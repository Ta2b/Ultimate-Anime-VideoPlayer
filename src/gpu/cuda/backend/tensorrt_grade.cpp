// Le grade exact dans l'anneau (chantier 2c, docs/adr/0009-grade-dans-l-anneau.md) dans le backend TensorRT :
// les ordres de KernelOps::set_grade (table d'entrée et pile de LUT en mémoire de l'appareil, les
// kGradeOrders derniers), les cartes et l'historique de la zone à la taille de l'anneau, et le grade d'un
// emplacement entre son écriture et son signal (finish_slot), aux trois endroits qui signalent un
// emplacement : write_ring, launch_into (warp, composition de GMFSS) et la queue du SPAN FP8.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>  // cudaCreateChannelDesc, make_cudaExtent
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/grade_kernels.h"
#include "gpu/cuda/backend/ring_slot.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace ajp::gpu::cuda {
namespace {

namespace g = kernels::grade;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Au-delà de cet écart de pts entre deux images de la même époque, la zone repart de sa mesure : l'image
// d'avant n'est plus la voisine (saut, image tenue longtemps).
constexpr std::int64_t kHistoryGapUs = 1'000'000;

}  // namespace

// Un ordre en mémoire de l'appareil.
struct GradeVersionDevice {
  std::uint32_t version = 0;  // 0 : case libre
  kernels::GradeUniforms uniforms;
  float* input_nits = nullptr;
  std::array<float, 9> input_matrix{};
  cudaArray_t look_array = nullptr;
  cudaTextureObject_t look = 0;
  int look_size = 0;

  void release() {
    if (look != 0) cudaDestroyTextureObject(look);
    if (look_array != nullptr) cudaFreeArray(look_array);
    if (input_nits != nullptr) cudaFree(input_nits);
    *this = GradeVersionDevice{};
  }
};

struct GradeState {
  std::array<GradeVersionDevice, kGradeOrders> versions{};
  std::size_t next = 0;
  // Les cartes, en un seul bloc (les voies s'ordonnent sur son adresse) : locale, zone, historique, max
  // local, puis le masque peau quand un ordre l'allume.
  void* block = nullptr;
  int width = 0;
  int height = 0;
  bool skin = false;
  GradeMapSizes sizes;
  // La chaîne d'images dont l'historique parle.
  bool kept = false;
  std::uint32_t epoch = 0;
  std::int64_t pts_us = 0;

  ~GradeState() {
    for (GradeVersionDevice& v : versions) v.release();
    if (block != nullptr) cudaFree(block);
  }
};

void GradeStateDeleter::operator()(GradeState* state) const {
  delete state;
}  // NOLINT(cppcoreguidelines-owning-memory)

std::expected<void, Refusal> TensorRtBackend::set_grade(const GradeOrder& order) {
  constexpr auto kTable = static_cast<std::size_t>(3 * g::kInputTablePoints);
  if (order.version == 0 || order.input_nits.size() != kTable) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("set_grade : version {} et {} nits d'entrée ({} attendus)", order.version,
                              order.input_nits.size(), kTable));
  }
  const auto n = static_cast<std::size_t>(std::max(order.look_size, 0));
  if (order.look_size != 0 && (order.look_size < 2 || order.look_rgb.size() != n * n * n * 3)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("set_grade : look de taille {} et {} valeurs",
                                                            order.look_size, order.look_rgb.size()));
  }
  if (order.look_size == 0 && order.uniforms.look_mix > 0.0F) {
    return refuse(RefusalCode::FieldOutOfRange, "set_grade : intensité du look sans look");
  }
  if (auto ok = bind(); !ok) return ok;
  if (!grade_) grade_.reset(new GradeState());  // NOLINT(cppcoreguidelines-owning-memory)
  GradeVersionDevice& entry = grade_->versions[grade_->next % kGradeOrders];
  if (entry.version != 0) {
    // La case la plus ancienne : des images encore en vol la lisent peut-être, sur n'importe quelle voie.
    if (auto ok = join(); !ok) return ok;
    if (auto ok = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); !ok) return ok;
    entry.release();
  }
  GradeVersionDevice made{.version = order.version,
                          .uniforms = order.uniforms,
                          .input_matrix = order.input_matrix,
                          .look_size = order.look_size};
  auto ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&made.input_nits), kTable * sizeof(float)),
                          "cudaMalloc");
  if (ready) {
    ready = cuda_check(
        cudaMemcpy(made.input_nits, order.input_nits.data(), kTable * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy");
  }
  if (ready && n != 0) {
    std::vector<float> rgba(n * n * n * 4);
    for (std::size_t i = 0; i < n * n * n; ++i) {
      rgba[i * 4 + 0] = order.look_rgb[i * 3 + 0];
      rgba[i * 4 + 1] = order.look_rgb[i * 3 + 1];
      rgba[i * 4 + 2] = order.look_rgb[i * 3 + 2];
      rgba[i * 4 + 3] = 1.0F;
    }
    const cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
    ready =
        cuda_check(cudaMalloc3DArray(&made.look_array, &f4, make_cudaExtent(n, n, n)), "cudaMalloc3DArray");
    if (ready) {
      cudaMemcpy3DParms copy{};
      copy.srcPtr = make_cudaPitchedPtr(rgba.data(), n * 4 * sizeof(float), n, n);
      copy.dstArray = made.look_array;
      copy.extent = make_cudaExtent(n, n, n);
      copy.kind = cudaMemcpyHostToDevice;
      ready = cuda_check(cudaMemcpy3D(&copy), "cudaMemcpy3D");
    }
    if (ready) {
      cudaResourceDesc resource{};
      resource.resType = cudaResourceTypeArray;
      resource.res.array.array = made.look_array;
      cudaTextureDesc tex{};
      tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = cudaAddressModeClamp;
      tex.filterMode = cudaFilterModeLinear;
      tex.readMode = cudaReadModeElementType;
      tex.normalizedCoords = 1;
      ready = cuda_check(cudaCreateTextureObject(&made.look, &resource, &tex, nullptr),
                         "cudaCreateTextureObject");
    }
  }
  if (!ready) {
    made.release();
    return ready;
  }
  entry = made;
  ++grade_->next;
  return {};
}

std::expected<void, Refusal> TensorRtBackend::finish_slot(const ImportedSlot& slot, const RingWrite& sync) {
  if (sync.grade.version != 0) {
    if (auto ok = grade_slot(slot, sync.grade); !ok) return ok;
  }
  return slot.signal(sync.signal_value, stream_);
}

std::expected<void, Refusal> TensorRtBackend::grade_slot(const ImportedSlot& slot, const GradeSlot& grade) {
  const GradeVersionDevice* version = nullptr;
  if (grade_) {
    for (const GradeVersionDevice& v : grade_->versions) {
      if (v.version == grade.version) version = &v;
    }
  }
  if (version == nullptr) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("grade de l'anneau : version {} inconnue", grade.version));
  }
  GradeState& state = *grade_;
  const RingSlotDesc& desc = slot.desc();
  const bool skin = version->uniforms.skin_on >= 0.5F;
  // Les cartes à la taille de l'anneau ; un changement de taille (ou le masque peau allumé) les refait.
  if (state.block == nullptr || state.width != desc.width || state.height != desc.height ||
      (skin && !state.skin)) {
    if (state.block != nullptr) {
      if (auto ok = join(); !ok) return ok;
      if (auto ok = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); !ok) return ok;
      cudaFree(state.block);
      state.block = nullptr;
    }
    const GradeMapSizes s = grade_map_sizes(desc.width, desc.height);
    const auto local = static_cast<std::size_t>(s.local_w) * static_cast<std::size_t>(s.local_h);
    const auto area = static_cast<std::size_t>(s.area_w) * static_cast<std::size_t>(s.area_h);
    const bool with_skin = skin || state.skin;
    const std::size_t floats =
        3 * local + 2 * area + local +
        (with_skin ? static_cast<std::size_t>(desc.width) * static_cast<std::size_t>(desc.height) : 0);
    if (auto ok = cuda_check(cudaMalloc(&state.block, floats * sizeof(float)), "cudaMalloc"); !ok) return ok;
    if (auto ok =
            cuda_check(cudaMemsetAsync(state.block, 0, floats * sizeof(float), stream_), "cudaMemsetAsync");
        !ok)
      return ok;
    state.width = desc.width;
    state.height = desc.height;
    state.skin = with_skin;
    state.sizes = s;
    state.kept = false;
  }
  const GradeMapSizes& s = state.sizes;
  const auto local = static_cast<std::size_t>(s.local_w) * static_cast<std::size_t>(s.local_h);
  const auto area = static_cast<std::size_t>(s.area_w) * static_cast<std::size_t>(s.area_h);
  auto* base = static_cast<float*>(state.block);
  GradeFrame frame{.surface = slot.surface(),
                   .width = desc.width,
                   .height = desc.height,
                   .input_nits = version->input_nits,
                   .look = version->look,
                   .look_size = version->look_size,
                   .uniforms = version->uniforms,
                   .local = base,
                   .area = base + 3 * local,
                   .history = base + 3 * local + area,
                   .localmax = base + 3 * local + 2 * area,
                   .skin = state.skin ? base + 4 * local + 2 * area : nullptr};
  std::copy(version->input_matrix.begin(), version->input_matrix.end(), frame.input_matrix);
  // La chaîne : même époque, pts qui avance de moins de kHistoryGapUs ; sinon la zone repart.
  const bool known = grade.pts_us != kNoPts;
  const bool follows = state.kept && known && grade.epoch == state.epoch && grade.pts_us > state.pts_us &&
                       grade.pts_us - state.pts_us <= kHistoryGapUs;
  const float dt_s = follows ? static_cast<float>(grade.pts_us - state.pts_us) * 1e-6F : 0.0F;
  state.kept = known;
  state.epoch = grade.epoch;
  state.pts_us = known ? grade.pts_us : 0;
  // Les cartes sont communes aux voies : l'image d'avant, sur l'autre voie peut-être, est finie avant.
  const auto block = reinterpret_cast<std::uintptr_t>(state.block);
  const auto nits = reinterpret_cast<std::uintptr_t>(version->input_nits);
  if (auto ok = order({nits}, {block}); !ok) return ok;
  GpuTiming& timing = stage("grade de l'anneau");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_grade_local(frame, stream_), "grade_local<<<>>>"); !ok) return ok;
  if (auto ok = cuda_check(launch_grade_area(frame, dt_s, !follows, stream_), "grade_area<<<>>>"); !ok)
    return ok;
  if (auto ok = cuda_check(launch_grade_slot(frame, stream_), "grade_slot<<<>>>"); !ok) return ok;
  timing.end(stream_);
  return {};
}

}  // namespace ajp::gpu::cuda
