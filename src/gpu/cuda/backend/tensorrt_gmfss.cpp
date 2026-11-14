// GMFSS Fortuna Union dans le backend TensorRT : entrée du graphe « paire » depuis deux images RGB
// (gmfss_frames, le noyau de rife_input sans ses plans constants) et softmax splatting entre les deux graphes
// (soft_splat_fill, gmfss_kernels.cu), l'instant de la fusion rempli au passage.
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// FP16 1 × channels × h × w ; channels 0 : quelconque.
bool fp16_planes(const Tensor& tensor, std::int64_t channels) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         (channels == 0 || tensor.shape.dims[1] == channels) && tensor.shape.dims[1] > 0 &&
         tensor.shape.dims[2] > 0 && tensor.shape.dims[3] > 0;
}

int width_of(const Tensor& tensor) {
  return static_cast<int>(tensor.shape.dims[3]);
}
int height_of(const Tensor& tensor) {
  return static_cast<int>(tensor.shape.dims[2]);
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::gmfss_frames(const Tensor& a, const Tensor& b,
                                                           const Tensor& frames) {
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid, "gmfss_frames : deux images RGB FP16 de même taille attendues");
  }
  if (!fp16_planes(frames, kernels::kGmfssFramePlanes) ||
      !kernels::rife_fits(width_of(a), height_of(a), width_of(frames), height_of(frames))) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("gmfss_frames : tenseur FP16 1×{}×h×w, multiples de 64 au moins égaux à "
                              "l'image {}x{}, attendu",
                              kernels::kGmfssFramePlanes, width_of(a), height_of(a)));
  }
  const kernels::RifeGeometry g =
      kernels::rife_window(width_of(a), height_of(a), width_of(frames), height_of(frames));
  if (auto ok = bind(); !ok) return ok;
  const RifeInputParams params{.a = address(a),
                               .b = address(b),
                               .width = g.width,
                               .height = g.height,
                               .padded_width = g.padded_width,
                               .padded_height = g.padded_height,
                               .left = g.left,
                               .top = g.top,
                               .black = 0.0F,  // la chaîne ingère en plage limitée : le noir vaut 0
                               .input = address(frames),
                               .planes = kernels::kGmfssFramePlanes};
  if (auto ok = order({a.handle, b.handle}, {frames.handle}); !ok) return ok;
  GpuTiming& timing = stage("entrée de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>> (GMFSS)");
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::soft_splat_fill(std::span<const KernelOps::SplatJob> jobs,
                                                              const Tensor& time, float t) {
  if (!fp16_planes(time, 0))
    return refuse(RefusalCode::ShapeInvalid, "soft_splat_fill : instant FP16 1 × c × h × w attendu");
  if (!(t >= 0.0F && t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("soft_splat_fill : instant {} hors de [0, 1]", t));
  if (jobs.empty() || jobs.size() > static_cast<std::size_t>(kMaxSplatJobs)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("soft_splat : {} splats en un passage, de 1 à {}", jobs.size(), kMaxSplatJobs));
  }
  // Les sommes de chaque travail à la suite, chacune alignée comme cudaMalloc l'aurait fait.
  constexpr std::size_t kAlign = 256;
  std::size_t total = 0;
  const auto same = [](const Tensor& a, const Tensor& b) {
    return a.handle == b.handle && a.offset == b.offset;
  };
  for (const KernelOps::SplatJob& job : jobs) {
    if (!fp16_planes(job.source, 0) || !fp16_planes(job.motion, kernels::kGmfssMotionChannels) ||
        !fp16_planes(job.out, job.source.shape.dims[1]) ||
        job.motion.shape.dims[2] != job.source.shape.dims[2] ||
        job.motion.shape.dims[3] != job.source.shape.dims[3] ||
        job.out.shape.dims[2] != job.source.shape.dims[2] ||
        job.out.shape.dims[3] != job.source.shape.dims[3]) {
      return refuse(RefusalCode::ShapeInvalid,
                    "soft_splat : source, mouvement (3 plans) et sortie FP16 de même taille attendus");
    }
    // La sortie n'est jamais une entrée : le splat lit la source et le mouvement de tous les travaux pendant
    // que la normalisation écrit les sorties. L'instant n'est ni lu ni écrit par un splat.
    if (same(job.out, job.motion) || same(job.out, job.source))
      return refuse(RefusalCode::ShapeInvalid, "soft_splat : la sortie est l'une des entrées");
    if (job.source.handle == time.handle || job.motion.handle == time.handle || job.out.handle == time.handle)
      return refuse(RefusalCode::ShapeInvalid, "soft_splat_fill : l'instant partage la mémoire d'un splat");
    if (!(job.s >= 0.0F && job.s <= 1.0F))
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("soft_splat : instant {} hors de [0, 1]", job.s));
    const std::size_t bytes = soft_splat_sums_bytes(width_of(job.source), height_of(job.source),
                                                    static_cast<int>(job.source.shape.dims[1]));
    total += (bytes + kAlign - 1) / kAlign * kAlign;
  }
  if (auto ok = bind(); !ok) return ok;
  if (total > splat_bytes_) {
    // Les splats en cours peuvent encore lire l'ancien tampon : rendu derrière toutes les voies (11.4).
    if (splat_sums_ != nullptr) free_after_lanes(splat_sums_);
    splat_sums_ = nullptr;
    splat_bytes_ = 0;
    if (auto ok = cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&splat_sums_), total, stream_),
                             "cudaMallocAsync");
        !ok)
      return ok;
    splat_bytes_ = total;
  }
  SoftSplatBatch batch{};
  batch.fill = address(time);
  batch.fill_count = 1;
  for (const auto d : time.shape.dims) batch.fill_count *= static_cast<std::size_t>(d);
  batch.fill_value = t;
  std::size_t offset = 0;
  std::vector<std::uintptr_t> reads;
  // Les sommes sous une clé fixe : elles changent d'adresse en grandissant.
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(&splat_sums_), time.handle};
  for (const KernelOps::SplatJob& job : jobs) {
    const int channels = static_cast<int>(job.source.shape.dims[1]);
    const std::size_t bytes = soft_splat_sums_bytes(width_of(job.source), height_of(job.source), channels);
    batch.jobs[batch.count++] = SoftSplatParams{.source = address(job.source),
                                                .motion = address(job.motion),
                                                .width = width_of(job.source),
                                                .height = height_of(job.source),
                                                .channels = channels,
                                                .s = job.s,
                                                .sums = splat_sums_ + offset / sizeof(float),
                                                .out = address(job.out)};
    offset += (bytes + kAlign - 1) / kAlign * kAlign;
    reads.push_back(job.source.handle);
    reads.push_back(job.motion.handle);
    writes.push_back(job.out.handle);
  }
  if (auto ok = order(reads, writes); !ok) return ok;
  GpuTiming& timing = stage("splatting de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_soft_splat_all(batch, splat_sums_, total, stream_), "soft_splat<<<>>>");
  timing.end(stream_);
  return launched;
}

}  // namespace ajp::gpu::cuda
