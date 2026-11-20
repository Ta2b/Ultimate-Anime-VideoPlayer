// GMFSS Fortuna Union dans le backend TensorRT : entrée du graphe « paire » depuis deux images RGB
// (gmfss_frames, le noyau de rife_input sans ses plans constants) et softmax splatting entre les deux graphes
// (soft_splat_fill, gmfss_kernels.cu), l'instant de la fusion rempli au passage ; composition du mélange à la
// taille de travail (gmfss_compose).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_fused_engine.h"
#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/gmfss_splat_schedule.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace kirari::gpu::cuda {
namespace {

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

// Les plans poussés par un travail : la source, ou le flot (fx, fy) d'un splat tiré.
int pushed_planes(const KernelOps::SplatJob& job) {
  return job.pull ? 2 : static_cast<int>(job.source.shape.dims[1]);
}

// Les sommes d'un travail, alignées comme cudaMalloc l'aurait fait : les travaux se suivent dans un tampon.
std::size_t aligned_sums(const KernelOps::SplatJob& job) {
  return gmfss_splat::aligned_bytes(
      soft_splat_sums_bytes(width_of(job.source), height_of(job.source), pushed_planes(job)));
}

// Le mouvement d'un travail ramené à la taille de sa source, FP16, aligné comme ses sommes.
std::size_t aligned_motion(const KernelOps::SplatJob& job) {
  constexpr std::size_t kAlign = 256;
  const std::size_t bytes = static_cast<std::size_t>(kernels::kGmfssMotionChannels) *
                            static_cast<std::size_t>(width_of(job.source)) *
                            static_cast<std::size_t>(height_of(job.source)) * sizeof(std::uint16_t);
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

bool same_size(const Tensor& a, const Tensor& b) {
  return a.shape.dims[2] == b.shape.dims[2] && a.shape.dims[3] == b.shape.dims[3];
}

// Le mouvement lu tel quel, sur la grille de la source (MotionGrid par défaut).
bool own_grid(const KernelOps::SplatJob& job) {
  return job.grid.scale == 1.0F && job.grid.x == 0.0F && job.grid.y == 0.0F;
}

std::expected<void, Refusal> check_job(const KernelOps::SplatJob& job, const Tensor& time) {
  const std::int64_t planes = job.source.shape.dims[1] + (job.coverage ? 1 : 0);
  if (!fp16_planes(job.source, 0) || !fp16_planes(job.motion, kernels::kGmfssMotionChannels) ||
      !fp16_planes(job.out, planes) || (own_grid(job) && !same_size(job.motion, job.source)) ||
      !same_size(job.out, job.source)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "soft_splat : source, mouvement (3 plans) et sortie FP16 de même taille attendus");
  }
  if (!(job.grid.scale > 0.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("soft_splat : échelle {}", job.grid.scale));
  // La sortie n'est jamais une entrée : le splat lit la source et le mouvement de tous les travaux pendant
  // que la normalisation écrit les sorties. L'instant n'est ni lu ni écrit par un splat.
  const auto same = [](const Tensor& a, const Tensor& b) {
    return a.handle == b.handle && a.offset == b.offset;
  };
  if (same(job.out, job.motion) || same(job.out, job.source))
    return refuse(RefusalCode::ShapeInvalid, "soft_splat : la sortie est l'une des entrées");
  if (time.handle != 0 &&
      (job.source.handle == time.handle || job.motion.handle == time.handle || job.out.handle == time.handle))
    return refuse(RefusalCode::ShapeInvalid, "soft_splat_fill : l'instant partage la mémoire d'un splat");
  if (!(job.s >= 0.0F && job.s <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("soft_splat : instant {} hors de [0, 1]", job.s));
  return {};
}

// Les paramètres des lancements d'un lot de splats, travail par travail : le mouvement ramené à la grille de
// la source, le splat, et pour un splat tiré le flot poussé dans le tampon des grilles puis la source tirée.
struct SplatPlan {
  SoftSplatBatch& batch;
  char* grid_base = nullptr;
  std::size_t grid_offset = 0;
  std::vector<MotionGridParams> regrids;
  std::vector<PullParams> pulls;
  std::vector<std::uintptr_t> reads;
  std::vector<std::uintptr_t> writes;

  void* take_grid(const KernelOps::SplatJob& job) {
    void* at = grid_base + grid_offset;
    grid_offset += aligned_motion(job);
    return at;
  }

  void add(const KernelOps::SplatJob& job, void* pixels, int pitch) {
    const void* motion = address(job.motion);
    if (!own_grid(job)) {  // le mouvement ramené d'abord à la grille de la source (motion_on_grid)
      void* regridded = take_grid(job);
      regrids.push_back({.motion = motion,
                         .motion_width = width_of(job.motion),
                         .motion_height = height_of(job.motion),
                         .scale = job.grid.scale,
                         .x = job.grid.x,
                         .y = job.grid.y,
                         .out = regridded,
                         .width = width_of(job.source),
                         .height = height_of(job.source)});
      motion = regridded;
    }
    const void* pushed = address(job.source);
    void* out = address(job.out);
    if (job.pull) {  // le flot (les deux premiers plans du mouvement) poussé vers le tampon, puis tiré
      void* flow = take_grid(job);
      pulls.push_back({.source = address(job.source),
                       .channels = static_cast<int>(job.source.shape.dims[1]),
                       .flow = flow,
                       .width = width_of(job.source),
                       .height = height_of(job.source),
                       .s = job.s,
                       .out = out,
                       .coverage = job.coverage ? 1 : 0});
      pushed = motion;
      out = flow;
    }
    batch.jobs[batch.count++] = SoftSplatParams{.source = pushed,
                                                .motion = motion,
                                                .width = width_of(job.source),
                                                .height = height_of(job.source),
                                                .channels = pushed_planes(job),
                                                .s = job.s,
                                                .out = pixels != nullptr ? pixels : out,
                                                .coverage = job.pull || job.coverage ? 1 : 0,
                                                .out_pitch = pitch};
    reads.push_back(job.source.handle);
    reads.push_back(job.motion.handle);
    writes.push_back(job.out.handle);
  }
};

constexpr std::size_t kMinSplatL2 = 32U * 1024 * 1024;

// Le tampon partage le L2 avec les entrées, les sorties et les autres voies de calcul.
std::size_t splat_l2_budget() {
  struct DeviceCache {
    int ordinal = -1;
    int l2 = 0;
  };
  static thread_local DeviceCache cached;
  int ordinal = -1;
  if (cudaGetDevice(&ordinal) != cudaSuccess) return 0;
  if (ordinal != cached.ordinal) {
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ordinal) != cudaSuccess) return 0;
    cached = {.ordinal = ordinal, .l2 = l2};
  }
  const auto l2 = static_cast<std::size_t>(std::max(0, cached.l2));
  return l2 >= kMinSplatL2 ? l2 / 2 : 0;
}

struct SplatStaging {
  std::array<int, kMaxSplatJobs> channels{};
  std::size_t bytes = 0;
};

std::optional<SplatStaging> stage_splats(std::span<const KernelOps::SplatJob> jobs, std::size_t total) {
  if (total <= kMinSplatL2 / 2) return std::nullopt;
  const std::size_t budget = splat_l2_budget();
  if (budget == 0 || total <= budget) return std::nullopt;
  SplatStaging plan;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const KernelOps::SplatJob& job = jobs[i];
    // Le lot intégral lit toutes ses entrées avant d'écrire. Conserver cet ordre si elles se partagent
    // un tampon avec une sortie, même si leurs tranches sont distinctes.
    if (std::ranges::any_of(jobs, [&](const KernelOps::SplatJob& other) {
          return other.source.handle == job.out.handle || other.motion.handle == job.out.handle;
        }))
      return std::nullopt;
    const int width = width_of(job.source);
    const int height = height_of(job.source);
    const auto plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const int channels = gmfss_splat::stage_channels(plane, pushed_planes(job), budget);
    if (channels == 0) return std::nullopt;
    plan.channels[i] = channels;
    plan.bytes =
        std::max(plan.bytes, gmfss_splat::aligned_bytes(soft_splat_sums_bytes(width, height, channels)));
  }
  return plan;
}

cudaError_t launch_splat_batch(SoftSplatBatch& batch, float* sums, std::size_t bytes, cudaStream_t stream) {
  std::size_t offset = 0;
  for (int j = 0; j < batch.count; ++j) {
    SoftSplatParams& p = batch.jobs[j];
    p.sums = sums + offset / sizeof(float);
    offset += gmfss_splat::aligned_bytes(soft_splat_sums_bytes(p.width, p.height, p.channels));
  }
  return launch_soft_splat_all(batch, sums, bytes, stream);
}

// Chaque étape applique le même splat FP32. Le premier lot accumule le dénominateur commun à tous les
// canaux. Les suivants effacent leurs numérateurs seulement. Le dernier écrit aussi la couverture.
cudaError_t launch_staged_splats(const SoftSplatBatch& batch, const SplatStaging& plan, float* sums,
                                 cudaStream_t stream) {
  SoftSplatBatch stage{
      .count = 1, .fill = batch.fill, .fill_count = batch.fill_count, .fill_value = batch.fill_value};
  for (int j = 0; j < batch.count; ++j) {
    const SoftSplatParams& job = batch.jobs[j];
    const auto plane = static_cast<std::size_t>(job.width) * static_cast<std::size_t>(job.height);
    const int stage_planes = plan.channels[static_cast<std::size_t>(j)];
    for (int first = 0; first < job.channels;) {
      const int count = std::min(job.channels - first, stage_planes);
      const std::size_t offset = static_cast<std::size_t>(first) * plane * sizeof(std::uint16_t);
      SoftSplatParams p = job;
      p.source = static_cast<const std::byte*>(job.source) + offset;
      if (job.out_pitch >= 0)
        p.out = static_cast<std::byte*>(job.out) +
                (job.out_pitch == 0 ? offset : static_cast<std::size_t>(first) * sizeof(std::uint16_t));
      p.out_first = first;
      p.sums = sums;
      p.channels = count;
      p.weight_planes = stage_planes;
      p.reuse_weights = first != 0;
      first += count;
      p.coverage = first == job.channels ? job.coverage : 0;
      stage.jobs[0] = p;
      const auto planes_to_clear = static_cast<std::size_t>(count + (p.reuse_weights ? 0 : 1));
      const auto bytes = planes_to_clear * plane * sizeof(float);
      if (const cudaError_t result = launch_soft_splat_all(stage, sums, bytes, stream); result != cudaSuccess)
        return result;
      stage.fill = nullptr;
    }
  }
  return cudaSuccess;
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
                               .planes = kernels::kGmfssFramePlanes,
                               .replicate = true};
  if (auto ok = order({a.handle, b.handle}, {frames.handle}); !ok) return ok;
  GpuTiming& timing = stage("entrée de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>> (GMFSS)");
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::reserve_grid_motion(std::size_t bytes) {
  if (bytes <= grid_motion_bytes_) return {};
  // Les splats en cours peuvent encore lire l'ancien tampon : rendu derrière toutes les voies, comme les
  // sommes.
  if (grid_motion_ != nullptr) free_after_lanes(grid_motion_);
  grid_motion_ = nullptr;
  grid_motion_bytes_ = 0;
  if (auto ok = cuda_check(cudaMallocAsync(&grid_motion_, bytes, stream_), "cudaMallocAsync"); !ok) return ok;
  grid_motion_bytes_ = bytes;
  return {};
}

std::expected<void, Refusal> TensorRtBackend::soft_splat_fill(std::span<const KernelOps::SplatJob> jobs,
                                                              const Tensor& time, float t) {
  return soft_splat(jobs, time, t, {}, nullptr);
}

std::expected<void, Refusal> TensorRtBackend::gmfss_fuse_splats(Engine& engine,
                                                               std::span<const SplatJob> jobs,
                                                               const Tensor& time, float t,
                                                               std::span<const Tensor> inputs,
                                                               std::span<Tensor> outputs) {
  auto* fusion = dynamic_cast<GmfssFusedEngine*>(&engine);
  const auto targets = fusion != nullptr ? fusion->packed_splats(inputs) : std::nullopt;
  if (!targets || jobs.size() != static_cast<std::size_t>(kMaxSplatJobs) || outputs.size() != 1)
    return refuse(RefusalCode::FormatUnsupported, "GMFSS : huit splats et un GridNet CUDA attendus");

  std::array<SplatDestination, kMaxSplatJobs> direct{};
  std::array<bool, 6> matched{};
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const SplatJob& job = jobs[i];
    const auto found = std::ranges::find_if(*targets, [&](const auto& target) {
      const Tensor& plane = target.planar;
      return job.out.handle == plane.handle && job.out.offset == plane.offset &&
             job.out.shape == plane.shape && job.out.type == plane.type;
    });
    if (found == targets->end()) continue;  // les deux images RVB restent en plans
    const auto rank = static_cast<std::size_t>(found - targets->begin());
    if (matched[rank] || job.pull || job.coverage || !own_grid(job))
      return refuse(RefusalCode::FormatUnsupported, "GMFSS : une tranche de GridNet par splat poussé");
    matched[rank] = true;
    direct[i] = {.pixels = found->pixels, .pitch = found->pitch};
  }
  if (!std::ranges::all_of(matched, [](bool ok) { return ok; }))
    return refuse(RefusalCode::FormatUnsupported, "GMFSS : caractéristiques à pousser incomplètes");
  if (auto ok = soft_splat(jobs, time, t, direct, fusion); !ok) return ok;
  return enqueue_gmfss_fused(*fusion, inputs, outputs, true);
}

std::expected<void, Refusal> TensorRtBackend::soft_splat(std::span<const SplatJob> jobs, const Tensor& time,
                                                        float t, std::span<const SplatDestination> direct,
                                                        const GmfssFusedEngine* fusion) {
  if (time.handle != 0 && !fp16_planes(time, 0))
    return refuse(RefusalCode::ShapeInvalid, "soft_splat_fill : instant FP16 1 × c × h × w attendu");
  if (!(t >= 0.0F && t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("soft_splat_fill : instant {} hors de [0, 1]", t));
  if (jobs.empty() || jobs.size() > static_cast<std::size_t>(kMaxSplatJobs)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("soft_splat : {} splats en un passage, de 1 à {}", jobs.size(), kMaxSplatJobs));
  }
  std::size_t total = 0;
  std::size_t grids = 0;  // octets des mouvements ramenés à la grille de leur source
  for (const KernelOps::SplatJob& job : jobs) {
    if (auto ok = check_job(job, time); !ok) return ok;
    total += aligned_sums(job);
    if (!own_grid(job)) grids += aligned_motion(job);
    if (job.pull) grids += aligned_motion(job);  // le flot poussé et sa couverture
  }
  if (auto ok = bind(); !ok) return ok;
  if (auto ok = reserve_grid_motion(grids); !ok) return ok;
  const auto staging = stage_splats(jobs, total);
  const std::size_t bytes = staging ? staging->bytes : total;
  if (bytes > splat_bytes_) {
    // Les splats en cours peuvent encore lire l'ancien tampon : rendu derrière toutes les voies (11.4).
    if (splat_sums_ != nullptr) free_after_lanes(splat_sums_);
    splat_sums_ = nullptr;
    splat_bytes_ = 0;
    if (auto ok = cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&splat_sums_), bytes, stream_),
                             "cudaMallocAsync");
        !ok)
      return ok;
    splat_bytes_ = bytes;
  }
  SoftSplatBatch batch{};
  // Les sommes sous une clé fixe : elles changent d'adresse en grandissant.
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(&splat_sums_)};
  if (fusion != nullptr) writes.push_back(reinterpret_cast<std::uintptr_t>(fusion));
  if (time.handle != 0) {
    batch.fill = address(time);
    batch.fill_count = 1;
    for (const auto d : time.shape.dims) batch.fill_count *= static_cast<std::size_t>(d);
    batch.fill_value = t;
    writes.push_back(time.handle);
  }
  std::vector<std::uintptr_t> reads;
  if (grids > 0) writes.push_back(reinterpret_cast<std::uintptr_t>(&grid_motion_));
  SplatPlan plan{.batch = batch, .grid_base = static_cast<char*>(grid_motion_)};
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const SplatDestination out = direct.empty() ? SplatDestination{} : direct[i];
    plan.add(jobs[i], out.pixels, out.pitch);
  }
  const std::vector<MotionGridParams>& regrids = plan.regrids;
  const std::vector<PullParams>& pulls = plan.pulls;
  reads = std::move(plan.reads);
  writes.insert(writes.end(), plan.writes.begin(), plan.writes.end());
  if (auto ok = order(reads, writes); !ok) return ok;
  // La composition de clé pousse aussi des splats, mais ne pose pas l'instant du GridNet.
  GpuTiming& timing = stage(time.handle != 0 ? "splatting de GMFSS" : "splats de composition GMFSS");
  timing.begin(stream_);
  for (const MotionGridParams& regrid : regrids) {
    if (auto ok = cuda_check(launch_motion_on_grid(regrid, stream_), "motion_on_grid<<<>>>"); !ok) {
      timing.end(stream_);
      return ok;
    }
  }
  auto launched = cuda_check(
      staging ? launch_staged_splats(batch, *staging, splat_sums_, stream_)
              : launch_splat_batch(batch, splat_sums_, bytes, stream_),
      "soft_splat<<<>>>");
  for (const PullParams& pull : pulls) {
    if (!launched) break;
    launched = cuda_check(launch_soft_pull(pull, stream_), "soft_pull<<<>>>");
  }
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::gmfss_compose(const KernelOps::ComposeJob& job) {
  constexpr std::int64_t kSplatted = 4;  // RGB et la couverture
  const bool shapes = fp16_planes(job.high_a, kSplatted) && fp16_planes(job.high_b, kSplatted) &&
                      fp16_planes(job.low_a, kSplatted) && fp16_planes(job.low_b, kSplatted) &&
                      fp16_planes(job.fusion, 3) && fp16_planes(job.out, 3) &&
                      same_size(job.high_a, job.out) && same_size(job.high_b, job.out) &&
                      same_size(job.low_a, job.low_b);
  const int w = width_of(job.low_a);
  const int h = height_of(job.low_a);
  if (!shapes || job.window.width != w || job.window.height != h || job.window.x < 0 || job.window.y < 0 ||
      job.window.x + w > width_of(job.fusion) || job.window.y + h > height_of(job.fusion) ||
      width_of(job.out) < w || height_of(job.out) < h) {
    return refuse(RefusalCode::ShapeInvalid,
                  "gmfss_compose : sources splattées (4 plans), fusion et sortie FP16 cohérentes attendues");
  }
  if (!(job.t >= 0.0F && job.t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("gmfss_compose : instant {} hors de [0, 1]", job.t));
  if (auto ok = bind(); !ok) return ok;
  const ComposeParams params{.high_a = address(job.high_a),
                             .high_b = address(job.high_b),
                             .low_a = address(job.low_a),
                             .low_b = address(job.low_b),
                             .fusion = address(job.fusion),
                             .width = w,
                             .height = h,
                             .out_width = width_of(job.out),
                             .out_height = height_of(job.out),
                             .fusion_width = width_of(job.fusion),
                             .fusion_height = height_of(job.fusion),
                             .left = job.window.x,
                             .top = job.window.y,
                             .t = job.t,
                             .out = address(job.out)};
  if (auto ok =
          order({job.high_a.handle, job.high_b.handle, job.low_a.handle, job.low_b.handle, job.fusion.handle},
                {job.out.handle});
      !ok)
    return ok;
  GpuTiming& timing = stage("composition de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_gmfss_compose(params, stream_), "gmfss_compose<<<>>>");
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::gmfss_still(const KernelOps::StillJob& job) {
  if (!fp16_planes(job.key, 3) || !fp16_planes(job.a, 3) || !fp16_planes(job.b, 3) ||
      !same_size(job.key, job.a) || !same_size(job.key, job.b)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "gmfss_still : clé et sources RGB FP16 de même taille attendues");
  }
  if (!(job.t >= 0.0F && job.t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("gmfss_still : instant {} hors de [0, 1]", job.t));
  if (auto ok = bind(); !ok) return ok;
  const StillParams params{.key = address(job.key),
                           .a = address(job.a),
                           .b = address(job.b),
                           .width = width_of(job.key),
                           .height = height_of(job.key),
                           .t = job.t};
  if (auto ok = order({job.key.handle, job.a.handle, job.b.handle}, {job.key.handle}); !ok) return ok;
  GpuTiming& timing = stage("décor fixe de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_gmfss_still(params, stream_), "gmfss_still<<<>>>");
  timing.end(stream_);
  return launched;
}

}  // namespace kirari::gpu::cuda
