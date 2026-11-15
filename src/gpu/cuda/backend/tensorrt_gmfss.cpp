// GMFSS Fortuna Union dans le backend TensorRT : entrée du graphe « paire » depuis deux images RGB
// (gmfss_frames, le noyau de rife_input sans ses plans constants) et softmax splatting entre les deux graphes
// (soft_splat_fill, gmfss_kernels.cu), l'instant de la fusion rempli au passage ; composition du mélange à la
// taille de travail (gmfss_compose).
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {
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

// Les sommes d'un travail, alignées comme cudaMalloc l'aurait fait : les travaux se suivent dans un tampon.
std::size_t aligned_sums(const KernelOps::SplatJob& job) {
  constexpr std::size_t kAlign = 256;
  const std::size_t bytes = soft_splat_sums_bytes(width_of(job.source), height_of(job.source),
                                                  static_cast<int>(job.source.shape.dims[1]));
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
  for (const KernelOps::SplatJob& job : jobs) {
    if (auto ok = check_job(job, time); !ok) return ok;
    total += aligned_sums(job);
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
  // Les sommes sous une clé fixe : elles changent d'adresse en grandissant.
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(&splat_sums_)};
  if (time.handle != 0) {
    batch.fill = address(time);
    batch.fill_count = 1;
    for (const auto d : time.shape.dims) batch.fill_count *= static_cast<std::size_t>(d);
    batch.fill_value = t;
    writes.push_back(time.handle);
  }
  std::size_t offset = 0;
  std::vector<std::uintptr_t> reads;
  for (const KernelOps::SplatJob& job : jobs) {
    const int channels = static_cast<int>(job.source.shape.dims[1]);
    const std::size_t first_sum = offset / sizeof(float);  // octets vers floats
    batch.jobs[batch.count++] = SoftSplatParams{.source = address(job.source),
                                                .motion = address(job.motion),
                                                .width = width_of(job.source),
                                                .height = height_of(job.source),
                                                .channels = channels,
                                                .s = job.s,
                                                .sums = splat_sums_ + first_sum,
                                                .out = address(job.out),
                                                .motion_width = width_of(job.motion),
                                                .motion_height = height_of(job.motion),
                                                .grid_scale = own_grid(job) ? 0.0F : job.grid.scale,
                                                .grid_x = job.grid.x,
                                                .grid_y = job.grid.y,
                                                .coverage = job.coverage ? 1 : 0};
    offset += aligned_sums(job);
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

}  // namespace ajp::gpu::cuda
