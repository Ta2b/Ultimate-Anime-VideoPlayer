// GMFSS Fortuna dans le backend TensorRT : entrée du graphe « paire » depuis deux images RGB
// (gmfss_frames, le noyau de rife_input sans ses plans constants), softmax splatting entre les
// deux graphes (soft_splat, gmfss_kernels.cu) et composition après SPAN (gmfss_compose).
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

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
  const kernels::RifeGeometry g = kernels::rife_geometry(width_of(a), height_of(a));
  if (!fp16_planes(frames, kernels::kGmfssFramePlanes) || width_of(frames) != g.padded_width ||
      height_of(frames) != g.padded_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("gmfss_frames : tenseur FP16 1×{}×{}×{} attendu", kernels::kGmfssFramePlanes,
                              g.padded_height, g.padded_width));
  }
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
  GpuTiming& timing = stage("entrée de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>> (GMFSS)");
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::fill(const Tensor& tensor, float value) {
  if (!fp16_planes(tensor, 0))
    return refuse(RefusalCode::ShapeInvalid, "fill : tenseur FP16 1 × c × h × w attendu");
  if (auto ok = bind(); !ok) return ok;
  std::size_t count = 1;
  for (const auto d : tensor.shape.dims) count *= static_cast<std::size_t>(d);
  return cuda_check(launch_fill(address(tensor), count, value, stream_), "fill<<<>>>");
}

std::expected<void, Refusal> TensorRtBackend::soft_splat(const Tensor& source, const Tensor& motion, float s,
                                                         const Tensor& out) {
  const KernelOps::SplatJob job{.source = source, .motion = motion, .s = s, .out = out};
  return soft_splat_all(std::span(&job, 1));
}

std::expected<void, Refusal> TensorRtBackend::soft_splat_all(std::span<const KernelOps::SplatJob> jobs) {
  if (jobs.empty()) return {};
  if (jobs.size() > static_cast<std::size_t>(kMaxSplatJobs)) {
    return refuse(
        RefusalCode::FieldOutOfRange,
        std::format("soft_splat : {} splats en un passage, {} au plus", jobs.size(), kMaxSplatJobs));
  }
  // Les sommes de chaque travail à la suite, chacune alignée comme cudaMalloc l'aurait fait.
  constexpr std::size_t kAlign = 256;
  std::size_t total = 0;
  for (const KernelOps::SplatJob& job : jobs) {
    // La sortie a les plans de la source, ou un de plus (la couverture) ; le mouvement a sa grille.
    const std::int64_t channels = job.source.shape.dims[1];
    if (!fp16_planes(job.source, 0) || !fp16_planes(job.motion, kernels::kGmfssMotionChannels) ||
        !fp16_planes(job.out, 0) || width_of(job.out) != width_of(job.source) ||
        height_of(job.out) != height_of(job.source) ||
        (job.out.shape.dims[1] != channels && job.out.shape.dims[1] != channels + 1)) {
      return refuse(RefusalCode::ShapeInvalid,
                    "soft_splat : source, mouvement (3 plans) et sortie FP16 de même taille attendus");
    }
    if (!(job.s >= 0.0F && job.s <= 1.0F))
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("soft_splat : instant {} hors de [0, 1]", job.s));
    if (!(job.grid.scale > 0.0F)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("soft_splat : échelle du mouvement {}", job.grid.scale));
    }
    const std::size_t bytes = soft_splat_sums_bytes(width_of(job.source), height_of(job.source),
                                                    static_cast<int>(job.source.shape.dims[1]));
    total += (bytes + kAlign - 1) / kAlign * kAlign;
  }
  if (auto ok = bind(); !ok) return ok;
  if (total > splat_bytes_) {
    // cudaFree attend la fin de tout ce que l'appareil exécute : les splats en cours compris.
    if (splat_sums_ != nullptr) cudaFree(splat_sums_);
    splat_sums_ = nullptr;
    splat_bytes_ = 0;
    if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&splat_sums_), total), "cudaMalloc"); !ok)
      return ok;
    splat_bytes_ = total;
  }
  SoftSplatBatch batch{};
  std::size_t offset = 0;
  for (const KernelOps::SplatJob& job : jobs) {
    const int channels = static_cast<int>(job.source.shape.dims[1]);
    const std::size_t bytes = soft_splat_sums_bytes(width_of(job.source), height_of(job.source), channels);
    // Même grille, sans décalage ni échelle : le mouvement du pixel, tel quel.
    const bool direct = job.grid.scale == 1.0F && job.grid.x == 0.0F && job.grid.y == 0.0F &&
                        width_of(job.motion) == width_of(job.source) &&
                        height_of(job.motion) == height_of(job.source);
    batch.jobs[batch.count++] =
        SoftSplatParams{.source = address(job.source),
                        .motion = address(job.motion),
                        .width = width_of(job.source),
                        .height = height_of(job.source),
                        .channels = channels,
                        .s = job.s,
                        .motion_width = width_of(job.motion),
                        .motion_height = height_of(job.motion),
                        .motion_scale = job.grid.scale,
                        .motion_x = job.grid.x,
                        .motion_y = job.grid.y,
                        .direct = direct ? 1 : 0,
                        .coverage = job.out.shape.dims[1] > job.source.shape.dims[1] ? 1 : 0,
                        .sums = splat_sums_ + offset / sizeof(float),
                        .out = address(job.out)};
    offset += (bytes + kAlign - 1) / kAlign * kAlign;
  }
  GpuTiming& timing = stage("splatting de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_soft_splat_all(batch, splat_sums_, total, stream_), "soft_splat<<<>>>");
  timing.end(stream_);
  return launched;
}

std::expected<void, Refusal> TensorRtBackend::gmfss_compose(const Tensor& high_a, const Tensor& high_b,
                                                            const Tensor& low_a, const Tensor& low_b,
                                                            const Tensor& fusion,
                                                            const kernels::Window& window, float t,
                                                            const Tensor& out) {
  if (!fp16_planes(out, 3) || !fp16_planes(high_a, 4) || !fp16_planes(high_b, 4) || !fp16_planes(low_a, 4) ||
      !fp16_planes(low_b, 4) || !fp16_planes(fusion, 3) || width_of(high_a) != width_of(out) ||
      height_of(high_a) != height_of(out) || width_of(high_b) != width_of(out) ||
      height_of(high_b) != height_of(out) || width_of(low_b) != width_of(low_a) ||
      height_of(low_b) != height_of(low_a)) {
    return refuse(
        RefusalCode::ShapeInvalid,
        "gmfss_compose : agrandies (4 plans, taille de la sortie), travail (4 plans), fusion (3 plans) "
        "FP16 attendus");
  }
  const int lw = width_of(low_a);
  const int lh = height_of(low_a);
  if (width_of(out) % lw != 0 || height_of(out) % lh != 0 || width_of(out) / lw != height_of(out) / lh) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("gmfss_compose : sortie {}x{} pour un travail {}x{}, multiple entier attendu",
                              width_of(out), height_of(out), lw, lh));
  }
  if (window.x < 0 || window.y < 0 || window.width != lw || window.height != lh ||
      window.x > width_of(fusion) - lw || window.y > height_of(fusion) - lh) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("gmfss_compose : fenêtre {}x{} en ({}, {}) hors de la fusion {}x{}", window.width,
                    window.height, window.x, window.y, width_of(fusion), height_of(fusion)));
  }
  if (!(t >= 0.0F && t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("gmfss_compose : instant {} hors de [0, 1]", t));
  if (auto ok = bind(); !ok) return ok;
  const GmfssComposeParams params{.high_a = address(high_a),
                                  .high_b = address(high_b),
                                  .width = width_of(out),
                                  .height = height_of(out),
                                  .low_a = address(low_a),
                                  .low_b = address(low_b),
                                  .low_width = lw,
                                  .low_height = lh,
                                  .scale = width_of(out) / lw,
                                  .fusion = address(fusion),
                                  .fusion_width = width_of(fusion),
                                  .fusion_height = height_of(fusion),
                                  .window_x = window.x,
                                  .window_y = window.y,
                                  .t = t,
                                  .out = address(out)};
  GpuTiming& timing = stage("composition de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_gmfss_compose(params, stream_), "gmfss_compose<<<>>>");
  timing.end(stream_);
  return launched;
}

}  // namespace ajp::gpu::cuda
