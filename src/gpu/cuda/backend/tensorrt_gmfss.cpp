// GMFSS Fortuna dans le backend TensorRT : entrée du graphe « paire » depuis deux images RGB
// (gmfss_frames, le noyau de rife_input sans ses plans constants) et softmax splatting entre les
// deux graphes (soft_splat, gmfss_kernels.cu).
#include <format>
#include <string>

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

std::expected<void, Refusal> TensorRtBackend::soft_splat(const Tensor& source, const Tensor& motion, float s,
                                                         const Tensor& out) {
  if (!fp16_planes(source, 0) || !fp16_planes(motion, kernels::kGmfssMotionChannels) ||
      source.shape.dims != out.shape.dims || !fp16_planes(out, source.shape.dims[1]) ||
      width_of(motion) != width_of(source) || height_of(motion) != height_of(source)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "soft_splat : source, mouvement (3 plans) et sortie FP16 de même taille attendus");
  }
  if (!(s >= 0.0F && s <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("soft_splat : instant {} hors de [0, 1]", s));
  if (auto ok = bind(); !ok) return ok;
  const int channels = static_cast<int>(source.shape.dims[1]);
  const std::size_t bytes = soft_splat_sums_bytes(width_of(source), height_of(source), channels);
  if (bytes > splat_bytes_) {
    // cudaFree attend la fin de tout ce que l'appareil exécute : les splats en cours compris.
    if (splat_sums_ != nullptr) cudaFree(splat_sums_);
    splat_sums_ = nullptr;
    splat_bytes_ = 0;
    if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&splat_sums_), bytes), "cudaMalloc"); !ok)
      return ok;
    splat_bytes_ = bytes;
  }
  const SoftSplatParams params{.source = address(source),
                               .motion = address(motion),
                               .width = width_of(source),
                               .height = height_of(source),
                               .channels = channels,
                               .s = s,
                               .sums = splat_sums_,
                               .out = address(out)};
  GpuTiming& timing = stage("splatting de GMFSS");
  timing.begin(stream_);
  auto launched = cuda_check(launch_soft_splat(params, stream_), "soft_splat<<<>>>");
  timing.end(stream_);
  return launched;
}

}  // namespace ajp::gpu::cuda
