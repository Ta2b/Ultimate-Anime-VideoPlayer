// RIFE dans le backend TensorRT de la tranche verticale : entrée remplie (rife_input), recadrage
// de la sortie (rife_crop, du contrat) et mesures immédiates d'une paire (scores_now).
#include <cstring>
#include <format>
#include <string>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool fp16_planes(const Tensor& tensor, std::int64_t channels) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         tensor.shape.dims[1] == channels && tensor.shape.dims[2] > 0 && tensor.shape.dims[3] > 0;
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

std::expected<void, Refusal> TensorRtBackend::rife_input(const Tensor& a, const Tensor& b, float t,
                                                         const Tensor& input) {
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid, "rife_input : deux images RGB FP16 de même taille attendues");
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(width_of(a), height_of(a));
  if (!fp16_planes(input, kernels::kRifeInputChannels) || width_of(input) != g.padded_width ||
      height_of(input) != g.padded_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("rife_input : tenseur FP16 1×{}×{}×{} attendu", kernels::kRifeInputChannels,
                              g.padded_height, g.padded_width));
  }
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("rife_input : instant {} hors de ]0, 1[", t));
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
                               .time = t,
                               .input = address(input)};
  return cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>>");
}

std::expected<void, Refusal> TensorRtBackend::rife_crop(const Tensor& padded, const Tensor& out) {
  if (!fp16_planes(padded, 3) || !fp16_planes(out, 3)) {
    return refuse(RefusalCode::ShapeInvalid, "rife_crop : tenseurs RGB FP16 attendus");
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(width_of(out), height_of(out));
  if (width_of(padded) != g.padded_width || height_of(padded) != g.padded_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("rife_crop : sortie de RIFE {}×{} attendue pour {}×{}", g.padded_height,
                              g.padded_width, g.height, g.width));
  }
  if (auto ok = bind(); !ok) return ok;
  const RifeCropParams params{.padded = address(padded),
                              .padded_width = g.padded_width,
                              .padded_height = g.padded_height,
                              .left = g.left,
                              .top = g.top,
                              .width = g.width,
                              .height = g.height,
                              .out = address(out)};
  return cuda_check(launch_rife_crop(params, stream_), "rife_crop<<<>>>");
}

std::expected<PairScores, Refusal> TensorRtBackend::scores_now(const Tensor& a, const Tensor& b) {
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid, "scores_now : deux images RGB FP16 de même taille attendues");
  }
  std::expected<void, Refusal> ready = bind();
  if (ready && pair_device_ == nullptr) {
    ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&pair_device_), sizeof(PairSums)), "cudaMalloc");
    if (ready) {
      ready = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&pair_host_), sizeof(PairSums)),
                         "cudaMallocHost");
    }
  }
  if (ready)
    ready = cuda_check(cudaMemsetAsync(pair_device_, 0, sizeof(PairSums), stream_), "cudaMemsetAsync");
  if (ready) {
    const PairDiffParams params{.a = address(a),
                                .b = address(b),
                                .width = width_of(a),
                                .height = height_of(a),
                                .levels = 219.0F,  // plage limitée, comme l'ingestion de la chaîne
                                .total = &pair_device_->total,
                                .worst = &pair_device_->worst};
    ready = cuda_check(launch_pair_diff(params, stream_), "pair_diff<<<>>>");
  }
  if (ready) {
    ready = cuda_check(
        cudaMemcpyAsync(pair_host_, pair_device_, sizeof(PairSums), cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync");
  }
  // Attente hôte voulue : la décision de lancer RIFE dépend de cette paire (kernels/spec/detect.h).
  if (ready) ready = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  if (!ready) return std::unexpected(std::move(ready.error()));
  const kernels::RifeGeometry g = kernels::rife_geometry(width_of(a), height_of(a));
  // Σ |ΔY| × (pas 8 bits / 255) : l'écart en codes rapporté à la crête, sur l'aire remplie de RIFE.
  const double scene = pair_host_->total * (219.0 / 255.0) /
                       (static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height));
  return PairScores{
      .scene_score = scene, .worst_block = pair_host_->worst, .identical = pair_host_->total == 0.0};
}

}  // namespace ajp::gpu::cuda
