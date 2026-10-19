// Warp et mélange par le mouvement de RIFE (warp_kernels.h) dans le backend TensorRT : warp_blend
// du contrat de KernelOps, et flow_max_now, extension de la tranche verticale.
#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda/backend/warp_kernels.h"

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

const void* address(const Tensor& tensor) {
  return reinterpret_cast<const void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

// Flot 1 × 4 et masque 1 × 1 de même taille, fenêtre non vide incluse dans leurs plans.
std::expected<void, Refusal> check_motion(const Motion& motion) {
  if (!fp16_planes(motion.flow, 4) || !fp16_planes(motion.mask, 1) ||
      motion.flow.shape.dims[2] != motion.mask.shape.dims[2] ||
      motion.flow.shape.dims[3] != motion.mask.shape.dims[3]) {
    return refuse(RefusalCode::ShapeInvalid,
                  "mouvement : flot FP16 1×4 et masque 1×1 de même taille attendus");
  }
  const kernels::Window& w = motion.area;
  if (w.x < 0 || w.y < 0 || w.width < 1 || w.height < 1 || w.width > width_of(motion.flow) - w.x ||
      w.height > height_of(motion.flow) - w.y) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("mouvement : fenêtre {}x{} en ({}, {}) hors des plans {}x{}", w.width, w.height,
                              w.x, w.y, width_of(motion.flow), height_of(motion.flow)));
  }
  return {};
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::warp_blend(const Tensor& a, const Tensor& b,
                                                         const Motion& motion, float t, const Tensor& out) {
  if (!fp16_planes(a, 3) || a.shape.dims != b.shape.dims || a.shape.dims != out.shape.dims ||
      b.type != TensorType::F16 || out.type != TensorType::F16 || b.handle == 0 || out.handle == 0) {
    return refuse(RefusalCode::ShapeInvalid, "warp_blend : trois images RGB FP16 de même taille attendues");
  }
  if (!(t > 0.0F && t < 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("warp_blend : instant {}", t));
  if (auto ok = check_motion(motion); !ok) return ok;
  if (auto ok = bind(); !ok) return ok;
  const WarpBlendParams params{.a = address(a),
                               .b = address(b),
                               .width = width_of(a),
                               .height = height_of(a),
                               .flow = address(motion.flow),
                               .mask = address(motion.mask),
                               .plane_width = width_of(motion.flow),
                               .plane_height = height_of(motion.flow),
                               .window_x = motion.area.x,
                               .window_y = motion.area.y,
                               .window_width = motion.area.width,
                               .window_height = motion.area.height,
                               .time = t,
                               .from_middle = t != 0.5F,
                               .out = const_cast<void*>(address(out))};  // NOLINT : écrit
  GpuTiming& timing = stage("warp et mélange");
  timing.begin(stream_);
  auto launched = cuda_check(launch_warp_blend(params, stream_), "warp_blend<<<>>>");
  timing.end(stream_);
  return launched;
}

std::expected<float, Refusal> TensorRtBackend::flow_max_now(const Motion& motion) {
  if (auto ok = check_motion(motion); !ok) return std::unexpected(std::move(ok.error()));
  std::expected<void, Refusal> ready = bind();
  if (ready && flow_device_ == nullptr) {
    ready =
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&flow_device_), sizeof(unsigned int)), "cudaMalloc");
    if (ready)
      ready = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&flow_host_), sizeof(unsigned int)),
                         "cudaMallocHost");
  }
  if (ready)
    ready = cuda_check(cudaMemsetAsync(flow_device_, 0, sizeof(unsigned int), stream_), "cudaMemsetAsync");
  if (ready) {
    const FlowMaxParams params{.flow = address(motion.flow),
                               .plane_width = width_of(motion.flow),
                               .plane_height = height_of(motion.flow),
                               .window_x = motion.area.x,
                               .window_y = motion.area.y,
                               .window_width = motion.area.width,
                               .window_height = motion.area.height,
                               .result = flow_device_};
    ready = cuda_check(launch_flow_max(params, stream_), "flow_max<<<>>>");
  }
  if (ready) {
    ready = cuda_check(
        cudaMemcpyAsync(flow_host_, flow_device_, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync");
  }
  // Attente hôte voulue : relancer RIFE aux autres instants dépend de ce maximum.
  if (ready) ready = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  if (!ready) return std::unexpected(std::move(ready.error()));
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(*flow_host_));
  std::memcpy(&value, flow_host_, sizeof(value));
  return value;
}

}  // namespace ajp::gpu::cuda
