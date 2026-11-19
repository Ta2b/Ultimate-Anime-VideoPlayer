// La carte de la scène (gpu/kernel_ops.h, scene_flow_start et scene_flow_take) dans le backend TensorRT : les
// lumas du flux écrites sur la voie courante, le flux et la réduction sur la file de NVOF (scene_flow.h).
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/scene_flow.h"
#include "gpu/cuda/backend/scene_flow_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/color.h"

namespace kirari::gpu::cuda {
namespace {

bool rgb_image(const Tensor& t) {
  return (t.type == TensorType::F16 || t.type == TensorType::F32) && t.handle != 0 && t.shape.dims[0] == 1 &&
         t.shape.dims[1] == 3 && t.shape.dims[2] > 0 && t.shape.dims[3] > 0;
}

const void* address(const Tensor& t) {
  return reinterpret_cast<const void*>(t.handle + t.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<std::uint64_t, Refusal> TensorRtBackend::scene_flow_start(const Tensor& a, const Tensor& b,
                                                                        bool restart) {
  if (!rgb_image(a) || !rgb_image(b) || a.type != b.type || a.shape.dims[2] != b.shape.dims[2] ||
      a.shape.dims[3] != b.shape.dims[3]) {
    return refuse(RefusalCode::ShapeInvalid,
                  "scene_flow_start : deux images RGB 1 × 3 × H × W de même taille, FP16 ou FP32, attendues");
  }
  const auto width = static_cast<int>(a.shape.dims[3]);
  const auto height = static_cast<int>(a.shape.dims[2]);
  const int divisor = kernels::scene_flow_divisor(height);
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));
  if (!scene_flow_) scene_flow_ = std::make_unique<SceneFlow>();
  auto fresh = scene_flow_->prepare(width / divisor, height / divisor, divisor);
  if (!fresh) return std::unexpected(std::move(fresh.error()));
  NvofSession& session = scene_flow_->session();
  const NvofSession::Pitched into_a = session.input();
  const NvofSession::Pitched into_b = session.reference();
  std::expected<void, Refusal> ready =
      order({a.handle, b.handle},
            {reinterpret_cast<std::uintptr_t>(into_a.data), reinterpret_cast<std::uintptr_t>(into_b.data)});
  if (ready) ready = scene_flow_->before_lumas(stream_);
  if (ready) {
    const kernels::Coefficients k = kernels::coefficients(kernels::Matrix::Bt709);
    SceneLumaParams params{.rgb = address(a),
                           .fp16 = a.type == TensorType::F16,
                           .width = width,
                           .height = height,
                           .divisor = divisor,
                           .kr = k.kr,
                           .kb = k.kb,
                           .luma = static_cast<unsigned char*>(into_a.data),
                           .pitch = into_a.pitch};
    GpuTiming& timing = stage("carte de la scène : lumas");
    timing.begin(stream_);
    ready = cuda_check(launch_scene_luma(params, stream_), "scene_luma<<<>>>");
    if (ready) {
      params.rgb = address(b);
      params.luma = static_cast<unsigned char*>(into_b.data);
      params.pitch = into_b.pitch;
      ready = cuda_check(launch_scene_luma(params, stream_), "scene_luma<<<>>>");
    }
    timing.end(stream_);
  }
  if (!ready) return std::unexpected(std::move(ready.error()));
  // Une session neuve n'a pas d'indices : la couper ne change rien, et le dit au pilote.
  return scene_flow_->run(stream_, restart || *fresh, stage("carte de la scène : NVOF et tuiles"));
}

std::expected<std::optional<kernels::SceneFlowTiles>, Refusal> TensorRtBackend::scene_flow_take(
    std::uint64_t ticket) {
  if (!scene_flow_)
    return refuse(RefusalCode::SequenceOrder, std::format("scene_flow_take : ticket {} périmé", ticket));
  return scene_flow_->take(ticket);
}

}  // namespace kirari::gpu::cuda
