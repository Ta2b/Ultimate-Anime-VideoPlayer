// La carte de la scène (gpu/kernel_ops.h, scene_flow_start et scene_flow_take) dans le backend TensorRT : la
// luma de b du flux écrite sur la voie courante (et celle de a si la paire ne suit pas la précédente), le
// flux et la réduction sur la file de NVOF (scene_flow.h).
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
  // Une session neuve n'a pas d'indices : la couper ne change rien, et le dit au pilote. Sans coupure, a est
  // l'image b de la paire d'avant (gpu/kernel_ops.h) : sa luma est déjà dans un tampon de NVOF.
  const bool cold = restart || *fresh;
  // NVOF lit encore un tampon à écrire : la paire n'est pas lancée, la voie n'attend pas (ticket 0).
  auto plan = scene_flow_->plan(!cold);
  if (!plan) return std::unexpected(std::move(plan.error()));
  if (!*plan) return std::uint64_t{0};
  const SceneFlow::Lumas lumas = **plan;
  const NvofSession& session = scene_flow_->session();
  const NvofSession::Pitched into_a = session.luma(lumas.a);
  const NvofSession::Pitched into_b = session.luma(lumas.b);
  const auto slot_a = reinterpret_cast<std::uintptr_t>(into_a.data);
  const auto slot_b = reinterpret_cast<std::uintptr_t>(into_b.data);
  std::expected<void, Refusal> ready =
      lumas.keep_a ? order({b.handle}, {slot_b}) : order({a.handle, b.handle}, {slot_a, slot_b});
  if (ready) ready = scene_flow_->before_lumas(stream_, lumas);
  if (ready) {
    const kernels::Coefficients k = kernels::coefficients(kernels::Matrix::Bt709);
    SceneLumaParams params{.rgb = address(b),
                           .fp16 = b.type == TensorType::F16,
                           .width = width,
                           .height = height,
                           .divisor = divisor,
                           .kr = k.kr,
                           .kb = k.kb,
                           .luma = static_cast<unsigned char*>(into_b.data),
                           .pitch = into_b.pitch};
    GpuTiming& timing = stage("carte de la scène : lumas");
    timing.begin(stream_);
    ready = cuda_check(launch_scene_luma(params, stream_), "scene_luma<<<>>>");
    if (ready && !lumas.keep_a) {
      params.rgb = address(a);
      params.luma = static_cast<unsigned char*>(into_a.data);
      params.pitch = into_a.pitch;
      ready = cuda_check(launch_scene_luma(params, stream_), "scene_luma<<<>>>");
    }
    timing.end(stream_);
  }
  if (!ready) return std::unexpected(std::move(ready.error()));
  // Le flux tourne sur le bloc NVOFA, à côté des SM : sa mesure est une latence, pas un temps ajouté à chaque
  // image. Ce qui s'ajoute, ce sont les lumas et les tuiles (docs/carte-de-la-scene.md).
  return scene_flow_->run(stream_, lumas, cold, stage("carte de la scène : flux (NVOFA)"),
                          stage("carte de la scène : tuiles"));
}

std::expected<std::optional<kernels::SceneFlowTiles>, Refusal> TensorRtBackend::scene_flow_take(
    std::uint64_t ticket) {
  if (!scene_flow_)
    return refuse(RefusalCode::SequenceOrder, std::format("scene_flow_take : ticket {} périmé", ticket));
  return scene_flow_->take(ticket);
}

std::expected<std::uint64_t, Refusal> TensorRtBackend::scene_camera_start(
    std::uint64_t flow_ticket, const kernels::SceneCameraRequest& request) {
  if (!scene_flow_) return std::uint64_t{0};  // aucune paire lancée : rien à affiner
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));
  // Sur la file de NVOF, sur les SM : un coût par paire, mesuré à part.
  return scene_flow_->refine(flow_ticket, request, stage("carte de la scène : caméra"));
}

std::expected<std::optional<kernels::SceneCameraRefined>, Refusal> TensorRtBackend::scene_camera_take(
    std::uint64_t ticket) {
  if (!scene_flow_)
    return refuse(RefusalCode::SequenceOrder, std::format("scene_camera_take : ticket {} périmé", ticket));
  return scene_flow_->refined(ticket);
}

}  // namespace kirari::gpu::cuda
