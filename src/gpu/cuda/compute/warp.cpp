#include "gpu/cuda/compute/warp.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda {
namespace {

dev::Motion motion_view(const MotionView& m) {
  return {.flow = m.flow.ptr,
          .mask = m.mask.ptr,
          .elem = m.flow.elem,
          .plane_width = m.flow.width,
          .plane_height = m.flow.height,
          .x = m.area.x,
          .y = m.area.y,
          .width = m.area.width,
          .height = m.area.height};
}

}  // namespace

Result<> check_motion(const MotionView& motion) {
  const int w = motion.flow.width;
  const int h = motion.flow.height;
  if (!kernels::valid_plane_dimension(w) || !kernels::valid_plane_dimension(h)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("mouvement : plans {}x{} hors de [1, {}]", w, h, kernels::kMaxDimension));
  }
  if (auto ok = check_tensor(motion.flow, w, h, 4, "flot"); !ok) return ok;
  if (auto ok = check_tensor(motion.mask, w, h, 1, "masque"); !ok) return ok;
  if (motion.mask.elem != motion.flow.elem) {
    return refuse(RefusalCode::FormatUnsupported, "mouvement : flot et masque de types différents");
  }
  const kernels::Window& a = motion.area;
  // Écrit sans a.x + a.width, qui pourrait déborder un int.
  if (a.x < 0 || a.y < 0 || a.width < 1 || a.height < 1 || a.width > w - a.x || a.height > h - a.y) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("mouvement : fenêtre {}x{} en ({}, {}) hors des plans {}x{}", a.width, a.height,
                              a.x, a.y, w, h));
  }
  return {};
}

Result<> warp_blend(const Tensor& a, const Tensor& b, const MotionView& motion, float t, const Tensor& out,
                    cudaStream_t stream) {
  if (!(t > 0.0F && t < 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant {} hors de ]0, 1[", t));
  if (!kernels::valid_plane_dimension(out.width) || !kernels::valid_plane_dimension(out.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("warp : sortie {}x{} hors de [1, {}]", out.width,
                                                         out.height, kernels::kMaxDimension));
  }
  for (const Tensor* image : {&a, &b, &out}) {
    if (auto ok = check_tensor(*image, out.width, out.height, 3, "warp"); !ok) return ok;
    if (image->elem != out.elem)
      return refuse(RefusalCode::FormatUnsupported, "warp : a, b et la sortie de types différents");
  }
  if (auto ok = check_motion(motion); !ok) return ok;
  // Le terme de l'instant, évalué en float comme dans la référence.
  const dev::BlendLaw law{.a = kernels::kBlendLawA,
                          .time_term = kernels::kBlendLawB * std::log((1.0F - t) / t),
                          .c = kernels::kBlendLawC,
                          .epsilon = kernels::kMaskEpsilon};
  return cuda_check(dev::warp_blend(a, b, motion_view(motion), t, law, out, stream),
                    "lancement de warp_blend");
}

Result<FlowMax> FlowMax::create(DevicePool& pool) {
  auto device = pool.allocate(sizeof(std::uint32_t), "plus grand déplacement");
  if (!device) return std::unexpected(device.error());
  auto host = PinnedBuffer::create(sizeof(std::uint32_t));
  if (!host) return std::unexpected(host.error());
  auto ready = Event::create();
  if (!ready) return std::unexpected(ready.error());
  FlowMax out;
  out.device_ = std::move(*device);
  out.host_ = std::move(*host);
  out.ready_ = std::move(*ready);
  return out;
}

Result<> FlowMax::launch(const MotionView& motion, cudaStream_t stream) {
  if (auto ok = check_motion(motion); !ok) return ok;
  auto* value = device_.as<std::uint32_t>();
  if (auto ok =
          cuda_check(cudaMemsetAsync(value, 0, sizeof(std::uint32_t), stream), "cudaMemsetAsync (flow_max)");
      !ok) {
    return ok;
  }
  if (auto ok = cuda_check(dev::flow_max(motion_view(motion), value, stream), "lancement de flow_max"); !ok)
    return ok;
  if (auto ok = cuda_check(
          cudaMemcpyAsync(host_.data(), value, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync (flow_max)");
      !ok) {
    return ok;
  }
  if (auto ok = ready_.record(stream); !ok) return ok;
  pending_ = true;
  return {};
}

Result<std::optional<float>> FlowMax::poll() {
  if (!pending_) return std::nullopt;
  auto done = ready_.done();
  if (!done) return std::unexpected(done.error());
  if (!*done) return std::nullopt;
  pending_ = false;
  float value = 0.0F;
  std::memcpy(&value, host_.data(), sizeof(value));
  return value;
}

}  // namespace ajp::gpu::cuda
