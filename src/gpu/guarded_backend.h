// Mandataire posé par chaque bibliothèque de backend autour du backend qu'elle rend (P2.4) : toute
// virtuelle de Backend et de KernelOps passe par un try, et une exception levée dans la bibliothèque
// devient un refus nommé (VendorCallFailed, qui cite l'appel) avant de traverser la frontière du module.
// Les appels qui ne rendent pas de refus rendent la valeur prudente (Unknown, nullopt, false, vide).
// Les moteurs ne sont pas enveloppés : le backend reconnaît les siens par dynamic_cast, et leurs
// virtuelles ne font que rendre des noms.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"

namespace ajp::gpu {
namespace guarded_detail {

template <class T>
inline constexpr bool kIsExpected = false;
template <class T, class E>
inline constexpr bool kIsExpected<std::expected<T, E>> = true;

// fn(), ou le refus de l'exception si fn lève ; un résultat qui n'est pas un std::expected rend alors
// sa valeur par défaut (nullopt, false, vide).
template <class Fn, class R = std::invoke_result_t<Fn>>
R call(std::string_view where, Fn&& fn) noexcept {
  try {
    return std::forward<Fn>(fn)();
  } catch (...) {
    if constexpr (kIsExpected<R>) {
      return std::unexpected(refusal_from_exception(RefusalCode::VendorCallFailed, where));
    } else {
      return R{};
    }
  }
}

template <class Fn>
void call_void(Fn&& fn) noexcept {
  try {
    std::forward<Fn>(fn)();
  } catch (...) {  // NOLINT(bugprone-empty-catch) : rien à rendre, l'appel suivant refusera
  }
}

}  // namespace guarded_detail

class GuardedKernelOps final : public KernelOps {
 public:
  explicit GuardedKernelOps(KernelOps* inner) : in_(inner) {}

  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                      const Tensor& rgb) override {
    return guarded_detail::call("ingest", [&] { return in_->ingest(frame, spec, rgb); });
  }
  std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter,
                                        const Tensor& out) override {
    return guarded_detail::call("resample", [&] { return in_->resample(in, filter, out); });
  }
  std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) override {
    return guarded_detail::call("rife_pad", [&] { return in_->rife_pad(frame, padded); });
  }
  std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) override {
    return guarded_detail::call("rife_crop", [&] { return in_->rife_crop(padded, out); });
  }
  std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion, float t,
                                          const Tensor& out) override {
    return guarded_detail::call("warp_blend", [&] { return in_->warp_blend(a, b, motion, t, out); });
  }
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override {
    return guarded_detail::call("write_ring", [&] { return in_->write_ring(rgb, slot, sync); });
  }
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override {
    return guarded_detail::call("measure", [&] { return in_->measure(a, b); });
  }
  std::optional<PairScores> previous_scores() override {
    return guarded_detail::call("previous_scores", [&] { return in_->previous_scores(); });
  }
  std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                          const Tensor& input) override {
    return guarded_detail::call("rife_input", [&] { return in_->rife_input(a, b, t, input); });
  }
  std::expected<void, Refusal> rife_time(const Tensor& input, float t) override {
    return guarded_detail::call("rife_time", [&] { return in_->rife_time(input, t); });
  }
  std::expected<void, Refusal> rife_pair(const Tensor& a, const Tensor& b, float t,
                                         const Tensor& input) override {
    return guarded_detail::call("rife_pair", [&] { return in_->rife_pair(a, b, t, input); });
  }
  std::expected<PairScores, Refusal> scores_now(const Tensor& a, const Tensor& b) override {
    return guarded_detail::call("scores_now", [&] { return in_->scores_now(a, b); });
  }
  std::expected<std::uint64_t, Refusal> scores_start(const Tensor& a, const Tensor& b) override {
    return guarded_detail::call("scores_start", [&] { return in_->scores_start(a, b); });
  }
  std::expected<PairScores, Refusal> scores_take(std::uint64_t ticket) override {
    return guarded_detail::call("scores_take", [&] { return in_->scores_take(ticket); });
  }
  std::expected<float, Refusal> flow_max_now(const Motion& motion) override {
    return guarded_detail::call("flow_max_now", [&] { return in_->flow_max_now(motion); });
  }
  std::expected<std::uint64_t, Refusal> flow_max_start(const Motion& motion) override {
    return guarded_detail::call("flow_max_start", [&] { return in_->flow_max_start(motion); });
  }
  std::expected<float, Refusal> flow_max_take(std::uint64_t ticket) override {
    return guarded_detail::call("flow_max_take", [&] { return in_->flow_max_take(ticket); });
  }
  std::expected<void, Refusal> warp_blend_poses(const Tensor& a, const Tensor& b, const Motion& motion,
                                                const Motion* previous, float t, const Tensor& out) override {
    return guarded_detail::call("warp_blend_poses",
                                [&] { return in_->warp_blend_poses(a, b, motion, previous, t, out); });
  }
  std::expected<void, Refusal> warp_blend_ring(const Tensor& a, const Tensor& b, const Motion& motion,
                                               float t, const RingTarget& target) override {
    return guarded_detail::call("warp_blend_ring",
                                [&] { return in_->warp_blend_ring(a, b, motion, t, target); });
  }
  std::expected<void, Refusal> warp_blend_poses_ring(const Tensor& a, const Tensor& b, const Motion& motion,
                                                     const Motion* previous, float t,
                                                     const RingTarget& target) override {
    return guarded_detail::call("warp_blend_poses_ring", [&] {
      return in_->warp_blend_poses_ring(a, b, motion, previous, t, target);
    });
  }
  std::expected<void, Refusal> write_yuv(const Tensor& rgb, const YuvTarget& target) override {
    return guarded_detail::call("write_yuv", [&] { return in_->write_yuv(rgb, target); });
  }
  std::expected<void, Refusal> deband(const Tensor& rgb) override {
    return guarded_detail::call("deband", [&] { return in_->deband(rgb); });
  }
  bool debands_on_ingest() const override {
    return guarded_detail::call("debands_on_ingest", [&] { return in_->debands_on_ingest(); });
  }
  std::optional<DeviceMemory> device_memory() override {
    return guarded_detail::call("device_memory", [&] { return in_->device_memory(); });
  }
  std::expected<std::uint64_t, Refusal> frame_stats_start(const Tensor& rgb,
                                                          kernels::FrameTransfer transfer) override {
    return guarded_detail::call("frame_stats_start", [&] { return in_->frame_stats_start(rgb, transfer); });
  }
  std::expected<std::optional<kernels::FrameStats>, Refusal> frame_stats_take(std::uint64_t ticket) override {
    return guarded_detail::call("frame_stats_take", [&] { return in_->frame_stats_take(ticket); });
  }
  std::expected<void, Refusal> health() override {
    return guarded_detail::call("health", [&] { return in_->health(); });
  }
  std::vector<GpuStageSummary> gpu_profile() override {
    return guarded_detail::call("gpu_profile", [&] { return in_->gpu_profile(); });
  }
  bool map_fences(std::span<std::byte> block) override {
    return guarded_detail::call("map_fences", [&] { return in_->map_fences(block); });
  }
  void unmap_fences(std::span<std::byte> block) override {
    guarded_detail::call_void([&] { in_->unmap_fences(block); });
  }

 private:
  KernelOps* in_;
};

class GuardedBackend final : public Backend {
 public:
  // Les noyaux vivent autant que le backend : leur mandataire est fait une fois, ici.
  explicit GuardedBackend(std::unique_ptr<Backend> inner) : in_(std::move(inner)) {
    if (KernelOps* ops = in_->kernels(); ops != nullptr) kernels_ = std::make_unique<GuardedKernelOps>(ops);
  }

  BackendCaps capabilities() const override {
    return guarded_detail::call("capabilities", [&] { return in_->capabilities(); });
  }
  DeviceMatch match(const DeviceUuid& render_device) const override {
    try {
      return in_->match(render_device);
    } catch (...) {
      return DeviceMatch::Unknown;
    }
  }
  std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                         Progress& progress) override {
    return guarded_detail::call("build", [&] { return in_->build(model, shapes, progress); });
  }
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) override {
    return guarded_detail::call("load", [&] { return in_->load(artifact); });
  }
  std::expected<Tensor, Refusal> import(const ExternalImage& image) override {
    return guarded_detail::call("import", [&] { return in_->import(image); });
  }
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override {
    return guarded_detail::call("alloc_tensor", [&] { return in_->alloc_tensor(type, shape); });
  }
  KernelOps* kernels() override { return kernels_.get(); }
  std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                       std::span<Tensor> outputs) override {
    return guarded_detail::call("enqueue", [&] { return in_->enqueue(engine, inputs, outputs); });
  }
  std::expected<Fence, Refusal> signal() override {
    return guarded_detail::call("signal", [&] { return in_->signal(); });
  }
  std::expected<void, Refusal> wait(const Fence& fence) override {
    return guarded_detail::call("wait", [&] { return in_->wait(fence); });
  }
  void drain() override {
    guarded_detail::call_void([&] { in_->drain(); });
  }
  void use_lane(Lane lane) override {
    guarded_detail::call_void([&] { in_->use_lane(lane); });
  }
  std::expected<SpanFp8Derivation, Refusal> derive_span_fp8(const SpanFp8Request& request,
                                                            Progress& progress) override {
    return guarded_detail::call("derive_span_fp8", [&] { return in_->derive_span_fp8(request, progress); });
  }
  std::expected<void, Refusal> enqueue_ring(Engine& engine, std::span<const Tensor> inputs,
                                            std::span<Tensor> outputs, const RingTarget& target) override {
    return guarded_detail::call("enqueue_ring",
                                [&] { return in_->enqueue_ring(engine, inputs, outputs, target); });
  }

 private:
  std::unique_ptr<Backend> in_;
  std::unique_ptr<GuardedKernelOps> kernels_;
};

}  // namespace ajp::gpu
