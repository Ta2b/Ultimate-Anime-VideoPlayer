// Warp et mélange par le mouvement de RIFE (warp_kernels.h) dans le backend TensorRT : warp_blend
// du contrat de KernelOps, warp_blend_poses (garder les poses) et flow_max_now, extensions de la
// tranche verticale ; leurs variantes _ring écrivent l'image directement dans le slot.
#include <array>
#include <cstring>
#include <format>
#include <span>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda/backend/warp_kernels.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::size_t kHistogramBytes = kernels::kFlowBins * sizeof(unsigned int);

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

// Le mouvement de la paire précédente, s'il y en a un : mêmes plans et même fenêtre que motion.
std::expected<void, Refusal> check_previous(const Motion& motion, const Motion* previous) {
  if (previous == nullptr) return {};
  if (auto ok = check_motion(*previous); !ok) return ok;
  if (previous->flow.shape.dims != motion.flow.shape.dims || previous->area.x != motion.area.x ||
      previous->area.y != motion.area.y || previous->area.width != motion.area.width ||
      previous->area.height != motion.area.height) {
    return refuse(RefusalCode::ShapeInvalid, "warp_blend_poses : mouvements de deux paires différents");
  }
  return {};
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::warp_blend(const Tensor& a, const Tensor& b,
                                                         const Motion& motion, float t, const Tensor& out) {
  return warp(a, b, motion, Poses{}, t, Target{.out = &out});
}

std::expected<void, Refusal> TensorRtBackend::warp_blend_poses(const Tensor& a, const Tensor& b,
                                                               const Motion& motion, const Motion* previous,
                                                               float t, const Tensor& out) {
  if (auto ok = check_previous(motion, previous); !ok) return ok;
  return warp(a, b, motion, Poses{.keep = true, .previous = previous}, t, Target{.out = &out});
}

std::expected<void, Refusal> TensorRtBackend::warp_blend_ring(const Tensor& a, const Tensor& b,
                                                              const Motion& motion, float t,
                                                              const RingTarget& ring) {
  return warp(a, b, motion, Poses{}, t, Target{.ring = &ring});
}

std::expected<void, Refusal> TensorRtBackend::warp_blend_poses_ring(const Tensor& a, const Tensor& b,
                                                                    const Motion& motion,
                                                                    const Motion* previous, float t,
                                                                    const RingTarget& ring) {
  if (auto ok = check_previous(motion, previous); !ok) return ok;
  return warp(a, b, motion, Poses{.keep = true, .previous = previous}, t, Target{.ring = &ring});
}

std::expected<void, Refusal> TensorRtBackend::warp(const Tensor& a, const Tensor& b, const Motion& motion,
                                                   const Poses& poses, float t, const Target& target) {
  const Tensor* out = target.out;
  if (!fp16_planes(a, 3) || a.shape.dims != b.shape.dims || b.type != TensorType::F16 || b.handle == 0 ||
      (out != nullptr &&
       (a.shape.dims != out->shape.dims || out->type != TensorType::F16 || out->handle == 0))) {
    return refuse(RefusalCode::ShapeInvalid, "warp_blend : trois images RGB FP16 de même taille attendues");
  }
  if (out == nullptr &&
      (target.ring == nullptr || target.ring->slot == nullptr || target.ring->slot->width != width_of(a) ||
       target.ring->slot->height != height_of(a))) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("warp_blend_ring : slot de la taille des images {}x{} attendu", width_of(a),
                              height_of(a)));
  }
  if (!(t > 0.0F && t < 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("warp_blend : instant {}", t));
  if (auto ok = check_motion(motion); !ok) return ok;
  if (auto ok = bind(); !ok) return ok;
  trim_textures();
  const WarpBlendParams params{
      .a = address(a),
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
      .from_middle = poses.keep || t != 0.5F,
      .keep_poses = poses.keep,
      .previous_flow = poses.previous != nullptr ? address(poses.previous->flow) : nullptr,
      .out = out != nullptr ? const_cast<void*>(address(*out)) : nullptr,  // NOLINT : écrit
      .textures = {.a = texture(a),
                   .b = texture(b),
                   .flow = texture(motion.flow),
                   .mask = texture(motion.mask),
                   .previous_flow = poses.previous != nullptr ? texture(poses.previous->flow) : 0}};
  const std::uintptr_t previous = poses.previous != nullptr ? poses.previous->flow.handle : 0;
  const std::array reads{a.handle, b.handle, motion.flow.handle, motion.mask.handle, previous};
  const std::array writes{out != nullptr ? out->handle : 0};
  if (auto ok = order(reads, std::span(writes).first(out != nullptr ? 1 : 0)); !ok) return ok;
  return launch_into(target, "warp et mélange", [&](cudaSurfaceObject_t surface) {
    WarpBlendParams into = params;
    into.surface = surface;
    return cuda_check(launch_warp_blend(into, stream_), "warp_blend<<<>>>");
  });
}

std::expected<float, Refusal> TensorRtBackend::flow_max_now(const Motion& motion) {
  return flow_max_start(motion).and_then([&](std::uint64_t ticket) { return flow_max_take(ticket); });
}

std::expected<std::uint64_t, Refusal> TensorRtBackend::flow_max_start(const Motion& motion) {
  if (auto ok = check_motion(motion); !ok) return std::unexpected(std::move(ok.error()));
  const std::uint64_t ticket = flow_tickets_ + 1;
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  std::expected<void, Refusal> ready = bind();
  if (ready && slot.device == nullptr) {
    ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device), kHistogramBytes), "cudaMalloc");
    if (ready)
      ready =
          cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), kHistogramBytes), "cudaMallocHost");
    if (ready)
      ready = cuda_check(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming), "cudaEventCreate");
  }
  if (ready) ready = order({motion.flow.handle}, {reinterpret_cast<std::uintptr_t>(slot.device)});
  if (ready) ready = cuda_check(cudaMemsetAsync(slot.device, 0, kHistogramBytes, stream_), "cudaMemsetAsync");
  if (ready) {
    const FlowMaxParams params{.flow = address(motion.flow),
                               .plane_width = width_of(motion.flow),
                               .plane_height = height_of(motion.flow),
                               .window_x = motion.area.x,
                               .window_y = motion.area.y,
                               .window_width = motion.area.width,
                               .window_height = motion.area.height,
                               .result = slot.device};
    ready = cuda_check(launch_flow_max(params, stream_), "flow_max<<<>>>");
  }
  if (ready) {
    ready =
        cuda_check(cudaMemcpyAsync(slot.host, slot.device, kHistogramBytes, cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync");
  }
  if (ready) ready = cuda_check(cudaEventRecord(slot.done, stream_), "cudaEventRecord");
  if (!ready) return std::unexpected(std::move(ready.error()));
  flow_tickets_ = ticket;
  slot.ticket = ticket;
  slot.points =
      static_cast<std::uint64_t>(motion.area.width) * static_cast<std::uint64_t>(motion.area.height);
  return ticket;
}

std::expected<float, Refusal> TensorRtBackend::flow_max_take(std::uint64_t ticket) {
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket)
    return refuse(RefusalCode::SequenceOrder, std::format("flow_max_take : ticket {} périmé", ticket));
  // Attente hôte voulue : relancer RIFE aux autres instants dépend de ce maximum.
  if (auto ok = wait_event(slot.done); !ok) return std::unexpected(std::move(ok.error()));
  slot.ticket = 0;
  return kernels::flow_percentile(slot.host, slot.points);
}

}  // namespace ajp::gpu::cuda
