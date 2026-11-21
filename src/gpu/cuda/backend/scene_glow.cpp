#include "gpu/cuda/backend/scene_glow.h"

#include <algorithm>

namespace kirari::gpu::cuda {

SceneGlow::SceneGlow(int width, int height)
    : width_(width),
      height_(height),
      cols_(glow_div_up(width, kernels::kGlowGuideBlock)),
      rows_(glow_div_up(height, kernels::kGlowGuideBlock)),
      cells_(static_cast<std::size_t>(cols_) * rows_) {}

std::unique_ptr<SceneGlow> SceneGlow::open(int width, int height) {
  if (width <= 0 || height <= 0 || width > 4096 || height > 4096) return {};
  auto guide = std::unique_ptr<SceneGlow>(new SceneGlow(width, height));
  if (!guide->allocate()) return {};
  return guide;
}

bool SceneGlow::allocate() {
  const auto count = static_cast<std::size_t>(glow_div_up(width_, kernels::kGlowAnchorBlock)) *
                     glow_div_up(height_, kernels::kGlowAnchorBlock);
  for (Source& image : sources_) {
    if (cudaMalloc(reinterpret_cast<void**>(&image.aggregates), count * sizeof(GlowAggregate)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&image.anchors), count * sizeof(GpuAnchor)) != cudaSuccess)
      return false;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_), cells_ * sizeof(std::uint32_t)) != cudaSuccess ||
      cudaEventCreateWithFlags(&anchors_ready_, cudaEventDisableTiming) != cudaSuccess)
    return false;
  for (Read& read : reads_) {
    if (cudaMallocHost(reinterpret_cast<void**>(&read.host), cells_ * sizeof(float)) != cudaSuccess ||
        cudaEventCreateWithFlags(&read.done, cudaEventDisableTiming) != cudaSuccess)
      return false;
  }
  return true;
}

SceneGlow::~SceneGlow() {
  if (anchors_ready_) static_cast<void>(cudaEventSynchronize(anchors_ready_));
  for (Read& read : reads_) {
    if (read.done) {
      static_cast<void>(cudaEventSynchronize(read.done));
      static_cast<void>(cudaEventDestroy(read.done));
    }
    if (read.host) static_cast<void>(cudaFreeHost(read.host));
  }
  if (anchors_ready_) static_cast<void>(cudaEventDestroy(anchors_ready_));
  if (device_) static_cast<void>(cudaFree(device_));
  for (Source& source : sources_) {
    if (source.anchors) static_cast<void>(cudaFree(source.anchors));
    if (source.aggregates) static_cast<void>(cudaFree(source.aggregates));
  }
}

void SceneGlow::forget(int a_slot, int b_slot, bool keep_a) {
  if (!keep_a) sources_[a_slot].ready = false;
  sources_[b_slot].ready = false;
}

bool SceneGlow::stage(cudaStream_t lane, int a_slot, int b_slot, bool keep_a, GlowSource a, GlowSource b,
                      const GlowProfile& profile) {
  Source& from = sources_[a_slot];
  Source& into = sources_[b_slot];
  // Même durée de vie et mêmes slots que les lumas. Une lecture refuse sa propre sortie seulement.
  const auto extract = [&](Source& out, GlowSource source) {
    if (profile.reduce) profile.reduce->begin(lane);
    const auto reduced = launch_glow_reduce(source, out.aggregates, lane);
    if (profile.reduce) profile.reduce->end(lane);
    if (reduced != cudaSuccess) return false;
    if (profile.crowns) profile.crowns->begin(lane);
    const auto crowned = launch_glow_crowns(source, out.aggregates, out.anchors, lane);
    if (profile.crowns) profile.crowns->end(lane);
    return crowned == cudaSuccess;
  };
  into.ready = extract(into, b);
  if (!keep_a || !from.ready) from.ready = extract(from, a);
  return cudaEventRecord(anchors_ready_, lane) == cudaSuccess && from.ready && into.ready;
}

bool SceneGlow::match(NvofSession& session, int divisor, int a_slot, int b_slot, std::uint64_t ticket,
                      GpuTiming* timing) {
  const auto stream = session.stream();
  if (cudaStreamWaitEvent(stream, anchors_ready_, 0) != cudaSuccess) return false;
  Read& read = reads_[ticket % reads_.size()];
  if (read.ticket && cudaEventQuery(read.done) != cudaSuccess) return true;  // une relecture en vol
  read.ticket = 0;
  const auto a = session.luma(a_slot), b = session.luma(b_slot), flow = session.flow(), cost = session.cost();
  const GlowMatch match{.width = width_,
                        .height = height_,
                        .divisor = divisor,
                        .anchor_block = kernels::kGlowAnchorBlock,
                        .guide_block = kernels::kGlowGuideBlock,
                        .first = sources_[a_slot].anchors,
                        .second = sources_[b_slot].anchors,
                        .flow = static_cast<const short2*>(flow.data),
                        .flow_pitch = flow.pitch,
                        .cost = static_cast<const std::uint8_t*>(cost.data),
                        .cost_pitch = cost.pitch,
                        .luma_first = static_cast<const std::uint8_t*>(a.data),
                        .luma_first_pitch = a.pitch,
                        .luma_second = static_cast<const std::uint8_t*>(b.data),
                        .luma_second_pitch = b.pitch,
                        .guide = device_,
                        .guide_count = cells_};
  if (timing) timing->begin(stream);
  const cudaError_t matched = launch_glow_match(match, stream);
  if (timing) timing->end(stream);
  if (matched != cudaSuccess) return false;
  if (cudaMemcpyAsync(read.host, device_, cells_ * sizeof(float), cudaMemcpyDeviceToHost, stream) !=
      cudaSuccess)
    return false;
  if (cudaEventRecord(read.done, stream) != cudaSuccess) return false;
  read.ticket = ticket;
  return true;
}

std::optional<kernels::SceneGlowTiles> SceneGlow::take(std::uint64_t ticket) {
  Read& read = reads_[ticket % reads_.size()];
  if (ticket == 0 || read.ticket != ticket || cudaEventQuery(read.done) != cudaSuccess) return std::nullopt;
  read.ticket = 0;
  if (std::none_of(read.host, read.host + cells_, [](float value) { return value > 0.0F; }))
    return std::nullopt;
  return kernels::SceneGlowTiles{.ticket = ticket,
                                 .cols = cols_,
                                 .rows = rows_,
                                 .cover_x = static_cast<double>(cols_ * kernels::kGlowGuideBlock) / width_,
                                 .cover_y = static_cast<double>(rows_ * kernels::kGlowGuideBlock) / height_,
                                 .scores = {read.host, read.host + cells_}};
}

}  // namespace kirari::gpu::cuda
