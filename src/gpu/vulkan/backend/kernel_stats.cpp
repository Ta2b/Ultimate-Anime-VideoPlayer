// L'analyse d'une image pour le grade (gpu/kernel_ops.h, frame_stats_start) : le noyau frame_stats
// sur la voie courante, ses comptes relus par l'hôte sans copie et sans attente, puis tirés en mesures
// par kernels/ref/frame_stats.h (la même fonction que la pré-analyse des plans).
#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "kernels/ref/frame_stats.h"

namespace ajp::gpu::vk {
namespace {

std::unexpected<Refusal> refuse_stats(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

std::expected<std::uint64_t, Refusal> VulkanKernelOps::frame_stats_start(const Tensor& rgb,
                                                                         kernels::FrameTransfer transfer) {
  auto view = detail::tensor_view(rgb, 3, "frame_stats_start : image");
  if (!view) return std::unexpected(view.error());
  const std::uint64_t ticket = stats_tickets_ + 1;
  StatsRead& slot = stats_reads_[ticket % stats_reads_.size()];
  auto counts_buffer = readback(slot.counts, sizeof(kernels::FrameCounts), slot.value);
  if (!counts_buffer) return std::unexpected(counts_buffer.error());
  const BufferView counts = (*counts_buffer)->view();
  OpKey key;
  key.add(OpCode::FrameStats);
  detail::add_view(key, *view);
  key.add(static_cast<std::uint64_t>(transfer)).add(detail::handle_bits(counts.buffer));
  auto rec = recording(key, [&](Recorder& r) { return runner_.frame_stats(r, *view, transfer, counts); });
  if (!rec) return std::unexpected(rec.error());
  if (auto ok = submit(**rec); !ok) return std::unexpected(ok.error());
  if (auto ok = queue_.flush(); !ok) return std::unexpected(ok.error());  // comme scores_start
  stats_tickets_ = ticket;
  slot.ticket = ticket;
  slot.value = queue_.ticket();
  return ticket;
}

std::expected<std::optional<kernels::FrameStats>, Refusal> VulkanKernelOps::frame_stats_take(
    std::uint64_t ticket) {
  StatsRead& slot = stats_reads_[ticket % stats_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket || !slot.counts) {
    return refuse_stats(RefusalCode::SequenceOrder,
                        std::format("frame_stats_take : ticket {} périmé", ticket));
  }
  // Jamais d'attente : le grade prend l'analyse d'une image dès qu'elle est finie.
  const auto completed = queue_.completed();
  if (!completed) return std::unexpected(completed.error());
  if (*completed < slot.value) return std::optional<kernels::FrameStats>{};
  slot.ticket = 0;
  kernels::FrameCounts counts;
  std::memcpy(&counts, slot.counts->mapped(), sizeof(counts));
  return std::optional(kernels::ref::frame_stats_of(counts));
}

}  // namespace ajp::gpu::vk
