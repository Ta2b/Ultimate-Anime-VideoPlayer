// Le débandage de la source, et l'analyse d'une image pour le grade (gpu/kernel_ops.h, frame_stats_start) :
// le noyau frame_stats sur la voie courante, ses comptes relus par l'hôte sans copie et sans attente, puis
// tirés en mesures par kernels/ref/frame_stats.h (la même fonction que la pré-analyse des plans). Aussi la
// mémoire de la carte pour les statistiques (device_memory).
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <utility>

#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/memory.h"
#include "kernels/ref/frame_stats.h"

namespace ajp::gpu::vk {
namespace {

std::unexpected<Refusal> refuse_stats(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

std::optional<KernelOps::DeviceMemory> VulkanKernelOps::device_memory() {
  const std::optional<HeapUse> use = device_heap_use(context_);
  if (!use) return std::nullopt;
  return DeviceMemory{.used = use->used, .total = use->total};
}

// Le débandage de la source avant SPAN (gpu/kernel_ops.h) : comme TensorRT, l'écran ne débande plus l'image
// agrandie à chaque présentation.
std::expected<void, Refusal> VulkanKernelOps::deband(const Tensor& rgb) {
  auto view = detail::tensor_view(rgb, 3, "débandage : image");
  if (!view) return std::unexpected(view.error());
  OpKey key;
  key.add(OpCode::Deband);
  detail::add_view(key, *view);
  auto rec = recording(key, [&](Recorder& r) { return runner_.deband(r, *view); });
  if (!rec) return std::unexpected(rec.error());
  return submit("débandage", **rec);
}

std::expected<void, Refusal> VulkanKernelOps::tensor_peaks(std::span<const Tensor> tensors,
                                                           const kernels::Window& area,
                                                           const BufferView& result) {
  for (std::size_t k = 0; k < tensors.size(); ++k) {
    auto view = detail::tensor_view(tensors[k], 0, "maximum d'un tenseur");
    if (!view) return std::unexpected(view.error());
    OpKey key;
    key.add(OpCode::TensorPeak);
    detail::add_view(key, *view);
    key.add(static_cast<std::uint64_t>(area.x)).add(static_cast<std::uint64_t>(area.y));
    key.add(static_cast<std::uint64_t>(area.width)).add(static_cast<std::uint64_t>(area.height));
    key.add(detail::handle_bits(result.buffer)).add(result.offset).add(k);
    const auto slot = static_cast<std::uint32_t>(k);
    auto rec = recording(key, [&](Recorder& r) { return runner_.tensor_peak(r, *view, area, result, slot); });
    if (!rec) return std::unexpected(rec.error());
    if (auto ok = submit("maximum des tenseurs", **rec); !ok) return ok;
  }
  return {};
}

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
  if (auto ok = submit("analyse de l'image", **rec); !ok) return std::unexpected(ok.error());
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
