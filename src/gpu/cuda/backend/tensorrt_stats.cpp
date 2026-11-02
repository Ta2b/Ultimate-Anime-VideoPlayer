// L'analyse d'une image pour le grade (gpu/kernel_ops.h, frame_stats_start) dans le backend
// TensorRT : le noyau frame_stats (stats_kernels.h) sur la voie courante, ses comptes copiés dans une
// mémoire hôte épinglée, puis tirés en mesures par kernels/ref/frame_stats.h quand l'événement de la
// copie est passé ; la relecture n'attend jamais.
#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/stats_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/ref/frame_stats.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::size_t kCountsBytes = sizeof(kernels::FrameCounts);

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<std::uint64_t, Refusal> TensorRtBackend::frame_stats_start(const Tensor& rgb,
                                                                         kernels::FrameTransfer transfer) {
  if ((rgb.type != TensorType::F16 && rgb.type != TensorType::F32) || rgb.handle == 0 ||
      rgb.shape.dims[0] != 1 || rgb.shape.dims[1] != 3 || rgb.shape.dims[2] <= 0 || rgb.shape.dims[3] <= 0) {
    return refuse(RefusalCode::ShapeInvalid,
                  "frame_stats_start : image RGB 1 × 3 × H × W FP16 ou FP32 attendue");
  }
  const std::uint64_t ticket = stats_tickets_ + 1;
  StatsRead& slot = stats_reads_[ticket % stats_reads_.size()];
  std::expected<void, Refusal> ready = bind();
  if (ready && slot.device == nullptr) {
    ready = cuda_check(cudaMalloc(&slot.device, kCountsBytes), "cudaMalloc");
    if (ready) ready = cuda_check(cudaMallocHost(&slot.host, kCountsBytes), "cudaMallocHost");
    if (ready)
      ready = cuda_check(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming), "cudaEventCreate");
  }
  if (ready) ready = order({rgb.handle}, {reinterpret_cast<std::uintptr_t>(slot.device)});
  if (ready) {
    const FrameStatsParams params{
        .rgb = reinterpret_cast<const void*>(rgb.handle + rgb.offset),  // NOLINT(performance-no-int-to-ptr)
        .fp16 = rgb.type == TensorType::F16,
        .width = static_cast<int>(rgb.shape.dims[3]),
        .height = static_cast<int>(rgb.shape.dims[2]),
        .transfer = static_cast<int>(transfer),
        .counts = slot.device};
    ready = cuda_check(launch_frame_stats(params, stream_), "frame_stats<<<>>>");
  }
  if (ready) {
    ready = cuda_check(cudaMemcpyAsync(slot.host, slot.device, kCountsBytes, cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync");
  }
  if (ready) ready = cuda_check(cudaEventRecord(slot.done, stream_), "cudaEventRecord");
  if (!ready) return std::unexpected(std::move(ready.error()));
  stats_tickets_ = ticket;
  slot.ticket = ticket;
  return ticket;
}

std::expected<std::optional<kernels::FrameStats>, Refusal> TensorRtBackend::frame_stats_take(
    std::uint64_t ticket) {
  StatsRead& slot = stats_reads_[ticket % stats_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket || slot.host == nullptr)
    return refuse(RefusalCode::SequenceOrder, std::format("frame_stats_take : ticket {} périmé", ticket));
  // Jamais d'attente : le grade prend l'analyse d'une image dès qu'elle est finie.
  const cudaError_t state = cudaEventQuery(slot.done);
  if (state == cudaErrorNotReady) return std::optional<kernels::FrameStats>{};
  if (auto ok = cuda_check(state, "cudaEventQuery"); !ok) return std::unexpected(std::move(ok.error()));
  slot.ticket = 0;
  kernels::FrameCounts counts;
  std::memcpy(&counts, slot.host, sizeof(counts));
  return std::optional(kernels::ref::frame_stats_of(counts));
}

}  // namespace ajp::gpu::cuda
