#include "gpu/cuda/backend/host_upload.h"

#include <cstring>
#include <format>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gpu_timing.h"

namespace ajp::gpu::cuda {
namespace {

std::size_t pitch_of(std::size_t row_bytes) {
  return (row_bytes + HostUploads::kPitchAlign - 1) / HostUploads::kPitchAlign * HostUploads::kPitchAlign;
}

}  // namespace

std::optional<HostUploads::Layout> HostUploads::layout(PixelFormat format, int width, int height) {
  if (width < 1 || height < 1) return std::nullopt;
  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  const std::size_t cw = (w + 1) / 2;
  const std::size_t ch = (h + 1) / 2;
  switch (format) {
    case PixelFormat::Nv12:
      return Layout{.planes = 2, .row_bytes = {w, 2 * cw, 0}, .rows = {h, ch, 0}};
    case PixelFormat::P010:
    case PixelFormat::P016:
      return Layout{.planes = 2, .row_bytes = {2 * w, 4 * cw, 0}, .rows = {h, ch, 0}};
    case PixelFormat::Yuv420p:
      return Layout{.planes = 3, .row_bytes = {w, cw, cw}, .rows = {h, ch, ch}};
    case PixelFormat::Yuv420p10:
      return Layout{.planes = 3, .row_bytes = {2 * w, 2 * cw, 2 * cw}, .rows = {h, ch, ch}};
    default:
      return std::nullopt;
  }
}

HostUploads::~HostUploads() {
  if (transfer_ != nullptr) cudaStreamSynchronize(transfer_);
  for (const Slot& slot : slots_) {
    if (slot.read != nullptr) {
      cudaEventSynchronize(slot.read);  // l'ingestion qui lit encore device
      cudaEventDestroy(slot.read);
    }
    if (slot.copied != nullptr) cudaEventDestroy(slot.copied);
    if (slot.device != nullptr) cudaFree(slot.device);
    if (slot.host != nullptr) cudaFreeHost(slot.host);
  }
  if (transfer_ != nullptr) cudaStreamDestroy(transfer_);
}

std::expected<void, Refusal> HostUploads::reserve(Slot& slot, std::size_t bytes) {
  if (slot.copied == nullptr) {
    if (auto ok =
            cuda_check(cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming), "cudaEventCreate");
        !ok)
      return ok;
    if (auto ok = cuda_check(cudaEventCreateWithFlags(&slot.read, cudaEventDisableTiming), "cudaEventCreate");
        !ok)
      return ok;
  }
  if (bytes <= slot.bytes) return {};
  if (slot.used) {
    // Image plus grande : l'ancien tampon d'appareil peut encore être lu.
    if (auto ok = cuda_check(cudaEventSynchronize(slot.read), "cudaEventSynchronize"); !ok) return ok;
  }
  if (slot.device != nullptr) cudaFree(slot.device);
  if (slot.host != nullptr) cudaFreeHost(slot.host);
  slot.device = nullptr;
  slot.host = nullptr;
  slot.bytes = 0;
  void* host = nullptr;
  if (auto ok = cuda_check(cudaHostAlloc(&host, bytes, cudaHostAllocWriteCombined), "cudaHostAlloc"); !ok)
    return ok;
  slot.host = static_cast<std::byte*>(host);
  if (auto ok = cuda_check(cudaMalloc(&slot.device, bytes), "cudaMalloc"); !ok) return ok;
  slot.bytes = bytes;
  return {};
}

std::expected<HostUploads::Planes, Refusal> HostUploads::upload(const FrameData& data, const Layout& layout,
                                                                cudaStream_t consumer, GpuTiming& timing) {
  if (layout.planes < 2 || layout.planes > 3 || data.plane_count() < layout.planes) {
    return std::unexpected(Refusal{
        RefusalCode::ShapeInvalid,
        std::format("ingestion hôte : {} plans pour {} attendus", data.plane_count(), layout.planes)});
  }
  std::array<std::size_t, 3> pitch{};
  std::array<std::size_t, 3> offset{};
  std::size_t bytes = 0;
  for (int p = 0; p < layout.planes; ++p) {
    const auto i = static_cast<std::size_t>(p);
    const std::span<const std::byte> plane = data.host_plane(p);
    const std::size_t stride = data.host_stride(p);
    if (stride < layout.row_bytes[i] || plane.size() < (layout.rows[i] - 1) * stride + layout.row_bytes[i]) {
      return std::unexpected(
          Refusal{RefusalCode::ShapeInvalid, std::format("ingestion hôte : plan {} trop court", p)});
    }
    pitch[i] = pitch_of(layout.row_bytes[i]);
    offset[i] = bytes;
    bytes += pitch[i] * layout.rows[i];
  }
  if (transfer_ == nullptr) {
    if (auto ok =
            cuda_check(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking), "cudaStreamCreate");
        !ok)
      return std::unexpected(ok.error());
  }
  Slot& slot = slots_[next_];
  next_ = (next_ + 1) % kDepth;
  if (slot.used) {
    // L'envoi d'il y a kDepth images lit peut-être encore le tampon épinglé.
    if (auto ok = cuda_check(cudaEventSynchronize(slot.copied), "cudaEventSynchronize"); !ok)
      return std::unexpected(ok.error());
  }
  if (auto ok = reserve(slot, bytes); !ok) return std::unexpected(ok.error());
  for (int p = 0; p < layout.planes; ++p) {
    const auto i = static_cast<std::size_t>(p);
    const std::byte* source = data.host_plane(p).data();
    const std::size_t stride = data.host_stride(p);
    std::byte* target = slot.host + offset[i];
    if (stride == pitch[i]) {
      std::memcpy(target, source, (layout.rows[i] - 1) * stride + layout.row_bytes[i]);
      continue;
    }
    for (std::size_t y = 0; y < layout.rows[i]; ++y)
      std::memcpy(target + y * pitch[i], source + y * stride, layout.row_bytes[i]);
  }
  // Le tampon d'appareil se réécrit après la lecture de l'ingestion précédente qui l'utilisait.
  if (slot.used) {
    if (auto ok = cuda_check(cudaStreamWaitEvent(transfer_, slot.read, 0), "cudaStreamWaitEvent"); !ok)
      return std::unexpected(ok.error());
  }
  timing.begin(transfer_);
  if (auto ok = cuda_check(cudaMemcpyAsync(slot.device, slot.host, bytes, cudaMemcpyHostToDevice, transfer_),
                           "cudaMemcpyAsync");
      !ok)
    return std::unexpected(ok.error());
  timing.end(transfer_);
  if (auto ok = cuda_check(cudaEventRecord(slot.copied, transfer_), "cudaEventRecord"); !ok)
    return std::unexpected(ok.error());
  slot.used = true;
  if (auto ok = cuda_check(cudaStreamWaitEvent(consumer, slot.copied, 0), "cudaStreamWaitEvent"); !ok)
    return std::unexpected(ok.error());
  pending_ = &slot;
  const auto* device = static_cast<const std::byte*>(slot.device);
  return Planes{.luma = device + offset[0],
                .chroma = device + offset[1],
                .chroma_v = layout.planes == 3 ? device + offset[2] : nullptr,
                .luma_pitch = pitch[0],
                .chroma_pitch = pitch[1]};
}

std::expected<void, Refusal> HostUploads::consumed(cudaStream_t consumer) {
  if (pending_ == nullptr) return {};
  const Slot& slot = *pending_;
  pending_ = nullptr;
  return cuda_check(cudaEventRecord(slot.read, consumer), "cudaEventRecord");
}

}  // namespace ajp::gpu::cuda
