#include "gpu/cuda/trt/cuda_objects.h"

#include <algorithm>
#include <format>
#include <utility>

namespace ajp::gpu::trt {

Refusal cuda_refusal(cudaError_t error, std::string_view call) {
  return Refusal{RefusalCode::CudaFailed,
                 std::format("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error))};
}

std::expected<void, Refusal> cuda_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(cuda_refusal(error, call));
}

// Les destructeurs ne peuvent rien refuser : une erreur de libération (pilote
// perdu) est ignorée, l'objet n'existant plus de toute façon.

std::expected<CudaStream, Refusal> CudaStream::create() {
  cudaStream_t stream = nullptr;
  if (auto ok = cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                           "cudaStreamCreateWithFlags(cudaStreamNonBlocking)");
      !ok) {
    return std::unexpected(ok.error());
  }
  return CudaStream(stream);
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  std::swap(stream_, other.stream_);
  return *this;
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
}

std::expected<void, Refusal> CudaStream::synchronize() const {
  return cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

std::expected<CudaEvent, Refusal> CudaEvent::create(EventTiming timing) {
  const unsigned flags = timing == EventTiming::Enabled ? cudaEventDefault : cudaEventDisableTiming;
  cudaEvent_t event = nullptr;
  if (auto ok = cuda_check(cudaEventCreateWithFlags(&event, flags), "cudaEventCreateWithFlags"); !ok) {
    return std::unexpected(ok.error());
  }
  return CudaEvent(event);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  std::swap(event_, other.event_);
  return *this;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) static_cast<void>(cudaEventDestroy(event_));
}

std::expected<void, Refusal> CudaEvent::record(cudaStream_t stream) const {
  return cuda_check(cudaEventRecord(event_, stream), "cudaEventRecord");
}

std::expected<void, Refusal> CudaEvent::synchronize() const {
  return cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

std::expected<bool, Refusal> CudaEvent::done() const {
  const cudaError_t state = cudaEventQuery(event_);
  if (state == cudaSuccess) return true;
  if (state == cudaErrorNotReady) return false;
  return std::unexpected(cuda_refusal(state, "cudaEventQuery"));
}

std::expected<float, Refusal> elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
  float ms = 0.0F;
  if (auto ok = cuda_check(cudaEventElapsedTime(&ms, start.get(), end.get()), "cudaEventElapsedTime"); !ok) {
    return std::unexpected(ok.error());
  }
  return ms;
}

std::expected<DeviceBuffer, Refusal> DeviceBuffer::allocate(std::size_t bytes) {
  void* data = nullptr;
  if (auto ok = cuda_check(cudaMalloc(&data, bytes), std::format("cudaMalloc({})", bytes)); !ok) {
    return std::unexpected(ok.error());
  }
  return DeviceBuffer(data, bytes);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
  return *this;
}

DeviceBuffer::~DeviceBuffer() {
  if (data_ != nullptr) static_cast<void>(cudaFree(data_));
}

std::expected<CudaDeviceInfo, Refusal> current_device() {
  CudaDeviceInfo info;
  if (auto ok = cuda_check(cudaGetDevice(&info.index), "cudaGetDevice"); !ok)
    return std::unexpected(ok.error());
  cudaDeviceProp props{};
  if (auto ok = cuda_check(cudaGetDeviceProperties(&props, info.index), "cudaGetDeviceProperties"); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = cuda_check(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion"); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = cuda_check(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion"); !ok) {
    return std::unexpected(ok.error());
  }
  info.name = props.name;
  info.sm_major = props.major;
  info.sm_minor = props.minor;
  std::ranges::transform(props.uuid.bytes, info.uuid.begin(),
                         [](char byte) { return static_cast<std::uint8_t>(byte); });
  return info;
}

}  // namespace ajp::gpu::trt
