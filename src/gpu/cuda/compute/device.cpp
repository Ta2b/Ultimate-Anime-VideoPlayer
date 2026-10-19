#include "gpu/cuda/compute/device.h"

#include <algorithm>
#include <format>
#include <span>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda {
namespace {

std::string hex(const Uuid& uuid) {
  std::string out;
  for (const std::uint8_t byte : uuid) out += std::format("{:02x}", byte);
  return out;
}

}  // namespace

Result<std::vector<DeviceInfo>> list_devices() {
  int count = 0;
  if (auto ok = cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount"); !ok) {
    return std::unexpected(ok.error());
  }
  std::vector<DeviceInfo> devices;
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop{};
    if (auto ok = cuda_check(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties"); !ok) {
      return std::unexpected(ok.error());
    }
    DeviceInfo info{.ordinal = i,
                    .name = prop.name,
                    .major = prop.major,
                    .minor = prop.minor,
                    .memory = prop.totalGlobalMem};
    const auto bytes = std::as_bytes(std::span(prop.uuid.bytes));
    std::ranges::transform(bytes, info.uuid.begin(),
                           [](std::byte b) { return static_cast<std::uint8_t>(b); });
    devices.push_back(std::move(info));
  }
  return devices;
}

Result<Device> Device::open(const Uuid& uuid) {
  auto devices = list_devices();
  if (!devices) return std::unexpected(devices.error());
  const auto found = std::ranges::find(*devices, uuid, &DeviceInfo::uuid);
  if (found == devices->end()) {
    return refuse(RefusalCode::CudaFailed,
                  std::format("cudaGetDeviceProperties : aucun des {} appareils CUDA n'a l'UUID {}",
                              devices->size(), hex(uuid)));
  }
  Device device(std::move(*found));
  if (auto ok = device.make_current(); !ok) return std::unexpected(ok.error());
  // Le premier lancement crée le contexte primaire et vérifie que le binaire a du code pour
  // cette architecture.
  if (auto ok = cuda_check(dev::probe_kernel_image(), "cudaFuncGetAttributes (noyau témoin)"); !ok) {
    return std::unexpected(ok.error());
  }
  return device;
}

Result<> Device::make_current() const {
  return cuda_check(cudaSetDevice(info_.ordinal), "cudaSetDevice");
}

Result<Stream> Stream::create() {
  cudaStream_t stream = nullptr;
  if (auto ok =
          cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
      !ok) {
    return std::unexpected(ok.error());
  }
  return Stream(stream);
}

Stream::Stream(Stream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

Stream::~Stream() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

Result<> Stream::wait(cudaEvent_t event) const {
  return cuda_check(cudaStreamWaitEvent(stream_, event, 0), "cudaStreamWaitEvent");
}

Result<> Stream::synchronize() const {
  return cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

Result<Event> Event::create(bool timing) {
  cudaEvent_t event = nullptr;
  const unsigned flags = timing ? cudaEventDefault : cudaEventDisableTiming;
  if (auto ok = cuda_check(cudaEventCreateWithFlags(&event, flags), "cudaEventCreateWithFlags"); !ok) {
    return std::unexpected(ok.error());
  }
  return Event(event);
}

Event::Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) cudaEventDestroy(event_);
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

Event::~Event() {
  if (event_ != nullptr) cudaEventDestroy(event_);
}

Result<> Event::record(cudaStream_t stream) const {
  return cuda_check(cudaEventRecord(event_, stream), "cudaEventRecord");
}

Result<bool> Event::done() const {
  const cudaError_t status = cudaEventQuery(event_);
  if (status == cudaErrorNotReady) {
    // « Pas encore » n'est pas une erreur : rien ne doit en rester pour le prochain lancement.
    static_cast<void>(cudaGetLastError());
    return false;
  }
  if (auto ok = cuda_check(status, "cudaEventQuery"); !ok) return std::unexpected(ok.error());
  return true;
}

Result<> Event::synchronize() const {
  return cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

Result<float> Event::elapsed_ms(const Event& start, const Event& end) {
  float ms = 0.0F;
  if (auto ok = cuda_check(cudaEventElapsedTime(&ms, start.event_, end.event_), "cudaEventElapsedTime");
      !ok) {
    return std::unexpected(ok.error());
  }
  return ms;
}

}  // namespace ajp::gpu::cuda
