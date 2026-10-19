// Objets du runtime CUDA possédés par le runtime TensorRT : flux non bloquant,
// événement, mémoire de l'appareil. Chaque erreur CUDA devient un refus
// CudaFailed qui cite l'appel ; aucun objet n'est rendu à moitié créé.
//
// Tous s'utilisent sur l'appareil courant du thread qui les crée (cudaSetDevice).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::gpu::trt {

// « cudaMalloc(1048576) : cudaErrorMemoryAllocation (out of memory) ».
Refusal cuda_refusal(cudaError_t error, std::string_view call);
std::expected<void, Refusal> cuda_check(cudaError_t error, std::string_view call);

// Flux créé avec cudaStreamNonBlocking : il ne se synchronise pas avec le flux
// par défaut, qu'un autre module pourrait utiliser.
class CudaStream {
 public:
  static std::expected<CudaStream, Refusal> create();

  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  ~CudaStream();

  cudaStream_t get() const { return stream_; }
  std::expected<void, Refusal> synchronize() const;

 private:
  explicit CudaStream(cudaStream_t stream) : stream_(stream) {}
  cudaStream_t stream_ = nullptr;
};

enum class EventTiming : std::uint8_t { Disabled, Enabled };

// Événement CUDA. Sans chronométrage (le défaut), il coûte moins à enregistrer
// et à attendre ; Enabled sert aux mesures.
class CudaEvent {
 public:
  static std::expected<CudaEvent, Refusal> create(EventTiming timing = EventTiming::Disabled);

  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  ~CudaEvent();

  cudaEvent_t get() const { return event_; }
  std::expected<void, Refusal> record(cudaStream_t stream) const;
  std::expected<void, Refusal> synchronize() const;
  // Vrai quand tout le travail enregistré avant l'événement est terminé.
  std::expected<bool, Refusal> done() const;

 private:
  explicit CudaEvent(cudaEvent_t event) : event_(event) {}
  cudaEvent_t event_ = nullptr;
};

// Millisecondes entre deux événements chronométrés et terminés.
std::expected<float, Refusal> elapsed_ms(const CudaEvent& start, const CudaEvent& end);

// Mémoire de l'appareil, libérée à la destruction.
class DeviceBuffer {
 public:
  static std::expected<DeviceBuffer, Refusal> allocate(std::size_t bytes);

  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer();

  void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  DeviceBuffer(void* data, std::size_t size) : data_(data), size_(size) {}
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

// L'appareil courant tel que les jetons de cache du constructeur le décrivent.
struct CudaDeviceInfo {
  int index = 0;
  std::string name;  // « NVIDIA GeForce RTX 4070 Ti »
  int sm_major = 0;
  int sm_minor = 0;
  std::array<std::uint8_t, 16> uuid{};
  int runtime_version = 0;  // cudaRuntimeGetVersion : 13040 pour CUDA 13.4
  int driver_version = 0;   // cudaDriverGetVersion
};

// Refuse sans appareil CUDA utilisable (pas de pilote, pas de GPU NVIDIA).
std::expected<CudaDeviceInfo, Refusal> current_device();

}  // namespace ajp::gpu::trt
