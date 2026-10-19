// Appareil CUDA choisi par UUID, flux non bloquants et événements.
//
// Le runtime CUDA travaille dans le contexte primaire de l'appareil : celui que FFmpeg partage
// avec AV_CUDA_USE_PRIMARY_CONTEXT, où les pointeurs de NVDEC sont valides, et celui où
// cudaImportExternalMemory importe les images Vulkan (l'équivalent de cuImportExternalMemory
// sur ce contexte). Le runtime est lié en statique et charge libcuda à l'exécution : sans pilote,
// open() refuse proprement au lieu d'empêcher le démarrage.
//
// Thread : un Device, ses flux et ses événements appartiennent au thread gpu. Le contexte courant
// est un état de thread : open() le pose sur le thread appelant, un autre thread qui lancerait sur
// ces flux appelle make_current() d'abord.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/compute/refusal.h"

namespace ajp::gpu::cuda {

// Identifiant d'un GPU, le même octet à octet que VkPhysicalDeviceIDProperties::deviceUUID,
// cuDeviceGetUuid et DeviceUuid (core/gpu_device.h) : il relie l'appareil CUDA à celui de rendu.
using Uuid = std::array<std::uint8_t, 16>;

struct DeviceInfo {
  int ordinal = 0;
  Uuid uuid{};
  std::string name;
  int major = 0;  // capacité de calcul : 8.9 pour la 4070 Ti
  int minor = 0;
  std::size_t memory = 0;  // octets
};

// Les appareils CUDA visibles ; refus si le pilote manque (cudaGetDeviceCount).
Result<std::vector<DeviceInfo>> list_devices();

class Device {
 public:
  // Le contexte primaire de l'appareil d'UUID donné, rendu courant sur le thread appelant.
  // Refus : pilote absent, aucun appareil de cet UUID, ou binaire sans code pour l'appareil
  // (architecture absente de AJP_CUDA_ARCHITECTURES).
  static Result<Device> open(const Uuid& uuid);

  const DeviceInfo& info() const { return info_; }
  Result<> make_current() const;

 private:
  explicit Device(DeviceInfo info) : info_(std::move(info)) {}
  DeviceInfo info_;
};

// Flux non bloquant (cudaStreamNonBlocking) : il ne se synchronise jamais avec le flux par défaut
// historique, que d'autres bibliothèques du processus peuvent utiliser.
class Stream {
 public:
  static Result<Stream> create();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;
  ~Stream();

  cudaStream_t get() const { return stream_; }
  // Attente côté GPU d'un événement, par exemple le CUevent de fin de décodage de NVDEC.
  Result<> wait(cudaEvent_t event) const;
  // Attente côté hôte : tests, mesures et arrêt seulement, jamais sur le chemin de lecture.
  Result<> synchronize() const;

 private:
  explicit Stream(cudaStream_t stream) : stream_(stream) {}
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  // timing : l'événement mesure le temps écoulé (elapsed_ms) ; sinon il ne sert qu'à l'ordre,
  // plus léger (cudaEventDisableTiming).
  static Result<Event> create(bool timing = false);

  Event() = default;  // sans événement : à remplacer par create()

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;
  ~Event();

  cudaEvent_t get() const { return event_; }
  Result<> record(cudaStream_t stream) const;
  // Le travail enregistré est-il fini ? N'attend jamais.
  Result<bool> done() const;
  Result<> synchronize() const;
  // Millisecondes entre deux événements de mesure, tous deux finis.
  static Result<float> elapsed_ms(const Event& start, const Event& end);

 private:
  explicit Event(cudaEvent_t event) : event_(event) {}
  cudaEvent_t event_ = nullptr;
};

}  // namespace ajp::gpu::cuda
