// Pilote CUDA chargé à l'exécution (libcuda.so.1, nvcuda.dll) : le lecteur
// démarre sans CUDA. Seulement ce que media en attend : trouver l'appareil du
// rendu par son UUID, créer le flux de décodage non bloquant, enregistrer un
// événement par image. Interne au module ; ne tourne pas sans GPU NVIDIA.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>

#include "core/types.h"
#include "media/frame_data.h"

namespace ajp::media {

using CudaUuid = std::array<std::uint8_t, 16>;

class CudaDriver : public std::enable_shared_from_this<CudaDriver> {
 public:
  // Refus nommé si la bibliothèque ou un symbole manque, ou si cuInit échoue.
  static std::expected<std::shared_ptr<CudaDriver>, Refusal> load();
  ~CudaDriver();
  CudaDriver(const CudaDriver&) = delete;
  CudaDriver& operator=(const CudaDriver&) = delete;

  // Ordinal de l'appareil dont l'UUID égale celui du rendu.
  std::expected<int, Refusal> ordinal_of(const CudaUuid& uuid) const;

  // Flux CU_STREAM_NON_BLOCKING dans le contexte donné (CUcontext).
  std::expected<std::uintptr_t, Refusal> create_stream(std::uintptr_t context) const;
  void destroy_stream(std::uintptr_t context, std::uintptr_t stream) const;

  // Événement enregistré sur le flux, détruit avec le dernier propriétaire.
  std::expected<ReadyEvent, Refusal> record(std::uintptr_t context, std::uintptr_t stream) const;

  struct Api;  // table des fonctions du pilote, définie dans le .cpp

 private:
  CudaDriver(void* library, std::unique_ptr<Api> api);

  void* library_;
  std::unique_ptr<Api> api_;
};

}  // namespace ajp::media
