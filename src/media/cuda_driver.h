// Pilote CUDA chargé à l'exécution (libcuda.so.1, nvcuda.dll) : le lecteur
// démarre sans CUDA. Seulement ce que media en attend : trouver l'appareil du
// rendu par son UUID, créer le flux de décodage non bloquant, enregistrer un
// événement par image. Interne au module ; ne tourne pas sans GPU NVIDIA.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "core/types.h"
#include "media/frame_data.h"

namespace ajp::media {

using CudaUuid = std::array<std::uint8_t, 16>;

// La bibliothèque n'est déchargée que si un symbole manque, avant cuInit. Dès
// cuInit appelé, elle reste chargée jusqu'à la fin du processus, même après
// le dernier CudaDriver :
// - cuInit lance les threads du pilote (cuda-EvtHandlr…) et pose ses
//   gestionnaires de fin de processus ; NVIDIA ne prend pas en charge le
//   déchargement qui suit, et ces threads tourneraient sur du code démappé ;
// - FFmpeg (hwcontext_cuda, nvdec), nos noyaux et TensorRT partagent le même
//   pilote et le même contexte primaire, chacun par son dlopen : notre
//   référence évite un cuInit complet à chaque réouverture du décodeur ;
// - LeakSanitizer ne nomme une fuite interne du pilote que si sa bibliothèque
//   est encore chargée à la sortie ; déchargée, la fuite devient
//   « <unknown module> » et tools/ci/lsan.supp ne la supprime plus.
// Chaque load() ajoute une référence au compte de dlopen, sans autre coût.
class CudaDriver : public std::enable_shared_from_this<CudaDriver> {
 public:
  // Refus nommé si la bibliothèque ou un symbole manque, ou si cuInit échoue.
  static std::expected<std::shared_ptr<CudaDriver>, Refusal> load();
  ~CudaDriver();
  CudaDriver(const CudaDriver&) = delete;
  CudaDriver& operator=(const CudaDriver&) = delete;

  // Ordinal de l'appareil dont l'UUID égale celui du rendu.
  std::expected<int, Refusal> ordinal_of(const CudaUuid& uuid) const;
  // UUID de chaque appareil, dans l'ordre des ordinaux.
  std::expected<std::vector<CudaUuid>, Refusal> uuids() const;
  // Contexte primaire de l'appareil (CUcontext), celui que FFmpeg reprend avec
  // AV_CUDA_USE_PRIMARY_CONTEXT. Lu sans le garder : il ne vit que tant qu'un
  // autre utilisateur le retient.
  std::expected<std::uintptr_t, Refusal> primary_context(int ordinal) const;

  // Flux CU_STREAM_NON_BLOCKING dans le contexte donné (CUcontext).
  std::expected<std::uintptr_t, Refusal> create_stream(std::uintptr_t context) const;
  void destroy_stream(std::uintptr_t context, std::uintptr_t stream) const;

  // Événement enregistré sur le flux, détruit avec le dernier propriétaire.
  std::expected<ReadyEvent, Refusal> record(std::uintptr_t context, std::uintptr_t stream) const;

  struct Api;  // table des fonctions du pilote, définie dans le .cpp

 private:
  explicit CudaDriver(std::unique_ptr<Api> api);

  std::unique_ptr<Api> api_;
};

}  // namespace ajp::media
