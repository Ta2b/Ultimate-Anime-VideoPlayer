// Ingestion des images décodées en mémoire hôte sous TensorRT (M1) : le décodage logiciel prend le
// relais de NVDEC pour ce qu'il ne décode pas (H.264 10 bits des fansubs, H.264 4:4:4, VP9 profils
// 1 et 3…). Les plans sont recopiés dans un tampon hôte épinglé, envoyés vers l'appareil sur une
// file de transfert, puis lus par le noyau d'ingestion habituel (slice_kernels.h).
//
// kDepth envois à tour de rôle : la copie de l'image suivante passe pendant que la chaîne traite la
// précédente. L'hôte n'attend un envoi que s'il a kDepth images d'avance sur la copie, et la file
// de transfert n'écrase un tampon d'appareil qu'après la lecture de l'ingestion qui l'utilisait.
#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <optional>

#include "core/frame.h"
#include "core/types.h"

namespace ajp::gpu::cuda {

class GpuTiming;

class HostUploads {
 public:
  static constexpr std::size_t kDepth = 3;
  // Pas de ligne des plans envoyés : un multiple de kPitchAlign octets.
  static constexpr std::size_t kPitchAlign = 256;

  // Les plans d'une image d'un format lisible par l'ingestion : NV12, P010, P016 (chroma
  // entrelacée, deux plans) ou YUV 4:2:0 planaire 8 et 10 bits (trois plans).
  struct Layout {
    int planes = 0;
    std::array<std::size_t, 3> row_bytes{};
    std::array<std::size_t, 3> rows{};
  };
  static std::optional<Layout> layout(PixelFormat format, int width, int height);

  // Les plans sur l'appareil : luma, chroma (CbCr entrelacé, ou Cb seul), Cr seul ou nul.
  struct Planes {
    const void* luma = nullptr;
    const void* chroma = nullptr;
    const void* chroma_v = nullptr;
    std::size_t luma_pitch = 0;
    std::size_t chroma_pitch = 0;
  };

  HostUploads() = default;
  ~HostUploads();
  HostUploads(const HostUploads&) = delete;
  HostUploads& operator=(const HostUploads&) = delete;

  // Envoie les plans de data (mémoire hôte, au format de layout) ; consumer attend la fin de
  // l'envoi avant ce qui y est enfilé ensuite. data peut être rendue au décodeur dès le retour.
  // Refus : ShapeInvalid (plan trop court ou pas de ligne trop petit), VendorCallFailed.
  std::expected<Planes, Refusal> upload(const FrameData& data, const Layout& layout, cudaStream_t consumer,
                                        GpuTiming& timing);
  // Après la lecture des plans enfilée sur consumer : leur tampon d'appareil se réutilise quand elle
  // est finie.
  std::expected<void, Refusal> consumed(cudaStream_t consumer);

 private:
  struct Slot {
    std::byte* host = nullptr;  // épinglé, en écriture combinée : l'hôte n'y relit rien
    void* device = nullptr;
    std::size_t bytes = 0;
    cudaEvent_t copied = nullptr;  // l'envoi fini : host se réécrit
    cudaEvent_t read = nullptr;    // l'ingestion finie : device se réécrit
    bool used = false;
  };
  std::expected<void, Refusal> reserve(Slot& slot, std::size_t bytes);

  cudaStream_t transfer_ = nullptr;
  std::array<Slot, kDepth> slots_{};
  std::size_t next_ = 0;
  Slot* pending_ = nullptr;  // l'envoi dont la lecture reste à marquer
};

}  // namespace ajp::gpu::cuda
