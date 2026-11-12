// Ingestion des images décodées en mémoire hôte sous TensorRT (M1) : le décodage logiciel prend le
// relais de NVDEC pour ce qu'il ne décode pas (H.264 10 bits des fansubs, H.264 4:4:4, VP9 profils
// 1 et 3, et toute piste 4:2:2 ou 4:4:4, que la chaîne garde entière). Les plans sont recopiés dans un
// tampon hôte épinglé, envoyés vers l'appareil sur une file de transfert, puis lus par le noyau
// d'ingestion habituel (slice_kernels.h).
//
// kDepth envois à tour de rôle : la copie de l'image suivante passe pendant que la chaîne traite la
// précédente. L'hôte n'attend un envoi que s'il a kDepth images d'avance sur la copie, et la file
// de transfert n'écrase un tampon d'appareil qu'après la lecture de l'ingestion qui l'utilisait.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <optional>
#include <vector>

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
  // entrelacée, deux plans), YUV 4:2:0 planaire 8 et 10 bits, ou 4:2:2 et 4:4:4 16 bits (trois plans).
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

  // Envoie les plans de frame (mémoire hôte, au format de layout) ; consumer attend la fin de
  // l'envoi avant ce qui y est enfilé ensuite. Les pixels peuvent être rendus au décodeur dès le retour.
  // La même source que l'envoi précédent (rgb_ puis small_ qui l'ingèrent l'une après l'autre, C32-03)
  // reprend ses plans sur l'appareil : ni copie hôte ni envoi de plus.
  // Refus : ShapeInvalid (plan trop court ou pas de ligne trop petit), VendorCallFailed.
  std::expected<Planes, Refusal> upload(const DecodedFrame& frame, const Layout& layout,
                                        cudaStream_t consumer, GpuTiming& timing);
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
  std::expected<Planes, Refusal> reuse(cudaStream_t consumer);

  cudaStream_t transfer_ = nullptr;
  std::array<Slot, kDepth> slots_{};
  std::size_t next_ = 0;
  Slot* pending_ = nullptr;  // l'envoi dont la lecture reste à marquer
  // L'image envoyée : son identité (le pool recycle les adresses, les SourceId repartent après un seek).
  struct Source {
    SourceId id;
    std::uint32_t epoch = 0;
    const FrameData* data = nullptr;
    friend bool operator==(const Source&, const Source&) = default;
  };
  // Le dernier envoi : sa source, ses plans, la file qui l'a lu en dernier (reuse).
  struct Last {
    Source source;
    Layout layout;
    Slot* slot = nullptr;
    Planes planes;
    cudaStream_t reader = nullptr;
  };
  std::optional<Last> last_;
  bool reusable(const Source& source, const Layout& layout) const;
  // Les tampons épinglés remplacés par un plus grand, rendus à la destruction (cudaFreeHost synchronise
  // l'appareil, C01-05).
  std::vector<std::byte*> outgrown_;
};

}  // namespace ajp::gpu::cuda
