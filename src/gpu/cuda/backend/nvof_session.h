// Le flux optique matériel de NVIDIA (NVIDIA Optical Flow, le bloc NVOFA des RTX ; API 2.0, en-têtes BSD dans
// third_party/nvidia-optical-flow) pour la carte de la scène (kernels/spec/scene_flow.h). Repris du commit
// 5dff1467, avec ce qui y manquait :
//   - sa propre file CUDA (nvOFSetIOCudaStreams), jamais celle d'une voie : sur la file du backend, NVOF
//     passait en série avec TensorRT et le bloc à part ne servait à rien. Priorité la plus haute : la
//     réduction et la copie qui suivent NVOF ne font pas la queue derrière les blocs de SPAN et de RIFE ;
//   - un seul sens (a vers b) et la carte de coût 8 bits, la confiance de chaque vecteur ;
//   - les indices temporels coupés à la demande (execute(restart)) : coupe, saut, nouvelle époque, export ;
//   - les capacités lues (nvOFGetCaps : grilles, tailles), rien de supposé. La carte de coût n'existe
//     qu'à partir d'Ampere : open refuse un GPU plus ancien (Turing) ;
//   - une session par taille : une autre taille demande une autre session.
// La bibliothèque vient du pilote, chargée ici et jamais déchargée ; aucun fichier de NVIDIA n'est livré.
// L'API n'est pas sûre entre threads : un seul thread se sert d'une session.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>

#include "core/types.h"

namespace kirari::gpu::cuda {

class NvofSession {
 public:
  // Une luma 8 bits de width × height, grille kernels::kSceneFlowGrid, sur une file créée pour elle.
  // Refus : bibliothèque absente (AMD, Intel, pilote sans flux optique : LibraryUnavailable), GPU sans carte
  // de coût, grille ou taille refusées (FormatUnsupported, ShapeInvalid), appel refusé par le pilote.
  static std::expected<std::unique_ptr<NvofSession>, Refusal> open(int width, int height);
  ~NvofSession();  // attend la file, puis rend tout au pilote
  NvofSession(const NvofSession&) = delete;
  NvofSession& operator=(const NvofSession&) = delete;

  int width() const { return width_; }
  int height() const { return height_; }
  int cells_x() const;
  int cells_y() const;
  cudaStream_t stream() const { return stream_; }

  // Mémoire de l'appareil et pas de ligne en octets.
  struct Pitched {
    void* data = nullptr;
    std::size_t pitch = 0;
  };
  // kLumas tampons de luma, chacun lu tour à tour comme a ou comme b : la luma de b d'une paire sert de a à
  // la suivante, et la voie écrit la nouvelle dans un tampon que NVOF a fini de lire (scene_flow.h).
  static constexpr int kLumas = 3;
  Pitched luma(int slot) const;  // slot dans [0, kLumas)
  Pitched flow() const;          // S10.5, cells_x × cells_y vecteurs (int16 x, int16 y)
  Pitched cost() const;          // un octet par vecteur

  // Le flux de luma(a) vers luma(b), enfilé sur stream() ; aucune attente hôte. restart : sans les vecteurs
  // de l'appel précédent comme indices.
  std::expected<void, Refusal> execute(int a, int b, bool restart);

  struct Api;
  struct Buffers;

 private:
  NvofSession() = default;
  std::unique_ptr<Api> api_;
  std::unique_ptr<Buffers> buffers_;
  cudaStream_t stream_ = nullptr;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace kirari::gpu::cuda
