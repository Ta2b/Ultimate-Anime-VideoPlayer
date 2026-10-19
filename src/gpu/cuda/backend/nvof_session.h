// Le flux optique matériel de NVIDIA (NVIDIA Optical Flow, l'OFA des RTX ; API 2.0, en-têtes dans
// third_party/nvidia-optical-flow) sur le contexte et le flux CUDA du backend. Deux sessions :
// avant (a → b) et arrière (b → a), chacune avec ses indices temporels, les vecteurs de la paire
// précédente guidant ceux de la suivante. La bibliothèque vient avec le pilote, chargée ici.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>

#include "core/types.h"

namespace ajp::gpu::cuda {

class NvofSession {
 public:
  // Une luma 8 bits de width × height ; la grille de sortie est la plus fine que l'appareil propose
  // parmi kernels::kPreferredFlowGrid et 4. Refus : bibliothèque absente (pilote sans flux
  // optique, GPU sans OFA), taille ou appel refusés par le pilote.
  static std::expected<std::unique_ptr<NvofSession>, Refusal> open(cudaStream_t stream, int width,
                                                                   int height);
  ~NvofSession();
  NvofSession(const NvofSession&) = delete;
  NvofSession& operator=(const NvofSession&) = delete;

  int width() const { return width_; }
  int height() const { return height_; }
  int grid() const { return grid_; }
  int cells_x() const;
  int cells_y() const;

  // Mémoire de l'appareil et pas de ligne en octets.
  struct Pitched {
    void* data = nullptr;
    std::size_t pitch = 0;
  };
  enum class Direction : std::uint8_t { Forward, Backward };
  // Luma lue par la session : l'image de départ (input) et celle d'arrivée (reference). Avant :
  // a puis b ; arrière : b puis a.
  Pitched input(Direction direction) const;
  Pitched reference(Direction direction) const;
  // Flot S10.5 écrit par execute : cells_x × cells_y vecteurs (int16 x, int16 y).
  Pitched flow(Direction direction) const;

  // Les deux flots, enfilés sur le flux du backend ; aucune attente hôte.
  std::expected<void, Refusal> execute();

  struct Api;
  struct Side;

 private:
  NvofSession() = default;
  std::unique_ptr<Api> api_;
  std::array<std::unique_ptr<Side>, 2> sides_;
  int width_ = 0;
  int height_ = 0;
  int grid_ = 0;
};

}  // namespace ajp::gpu::cuda
