// Décisions temporelles d'une paire de sources (kernels/spec/detect.h), en une lecture des deux
// lumas : score de coupe de scène et pire bloc 32 × 32 des dessins tenus. Le résultat se relit
// sans attente, comme k_flow_max dans le moteur de référence : measure() lance la paire,
// previous() rend la dernière mesure finie, servie à la paire suivante.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <optional>

#include "gpu/cuda/compute/device.h"
#include "gpu/cuda/compute/image.h"
#include "gpu/cuda/compute/memory.h"

namespace ajp::gpu::cuda {

struct PairScores {
  double scene_score = 0.0;       // kernels::is_scene_cut(scene_score)
  std::uint32_t worst_block = 0;  // en 1/256 de niveau 8 bits ; kernels::is_held(worst_block)
  bool identical = false;         // luma égale au bit : somme des écarts nulle
};

class PairMeter {
 public:
  // Lumas de format et de taille donnés (la source, non remplie). Le score se divise par l'aire
  // remplie de kernels::rife_geometry(width, height), que la mesure à 720p a fixée.
  static Result<PairMeter> create(DevicePool& pool, kernels::Format format, int width, int height);

  // Lance la mesure de a contre b (seule leur luma est lue). Un measure() avant que previous()
  // ait rendu la mesure précédente la remplace.
  Result<> measure(const YuvPlanes& a, const YuvPlanes& b, cudaStream_t stream);

  // La dernière mesure finie, une seule fois ; rien si aucune n'a fini depuis l'appel précédent.
  // N'attend jamais côté hôte.
  Result<std::optional<PairScores>> previous();

 private:
  kernels::Format format_ = kernels::Format::Nv12;
  int width_ = 0;
  int height_ = 0;
  int tiles_ = 0;
  DeviceBuffer partials_;
  DeviceBuffer result_;
  PinnedBuffer host_;
  Event ready_;
  bool pending_ = false;
};

}  // namespace ajp::gpu::cuda
