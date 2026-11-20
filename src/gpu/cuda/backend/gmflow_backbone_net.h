// Le squelette CNN de GMFlow natif (docs/gmflow-natif.md) : le plan de prepare/gmflow_backbone.h exécuté à la
// lettre par les noyaux de gmflow_backbone_kernels.cuh, sur les deux images de GMFlow en lot. Le trident
// réutilise la convolution au quart lorsque les deux branches partagent exactement les mêmes poids.
// Sans std::expected, car nvcc lit cet en-tête en C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmflow_backbone.h"

namespace kirari::gpu::cuda {

// Les deux sorties, 128 caractéristiques NHWC FP16 par pixel : le trident de pas 1 au quart
// (2 × height/4 × width/4 × 128) et de pas 2 au huitième (2 × height/8 × width/8 × 128), sans la position.
struct GmflowFeatures {
  void* quarter = nullptr;
  void* eighth = nullptr;
  bool quarter_swapped = false;  // au quart seulement, ordre physique B, A si vrai
  bool eighth_from_quarter = false;  // le consommateur lira directement les centres pairs du quart
};

class GmflowBackboneNet {
 public:
  virtual ~GmflowBackboneNet() = default;
  // height × width : la taille des images de GMFlow (la moitié de la paire au flot entier, son quart au
  // demi-flot), multiples de 32. Sommes des convolutions et des statistiques en f32. Les tenseurs sont placés
  // une fois dans une zone allouée ici (deux qui vivent ensemble ne se recouvrent pas). Rien et error rempli
  // si le plan sort de ce que les noyaux font, ou si la mémoire manque.
  static std::unique_ptr<GmflowBackboneNet> create(const prepare::GmflowBackbone& backbone, int height,
                                                   int width, std::string& error);
  virtual std::int64_t workspace_bytes() const = 0;
  // images : les deux images, trois plans FP16 de height × width chacune dans [0, 1], l'image 0 puis l'image
  // 1 (la normalisation se fait à la lecture). Les statistiques sont reconstruites dans un ordre fixe à
  // chaque convolution normalisée. Les opérations sont capturables dans un graphe CUDA.
  // Avec reuse_left, l'ancien côté droit doit être intact dans out.quarter, à l'adresse indiquée
  // par quarter_swapped. Le noyau échange alors les rôles des deux emplacements sans copier.
  // La sortie indique dans eighth_from_quarter si le trident au huitième se lit depuis quarter.
  // Sinon le huitième est calculé en ordre logique A, B dans sa propre zone.
  virtual cudaError_t run(cudaStream_t stream, const void* images, GmflowFeatures& out,
                          bool reuse_left = false) const = 0;
};

}  // namespace kirari::gpu::cuda
