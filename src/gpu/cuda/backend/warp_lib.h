// Bibliothèque de warp CUDA, côté hôte (lot E, docs/warp-lib.md) : le warp générique d'une image par un flot
// de n'importe quelle source et à n'importe quelle échelle, avec masque de confiance facultatif. Les noyaux
// spécialisés (la colle de RIFE fusionné) prennent directement les briques de warp_lib.cuh ; ce point
// d'entrée sert les clients qui n'ont pas besoin d'un noyau à eux : TemporalFix (lot B, étape 2), la
// composition par calques, le décalage des tuiles, le banc du flot NVOF.
#pragma once

#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

// Un flot. Soit un tenseur FP16 (dx, dy aux canaux channel et channel + 1) de height × width points, rangé en
// plans (pitch 0) ou par point (pitch canaux), agrandi d'un facteur scale jusqu'à l'image en bilinéaire
// demi-pixel (le Resize de RIFE) et multiplié par gain : la sortie d'un bloc de RIFE (gain = scale), un flot
// NVOF en grille 4 (scale 4, gain 1 : déjà en pixels de l'image), un flot vers un autre instant (gain t).
// Soit, planes nul, un mouvement affine évalué au pixel (caméra) :
//   dx = affine[0] + affine[1] x + affine[2] y,  dy = affine[3] + affine[4] x + affine[5] y.
struct WarpFlow {
  const void* planes = nullptr;
  int height = 0;
  int width = 0;
  int pitch = 0;
  int channel = 0;
  int scale = 1;
  float gain = 1.0F;
  float affine[6] = {};  // NOLINT(modernize-avoid-c-arrays) : passé tel quel au noyau
};

struct WarpPlanesParams {
  const void* image = nullptr;  // channels plans FP16 height × width
  int channels = 0;             // 1 à 16
  int height = 0;
  int width = 0;
  // La fenêtre lisible de l'image (sans la bordure d'un tenseur rempli) ; vide : l'image entière.
  int window_left = 0;
  int window_top = 0;
  int window_width = 0;
  int window_height = 0;
  WarpFlow flow;
  // Confiance (facultative) : un plan FP16 sur la grille du flot tensoriel (même taille, même échelle), lu
  // comme lui et borné à [0, 1]. Avec fallback (channels plans de la taille de l'image, par exemple l'image
  // non déplacée), la sortie est c · warp + (1 − c) · fallback au point lu ; confidence_out, s'il est donné,
  // reçoit c à la taille de la sortie.
  const void* confidence = nullptr;
  const void* fallback = nullptr;
  void* confidence_out = nullptr;
  int cubic = 0;   // 1 : Catmull-Rom borné (kernels::RifeWarp), sinon bilinéaire
  int reduce = 1;  // d : sortie (height / d) × (width / d), moyenne des 2 × 2 points d x + d/2 − 1 + {0, 1}
  void* out = nullptr;
  int out_pitch = 0;  // 0 : plans ; sinon canaux par pixel (au moins channels)
};

// cudaErrorInvalidValue, sans rien lancer : forme vide ou hors bornes (1 à 16 plans, d = 1 ou pair, sortie
// non vide, out_pitch ≥ channels), fenêtre hors de l'image, flot tensoriel sans grille ou dont dx, dy sortent
// du pas, confiance sur un flot affine.
cudaError_t launch_warp_planes(const WarpPlanesParams& params, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
