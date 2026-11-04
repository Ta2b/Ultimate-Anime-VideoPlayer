// Le grade exact dans l'anneau (chantier 2c, docs/adr/0009-grade-dans-l-anneau.md) : entre l'écriture d'un
// emplacement et son signal, trois lancements refont ce que la chaîne fusionnée de libplacebo fait sur
// OUTPUT, à la taille de l'anneau, et laissent dans l'emplacement les codes PQ gradés. Les formules sont
// celles de kernels/spec/grade_pixel.h, partagées avec la référence kernels/ref/grade.cpp (test « grade de
// l'anneau contre la référence », [backend_tensorrt][cuda]).
#pragma once

#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/ring_surface.h"
#include "kernels/spec/grade.h"

namespace ajp::gpu::cuda {

struct GradeMapSizes {
  int local_w = 0;
  int local_h = 0;
  int area_w = 0;
  int area_h = 0;
};
// Les tailles des cartes d'une image width × height (kernels::grade::map_size).
GradeMapSizes grade_map_sizes(int width, int height);

// Une image de l'anneau à grader, et ses tampons (mémoire de l'appareil, tailles de grade_map_sizes).
struct GradeFrame {
  RingSurface surface;  // lue puis réécrite en place
  int width = 0;
  int height = 0;
  // L'entrée SDR → PQ de libplacebo (render::capture_grade_input) : 3 × kInputTablePoints nits (R, G, B bout
  // à bout), et la matrice des primaires vers BT.2020, ligne par ligne.
  const float* input_nits = nullptr;
  float input_matrix[9] = {};
  // La pile de LUT composée : texture 3D float4, filtre linéaire, coordonnées normalisées, bords tenus ;
  // 0 : aucun look (uniforms.look_mix doit alors valoir 0).
  cudaTextureObject_t look = 0;
  int look_size = 0;
  kernels::GradeUniforms uniforms;
  float* local = nullptr;     // 3 × local_w × local_h
  float* area = nullptr;      // area_w × area_h, la zone lissée
  float* history = nullptr;   // area_w × area_h, gardé par chaîne d'images (valeur + 1, 0 : jamais écrit)
  float* localmax = nullptr;  // local_w × local_h
  float* skin = nullptr;      // width × height, lu seulement si uniforms.skin_on
};

// La carte locale 1/8 (et, masque peau allumé, le masque de chaque pixel, lu avant que l'image ne change).
cudaError_t launch_grade_local(const GradeFrame& frame, cudaStream_t stream);
// La zone : moyenne 4 × 4, gaussienne en mémoire partagée, puis le lissage par l'historique de la chaîne,
// α = 1 − 0,7^(48 dt_s) ; `reset` (nouvelle époque, coupe annoncée) : l'historique est ignoré et réécrit.
cudaError_t launch_grade_area(const GradeFrame& frame, float dt_s, bool reset, cudaStream_t stream);
// Le max local du vrai noir (s'il est allumé), puis chaque pixel : entrée, grade, look, pied, KWin, et le
// code PQ écrit en place.
cudaError_t launch_grade_slot(const GradeFrame& frame, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
