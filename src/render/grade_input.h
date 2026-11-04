// L'entrée du grade de l'anneau (kernels/spec/grade.h, chantier 2.7) : ce que libplacebo fait d'une image du
// ring avant les hooks d'OUTPUT, relevé par un rendu de rampe sans hook, en une table 1D par canal et une
// matrice de primaires. Le grade exact la rejoue sur le GPU du ring (kernels/ref/grade.h, GradeInput).
#pragma once

#include <array>
#include <expected>
#include <vector>

#include <libplacebo/colorspace.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

#include "core/types.h"

namespace ajp::render {

inline constexpr int kGradeInputPoints = 4096;  // kernels::grade::kInputTablePoints

struct GradeInputTable {
  // Par canal, la luminance (nits) rendue pour le code i / (kGradeInputPoints − 1) d'une rampe grise : la
  // matrice garde les gris, chaque canal de sortie vaut donc la courbe de son canal d'entrée.
  std::array<std::vector<float>, 3> nits;
  // Primaires de la source vers BT.2020, ligne par ligne (pl_get_color_mapping_matrix, relatif).
  std::array<float, 9> matrix{};
};

// La rampe de `source` rendue vers `target` (PQ BT.2020), sans hook ni tramage, relue tout de suite.
// HdrUnavailable : la cible n'est pas en PQ ; VendorCallFailed : rendu ou relecture.
std::expected<GradeInputTable, Refusal> capture_grade_input(pl_renderer renderer, pl_gpu gpu,
                                                            const pl_color_space& source,
                                                            const pl_color_space& target);

}  // namespace ajp::render
