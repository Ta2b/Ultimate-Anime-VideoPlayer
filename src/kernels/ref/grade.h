// Référence CPU du grade exact dans l'anneau (kernels/spec/grade.h, kernels/spec/grade_pixel.h) : l'entrée
// SDR → PQ en table, les cartes (locale, zone lissée, max local) et le grade d'une image, comme la chaîne
// fusionnée de libplacebo les fait sur OUTPUT. Tests : tests/unit/kernels/grade_test.cpp et, contre le
// shader, tests/unit/render/grade_ring_ref_test.cpp.
#pragma once

#include <array>
#include <span>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/grade.h"

namespace ajp::kernels::ref {

// L'entrée SDR → PQ BT.2020 de libplacebo, relevée par un rendu de rampe : par canal, kInputTablePoints
// luminances (nits) aux codes i / (kInputTablePoints − 1), lues par interpolation linéaire, puis la matrice
// BT.709 → BT.2020 (ligne par ligne) et l'encodage PQ.
struct GradeInput {
  std::array<std::vector<float>, 3> nits;
  std::array<float, 9> matrix{};
};
// La table analytique : BT.1886 γ `gamma`, noir 0, blanc à `white_nits`, matrice BT.709 → BT.2020.
GradeInput sdr_input(float white_nits = 203.0F, float gamma = 2.4F);
// Le code PQ d'une image SDR encodée (R'G'B' dans [0, 1]) par la table.
Result<RgbImage> apply_input(const RgbImage& sdr, const GradeInput& input);

// La pile de LUT composée (render/cube_lut.h, compose_luts), size³ couleurs RVB dans [0, 1], R le plus
// rapide ; vide : aucun look. Son intensité est GradeUniforms::look_mix.
struct GradeLook {
  int size = 0;
  std::span<const float> rgb;
};

// Les cartes d'une image (kernels/spec/grade.h).
struct GradeMaps {
  int local_w = 0;
  int local_h = 0;
  std::vector<float> local;  // 3 composantes par texel
  int area_w = 0;
  int area_h = 0;
  std::vector<float> area;      // la zone lissée, celle que le grade lit
  std::vector<float> history;   // ce que la chaîne garde pour l'image suivante (valeur + 1, 0 : jamais écrit)
  std::vector<float> localmax;  // à la taille de la carte locale, 0 sans vrai noir
};

// Les cartes de `pq` (codes de la cible, au moins 2 × 2). `previous` : l'historique de l'image précédente de
// la même chaîne (GradeMaps::history), vide à une nouvelle époque ; `dt_s` l'écart de pts.
Result<GradeMaps> grade_maps(const RgbImage& pq, const GradeUniforms& uniforms, const GradeLook& look,
                             std::span<const float> previous, float dt_s);

// Un pixel : grade, look, vrai noir, KWin (grade_pixel.h, grade_pixel).
Rgb grade_pixel(const Rgb& code, float skin, float local, float area, float black_local,
                const GradeUniforms& uniforms, const GradeLook& look);

// L'image gradée : chaque pixel lit ses cartes par lecture bilinéaire à sa position.
Result<RgbImage> grade_image(const RgbImage& pq, const GradeMaps& maps, const GradeUniforms& uniforms,
                             const GradeLook& look);

}  // namespace ajp::kernels::ref
