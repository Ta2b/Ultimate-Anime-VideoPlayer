// La chaîne complète du grade HDR dans les tests : la passe anime, la pile de LUT livrée (data/luts) et la
// dernière passe, comme le lecteur les pose sur une sortie HDR10.
#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "grade_gpu_support.h"
#include "render/cube_lut.h"

namespace ajp::render::gpu_test {

inline CubeLut shipped_lut(const std::string& name) {
  const std::filesystem::path path = std::filesystem::path(AJP_DATA_DIR) / "luts" / name;
  std::ifstream file(path, std::ios::binary);
  REQUIRE(file);
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto lut = parse_cube(text);
  REQUIRE(lut);
  fill_unreachable(*lut);  // comme load_look pour les LUT livrées
  return *lut;
}

// La pile par défaut du look HDR anime (app/play/look_catalog.cpp, curseurs à 0) : la base, 3 LUT de
// contraste, 1,5 LUT d'éclats, « skin fix », puis 4 LUT de contraste.
inline CubeLut default_stack() {
  const CubeLut base = shipped_lut("HDR grading base ver.9.0 33 point.cube");
  const CubeLut contrast = shipped_lut("HDR-contrast-Enhancement-L.cube");
  const CubeLut highlight = shipped_lut("Highlight-Enhancement-L.cube");
  const CubeLut skin = shipped_lut("skin fix.cube");
  const std::vector<const CubeLut*> luts{&base, &contrast, &contrast, &contrast, &highlight, &highlight,
                                         &skin, &contrast, &contrast, &contrast, &contrast};
  const std::vector<double> mixes{1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0};
  return compose_luts(luts, mixes);
}

// Le grade que le lecteur pose sur une sortie HDR10 de `peak` nits pour une source SDR (route native,
// réglages par défaut de decide_grade), la pile en premier emplacement.
inline GradeParams player_params(double peak) {
  GradeParams p;
  p.peak = peak;
  p.paper = 0.5 * peak;
  p.adapt = 1.0;
  p.sdr_white = 203.0;
  p.shadows = 0.0;  // « Noirs profonds » à 0 : la courbe de la chaîne seule
  p.shadow_knee = 0.25 * 203.0;
  p.spec_gain = 1.661;  // highlights_rtx = 0,25
  p.look_mixes[0] = 1.0;
  p.look_pq = 1.0;
  p.look_clean = 1.0;
  return p;
}

// Les nits affichés d'un code SDR 8 bits (gamma 2,4, blanc à 203 nits) avant le grade.
inline double sdr_code_nits(double code) {
  return std::pow(code / 255.0, 2.4) * 203.0;
}

// Le lecteur, sortie HDR10 : la pile posée, la dernière passe réglée pour l'écran (blanc SDR de 203 nits).
inline void set_player_output(Gpu& gpu, const CubeLut& stack, double peak) {
  std::array<const CubeLut*, kLookSlots> slots{};
  slots[0] = &stack;
  REQUIRE(gpu.grade->set_looks(slots));
  gpu.grade->set_kwin_match(Grade::KwinMatch{.reference = 203.0, .peak = peak});
}

// Le centre d'une image uniforme passée par la chaîne : ses nits (R).
inline double chain_level(Gpu& gpu, double nits, const GradeParams& params) {
  return graded(gpu, uniform(nits), params)[kSide / 2];
}

}  // namespace ajp::render::gpu_test
