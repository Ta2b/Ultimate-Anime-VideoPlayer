// Le corpus de cas durs de la porte du SPAN FP8 (étape 3.7a) : des images synthétiques, déterministes, que
// la vidéo de calibration n'apporte pas toujours et où la quantification se voit d'abord : traits d'un pixel
// à 0, 30, 45 et 90°, rampes sombres en paliers de 8 bits, bords saturés, motifs fins, hautes lumières,
// grain. Elles ne calibrent pas : la dérivation les juge seulement, au 99,9e centile de l'erreur et au pire
// PSNR (derive.h, judge_hardened).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ajp::gpu::span_fp8 {

struct HardCase {
  std::string name;
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;  // planaire 3×H×W, 8 bits
};

// Les cas durs à width × height (paires, au moins 16 × 16), toujours dans le même ordre et aux mêmes
// octets. Vide pour une taille plus petite.
std::vector<HardCase> hard_cases(int width, int height);

}  // namespace ajp::gpu::span_fp8
