// Outils des tests sur le vrai SDK Maxine et la RTX : emplacement du SDK, GPU, image source.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "cuda_driver.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr::test {

// Racine du SDK installé (AJP_MAXINE_SDK_DIR) ; SKIP si ses en-têtes n'y sont pas.
std::filesystem::path require_maxine_sdk();

// Contexte CUDA courant sur l'appareil 0 ; SKIP sans pilote ni appareil.
std::unique_ptr<Cuda> require_cuda();

// Configuration d'une source SDR BT.709 plage limitée, aux réglages par défaut du plan.
TrueHdrConfig sdr_config(int width, int height, std::uintptr_t cuda_context);

// Image source synthétique en Y'CbCr 10 bits BT.709 plage limitée 4:4:4, puis convertie comme
// gpu le fera : vraie matrice et vraie plage (kernels/spec/color.h), tramage triangulaire de
// ±1 code 8 bits avant la quantification (AH-371, AH-392), ou pas de tramage.
// Bandes horizontales : rampe de gris complète (0 à 25 %), rampe de chroma (25 à 50 %),
// rampe quasi noire de 0 à 16 niveaux 8 bits, soit 4 codes 10 bits par niveau (50 à 100 %).
std::vector<std::uint32_t> source_rgba8(int width, int height, bool dither);

// Ligne de début de la rampe quasi noire de source_rgba8.
int near_black_row(int height);

// Image unie R = G = B = level (8 bits), alpha opaque.
std::vector<std::uint32_t> flat_rgba8(int width, int height, std::uint32_t level);

// Composantes d'un pixel RGB10A2 : R bits 0-9, G 10-19, B 20-29, A 30-31.
constexpr std::uint32_t red10(std::uint32_t pixel) {
  return pixel & 0x3FFU;
}
constexpr std::uint32_t alpha2(std::uint32_t pixel) {
  return pixel >> 30U;
}

}  // namespace ajp::vendor_hdr::test
