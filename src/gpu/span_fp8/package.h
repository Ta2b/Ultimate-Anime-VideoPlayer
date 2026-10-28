// Le paquet de poids du SPAN entier en FP8 : ce que tools/span/derive_span_fp8.py écrit et ce que les noyaux
// CUTLASS du backend TensorRT chargent (docs/span-fp8-cutlass.md). Un fichier « .span8 », petit-boutiste :
//
//   en-tête de 40 octets : « AJPSPAN8 », version (u32 = 1), blocs (u32 = 8), canaux (u32 = 64), réservé
//   (u32 = 0), taille de la charge (u64), CRC-32 de la charge (u32), réservé (u32 = 0) ;
//   charge : les 25 échelles E4M3 des activations (f32, voir ci-dessous), puis conv_1, les 8 blocs, conv_cat
//   et to_pixel. Poids en FP16 (ceux du modèle, sans perte), rangés [sortie][ligne][colonne][entrée] ; biais
//   en f32. Les poids des convolutions de bloc sont ramenés en FP8 au chargement, à l'échelle de leur tenseur
//   (maximum / 448) : le paquet ne porte donc que des valeurs que le modèle d'origine contient.
//
// Échelles : [0] sortie de conv_1 ; pour le bloc b (0 à 7) : [1+3b] sortie du premier Relu, [2+3b] du second,
// [3+3b] sortie du bloc. Une échelle est « maximum observé / 448 » sur le corpus de calibration.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::span_fp8 {

inline constexpr std::size_t kHeaderBytes = 40;
inline constexpr std::uint32_t kVersion = 1;
inline constexpr int kBlocks = 8;
inline constexpr int kChannels = 64;
inline constexpr int kUnshuffled = 12;        // canaux après le dégroupage 2×2 des 3 plans
inline constexpr int kUnshuffledPadded = 16;  // conv_1 lit 16 canaux, les 4 derniers nuls
inline constexpr int kConcatChannels = 4 * kChannels;
inline constexpr int kPixelChannels = 48;  // 3 plans × 4 × 4, avant DepthToSpace
inline constexpr int kScaleCount = 1 + 3 * kBlocks;
inline constexpr std::size_t kMaxPackageBytes = std::size_t{64} << 20;  // le vrai pèse environ 3 Mo

// Une convolution : poids FP16 (motif binaire IEEE 754 demi-précision) et biais f32.
struct Conv {
  std::vector<std::uint16_t> weights;
  std::vector<float> bias;
};

struct Block {
  std::array<Conv, 3> conv;  // c1, c2, c3 (3×3, 64 → 64)
  Conv attention;            // 1×1, 64 → 64
};

struct Package {
  std::array<float, kScaleCount> scales{};
  Conv first;  // conv_1 : 3×3, 16 → 64 (4 canaux nuls)
  std::array<Block, kBlocks> blocks;
  Conv concat;  // conv_cat : 1×1, 256 → 64
  Conv pixel;   // to_pixel : 3×3, 64 → 48
};

// Les noyaux traitent des lignes de 128 pixels de caractéristiques (demi-résolution) sans garde de bord, et
// le dégroupage prend des blocs 2×2 : hauteur et largeur paires, (h/2)·(w/2) multiple de 128.
bool size_supported(int height, int width);
// Pourquoi une taille n'est pas prise ; vide si elle l'est.
std::string size_refusal(int height, int width);

// CRC-32 (IEEE 802.3) d'une suite d'octets ; le même polynôme que zlib.crc32 de l'outil Python.
std::uint32_t crc32(std::span<const std::byte> bytes);

// Lit et vérifie un paquet entier : magie, version, tailles, CRC, valeurs finies, échelles strictement
// positives. Refus FormatUnsupported (version, structure) ou FileUnreadable (tronqué, CRC faux).
std::expected<Package, Refusal> parse(std::span<const std::byte> bytes);

// Octets d'un paquet valide (pour les tests et l'outil de contrôle) ; l'inverse de parse.
std::vector<std::byte> serialize(const Package& package);

// Un paquet de la forme du vrai, rempli de valeurs régulières (tests sans modèle).
Package synthetic(float fill = 0.01F);

}  // namespace ajp::gpu::span_fp8
