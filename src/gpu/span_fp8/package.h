// Le paquet de poids du SPAN entier en FP8 : ce que tools/span/derive_span_fp8.py écrit et ce que les noyaux
// CUTLASS du backend TensorRT chargent (docs/span-fp8-cutlass.md). Un fichier « .span8 », petit-boutiste :
//
//   en-tête de 40 octets : « AJPSPAN8 », version (u32 = 2 ; 1 reste lue), blocs (u32, 2 à 8 ; 8 en
//   version 1), canaux rangés (u32 = 64), canaux réels du modèle (u32 : 48 ou 64 ; 0 en version 1, lu 64),
//   taille de la charge (u64), CRC-32 de la charge (u32), drapeaux (u32 : bit 0 = accumulateur f16 hors
//   blocs, étape 3.5 ; bit 1 = blocs en FP16 sans quantification, le SPAN FP16 natif ; 0 en version 1, les
//   autres bits refusés) ; charge : les 1 + 3·blocs échelles E4M3 des
//   activations (f32, voir ci-dessous), puis conv_1, les blocs, conv_cat et to_pixel. Un modèle à 48 canaux
//   (b5f48) est rangé en 64 canaux remplis de zéros (étape 3.2b) : Relu(0) = 0 et (t + x)·tanh(0) = 0, les
//   canaux ajoutés restent nuls d'un bout à l'autre. Poids en FP16 (ceux du modèle, sans perte), rangés
//   [sortie][ligne][colonne][entrée] ; biais en f32. Les poids des convolutions de bloc sont ramenés en FP8
//   au chargement, à une échelle par canal de sortie (maximum de |poids| du canal / 448, quantize.h) : le
//   paquet ne porte donc que des valeurs que le modèle d'origine contient.
//
// Échelles : [0] sortie de conv_1 ; pour le bloc b (0 à 7) : [1+3b] sortie du premier Relu, [2+3b] du second,
// [3+3b] sortie du bloc. Une échelle est « maximum observé × kScaleHeadroom (1,25) / 448 » sur le corpus de
// calibration (derive.h, scales_from_peaks).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/span_fp8/e4m3.h"

namespace ajp::gpu::span_fp8 {

inline constexpr std::size_t kHeaderBytes = 40;
inline constexpr std::uint32_t kVersion = 2;
inline constexpr int kBlocks = 8;  // au plus ; le b8f64 en a 8, le b5f48 5
// Au moins : le premier et le dernier bloc écrivent chacun leur tranche de la concaténation, et les noyaux
// (span_fp8_net.cu) refusent un seul bloc (A02-10).
inline constexpr int kMinBlocks = 2;
inline constexpr int kChannels = 64;
inline constexpr int kUnshuffled = 12;        // canaux après le dégroupage 2×2 des 3 plans
inline constexpr int kUnshuffledPadded = 16;  // conv_1 lit 16 canaux, les 4 derniers nuls
inline constexpr int kConcatChannels = 4 * kChannels;
inline constexpr int kPixelChannels = 48;            // 3 plans × 4 × 4, avant DepthToSpace
inline constexpr int kScaleCount = 1 + 3 * kBlocks;  // au plus : 1 + 3·blocs servent
inline constexpr std::size_t kMaxPackageBytes = std::size_t{64} << 20U;  // le vrai pèse environ 2 Mo

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
  std::array<float, kScaleCount> scales{};  // 1 + 3·blocks.size() servent, les autres à 1
  Conv first;                               // conv_1 : 3×3, 16 → 64 (4 canaux nuls)
  std::vector<Block> blocks{static_cast<std::size_t>(kBlocks)};  // kMinBlocks à kBlocks
  int real_channels = kChannels;  // 48 (b5f48, rangé en 64 avec des zéros) ou 64
  Conv concat;                    // conv_cat : 1×1, 256 → 64
  Conv pixel;                     // to_pixel : 3×3, 64 → 48
  // conv_cat et to_pixel accumulent en f16 (étape 3.5a, mesuré −2,7 % au banc) ; posé par la dérivation
  // seulement si la porte passe et que le PSNR moyen perd au plus kMaxAcc16PsnrLossDb (derive.h).
  bool acc16 = false;
  // Les blocs en FP16, sans quantification (SpanFp8Weights::fp16) : le SPAN FP16 natif, tiré de l'ONNX sans
  // calibration ni porte ; les échelles valent alors 1 et ne servent pas.
  bool fp16 = false;
};

inline constexpr std::uint32_t kFlagAcc16 = 1U;
inline constexpr std::uint32_t kFlagFp16 = 2U;

// Le dégroupage prend des blocs 2×2 : hauteur et largeur paires. La dernière tuile de 128 lignes des noyaux
// garde ses lignes au-delà de M (étape 3.1) : plus de multiple de 128. Plafonds (B02-04) : kMaxSide par côté,
// et kMaxPixels en tout, parce que Launcher::init calcule les pas CUTLASS en int (256 canaux × W/2 × H/2 ne
// tient plus au-delà de 33,5 Mpx).
inline constexpr int kMaxSide = 16384;
inline constexpr long long kMaxPixels = 7680LL * 4320LL;  // 8K UHD
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

// Échelles qui servent pour ce nombre de blocs.
constexpr std::size_t scale_count(std::size_t blocks) {
  return 1 + 3 * blocks;
}

}  // namespace ajp::gpu::span_fp8
