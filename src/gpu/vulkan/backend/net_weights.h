// Poids des réseaux rangés pour les noyaux (blobs du programme), une fois, à l'abaissement.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/ir/ir.h"

namespace ajp::gpu::vk::net {

enum class BlobType : std::uint8_t { F32, F16 };

// Poids d'une Conv directe : ONNX [cout][cin / groupes][kh][kw] vers
// [groupes × blocs][kh][kw][cin / groupes][lanes], un bloc portant lanes canaux de sortie d'un
// même groupe, complété de zéros (net_conv.comp).
std::vector<std::byte> conv_weights(const ir::Tensor& w, std::uint32_t groups, std::uint32_t lanes,
                                    BlobType type);

// Même rangement depuis les poids ONNX d'une ConvTranspose [cin][cout / groupes][kh][kw].
std::vector<std::byte> conv_transpose_weights(const ir::Tensor& w, std::uint32_t groups, std::uint32_t lanes);

// Poids d'une Conv par matrices coopératives : la matrice B du produit implicite, K × Np en
// float16, ligne k = (ky × kw + kx) × cinp + ci (cinp : cin arrondi à 16, rangs de bourrage
// nuls), colonne = canal de sortie (Np : cout arrondi à 16, colonnes nulles). Groupe unique.
std::vector<std::byte> matrix_weights(const ir::Tensor& w, std::uint32_t cin_padded,
                                      std::uint32_t cout_padded);

// Poids d'une Conv par matrices coopératives FP8 : la même matrice B en octets E4M3
// (fp8.bytes, ir::recover_fp8_weights), K × Np octets, rangs et colonnes de bourrage nuls.
std::vector<std::byte> matrix_weights_fp8(const ir::Dims& dims, const std::vector<std::uint8_t>& e4m3,
                                          std::uint32_t cin_padded, std::uint32_t cout_padded);

// Poids d'une Conv 3 × 3 par Winograd F(2×2, 3×3) : U = G g Gᵀ de chaque paire (cout, cin),
// 4 × 4 valeurs, rangées [bloc][cin][position][lanes] (en float ou en float16), un bloc portant
// lanes canaux de sortie complété de zéros (net_conv_wino.glsl). Groupe unique.
std::vector<std::byte> winograd_weights(const ir::Tensor& w, std::uint32_t lanes, BlobType type);

// Un vecteur par canal de sortie en float (biais, pente PRelu), complété de zéros jusqu'à
// count ; absent (nullptr) : des zéros.
std::vector<std::byte> channel_floats(const ir::Tensor* values, std::uint32_t count);

// Biais puis échelles du noyau FP8, 2 × count floats : les biais complétés de zéros, puis
// in_scale × échelle de chaque canal de sortie (net_conv_coopmat_fp8.glsl, bias.v[count + canal]).
std::vector<std::byte> fp8_bias_and_scales(const ir::Tensor* bias, const std::vector<float>& channel_scale,
                                           float in_scale, std::uint32_t count);

// Une constante lue comme tenseur par un noyau : ses valeurs dans son type, en ordre ligne.
std::vector<std::byte> constant_blob(const ir::Tensor& constant);

}  // namespace ajp::gpu::vk::net
