// Le paquet « .span8 » du SPAN entier en FP8 (gpu/span_fp8/package.h, docs/span-fp8-cutlass.md) sous le
// moteur Vulkan : son graphe (gpu/span_fp8/package_model.h), importé comme un ONNX, passe par les mêmes
// passes, le même autotuner, la même porte et le même cache que les autres réseaux. Les convolutions des
// blocs y lisent leur entrée arrondie à la grille E4M3 (ConvAttrs::in_quant) avec des poids sur cette grille
// : sur une carte à matrices coopératives FP8 (RDNA 4, Target::matrix_f8), l'autotuner les essaie en
// coopmat_fp8 ; ailleurs, le noyau direct calcule le même arrondi en float.
#pragma once

#include <expected>
#include <filesystem>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/span_fp8/package.h"
#include "gpu/span_fp8/package_model.h"

namespace ajp::gpu::vk::net {

// La version du graphe d'un paquet : à augmenter à chaque changement de package_model ; elle entre dans
// BackendCaps::span_fp8_kernels (les reçus d'un autre graphe se refont, app/play/span_int8.cpp).
inline constexpr int kSpan8GraphVersion = 1;

// L'extension d'un paquet : « .span8 ».
bool is_span8(const std::filesystem::path& path);

// Le graphe importé d'un paquet lu (formes encore à inférer). Refus : ceux de package_model et de l'import.
std::expected<ir::Graph, Refusal> span8_graph(const span_fp8::Package& package,
                                              const span_fp8::ModelOptions& options = {});

// Le graphe d'un fichier de réseau : un paquet « .span8 » (lu, vérifié par span_fp8::parse), sinon un ONNX
// (ir::load_onnx_file). Refus FileUnreadable : paquet illisible ; ceux de parse et de l'import.
std::expected<ir::Graph, Refusal> load_network_graph(const std::filesystem::path& path);

}  // namespace ajp::gpu::vk::net
