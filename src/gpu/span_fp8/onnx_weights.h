// Les poids du SPAN lus dans son ONNX, rangés comme le paquet « .span8 » les veut : ce que
// tools/span/derive_span_fp8.py fait avec numpy et onnx (weights_from_onnx, krsc, build_package), en C++ et
// sur le lecteur ONNX du module gpu_ir. Sans GPU : les échelles d'activation viennent de la calibration.
#pragma once

#include <expected>
#include <filesystem>
#include <span>

#include "core/types.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::span_fp8 {

// Le graphe d'un SPAN « unshuffle » d'AnimeJaNai : noms et formes des poids, les 1 + 3·blocs tenseurs que les
// noyaux quantifient, le dégroupage 2×2 et DepthToSpace(4) du modèle. Le b8f64 (8 blocs de 64 canaux) et le
// b5f48 (5 blocs de 48 canaux réels, rangés en 64) se prennent ; un autre graphe est refusé
// (FormatUnsupported), avec le motif.
// Les échelles du paquet rendu valent 1 : à remplacer par celles de la calibration.
std::expected<Package, Refusal> package_from_onnx(const ir::onnx::Model& model);

// read_onnx puis package_from_onnx. Refus FileUnreadable si le fichier ne se lit pas.
std::expected<Package, Refusal> package_from_onnx_file(const std::filesystem::path& path);

}  // namespace ajp::gpu::span_fp8
