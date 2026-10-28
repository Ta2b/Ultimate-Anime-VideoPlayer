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

// Le graphe du SPAN Balanced HD V3.1 à 8 blocs de 64 canaux « unshuffle » : noms et formes des poids, les 25
// tenseurs que les noyaux quantifient, le dégroupage 2×2 et DepthToSpace(4) du modèle. Refus
// FormatUnsupported, avec le motif : un autre SPAN (le b5f48 à 48 canaux, par exemple) ne se prend pas.
// Les échelles du paquet rendu valent 1 : à remplacer par celles de la calibration.
std::expected<Package, Refusal> package_from_onnx(const ir::onnx::Model& model);

// read_onnx puis package_from_onnx. Refus FileUnreadable si le fichier ne se lit pas.
std::expected<Package, Refusal> package_from_onnx_file(const std::filesystem::path& path);

}  // namespace ajp::gpu::span_fp8
