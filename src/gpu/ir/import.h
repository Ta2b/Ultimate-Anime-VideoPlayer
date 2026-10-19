// Import d'un modèle ONNX vers l'IR : tenseurs, nœuds triés, attributs typés.
#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/ir/onnx_model.h"

namespace ajp::gpu::ir {

// Opsets du domaine ONNX par défaut dont la sémantique est connue de l'import.
inline constexpr std::int64_t kMinOpset = 11;
inline constexpr std::int64_t kMaxOpset = 23;

// Les nœuds Constant deviennent des tenseurs constants ; un opérateur hors du
// jeu de l'IR est refusé (OperatorUnknown) avec son nom.
std::expected<Graph, Refusal> import_model(const onnx::Model& model);

// read_onnx() puis import_model().
std::expected<Graph, Refusal> load_onnx(std::span<const std::byte> bytes);
std::expected<Graph, Refusal> load_onnx_file(const std::filesystem::path& path);

}  // namespace ajp::gpu::ir
