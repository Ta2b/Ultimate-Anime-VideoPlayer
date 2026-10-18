// Le modèle tel que la construction le lit, vérifié contre son ModelSpec.
//
// Rien n'est corrigé : un hachage, une précision, une entrée ou une forme qui
// ne correspondent pas sont refusés avant de lancer le constructeur.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "prepare/onnx_scan.h"
#include "prepare/trtexec_command.h"

namespace ajp::prepare {

// Plus grand ONNX lu : 2 Gio, la limite d'un message protobuf. Au-delà, les
// poids sont forcément externes.
inline constexpr std::uintmax_t kMaxOnnxBytes = std::uintmax_t{2} << 30U;

struct ModelSource {
  std::vector<std::byte> graph;
  OnnxSummary summary;
  OnnxInput input;          // l'entrée nommée du graphe
  std::string content_key;  // onnx_source_key : graphe et poids externes
};

// Lit l'ONNX, vérifie son SHA-256, ses poids externes, son entrée et sa précision.
std::expected<ModelSource, Refusal> read_model(const gpu::ModelSpec& model, std::string_view input_name);

// Le type d'entrée du graphe décide de la précision sous TensorRT 11 : il doit
// être celui que ModelSpec annonce.
std::expected<void, Refusal> check_precision(const gpu::ModelSpec& model, const OnnxInput& input);

// Un graphe à GridSample est un RIFE : livré en FP16 avec ses îlots FP32.
std::expected<Recipe, Refusal> recipe_for(const gpu::ModelSpec& model, const OnnxSummary& summary);

// Formes du profil de construction. Une entrée entièrement statique n'a pas de
// profil (nullopt) et doit égaler les formes demandées ; sinon chaque dimension
// fixe du graphe doit se retrouver dans min, opt et max.
std::expected<std::optional<gpu::ShapeSet>, Refusal> profile_shapes(const OnnxInput& input,
                                                                    const gpu::ShapeSet& shapes);

}  // namespace ajp::prepare
