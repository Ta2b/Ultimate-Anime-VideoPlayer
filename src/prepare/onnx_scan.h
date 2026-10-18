// Lecture minimale d'un ONNX au niveau du format protobuf, sans dépendance onnx.
//
// Repris de animejanai-inference-rtx@118c272 src/onnx_flow.cpp:14-58 (lecteur
// protobuf) et 104-176 (aji_onnx_external_locations). Le résumé du graphe
// (entrées, nombre de GridSample) est propre à ajp : il sert à choisir les
// réglages de construction et à refuser une précision qui ne correspond pas.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

// TensorProto.DataType d'ONNX, pour les types qui nous concernent.
enum class OnnxType : std::int32_t {
  Undefined = 0,
  Float = 1,
  Uint8 = 2,
  Int8 = 3,
  Float16 = 10,
};

struct OnnxInput {
  std::string name;
  std::int32_t elem_type = 0;      // OnnxType, ou un autre code ONNX
  std::vector<std::int64_t> dims;  // -1 : dimension symbolique ou absente
  bool is_static() const;
};

struct OnnxSummary {
  std::vector<OnnxInput> inputs;                // entrées du graphe, initialiseurs exclus
  std::vector<std::string> external_locations;  // triés, sans doublon
  std::size_t grid_samples = 0;                 // nœuds GridSample du graphe principal
};

// Emplacements de tous les poids externes (graphes imbriqués, tenseurs creux et
// fonctions compris), dédoublonnés. Des métadonnées malformées sont refusées.
std::expected<std::vector<std::string>, Refusal> onnx_external_locations(std::span<const std::byte> model);

std::expected<OnnxSummary, Refusal> scan_onnx(std::span<const std::byte> model);

// Un emplacement de poids reste sous le dossier du modèle : relatif, sans « .. »,
// sans « \ » ni « : » (repris de model_identity.cpp:52-60).
bool is_safe_location(const std::string& location);

}  // namespace ajp::prepare
