// Aides communes aux deux moitiés du décodeur ONNX (onnx_reader.cpp : graphe, nœuds, entrées ;
// onnx_tensor.cpp : tenseurs et attributs).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"

namespace ajp::gpu::ir::onnx::detail {

using proto::Bytes;
using proto::Field;
using proto::for_each_field;
using proto::unsupported;

// Lit un champ chaîne dans out.
inline std::expected<void, Refusal> read_string(const Field& field, std::string& out) {
  auto value = proto::as_string(field);
  if (!value) return std::unexpected(value.error());
  out = std::move(*value);
  return {};
}

// A44-04 : un entier empaqueté sur un octet en coûte huit en mémoire, une chaîne vide de deux octets en coûte
// trente-deux. Sans borne, un fichier hostile proche de 2 Gio demandait 16 à 32 Gio et finissait en bad_alloc
// au lieu d'un refus. Aucun modèle réel n'approche ces bornes : au-delà, les valeurs sont en raw_data.
inline constexpr std::size_t kMaxTypedInts = std::size_t{1} << 24U;
inline constexpr std::size_t kMaxStrings = std::size_t{1} << 20U;
// B44-01 : de même, un nœud vide de deux octets en coûte plus de deux cents, un initialiseur ou une entrée
// autant. Aucun modèle réel n'approche ces plafonds (RIFE : quelques milliers de nœuds) ; au-delà, un refus
// typé plutôt qu'un bad_alloc loin de la cause.
inline constexpr std::size_t kMaxNodes = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxInitializers = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxValueInfos = std::size_t{1} << 16U;
inline constexpr std::size_t kMaxAttributes = std::size_t{1} << 12U;
inline constexpr std::size_t kMaxOpsets = std::size_t{1} << 10U;

inline std::expected<void, Refusal> push_string(const Field& field, std::vector<std::string>& out) {
  if (out.size() >= kMaxStrings) return proto::malformed("trop de chaînes dans une liste");
  auto value = proto::as_string(field);
  if (!value) return std::unexpected(value.error());
  out.push_back(std::move(*value));
  return {};
}

// Un champ empaqueté tient au plus un entier par octet : la borne se vérifie avant de le lire.
inline std::expected<void, Refusal> append_ints(const Field& field, std::vector<std::int64_t>& out) {
  const std::size_t more = field.wire == proto::WireType::Length ? field.bytes.size() : 1;
  if (out.size() + more > kMaxTypedInts) return proto::malformed("trop de valeurs entières typées");
  return proto::append_int64s(field, out);
}

// Un TensorProto (initialiseur ou valeur d'attribut) ajouté à out.
std::expected<void, Refusal> push_tensor(const Field& field, std::vector<Tensor>& out);
// Un AttributeProto.
std::expected<Attribute, Refusal> read_attribute(Bytes bytes);

}  // namespace ajp::gpu::ir::onnx::detail
