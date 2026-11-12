// Décodeur ONNX sans dépendance, au niveau du format filaire protobuf.
//
// Numéros de champ d'onnx.proto (ONNX 1.17). Seuls les champs déclarés comme
// messages sont relus en protobuf : les données brutes et les chaînes ne le sont
// jamais. Approche voisine de animejanai-inference-rtx@118c272 src/onnx_shape.cpp,
// réécrite pour décoder tout le graphe et refuser au lieu d'ignorer.
#include <format>
#include <limits>
#include <optional>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/ir/onnx_reader_detail.h"
#include "gpu/ir/protobuf.h"

namespace ajp::gpu::ir::onnx {
namespace {

using namespace detail;

std::expected<Node, Refusal> read_node(Bytes bytes) {
  Node node;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1:
        return push_string(field, node.inputs);
      case 2:
        return push_string(field, node.outputs);
      case 3:
        return read_string(field, node.name);
      case 4:
        return read_string(field, node.op_type);
      case 5: {
        if (field.wire != proto::WireType::Length) return proto::malformed("attribut : message attendu");
        if (node.attributes.size() >= kMaxAttributes) return proto::malformed("trop d'attributs sur un nœud");
        auto attr = read_attribute(field.bytes);
        if (!attr) return std::unexpected(attr.error());
        node.attributes.push_back(std::move(*attr));
        return {};
      }
      case 7:
        return read_string(field, node.domain);
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  return node;
}

std::expected<void, Refusal> read_dimension(Bytes bytes, ValueInfo& info) {
  std::int64_t value = -1;
  std::string symbol;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) {
      auto dim = proto::as_int64(field);
      if (!dim) return std::unexpected(dim.error());
      // Certains exports (RIFE 4.26) écrivent -1 pour une dimension dynamique : elle reste
      // inconnue, comme une dimension sans valeur. Toute autre valeur négative est refusée.
      if (*dim < -1) return proto::malformed(std::format("{} : dimension {}", info.name, *dim));
      value = *dim;
    } else if (field.number == 2) {
      return read_string(field, symbol);
    }
    return {};
  });
  if (!ok) return ok;
  // Un rang au-delà de kMaxRank est refusé à l'import : le décodeur le dit dès la dimension qui le dépasse,
  // sans accumuler des millions de dimensions vides (B44-01).
  if (info.dims.size() >= kMaxRank)
    return proto::malformed(std::format("{} : rang au-delà de {}", info.name, kMaxRank));
  info.dims.push_back(value);
  info.symbols.push_back(value >= 0 ? std::string{} : (symbol.empty() ? std::string{"?"} : symbol));
  return {};
}

std::expected<void, Refusal> read_tensor_type(Bytes bytes, ValueInfo& info) {
  return for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) {
      auto type = proto::as_int32(field);
      if (!type) return std::unexpected(type.error());
      info.elem_type = *type;
    } else if (field.number == 2) {
      if (field.wire != proto::WireType::Length) return proto::malformed("forme : message attendu");
      info.has_shape = true;
      return for_each_field(field.bytes, [&](const Field& dim) -> std::expected<void, Refusal> {
        if (dim.number != 1) return {};
        if (dim.wire != proto::WireType::Length) return proto::malformed("dimension : message attendu");
        return read_dimension(dim.bytes, info);
      });
    }
    return {};
  });
}

std::expected<ValueInfo, Refusal> read_value_info(Bytes bytes) {
  ValueInfo info;
  bool tensor_type = false;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) return read_string(field, info.name);
    if (field.number != 2) return {};
    if (field.wire != proto::WireType::Length) return proto::malformed("type : message attendu");
    return for_each_field(field.bytes, [&](const Field& type) -> std::expected<void, Refusal> {
      if (type.number != 1) return {};
      if (type.wire != proto::WireType::Length) return proto::malformed("type tenseur : message attendu");
      tensor_type = true;
      return read_tensor_type(type.bytes, info);
    });
  });
  if (!ok) return std::unexpected(ok.error());
  if (!tensor_type) return unsupported(std::format("{} : seules les entrées et sorties tenseurs", info.name));
  return info;
}

std::expected<void, Refusal> push_value_info(const Field& field, std::vector<ValueInfo>& out) {
  if (field.wire != proto::WireType::Length) return proto::malformed("ValueInfoProto : message attendu");
  if (out.size() >= kMaxValueInfos) return proto::malformed("trop d'entrées ou de sorties");
  auto info = read_value_info(field.bytes);
  if (!info) return std::unexpected(info.error());
  out.push_back(std::move(*info));
  return {};
}

std::expected<Graph, Refusal> read_graph(Bytes bytes) {
  Graph graph;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1: {
        if (field.wire != proto::WireType::Length) return proto::malformed("nœud : message attendu");
        if (graph.nodes.size() >= kMaxNodes) return proto::malformed("trop de nœuds");
        auto node = read_node(field.bytes);
        if (!node) return std::unexpected(node.error());
        graph.nodes.push_back(std::move(*node));
        return {};
      }
      case 2:
        return read_string(field, graph.name);
      case 5:
        if (graph.initializers.size() >= kMaxInitializers) return proto::malformed("trop d'initialiseurs");
        return push_tensor(field, graph.initializers);
      case 11:
        return push_value_info(field, graph.inputs);
      case 12:
        return push_value_info(field, graph.outputs);
      case 15:
        return unsupported("initialiseurs creux");
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  return graph;
}

std::expected<OpsetId, Refusal> read_opset(Bytes bytes) {
  OpsetId opset;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) return read_string(field, opset.domain);
    if (field.number != 2) return {};
    auto version = proto::as_int64(field);
    if (!version) return std::unexpected(version.error());
    opset.version = *version;
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return opset;
}

}  // namespace

std::expected<Model, Refusal> read_onnx(std::span<const std::byte> bytes) {
  Model model;
  bool has_graph = false;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1: {
        auto version = proto::as_int64(field);
        if (!version) return std::unexpected(version.error());
        model.ir_version = *version;
        return {};
      }
      case 2:
        return read_string(field, model.producer_name);
      case 3:
        return read_string(field, model.producer_version);
      case 7: {
        if (field.wire != proto::WireType::Length) return proto::malformed("graphe : message attendu");
        if (has_graph) return proto::malformed("plusieurs graphes");
        auto graph = read_graph(field.bytes);
        if (!graph) return std::unexpected(graph.error());
        model.graph = std::move(*graph);
        has_graph = true;
        return {};
      }
      case 8: {
        if (field.wire != proto::WireType::Length) return proto::malformed("opset : message attendu");
        if (model.opsets.size() >= kMaxOpsets) return proto::malformed("trop de domaines d'opérateurs");
        auto opset = read_opset(field.bytes);
        if (!opset) return std::unexpected(opset.error());
        model.opsets.push_back(std::move(*opset));
        return {};
      }
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  if (!has_graph) return proto::malformed("ModelProto sans graphe");
  return model;
}

}  // namespace ajp::gpu::ir::onnx
