#include "gpu/ir/onnx_io_type.h"

#include <algorithm>
#include <array>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"

namespace ajp::gpu::ir::onnx {
namespace {

using proto::Bytes;
using proto::Field;
using proto::Writer;

// Numéros de champ d'onnx.proto (ONNX 1.17).
constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kGraphNode = 1;
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kGraphValueInfo = 13;
constexpr std::uint32_t kNodeInput = 1;
constexpr std::uint32_t kNodeOutput = 2;
constexpr std::uint32_t kNodeName = 3;
constexpr std::uint32_t kNodeOpType = 4;
constexpr std::uint32_t kNodeAttribute = 5;
constexpr std::uint32_t kAttributeName = 1;
constexpr std::uint32_t kAttributeInt = 3;
constexpr std::uint32_t kAttributeType = 20;
constexpr std::uint64_t kAttributeTypeInt = 2;
constexpr std::uint32_t kValueName = 1;
constexpr std::uint32_t kValueType = 2;
constexpr std::uint32_t kTypeTensor = 1;
constexpr std::uint32_t kTensorElemType = 1;
// Cast prend son type cible en entier (attribut to) depuis l'opset 6.
constexpr std::int64_t kCastOpset = 6;

std::unexpected<Refusal> unavailable(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, std::move(detail)});
}

// Appelle fn sur chaque champ du message ; s'arrête au premier refus.
template <class Fn>
std::expected<void, Refusal> for_each_field(Bytes bytes, Fn&& fn) {
  proto::Reader reader(bytes);
  for (;;) {
    auto field = reader.next();
    if (!field) return std::unexpected(field.error());
    if (!*field) return {};
    if (auto ok = fn(**field); !ok) return ok;
  }
}

bool is_float(std::int32_t elem_type) {
  return elem_type == kFloat || elem_type == kFloat16;
}

// Une entrée ou sortie retypée : son nom dans le fichier (qui prend le type voulu), le nom
// interne que les nœuds prennent à sa place, et son type d'origine, celui du nom interne.
struct Retyped {
  std::string name;
  std::string internal;
  std::int32_t from = 0;
};

struct Plan {
  std::vector<Retyped> inputs;
  std::vector<Retyped> outputs;
  std::unordered_map<std::string, std::string> rename;  // nom du fichier → nom interne
};

std::expected<Plan, Refusal> plan_of(const Model& model, std::int32_t elem_type) {
  const Graph& g = model.graph;
  for (const Node& node : g.nodes) {
    for (const Attribute& a : node.attributes) {
      if (a.type == AttributeType::Graph || a.type == AttributeType::Graphs)
        return unavailable(std::format("{} : sous-graphe, interface non réécrite", node.op_type));
    }
  }
  for (const OpsetId& opset : model.opsets) {
    if ((opset.domain.empty() || opset.domain == "ai.onnx") && opset.version < kCastOpset)
      return unavailable(std::format("opset {} : Cast sans type cible", opset.version));
  }
  std::unordered_set<std::string> weights;
  for (const Tensor& w : g.initializers) weights.insert(w.name);
  std::unordered_set<std::string> taken = weights;  // tous les noms du graphe
  for (const Node& n : g.nodes) {
    taken.insert(n.inputs.begin(), n.inputs.end());
    taken.insert(n.outputs.begin(), n.outputs.end());
  }
  for (const ValueInfo& v : g.inputs) taken.insert(v.name);
  for (const ValueInfo& v : g.outputs) taken.insert(v.name);
  Plan plan;
  const auto add = [&](const ValueInfo& v, std::vector<Retyped>& into) -> std::expected<void, Refusal> {
    if (!is_float(v.elem_type) || v.elem_type == elem_type) return {};
    if (plan.rename.contains(v.name))
      return unavailable(std::format("{} : à la fois entrée et sortie", v.name));
    const std::string_view suffix = v.elem_type == kFloat ? "f32" : "f16";
    std::string internal = std::format("{}.{}", v.name, suffix);
    for (int k = 2; taken.contains(internal); ++k) internal = std::format("{}.{}{}", v.name, suffix, k);
    taken.insert(internal);
    plan.rename.emplace(v.name, internal);
    into.push_back({.name = v.name, .internal = internal, .from = v.elem_type});
    return {};
  };
  for (const ValueInfo& v : g.inputs) {
    if (weights.contains(v.name)) continue;  // avant l'IR 4, les poids figurent parmi les entrées
    if (auto ok = add(v, plan.inputs); !ok) return std::unexpected(ok.error());
  }
  for (const ValueInfo& v : g.outputs) {
    if (weights.contains(v.name)) return unavailable(std::format("{} : sortie qui est un poids", v.name));
    if (auto ok = add(v, plan.outputs); !ok) return std::unexpected(ok.error());
  }
  return plan;
}

// Un nœud Cast de from vers to, dans le type to_type.
void write_cast(Writer& graph, std::string_view from, std::string_view to, std::int32_t to_type) {
  Writer attr;
  attr.string(kAttributeName, "to");
  attr.varint(kAttributeInt, static_cast<std::uint64_t>(to_type));
  attr.varint(kAttributeType, kAttributeTypeInt);
  Writer node;
  node.string(kNodeInput, from);
  node.string(kNodeOutput, to);
  node.string(kNodeName, std::format("ajp.cast.{}", to));
  node.string(kNodeOpType, "Cast");
  node.message(kNodeAttribute, attr);
  graph.message(kGraphNode, node);
}

// Le nom (champ name) d'un ValueInfoProto.
std::expected<std::string, Refusal> name_of(Bytes info) {
  std::string name;
  auto ok = for_each_field(info, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kValueName) return {};
    auto value = proto::as_string(field);
    if (!value) return std::unexpected(value.error());
    name = std::move(*value);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return name;
}

// Recopie le message en remplaçant, dans les champs chaîne de ces numéros, les noms renommés.
std::expected<void, Refusal> copy_renamed(Bytes message, std::span<const std::uint32_t> numbers,
                                          const Plan& plan, std::uint32_t as, Writer& out) {
  Writer w;
  auto ok = for_each_field(message, [&](const Field& field) -> std::expected<void, Refusal> {
    if (std::ranges::find(numbers, field.number) != numbers.end() && field.wire == proto::WireType::Length) {
      auto name = proto::as_string(field);
      if (!name) return std::unexpected(name.error());
      if (const auto renamed = plan.rename.find(*name); renamed != plan.rename.end()) {
        w.string(field.number, renamed->second);
        return {};
      }
    }
    w.field(field);
    return {};
  });
  if (!ok) return ok;
  out.message(as, w);
  return {};
}

// Le ValueInfoProto avec elem_type remplacé : type et tensor_type sont recopiés champ par
// champ jusqu'à lui, nom et forme restent.
std::expected<void, Refusal> copy_retyped(Bytes info, std::int32_t elem_type, std::uint32_t as, Writer& out) {
  Writer w;
  auto ok = for_each_field(info, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kValueType || field.wire != proto::WireType::Length) {
      w.field(field);
      return {};
    }
    Writer type;
    auto inner = for_each_field(field.bytes, [&](const Field& t) -> std::expected<void, Refusal> {
      if (t.number != kTypeTensor || t.wire != proto::WireType::Length) {
        type.field(t);
        return {};
      }
      Writer tensor;
      bool typed = false;
      auto fields = for_each_field(t.bytes, [&](const Field& e) -> std::expected<void, Refusal> {
        if (e.number == kTensorElemType) {
          tensor.varint(kTensorElemType, static_cast<std::uint64_t>(elem_type));
          typed = true;
        } else {
          tensor.field(e);
        }
        return {};
      });
      if (!fields) return fields;
      if (!typed) tensor.varint(kTensorElemType, static_cast<std::uint64_t>(elem_type));
      type.message(kTypeTensor, tensor);
      return {};
    });
    if (!inner) return inner;
    w.message(kValueType, type);
    return {};
  });
  if (!ok) return ok;
  out.message(as, w);
  return {};
}

std::expected<void, Refusal> copy_graph(Bytes graph, const Plan& plan, std::int32_t elem_type, Writer& out) {
  Writer w;
  // Les Cast des entrées avant le premier nœud : les nœuds restent dans l'ordre topologique.
  bool casts_in = false;
  const auto input_casts = [&] {
    if (casts_in) return;
    casts_in = true;
    for (const Retyped& r : plan.inputs) write_cast(w, r.name, r.internal, r.from);
  };
  constexpr std::array<std::uint32_t, 2> kNodeNames{kNodeInput, kNodeOutput};
  constexpr std::array<std::uint32_t, 1> kValueNames{kValueName};
  auto ok = for_each_field(graph, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.wire != proto::WireType::Length) {
      w.field(field);
      return {};
    }
    switch (field.number) {
      case kGraphNode:
        input_casts();
        return copy_renamed(field.bytes, kNodeNames, plan, kGraphNode, w);
      case kGraphInput:
      case kGraphOutput: {
        auto name = name_of(field.bytes);
        if (!name) return std::unexpected(name.error());
        if (!plan.rename.contains(*name)) {
          w.field(field);
          return {};
        }
        return copy_retyped(field.bytes, elem_type, field.number, w);
      }
      case kGraphValueInfo:
        // Une forme connue d'un tenseur retypé suit son nom interne, dans son type d'origine.
        return copy_renamed(field.bytes, kValueNames, plan, kGraphValueInfo, w);
      default:
        w.field(field);
        return {};
    }
  });
  if (!ok) return ok;
  input_casts();
  for (const Retyped& r : plan.outputs) write_cast(w, r.internal, r.name, elem_type);
  out.message(kModelGraph, w);
  return {};
}

}  // namespace

std::expected<std::vector<std::byte>, Refusal> with_io_type(std::span<const std::byte> model,
                                                            std::int32_t elem_type) {
  if (!is_float(elem_type))
    return unavailable(std::format("type d'interface {} : ni FLOAT ni FLOAT16", elem_type));
  auto decoded = read_onnx(model);
  if (!decoded) return std::unexpected(decoded.error());
  auto plan = plan_of(*decoded, elem_type);
  if (!plan) return std::unexpected(plan.error());
  if (plan->rename.empty()) return std::vector<std::byte>{};
  Writer out;
  auto ok = for_each_field(model, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == kModelGraph && field.wire == proto::WireType::Length)
      return copy_graph(field.bytes, *plan, elem_type, out);
    out.field(field);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return std::move(out).take();
}

}  // namespace ajp::gpu::ir::onnx
