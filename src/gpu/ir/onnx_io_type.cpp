#include "gpu/ir/onnx_io_type.h"

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir::onnx {
namespace {

using proto::Bytes;
using proto::Field;
using proto::Writer;

// Numéros de champ d'onnx.proto (ONNX 1.17).
constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kModelMetadata = 14;
constexpr std::uint32_t kEntryKey = 1;
constexpr std::uint32_t kEntryValue = 2;
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
constexpr std::uint32_t kTensorShape = 2;
constexpr std::uint32_t kShapeDim = 1;
constexpr std::uint32_t kDimValue = 1;
// Cast prend son type cible en entier (attribut to) depuis l'opset 6.
constexpr std::int64_t kCastOpset = 6;
constexpr std::uint32_t kModelFunctions = 25;
constexpr std::uint32_t kGraphInitializer = 5;
constexpr std::uint32_t kGraphSparseInitializer = 15;
constexpr std::uint32_t kNodeDomain = 7;
constexpr std::uint32_t kAttributeTensor = 5;
constexpr std::uint32_t kAttributeGraph = 6;
constexpr std::uint32_t kAttributeTensors = 10;
constexpr std::uint32_t kAttributeGraphs = 11;
constexpr std::uint32_t kAttributeSparseTensor = 22;
constexpr std::uint32_t kAttributeSparseTensors = 23;
constexpr std::uint32_t kTensorDims = 1;
constexpr std::uint32_t kTensorDataType = 2;
constexpr std::uint32_t kTensorInt32Data = 5;
constexpr std::uint32_t kTensorRawData = 9;
constexpr std::uint32_t kTensorDataLocation = 14;
constexpr std::int64_t kDataLocationExternal = 1;

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

// Le ValueInfoProto à la forme dims : tensor_type garde son type, sa forme est réécrite.
std::expected<void, Refusal> copy_with_dims(Bytes info, std::span<const std::int64_t> dims, Writer& out) {
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
      auto fields = for_each_field(t.bytes, [&](const Field& e) -> std::expected<void, Refusal> {
        if (e.number != kTensorShape) tensor.field(e);
        return {};
      });
      if (!fields) return fields;
      Writer shape;
      for (const std::int64_t d : dims) {
        Writer dim;
        dim.varint(kDimValue, static_cast<std::uint64_t>(d));
        shape.message(kShapeDim, dim);
      }
      tensor.message(kTensorShape, shape);
      type.message(kTypeTensor, tensor);
      return {};
    });
    if (!inner) return inner;
    w.message(kValueType, type);
    return {};
  });
  if (!ok) return ok;
  out.message(kGraphInput, w);
  return {};
}

// Le TensorProto en FLOAT s'il est en FLOAT16 (raw_data, ou int32_data : un motif de 16 bits par
// élément), ses valeurs converties dans raw_data, le reste recopié ; sinon tel quel. changed : un
// tenseur converti.
std::expected<void, Refusal> copy_tensor_f32(Bytes tensor, std::uint32_t as, Writer& out, bool& changed) {
  std::int64_t data_type = 0;
  std::int64_t location = 0;
  std::vector<std::int64_t> dims;
  std::vector<std::int64_t> words;
  std::optional<Bytes> raw;
  auto scanned = for_each_field(tensor, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case kTensorDims:
        return proto::append_int64s(field, dims);
      case kTensorInt32Data:
        return proto::append_int64s(field, words);
      case kTensorDataType:
      case kTensorDataLocation: {
        auto value = proto::as_int64(field);
        if (!value) return std::unexpected(value.error());
        (field.number == kTensorDataType ? data_type : location) = *value;
        return {};
      }
      case kTensorRawData:
        if (field.wire != proto::WireType::Length)
          return proto::malformed("raw_data : pas une chaîne d'octets");
        raw = field.bytes;
        return {};
      default:
        return {};
    }
  });
  if (!scanned) return scanned;
  if (data_type != kFloat16) {
    out.bytes(as, tensor);
    return {};
  }
  if (location == kDataLocationExternal) return unavailable("tenseur FLOAT16 hors du fichier ONNX");
  std::size_t count = 1;
  for (const std::int64_t d : dims) {
    if (d < 0) return proto::malformed(std::format("tenseur FLOAT16 de dimension {}", d));
    count *= static_cast<std::size_t>(d);
  }
  if (raw ? raw->size() != count * 2 : words.size() != count) {
    return proto::malformed(std::format("tenseur FLOAT16 : {} valeurs pour {} éléments",
                                        raw ? raw->size() / 2 : words.size(), count));
  }
  std::vector<std::byte> floats(count * 4);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t half = 0;
    if (raw) {
      half = static_cast<std::uint16_t>(std::to_integer<std::uint16_t>((*raw)[2 * i]) |
                                        (std::to_integer<std::uint16_t>((*raw)[(2 * i) + 1]) << 8U));
    } else {
      if (words[i] < 0 || words[i] > 0xFFFF)
        return proto::malformed(std::format("motif FLOAT16 {} hors de 16 bits", words[i]));
      half = static_cast<std::uint16_t>(words[i]);
    }
    const auto bits = std::bit_cast<std::uint32_t>(kernels::ref::half_to_float(half));
    for (std::size_t b = 0; b < 4; ++b)
      floats[(4 * i) + b] = static_cast<std::byte>((bits >> (8U * b)) & 0xFFU);
  }
  Writer w;
  auto copied = for_each_field(tensor, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kTensorDataType && field.number != kTensorRawData && field.number != kTensorInt32Data)
      w.field(field);
    return {};
  });
  if (!copied) return copied;
  w.varint(kTensorDataType, static_cast<std::uint64_t>(kFloat));
  w.bytes(kTensorRawData, floats);
  out.message(as, w);
  changed = true;
  return {};
}

// Le ValueInfoProto en FLOAT s'il est un tenseur FLOAT16 ; nom et forme restent.
std::expected<void, Refusal> copy_value_f32(Bytes info, std::uint32_t as, Writer& out, bool& changed) {
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
      auto fields = for_each_field(t.bytes, [&](const Field& e) -> std::expected<void, Refusal> {
        if (e.number == kTensorElemType && e.wire == proto::WireType::Varint &&
            e.scalar == static_cast<std::uint64_t>(kFloat16)) {
          tensor.varint(kTensorElemType, static_cast<std::uint64_t>(kFloat));
          changed = true;
        } else {
          tensor.field(e);
        }
        return {};
      });
      if (!fields) return fields;
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

// Le nœud en FLOAT : ses tenseurs d'attribut (Constant, ConstantOfShape…) convertis, et le type
// cible FLOAT16 d'un Cast (to) ou d'un générateur (dtype : RandomNormal, EyeLike…) du domaine
// ONNX, FLOAT.
std::expected<void, Refusal> copy_node_f32(Bytes node, Writer& out, bool& changed) {
  std::string op_type;
  std::string domain;
  auto named = for_each_field(node, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kNodeOpType && field.number != kNodeDomain) return {};
    auto text = proto::as_string(field);
    if (!text) return std::unexpected(text.error());
    (field.number == kNodeOpType ? op_type : domain) = std::move(*text);
    return {};
  });
  if (!named) return named;
  const bool onnx_domain = domain.empty() || domain == "ai.onnx";
  Writer w;
  auto ok = for_each_field(node, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kNodeAttribute || field.wire != proto::WireType::Length) {
      w.field(field);
      return {};
    }
    std::string name;
    auto scanned = for_each_field(field.bytes, [&](const Field& a) -> std::expected<void, Refusal> {
      if (a.number == kAttributeGraph || a.number == kAttributeGraphs)
        return unavailable(std::format("{} : sous-graphe, non converti en FLOAT", op_type));
      if (a.number == kAttributeSparseTensor || a.number == kAttributeSparseTensors)
        return unavailable(std::format("{} : tenseur creux, non converti en FLOAT", op_type));
      if (a.number != kAttributeName) return {};
      auto text = proto::as_string(a);
      if (!text) return std::unexpected(text.error());
      name = std::move(*text);
      return {};
    });
    if (!scanned) return scanned;
    const bool type_target = onnx_domain && ((op_type == "Cast" && name == "to") || name == "dtype");
    Writer attr;
    auto copied = for_each_field(field.bytes, [&](const Field& a) -> std::expected<void, Refusal> {
      if ((a.number == kAttributeTensor || a.number == kAttributeTensors) &&
          a.wire == proto::WireType::Length)
        return copy_tensor_f32(a.bytes, a.number, attr, changed);
      if (type_target && a.number == kAttributeInt && a.wire == proto::WireType::Varint &&
          a.scalar == static_cast<std::uint64_t>(kFloat16)) {
        attr.varint(kAttributeInt, static_cast<std::uint64_t>(kFloat));
        changed = true;
        return {};
      }
      attr.field(a);
      return {};
    });
    if (!copied) return copied;
    w.message(kNodeAttribute, attr);
    return {};
  });
  if (!ok) return ok;
  out.message(kGraphNode, w);
  return {};
}

std::expected<void, Refusal> copy_graph_f32(Bytes graph, Writer& out, bool& changed) {
  Writer w;
  auto ok = for_each_field(graph, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.wire != proto::WireType::Length) {
      w.field(field);
      return {};
    }
    switch (field.number) {
      case kGraphNode:
        return copy_node_f32(field.bytes, w, changed);
      case kGraphInitializer:
        return copy_tensor_f32(field.bytes, kGraphInitializer, w, changed);
      case kGraphSparseInitializer:
        return unavailable("poids creux, non convertis en FLOAT");
      case kGraphInput:
      case kGraphOutput:
      case kGraphValueInfo:
        return copy_value_f32(field.bytes, field.number, w, changed);
      default:
        w.field(field);
        return {};
    }
  });
  if (!ok) return ok;
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

std::expected<std::vector<std::byte>, Refusal> with_input_dims(std::span<const std::byte> model,
                                                               std::span<const InputDims> inputs) {
  auto decoded = read_onnx(model);
  if (!decoded) return std::unexpected(decoded.error());
  const auto mismatch = [](std::string detail) {
    return std::unexpected(Refusal{RefusalCode::InputMismatch, std::move(detail)});
  };
  bool changes = false;
  for (const InputDims& wanted : inputs) {
    const auto& graph_inputs = decoded->graph.inputs;
    const auto at = std::ranges::find(graph_inputs, wanted.name, &ValueInfo::name);
    if (at == graph_inputs.end()) return mismatch(std::format("{} : pas une entrée du graphe", wanted.name));
    if (std::ranges::any_of(wanted.dims, [](std::int64_t d) { return d <= 0; }))
      return mismatch(std::format("{} : forme voulue avec une dimension libre", wanted.name));
    if (!at->has_shape) {
      changes = true;
      continue;
    }
    if (at->dims.size() != wanted.dims.size())
      return mismatch(
          std::format("{} : rang {} au lieu de {}", wanted.name, at->dims.size(), wanted.dims.size()));
    for (std::size_t k = 0; k < at->dims.size(); ++k) {
      if (!at->symbols[k].empty()) {
        changes = true;
      } else if (at->dims[k] != wanted.dims[k]) {
        return mismatch(
            std::format("{} : dimension {} fixée à {}, pas {}", wanted.name, k, at->dims[k], wanted.dims[k]));
      }
    }
  }
  if (!changes) return std::vector<std::byte>{};
  Writer out;
  auto ok = for_each_field(model, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kModelGraph || field.wire != proto::WireType::Length) {
      out.field(field);
      return {};
    }
    Writer graph;
    auto copied = for_each_field(field.bytes, [&](const Field& g) -> std::expected<void, Refusal> {
      if (g.number != kGraphInput || g.wire != proto::WireType::Length) {
        graph.field(g);
        return {};
      }
      auto name = name_of(g.bytes);
      if (!name) return std::unexpected(name.error());
      const auto at = std::ranges::find(inputs, std::string_view(*name), &InputDims::name);
      if (at == inputs.end()) {
        graph.field(g);
        return {};
      }
      return copy_with_dims(g.bytes, at->dims, graph);
    });
    if (!copied) return copied;
    out.message(kModelGraph, graph);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return std::move(out).take();
}

std::expected<std::vector<std::byte>, Refusal> with_metadata(std::span<const std::byte> model,
                                                             std::string_view key, std::string_view value) {
  // La clé et la valeur d'une entrée de métadonnées ; les autres champs ignorés.
  const auto entry_of = [](Bytes bytes) -> std::expected<std::pair<std::string, std::string>, Refusal> {
    std::pair<std::string, std::string> entry;
    proto::Reader reader(bytes);
    while (true) {
      auto field = reader.next();
      if (!field) return std::unexpected(field.error());  // entrée tronquée : refusée, pas lue en partie
      if (!*field) break;
      if ((*field)->number != kEntryKey && (*field)->number != kEntryValue) continue;
      auto text = proto::as_string(**field);
      if (!text) return std::unexpected(text.error());
      ((*field)->number == kEntryKey ? entry.first : entry.second) = std::move(*text);
    }
    return entry;
  };
  // Une seule entrée de cette clé, à la bonne valeur : le modèle est déjà tel qu'on le veut. Des
  // doublons sont tous retirés, la seule entrée voulue est réécrite à la fin.
  int matching = 0;
  bool present = false;
  Writer out;
  auto ok = for_each_field(model, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == kModelMetadata && field.wire == proto::WireType::Length) {
      auto entry = entry_of(field.bytes);
      if (!entry) return std::unexpected(entry.error());
      if (entry->first == key) {
        ++matching;
        present = entry->second == value;
        return {};  // remplacée à la fin
      }
    }
    out.field(field);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  if (present && matching == 1) return std::vector<std::byte>{};
  Writer entry;
  entry.string(kEntryKey, key);
  entry.string(kEntryValue, value);
  out.message(kModelMetadata, entry);
  return std::move(out).take();
}

std::expected<std::vector<std::byte>, Refusal> with_float32(std::span<const std::byte> model) {
  auto decoded = read_onnx(model);
  if (!decoded) return std::unexpected(decoded.error());
  bool changed = false;
  Writer out;
  auto ok = for_each_field(model, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.wire == proto::WireType::Length && field.number == kModelGraph)
      return copy_graph_f32(field.bytes, out, changed);
    if (field.wire == proto::WireType::Length && field.number == kModelFunctions)
      return unavailable("fonctions locales, non converties en FLOAT");
    out.field(field);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  if (!changed) return std::vector<std::byte>{};
  return std::move(out).take();
}

}  // namespace ajp::gpu::ir::onnx
