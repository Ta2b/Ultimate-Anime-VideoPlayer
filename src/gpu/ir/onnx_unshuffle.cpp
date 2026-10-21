#include "gpu/ir/onnx_unshuffle.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
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
constexpr std::uint32_t kGraphInitializer = 5;
constexpr std::uint32_t kGraphValueInfo = 13;
constexpr std::uint32_t kNodeInput = 1;
constexpr std::uint32_t kNodeOutput = 2;
constexpr std::uint32_t kNodeName = 3;
constexpr std::uint32_t kNodeOpType = 4;
constexpr std::uint32_t kNodeAttribute = 5;
constexpr std::uint32_t kAttributeName = 1;
constexpr std::uint32_t kAttributeInts = 8;
constexpr std::uint32_t kAttributeType = 20;
constexpr std::uint64_t kAttributeTypeInts = 7;
constexpr std::uint32_t kTensorDims = 1;
constexpr std::uint32_t kTensorDataType = 2;
constexpr std::uint32_t kTensorInt64Data = 7;
constexpr std::uint32_t kTensorName = 8;
constexpr std::uint32_t kValueName = 1;
// Slice prend ses bornes en entrées depuis l'opset 10 ; les poids hors des entrées du graphe
// depuis l'IR 4.
constexpr std::int64_t kSliceOpset = 10;
constexpr std::int64_t kIrInitializers = 4;

// pixel_unshuffle : [N, C, H/r, r, W/r, r] → [N, C, r, r, H/r, W/r] ; sans N, la même chose.
constexpr std::array<std::int64_t, 6> kUnshufflePerm{0, 1, 3, 5, 2, 4};
constexpr std::array<std::int64_t, 5> kUnbatchedPerm{0, 2, 4, 1, 3};
constexpr std::array<std::int64_t, 1> kStarts{1};
constexpr std::array<std::int64_t, 1> kEnds{6};

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

// Le champ name (numéro number) d'un message.
std::expected<std::string, Refusal> name_of(Bytes message, std::uint32_t number) {
  std::string name;
  auto ok = for_each_field(message, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != number) return {};
    auto value = proto::as_string(field);
    if (!value) return std::unexpected(value.error());
    name = std::move(*value);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return name;
}

bool default_domain(const Node& node) {
  return node.domain.empty() || node.domain == "ai.onnx";
}

bool is_op(const Node& node, std::string_view op_type) {
  return node.op_type == op_type && default_domain(node);
}

std::span<const std::int64_t> perm_of(const Node& node) {
  for (const Attribute& a : node.attributes) {
    if (a.name == "perm") return a.ints;
  }
  return {};
}

// Un dépliage à réécrire : les rangs du Reshape et du Transpose parmi les nœuds, la forme 6-D
// que lit le Reshape et la forme 5-D, sortie du Slice ajouté devant lui.
struct Site {
  std::size_t reshape = 0;
  std::size_t transpose = 0;
  std::string shape;
  std::string unbatched;
};

struct Plan {
  std::vector<Site> sites;
  std::string starts;  // poids du Slice, partagés par tous les dépliages
  std::string ends;
  std::unordered_set<std::string> stale;  // tenseurs changés de rang : leurs value_info partent
};

Plan plan_of(const Graph& g) {
  std::unordered_map<std::string, std::size_t> producer;
  std::unordered_map<std::string, std::vector<std::size_t>> readers;
  std::unordered_set<std::string> taken;  // tous les noms du graphe
  for (const Tensor& w : g.initializers) taken.insert(w.name);
  for (const ValueInfo& v : g.inputs) taken.insert(v.name);
  std::unordered_set<std::string> outputs;
  for (const ValueInfo& v : g.outputs) {
    outputs.insert(v.name);
    taken.insert(v.name);
  }
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    for (const std::string& in : g.nodes[i].inputs) {
      if (in.empty()) continue;
      readers[in].push_back(i);
      taken.insert(in);
    }
    for (const std::string& out : g.nodes[i].outputs) {
      producer.emplace(out, i);
      taken.insert(out);
    }
  }
  const auto unique = [&](const std::string& base) {
    std::string name = base;
    for (int k = 2; taken.contains(name); ++k) name = std::format("{}{}", base, k);
    taken.insert(name);
    return name;
  };
  Plan plan;
  for (std::size_t t = 0; t < g.nodes.size(); ++t) {
    const Node& transpose = g.nodes[t];
    if (!is_op(transpose, "Transpose") || transpose.inputs.size() != 1 || transpose.outputs.size() != 1 ||
        !std::ranges::equal(perm_of(transpose), kUnshufflePerm))
      continue;
    const auto made = producer.find(transpose.inputs[0]);
    if (made == producer.end()) continue;
    const Node& reshape = g.nodes[made->second];
    if (!is_op(reshape, "Reshape") || reshape.inputs.size() != 2 || reshape.inputs[1].empty() ||
        reshape.outputs.size() != 1)
      continue;
    const std::string& folded = reshape.outputs[0];
    if (readers[folded].size() != 1 || outputs.contains(folded)) continue;
    const std::string& permuted = transpose.outputs[0];
    const std::vector<std::size_t>& next = readers[permuted];
    const bool reshaped = std::ranges::all_of(next, [&](std::size_t r) {
      const Node& n = g.nodes[r];
      return is_op(n, "Reshape") && n.inputs.size() == 2 && n.inputs[0] == permuted &&
             n.inputs[1] != permuted;
    });
    if (next.empty() || !reshaped || outputs.contains(permuted)) continue;
    if (plan.sites.empty()) {
      plan.starts = unique("ajp.unbatch.starts");
      plan.ends = unique("ajp.unbatch.ends");
    }
    plan.sites.push_back({.reshape = made->second,
                          .transpose = t,
                          .shape = reshape.inputs[1],
                          .unbatched = unique(reshape.inputs[1] + ".5d")});
    plan.stale.insert(folded);
    plan.stale.insert(permuted);
  }
  return plan;
}

// Un poids INT64 à une dimension.
void write_ints(Writer& graph, std::string_view name, std::span<const std::int64_t> values) {
  Writer tensor;
  tensor.varint(kTensorDims, values.size());
  tensor.varint(kTensorDataType, static_cast<std::uint64_t>(kInt64));
  for (const std::int64_t v : values) tensor.varint(kTensorInt64Data, static_cast<std::uint64_t>(v));
  tensor.string(kTensorName, name);
  graph.message(kGraphInitializer, tensor);
}

// La forme 5-D : la forme 6-D sans sa première dimension.
void write_slice(Writer& graph, const Site& site, const Plan& plan) {
  Writer node;
  node.string(kNodeInput, site.shape);
  node.string(kNodeInput, plan.starts);
  node.string(kNodeInput, plan.ends);
  node.string(kNodeOutput, site.unbatched);
  node.string(kNodeName, std::format("ajp.unbatch.{}", site.unbatched));
  node.string(kNodeOpType, "Slice");
  graph.message(kGraphNode, node);
}

// Le Reshape, sa seconde entrée (la forme) remplacée par la forme 5-D.
std::expected<void, Refusal> copy_reshape(Bytes node, const Site& site, Writer& graph) {
  Writer w;
  int input = 0;
  auto ok = for_each_field(node, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == kNodeInput && field.wire == proto::WireType::Length && input++ == 1) {
      w.string(kNodeInput, site.unbatched);
    } else {
      w.field(field);
    }
    return {};
  });
  if (!ok) return ok;
  graph.message(kGraphNode, w);
  return {};
}

// Le Transpose, sa permutation remplacée par la permutation 5-D.
std::expected<void, Refusal> copy_transpose(Bytes node, Writer& graph) {
  Writer w;
  auto ok = for_each_field(node, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number != kNodeAttribute || field.wire != proto::WireType::Length) {
      w.field(field);
      return {};
    }
    auto name = name_of(field.bytes, kAttributeName);
    if (!name) return std::unexpected(name.error());
    if (*name != "perm") {
      w.field(field);
      return {};
    }
    Writer attr;
    attr.string(kAttributeName, "perm");
    for (const std::int64_t p : kUnbatchedPerm) attr.varint(kAttributeInts, static_cast<std::uint64_t>(p));
    attr.varint(kAttributeType, kAttributeTypeInts);
    w.message(kNodeAttribute, attr);
    return {};
  });
  if (!ok) return ok;
  graph.message(kGraphNode, w);
  return {};
}

std::expected<void, Refusal> copy_graph(Bytes graph, const Plan& plan, Writer& out) {
  std::unordered_map<std::size_t, const Site*> reshapes;
  std::unordered_set<std::size_t> transposes;
  for (const Site& site : plan.sites) {
    reshapes.emplace(site.reshape, &site);
    transposes.insert(site.transpose);
  }
  Writer w;
  std::size_t index = 0;  // rang du nœud, dans l'ordre du fichier comme read_onnx
  auto ok = for_each_field(graph, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.wire == proto::WireType::Length && field.number == kGraphNode) {
      const std::size_t at = index++;
      if (const auto r = reshapes.find(at); r != reshapes.end()) {
        write_slice(w, *r->second, plan);
        return copy_reshape(field.bytes, *r->second, w);
      }
      if (transposes.contains(at)) return copy_transpose(field.bytes, w);
    } else if (field.wire == proto::WireType::Length && field.number == kGraphValueInfo) {
      auto name = name_of(field.bytes, kValueName);
      if (!name) return std::unexpected(name.error());
      if (plan.stale.contains(*name)) return {};
    }
    w.field(field);
    return {};
  });
  if (!ok) return ok;
  write_ints(w, plan.starts, kStarts);
  write_ints(w, plan.ends, kEnds);
  out.message(kModelGraph, w);
  return {};
}

}  // namespace

std::expected<std::vector<std::byte>, Refusal> with_unshuffle_5d(std::span<const std::byte> model) {
  auto decoded = read_onnx(model);
  if (!decoded) return std::unexpected(decoded.error());
  const bool slices = std::ranges::any_of(decoded->opsets, [](const OpsetId& o) {
    return (o.domain.empty() || o.domain == "ai.onnx") && o.version >= kSliceOpset;
  });
  if (!slices || decoded->ir_version < kIrInitializers) return std::vector<std::byte>{};
  const Plan plan = plan_of(decoded->graph);
  if (plan.sites.empty()) return std::vector<std::byte>{};
  Writer out;
  auto ok = for_each_field(model, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == kModelGraph && field.wire == proto::WireType::Length)
      return copy_graph(field.bytes, plan, out);
    out.field(field);
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return std::move(out).take();
}

}  // namespace ajp::gpu::ir::onnx
