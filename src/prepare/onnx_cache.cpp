// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1 ; NodeProto.input
// = 1 (les autres : prepare/onnx_build.h).
#include "prepare/onnx_cache.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>

#include "prepare/onnx_build.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/protobuf_fields.h"

namespace kirari::prepare {
namespace {

using onnx_build::as_text;
using onnx_build::int_attribute;
using onnx_build::kNodeAttribute;
using onnx_build::kNodeInput;
using onnx_build::Message;
using onnx_build::node;
using onnx_build::pool_node;
using protobuf::Bytes;
using protobuf::Fields;
using protobuf::RawField;

constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kGraphNode = 1;

// Les noms que la découpe ajoute au graphe.
constexpr std::string_view kPooled = "kirari_cache_pooled";
constexpr std::string_view kStacked = "kirari_cache_stacked";
constexpr std::string_view kEncodedOut = "kirari_cache_features";

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

// DepthToSpace ou SpaceToDepth de bloc 2 (ordre DCR, l'un défait l'autre).
Message block_node(std::string_view op, std::string_view name, std::string_view input,
                   std::string_view output) {
  Message n = node(op, name, std::array{input}, output);
  n.message(kNodeAttribute, int_attribute("blocksize", 2));
  return n;
}

// Le nœud recopié, ses entrées nommées from lues sur to ; found compte les remplacements.
std::expected<Message, Refusal> rewired(Bytes node_bytes, std::string_view from, std::string_view to,
                                        int& found) {
  Message out;
  Fields fields(node_bytes);
  RawField part;
  while (fields.next(part)) {
    if (part.number == kNodeInput && part.delimited && as_text(part.payload) == from) {
      out.text(kNodeInput, to);
      ++found;
    } else {
      out.raw(part.raw);
    }
  }
  if (!fields.ok()) return malformed("ONNX : nœud malformé");
  return out;
}

// Le modèle où head précède les nœuds du graphe, dont les entrées from lisent to, et où tail les suit. Le
// reste est recopié ; les nœuds devenus inutiles restent, extract_subgraph les retire.
std::expected<std::vector<std::byte>, Refusal> spliced(Bytes model, std::span<const Message> head,
                                                       std::string_view from, std::string_view to,
                                                       std::span<const Message> tail) {
  auto graph = onnx_graph_field(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  Message new_graph;
  for (const Message& n : head) new_graph.message(kGraphNode, n);
  int found = 0;
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    if (field.number == kGraphNode && field.delimited) {
      auto n = rewired(field.payload, from, to, found);
      if (!n) return std::unexpected(std::move(n.error()));
      new_graph.message(kGraphNode, *n);
    } else {
      new_graph.raw(field.raw);
    }
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");
  if (found == 0)
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, std::format("ONNX : aucun nœud ne lit « {} »", from)});
  for (const Message& n : tail) new_graph.message(kGraphNode, n);
  std::vector<std::byte> out;
  out.reserve(model.size() + 1024);
  Fields model_fields(model);
  while (model_fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) {
      protobuf::put_delimited(out, kModelGraph, new_graph.data());
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  if (!model_fields.ok()) return malformed("ONNX : protobuf du modèle malformé");
  return out;
}

std::expected<void, Refusal> check_cache(const EncoderCache& cache) {
  if (cache.frames < 1 || cache.channels < 1 || (cache.half && cache.channels % 4 != 0)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("ONNX : cache de {} images à {} canaux{}", cache.frames,
                                               cache.channels, cache.half ? " en demi-taille" : "")});
  }
  return {};
}

// Le type des entrées du modèle (celui de la première) : les traits gardés ont le même.
std::expected<std::int32_t, Refusal> input_type(const OnnxSummary& summary) {
  if (summary.inputs.empty())
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "ONNX : sans entrée"});
  return summary.inputs.front().elem_type;
}

}  // namespace

std::string feature_input_name(std::string_view features, int k) {
  return std::format("{}_{}", features, k);
}

int feature_channels(const EncoderCache& cache) {
  return cache.half ? cache.channels / 4 : cache.channels;
}

std::expected<std::vector<std::byte>, Refusal> cached_encoder(std::span<const std::byte> model,
                                                              const EncoderCache& cache) {
  if (auto ok = check_cache(cache); !ok) return std::unexpected(std::move(ok.error()));
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  auto type = input_type(*summary);
  if (!type) return std::unexpected(std::move(type.error()));
  std::vector<Message> head;
  std::vector<Message> tail;
  if (cache.half) {
    head.push_back(pool_node("kirari_cache_pool", cache.input, kPooled));
    tail.push_back(block_node("DepthToSpace", "kirari_cache_depth", cache.encoded, kEncodedOut));
  }
  // L'entrée du modèle prend le nom cache.input, déjà celui de l'entrée empilée d'origine : les nœuds qui la
  // lisaient ne mènent pas aux traits et partent avec le reste du réseau.
  auto joined = spliced(model, head, cache.source, cache.half ? kPooled : cache.input, tail);
  if (!joined) return std::unexpected(std::move(joined.error()));
  const std::vector<SplitTensor> inputs{
      SplitTensor{.name = std::string(cache.input), .elem_type = *type, .dims = {1, 3, -1, -1}}};
  const std::vector<SplitTensor> outputs{SplitTensor{
      .name = std::string(cache.half ? kEncodedOut : cache.encoded), .elem_type = *type, .dims = {}}};
  return extract_subgraph(*joined, inputs, outputs, "temporal_encoder");
}

std::expected<std::vector<std::byte>, Refusal> with_cached_encoder(std::span<const std::byte> model,
                                                                   const EncoderCache& cache) {
  if (auto ok = check_cache(cache); !ok) return std::unexpected(std::move(ok.error()));
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  auto type = input_type(*summary);
  if (!type) return std::unexpected(std::move(type.error()));
  auto graph = read_onnx_graph(model);
  if (!graph) return std::unexpected(std::move(graph.error()));

  std::vector<SplitTensor> inputs;
  for (const OnnxInput& each : summary->inputs)
    inputs.push_back(SplitTensor{.name = each.name, .elem_type = each.elem_type, .dims = each.dims});
  std::vector<Message> head;
  std::vector<std::string> stacked;
  for (int k = 0; k < cache.frames; ++k) {
    const std::string name = feature_input_name(cache.features, k);
    if (std::ranges::contains(summary->inputs, name, &OnnxInput::name))
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                     std::format("ONNX : une entrée s'appelle déjà « {} »", name)});
    inputs.push_back(
        SplitTensor{.name = name, .elem_type = *type, .dims = {1, feature_channels(cache), -1, -1}});
    if (cache.half) {
      stacked.push_back(std::format("kirari_cache_space_{}", k));
      head.push_back(block_node("SpaceToDepth", stacked.back(), name, stacked.back()));
    } else {
      stacked.push_back(name);
    }
  }
  Message concat;
  for (const std::string& name : stacked) concat.text(kNodeInput, name);
  concat.text(onnx_build::kNodeOutput, kStacked)
      .text(onnx_build::kNodeName, "kirari_cache_concat")
      .text(onnx_build::kNodeOpType, "Concat")
      .message(kNodeAttribute, int_attribute("axis", 0));
  head.push_back(std::move(concat));
  auto joined = spliced(model, head, cache.encoded, kStacked, {});
  if (!joined) return std::unexpected(std::move(joined.error()));
  std::vector<SplitTensor> outputs;
  for (const std::string& name : graph->outputs)
    outputs.push_back(SplitTensor{.name = name, .elem_type = *type, .dims = {}});
  return extract_subgraph(*joined, inputs, outputs, "temporal_window");
}

}  // namespace kirari::prepare
