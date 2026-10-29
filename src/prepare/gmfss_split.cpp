#include "prepare/gmfss_split.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>

#include "kernels/spec/gmfss.h"
#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/sha256.h"

namespace ajp::prepare {
namespace {

std::unexpected<Refusal> unknown(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                 "GMFSS : structure non reconnue pour la découpe : " + std::move(detail)});
}

// Le graphe indexé par nom de nœud et par tenseur produit.
class Index {
 public:
  explicit Index(const OnnxGraph& graph) : graph_(graph) {
    for (const OnnxFullNode& node : graph.nodes) {
      by_name_.emplace(node.name, &node);
      for (const std::string& output : node.outputs) producer_.emplace(output, &node);
    }
    for (const std::string& name : graph.initializers) initializers_.insert(name);
  }
  // Le nœud de ce nom s'il fait cet opérateur.
  const OnnxFullNode* named(const std::string& name, std::string_view op) const {
    const auto found = by_name_.find(name);
    return found != by_name_.end() && found->second->op_type == op ? found->second : nullptr;
  }
  const OnnxFullNode* producer(const std::string& tensor) const {
    const auto found = producer_.find(tensor);
    return found != producer_.end() ? found->second : nullptr;
  }
  // Les valeurs du tenseur constant, vides s'il n'en est pas un.
  const std::vector<double>& constant(const std::string& tensor) const {
    static const std::vector<double> none;
    const auto found = graph_.constants.find(tensor);
    return found != graph_.constants.end() ? found->second : none;
  }
  bool is_initializer(const std::string& tensor) const { return initializers_.contains(tensor); }
  const OnnxGraph& graph() const { return graph_; }

 private:
  const OnnxGraph& graph_;
  std::map<std::string, const OnnxFullNode*, std::less<>> by_name_;
  std::map<std::string, const OnnxFullNode*, std::less<>> producer_;
  std::set<std::string, std::less<>> initializers_;
};

bool same(const std::vector<double>& got, std::initializer_list<double> want) {
  if (got.size() != want.size()) return false;
  std::size_t i = 0;
  for (const double w : want) {
    if (std::abs(got[i++] - w) > 1e-6) return false;
  }
  return true;
}

// Un Slice de data, sur l'axe 1, de first (inclus) à last (exclu), pas 1 : les cinq entrées data, starts,
// ends, axes, steps.
bool is_plane_slice(const Index& index, const OnnxFullNode* node, const std::string& data, int first,
                    int last) {
  return node != nullptr && node->inputs.size() == 5 && node->outputs.size() == 1 &&
         node->inputs[0] == data && same(index.constant(node->inputs[1]), {static_cast<double>(first)}) &&
         same(index.constant(node->inputs[2]), {static_cast<double>(last)}) &&
         same(index.constant(node->inputs[3]), {1.0}) && same(index.constant(node->inputs[4]), {1.0});
}

// Les nœuds qui mènent de stop aux outputs, dans l'ordre du graphe, chacun avec ce qui le distingue d'un nœud
// d'un autre côté du réseau : opérateur, attributs, poids (par nom) et constantes (par valeur). Les tenseurs
// d'activation n'y sont pas : deux côtés qui font le même calcul avec les mêmes poids ont la même empreinte.
using Signature = std::tuple<std::string, std::vector<std::pair<std::string, std::int64_t>>,
                             std::vector<std::string>, std::vector<std::vector<double>>>;

std::vector<Signature> signature(const Index& index, std::span<const std::string> outputs,
                                 const std::string& stop) {
  std::set<const OnnxFullNode*> kept;
  std::vector<std::string> pending(outputs.begin(), outputs.end());
  std::set<std::string, std::less<>> seen;
  while (!pending.empty()) {
    std::string name = std::move(pending.back());
    pending.pop_back();
    if (name.empty() || name == stop || !seen.insert(name).second) continue;
    const OnnxFullNode* node = index.producer(name);
    if (node == nullptr || !kept.insert(node).second) continue;
    for (const std::string& input : node->inputs) pending.push_back(input);
  }
  std::vector<Signature> out;
  for (const OnnxFullNode& node : index.graph().nodes) {
    if (!kept.contains(&node) || node.op_type == "Constant") continue;
    Signature item{node.op_type, {}, {}, {}};
    for (const OnnxAttribute& attribute : node.attributes) {
      std::get<1>(item).emplace_back(attribute.name + attribute.text.value_or(""),
                                     attribute.integer.value_or(0));
    }
    for (const std::string& input : node.inputs) {
      if (index.is_initializer(input))
        std::get<2>(item).push_back(input);
      else if (!index.constant(input).empty())
        std::get<3>(item).push_back(index.constant(input));
    }
    out.push_back(std::move(item));
  }
  return out;
}

}  // namespace

std::expected<void, Refusal> analyze_gmfss(std::span<const std::byte> pair) {
  const auto Float16 = static_cast<std::int32_t>(OnnxType::Float16);
  // La paire.
  auto pair_summary = scan_onnx(pair);
  if (!pair_summary) return std::unexpected(std::move(pair_summary.error()));
  if (pair_summary->inputs.size() != 1 || pair_summary->inputs[0].name != "frames" ||
      pair_summary->inputs[0].dims.size() != 4 ||
      pair_summary->inputs[0].dims[1] != kernels::kGmfssFramePlanes ||
      pair_summary->inputs[0].elem_type != Float16) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch,
                "GMFSS : entrée « frames » 1 × 6 × H × W en FP16 attendue pour la découpe"});
  }
  auto pair_graph = read_onnx_graph(pair);
  if (!pair_graph) return std::unexpected(std::move(pair_graph.error()));
  if (pair_graph->has_external_data || pair_graph->has_sparse_initializers || pair_graph->has_subgraphs)
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   "GMFSS : poids externes, initialiseurs creux ou graphes imbriqués"});
  const Index pairs(*pair_graph);
  // Les images a et b : deux Slice de « frames », les plans 0 à 2 et 3 à 5.
  const OnnxFullNode* slice_a = pairs.named("/Slice", "Slice");
  const OnnxFullNode* slice_b = pairs.named("/Slice_1", "Slice");
  if (!is_plane_slice(pairs, slice_a, "frames", 0, 3) || !is_plane_slice(pairs, slice_b, "frames", 3, 6) ||
      slice_a->outputs[0] != kGmfssFeatureInput || slice_b->outputs[0] != "/Slice_1_output_0")
    return unknown("images de « frames »");
  // FeatureNet : les trois sorties de chaque côté, calculées de l'image seule (version 2 des exports : les
  // 64 caractéristiques du niveau 1 en feature{k}_1), les deux côtés avec les mêmes poids.
  const std::vector<std::string> side0 = {"feature0_1", "source0_2", "source0_3"};
  const std::vector<std::string> side1 = {"feature1_1", "source1_2", "source1_3"};
  for (const std::string& name : side0) {
    if (pairs.producer(name) == nullptr) return unknown("« " + name + " » (export version 2 attendu)");
  }
  for (const std::string& name : side1) {
    if (pairs.producer(name) == nullptr) return unknown("« " + name + " » (export version 2 attendu)");
  }
  const std::vector<Signature> net_a = signature(pairs, side0, std::string(kGmfssFeatureInput));
  const std::vector<Signature> net_b = signature(pairs, side1, "/Slice_1_output_0");
  if (net_a.empty() || net_a != net_b) return unknown("FeatureNet : les deux côtés diffèrent");
  for (const char* name : {"half0", "half1"}) {
    const OnnxFullNode* resize = pairs.producer(name);
    if (resize == nullptr || resize->op_type != "Resize") return unknown(std::format("« {} »", name));
  }
  for (int side = 0; side < 2; ++side) {
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      if (pairs.producer(std::format("motion{}_{}", side, level)) == nullptr)
        return unknown(std::format("« motion{}_{} »", side, level));
    }
  }

  return {};
}

std::expected<GmfssPieces, Refusal> split_gmfss(std::span<const std::byte> pair, int height, int width) {
  if (height <= 0 || width <= 0 || height % 8 != 0 || width % 8 != 0) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeInvalid,
                std::format("GMFSS fusionné : taille {}x{} que la découpe ne prend pas", width, height)});
  }
  if (auto ok = analyze_gmfss(pair); !ok) return std::unexpected(std::move(ok.error()));
  constexpr std::int32_t kFloat16 = 10;
  GmfssPieces pieces;
  {
    const SplitTensor in{
        .name = std::string(kGmfssFeatureInput), .elem_type = kFloat16, .dims = {1, 3, height, width}};
    std::vector<SplitTensor> outs;
    outs.reserve(kGmfssFeatureOutputs.size());
    for (const std::string_view name : kGmfssFeatureOutputs)
      outs.push_back({.name = std::string(name), .elem_type = kFloat16});
    auto cut = extract_subgraph(pair, std::span(&in, 1), outs, "gmfss_features");
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.features = std::move(*cut);
  }
  {
    const SplitTensor in{
        .name = "frames", .elem_type = kFloat16, .dims = {1, kernels::kGmfssFramePlanes, height, width}};
    std::vector<SplitTensor> outs;
    for (int side = 0; side < 2; ++side) {
      for (int level = 1; level <= kernels::kGmfssLevels; ++level)
        outs.push_back({.name = std::format("motion{}_{}", side, level), .elem_type = kFloat16});
    }
    outs.push_back({.name = "half0", .elem_type = kFloat16});
    outs.push_back({.name = "half1", .elem_type = kFloat16});
    auto cut = extract_subgraph(pair, std::span(&in, 1), outs, "gmfss_flow");
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.flow = std::move(*cut);
  }
  return pieces;
}

namespace {

// Écrit les octets à côté de target puis les y renomme.
std::expected<void, Refusal> write_atomic(const std::filesystem::path& target,
                                          std::span<const std::byte> bytes) {
  std::filesystem::path temp = target;
  temp += std::format(
      ".{:x}.tmp", std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
                       static_cast<std::size_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::error_code error;
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out.flush()) {
      std::filesystem::remove(temp, error);
      return std::unexpected(Refusal{RefusalCode::FileUnwritable, temp.string()});
    }
  }
  std::filesystem::rename(temp, target, error);
  if (error) {
    std::filesystem::remove(temp, error);
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), error.message())});
  }
  return {};
}

bool exists_nonempty(const std::filesystem::path& file) {
  std::error_code error;
  const auto size = std::filesystem::file_size(file, error);
  return !error && size > 0;
}

}  // namespace

std::expected<GmfssFusedFiles, Refusal> derive_gmfss_fused(const std::filesystem::path& pair_onnx,
                                                           const std::filesystem::path& dir, int height,
                                                           int width) {
  auto sha = sha256_file_known(pair_onnx);
  if (!sha) return std::unexpected(std::move(sha.error()));
  const std::string stem = std::format("{}-{}-{}x{}-{}", pair_onnx.stem().string(), kGmfssFusedDerivation,
                                       height, width, sha->substr(0, 16));
  GmfssFusedFiles files{.features = dir / (stem + "-features.onnx"), .flow = dir / (stem + "-flow.onnx")};
  if (exists_nonempty(files.features) && exists_nonempty(files.flow)) return files;
  auto bytes = read_onnx_bytes(pair_onnx);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto pieces = split_gmfss(*bytes, height, width);
  if (!pieces) return std::unexpected(std::move(pieces.error()));
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  for (const auto& [file, content] :
       {std::pair{&files.features, &pieces->features}, std::pair{&files.flow, &pieces->flow}}) {
    if (exists_nonempty(*file)) continue;
    if (auto written = write_atomic(*file, *content); !written)
      return std::unexpected(std::move(written.error()));
  }
  return files;
}

}  // namespace ajp::prepare
