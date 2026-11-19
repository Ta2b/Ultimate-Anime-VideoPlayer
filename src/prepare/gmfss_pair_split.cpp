#include "prepare/gmfss_pair_split.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "prepare/derived_files.h"
#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/sha256.h"

namespace kirari::prepare {
namespace {

constexpr std::string_view kHeader = "kirari-gmfss-pair";
constexpr std::size_t kPieces = 2;
constexpr std::string_view kMetricWeight = "gmfss.metricnet.metric_in.weight";

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFSS natif : coupe de la paire : " + std::move(detail)});
}

// Un nom que le manifeste écrit sur une ligne « clé nom » : ni vide, ni blanc, ni fin de ligne.
bool name_writable(std::string_view name) {
  return !name.empty() &&
         std::ranges::none_of(name, [](char c) { return c == ' ' || c == '\n' || c == '\r'; });
}

// Le tenseur d'où vient name, en remontant les QuantizeLinear et DequantizeLinear (paire FP8).
std::string before_qdq(const std::map<std::string_view, const OnnxFullNode*>& producers, std::string name) {
  for (int hops = 0; hops < 8; ++hops) {
    const auto found = producers.find(name);
    if (found == producers.end()) break;
    const OnnxFullNode& node = *found->second;
    if ((node.op_type != "QuantizeLinear" && node.op_type != "DequantizeLinear") || node.inputs.empty())
      break;
    name = node.inputs[0];
  }
  return name;
}

std::string motion_name(int index) {
  return std::format("motion{}_{}", index / 3, index % 3 + 1);
}

// Les noyaux prennent une fusion dont les deux côtés sont des multiples de 32 (comme gmfss_fused.cpp), et au
// plus 4K.
bool size_valid(int height, int width) {
  return height >= 32 && width >= 32 && height % 32 == 0 && width % 32 == 0 && height <= 2176 &&
         width <= 3840;
}

std::optional<std::string> read_size(std::string_view rest, GmfssPairManifest& manifest) {
  const std::size_t gap = rest.find(' ');
  const std::string_view hs = rest.substr(0, gap);
  const std::string_view ws = gap == std::string_view::npos ? std::string_view() : rest.substr(gap + 1);
  const auto h = std::from_chars(hs.data(), hs.data() + hs.size(), manifest.height);
  const auto w = std::from_chars(ws.data(), ws.data() + ws.size(), manifest.width);
  if (h.ec != std::errc{} || w.ec != std::errc{} || h.ptr != hs.data() + hs.size() ||
      w.ptr != ws.data() + ws.size())
    return "taille illisible";
  return std::nullopt;
}

struct Seen {
  bool header = false;
  bool size = false;
  std::size_t flows = 0;
};

// Une ligne « clé reste » du manifeste ; rien : lue.
std::optional<std::string> read_field(std::string_view key, std::string_view rest,
                                      GmfssPairManifest& manifest, Seen& seen) {
  if (key == kHeader) {
    if (rest != "1") return "version " + std::string(rest);
    seen.header = true;
  } else if (key == "size") {
    if (auto problem = read_size(rest, manifest)) return problem;
    seen.size = true;
  } else if (key == "piece") {
    if (!is_piece_file_name(rest)) return "morceau « " + std::string(rest) + " »";
    manifest.pieces.emplace_back(rest);
  } else if (key == "sizes") {
    auto sizes = parse_sizes(rest);
    if (!sizes) return "tailles illisibles";
    manifest.sizes = std::move(*sizes);
  } else if (key == "digests") {
    auto digests = parse_digests(rest);
    if (!digests) return "empreintes illisibles";
    manifest.digests = std::move(*digests);
  } else if (key == "metric" && name_writable(rest)) {
    manifest.cut.metric_input = rest;
  } else if (key == "flow" && name_writable(rest)) {
    if (seen.flows >= manifest.cut.flows.size()) return "flots en trop";
    manifest.cut.flows[seen.flows++] = rest;
  } else {
    return "champ « " + std::string(key) + " » inconnu";
  }
  return std::nullopt;
}

}  // namespace

std::expected<GmfssPairCut, Refusal> gmfss_pair_cut(std::span<const std::byte> pair) {
  auto graph = read_onnx_graph(pair);
  if (!graph) return std::unexpected(std::move(graph.error()));
  std::map<std::string_view, const OnnxFullNode*> producers;
  for (const OnnxFullNode& node : graph->nodes) {
    for (const std::string& output : node.outputs) producers.emplace(output, &node);
  }
  GmfssPairCut cut;
  for (const OnnxFullNode& node : graph->nodes) {
    if (node.op_type != "Conv" || node.inputs.size() < 2) continue;
    if (before_qdq(producers, node.inputs[1]) == kMetricWeight)
      cut.metric_input = before_qdq(producers, node.inputs[0]);
  }
  if (cut.metric_input.empty()) return missing("pas de convolution metric_in");
  for (int i = 0; i < kGmfssPairFlows; ++i) {
    const std::string motion = motion_name(i);
    const auto found = producers.find(motion);
    if (found == producers.end() || found->second->op_type != "Concat" || found->second->inputs.size() != 2)
      return missing(motion + " : un Concat du flot et de la métrique attendu");
    cut.flows[static_cast<std::size_t>(i)] = before_qdq(producers, found->second->inputs[0]);
  }
  if (!name_writable(cut.metric_input) || !std::ranges::all_of(cut.flows, name_writable))
    return missing("un nom de tenseur à blanc");
  return cut;
}

std::expected<std::vector<std::byte>, Refusal> gmfss_flow_piece(std::span<const std::byte> pair,
                                                                const GmfssPairCut& cut, int height,
                                                                int width) {
  const std::vector<SplitTensor> inputs{{.name = "frames", .dims = {1, 6, 2 * height, 2 * width}}};
  std::vector<SplitTensor> outputs{{.name = cut.metric_input}};
  for (const std::string& flow : cut.flows) outputs.push_back({.name = flow});
  outputs.push_back({.name = "half0"});
  outputs.push_back({.name = "half1"});
  return extract_subgraph(pair, inputs, outputs, "gmfss_flow");
}

std::string format_gmfss_pair_manifest(const GmfssPairManifest& manifest) {
  std::string text = std::format("{} 1\n", kHeader);
  text += std::format("size {} {}\n", manifest.height, manifest.width);
  for (const std::string& piece : manifest.pieces) text += std::format("piece {}\n", piece);
  text += format_sizes(manifest.sizes);
  text += format_digests(manifest.digests);
  text += std::format("metric {}\n", manifest.cut.metric_input);
  for (const std::string& flow : manifest.cut.flows) text += std::format("flow {}\n", flow);
  return text;
}

std::expected<GmfssPairManifest, Refusal> parse_gmfss_pair_manifest(std::string_view text) {
  const auto bad = [](std::string detail) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "GMFSS natif : manifeste de la paire : " + std::move(detail)});
  };
  GmfssPairManifest manifest;
  Seen seen;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    const std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
    if (line.empty()) continue;
    const std::size_t space = line.find(' ');
    const std::string_view rest =
        space == std::string_view::npos ? std::string_view() : line.substr(space + 1);
    if (auto problem = read_field(line.substr(0, space), rest, manifest, seen))
      return bad(std::move(*problem));
  }
  if (!seen.header || !seen.size) return bad("en-tête ou taille absents");
  if (!size_valid(manifest.height, manifest.width)) return bad("taille que les noyaux ne prennent pas");
  if (manifest.pieces.size() != kPieces) return bad("la paire des poids et le flot attendus");
  if (manifest.sizes.size() != kPieces || manifest.digests.size() != kPieces)
    return bad("une taille et une empreinte par morceau attendues");
  if (manifest.cut.metric_input.empty() || seen.flows != manifest.cut.flows.size())
    return bad("l'entrée du MetricNet et six flots attendus");
  return manifest;
}

bool is_gmfss_pair_manifest(const std::filesystem::path& path) {
  return path.extension() == kGmfssPairExtension;
}

std::expected<std::filesystem::path, Refusal> derive_gmfss_pair(const std::filesystem::path& pair,
                                                                const std::filesystem::path& weights,
                                                                int height, int width) {
  if (!size_valid(height, width))
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("GMFSS natif : fusion de {}x{}, des multiples de 32 attendus", width, height));
  if (weights.parent_path() != pair.parent_path())
    return refuse(RefusalCode::ShapeInvalid, "GMFSS natif : la paire des poids hors du dossier de la paire");
  auto sha = sha256_file_known(pair);
  if (!sha) return std::unexpected(std::move(sha.error()));
  auto weights_sha = weights == pair ? sha : sha256_file_known(weights);
  if (!weights_sha) return std::unexpected(std::move(weights_sha.error()));
  const std::filesystem::path dir = pair.parent_path();
  const std::string stem = std::format("{}-{}-{}x{}-{}", core::to_utf8(pair.stem()), kGmfssPairDerivation,
                                       height, width, sha->substr(0, 16));
  const std::filesystem::path manifest_path =
      dir / core::path_from_utf8(std::format("{}{}", stem, kGmfssPairExtension));
  const std::string weights_name = core::to_utf8(weights.filename());
  // Déjà dérivé : le manifeste est écrit en dernier, sa présence dit que le flot y est.
  if (std::ifstream known(manifest_path); known) {
    const std::string text{std::istreambuf_iterator<char>(known), std::istreambuf_iterator<char>()};
    if (const auto parsed = parse_gmfss_pair_manifest(text);
        parsed && parsed->height == height && parsed->width == width &&
        parsed->pieces.front() == weights_name && pieces_intact(dir, *parsed))
      return manifest_path;
  }
  auto bytes = read_onnx_bytes(pair);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto cut = gmfss_pair_cut(*bytes);
  if (!cut) return std::unexpected(std::move(cut.error()));
  auto flow = gmfss_flow_piece(*bytes, *cut, height, width);
  if (!flow) return std::unexpected(std::move(flow.error()));
  const std::string file = std::format("{}-flow.onnx", stem);
  const std::string digest = sha256_hex(byte_text(*flow));
  if (!file_matches(dir / core::path_from_utf8(file), flow->size(), digest)) {
    if (auto written = core::write_atomic(dir / core::path_from_utf8(file), *flow); !written)
      return std::unexpected(std::move(written.error()));
  }
  std::error_code error;
  const auto weights_size = std::filesystem::file_size(weights, error);
  if (error) return refuse(RefusalCode::ModelUnreadable, "GMFSS natif : taille de " + weights_name);
  const GmfssPairManifest manifest{.height = height,
                                   .width = width,
                                   .pieces = {weights_name, file},
                                   .sizes = {weights_size, flow->size()},
                                   .digests = {*weights_sha, digest},
                                   .cut = std::move(*cut)};
  const std::string text = format_gmfss_pair_manifest(manifest);
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  if (auto written = core::write_atomic(manifest_path, std::span(raw, text.size())); !written)
    return std::unexpected(std::move(written.error()));
  return manifest_path;
}

}  // namespace kirari::prepare
