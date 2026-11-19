#include "prepare/tfix_split.h"

#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "kernels/spec/tfix_fused.h"
#include "prepare/derived_files.h"
#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/sha256.h"
#include "prepare/split_index.h"

namespace kirari::prepare {
namespace {

using split::Index;
using split::text_is;

std::unexpected<Refusal> unknown(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing,
              "TemporalFix : structure non reconnue pour la découpe : " + std::move(detail)});
}

std::string block_node(int b, std::string_view tail) {
  return std::format("/aligner/blocks.{}/{}", b, tail);
}

// La constante d'entrée input du nœud name vaut-elle values ?
bool constant_is(const Index& index, const std::string& name, std::size_t input,
                 std::span<const double> values) {
  const OnnxFullNode* node = index.named(name);
  if (node == nullptr || input >= node->inputs.size()) return false;
  const std::vector<double>& got = index.constant(node->inputs[input]);
  if (got.size() != values.size()) return false;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (std::abs(got[i] - values[i]) > 1e-9) return false;
  }
  return true;
}

// Ce que les noyaux refont et qui doit être tel qu'ils le refont : Resize linéaire demi-pixel sans
// anticrénelage, GridSample bilinéaire à zéros dehors, sans align_corners ; rien : bon.
std::optional<std::string> operator_problem(const OnnxGraph& graph) {
  int grids = 0;
  for (const OnnxFullNode& node : graph.nodes) {
    if (node.op_type == "Resize") {
      const OnnxAttribute* mode = node.attribute("coordinate_transformation_mode");
      const bool half_pixel = mode == nullptr || (mode->text && *mode->text == "half_pixel");
      if (!text_is(node, "mode", "linear") || !half_pixel || !node.resize_without_antialias())
        return std::format("{} : Resize autre que linéaire demi-pixel", node.name);
    } else if (node.op_type == "GridSample") {
      ++grids;
      const OnnxAttribute* align = node.attribute("align_corners");
      const OnnxAttribute* padding = node.attribute("padding_mode");
      const bool zeros = padding == nullptr || (padding->text && *padding->text == "zeros");
      if (!text_is(node, "mode", "bilinear") || !zeros ||
          (align != nullptr && align->integer.value_or(0) != 0))
        return std::format("{} : GridSample autre que bilinéaire à zéros", node.name);
    }
  }
  if (grids != 5) return std::format("{} GridSample au lieu de 5", grids);
  return std::nullopt;
}

// Les trois blocs du flot : leur Concat d'entrée et leur dernière convolution, notées dans split ; rien :
// bon.
std::optional<std::string> blocks_problem(const Index& index, TfixSplit& split) {
  for (int b = 0; b < kernels::kTfixLevels; ++b) {
    const OnnxFullNode* first = index.named(block_node(b, "conv0/conv0.0/conv0.0.0/Conv"), "Conv");
    const OnnxFullNode* last = index.named(block_node(b, "out_conv/Conv"), "Conv");
    if (first == nullptr || last == nullptr || first->inputs.empty() || last->outputs.size() != 1)
      return std::format("bloc {}", b);
    const OnnxFullNode* concat = index.producer(first->inputs[0]);
    // X0 : cinq morceaux (images, traits, temps) ; X1, X2 : les 24 canaux puis le flot.
    const std::size_t arity = b == 0 ? 5U : 2U;
    if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != arity)
      return std::format("bloc {} : Concat d'entrée", b);
    split.block_input.push_back(first->inputs[0]);
    split.block_output.push_back(last->outputs[0]);
  }
  if (index.named(block_node(3, "conv0/conv0.0/conv0.0.0/Conv")) != nullptr) return "plus de trois blocs";

  return std::nullopt;
}

// La tête de confiance, les temps des six appuis et les constantes du mélange que les noyaux écrivent en dur
// ; rien : bon.
std::optional<std::string> head_problem(const Index& index, const TfixSplit& split) {
  const OnnxFullNode* conf_in = index.named("/conf_head/net/net.0/net.0.0/Conv", "Conv");
  const OnnxFullNode* conf_out = index.named("/conf_head/net/net.2/Conv", "Conv");
  const OnnxFullNode* conf_concat = index.producer(split.conf_input);
  if (conf_in == nullptr || conf_out == nullptr || conf_in->inputs.empty() ||
      conf_in->inputs[0] != split.conf_input || conf_out->outputs.size() != 1 ||
      conf_out->outputs[0] != split.conf_output || conf_concat == nullptr ||
      conf_concat->op_type != "Concat" || conf_concat->inputs.size() != 9)
    return "tête de confiance";

  // Les temps des six appuis, et les constantes du mélange que les noyaux écrivent en dur.
  std::vector<double> times;
  for (const float t : kernels::kTfixTime) times.push_back(t);
  if (!constant_is(index, "/Expand", 0, times)) return "temps des appuis";
  const struct {
    const char* node;
    double value;
  } mix[] = {
      {"/Sub_6", 0.60009765625}, {"/Mul_16", 12.0}, {"/Add_7", 1.0}, {"/Add_8", 0.5}, {"/Mul_19", 4.0}};
  for (const auto& want : mix) {
    if (!constant_is(index, want.node, 1, std::span(&want.value, 1)))
      return std::format("{} : constante du mélange différente", want.node);
  }
  return std::nullopt;
}

}  // namespace

std::string_view role_name(TfixRole role) {
  return role == TfixRole::Encoder ? "encoder" : "window";
}

std::size_t tfix_piece_count(TfixRole role) {
  return role == TfixRole::Encoder ? 1U : 4U;
}

std::expected<TfixSplit, Refusal> analyze_tfix(std::span<const std::byte> model) {
  auto graph = read_onnx_graph(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  if (graph->has_external_data || graph->has_sparse_initializers || graph->has_subgraphs)
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   "TemporalFix : poids externes, creux ou graphes imbriqués non découpés"});
  const Index index(*graph);
  TfixSplit split;

  const OnnxFullNode* encode_in = index.named("/encode/cnn0/Conv", "Conv");
  const OnnxFullNode* encode_out = index.named("/encode/Resize", "Resize");
  if (encode_in == nullptr || encode_out == nullptr || encode_in->inputs.empty() ||
      encode_in->inputs[0] != split.encode_input || encode_out->outputs.size() != 1 ||
      encode_out->outputs[0] != split.encode_output)
    return unknown("encodeur");

  if (auto problem = blocks_problem(index, split)) return unknown(*problem);
  if (auto problem = head_problem(index, split)) return unknown(*problem);
  if (const auto problem = operator_problem(*graph)) return unknown(*problem);
  return split;
}

std::expected<std::vector<TfixPiece>, Refusal> split_tfix(std::span<const std::byte> model,
                                                          const TfixSplit& split, TfixRole role, int height,
                                                          int width) {
  if (!kernels::tfix_size_valid(height, width)) {
    return std::unexpected(Refusal{
        RefusalCode::ShapeInvalid,
        std::format("TemporalFix fusionné : taille {}x{} que la découpe ne prend pas", width, height)});
  }
  std::vector<TfixPiece> pieces;
  const auto cut = [&](std::string name, const std::string& input, std::vector<std::int64_t> dims,
                       const std::string& output) -> std::expected<void, Refusal> {
    const SplitTensor in{.name = input, .elem_type = 10, .dims = std::move(dims)};
    const SplitTensor out{.name = output, .elem_type = 10};
    auto piece = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), name);
    if (!piece) return std::unexpected(std::move(piece.error()));
    pieces.push_back({.name = std::move(name), .model = std::move(*piece)});
    return {};
  };
  if (role == TfixRole::Encoder) {
    if (auto ok =
            cut("encode", split.encode_input, {1, kernels::kTfixImage, height, width}, split.encode_output);
        !ok)
      return std::unexpected(std::move(ok.error()));
    return pieces;
  }
  for (int b = 0; b < kernels::kTfixLevels; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::TfixDims x = kernels::tfix_block_x(b, height, width);
    if (auto ok = cut(std::format("bloc{}", b), split.block_input[i],
                      {kernels::kTfixSupports, x.channels, x.height, x.width}, split.block_output[i]);
        !ok)
      return std::unexpected(std::move(ok.error()));
  }
  const kernels::TfixDims half = kernels::tfix_level(2, height, width);
  if (auto ok = cut("conf", split.conf_input,
                    {kernels::kTfixSupports, kernels::kTfixConf, half.height, half.width}, split.conf_output);
      !ok)
    return std::unexpected(std::move(ok.error()));
  return pieces;
}

std::string format_tfix_manifest(const TfixFusedManifest& manifest) {
  std::string text = "kirari-tfix-fused 1\n";
  text += std::format("role {}\n", role_name(manifest.role));
  text += std::format("size {} {}\n", manifest.height, manifest.width);
  text += std::format("half {}\n", manifest.half ? 1 : 0);
  for (const std::string& piece : manifest.pieces) text += std::format("piece {}\n", piece);
  text += format_sizes(manifest.sizes);
  text += format_digests(manifest.digests);
  return text;
}

namespace {

struct ManifestSeen {
  bool header = false;
  bool role = false;
  bool size = false;
  bool half = false;
};

std::optional<std::string> read_size(std::string_view rest, TfixFusedManifest& manifest) {
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

// Une ligne « clé reste » du manifeste ; rien : lue.
std::optional<std::string> read_field(std::string_view key, std::string_view rest,
                                      TfixFusedManifest& manifest, ManifestSeen& seen) {
  if (key == "kirari-tfix-fused") {
    if (rest != "1") return "version " + std::string(rest);
    seen.header = true;
  } else if (key == "role") {
    if (rest != "encoder" && rest != "window") return "rôle « " + std::string(rest) + " »";
    manifest.role = rest == "encoder" ? TfixRole::Encoder : TfixRole::Window;
    seen.role = true;
  } else if (key == "size") {
    if (auto problem = read_size(rest, manifest)) return problem;
    seen.size = true;
  } else if (key == "half") {
    if (rest != "0" && rest != "1") return "demi-taille illisible";
    manifest.half = rest == "1";
    seen.half = true;
  } else if (key == "piece") {
    if (rest == "." || !is_piece_file_name(rest)) return "morceau « " + std::string(rest) + " »";
    manifest.pieces.emplace_back(rest);
  } else if (key == "sizes") {
    auto sizes = parse_sizes(rest);
    if (!sizes) return "tailles illisibles";
    manifest.sizes = std::move(*sizes);
  } else if (key == "digests") {
    auto digests = parse_digests(rest);
    if (!digests) return "empreintes illisibles";
    manifest.digests = std::move(*digests);
  } else {
    return "champ « " + std::string(key) + " » inconnu";
  }
  return std::nullopt;
}

// Le manifeste entier, une fois lu ; rien : bon.
std::optional<std::string> manifest_problem(const TfixFusedManifest& manifest, const ManifestSeen& seen) {
  if (!seen.header || !seen.role || !seen.size || !seen.half)
    return "en-tête, rôle, taille ou demi-taille absents";
  if (!kernels::tfix_size_valid(manifest.height, manifest.width))
    return "taille que les noyaux ne prennent pas";
  if (manifest.pieces.size() != tfix_piece_count(manifest.role))
    return "nombre de morceaux différent du rôle";
  if (!manifest.sizes.empty() && manifest.sizes.size() != manifest.pieces.size())
    return "autant de tailles que de morceaux attendu";
  if (!manifest.digests.empty() && manifest.digests.size() != manifest.pieces.size())
    return "autant d'empreintes que de morceaux attendu";
  return std::nullopt;
}

}  // namespace

std::expected<TfixFusedManifest, Refusal> parse_tfix_manifest(std::string_view text) {
  const auto bad = [](std::string detail) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "TemporalFix fusionné : manifeste : " + std::move(detail)});
  };
  TfixFusedManifest manifest;
  ManifestSeen seen;
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
  if (auto problem = manifest_problem(manifest, seen)) return bad(std::move(*problem));
  return manifest;
}

bool is_tfix_fused_manifest(const std::filesystem::path& path) {
  return path.extension() == kTfixFusedExtension;
}

std::expected<std::filesystem::path, Refusal> derive_tfix_fused(const std::filesystem::path& onnx,
                                                                const std::filesystem::path& dir,
                                                                TfixRole role, int height, int width,
                                                                bool half) {
  auto sha = sha256_file_known(onnx);
  if (!sha) return std::unexpected(std::move(sha.error()));
  const std::string stem =
      std::format("{}-{}-{}{}-{}x{}-{}", core::to_utf8(onnx.stem()), kTfixFusedDerivation, role_name(role),
                  half ? "-half" : "", height, width, sha->substr(0, 16));
  std::filesystem::path manifest_path =
      dir / core::path_from_utf8(std::format("{}{}", stem, kTfixFusedExtension));
  // Déjà dérivé : le manifeste est écrit en dernier, sa présence dit que les morceaux y sont.
  if (std::ifstream known(manifest_path); known) {
    const std::string text{std::istreambuf_iterator<char>(known), std::istreambuf_iterator<char>()};
    if (const auto parsed = parse_tfix_manifest(text);
        parsed && parsed->role == role && parsed->height == height && parsed->width == width &&
        parsed->half == half && !parsed->sizes.empty() && pieces_intact(dir, *parsed))
      return manifest_path;
  }
  auto bytes = read_onnx_bytes(onnx);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto split = analyze_tfix(*bytes);
  if (!split) return std::unexpected(std::move(split.error()));
  auto pieces = split_tfix(*bytes, *split, role, height, width);
  if (!pieces) return std::unexpected(std::move(pieces.error()));
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  TfixFusedManifest manifest{.role = role, .height = height, .width = width, .half = half};
  for (const TfixPiece& piece : *pieces) {
    const std::string file = std::format("{}-{}.onnx", stem, piece.name);
    const std::string digest = sha256_hex(byte_text(piece.model));
    if (!file_matches(dir / core::path_from_utf8(file), piece.model.size(), digest)) {
      if (auto written = core::write_atomic(dir / core::path_from_utf8(file), piece.model); !written)
        return std::unexpected(std::move(written.error()));
    }
    manifest.pieces.push_back(file);
    manifest.sizes.push_back(piece.model.size());
    manifest.digests.push_back(digest);
  }
  const std::string text = format_tfix_manifest(manifest);
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  if (auto written = core::write_atomic(manifest_path, std::span(raw, text.size())); !written)
    return std::unexpected(std::move(written.error()));
  return manifest_path;
}

}  // namespace kirari::prepare
