#include "prepare/rife_split.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "prepare/model_source.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/sha256.h"

namespace ajp::prepare {
namespace {

std::unexpected<Refusal> unknown(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                 "RIFE : structure non reconnue pour la découpe : " + std::move(detail)});
}

// Le graphe indexé par nom de nœud et par tenseur produit.
class Index {
 public:
  explicit Index(const OnnxGraph& graph) : graph_(graph) {
    for (const OnnxFullNode& node : graph.nodes) {
      by_name_.emplace(node.name, &node);
      for (const std::string& output : node.outputs) producer_.emplace(output, &node);
    }
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

 private:
  const OnnxGraph& graph_;
  std::map<std::string, const OnnxFullNode*, std::less<>> by_name_;
  std::map<std::string, const OnnxFullNode*, std::less<>> producer_;
};

bool text_is(const OnnxFullNode& node, std::string_view attribute, std::string_view value) {
  const OnnxAttribute* found = node.attribute(attribute);
  return found != nullptr && found->text && *found->text == value;
}

bool same(const std::vector<double>& got, std::initializer_list<double> want) {
  if (got.size() != want.size()) return false;
  std::size_t i = 0;
  for (const double w : want) {
    if (std::abs(got[i++] - w) > 1e-6) return false;
  }
  return true;
}

// Un Resize linéaire en demi-pixel d'échelle (1, 1, s, s) : s, ou 0 si ce n'est pas lui.
double resize_scale(const Index& index, const OnnxFullNode& node) {
  if (node.op_type != "Resize" || node.inputs.size() != 3 || !node.inputs[1].empty() ||
      !text_is(node, "mode", "linear") || !text_is(node, "coordinate_transformation_mode", "half_pixel"))
    return 0.0;
  const std::vector<double>& scales = index.constant(node.inputs[2]);
  if (scales.size() != 4 || scales[0] != 1.0 || scales[1] != 1.0 || scales[2] != scales[3]) return 0.0;
  return scales[2];
}

// 2, 4 ou 8 pour un facteur 1/d ou d ; 0 sinon.
int power_of_two_factor(double value) {
  for (const int d : {2, 4, 8}) {
    if (std::abs(value - d) < 1e-6 || std::abs(value * d - 1.0) < 1e-6) return d;
  }
  return 0;
}

std::string block_name(int b, std::string_view tail) {
  return std::format("/block{}/{}", b, tail);
}

}  // namespace

std::expected<RifeSplit, Refusal> analyze_rife(std::span<const std::byte> model) {
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  if (summary->inputs.size() != 1 || summary->inputs[0].name != "input" ||
      summary->inputs[0].dims.size() != 4 || summary->inputs[0].dims[1] != kernels::kFusedInputPlanes ||
      summary->inputs[0].elem_type != static_cast<std::int32_t>(OnnxType::Float16)) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch,
                "RIFE : entrée « input » 1 × 11 × H × W en FP16 attendue pour la découpe"});
  }
  auto graph = read_onnx_graph(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  if (graph->has_external_data || graph->has_sparse_initializers || graph->has_subgraphs)
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   "RIFE : poids externes, initialiseurs creux ou graphes imbriqués"});
  const Index index(*graph);
  RifeSplit split;
  split.elem_type = summary->inputs[0].elem_type;

  // Les attributs de tous les Resize et GridSample : ce que les noyaux refont.
  std::size_t resizes = 0;
  std::size_t grids = 0;
  std::size_t shuffles = 0;
  for (const OnnxFullNode& node : graph->nodes) {
    if (node.op_type == "Resize") {
      ++resizes;
      if (resize_scale(index, node) == 0.0)
        return unknown(node.name + " : Resize non linéaire/demi-pixel à échelle constante");
    } else if (node.op_type == "GridSample") {
      ++grids;
      const OnnxAttribute* align = node.attribute("align_corners");
      if (align == nullptr || !align->integer || *align->integer != 1 ||
          !text_is(node, "padding_mode", "border") || !text_is(node, "mode", "bilinear"))
        return unknown(node.name + " : GridSample hors coins alignés, bords répétés, bilinéaire");
    } else if (node.op_type == "DepthToSpace") {
      ++shuffles;
      const OnnxAttribute* size = node.attribute("blocksize");
      if (size == nullptr || !size->integer || *size->integer != 2 || !text_is(node, "mode", "CRD"))
        return unknown(node.name + " : DepthToSpace hors CRD de bloc 2");
    }
  }

  // L'encodeur.
  const OnnxFullNode* head_split = index.named("/Split", "Split");
  const OnnxFullNode* head_reshape = index.named("/Reshape", "Reshape");
  const OnnxFullNode* encoder = index.named("/encode/cnn3/ConvTranspose", "ConvTranspose");
  if (head_split == nullptr || head_reshape == nullptr || encoder == nullptr ||
      head_split->inputs.size() != 2 || head_split->inputs[0] != "input" ||
      !same(index.constant(head_split->inputs[1]), {6, 1, 2, 2}) || head_reshape->inputs.empty() ||
      head_reshape->inputs[0] != "/Split_output_0" || head_reshape->outputs.size() != 1 ||
      encoder->outputs.size() != 1) {
    return unknown("découpe de l'entrée ou encodeur");
  }
  split.encode_input = head_reshape->outputs[0];
  split.encode_output = encoder->outputs[0];

  // Les blocs : autant que de ConvTranspose « lastconv ».
  int blocks = 0;
  while (blocks < kernels::kFusedMaxBlocks + 1 &&
         index.named(block_name(blocks, "lastconv/lastconv.0/ConvTranspose"), "ConvTranspose") != nullptr)
    ++blocks;
  if (blocks < 2 || blocks > kernels::kFusedMaxBlocks) return unknown(std::format("{} blocs", blocks));
  split.plan.blocks = blocks;
  for (int b = 0; b < blocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const OnnxFullNode* last =
        index.named(block_name(b, "lastconv/lastconv.0/ConvTranspose"), "ConvTranspose");
    const OnnxFullNode* first = index.named(block_name(b, "conv0/conv0.0/conv0.0.0/Conv"), "Conv");
    const OnnxFullNode* shuffle =
        index.named(block_name(b, "lastconv/lastconv.1/DepthToSpace"), "DepthToSpace");
    const OnnxFullNode* sizes = index.named(block_name(b, "Split"), "Split");
    if (last == nullptr || first == nullptr || shuffle == nullptr || sizes == nullptr ||
        last->outputs.size() != 1 || first->inputs.empty() || shuffle->inputs.empty() ||
        shuffle->inputs[0] != last->outputs[0] || shuffle->outputs.size() != 1 || sizes->inputs.size() != 2 ||
        !same(index.constant(sizes->inputs[1]), {4, 1, 8}))
      return unknown(std::format("bloc {}", b));
    split.block_input[i] = first->inputs[0];
    split.block_output[i] = last->outputs[0];

    // Réduction d'entrée : un Resize d'échelle 1/d juste avant la première convolution, ou rien (d = 1).
    int d = 1;
    const OnnxFullNode* source = index.producer(first->inputs[0]);
    if (source == nullptr) return unknown(std::format("bloc {} : entrée sans producteur", b));
    if (source->op_type == "Resize") {
      d = power_of_two_factor(1.0 / resize_scale(index, *source));
      if (d == 0 || source->name != block_name(b, "Resize"))
        return unknown(std::format("bloc {} : Resize d'entrée", b));
      source = index.producer(source->inputs[0]);
    }
    split.plan.reduce[i] = d;

    // Agrandissement de sortie : un Resize d'échelle d entre le DepthToSpace et le Split, ou rien (d = 1).
    const OnnxFullNode* grown = index.named(block_name(b, "Resize_1"), "Resize");
    if (d == 1) {
      if (grown != nullptr || sizes->inputs[0] != shuffle->outputs[0])
        return unknown(std::format("bloc {} : sortie", b));
    } else {
      if (grown == nullptr || grown->inputs[0] != shuffle->outputs[0] || grown->outputs.size() != 1 ||
          sizes->inputs[0] != grown->outputs[0] || power_of_two_factor(resize_scale(index, *grown)) != d ||
          resize_scale(index, *grown) < 1.0)
        return unknown(std::format("bloc {} : Resize de sortie", b));
      // Le flot du bloc est multiplié par d.
      const OnnxFullNode* scale = index.named(block_name(b, "Mul"), "Mul");
      if (scale == nullptr || scale->inputs.size() != 2 ||
          scale->inputs[0] != block_name(b, "Split_output_0") ||
          !same(index.constant(scale->inputs[1]), {static_cast<double>(d)}))
        return unknown(std::format("bloc {} : Mul du flot", b));
    }

    // Ce qui précède l'entrée : le Concat de la colle, dont l'ordre des canaux fixe la disposition de X_k.
    if (source == nullptr || source->op_type != "Concat")
      return unknown(std::format("bloc {} : Concat d'entrée", b));
    if (b == 0) {
      if (source->inputs.size() != 3 || source->inputs[0] != "/Split_output_0" ||
          source->inputs[2] != "/Split_output_1")
        return unknown("Concat du bloc 0");
    } else {
      const int previous = b - 1;
      if (source->inputs.size() != 6 || source->inputs[2] != "/Split_output_1" ||
          source->inputs[3] != block_name(previous, "Split_output_1") ||
          source->inputs[4] != block_name(previous, "Split_output_2"))
        return unknown(std::format("Concat du bloc {}", b));
      // Le flot accumulé arrive multiplié par 1/d (ou tel quel pour d = 1).
      const OnnxFullNode* flow = index.producer(source->inputs[5]);
      if (flow == nullptr) return unknown(std::format("bloc {} : flot", b));
      if (d > 1) {
        if (flow->op_type != "Mul" || flow->inputs.size() != 2 ||
            !same(index.constant(flow->inputs[1]), {1.0 / static_cast<double>(d)}))
          return unknown(std::format("bloc {} : Mul 1/d du flot", b));
      } else if (flow->op_type == "Mul") {
        return unknown(std::format("bloc {} : Mul du flot à d = 1", b));
      }
    }
  }
  if (!kernels::rife_fused_plan_valid(split.plan)) return unknown("plan des réductions");
  // Un GridSample et un DepthToSpace par bloc, un Resize à l'entrée et un à la sortie de chaque bloc qui
  // réduit.
  const auto n = static_cast<std::size_t>(blocks);
  std::size_t expected = 0;
  for (int b = 0; b < blocks; ++b) expected += split.plan.reduce[static_cast<std::size_t>(b)] > 1 ? 2U : 0U;
  if (grids != n || shuffles != n || resizes != expected) return unknown("décompte des opérateurs");
  const OnnxFullNode* mask = index.named("/Sigmoid", "Sigmoid");
  if (mask == nullptr || mask->inputs.size() != 1 ||
      mask->inputs[0] != block_name(blocks - 1, "Split_output_1"))
    return unknown("masque sigmoïde du dernier bloc");
  return split;
}

std::string rife_piece_name(int index) {
  return index == 0 ? "encode" : std::format("bloc{}", index - 1);
}

std::expected<std::vector<RifePiece>, Refusal> split_rife(std::span<const std::byte> model,
                                                          const RifeSplit& split, int height, int width) {
  if (!kernels::rife_fused_size_valid(split.plan, height, width)) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeInvalid,
                std::format("RIFE fusionné : taille {}x{} que la découpe ne prend pas", width, height)});
  }
  std::vector<RifePiece> pieces;
  {
    const SplitTensor in{
        .name = split.encode_input, .elem_type = split.elem_type, .dims = {2, 3, height, width}};
    const SplitTensor out{.name = split.encode_output, .elem_type = split.elem_type};
    auto cut = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), rife_piece_name(0));
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.push_back({.name = rife_piece_name(0), .model = std::move(*cut)});
  }
  for (int b = 0; b < split.plan.blocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::FusedDims x = kernels::rife_fused_x(split.plan, b, height, width);
    const SplitTensor in{.name = split.block_input[i],
                         .elem_type = split.elem_type,
                         .dims = {1, x.channels, x.height, x.width}};
    const SplitTensor out{.name = split.block_output[i], .elem_type = split.elem_type};
    auto cut = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), rife_piece_name(b + 1));
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.push_back({.name = rife_piece_name(b + 1), .model = std::move(*cut)});
  }
  return pieces;
}

std::string format_manifest(const RifeFusedManifest& manifest) {
  std::string text = "ajp-rife-fused 1\n";
  text += std::format("output {}\n", manifest.output == RifeFusedOutput::Image ? "image" : "motion");
  text += std::format("size {} {}\n", manifest.height, manifest.width);
  text += "reduce";
  for (int b = 0; b < manifest.plan.blocks; ++b)
    text += std::format(" {}", manifest.plan.reduce[static_cast<std::size_t>(b)]);
  text += '\n';
  for (const std::string& piece : manifest.pieces) text += std::format("piece {}\n", piece);
  return text;
}

std::expected<RifeFusedManifest, Refusal> parse_manifest(std::string_view text) {
  const auto bad = [](std::string detail) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "RIFE fusionné : manifeste : " + std::move(detail)});
  };
  RifeFusedManifest manifest;
  bool header = false;
  bool has_output = false;
  bool has_size = false;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
    if (line.empty()) continue;
    const std::size_t space = line.find(' ');
    const std::string_view key = line.substr(0, space);
    std::string_view rest = space == std::string_view::npos ? std::string_view() : line.substr(space + 1);
    const auto number = [&rest] -> std::optional<int> {
      int value = 0;
      const char* first = rest.data();
      const auto [stop, error] = std::from_chars(first, first + rest.size(), value);
      if (error != std::errc{}) return std::nullopt;
      rest.remove_prefix(static_cast<std::size_t>(stop - first));
      while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1);
      return value;
    };
    if (key == "ajp-rife-fused") {
      if (rest != "1") return bad("version " + std::string(rest));
      header = true;
    } else if (key == "output") {
      if (rest == "image")
        manifest.output = RifeFusedOutput::Image;
      else if (rest == "motion")
        manifest.output = RifeFusedOutput::Motion;
      else
        return bad("sortie « " + std::string(rest) + " »");
      has_output = true;
    } else if (key == "size") {
      const auto h = number();
      const auto w = h ? number() : std::nullopt;
      if (!h || !w) return bad("taille illisible");
      manifest.height = *h;
      manifest.width = *w;
      has_size = true;
    } else if (key == "reduce") {
      while (!rest.empty()) {
        const auto d = number();
        if (!d || manifest.plan.blocks >= kernels::kFusedMaxBlocks) return bad("réductions illisibles");
        manifest.plan.reduce[static_cast<std::size_t>(manifest.plan.blocks++)] = *d;
      }
    } else if (key == "piece") {
      // Un nom de fichier seul : jamais de dossier ni de « .. ».
      if (rest.empty() || rest.find_first_of("/\\") != std::string_view::npos || rest == "." || rest == "..")
        return bad("morceau « " + std::string(rest) + " »");
      manifest.pieces.emplace_back(rest);
    } else {
      return bad("champ « " + std::string(key) + " » inconnu");
    }
  }
  if (!header || !has_output || !has_size) return bad("en-tête, sortie ou taille absents");
  if (!kernels::rife_fused_size_valid(manifest.plan, manifest.height, manifest.width))
    return bad("découpage ou taille que les noyaux ne prennent pas");
  if (manifest.pieces.size() != static_cast<std::size_t>(manifest.plan.blocks) + 1)
    return bad("nombre de morceaux différent de 1 + blocs");
  return manifest;
}

bool is_rife_fused_manifest(const std::filesystem::path& path) {
  return path.extension() == kRifeFusedExtension;
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

std::expected<std::filesystem::path, Refusal> derive_rife_fused(const std::filesystem::path& rife_onnx,
                                                                const std::filesystem::path& dir, int height,
                                                                int width, RifeFusedOutput output) {
  auto sha = sha256_file_known(rife_onnx);
  if (!sha) return std::unexpected(std::move(sha.error()));
  const std::string stem = std::format("{}-{}-{}x{}-{}", rife_onnx.stem().string(), kRifeFusedDerivation,
                                       height, width, sha->substr(0, 16));
  std::filesystem::path manifest_path =
      dir / std::format("{}-{}{}", stem, output == RifeFusedOutput::Image ? "image" : "motion",
                        kRifeFusedExtension);
  // Déjà dérivé : le manifeste est écrit en dernier, sa présence dit que les morceaux y sont.
  if (std::ifstream known(manifest_path); known) {
    const std::string text{std::istreambuf_iterator<char>(known), std::istreambuf_iterator<char>()};
    if (const auto parsed = parse_manifest(text);
        parsed && parsed->height == height && parsed->width == width &&
        std::ranges::all_of(parsed->pieces,
                            [&](const std::string& piece) { return exists_nonempty(dir / piece); }))
      return manifest_path;
  }
  auto bytes = read_onnx_bytes(rife_onnx);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto split = analyze_rife(*bytes);
  if (!split) return std::unexpected(std::move(split.error()));
  auto pieces = split_rife(*bytes, *split, height, width);
  if (!pieces) return std::unexpected(std::move(pieces.error()));
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  RifeFusedManifest manifest{.output = output, .height = height, .width = width, .plan = split->plan};
  for (const RifePiece& piece : *pieces) {
    const std::string file = std::format("{}-{}.onnx", stem, piece.name);
    if (!exists_nonempty(dir / file)) {
      if (auto written = write_atomic(dir / file, piece.model); !written)
        return std::unexpected(std::move(written.error()));
    }
    manifest.pieces.push_back(file);
  }
  const std::string text = format_manifest(manifest);
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  if (auto written = write_atomic(manifest_path, std::span(raw, text.size())); !written)
    return std::unexpected(std::move(written.error()));
  return manifest_path;
}

}  // namespace ajp::prepare
