#include "prepare/drba_split.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
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
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing,
              "DistilDRBA : structure non reconnue pour la découpe : " + std::move(detail)});
}

constexpr double kEnd = static_cast<double>(std::numeric_limits<std::int64_t>::max());

// Les tenseurs sortis des encodeurs (f0, f1, f2) et les noms que la colle leur donne.
const std::array<std::string, 3> kEncodeOut{"/encode/cnn3/ConvTranspose_output_0",
                                            "/encode/cnn3_1/ConvTranspose_output_0",
                                            "/encode/cnn3_2/ConvTranspose_output_0"};
const std::array<std::string, 3> kEncodeIn{"/Resize_output_0/cast16", "/Resize_1_output_0/cast16",
                                           "/Resize_2_output_0/cast16"};

std::string out0(std::string_view node) {
  return std::format("{}_output_0", node);
}
std::string cast16(std::string_view node) {
  return std::format("{}_output_0/cast16", node);
}
std::string cast32(std::string_view tensor) {
  return std::format("{}/cast32", tensor);
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
  const OnnxFullNode* named(const std::string& name) const {
    const auto found = by_name_.find(name);
    return found != by_name_.end() ? found->second : nullptr;
  }
  const OnnxFullNode* producer(const std::string& tensor) const {
    const auto found = producer_.find(tensor);
    return found != producer_.end() ? found->second : nullptr;
  }
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

bool same(const std::vector<double>& got, const std::vector<double>& want) {
  if (got.size() != want.size()) return false;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::abs(got[i] - want[i]) > 1e-6 * std::max(1.0, std::abs(want[i]))) return false;
  }
  return true;
}

std::string block_name(int b, std::string_view tail) {
  return std::format("/block{}/{}", b, tail);
}

}  // namespace

DrbaExpectedWiring drba_expected_wiring(int tail) {
  DrbaExpectedWiring wiring;
  const double t = static_cast<double>(tail);
  const auto node = [&](std::string name, std::string op, std::vector<std::string> inputs) {
    DrbaExpectedNode n{.name = std::move(name), .op = std::move(op), .inputs = std::move(inputs)};
    wiring.nodes.push_back(std::move(n));
  };
  const auto constant = [&](std::string name, std::size_t input, std::vector<double> values) {
    wiring.constants.push_back({.node = std::move(name), .input = input, .values = std::move(values)});
  };
  const auto slice = [&](const std::string& name, std::string data, double begin, double end) {
    node(name, "Slice", {std::move(data)});
    constant(name, 1, {begin});
    constant(name, 2, {end});
  };
  const auto resize = [&](int n, std::string input, double scale) {
    const std::string name = n == 0 ? "/Resize" : std::format("/Resize_{}", n);
    node(name, "Resize", {std::move(input), "", ""});
    constant(name, 2, {1.0, 1.0, scale, scale});
  };

  // Les plans de pair01 et pair12, le temps au coin et le côté choisi.
  slice("/Slice", "pair01", 0, 3);
  slice("/Slice_1", "pair01", 3, 6);
  slice("/Slice_2", "pair12", 3, 6);
  slice("/Slice_3", "pair01", 6, 7);
  slice("/Slice_4", out0("/Slice_3"), 0, 1);
  slice("/Slice_5", out0("/Slice_4"), 0, 1);
  node("/Mul", "Mul", {out0("/Slice_5"), ""});
  constant("/Mul", 1, {2.0});
  node("/Sub", "Sub", {"", out0("/Mul")});
  constant("/Sub", 0, {1.0});
  node("/Mul_1", "Mul", {out0("/Sub"), ""});
  constant("/Mul_1", 1, {4096.0});
  node("/Clip", "Clip", {out0("/Mul_1"), "", ""});
  constant("/Clip", 1, {0.0});
  constant("/Clip", 2, {1.0});
  node("/Sub_1", "Sub", {"", out0("/Clip")});
  constant("/Sub_1", 0, {1.0});
  node("/Mul_2", "Mul", {out0("/Clip"), out0("/Slice")});
  node("/Mul_3", "Mul", {out0("/Sub_1"), out0("/Slice_2")});
  node("/Add", "Add", {out0("/Mul_2"), out0("/Mul_3")});
  node("/Mul_4", "Mul", {out0("/Clip"), kEncodeOut[0]});
  node("/Mul_5", "Mul", {out0("/Sub_1"), kEncodeOut[2]});
  node("/Add_1", "Add", {out0("/Mul_4"), out0("/Mul_5")});

  // L'encodeur : trois copies sur les images réduites de t.
  const std::array<std::string, 3> images{out0("/Slice"), out0("/Slice_1"), out0("/Slice_2")};
  const std::array<std::string, 3> suffix{"", "_1", "_2"};
  for (std::size_t i = 0; i < 3; ++i) {
    resize(static_cast<int>(i), cast32(images[i]), 1.0 / t);
    node(std::format("/encode/cnn0{}/Conv", suffix[i]), "Conv", {kEncodeIn[i]});
    node(std::format("/encode/cnn3{}/ConvTranspose", suffix[i]), "ConvTranspose", {});
  }

  // Bloc 0 : les images au 1/(4 t), les encodages au quart, le temps au 1/(4 t).
  resize(3, cast32(kEncodeOut[0]), 0.25);
  resize(4, cast32(kEncodeOut[1]), 0.25);
  resize(5, cast32(kEncodeOut[2]), 0.25);
  resize(6, cast32(images[0]), 1.0 / (4 * t));
  resize(7, cast32(images[1]), 1.0 / (4 * t));
  resize(8, cast32(images[2]), 1.0 / (4 * t));
  slice("/Slice_6", out0("/Slice_3"), 0, kEnd);
  constant("/Slice_6", 4, {4 * t});
  slice("/Slice_7", out0("/Slice_6"), 0, kEnd);
  constant("/Slice_7", 4, {4 * t});
  node("/Mul_6", "Mul", {out0("/Slice_7"), ""});
  constant("/Mul_6", 1, {2.0});
  node("/Concat", "Concat",
       {cast16("/Resize_6"), cast16("/Resize_7"), cast16("/Resize_8"), cast16("/Resize_3"),
        cast16("/Resize_4"), cast16("/Resize_5"), out0("/Mul_6")});

  // Les trois blocs : coupes de la sortie en flot, masque et attributs.
  for (int b = 0; b < 3; ++b) {
    const std::string p = block_name(b, "");
    const std::string shuffle = p + "lastconv/lastconv.1/DepthToSpace";
    node(p + "conv0/conv0.0/conv0.0.0/Conv", "Conv", {b == 0 ? out0("/Concat") : out0(p + "Concat")});
    node(p + "lastconv/lastconv.0/ConvTranspose", "ConvTranspose", {});
    node(shuffle, "DepthToSpace", {out0(p + "lastconv/lastconv.0/ConvTranspose")});
    slice(p + "Slice", out0(shuffle), 0, 4);
    slice(p + "Slice_1", out0(shuffle), 4, 5);
    if (b < 2) slice(p + "Slice_2", out0(shuffle), 5, kEnd);  // le dernier bloc n'a pas d'attributs à passer
  }
  // Flot, masque et attributs du bloc 0 doublés vers le bloc 1, ceux du bloc 1 vers le bloc 2.
  resize(9, cast32(out0("/block0/Slice")), 2.0);
  resize(10, cast32(out0("/block0/Slice_1")), 2.0);
  resize(11, cast32(out0("/block0/Slice_2")), 2.0);
  node("/Mul_7", "Mul", {cast16("/Resize_9"), ""});
  constant("/Mul_7", 1, {2.0});
  resize(20, cast32(out0("/block1/Slice")), 2.0);
  resize(21, cast32(out0("/block1/Slice_1")), 2.0);
  resize(22, cast32(out0("/block1/Slice_2")), 2.0);
  node("/Mul_33", "Mul", {cast16("/Resize_20"), ""});
  constant("/Mul_33", 1, {2.0});

  // Bloc 1 : images au 1/(2 t), encodages au demi, temps au 1/(2 t), warps au 1/(2 t).
  resize(12, cast32(kEncodeOut[1]), 0.5);
  resize(13, cast32(out0("/Add_1")), 0.5);
  resize(14, cast32(images[1]), 1.0 / (2 * t));
  resize(15, cast32(out0("/Add")), 1.0 / (2 * t));
  resize(16, cast32(images[0]), 1.0 / (2 * t));
  resize(17, cast32(images[2]), 1.0 / (2 * t));
  resize(18, cast32(kEncodeOut[0]), 0.5);
  resize(19, cast32(kEncodeOut[2]), 0.5);
  slice("/Slice_8", out0("/Slice_3"), 0, kEnd);
  constant("/Slice_8", 4, {2 * t});
  slice("/Slice_9", out0("/Slice_8"), 0, kEnd);
  constant("/Slice_9", 4, {2 * t});
  node("/Mul_8", "Mul", {out0("/Slice_9"), ""});
  constant("/Mul_8", 1, {2.0});
  slice("/Slice_10", out0("/Mul_7"), 0, 2);
  slice("/Slice_13", out0("/Mul_7"), 2, 4);
  node("/GridSample", "GridSample", {out0("/Resize_12"), cast32(out0("/Cast_8"))});
  node("/GridSample_1", "GridSample", {out0("/Resize_14"), cast32(out0("/Cast_11"))});
  node("/GridSample_2", "GridSample", {out0("/Resize_13"), cast32(out0("/Cast_20"))});
  node("/GridSample_3", "GridSample", {out0("/Resize_15"), cast32(out0("/Cast_23"))});
  node("/block1/Concat", "Concat",
       {cast16("/Resize_16"), cast16("/Resize_14"), cast16("/Resize_17"), cast16("/Resize_18"),
        cast16("/Resize_12"), cast16("/Resize_19"), cast16("/GridSample_1"), cast16("/GridSample_3"),
        cast16("/GridSample"), cast16("/GridSample_2"), out0("/Mul_8"), cast16("/Resize_10"),
        cast16("/Resize_11"), out0("/Mul_7")});

  // Bloc 2 : images au 1/t (celles de l'encodeur), encodages entiers, temps au 1/t.
  resize(23, cast32(out0("/Add")), 1.0 / t);
  slice("/Slice_16", out0("/Slice_3"), 0, kEnd);
  constant("/Slice_16", 4, {t});
  slice("/Slice_17", out0("/Slice_16"), 0, kEnd);
  constant("/Slice_17", 4, {t});
  node("/Mul_34", "Mul", {out0("/Slice_17"), ""});
  constant("/Mul_34", 1, {2.0});
  slice("/Slice_18", out0("/Mul_33"), 0, 2);
  slice("/Slice_21", out0("/Mul_33"), 2, 4);
  node("/GridSample_4", "GridSample", {cast32(kEncodeOut[1]), cast32(out0("/Cast_32"))});
  node("/GridSample_5", "GridSample", {out0("/Resize_1"), cast32(out0("/Cast_35"))});
  node("/GridSample_6", "GridSample", {cast32(out0("/Add_1")), cast32(out0("/Cast_44"))});
  node("/GridSample_7", "GridSample", {out0("/Resize_23"), cast32(out0("/Cast_47"))});
  node("/block2/Concat", "Concat",
       {kEncodeIn[0], kEncodeIn[1], kEncodeIn[2], kEncodeOut[0], kEncodeOut[1], kEncodeOut[2],
        cast16("/GridSample_5"), cast16("/GridSample_7"), cast16("/GridSample_4"), cast16("/GridSample_6"),
        out0("/Mul_34"), cast16("/Resize_21"), cast16("/Resize_22"), out0("/Mul_33")});

  // Fin : la sortie du bloc 2 agrandie ×t, le masque, deux warps pleine résolution et le mélange.
  resize(24, cast32(out0("/block2/Slice")), t);
  resize(25, cast32(out0("/block2/Slice_1")), t);
  node("/Mul_59", "Mul", {cast16("/Resize_24"), ""});
  constant("/Mul_59", 1, {t});
  slice("/Slice_24", out0("/Mul_59"), 0, 2);
  slice("/Slice_27", out0("/Mul_59"), 2, 4);
  node("/Sigmoid", "Sigmoid", {cast16("/Resize_25")});
  node("/GridSample_8", "GridSample", {cast32(out0("/Slice_1")), cast32(out0("/Cast_56"))});
  node("/GridSample_9", "GridSample", {cast32(out0("/Add")), cast32(out0("/Cast_65"))});
  node("/Mul_68", "Mul", {cast16("/GridSample_8"), out0("/Sigmoid")});
  node("/Sub_46", "Sub", {"", out0("/Sigmoid")});
  constant("/Sub_46", 0, {1.0});
  node("/Mul_77", "Mul", {cast16("/GridSample_9"), out0("/Sub_46")});
  node("/Add_22", "Add", {out0("/Mul_68"), out0("/Mul_77")});
  return wiring;
}

std::expected<DrbaSplit, Refusal> analyze_drba(std::span<const std::byte> model) {
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  if (summary->inputs.size() != 2 || summary->inputs[0].name != "pair01" ||
      summary->inputs[1].name != "pair12") {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch,
                "DistilDRBA : entrées « pair01 » et « pair12 » attendues pour la découpe"});
  }
  for (const OnnxInput& input : summary->inputs) {
    if (input.dims.size() != 4 || input.dims[1] != kernels::kDrbaFusedInputPlanes ||
        input.elem_type != static_cast<std::int32_t>(OnnxType::Float16))
      return std::unexpected(Refusal{RefusalCode::InputMismatch,
                                     "DistilDRBA : entrées 1 × 7 × H × W en FP16 attendues pour la découpe"});
  }
  auto graph = read_onnx_graph(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  if (graph->has_external_data || graph->has_sparse_initializers || graph->has_subgraphs)
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   "DistilDRBA : poids externes, initialiseurs creux ou graphes imbriqués"});
  const Index index(*graph);
  DrbaSplit split;
  split.elem_type = summary->inputs[0].elem_type;

  // Les échelles : celle du Resize d'entrée de l'encodeur donne t.
  const OnnxFullNode* first_resize = index.named("/Resize");
  if (first_resize == nullptr || first_resize->inputs.size() != 3)
    return unknown("Resize d'entrée de l'encodeur");
  const std::vector<double>& first_scale = index.constant(first_resize->inputs[2]);
  if (first_scale.size() != 4 || first_scale[2] <= 0.0) return unknown("échelle d'entrée de l'encodeur");
  split.plan.tail = static_cast<int>(std::lround(1.0 / first_scale[2]));
  if (!kernels::drba_fused_plan_valid(split.plan))
    return unknown(std::format("réduction t = {}", split.plan.tail));

  // Tous les nœuds de la colle, avec leurs entrées et leurs constantes.
  const DrbaExpectedWiring wiring = drba_expected_wiring(split.plan.tail);
  for (const DrbaExpectedNode& want : wiring.nodes) {
    const OnnxFullNode* got = index.named(want.name);
    if (got == nullptr || got->op_type != want.op)
      return unknown(want.name + " : nœud absent ou d'un autre type");
    for (std::size_t i = 0; i < want.inputs.size(); ++i) {
      if (!want.inputs[i].empty() && (i >= got->inputs.size() || got->inputs[i] != want.inputs[i]))
        return unknown(std::format("{} : entrée {} différente", want.name, i));
    }
  }
  for (const DrbaExpectedConstant& want : wiring.constants) {
    const OnnxFullNode* got = index.named(want.node);
    if (got == nullptr || want.input >= got->inputs.size() ||
        !same(index.constant(got->inputs[want.input]), want.values))
      return unknown(std::format("{} : constante d'entrée {} différente", want.node, want.input));
  }

  // Les attributs et le nombre des opérateurs que les noyaux refont : rien de plus que ce que le câblage a
  // vu.
  std::size_t resizes = 0;
  std::size_t grids = 0;
  std::size_t shuffles = 0;
  for (const OnnxFullNode& node : graph->nodes) {
    if (node.op_type == "Resize") {
      ++resizes;
      if (node.inputs.size() != 3 || !node.inputs[1].empty() || !text_is(node, "mode", "linear") ||
          !text_is(node, "coordinate_transformation_mode", "half_pixel"))
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
  if (resizes != 26 || grids != 10 || shuffles != 3) return unknown("décompte des opérateurs");

  // L'encodeur : l'image réduite en entrée de la première convolution, l'encodage en sortie du ConvTranspose.
  const OnnxFullNode* encoder_in = index.named("/encode/cnn0/Conv");
  const OnnxFullNode* encoder_out = index.named("/encode/cnn3/ConvTranspose");
  if (encoder_in == nullptr || encoder_out == nullptr || encoder_in->inputs.empty() ||
      encoder_out->outputs.size() != 1)
    return unknown("encodeur");
  split.encode_input = encoder_in->inputs[0];
  split.encode_output = encoder_out->outputs[0];

  // Les blocs.
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const OnnxFullNode* first = index.named(block_name(b, "conv0/conv0.0/conv0.0.0/Conv"));
    const OnnxFullNode* last = index.named(block_name(b, "lastconv/lastconv.0/ConvTranspose"));
    if (first == nullptr || last == nullptr || first->inputs.empty() || last->outputs.size() != 1)
      return unknown(std::format("bloc {}", b));
    const OnnxFullNode* concat = index.producer(first->inputs[0]);
    const std::size_t arity = b == 0 ? 7U : 14U;
    if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != arity)
      return unknown(std::format("bloc {} : Concat d'entrée", b));
    split.block_input[i] = first->inputs[0];
    split.block_output[i] = last->outputs[0];
  }
  if (index.named(block_name(3, "conv0/conv0.0/conv0.0.0/Conv")) != nullptr)
    return unknown("plus de trois blocs");
  return split;
}

std::string drba_piece_name(int index) {
  return index == 0 ? "encode" : std::format("bloc{}", index - 1);
}

std::expected<std::vector<DrbaPiece>, Refusal> split_drba(std::span<const std::byte> model,
                                                          const DrbaSplit& split, int height, int width) {
  if (!kernels::drba_fused_size_valid(split.plan, height, width)) {
    return std::unexpected(Refusal{
        RefusalCode::ShapeInvalid,
        std::format("DistilDRBA fusionné : taille {}x{} que la découpe ne prend pas", width, height)});
  }
  const int t = split.plan.tail;
  std::vector<DrbaPiece> pieces;
  {
    // Le lot de trois images : les poids de l'encodeur sont partagés, la convolution ne dépend pas du lot.
    const SplitTensor in{
        .name = split.encode_input, .elem_type = split.elem_type, .dims = {3, 3, height / t, width / t}};
    const SplitTensor out{.name = split.encode_output, .elem_type = split.elem_type};
    auto cut = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), drba_piece_name(0));
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.push_back({.name = drba_piece_name(0), .model = std::move(*cut)});
  }
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::DrbaFusedDims x = kernels::drba_fused_x(split.plan, b, height, width);
    const SplitTensor in{.name = split.block_input[i],
                         .elem_type = split.elem_type,
                         .dims = {1, x.channels, x.height, x.width}};
    const SplitTensor out{.name = split.block_output[i], .elem_type = split.elem_type};
    auto cut = extract_subgraph(model, std::span(&in, 1), std::span(&out, 1), drba_piece_name(b + 1));
    if (!cut) return std::unexpected(std::move(cut.error()));
    pieces.push_back({.name = drba_piece_name(b + 1), .model = std::move(*cut)});
  }
  return pieces;
}

std::string format_drba_manifest(const DrbaFusedManifest& manifest) {
  std::string text = "ajp-drba-fused 1\n";
  text += std::format("size {} {}\n", manifest.height, manifest.width);
  text += std::format("tail {}\n", manifest.plan.tail);
  for (const std::string& piece : manifest.pieces) text += std::format("piece {}\n", piece);
  return text;
}

std::expected<DrbaFusedManifest, Refusal> parse_drba_manifest(std::string_view text) {
  const auto bad = [](std::string detail) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "DistilDRBA fusionné : manifeste : " + std::move(detail)});
  };
  DrbaFusedManifest manifest;
  bool header = false;
  bool has_size = false;
  bool has_tail = false;
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
    if (key == "ajp-drba-fused") {
      if (rest != "1") return bad("version " + std::string(rest));
      header = true;
    } else if (key == "size") {
      const auto h = number();
      const auto w = h ? number() : std::nullopt;
      if (!h || !w) return bad("taille illisible");
      manifest.height = *h;
      manifest.width = *w;
      has_size = true;
    } else if (key == "tail") {
      const auto t = number();
      if (!t) return bad("réduction illisible");
      manifest.plan.tail = *t;
      has_tail = true;
    } else if (key == "piece") {
      // Un nom de fichier seul : jamais de dossier ni de « .. ».
      if (rest.empty() || rest.find_first_of("/\\") != std::string_view::npos || rest == "." || rest == "..")
        return bad("morceau « " + std::string(rest) + " »");
      manifest.pieces.emplace_back(rest);
    } else {
      return bad("champ « " + std::string(key) + " » inconnu");
    }
  }
  if (!header || !has_size || !has_tail) return bad("en-tête, taille ou réduction absents");
  if (!kernels::drba_fused_size_valid(manifest.plan, manifest.height, manifest.width))
    return bad("découpage ou taille que les noyaux ne prennent pas");
  if (manifest.pieces.size() != static_cast<std::size_t>(kernels::kDrbaFusedBlocks) + 1)
    return bad("nombre de morceaux différent de 4");
  return manifest;
}

bool is_drba_fused_manifest(const std::filesystem::path& path) {
  return path.extension() == kDrbaFusedExtension;
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

std::expected<std::filesystem::path, Refusal> derive_drba_fused(const std::filesystem::path& drba_onnx,
                                                                const std::filesystem::path& dir, int height,
                                                                int width) {
  auto sha = sha256_file_known(drba_onnx);
  if (!sha) return std::unexpected(std::move(sha.error()));
  const std::string stem = std::format("{}-{}-{}x{}-{}", drba_onnx.stem().string(), kDrbaFusedDerivation,
                                       height, width, sha->substr(0, 16));
  std::filesystem::path manifest_path = dir / std::format("{}{}", stem, kDrbaFusedExtension);
  // Déjà dérivé : le manifeste est écrit en dernier, sa présence dit que les morceaux y sont.
  if (std::ifstream known(manifest_path); known) {
    const std::string text{std::istreambuf_iterator<char>(known), std::istreambuf_iterator<char>()};
    if (const auto parsed = parse_drba_manifest(text);
        parsed && parsed->height == height && parsed->width == width &&
        std::ranges::all_of(parsed->pieces,
                            [&](const std::string& piece) { return exists_nonempty(dir / piece); }))
      return manifest_path;
  }
  auto bytes = read_onnx_bytes(drba_onnx);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto split = analyze_drba(*bytes);
  if (!split) return std::unexpected(std::move(split.error()));
  auto pieces = split_drba(*bytes, *split, height, width);
  if (!pieces) return std::unexpected(std::move(pieces.error()));
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  DrbaFusedManifest manifest{.height = height, .width = width, .plan = split->plan};
  for (const DrbaPiece& piece : *pieces) {
    const std::string file = std::format("{}-{}.onnx", stem, piece.name);
    if (!exists_nonempty(dir / file)) {
      if (auto written = write_atomic(dir / file, piece.model); !written)
        return std::unexpected(std::move(written.error()));
    }
    manifest.pieces.push_back(file);
  }
  const std::string text = format_drba_manifest(manifest);
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  if (auto written = write_atomic(manifest_path, std::span(raw, text.size())); !written)
    return std::unexpected(std::move(written.error()));
  return manifest_path;
}

}  // namespace ajp::prepare
