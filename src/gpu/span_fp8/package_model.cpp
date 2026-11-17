#include "gpu/span_fp8/package_model.h"

#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <utility>

#include "gpu/span_fp8/quantize.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

using ir::onnx::Attribute;
using ir::onnx::AttributeType;
using ir::onnx::Node;
using ir::onnx::Tensor;

std::unexpected<Refusal> refuse(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FormatUnsupported, "graphe du paquet SPAN : " + std::move(detail)});
}

float half_round(float v) {
  return kernels::ref::half_to_float(kernels::ref::float_to_half(v));
}

// Les poids d'une convolution du paquet, rangés [sortie][ligne][colonne][groupe × stored], en ordre ONNX
// [sortie][groupe × real][ligne][colonne] : les outs premières sorties et les real premières entrées de
// chaque groupe. input_of (facultatif) donne, pour l'entrée i du graphe, celle du paquet.
std::vector<float> onnx_weights(const Conv& conv, int outs, int side, int groups, int stored, int real,
                                const std::vector<int>& input_of = {}) {
  const int inputs = groups * real;
  const int taps = side * side;
  std::vector<float> out(static_cast<std::size_t>(outs) * static_cast<std::size_t>(inputs) *
                         static_cast<std::size_t>(taps));
  for (int o = 0; o < outs; ++o)
    for (int i = 0; i < inputs; ++i) {
      const int from_input = input_of.empty() ? i : input_of[static_cast<std::size_t>(i)];
      const int g = from_input / real;
      const int c = from_input % real;
      for (int t = 0; t < taps; ++t) {
        const std::size_t from = (static_cast<std::size_t>(o * taps + t) * static_cast<std::size_t>(groups) +
                                  static_cast<std::size_t>(g)) *
                                     static_cast<std::size_t>(stored) +
                                 static_cast<std::size_t>(c);
        out[(static_cast<std::size_t>(o) * static_cast<std::size_t>(inputs) + static_cast<std::size_t>(i)) *
                static_cast<std::size_t>(taps) +
            static_cast<std::size_t>(t)] = kernels::ref::half_to_float(conv.weights[from]);
      }
    }
  return out;
}

Attribute ints(std::string name, std::vector<std::int64_t> values) {
  return Attribute{.name = std::move(name), .type = AttributeType::Ints, .ints = std::move(values)};
}

Attribute integer(std::string name, std::int64_t value) {
  return Attribute{.name = std::move(name), .type = AttributeType::Int, .i = value};
}

Attribute text(std::string name, std::string value) {
  return Attribute{.name = std::move(name), .type = AttributeType::String, .s = std::move(value)};
}

class Builder {
 public:
  explicit Builder(ir::onnx::Model& model) : graph_(model.graph) {}

  void constant(const std::string& name, std::vector<std::int64_t> dims, std::vector<float> values,
                std::int32_t type = ir::onnx::kFloat16) {
    if (type == ir::onnx::kFloat16)
      for (float& v : values) v = half_round(v);
    graph_.initializers.push_back(
        Tensor{.name = name, .data_type = type, .dims = std::move(dims), .floats = std::move(values)});
  }

  std::string node(const std::string& op, std::vector<std::string> inputs, const std::string& output,
                   std::vector<Attribute> attributes = {}) {
    graph_.nodes.push_back(Node{.name = std::format("{}_{}", op, ++count_),
                                .op_type = op,
                                .domain = "",
                                .inputs = std::move(inputs),
                                .outputs = {output},
                                .attributes = std::move(attributes)});
    return output;
  }

  // Conv au remplissage de side / 2 de chaque côté, poids et biais en constantes FP16.
  std::string conv(const std::string& name, const std::string& x, std::vector<float> weights, int outs,
                   int ins, int side, const std::vector<float>& bias) {
    constant(name + ".weight", {outs, ins, side, side}, std::move(weights));
    constant(name + ".bias", {outs}, std::vector<float>(bias.begin(), bias.begin() + outs));
    const std::int64_t pad = side / 2;
    return node("Conv", {x, name + ".weight", name + ".bias"}, name + "_out",
                {ints("kernel_shape", {side, side}), ints("pads", {pad, pad, pad, pad})});
  }

  // x arrondi à la grille E4M3 de son échelle, une constante float scalaire.
  std::string quant(const std::string& x, float scale) {
    const std::string name = std::format("{}_scale_{}", x, ++count_);
    constant(name, {}, {scale}, ir::onnx::kFloat);
    return node("FakeQuantE4M3", {x, name}, x + "_q");
  }

 private:
  ir::onnx::Graph& graph_;
  int count_ = 0;
};

// Les poids d'un bloc sur la grille E4M3 de leur canal de sortie (étape 10.3) : les valeurs que les
// matrices FP8 calculent, retrouvées par ir::recover_fp8_weights.
std::vector<float> on_e4m3_grid(std::vector<float> weights, int outs) {
  const std::vector<float> scales = channel_scales(weights, static_cast<std::size_t>(outs));
  return reconstruct(weights, scales);
}

}  // namespace

std::vector<std::string> peak_names(std::size_t blocks) {
  std::vector<std::string> names{"conv_1_out"};
  for (std::size_t b = 0; b < blocks; ++b) {
    names.push_back(std::format("block_{}.r1", b + 1));
    names.push_back(std::format("block_{}.r2", b + 1));
    names.push_back(std::format("block_{}.out", b + 1));
  }
  return names;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<ir::onnx::Model, Refusal> package_model(const Package& package, const ModelOptions& options) {
  const std::size_t blocks = package.blocks.size();
  if (blocks < static_cast<std::size_t>(kMinBlocks) || blocks > static_cast<std::size_t>(kBlocks))
    return refuse(std::format("{} blocs, {} à {} attendus", blocks, kMinBlocks, kBlocks));
  const int ch = package.real_channels;
  if (ch < 1 || ch > kChannels) return refuse(std::format("{} canaux réels, 1 à {} attendus", ch, kChannels));
  const bool quantize = options.quantize && !package.fp16;
  if (quantize) {
    for (std::size_t k = 0; k < scale_count(blocks); ++k) {
      const float s = package.scales[k];
      if (!(s > 0.0F) || !std::isfinite(s)) return refuse(std::format("échelle {} : {}", k, s));
    }
  }

  ir::onnx::Model model{.ir_version = 8,
                        .producer_name = "ajp-span8",
                        .producer_version = "1",
                        .opsets = {{.domain = "", .version = 17}},
                        .graph = {}};
  model.graph.name = "span8";
  model.graph.inputs.push_back({.name = kModelInput,
                                .elem_type = ir::onnx::kFloat16,
                                .has_shape = true,
                                .dims = {1, 3, -1, -1},
                                .symbols = {"", "", "height", "width"}});
  Builder g(model);

  // Dégroupage : SpaceToDepth (DCR) puis conv_1, dont l'entrée i du graphe est l'entrée c·4 + dy·2 + dx du
  // paquet, i = (dy·2 + dx)·3 + c.
  std::vector<int> unshuffled(kUnshuffled);
  for (int i = 0; i < kUnshuffled; ++i) unshuffled[static_cast<std::size_t>(i)] = (i % 3) * 4 + i / 3;
  const std::string s2d = g.node("SpaceToDepth", {kModelInput}, "unshuffled", {integer("blocksize", 2)});
  const std::string first =
      g.conv("conv_1", s2d, onnx_weights(package.first, ch, 3, 1, kUnshuffledPadded, kUnshuffled, unshuffled),
             ch, kUnshuffled, 3, package.first.bias);
  const std::vector<std::string> peaks = peak_names(blocks);  // peaks[0] : la sortie de conv_1, first
  const std::string& first_out = first;

  std::string x = first_out;
  std::string first_block;
  std::string last_relu;
  for (std::size_t b = 0; b < blocks; ++b) {
    const Block& block = package.blocks[b];
    const std::string base = std::format("block_{}", b + 1);
    const float in_scale = b == 0 ? package.scales[0] : package.scales[3 + 3 * (b - 1)];
    const auto block_conv = [&](std::size_t i, const std::string& in) {
      std::vector<float> w = onnx_weights(block.conv[i], ch, 3, 1, kChannels, ch);
      if (quantize) w = on_e4m3_grid(std::move(w), ch);
      return g.conv(std::format("{}.c{}", base, i + 1), in, std::move(w), ch, ch, 3, block.conv[i].bias);
    };
    const std::string t1 = block_conv(0, quantize ? g.quant(x, in_scale) : x);
    const std::string r1 = g.node("Relu", {t1}, peaks[1 + 3 * b]);
    const std::string t2 = block_conv(1, quantize ? g.quant(r1, package.scales[1 + 3 * b]) : r1);
    const std::string r2 = g.node("Relu", {t2}, peaks[2 + 3 * b]);
    const std::string t3 = block_conv(2, quantize ? g.quant(r2, package.scales[2 + 3 * b]) : r2);
    const std::string att = g.conv(base + ".att", t3, onnx_weights(block.attention, ch, 1, 1, kChannels, ch),
                                   ch, ch, 1, block.attention.bias);
    const std::string gate = g.node("Tanh", {att}, base + ".tanh");
    const std::string sum = g.node("Add", {x, t3}, base + ".add");
    x = g.node("Mul", {sum, gate}, peaks[3 + 3 * b]);
    if (b == 0) first_block = x;
    if (b + 1 == blocks) last_relu = r1;
  }

  // La concaténation dans l'ordre des tranches des noyaux (onnx_weights.cpp) : conv_1, dernier bloc, premier
  // bloc, premier Relu du dernier bloc.
  const std::string cat_in =
      g.node("Concat", {first_out, x, first_block, last_relu}, "concat", {integer("axis", 1)});
  const std::string cat = g.conv("conv_cat", cat_in, onnx_weights(package.concat, ch, 1, 4, kChannels, ch),
                                 ch, 4 * ch, 1, package.concat.bias);
  const std::string pixel =
      g.conv("to_pixel", cat, onnx_weights(package.pixel, kPixelChannels, 3, 1, kChannels, ch),
             kPixelChannels, ch, 3, package.pixel.bias);
  g.node("DepthToSpace", {pixel}, kModelOutput, {integer("blocksize", 4), text("mode", "CRD")});

  model.graph.outputs.push_back({.name = kModelOutput, .elem_type = ir::onnx::kFloat16});
  if (options.peaks)
    for (const std::string& name : peaks)
      model.graph.outputs.push_back({.name = name, .elem_type = ir::onnx::kFloat16});
  return model;
}

}  // namespace ajp::gpu::span_fp8
