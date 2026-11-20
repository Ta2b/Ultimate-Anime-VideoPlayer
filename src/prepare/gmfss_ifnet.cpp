// Les quatre chaînes de convolutions de l'IFNet d'origine de Fortuna Union. Le modèle dérivé (gmfss_ifnet)
// garde les warps, le flot accumulé et les masques ; les morceaux s'arrêtent avant le PixelShuffle.
#include <algorithm>
#include <cmath>
#include <format>
#include <ranges>
#include <string_view>
#include <utility>

#include "prepare/gmflow_onnx_view.h"
#include "prepare/gmfss_fused.h"
#include "prepare/onnx_split.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;

struct BlockShape {
  int reduce;
  int input;
  int first;
  int body;
};

constexpr std::array<BlockShape, 4> kShapes{
    {{8, 7, 96, 192}, {4, 12, 64, 128}, {2, 12, 48, 96}, {1, 12, 32, 64}}};
constexpr int kTailChannels = 24;  // PixelShuffle ×2 : quatre canaux de flot et deux de masque
constexpr std::size_t kLayers = 11;

std::unexpected<Refusal> unsupported(std::string detail) {
  return refuse(RefusalCode::FormatUnsupported, "IFNet natif : " + std::move(detail));
}

bool inputs_match(const onnx::Graph& graph, int height, int width) {
  const auto matches = [&](std::string_view name, const std::vector<std::int64_t>& dims) {
    const auto i = std::ranges::find(graph.inputs, name, &onnx::ValueInfo::name);
    return i != graph.inputs.end() && i->elem_type == onnx::kFloat16 && i->has_shape && i->dims == dims;
  };
  const std::vector<std::int64_t> image{1, 3, height, width};
  return graph.inputs.size() == 3 && matches("half0", image) && matches("half1", image) &&
         matches("time", {1, 1, 1, 1}) && graph.outputs.size() == 1 &&
         graph.outputs.front().elem_type == onnx::kFloat16;
}

bool ordinary(const onnx::Node* node, std::size_t inputs) {
  return node != nullptr && onnx::is_default_domain(node->domain) && node->inputs.size() == inputs &&
         node->outputs.size() == 1;
}

bool int_attribute(const onnx::Node& node, std::string_view attribute, std::int64_t expected) {
  const auto* value = onnx_attribute(node, attribute);
  return value == nullptr ? expected == 0 : value->i == expected;
}

bool linear_resize(const OnnxView& view, const onnx::Node* node, float scale) {
  if (!ordinary(node, 3) || !node->inputs[1].empty() || !int_attribute(*node, "antialias", 0) ||
      onnx_attribute(*node, "axes") != nullptr)
    return false;
  const auto* mode = onnx_attribute(*node, "mode");
  const auto* coordinate = onnx_attribute(*node, "coordinate_transformation_mode");
  const auto* scales = view.constant(node->inputs[2]);
  return mode != nullptr && mode->s == "linear" && coordinate != nullptr && coordinate->s == "half_pixel" &&
         scales != nullptr && scales->floats == std::vector<float>{1.F, 1.F, scale, scale};
}

bool entry_matches(const OnnxView& view, const onnx::Node& first, const std::string& prefix,
                   const BlockShape& shape) {
  const float scale = 1.F / static_cast<float>(shape.reduce);
  const auto* resize = view.node(prefix + "Resize", "Resize");
  if (!linear_resize(view, resize, scale)) return false;
  if (shape.input == 7) return first.inputs[0] == resize->outputs[0];
  const auto* concat = view.node(prefix + "Concat", "Concat");
  const auto* flow = view.node(prefix + "Resize_1", "Resize");
  if (!ordinary(concat, 2) || !linear_resize(view, flow, scale)) return false;
  const auto* unit_flow = view.node(prefix + "Mul", "Mul");
  const auto* scale_flow = view.producer(concat->inputs[1]);
  if (!ordinary(unit_flow, 2) || !ordinary(scale_flow, 2) || scale_flow->op_type != "Div") return false;
  return first.inputs[0] == concat->outputs[0] && concat->inputs[0] == resize->outputs[0] &&
         int_attribute(*concat, "axis", 1) && unit_flow->inputs[0] == flow->outputs[0] &&
         view.scalar(unit_flow->inputs[1]) == 1.F && scale_flow->inputs[0] == unit_flow->outputs[0] &&
         view.scalar(scale_flow->inputs[1]) == static_cast<float>(shape.reduce);
}

// Les deux formes de PixelShuffle : [N,C/4,2,2,H,W] puis [N,C/4,2H,2W]. Les Gather
// lisent la forme de la transposée et les petits sous-graphes ne font que Div par 4 ou Mul par 2.
bool shape_dim(const OnnxView& view, std::string_view expr, std::string_view tensor, std::int64_t axis) {
  const auto* gather = view.made_by(expr, "Gather");
  if (!ordinary(gather, 2) || !int_attribute(*gather, "axis", 0) || view.integer(gather->inputs[1]) != axis)
    return false;
  const auto* shape = view.made_by(gather->inputs[0], "Shape");
  return ordinary(shape, 1) && shape->inputs[0] == tensor && shape->attributes.empty();
}

bool scaled_dim(const OnnxView& view, std::string_view expr, std::string_view op, std::string_view tensor,
                std::int64_t axis, std::int64_t factor) {
  const auto* scaled = view.made_by(expr, op);
  return ordinary(scaled, 2) && shape_dim(view, scaled->inputs[0], tensor, axis) &&
         view.integer(scaled->inputs[1]) == factor;
}

bool shuffle_shapes(const OnnxView& view, const onnx::Node& first, const onnx::Node& second) {
  const auto* split = view.made_by(first.inputs[1], "Concat");
  const auto* joined = view.made_by(second.inputs[1], "Concat");
  if (!ordinary(split, 6) || !ordinary(joined, 4) || !int_attribute(*split, "axis", 0) ||
      !int_attribute(*joined, "axis", 0))
    return false;
  const std::vector<std::string>& from = split->inputs;
  const std::vector<std::string>& to = joined->inputs;
  const std::string& tensor = first.inputs[0];
  return shape_dim(view, from[0], tensor, 0) && scaled_dim(view, from[1], "Div", tensor, 1, 4) &&
         view.integer(from[2]) == 2 && view.integer(from[3]) == 2 && shape_dim(view, from[4], tensor, 2) &&
         shape_dim(view, from[5], tensor, 3) && view.base(to[0]) == view.base(from[0]) &&
         view.base(to[1]) == view.base(from[1]) && scaled_dim(view, to[2], "Mul", tensor, 2, 2) &&
         scaled_dim(view, to[3], "Mul", tensor, 3, 2);
}

bool tail_matches(const OnnxView& view, const onnx::Node& tail, const std::string& prefix,
                  const BlockShape& shape) {
  const std::string shuffle = prefix + "lastconv/lastconv.1/";
  const auto* first = view.node(shuffle + "Reshape", "Reshape");
  const auto* permute = view.node(shuffle + "Transpose", "Transpose");
  const auto* second = view.node(shuffle + "Reshape_1", "Reshape");
  const auto* resize = view.node(prefix + (shape.input == 7 ? "Resize_1" : "Resize_2"), "Resize");
  if (!ordinary(first, 2) || !ordinary(permute, 1) || !ordinary(second, 2) ||
      !linear_resize(view, resize, static_cast<float>(shape.reduce)))
    return false;
  const auto* perm = onnx_attribute(*permute, "perm");
  return first->inputs[0] == tail.outputs[0] && permute->inputs[0] == first->outputs[0] &&
         second->inputs[0] == permute->outputs[0] && resize->inputs[0] == second->outputs[0] &&
         perm != nullptr && perm->ints == std::vector<std::int64_t>{0, 1, 4, 2, 5, 3} &&
         int_attribute(*first, "allowzero", 0) && int_attribute(*second, "allowzero", 0) &&
         shuffle_shapes(view, *first, *second);
}

bool finite(const std::vector<float>& values) {
  return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
}

bool fp16_params(const OnnxView& view, const onnx::Node& node) {
  if (node.op_type == "Conv" || node.op_type == "ConvTranspose") {
    if (!ordinary(&node, 3) || !int_attribute(node, "group", 1)) return false;
    const auto* weight = view.constant(node.inputs[1]);
    const auto* bias = view.constant(node.inputs[2]);
    const int kernel = node.op_type == "ConvTranspose" ? 4 : 3;
    return weight != nullptr && weight->data_type == onnx::kFloat16 && weight->dims.size() == 4 &&
           weight->dims[2] == kernel && weight->dims[3] == kernel && bias != nullptr &&
           bias->data_type == onnx::kFloat16;
  }
  if (node.op_type == "Mul") {
    return ordinary(&node, 2) && std::ranges::any_of(node.inputs, [&](const std::string& in) {
             const auto* factor = view.constant(in);
             return factor != nullptr && factor->data_type == onnx::kFloat16;
           });
  }
  return node.op_type == "LeakyRelu" || node.op_type == "Add";
}

bool layers_match(std::span<const RifeNativeLayer> layers, const BlockShape& shape) {
  if (layers.size() != kLayers || !rife_native_check(layers, shape.input, kTailChannels) ||
      layers[0].cout != shape.first || layers[1].cout != shape.body)
    return false;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const RifeNativeLayer& layer = layers[i];
    const bool last = i + 1 == layers.size();
    const bool residual = i >= 2 && !last;
    const auto channels = static_cast<std::size_t>(layer.cout);
    const int kernel = last ? 4 : 3;
    const auto weights =
        static_cast<std::size_t>(layer.cin) * channels * static_cast<std::size_t>(kernel * kernel);
    if (layer.kernel != kernel || layer.stride != (residual ? 1 : 2) ||
        (residual && layer.cout != shape.body) || layer.weights.size() != weights ||
        layer.bias.size() != channels || layer.slope.size() != (last ? 0 : channels) ||
        layer.scale.size() != (residual ? channels : 0) || !finite(layer.weights) || !finite(layer.bias) ||
        !finite(layer.scale) || !std::ranges::all_of(layer.slope, [](float s) { return s == 0.2F; }))
      return false;
  }
  return true;
}

std::expected<std::vector<RifeNativeLayer>, Refusal> read_block(std::span<const std::byte> ifnet,
                                                                const onnx::Graph& graph,
                                                                const OnnxView& view, std::size_t block,
                                                                int height, int width) {
  const BlockShape& shape = kShapes[block];
  const std::string prefix = std::format("/ifnet/block{}/", block);
  const auto* first = view.node(prefix + "conv0/conv0.0/conv0.0.0/Conv", "Conv");
  const auto* tail = view.node(prefix + "lastconv/lastconv.0/ConvTranspose", "ConvTranspose");
  if (!ordinary(first, 3) || !ordinary(tail, 3) || !entry_matches(view, *first, prefix, shape) ||
      !tail_matches(view, *tail, prefix, shape))
    return unsupported(std::format("frontières du bloc {}", block));
  const std::string conv = prefix + "conv";
  const std::string transpose = prefix + "lastconv/lastconv.0/";
  auto body = graph.nodes | std::views::filter([&](const onnx::Node& node) {
                return node.name.starts_with(conv) || node.name.starts_with(transpose);
              });
  if (!std::ranges::all_of(body, [&](const onnx::Node& node) { return fp16_params(view, node); }))
    return unsupported(std::format("opérateur ou poids non FP16 dans le bloc {}", block));

  const int h = height / shape.reduce, w = width / shape.reduce;
  const SplitTensor input{.name = first->inputs[0], .dims = {1, shape.input, h, w}};
  const SplitTensor output{.name = tail->outputs[0], .dims = {1, kTailChannels, h / 2, w / 2}};
  auto piece = extract_subgraph(ifnet, std::span(&input, 1), std::span(&output, 1), prefix);
  if (!piece) return std::unexpected(std::move(piece.error()));
  auto layers = rife_native_layers(*piece);
  if (!layers) return std::unexpected(std::move(layers.error()));
  if (!layers_match(*layers, shape)) return unsupported(std::format("couches du bloc {}", block));
  return layers;
}

}  // namespace

std::expected<GmfssIfnetBlocks, Refusal> gmfss_ifnet_blocks(std::span<const std::byte> ifnet, int height,
                                                            int width) {
  // Deux convolutions de pas 2 après une réduction par 8 : il faut une grille multiple de 32.
  if (height <= 0 || width <= 0 || height % 32 != 0 || width % 32 != 0)
    return refuse(RefusalCode::ShapeInvalid, "IFNet natif : dimensions multiples de 32 attendues");
  auto model = onnx::read_onnx(ifnet);
  if (!model) return std::unexpected(std::move(model.error()));
  if (!inputs_match(model->graph, height, width)) return unsupported("entrées/sortie de l'IFNet dérivé");
  const OnnxView view(*model);
  GmfssIfnetBlocks blocks;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    auto layers = read_block(ifnet, model->graph, view, b, height, width);
    if (!layers) return std::unexpected(std::move(layers.error()));
    blocks[b] = std::move(*layers);
  }
  return blocks;
}

}  // namespace kirari::prepare
