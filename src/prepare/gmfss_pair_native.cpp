#include "prepare/gmfss_pair_native.h"

#include <array>
#include <cstdint>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/onnx_model.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFSS natif : paire non reconnue : " + std::move(detail)});
}

struct Model {
  std::map<std::string, const onnx::Tensor*, std::less<>> constants;
  std::map<std::string, const onnx::Node*, std::less<>> nodes;
};

// La convolution 3 × 3 « gmfss.<name> » de pas stride : poids ONNX [cout][cin][3][3] rangés [cout][9][cin],
// et biais.
std::expected<GmfssGridConv, Refusal> read_conv(const Model& model, std::string_view name, int stride) {
  const auto weight = model.constants.find(std::format("gmfss.{}.weight", name));
  const auto bias = model.constants.find(std::format("gmfss.{}.bias", name));
  if (weight == model.constants.end() || bias == model.constants.end())
    return missing(std::format("poids ou biais de {}", name));
  const std::vector<std::int64_t>& d = weight->second->dims;
  if (d.size() != 4 || d[2] != 3 || d[3] != 3) return missing(std::format("{} : poids 3×3 attendus", name));
  GmfssGridConv conv{.cin = static_cast<int>(d[1]), .cout = static_cast<int>(d[0]), .stride = stride};
  const auto cin = static_cast<std::size_t>(conv.cin);
  const auto cout = static_cast<std::size_t>(conv.cout);
  const std::vector<float>& from = weight->second->floats;
  if (from.size() != cin * cout * 9 || bias->second->floats.size() != cout)
    return missing(std::format("{} : taille des poids ou du biais", name));
  conv.weights.resize(from.size());
  for (std::size_t o = 0; o < cout; ++o)
    for (std::size_t c = 0; c < cin; ++c)
      for (std::size_t t = 0; t < 9; ++t) conv.weights[(o * 9 + t) * cin + c] = from[(o * cin + c) * 9 + t];
  conv.bias = bias->second->floats;
  return conv;
}

// La pente du PReLU exporté dont le Mul s'appelle node : sa constante d'un seul élément.
std::expected<float, Refusal> read_slope(const Model& model, const std::string& node) {
  const auto found = model.nodes.find(node);
  if (found == model.nodes.end() || found->second->op_type != "Mul") return missing("PReLU " + node);
  for (const std::string& input : found->second->inputs) {
    const auto constant = model.constants.find(input);
    if (constant != model.constants.end() && constant->second->floats.size() == 1)
      return constant->second->floats[0];
  }
  return missing("pente de " + node);
}

// Le tenseur nouveau de level, aux canaux de la convolution (rembourrés à 16, ou à 8 pour une petite sortie).
int add_tensor(GmfssGrid& grid, int level, int channels) {
  grid.tensors.push_back({.level = level, .channels = channels});
  return static_cast<int>(grid.tensors.size()) - 1;
}

int padded(int channels) {
  return channels <= 8 ? 8 : (channels + 15) / 16 * 16;
}

// FeatureNet : trois blocs PReLU, convolution de pas 2, PReLU, convolution. Le second PReLU passe en sortie
// de la première convolution, le premier à la lecture.
std::expected<GmfssGrid, Refusal> features(const Model& model) {
  GmfssGrid grid;
  add_tensor(grid, kGmfssFullLevel, 16);
  constexpr std::array<int, kGmfssFeatureLevels> kChannels{64, 128, 192};
  for (int level = 0; level < kGmfssFeatureLevels; ++level)
    add_tensor(grid, level, kChannels[static_cast<std::size_t>(level)]);
  int from = kGmfssFeatureImage;
  for (int b = 1; b <= kGmfssFeatureLevels; ++b) {
    auto s0 = read_slope(model, std::format("/feat_ext/block{0}/block{0}.0/Mul", b));
    auto s2 = read_slope(model, std::format("/feat_ext/block{0}/block{0}.2/Mul", b));
    auto first = read_conv(model, std::format("feat_ext.block{}.1", b), 2);
    auto second = read_conv(model, std::format("feat_ext.block{}.3", b), 1);
    if (!s0) return std::unexpected(std::move(s0.error()));
    if (!s2) return std::unexpected(std::move(s2.error()));
    if (!first) return std::unexpected(std::move(first.error()));
    if (!second) return std::unexpected(std::move(second.error()));
    first->in_slope = *s0;
    first->out_slope = *s2;
    const int temp = add_tensor(grid, b - 1, padded(first->cout));
    grid.steps.push_back({.conv = std::move(*first), .from = from, .to = temp});
    grid.steps.push_back({.conv = std::move(*second), .from = temp, .to = b});
    from = b;
  }
  return grid;
}

// MetricNet : metric_in, trois résiduels (PReLU, convolution, plus l'entrée du bloc), PReLU et metric_out. La
// queue tanh × 10 est celle du noyau qui écrit la métrique dans les mouvements.
std::expected<GmfssGrid, Refusal> metric(const Model& model) {
  GmfssGrid grid;
  add_tensor(grid, 0, 16);
  const int out = add_tensor(grid, 0, 8);
  auto head = read_conv(model, "metricnet.metric_in", 1);
  if (!head) return std::unexpected(std::move(head.error()));
  int from = add_tensor(grid, 0, padded(head->cout));
  grid.steps.push_back({.conv = std::move(*head), .from = kGmfssMetricInput, .to = from});
  for (int i = 1; i <= 3; ++i) {
    auto slope = read_slope(model, std::format("/metricnet/metric_net{0}/metric_net{0}.0/Mul", i));
    auto conv = read_conv(model, std::format("metricnet.metric_net{}.1", i), 1);
    if (!slope) return std::unexpected(std::move(slope.error()));
    if (!conv) return std::unexpected(std::move(conv.error()));
    conv->in_slope = *slope;
    const int to = add_tensor(grid, 0, padded(conv->cout));
    grid.steps.push_back({.conv = std::move(*conv), .from = from, .skip = from, .to = to});
    from = to;
  }
  auto slope = read_slope(model, "/metricnet/metric_out/metric_out.0/Mul");
  auto last = read_conv(model, "metricnet.metric_out.1", 1);
  if (!slope) return std::unexpected(std::move(slope.error()));
  if (!last) return std::unexpected(std::move(last.error()));
  if (last->cout != 2) return missing("metric_out : deux sorties attendues");
  last->in_slope = *slope;
  grid.steps.push_back({.conv = std::move(*last), .from = from, .to = out});
  return grid;
}

int level_after(int level, int stride) {
  if (stride == 1) return level;
  return level == kGmfssFullLevel ? 0 : level + 1;
}

std::expected<void, Refusal> check_step(const GmfssGrid& grid, const GmfssGridStep& step, bool image) {
  const GmfssGridTensor& in = grid.tensors[static_cast<std::size_t>(step.from)];
  const GmfssGridTensor& out = grid.tensors[static_cast<std::size_t>(step.to)];
  const GmfssGridConv& conv = step.conv;
  if (conv.transposed || step.shuffle || (conv.stride != 1 && conv.stride != 2))
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "3 × 3 de pas 1 ou 2 attendue"});
  if (image && (conv.stride != 2 || conv.cin > 3))
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "l'image : 3 canaux, pas 2"});
  if ((conv.cin + 15) / 16 * 16 != in.channels)
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "canaux d'entrée"});
  if (level_after(in.level, conv.stride) != out.level)
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "taille de sortie"});
  if (conv.cout > out.channels || out.channels > (conv.cout + 15) / 16 * 16 || out.channels % 8 != 0)
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "canaux de sortie"});
  if (step.skip >= 0) {
    const GmfssGridTensor& skip = grid.tensors[static_cast<std::size_t>(step.skip)];
    if (skip.level != out.level || skip.channels != out.channels)
      return std::unexpected(Refusal{RefusalCode::FieldMissing, "résiduel"});
  }
  if (conv.weights.size() != static_cast<std::size_t>(conv.cout) * static_cast<std::size_t>(conv.cin) * 9U ||
      conv.bias.size() != static_cast<std::size_t>(conv.cout))
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "poids"});
  return {};
}

// Un réseau dont le tenseur 0 est l'entrée (image : trois plans) et les tenseurs 1 à outputs les sorties.
std::expected<void, Refusal> check_net(const GmfssGrid& grid, std::string_view what, int outputs,
                                       bool image) {
  const auto count = static_cast<int>(grid.tensors.size());
  if (count <= outputs || grid.steps.empty()) return missing(std::format("{} : réseau vide", what));
  std::vector<bool> written(grid.tensors.size(), false);
  written[0] = true;
  for (std::size_t s = 0; s < grid.steps.size(); ++s) {
    const GmfssGridStep& step = grid.steps[s];
    const auto bad = [&](std::string_view why) {
      return missing(std::format("{}, convolution {} : {}", what, s, why));
    };
    if (step.from < 0 || step.from >= count || step.to < 1 || step.to >= count || step.skip >= count ||
        step.to == step.from || step.to == step.skip)
      return bad("tenseurs");
    if (!written[static_cast<std::size_t>(step.from)] ||
        (step.skip >= 0 && !written[static_cast<std::size_t>(step.skip)]))
      return bad("lit un tenseur pas encore écrit");
    if (step.skip == 0) return bad("l'entrée n'est pas un résiduel");
    const bool reads_image = image && step.from == 0;
    if (auto ok = check_step(grid, step, reads_image); !ok) return bad(ok.error().detail);
    written[static_cast<std::size_t>(step.to)] = true;
  }
  for (int i = 1; i <= outputs; ++i) {
    if (!written[static_cast<std::size_t>(i)])
      return missing(std::format("{} : sortie {} pas écrite", what, i));
  }
  return {};
}

}  // namespace

std::expected<GmfssPairNets, Refusal> gmfss_pair_nets(std::span<const std::byte> pair) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, parsed.error().detail});
  Model model;
  for (const onnx::Tensor& tensor : parsed->graph.initializers) model.constants.emplace(tensor.name, &tensor);
  for (const onnx::Node& node : parsed->graph.nodes) {
    model.nodes.emplace(node.name, &node);
    if (node.op_type != "Constant" || node.outputs.size() != 1) continue;
    for (const onnx::Attribute& attribute : node.attributes) {
      if (attribute.name == "value" && attribute.tensors.size() == 1)
        model.constants.emplace(node.outputs[0], attribute.tensors.data());
    }
  }
  auto feature_net = features(model);
  if (!feature_net) return std::unexpected(std::move(feature_net.error()));
  auto metric_net = metric(model);
  if (!metric_net) return std::unexpected(std::move(metric_net.error()));
  GmfssPairNets nets{.features = std::move(*feature_net), .metric = std::move(*metric_net)};
  if (auto ok = gmfss_pair_check(nets); !ok) return std::unexpected(std::move(ok.error()));
  return nets;
}

std::expected<void, Refusal> gmfss_pair_check(const GmfssPairNets& nets) {
  const auto image = nets.features.tensors.empty() ? GmfssGridTensor{} : nets.features.tensors.front();
  if (image.level != kGmfssFullLevel || image.channels != 16) return missing("FeatureNet : l'image d'entrée");
  if (auto ok = check_net(nets.features, "FeatureNet", kGmfssFeatureLevels, true); !ok) return ok;
  for (int level = 0; level < kGmfssFeatureLevels; ++level) {
    if (nets.features.tensors[static_cast<std::size_t>(level) + 1].level != level)
      return missing(std::format("FeatureNet : sortie {} hors de son niveau", level + 1));
  }
  const auto input = nets.metric.tensors.empty() ? GmfssGridTensor{} : nets.metric.tensors.front();
  if (input.level != 0 || input.channels != 16) return missing("MetricNet : l'entrée");
  if (auto ok = check_net(nets.metric, "MetricNet", kGmfssMetricOutput, false); !ok) return ok;
  const GmfssGridTensor& out = nets.metric.tensors[kGmfssMetricOutput];
  if (out.level != 0 || out.channels != 8) return missing("MetricNet : la sortie, 8 canaux attendus");
  return {};
}

}  // namespace kirari::prepare
