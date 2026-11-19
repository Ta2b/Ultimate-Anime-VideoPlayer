#include "prepare/gmfss_native.h"

#include <algorithm>
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
      Refusal{RefusalCode::FieldMissing, "GMFSS natif : fusion non reconnue : " + std::move(detail)});
}

// Les tenseurs nommés du GridNet : les entrées, puis les sommes de chaque ligne (64 canaux à la taille de la
// fusion, 128 à la moitié, 192 au quart), dans l'ordre où le graphe exporté les écrit.
enum Named : int {
  kX0,
  kX1,
  kX2,
  kX3,
  kP,  // ligne 0
  kA,
  kB,
  kJ,
  kJ2,
  kL,
  kL2,
  kQ,  // ligne 1
  kD,
  kF,
  kF2,
  kI,
  kI2,
  kK,
  kK2,
  kR,  // ligne 2
  kE,
  kG,
  kG2,
  kH8,
  kH9,
  kNamedCount,
};

GmfssGridTensor named_tensor(int id) {
  switch (id) {
    case kX0:
      return {.level = 0, .channels = 16};
    case kX1:
      return {.level = 0, .channels = 2 * kGmfssFeatureChannels};
    case kX2:
      return {.level = 1, .channels = 256};
    case kX3:
      return {.level = 2, .channels = 384};
    default:
      break;
  }
  if (id < kQ) return {.level = 0, .channels = 64};
  if (id < kR) return {.level = 1, .channels = 128};
  return {.level = 2, .channels = 192};
}

// Un bloc : PReLU, convolution (pas 1, pas 2, ou transposée de pas 2), PReLU, convolution 3 × 3 ; la sortie
// plus skip (-1 : rien).
enum class Kind : std::uint8_t { Same, Down, Up };
struct Block {
  std::string_view module;
  Kind kind = Kind::Same;
  int from = 0;
  int skip = -1;
  int to = 0;
};

// FusionNet_u de GMFSS Fortuna Union, colonne par colonne : têtes et descente, puis la montée. Une somme de
// deux blocs (A = head0 + head1) s'écrit en deux : le premier bloc écrit seul, le second l'ajoute.
constexpr std::array<Block, 21> kBlocks{{
    {"residual_model_head1", Kind::Same, kX1, -1, kP}, {"residual_model_head0", Kind::Same, kX0, kP, kA},
    {"residual_model_01", Kind::Same, kA, kA, kB},     {"residual_model_head2", Kind::Same, kX2, -1, kQ},
    {"downsample_model_10", Kind::Down, kA, kQ, kD},   {"residual_model_head3", Kind::Same, kX3, -1, kR},
    {"downsample_model_20", Kind::Down, kD, kR, kE},   {"residual_model_11", Kind::Same, kD, kD, kF},
    {"downsample_model_11", Kind::Down, kB, kF, kF2},  {"residual_model_21", Kind::Same, kE, kE, kG},
    {"downsample_model_21", Kind::Down, kF2, kG, kG2}, {"residual_model_24", Kind::Same, kG2, kG2, kH8},
    {"residual_model_25", Kind::Same, kH8, kH8, kH9},  {"residual_model_14", Kind::Same, kF2, kF2, kI},
    {"upsample_model_14", Kind::Up, kH8, kI, kI2},     {"residual_model_04", Kind::Same, kB, kB, kJ},
    {"upsample_model_04", Kind::Up, kI2, kJ, kJ2},     {"residual_model_15", Kind::Same, kI2, kI2, kK},
    {"upsample_model_15", Kind::Up, kH9, kK, kK2},     {"residual_model_05", Kind::Same, kJ2, kJ2, kL},
    {"upsample_model_05", Kind::Up, kK2, kL, kL2},
}};

constexpr std::string_view kWeights = "gmfss.fusionnet.";

struct Model {
  std::map<std::string, const onnx::Tensor*, std::less<>> constants;
  std::map<std::string, const onnx::Node*, std::less<>> nodes;
};

// Les poids ONNX ([cout][cin][k][k], ou [cin][cout][k][k] pour la transposée) rangés [cout][k²][cin].
std::vector<float> reorder(const std::vector<float>& from, const GmfssGridConv& conv, int kernel) {
  const auto taps = static_cast<std::size_t>(kernel) * static_cast<std::size_t>(kernel);
  const auto cin = static_cast<std::size_t>(conv.cin);
  const auto cout = static_cast<std::size_t>(conv.cout);
  std::vector<float> out(from.size());
  for (std::size_t o = 0; o < cout; ++o)
    for (std::size_t c = 0; c < cin; ++c)
      for (std::size_t t = 0; t < taps; ++t) {
        const std::size_t at = conv.transposed ? (c * cout + o) * taps + t : (o * cin + c) * taps + t;
        out[(o * taps + t) * cin + c] = from[at];
      }
  return out;
}

// La convolution « gmfss.fusionnet.<name> » : poids et biais.
std::expected<GmfssGridConv, Refusal> read_conv(const Model& model, std::string_view name, bool transposed,
                                                int stride) {
  const auto weight = model.constants.find(std::format("{}{}.weight", kWeights, name));
  const auto bias = model.constants.find(std::format("{}{}.bias", kWeights, name));
  if (weight == model.constants.end() || bias == model.constants.end())
    return missing(std::format("poids ou biais de {}", name));
  const std::vector<std::int64_t>& d = weight->second->dims;
  const int kernel = transposed ? 4 : 3;
  if (d.size() != 4 || d[2] != kernel || d[3] != kernel)
    return missing(std::format("{0} : poids {1}×{1} attendus", name, kernel));
  GmfssGridConv conv{.transposed = transposed,
                     .cin = static_cast<int>(transposed ? d[0] : d[1]),
                     .cout = static_cast<int>(transposed ? d[1] : d[0]),
                     .stride = transposed ? 2 : stride};
  const auto count = static_cast<std::size_t>(conv.cin) * static_cast<std::size_t>(conv.cout) *
                     static_cast<std::size_t>(kernel * kernel);
  if (weight->second->floats.size() != count ||
      bias->second->floats.size() != static_cast<std::size_t>(conv.cout))
    return missing(std::format("{} : taille des poids ou du biais", name));
  conv.weights = reorder(weight->second->floats, conv, kernel);
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

std::expected<void, Refusal> add_block(const Model& model, const Block& block, GmfssGrid& grid) {
  const std::string m(block.module);
  auto s1 = read_slope(model, std::format("/fusionnet/{0}/{0}.0/Mul", m));
  auto s2 = read_slope(model, std::format("/fusionnet/{0}/{0}.2/Mul", m));
  auto first = read_conv(model, m + ".1", block.kind == Kind::Up, block.kind == Kind::Down ? 2 : 1);
  auto second = read_conv(model, m + ".3", false, 1);
  if (!s1) return std::unexpected(std::move(s1.error()));
  if (!s2) return std::unexpected(std::move(s2.error()));
  if (!first) return std::unexpected(std::move(first.error()));
  if (!second) return std::unexpected(std::move(second.error()));
  // La sortie de la première convolution, rangée à la taille de la sortie du bloc.
  const int temp = static_cast<int>(grid.tensors.size());
  grid.tensors.push_back({.level = named_tensor(block.to).level, .channels = (first->cout + 15) / 16 * 16});
  first->in_slope = *s1;
  // Après une transposée, écrite par phases sans activation, le PReLU passe à la lecture suivante.
  if (block.kind == Kind::Up)
    second->in_slope = *s2;
  else
    first->out_slope = *s2;
  grid.steps.push_back({.conv = std::move(*first), .from = block.from, .to = temp});
  grid.steps.push_back({.conv = std::move(*second), .from = temp, .skip = block.skip, .to = block.to});
  return {};
}

// La queue : convolution 64 → 64 et PReLU, convolution 64 → 256 et PixelShuffle(2), convolution 64 → 3.
std::expected<void, Refusal> add_tail(const Model& model, GmfssGrid& grid) {
  constexpr std::string_view kTail = "residual_model_tail";
  auto slope =
      read_slope(model, std::format("/fusionnet/{0}/conv_before_upsample/conv_before_upsample.1/Mul", kTail));
  auto before = read_conv(model, std::format("{}.conv_before_upsample.0", kTail), false, 1);
  auto up = read_conv(model, std::format("{}.upsample.0", kTail), false, 1);
  auto last = read_conv(model, std::format("{}.conv_last", kTail), false, 1);
  if (!slope) return std::unexpected(std::move(slope.error()));
  if (!before) return std::unexpected(std::move(before.error()));
  if (!up) return std::unexpected(std::move(up.error()));
  if (!last) return std::unexpected(std::move(last.error()));
  if (up->cout % 4 != 0) return missing("PixelShuffle : canaux de la convolution");
  before->out_slope = *slope;
  const int t = static_cast<int>(grid.tensors.size());
  grid.tensors.push_back({.level = 0, .channels = (before->cout + 15) / 16 * 16});
  grid.tensors.push_back({.level = kGmfssFullLevel, .channels = (up->cout / 4 + 15) / 16 * 16});
  grid.tensors.push_back({.level = kGmfssFullLevel, .channels = (last->cout + 7) / 8 * 8});
  grid.steps.push_back({.conv = std::move(*before), .from = kL2, .to = t});
  grid.steps.push_back({.conv = std::move(*up), .from = t, .to = t + 1, .shuffle = true});
  grid.steps.push_back({.conv = std::move(*last), .from = t + 1, .to = t + 2});
  return {};
}

std::pair<int, int> level_of(const GmfssGrid& grid, int id) {
  const GmfssGridTensor& t = grid.tensors[static_cast<std::size_t>(id)];
  return {t.level, t.channels};
}

// La taille que la convolution écrit, depuis celle qu'elle lit.
int written_level(const GmfssGridStep& step, int from) {
  if (step.conv.transposed || step.shuffle) return from == 0 ? kGmfssFullLevel : from - 1;
  return step.conv.stride == 2 ? from + 1 : from;
}

// Ce qu'une convolution lit et écrit : des tenseurs du réseau, déjà écrits, la sortie distincte des
// lectures ; rien : bon.
const char* read_problem(const GmfssGrid& grid, const std::vector<bool>& written, const GmfssGridStep& step) {
  const auto count = static_cast<int>(grid.tensors.size());
  if (step.from < 0 || step.from >= count || step.to < kGmfssGridInputs || step.to >= count ||
      step.skip >= count || step.to == step.from || step.to == step.skip)
    return "tenseurs";
  if (!written[static_cast<std::size_t>(step.from)] ||
      (step.skip >= 0 && !written[static_cast<std::size_t>(step.skip)]))
    return "lit un tenseur pas encore écrit";
  return nullptr;
}

// Canaux, tailles, résiduel et poids d'une convolution ; rien : bon.
const char* shape_problem(const GmfssGrid& grid, const GmfssGridStep& step) {
  const auto [in_level, in_channels] = level_of(grid, step.from);
  const auto [out_level, out_channels] = level_of(grid, step.to);
  const GmfssGridConv& conv = step.conv;
  const bool phases = step.shuffle || conv.transposed;
  if ((conv.cin + 15) / 16 * 16 != in_channels ||
      (in_level == kGmfssFullLevel && (conv.stride != 1 || phases)))
    return "canaux d'entrée";
  if (written_level(step, in_level) != out_level) return "taille de sortie";
  const int cout = step.shuffle ? conv.cout / 4 : conv.cout;
  if ((step.shuffle && conv.cout % 4 != 0) || cout > out_channels || out_channels > (cout + 15) / 16 * 16)
    return "canaux de sortie";
  if (step.skip >= 0 && level_of(grid, step.skip) != level_of(grid, step.to)) return "résiduel";
  if (phases && (step.skip >= 0 || conv.out_slope != 1.0F))
    return "une sortie par phases n'a ni résiduel ni activation";
  const std::size_t taps = conv.transposed ? 16U : 9U;
  if (conv.weights.size() !=
          static_cast<std::size_t>(conv.cout) * static_cast<std::size_t>(conv.cin) * taps ||
      conv.bias.size() != static_cast<std::size_t>(conv.cout))
    return "poids";
  return nullptr;
}

}  // namespace

std::expected<GmfssGrid, Refusal> gmfss_grid(std::span<const std::byte> fusion) {
  auto parsed = onnx::read_onnx(fusion);
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
  GmfssGrid grid;
  for (int id = 0; id < kNamedCount; ++id) grid.tensors.push_back(named_tensor(id));
  for (const Block& block : kBlocks) {
    if (auto ok = add_block(model, block, grid); !ok) return std::unexpected(std::move(ok.error()));
  }
  if (auto ok = add_tail(model, grid); !ok) return std::unexpected(std::move(ok.error()));
  if (auto ok = gmfss_grid_check(grid); !ok) return std::unexpected(std::move(ok.error()));
  return grid;
}

std::expected<void, Refusal> gmfss_grid_check(const GmfssGrid& grid) {
  std::vector<bool> written(grid.tensors.size(), false);
  for (std::size_t i = 0; i < written.size() && i < static_cast<std::size_t>(kGmfssGridInputs); ++i)
    written[i] = true;
  for (std::size_t s = 0; s < grid.steps.size(); ++s) {
    const GmfssGridStep& step = grid.steps[s];
    const char* why = read_problem(grid, written, step);
    if (why == nullptr) why = shape_problem(grid, step);
    if (why != nullptr) return missing(std::format("convolution {} : {}", s, why));
    written[static_cast<std::size_t>(step.to)] = true;
  }
  if (grid.steps.empty() || grid.tensors.back().level != kGmfssFullLevel ||
      grid.steps.back().to != static_cast<int>(grid.tensors.size()) - 1)
    return missing("la dernière convolution n'écrit pas la sortie");
  return {};
}

}  // namespace kirari::prepare
