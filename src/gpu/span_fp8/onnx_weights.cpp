#include "gpu/span_fp8/onnx_weights.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

using ir::onnx::Tensor;

// Les SPAN du lecteur pèsent quelques mégaoctets en FP32 ; au-delà, ce n'en est pas un.
constexpr std::uintmax_t kMaxOnnxBytes = std::uintmax_t{256} << 20U;

std::unexpected<Refusal> refuse(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FormatUnsupported, std::move(detail)});
}

std::string shape_text(const std::vector<std::int64_t>& dims) {
  std::string out = "(";
  for (std::size_t i = 0; i < dims.size(); ++i) out += std::format("{}{}", i == 0 ? "" : ", ", dims[i]);
  return out + ")";
}

// La forme attendue, et des valeurs flottantes en nombre : un initialiseur entier de bonne forme a ses
// valeurs dans ints, krsc_padded et flat_padded liraient floats hors bornes (A02-06).
std::expected<void, Refusal> check_weight(const Tensor& w, const std::string& weight,
                                          const std::vector<std::int64_t>& shape) {
  if (w.dims != shape)
    return refuse(std::format("{} : {}, {} attendu", weight, shape_text(w.dims), shape_text(shape)));
  std::size_t count = 1;
  for (const std::int64_t d : shape) count *= static_cast<std::size_t>(d);
  if ((w.data_type != ir::onnx::kFloat && w.data_type != ir::onnx::kFloat16) || w.floats.size() != count)
    return refuse(std::format("{} : {} valeurs flottantes attendues", weight, count));
  return {};
}

// (sortie, entrée, ligne, colonne) -> (sortie, ligne, colonne, entrée) en FP16, sorties complétées de zéros
// jusqu'à out_to et entrées jusqu'à in_to (0 : pas de complément). Étape 3.2b : un modèle à 48 canaux rangé
// en 64.
std::vector<std::uint16_t> krsc_padded(const Tensor& w, std::int64_t out_to, std::int64_t in_to) {
  const auto o = static_cast<std::size_t>(w.dims[0]), i = static_cast<std::size_t>(w.dims[1]);
  const auto r = static_cast<std::size_t>(w.dims[2]), s = static_cast<std::size_t>(w.dims[3]);
  const std::size_t outs = std::max(o, static_cast<std::size_t>(out_to));
  const std::size_t padded = std::max(i, static_cast<std::size_t>(in_to));
  std::vector<std::uint16_t> out(outs * r * s * padded, 0);
  for (std::size_t oo = 0; oo < o; ++oo)
    for (std::size_t ii = 0; ii < i; ++ii)
      for (std::size_t rr = 0; rr < r; ++rr)
        for (std::size_t ss = 0; ss < s; ++ss)
          out[((oo * r + rr) * s + ss) * padded + ii] =
              kernels::ref::float_to_half(w.floats[((oo * i + ii) * r + rr) * s + ss]);
  return out;
}

// Une convolution 1×1 : (sortie, entrée, 1, 1) -> (sortie, entrée), complétée de zéros à out_to × in_to.
// group : les entrées viennent par groupes de dims[1] / groups canaux réels (le Concat de conv_cat : quatre
// tenseurs de C canaux), chaque groupe rangé sur in_to / groups places (la colonne s·C + j va en s·64 + j).
std::vector<std::uint16_t> flat_padded(const Tensor& w, std::size_t out_to, std::size_t in_to,
                                       std::size_t groups = 1) {
  const auto o = static_cast<std::size_t>(w.dims[0]), i = static_cast<std::size_t>(w.dims[1]);
  const std::size_t outs = std::max(o, out_to), ins = std::max(i, in_to);
  const std::size_t real_group = i / groups, padded_group = ins / groups;
  std::vector<std::uint16_t> out(outs * ins, 0);
  for (std::size_t oo = 0; oo < o; ++oo)
    for (std::size_t ii = 0; ii < i; ++ii)
      out[oo * ins + (ii / real_group) * padded_group + ii % real_group] =
          kernels::ref::float_to_half(w.floats[oo * i + ii]);
  return out;
}

std::vector<float> bias_padded(const std::vector<float>& bias, std::size_t to) {
  std::vector<float> out = bias;
  if (out.size() < to) out.resize(to, 0.0F);
  return out;
}

const ir::onnx::Attribute* attribute(const ir::onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &ir::onnx::Attribute::name);
  return found == node.attributes.end() ? nullptr : &*found;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<Package, Refusal> package_from_onnx(const ir::onnx::Model& model) {
  const ir::onnx::Graph& graph = model.graph;
  if (graph.inputs.size() != 1 || graph.outputs.size() != 1)
    return refuse("un SPAN a une entrée et une sortie");
  std::map<std::string, const Tensor*> init;
  for (const Tensor& tensor : graph.initializers) init[tensor.name] = &tensor;

  const auto find = [&](const std::string& name) -> const Tensor* {
    const auto it = init.find(name);
    return it == init.end() ? nullptr : it->second;
  };
  const Tensor* first = find("conv_1.1.eval_conv.weight");
  if (first == nullptr)
    return refuse("pas un SPAN AnimeJaNai « unshuffle » (conv_1.1.eval_conv.weight absent)");
  // 64 canaux (b8f64) ou 48 (b5f48, rangé en 64 avec des zéros : étape 3.2b).
  const std::int64_t channels = first->dims.size() == 4 ? first->dims[0] : 0;
  if ((channels != kChannels && channels != 48) ||
      first->dims != std::vector<std::int64_t>{channels, kUnshuffled, 3, 3})
    return refuse(std::format("conv_1 de forme {} : seuls les SPAN à 48 ou {} canaux sont pris",
                              shape_text(first->dims), kChannels));
  std::set<int> blocks;
  for (const Tensor& tensor : graph.initializers) {
    if (!tensor.name.starts_with("block_")) continue;
    int number = 0;
    const char* begin = tensor.name.c_str() + 6;
    const char* last = tensor.name.c_str() + tensor.name.size();
    const auto [end, error] = std::from_chars(begin, last, number);
    if (error == std::errc{} && end != last && *end == '.') blocks.insert(number);
  }
  const int block_count = static_cast<int>(blocks.size());
  if (std::clamp(block_count, kMinBlocks, kBlocks) != block_count || *blocks.begin() != 1 ||
      *blocks.rbegin() != block_count)
    return refuse(std::format("{} blocs, {} à {} attendus (depuis 1)", block_count, kMinBlocks, kBlocks));

  // shape : la forme du modèle ; out_to, in_to : celle du paquet (64 canaux) ; groups : voir flat_padded.
  const auto conv = [&](const std::string& weight, const std::string& bias,
                        const std::vector<std::int64_t>& shape, Conv& out, bool three_by_three,
                        std::int64_t out_to, std::int64_t in_to,
                        std::size_t groups = 1) -> std::expected<void, Refusal> {
    const Tensor* w = find(weight);
    const Tensor* b = find(bias);
    if (w == nullptr) return refuse(std::format("{} : absent, {} attendu", weight, shape_text(shape)));
    if (auto ok = check_weight(*w, weight, shape); !ok) return ok;
    if (b == nullptr || b->floats.size() != static_cast<std::size_t>(shape[0]))
      return refuse(std::format("{} : biais absent ou de mauvaise taille", bias));
    out.weights = three_by_three ? krsc_padded(*w, out_to, in_to)
                                 : flat_padded(*w, static_cast<std::size_t>(out_to),
                                               static_cast<std::size_t>(in_to), groups);
    out.bias = bias_padded(b->floats, static_cast<std::size_t>(std::max(out_to, shape[0])));
    return {};
  };

  Package package;
  package.scales.fill(1.0F);
  package.blocks.resize(static_cast<std::size_t>(block_count));
  package.real_channels = static_cast<int>(channels);
  const std::int64_t c = channels;
  if (auto ok = conv("conv_1.1.eval_conv.weight", "conv_1.1.eval_conv.bias", {c, kUnshuffled, 3, 3},
                     package.first, true, kChannels, kUnshuffledPadded);
      !ok)
    return std::unexpected(std::move(ok.error()));
  for (int k = 1; k <= block_count; ++k) {
    Block& block = package.blocks[static_cast<std::size_t>(k - 1)];
    constexpr std::array<const char*, 3> kNames{"c1_r", "c2_r", "c3_r"};
    for (std::size_t i = 0; i < kNames.size(); ++i) {
      const std::string base = std::format("block_{}.{}.eval_conv", k, kNames[i]);
      if (auto ok =
              conv(base + ".weight", base + ".bias", {c, c, 3, 3}, block.conv[i], true, kChannels, kChannels);
          !ok)
        return std::unexpected(std::move(ok.error()));
    }
    const std::string att = std::format("block_{}.att_conv", k);
    if (auto ok =
            conv(att + ".weight", att + ".bias", {c, c, 1, 1}, block.attention, false, kChannels, kChannels);
        !ok)
      return std::unexpected(std::move(ok.error()));
  }
  if (auto ok = conv("conv_cat.weight", "conv_cat.bias", {c, 4 * c, 1, 1}, package.concat, false, kChannels,
                     kConcatChannels, 4);
      !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = conv("to_pixel.eval_conv.weight", "to_pixel.eval_conv.bias", {kPixelChannels, c, 3, 3},
                     package.pixel, true, kPixelChannels, kChannels);
      !ok)
    return std::unexpected(std::move(ok.error()));

  // Le graphe : les tenseurs que les noyaux quantifient existent, et le dégroupage et DepthToSpace sont ceux
  // que les noyaux supposent (canal = c·4 + a·2 + b ; DepthToSpace(4) en mode CRD).
  std::set<std::string> produced;
  std::map<std::string, const ir::onnx::Node*> producer;
  for (const ir::onnx::Node& node : graph.nodes)
    for (const std::string& out : node.outputs) {
      produced.insert(out);
      producer[out] = &node;
    }
  std::vector<std::string> needed{"/conv_1/conv_1.1/eval_conv/Conv_output_0"};
  for (int k = 1; k <= block_count; ++k) {
    needed.push_back(std::format("/block_{}/act1/Relu_output_0", k));
    needed.push_back(std::format("/block_{}/act1_1/Relu_output_0", k));
    needed.push_back(std::format("/block_{}/Mul_output_0", k));
  }
  needed.emplace_back("/conv_1/conv_1.0/Reshape_1_output_0");
  for (const std::string& name : needed)
    if (!produced.contains(name))
      return refuse(std::format(
          "tenseur {} absent du graphe : ce SPAN n'a pas la forme que les noyaux supposent", name));
  const ir::onnx::Node* reshape = producer["/conv_1/conv_1.0/Reshape_1_output_0"];
  const ir::onnx::Node* transpose = nullptr;
  if (reshape->op_type == "Reshape" && !reshape->inputs.empty() && producer.contains(reshape->inputs[0]))
    transpose = producer[reshape->inputs[0]];
  const ir::onnx::Attribute* perm =
      transpose != nullptr && transpose->op_type == "Transpose" ? attribute(*transpose, "perm") : nullptr;
  if (perm == nullptr || perm->ints != std::vector<std::int64_t>{0, 1, 3, 5, 2, 4})
    return refuse("dégroupage : ce SPAN n'a pas l'ordre des canaux supposé par les noyaux");
  // L'ordre des quatre entrées du Concat de conv_cat (étape 3.2a) : les noyaux écrivent la tranche 0 depuis
  // conv_1, la 1 depuis la sortie du dernier bloc, la 2 depuis celle du bloc 1 et la 3 depuis le premier Relu
  // du dernier bloc. Une entrée lue à travers des Identity.
  const auto through_identity = [&](std::string name) {
    for (int hops = 0; hops < 8; ++hops) {
      const auto it = producer.find(name);
      if (it == producer.end() || it->second->op_type != "Identity" || it->second->inputs.empty()) break;
      name = it->second->inputs[0];
    }
    return name;
  };
  const auto cat_conv = std::ranges::find_if(graph.nodes, [](const ir::onnx::Node& node) {
    return node.op_type == "Conv" && std::ranges::find(node.inputs, "conv_cat.weight") != node.inputs.end();
  });
  if (cat_conv == graph.nodes.end() || cat_conv->inputs.empty())
    return refuse("conv_cat : la convolution qui lit conv_cat.weight est absente du graphe");
  const std::string cat_input = through_identity(cat_conv->inputs[0]);
  const ir::onnx::Node* concat = producer.contains(cat_input) ? producer[cat_input] : nullptr;
  if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != 4)
    return refuse("conv_cat : son entrée n'est pas un Concat de quatre tenseurs");
  const std::array<std::string, 4> expected_order{
      "/conv_1/conv_1.1/eval_conv/Conv_output_0", std::format("/block_{}/Mul_output_0", block_count),
      "/block_1/Mul_output_0", std::format("/block_{}/act1/Relu_output_0", block_count)};
  for (std::size_t i = 0; i < expected_order.size(); ++i) {
    const std::string got = through_identity(concat->inputs[i]);
    if (got != expected_order[i])
      return refuse(
          std::format("Concat de conv_cat : entrée {} « {} », « {} » attendue (ordre des tranches des "
                      "noyaux)",
                      i, got, expected_order[i]));
  }
  const auto depth_to_space = std::ranges::find(graph.nodes, "DepthToSpace", &ir::onnx::Node::op_type);
  if (depth_to_space == graph.nodes.end()) return refuse("DepthToSpace absent du graphe");
  const ir::onnx::Attribute* block_size = attribute(*depth_to_space, "blocksize");
  const ir::onnx::Attribute* mode = attribute(*depth_to_space, "mode");
  if (block_size == nullptr || block_size->i != 4 || mode == nullptr || mode->s != "CRD")
    return refuse("DepthToSpace : blocksize 4 en mode CRD attendu");
  return package;
}

std::expected<Package, Refusal> package_from_onnx_file(const std::filesystem::path& path) {
  // Un fichier ordinaire, de taille connue et plafonnée : un dossier ou un fichier spécial ne fait pas
  // lever std::length_error hors de std::expected (tellg à -1), un fichier énorme n'est pas lu (A02-07).
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::is_regular_file(path, error) ? std::filesystem::file_size(path, error) : 0;
  if (error || size == 0 || size > kMaxOnnxBytes) {
    return std::unexpected(
        Refusal{RefusalCode::FileUnreadable, "modèle SPAN illisible : " + core::to_utf8(path)});
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size))
    return std::unexpected(
        Refusal{RefusalCode::FileUnreadable, "modèle SPAN tronqué : " + core::to_utf8(path)});
  auto model = ir::onnx::read_onnx(bytes);
  if (!model) return std::unexpected(std::move(model.error()));
  return package_from_onnx(*model);
}

}  // namespace ajp::gpu::span_fp8
