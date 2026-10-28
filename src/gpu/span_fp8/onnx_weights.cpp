#include "gpu/span_fp8/onnx_weights.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

using ir::onnx::Tensor;

std::unexpected<Refusal> refuse(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FormatUnsupported, std::move(detail)});
}

std::string shape_text(const std::vector<std::int64_t>& dims) {
  std::string out = "(";
  for (std::size_t i = 0; i < dims.size(); ++i) out += std::format("{}{}", i == 0 ? "" : ", ", dims[i]);
  return out + ")";
}

// (sortie, entrée, ligne, colonne) -> (sortie, ligne, colonne, entrée) en FP16, entrées complétées de zéros
// jusqu'à pad_to (0 : pas de complément).
std::vector<std::uint16_t> krsc(const Tensor& w, std::int64_t pad_to) {
  const auto o = static_cast<std::size_t>(w.dims[0]), i = static_cast<std::size_t>(w.dims[1]);
  const auto r = static_cast<std::size_t>(w.dims[2]), s = static_cast<std::size_t>(w.dims[3]);
  const std::size_t padded = std::max(i, static_cast<std::size_t>(pad_to));
  std::vector<std::uint16_t> out(o * r * s * padded, 0);
  for (std::size_t oo = 0; oo < o; ++oo)
    for (std::size_t ii = 0; ii < i; ++ii)
      for (std::size_t rr = 0; rr < r; ++rr)
        for (std::size_t ss = 0; ss < s; ++ss)
          out[((oo * r + rr) * s + ss) * padded + ii] =
              kernels::ref::float_to_half(w.floats[((oo * i + ii) * r + rr) * s + ss]);
  return out;
}

// Une convolution 1×1 : (sortie, entrée, 1, 1) -> (sortie, entrée).
std::vector<std::uint16_t> flat(const Tensor& w) {
  std::vector<std::uint16_t> out(w.floats.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = kernels::ref::float_to_half(w.floats[i]);
  return out;
}

const ir::onnx::Attribute* attribute(const ir::onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &ir::onnx::Attribute::name);
  return found == node.attributes.end() ? nullptr : &*found;
}

}  // namespace

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
  if (first->dims != std::vector<std::int64_t>{kChannels, kUnshuffled, 3, 3})
    return refuse(
        std::format("conv_1 de forme {} : seul le SPAN à {} canaux est pris (le modèle b5f48 à 48 "
                    "canaux reste sur TensorRT FP16)",
                    shape_text(first->dims), kChannels));
  std::set<int> blocks;
  for (const Tensor& tensor : graph.initializers) {
    if (!tensor.name.starts_with("block_")) continue;
    int number = 0;
    const char* begin = tensor.name.c_str() + 6;
    char* end = nullptr;
    number = static_cast<int>(std::strtol(begin, &end, 10));
    if (end != begin && *end == '.') blocks.insert(number);
  }
  if (blocks.size() != static_cast<std::size_t>(kBlocks) || *blocks.begin() != 1 ||
      *blocks.rbegin() != kBlocks)
    return refuse(std::format("{} blocs trouvés, {} attendus", blocks.size(), kBlocks));

  const auto conv = [&](const std::string& weight, const std::string& bias,
                        const std::vector<std::int64_t>& shape, Conv& out, bool three_by_three,
                        std::int64_t pad) -> std::expected<void, Refusal> {
    const Tensor* w = find(weight);
    const Tensor* b = find(bias);
    if (w == nullptr) return refuse(std::format("{} : absent, {} attendu", weight, shape_text(shape)));
    if (w->dims != shape)
      return refuse(std::format("{} : {}, {} attendu", weight, shape_text(w->dims), shape_text(shape)));
    if (b == nullptr || b->floats.size() != static_cast<std::size_t>(shape[0]))
      return refuse(std::format("{} : biais absent ou de mauvaise taille", bias));
    out.weights = three_by_three ? krsc(*w, pad) : flat(*w);
    out.bias = b->floats;
    return {};
  };

  Package package;
  package.scales.fill(1.0F);
  if (auto ok = conv("conv_1.1.eval_conv.weight", "conv_1.1.eval_conv.bias", {kChannels, kUnshuffled, 3, 3},
                     package.first, true, kUnshuffledPadded);
      !ok)
    return std::unexpected(std::move(ok.error()));
  for (int k = 1; k <= kBlocks; ++k) {
    Block& block = package.blocks[static_cast<std::size_t>(k - 1)];
    constexpr std::array<const char*, 3> kNames{"c1_r", "c2_r", "c3_r"};
    for (std::size_t i = 0; i < kNames.size(); ++i) {
      const std::string base = std::format("block_{}.{}.eval_conv", k, kNames[i]);
      if (auto ok =
              conv(base + ".weight", base + ".bias", {kChannels, kChannels, 3, 3}, block.conv[i], true, 0);
          !ok)
        return std::unexpected(std::move(ok.error()));
    }
    const std::string att = std::format("block_{}.att_conv", k);
    if (auto ok =
            conv(att + ".weight", att + ".bias", {kChannels, kChannels, 1, 1}, block.attention, false, 0);
        !ok)
      return std::unexpected(std::move(ok.error()));
  }
  if (auto ok = conv("conv_cat.weight", "conv_cat.bias", {kChannels, kConcatChannels, 1, 1}, package.concat,
                     false, 0);
      !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = conv("to_pixel.eval_conv.weight", "to_pixel.eval_conv.bias",
                     {kPixelChannels, kChannels, 3, 3}, package.pixel, true, 0);
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
  for (int k = 1; k <= kBlocks; ++k) {
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
  const auto depth_to_space = std::ranges::find(graph.nodes, "DepthToSpace", &ir::onnx::Node::op_type);
  if (depth_to_space == graph.nodes.end()) return refuse("DepthToSpace absent du graphe");
  const ir::onnx::Attribute* block_size = attribute(*depth_to_space, "blocksize");
  const ir::onnx::Attribute* mode = attribute(*depth_to_space, "mode");
  if (block_size == nullptr || block_size->i != 4 || mode == nullptr || mode->s != "CRD")
    return refuse("DepthToSpace : blocksize 4 en mode CRD attendu");
  return package;
}

std::expected<Package, Refusal> package_from_onnx_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return std::unexpected(Refusal{RefusalCode::FileUnreadable, "modèle SPAN illisible : " + path.string()});
  const std::streamsize size = file.tellg();
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!file)
    return std::unexpected(Refusal{RefusalCode::FileUnreadable, "modèle SPAN tronqué : " + path.string()});
  auto model = ir::onnx::read_onnx(bytes);
  if (!model) return std::unexpected(std::move(model.error()));
  return package_from_onnx(*model);
}

}  // namespace ajp::gpu::span_fp8
