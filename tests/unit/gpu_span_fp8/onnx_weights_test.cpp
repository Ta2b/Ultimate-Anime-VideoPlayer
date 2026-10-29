// Les poids du SPAN lus dans son ONNX : refus des autres réseaux, rangement des poids, et (test caché
// [.span_compat]) accord octet pour octet avec le paquet de tools/span/derive_span_fp8.py.
#include "gpu/span_fp8/onnx_weights.h"

#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {
namespace {

using ir::onnx::Attribute;
using ir::onnx::AttributeType;
using ir::onnx::Model;
using ir::onnx::Node;
using ir::onnx::Tensor;

Tensor tensor(std::string name, std::vector<std::int64_t> dims, float fill = 0.25F) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
  return Tensor{.name = std::move(name),
                .data_type = ir::onnx::kFloat16,
                .dims = std::move(dims),
                .floats = std::vector<float>(count, fill)};
}

Node node(std::string op, std::vector<std::string> in, std::string out) {
  return Node{
      .name = op + out, .op_type = std::move(op), .inputs = std::move(in), .outputs = {std::move(out)}};
}

// Un SPAN b8f64 aux poids réguliers, du graphe que les noyaux supposent (les noms, pas les calculs).
Model synthetic_model(int channels = kChannels) {
  Model model;
  auto& g = model.graph;
  g.inputs.push_back({.name = "input"});
  g.outputs.push_back({.name = "output"});
  g.initializers.push_back(tensor("conv_1.1.eval_conv.weight", {channels, 12, 3, 3}));
  g.initializers.push_back(tensor("conv_1.1.eval_conv.bias", {channels}));
  for (int k = 1; k <= kBlocks; ++k) {
    for (const char* n : {"c1_r", "c2_r", "c3_r"}) {
      g.initializers.push_back(tensor(std::format("block_{}.{}.eval_conv.weight", k, n), {64, 64, 3, 3}));
      g.initializers.push_back(tensor(std::format("block_{}.{}.eval_conv.bias", k, n), {64}));
    }
    g.initializers.push_back(tensor(std::format("block_{}.att_conv.weight", k), {64, 64, 1, 1}));
    g.initializers.push_back(tensor(std::format("block_{}.att_conv.bias", k), {64}));
    g.nodes.push_back(node("Relu", {"x"}, std::format("/block_{}/act1/Relu_output_0", k)));
    g.nodes.push_back(node("Relu", {"x"}, std::format("/block_{}/act1_1/Relu_output_0", k)));
    g.nodes.push_back(node("Mul", {"x"}, std::format("/block_{}/Mul_output_0", k)));
  }
  g.initializers.push_back(tensor("conv_cat.weight", {64, 256, 1, 1}));
  g.initializers.push_back(tensor("conv_cat.bias", {64}));
  g.initializers.push_back(tensor("to_pixel.eval_conv.weight", {48, 64, 3, 3}));
  g.initializers.push_back(tensor("to_pixel.eval_conv.bias", {48}));
  g.nodes.push_back(node("Conv", {"x"}, "/conv_1/conv_1.1/eval_conv/Conv_output_0"));
  Node transpose = node("Transpose", {"a"}, "/conv_1/conv_1.0/Transpose_output_0");
  transpose.attributes.push_back({.name = "perm", .type = AttributeType::Ints, .ints = {0, 1, 3, 5, 2, 4}});
  g.nodes.push_back(transpose);
  g.nodes.push_back(
      node("Reshape", {"/conv_1/conv_1.0/Transpose_output_0"}, "/conv_1/conv_1.0/Reshape_1_output_0"));
  Node d2s = node("DepthToSpace", {"y"}, "/upsampler/DepthToSpace_output_0");
  d2s.attributes.push_back({.name = "blocksize", .type = AttributeType::Int, .i = 4});
  d2s.attributes.push_back({.name = "mode", .type = AttributeType::String, .s = "CRD"});
  g.nodes.push_back(d2s);
  return model;
}

std::vector<std::byte> read_file(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

}  // namespace

TEST_CASE("un SPAN b8f64 se range comme le paquet le veut", "[span_fp8][onnx]") {
  Model model = synthetic_model();
  // conv_1 : poids (sortie 1, entrée 2, ligne 0, colonne 1) = 0,5 ; le paquet range
  // [sortie][ligne][colonne][entrée] et complète les entrées à 16.
  model.graph.initializers[0].floats[((1 * 12 + 2) * 3 + 0) * 3 + 1] = 0.5F;
  model.graph.initializers[1].floats[5] = -1.5F;
  const auto package = package_from_onnx(model);
  REQUIRE(package);
  CHECK(package->first.weights.size() == std::size_t{64} * 9 * 16);
  CHECK(package->first.weights[((1 * 3 + 0) * 3 + 1) * 16 + 2] == 0x3800);  // 0,5 en demi-précision
  CHECK(package->first.weights[((1 * 3 + 0) * 3 + 1) * 16 + 12] == 0);      // canal de complément
  CHECK(package->first.weights[0] == 0x3400);                               // 0,25
  CHECK(package->first.bias[5] == -1.5F);
  CHECK(package->blocks[7].conv[2].weights.size() == std::size_t{64} * 9 * 64);
  CHECK(package->blocks[3].attention.weights.size() == std::size_t{64} * 64);
  CHECK(package->concat.weights.size() == std::size_t{64} * 256);
  CHECK(package->pixel.bias.size() == 48U);
  // Le paquet qui en sort est valide pour parse() une fois ses échelles posées.
  Package ready = *package;
  ready.scales.fill(0.05F);
  CHECK(parse(serialize(ready)));
}

TEST_CASE("un autre SPAN que le b8f64 est refusé avant tout calcul", "[span_fp8][onnx]") {
  const auto narrow = package_from_onnx(synthetic_model(48));
  REQUIRE_FALSE(narrow);
  CHECK(narrow.error().code == RefusalCode::FormatUnsupported);
  CHECK(narrow.error().detail.find("64 canaux") != std::string::npos);

  Model not_span;
  not_span.graph.inputs.push_back({.name = "input"});
  not_span.graph.outputs.push_back({.name = "output"});
  CHECK_FALSE(package_from_onnx(not_span));

  Model fewer = synthetic_model();
  std::erase_if(fewer.graph.initializers, [](const Tensor& t) { return t.name.starts_with("block_8."); });
  const auto seven = package_from_onnx(fewer);
  REQUIRE_FALSE(seven);
  CHECK(seven.error().detail.find("7 blocs") != std::string::npos);

  Model wrong_shape = synthetic_model();
  for (Tensor& t : wrong_shape.graph.initializers)
    if (t.name == "block_2.c2_r.eval_conv.weight") t = tensor(t.name, {64, 32, 3, 3});
  CHECK_FALSE(package_from_onnx(wrong_shape));
}

TEST_CASE("un graphe dont le dégroupage ou le DepthToSpace diffèrent est refusé", "[span_fp8][onnx]") {
  Model wrong_perm = synthetic_model();
  for (Node& n : wrong_perm.graph.nodes)
    if (n.op_type == "Transpose") n.attributes[0].ints = {0, 1, 2, 4, 3, 5};
  const auto perm = package_from_onnx(wrong_perm);
  REQUIRE_FALSE(perm);
  CHECK(perm.error().detail.find("dégroupage") != std::string::npos);

  Model wrong_mode = synthetic_model();
  for (Node& n : wrong_mode.graph.nodes)
    if (n.op_type == "DepthToSpace") n.attributes[1].s = "DCR";
  CHECK_FALSE(package_from_onnx(wrong_mode));

  Model missing = synthetic_model();
  std::erase_if(missing.graph.nodes,
                [](const Node& n) { return n.outputs[0] == "/block_5/act1_1/Relu_output_0"; });
  const auto absent = package_from_onnx(missing);
  REQUIRE_FALSE(absent);
  CHECK(absent.error().detail.find("/block_5/act1_1/Relu_output_0") != std::string::npos);
}

TEST_CASE("les poids lus ici sont ceux du paquet de derive_span_fp8.py", "[.span_compat][span_fp8][onnx]") {
  const std::string model_path = AJP_SPAN_FP8_MODEL;
  const std::string package_path = AJP_SPAN_FP8_PYTHON_PACKAGE;
  if (model_path.empty() || package_path.empty())
    SKIP("-DAJP_SPAN_FP8_MODEL=… -DAJP_SPAN_FP8_PYTHON_PACKAGE=… (paquet écrit par derive_span_fp8.py)");
  const std::vector<std::byte> theirs = read_file(package_path.c_str());
  const auto parsed = parse(theirs);
  REQUIRE(parsed);  // un paquet de l'outil Python se lit avec le lecteur C++
  auto mine = package_from_onnx_file(model_path);
  REQUIRE(mine);
  mine->scales = parsed->scales;  // la calibration est ce qui diffère ; les poids doivent être les mêmes
  const std::vector<std::byte> rewritten = serialize(*mine);
  REQUIRE(rewritten.size() == theirs.size());
  CHECK(std::memcmp(rewritten.data(), theirs.data(), theirs.size()) == 0);  // en-tête, CRC et charge compris
}

}  // namespace ajp::gpu::span_fp8
