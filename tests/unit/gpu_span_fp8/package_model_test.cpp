// Le graphe ONNX rebâti d'un paquet (package_model.h) : sans quantification, l'interpréteur de l'IR y rend ce
// que le SPAN du processeur (forward.h) rend du paquet ; avec, un FakeQuantE4M3 devant chaque tenseur que le
// paquet quantifie devant un bloc, des poids de blocs que recover_fp8_weights retrouve, et une image proche
// de la FP16 aux échelles tirées des maxima de ses sorties de relevé.
#include "gpu/span_fp8/package_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/fp8_weights.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/forward.h"
#include "gpu/span_fp8/metrics.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

Conv noise_conv(int outs, int taps, int inputs, std::uint32_t& seed, float amplitude) {
  Conv conv{.weights = std::vector<std::uint16_t>(static_cast<std::size_t>(outs * taps * inputs)),
            .bias = std::vector<float>(static_cast<std::size_t>(outs))};
  const auto next = [&seed] {
    seed = seed * 1664525U + 1013904223U;
    return static_cast<float>(seed >> 8U) / 16777216.0F - 0.5F;
  };
  for (std::uint16_t& w : conv.weights) w = kernels::ref::float_to_half(amplitude * next());
  for (float& b : conv.bias) b = kernels::ref::round_to_half(0.05F * next());
  return conv;
}

// Des poids pseudo-aléatoires partout, assez petits pour que le réseau ne sature pas, et une échelle 1 sur
// toutes les places.
Package noise_package(int channels, int blocks) {
  std::uint32_t seed = 777U;
  Package p;
  p.real_channels = channels;
  p.first = noise_conv(kChannels, 9, kUnshuffledPadded, seed, 0.3F);
  p.blocks.assign(static_cast<std::size_t>(blocks), Block{});
  for (Block& b : p.blocks) {
    for (Conv& conv : b.conv) conv = noise_conv(kChannels, 9, kChannels, seed, 0.12F);
    b.attention = noise_conv(kChannels, 1, kChannels, seed, 0.3F);
  }
  p.concat = noise_conv(kChannels, 1, kConcatChannels, seed, 0.15F);
  p.pixel = noise_conv(kPixelChannels, 9, kChannels, seed, 0.12F);
  p.scales.fill(1.0F);
  return p;
}

std::vector<float> pattern(int height, int width) {
  std::vector<float> rgb(std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = kernels::ref::round_to_half(
        0.5F + 0.45F * std::sin(0.37F * static_cast<float>(i % 97) + 0.011F * static_cast<float>(i)));
  return rgb;
}

ir::Graph graph_of(const Package& p, const ModelOptions& options) {
  auto model = package_model(p, options);
  REQUIRE(model);
  auto graph = ir::import_model(*model);
  REQUIRE(graph);
  return std::move(*graph);
}

std::vector<ir::NamedValue> run(const ir::Graph& graph, const std::vector<float>& rgb, int height,
                                int width) {
  const ir::Value input{.type = ir::DataType::F16, .dims = {1, 3, height, width}, .f = rgb, .i = {}};
  auto ran = ir::run(graph, std::vector<ir::NamedValue>{{.name = kModelInput, .value = input}});
  REQUIRE(ran);
  return std::move(*ran);
}

float worst_gap(std::span<const float> a, std::span<const float> b) {
  REQUIRE(a.size() == b.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

std::size_t count_ops(const ir::Graph& graph, ir::OpKind op) {
  return static_cast<std::size_t>(
      std::count_if(graph.nodes.begin(), graph.nodes.end(), [op](const ir::Node& n) { return n.op == op; }));
}

}  // namespace

TEST_CASE("le graphe d'un paquet rend, sans quantification, ce que rend le SPAN du processeur",
          "[span_fp8][package_model]") {
  for (const auto& [channels, blocks] : {std::pair{48, 2}, std::pair{64, 3}}) {
    const Package p = noise_package(channels, blocks);
    constexpr int kH = 10, kW = 12;
    const std::vector<float> rgb = pattern(kH, kW);
    const ir::Graph graph = graph_of(p, {.quantize = false, .peaks = false});
    CHECK(count_ops(graph, ir::OpKind::FakeQuantE4M3) == 0);
    const std::vector<ir::NamedValue> out = run(graph, rgb, kH, kW);
    REQUIRE(out.size() == 1);
    CHECK(out[0].value.dims == ir::Dims{1, 3, 2 * kH, 2 * kW});
    const std::vector<float> want = forward(p, rgb, kH, kW);
    CHECK(worst_gap(out[0].value.f, want) < 1e-4F);
  }
}

TEST_CASE("les sorties de relevé du graphe suivent l'image, dans l'ordre des échelles",
          "[span_fp8][package_model]") {
  const Package p = noise_package(48, 3);
  constexpr int kH = 8, kW = 8;
  const std::vector<float> rgb = pattern(kH, kW);
  const std::vector<std::string> names = peak_names(3);
  REQUIRE(names.size() == scale_count(3));
  CHECK(names.front() == "conv_1_out");
  CHECK(names[1] == "block_1.r1");
  CHECK(names.back() == "block_3.out");
  const ir::Graph plain = graph_of(p, {.quantize = false, .peaks = false});
  const ir::Graph tapped = graph_of(p, {.quantize = false, .peaks = true});
  const std::vector<ir::NamedValue> a = run(plain, rgb, kH, kW);
  const std::vector<ir::NamedValue> b = run(tapped, rgb, kH, kW);
  REQUIRE(b.size() == 1 + names.size());
  CHECK(worst_gap(a[0].value.f, b[0].value.f) == 0.0F);
  for (std::size_t k = 0; k < names.size(); ++k) {
    INFO(names[k]);
    CHECK(b[1 + k].name == names[k]);
    CHECK(b[1 + k].value.dims == ir::Dims{1, 48, kH / 2, kW / 2});
    // Les Relu ne rendent rien de négatif.
    if (k % 3 == 1 || k % 3 == 2)
      CHECK(*std::min_element(b[1 + k].value.f.begin(), b[1 + k].value.f.end()) >= 0.0F);
  }
}

TEST_CASE("le graphe quantifié : un FakeQuant par entrée de bloc, des convolutions FP8 aux poids retrouvés",
          "[span_fp8][package_model]") {
  constexpr int kBlocksHere = 3;
  Package p = noise_package(48, kBlocksHere);
  constexpr int kH = 12, kW = 16;
  const std::vector<float> rgb = pattern(kH, kW);
  // Les échelles des maxima de la référence FP16, comme la dérivation les tire.
  const std::vector<ir::NamedValue> tapped =
      run(graph_of(p, {.quantize = false, .peaks = true}), rgb, kH, kW);
  std::array<float, kScaleCount> peaks{};
  for (std::size_t k = 0; k < scale_count(kBlocksHere); ++k)
    for (const float v : tapped[1 + k].value.f) peaks[k] = std::max(peaks[k], std::abs(v));
  auto scales = scales_from_peaks(peaks, kBlocksHere);
  REQUIRE(scales);
  p.scales = *scales;

  ir::Graph graph = graph_of(p, {.quantize = true, .peaks = false});
  // La sortie du dernier bloc ne va qu'à la concaténation, en FP16 : son échelle ne sert pas (comme les
  // noyaux CUTLASS, span_fp8_body.cuh).
  CHECK(count_ops(graph, ir::OpKind::FakeQuantE4M3) == scale_count(kBlocksHere) - 1);
  const std::vector<float> fp8 = run(graph, rgb, kH, kW)[0].value.f;
  const std::vector<float> fp16 = tapped[0].value.f;
  const double psnr_db = psnr(fp16, fp8);
  WARN("PSNR FP8 contre FP16 : " << psnr_db << " dB");
  CHECK(psnr_db > 30.0);
  CHECK(worst_gap(fp16, fp8) > 0.0F);  // l'arrondi E4M3 a bien lieu

  // Après les passes du moteur : chaque convolution de bloc lit son entrée arrondie, ses poids sur la grille.
  REQUIRE(
      ir::infer_shapes(graph, std::vector<ir::InputShape>{{.name = kModelInput, .dims = {1, 3, kH, kW}}}));
  REQUIRE(ir::optimize(graph));
  CHECK(count_ops(graph, ir::OpKind::FakeQuantE4M3) == 0);
  std::size_t fp8_convs = 0;
  for (const ir::Node& node : graph.nodes) {
    if (node.op != ir::OpKind::Conv) continue;
    const auto& attrs = std::get<ir::ConvAttrs>(node.attrs);
    if (!(attrs.in_quant > 0.0F)) continue;
    ++fp8_convs;
    const ir::Tensor& weights = graph.tensors[node.inputs[1]];
    INFO(node.name);
    CHECK(ir::recover_fp8_weights(weights.value).has_value());
  }
  CHECK(fp8_convs == 3 * kBlocksHere);
  // Les passes ne changent pas le calcul.
  CHECK(worst_gap(run(graph, rgb, kH, kW)[0].value.f, fp8) < 1e-5F);
}

TEST_CASE("le graphe d'un paquet refuse un nombre de blocs ou une échelle impossible",
          "[span_fp8][package_model]") {
  Package p = noise_package(48, 2);
  p.blocks.resize(1);
  CHECK_FALSE(package_model(p));
  p = noise_package(48, 2);
  p.scales[2] = 0.0F;
  CHECK_FALSE(package_model(p));
  CHECK(package_model(p, {.quantize = false, .peaks = false}));  // la référence n'en lit aucune
  p.fp16 = true;
  CHECK(package_model(p));  // un paquet FP16 ne quantifie rien
}

}  // namespace ajp::gpu::span_fp8
