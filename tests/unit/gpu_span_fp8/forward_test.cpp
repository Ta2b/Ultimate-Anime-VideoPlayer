// Le SPAN f32 du processeur (forward.h), juge des noyaux CUTLASS au chargement : un paquet construit pour
// rendre l'agrandissement au plus proche, et (test caché [.span_b5f48]) l'accord avec l'interpréteur de l'IR
// sur le vrai b5f48.
#include "gpu/span_fp8/forward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

Conv zero_conv(int outs, int taps, int inputs) {
  return Conv{.weights = std::vector<std::uint16_t>(static_cast<std::size_t>(outs * taps * inputs), 0),
              .bias = std::vector<float>(static_cast<std::size_t>(outs), 0.0F)};
}

// conv_1 recopie les 12 canaux dégroupés (position centrale), chaque bloc rend son entrée (c3 nul,
// tanh(20) = 1 à l'arrondi f32), conv_cat garde la sortie de conv_1 et to_pixel envoie au canal
// c·16 + by·4 + bx le canal dégroupé c·4 + (by/2)·2 + bx/2 : l'image doublée au plus proche.
Package nearest_package(int channels, int blocks) {
  constexpr std::uint16_t kOne = 0x3C00;
  constexpr int kCentre = 4;
  Package p;
  p.real_channels = channels;
  p.first = zero_conv(kChannels, 9, kUnshuffledPadded);
  for (int c = 0; c < kUnshuffled; ++c)
    p.first.weights[static_cast<std::size_t>((c * 9 + kCentre) * kUnshuffledPadded + c)] = kOne;
  p.blocks.assign(static_cast<std::size_t>(blocks), Block{});
  for (Block& b : p.blocks) {
    for (Conv& conv : b.conv) conv = zero_conv(kChannels, 9, kChannels);
    b.attention = zero_conv(kChannels, 1, kChannels);
    std::fill(b.attention.bias.begin(), b.attention.bias.end(), 20.0F);
  }
  p.concat = zero_conv(kChannels, 1, kConcatChannels);
  for (int c = 0; c < channels; ++c)
    p.concat.weights[static_cast<std::size_t>(c * kConcatChannels + c)] = kOne;
  p.pixel = zero_conv(kPixelChannels, 9, kChannels);
  for (int c = 0; c < 3; ++c)
    for (int by = 0; by < 4; ++by)
      for (int bx = 0; bx < 4; ++bx) {
        const int out = c * 16 + by * 4 + bx, in = c * 4 + (by / 2) * 2 + bx / 2;
        p.pixel.weights[static_cast<std::size_t>((out * 9 + kCentre) * kChannels + in)] = kOne;
      }
  return p;
}

std::vector<float> pattern(int height, int width) {
  std::vector<float> rgb(std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = kernels::ref::round_to_half(
        0.5F + 0.45F * std::sin(0.37F * static_cast<float>(i % 97) + 0.011F * static_cast<float>(i)));
  return rgb;
}

}  // namespace

TEST_CASE("le SPAN du processeur rend l'agrandissement au plus proche d'un paquet fait pour", "[span_fp8]") {
  for (const auto& [channels, blocks] : {std::pair{48, 5}, std::pair{64, 8}}) {
    const Package p = nearest_package(channels, blocks);
    constexpr int kH = 6, kW = 8;
    const std::vector<float> rgb = pattern(kH, kW);
    const std::vector<float> out = forward(p, rgb, kH, kW);
    REQUIRE(out.size() == rgb.size() * 4);
    float worst = 0.0F;
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 2 * kH; ++y)
        for (int x = 0; x < 2 * kW; ++x) {
          const float want = rgb[static_cast<std::size_t>((c * kH + y / 2) * kW + x / 2)];
          const float got = out[static_cast<std::size_t>((c * 2 * kH + y) * 2 * kW + x)];
          worst = std::max(worst, std::abs(got - want));
        }
    CHECK(worst < 1e-6F);
  }
}

TEST_CASE("le SPAN du processeur refuse une taille impaire ou une entrée de la mauvaise longueur",
          "[span_fp8]") {
  const Package p = nearest_package(48, 5);
  CHECK(forward(p, pattern(5, 8), 5, 8).empty());
  CHECK(forward(p, pattern(6, 8), 6, 6).empty());
  CHECK(forward(p, {}, 0, 0).empty());
}

TEST_CASE("pixels de sortie qu'une découpe calcule comme l'image entière", "[span_fp8]") {
  CHECK(exact_side(64, 5) == 60);  // 32 au demi-plan, moins 2 + 15 lignes mangées, fois 4
  CHECK(exact_side(64, 8) == 24);
  CHECK(exact_side(32, 8) == 0);
}

TEST_CASE("une découpe rend le coin de l'image entière sur exact_side pixels", "[span_fp8]") {
  // Poids pseudo-aléatoires à toutes les positions des filtres : chaque convolution étale vraiment.
  Package p = nearest_package(48, 2);
  std::uint32_t seed = 12345U;
  const auto noise = [&seed](Conv& conv) {
    for (std::uint16_t& w : conv.weights) {
      seed = seed * 1664525U + 1013904223U;
      w = kernels::ref::float_to_half(0.08F * (static_cast<float>(seed >> 8U) / 16777216.0F - 0.5F));
    }
  };
  noise(p.first);
  for (Block& b : p.blocks) {
    for (Conv& conv : b.conv) noise(conv);
    noise(b.attention);
  }
  noise(p.concat);
  noise(p.pixel);
  constexpr int kSide = 32, kH = 40, kW = 48;
  const int exact = exact_side(kSide, 2);
  REQUIRE(exact == 32);  // 16 au demi-plan, moins 2 + 6 lignes mangées, fois 4
  const std::vector<float> whole = pattern(kH, kW);
  std::vector<float> crop(std::size_t{3} * kSide * kSide);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < kSide; ++y)
      for (int x = 0; x < kSide; ++x)
        crop[static_cast<std::size_t>((c * kSide + y) * kSide + x)] =
            whole[static_cast<std::size_t>((c * kH + y) * kW + x)];
  const std::vector<float> big = forward(p, whole, kH, kW), small = forward(p, crop, kSide, kSide);
  float inside = 0.0F, beyond = 0.0F;
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 2 * kSide; ++y)
      for (int x = 0; x < 2 * kSide; ++x) {
        const float d = std::abs(big[static_cast<std::size_t>((c * 2 * kH + y) * 2 * kW + x)] -
                                 small[static_cast<std::size_t>((c * 2 * kSide + y) * 2 * kSide + x)]);
        float& worst = y < exact && x < exact ? inside : beyond;
        worst = std::max(worst, d);
      }
  CHECK(inside == 0.0F);
  CHECK(beyond > 0.0F);  // au-delà, le bord de la découpe se voit
}

TEST_CASE("le SPAN du processeur rend ce que l'interpréteur de l'IR rend du vrai b5f48",
          "[.span_b5f48][span_fp8]") {
  const std::string path = AJP_SPAN_B5F48_MODEL;
  if (path.empty() || !std::filesystem::exists(path))
    SKIP("-DAJP_SPAN_B5F48_MODEL=… (SPAN b5f48 d'AnimeJaNai)");
  const auto package = package_from_onnx_file(path);
  REQUIRE(package);
  const auto graph = ir::load_onnx_file(path);
  REQUIRE(graph);
  constexpr int kH = 24, kW = 40;
  const std::vector<float> rgb = pattern(kH, kW);
  ir::Value input{.type = ir::DataType::F16, .dims = {1, 3, kH, kW}, .f = rgb, .i = {}};
  const auto ran = ir::run(*graph, std::vector<ir::NamedValue>{{.name = "input", .value = input}});
  REQUIRE(ran);
  const std::vector<float>& theirs = (*ran)[0].value.f;
  const std::vector<float> mine = forward(*package, rgb, kH, kW);
  REQUIRE(mine.size() == theirs.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < mine.size(); ++i) worst = std::max(worst, std::abs(mine[i] - theirs[i]));
  WARN("écart maximal au b5f48 : " << worst);
  CHECK(worst < 1e-3F);
}

}  // namespace ajp::gpu::span_fp8
