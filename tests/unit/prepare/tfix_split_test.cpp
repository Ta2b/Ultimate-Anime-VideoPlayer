// Découpe de TemporalFix en cinq morceaux (prepare/tfix_split.h) : le manifeste et ses refus, et, quand les
// modèles du lecteur sont sur la machine (KIRARI_TFIX_MODEL_DIR), la reconnaissance et la découpe
// des vrais s1, s2 et s3, et leurs morceaux lus pour les noyaux natifs (--tfix-native).
#include "prepare/tfix_split.h"

#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/tfix_fused.h"
#include "prepare/model_source.h"
#include "prepare/onnx_split.h"
#include "prepare/rife_native.h"

#ifndef KIRARI_TFIX_MODEL_DIR
#define KIRARI_TFIX_MODEL_DIR ""
#endif

namespace kirari::prepare {
namespace {

TEST_CASE("Le manifeste de TemporalFix fusionné se relit", "[prepare][tfix_fused]") {
  const TfixFusedManifest window{.role = TfixRole::Window,
                                 .height = 540,
                                 .width = 960,
                                 .half = true,
                                 .pieces = {"a-bloc0.onnx", "a-bloc1.onnx", "a-bloc2.onnx", "a-conf.onnx"},
                                 .sizes = {1, 2, 3, 4},
                                 .digests = {}};
  const auto read = parse_tfix_manifest(format_tfix_manifest(window));
  REQUIRE(read);
  CHECK(read->role == TfixRole::Window);
  CHECK(read->height == 540);
  CHECK(read->width == 960);
  CHECK(read->half);
  CHECK(read->pieces == window.pieces);
  CHECK(read->sizes == window.sizes);

  TfixFusedManifest encoder = window;
  encoder.role = TfixRole::Encoder;
  CHECK_FALSE(parse_tfix_manifest(format_tfix_manifest(encoder)));  // un morceau attendu, quatre donnés
  encoder.pieces = {"a-encode.onnx"};
  encoder.sizes = {9};
  CHECK(parse_tfix_manifest(format_tfix_manifest(encoder)));
  TfixFusedManifest odd = window;
  odd.height = 541;
  CHECK_FALSE(parse_tfix_manifest(format_tfix_manifest(odd)));
  CHECK_FALSE(parse_tfix_manifest("kirari-tfix-fused 1\nrole window\nsize 540 960\n"));
  CHECK_FALSE(parse_tfix_manifest(format_tfix_manifest(window) + "inconnu 1\n"));
  CHECK(is_tfix_fused_manifest("x/m.tfixfused"));
}

TEST_CASE("Les TemporalFix du catalogue se coupent en morceaux", "[prepare][onnx][tfix_fused]") {
  const std::filesystem::path dir = std::filesystem::path(KIRARI_TFIX_MODEL_DIR);
  bool any = false;
  for (const std::string_view file :
       {"temporalfix_s1_v1.1_op18_fp16.onnx", "temporalfix_s2_v1_op18_fp16.onnx",
        "temporalfix_s3_v1_op18_fp16.onnx"}) {
    const std::filesystem::path path = dir / file;
    if (std::string_view(KIRARI_TFIX_MODEL_DIR).empty() || !std::filesystem::is_regular_file(path)) continue;
    any = true;
    INFO(path.string());
    auto bytes = read_onnx_bytes(path);
    REQUIRE(bytes);
    const auto split = analyze_tfix(*bytes);
    REQUIRE(split);
    const auto window = split_tfix(*bytes, *split, TfixRole::Window, 68, 120);
    REQUIRE(window);
    CHECK(window->size() == 4);
    const auto encoder = split_tfix(*bytes, *split, TfixRole::Encoder, 68, 120);
    REQUIRE(encoder);
    CHECK(encoder->size() == 1);
    CHECK_FALSE(split_tfix(*bytes, *split, TfixRole::Window, 67, 120));
    // Les noyaux natifs : l'encodeur coupé avant son Resize, puis des chaînes de convolutions 3 × 3 (les
    // séparables repliées).
    const SplitTensor from{.name = split->encode_input, .dims = {1, 3, 68, 120}};
    const SplitTensor to{.name = split->encode_last};
    const auto cut = extract_subgraph((*encoder)[0].model, std::span(&from, 1), std::span(&to, 1), "e");
    REQUIRE(cut);
    const auto encode = rife_native_layers(*cut);
    REQUIRE(encode);
    CHECK(encode->size() == 4);
    CHECK(conv_chain_check(*encode, kernels::kTfixImage, kernels::kTfixTraits));
    const int ins[] = {kernels::kTfixX0, kernels::kTfixX, kernels::kTfixX, kernels::kTfixConf};
    const std::size_t counts[] = {7, 7, 7, 3};
    for (std::size_t i = 0; i < window->size(); ++i) {
      const auto layers = rife_native_layers((*window)[i].model);
      REQUIRE(layers);
      CHECK(layers->size() == counts[i]);
      CHECK(conv_chain_check(*layers, ins[i], i < 3 ? kernels::kTfixY : 1));
    }
  }
  if (!any) SKIP("modèles TemporalFix du lecteur absents (KIRARI_TFIX_MODEL_DIR)");
}

}  // namespace
}  // namespace kirari::prepare

TEST_CASE("TemporalFix natif : un rôle à la fois, l'encodeur seul ou la fenêtre seule", "[prepare][tfix]") {
  using kirari::kernels::tfix_native_encoder;
  const std::array encoder{true, false, false, false, false};
  const std::array window{false, true, true, true, true};
  CHECK(tfix_native_encoder(encoder) == true);
  CHECK(tfix_native_encoder(window) == false);
  const std::array both{true, true, true, true, true};
  const std::array none{false, false, false, false, false};
  const std::array hole{false, true, false, true, true};
  CHECK_FALSE(tfix_native_encoder(both).has_value());
  CHECK_FALSE(tfix_native_encoder(none).has_value());
  CHECK_FALSE(tfix_native_encoder(hole).has_value());
  CHECK_FALSE(tfix_native_encoder(std::array{true, false}).has_value());
}
