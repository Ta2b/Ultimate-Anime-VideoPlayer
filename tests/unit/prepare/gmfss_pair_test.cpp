// La paire de GMFSS coupée pour le FeatureNet et le MetricNet natifs (prepare/gmfss_pair_split.h,
// prepare/gmfss_pair_native.h) : le manifeste relu et ses refus, la vérification de deux réseaux écrits à la
// main, et sur la paire exportée quand elle est là (KIRARI_RIFE_MODEL_DIR) : les réseaux lus, la coupe
// relevée et le flot extrait.
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "prepare/gmfss_pair_native.h"
#include "prepare/gmfss_pair_split.h"
#include "prepare/model_source.h"

#ifndef KIRARI_RIFE_MODEL_DIR
#define KIRARI_RIFE_MODEL_DIR ""
#endif

namespace kirari::prepare {
namespace {

const std::string kDigest(64, 'b');

GmfssGridConv conv(int cin, int cout, int stride = 1) {
  return {.cin = cin,
          .cout = cout,
          .stride = stride,
          .weights = std::vector<float>(9U * static_cast<std::size_t>(cin * cout), 0.25F),
          .bias = std::vector<float>(static_cast<std::size_t>(cout), 0.0F)};
}

GmfssPairManifest manifest() {
  GmfssPairManifest m{.height = 384,
                      .width = 640,
                      .pieces = {"pair.onnx", "pair-flow.onnx"},
                      .sizes = {10, 20},
                      .digests = {kDigest, kDigest}};
  m.cut.metric_input = "/metricnet/Concat_output_0";
  for (int i = 0; i < kGmfssPairFlows; ++i)
    m.cut.flows[static_cast<std::size_t>(i)] = "/flow_" + std::to_string(i);
  return m;
}

// Le FeatureNet (l'image, ses trois sorties, puis les tenseurs des blocs) et le MetricNet (l'entrée, la
// sortie, une somme), au plus court.
GmfssPairNets small_nets() {
  GmfssPairNets nets;
  nets.features.tensors = {{kGmfssFullLevel, 16}, {0, 64}, {1, 128}, {2, 192}};
  nets.features.steps.push_back({.conv = conv(3, 64, 2), .from = 0, .to = 1});
  nets.features.steps.push_back({.conv = conv(64, 128, 2), .from = 1, .to = 2});
  nets.features.steps.push_back({.conv = conv(128, 192, 2), .from = 2, .to = 3});
  nets.metric.tensors = {{0, 16}, {0, 8}, {0, 64}, {0, 64}};
  nets.metric.steps.push_back({.conv = conv(14, 64), .from = 0, .to = 2});
  nets.metric.steps.push_back({.conv = conv(64, 64), .from = 2, .skip = 2, .to = 3});
  nets.metric.steps.push_back({.conv = conv(64, 2), .from = 3, .to = 1});
  return nets;
}

}  // namespace

TEST_CASE("GMFSS natif : le manifeste de la paire se relit, un manifeste abîmé est refusé",
          "[prepare][gmfss]") {
  const GmfssPairManifest m = manifest();
  const std::string text = format_gmfss_pair_manifest(m);
  const auto parsed = parse_gmfss_pair_manifest(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 384);
  CHECK(parsed->width == 640);
  CHECK(parsed->pieces == m.pieces);
  CHECK(parsed->digests == m.digests);
  CHECK(parsed->cut.metric_input == m.cut.metric_input);
  CHECK(parsed->cut.flows == m.cut.flows);
  CHECK(is_gmfss_pair_manifest("a/b-natif1.gmfsspair"));
  CHECK_FALSE(is_gmfss_pair_manifest("a/b.gmfssfused"));

  const auto without = [&](std::string_view line) {
    std::string cut = text;
    const std::size_t at = cut.find(line);
    REQUIRE(at != std::string::npos);
    cut.erase(at, cut.find('\n', at) + 1 - at);
    return cut;
  };
  CHECK_FALSE(parse_gmfss_pair_manifest(without("flow /flow_5")));          // un flot manque
  CHECK_FALSE(parse_gmfss_pair_manifest(without("metric ")));               // l'entrée du MetricNet manque
  CHECK_FALSE(parse_gmfss_pair_manifest(text + "flow /de_trop\n"));         // un flot de trop
  CHECK_FALSE(parse_gmfss_pair_manifest(text + "inconnu 1\n"));             // champ inconnu
  CHECK_FALSE(parse_gmfss_pair_manifest(without("piece pair-flow.onnx")));  // le flot manque
  GmfssPairManifest odd = m;
  odd.width = 630;
  CHECK_FALSE(parse_gmfss_pair_manifest(format_gmfss_pair_manifest(odd)));  // pas un multiple de 32
}

TEST_CASE("GMFSS natif : les réseaux de la paire se vérifient, leurs fautes sont refusées",
          "[prepare][gmfss]") {
  CHECK(gmfss_pair_check(small_nets()));

  GmfssPairNets same_size = small_nets();
  same_size.features.steps[0].conv.stride = 1;  // l'image ne se lit qu'en pas 2
  CHECK_FALSE(gmfss_pair_check(same_size));

  GmfssPairNets unwritten = small_nets();
  unwritten.features.steps.pop_back();  // source_3 jamais écrite
  CHECK_FALSE(gmfss_pair_check(unwritten));

  GmfssPairNets skip = small_nets();
  skip.metric.tensors[2].channels = 48;  // plus étroit que la convolution qui l'écrit
  CHECK_FALSE(gmfss_pair_check(skip));

  GmfssPairNets wide = small_nets();
  wide.metric.tensors[1].channels = 16;  // la sortie du MetricNet : 8 canaux
  CHECK_FALSE(gmfss_pair_check(wide));
}

TEST_CASE("GMFSS natif : la paire exportée se lit et se coupe", "[prepare][gmfss]") {
  const bool half = GENERATE(false, true);
  const std::filesystem::path dir(KIRARI_RIFE_MODEL_DIR);
  const std::filesystem::path pair =
      dir / (half ? "gmfss_fortuna_union_s05_pair.onnx" : "gmfss_fortuna_union_pair.onnx");
  CAPTURE(pair);
  if (std::string(KIRARI_RIFE_MODEL_DIR).empty() || !std::filesystem::is_regular_file(pair))
    SKIP("paire GMFSS absente de KIRARI_RIFE_MODEL_DIR");
  const auto bytes = read_onnx_bytes(pair);
  REQUIRE(bytes);
  const auto nets = gmfss_pair_nets(*bytes);
  REQUIRE(nets);
  CHECK(nets->features.steps.size() == 6);  // trois blocs de deux convolutions
  CHECK(nets->metric.steps.size() == 5);    // metric_in, trois résiduels, metric_out
  CHECK(nets->features.steps[0].conv.stride == 2);
  const auto cut = gmfss_pair_cut(*bytes);
  REQUIRE(cut);
  CHECK_FALSE(cut->metric_input.empty());
  for (const std::string& flow : cut->flows) CHECK_FALSE(flow.empty());
  const auto flow = gmfss_flow_piece(*bytes, *cut, half ? 128 : 96, 128);
  REQUIRE(flow);
  CHECK(flow->size() < bytes->size());  // ni le FeatureNet ni le MetricNet
}

}  // namespace kirari::prepare
