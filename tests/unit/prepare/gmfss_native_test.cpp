// La fusion de GMFSS coupée pour le GridNet natif (prepare/gmfss_fused.h, prepare/gmfss_native.h) : le
// manifeste relu et ses refus, la vérification d'un réseau écrit à la main, et sur la fusion exportée quand
// elle est là (KIRARI_RIFE_MODEL_DIR) : le GridNet lu et l'IFNet extrait.
#include "prepare/gmfss_native.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/gmfss_fused.h"
#include "prepare/model_source.h"

#ifndef KIRARI_RIFE_MODEL_DIR
#define KIRARI_RIFE_MODEL_DIR ""
#endif

namespace kirari::prepare {
namespace {

const std::string kDigest(64, 'a');

GmfssGridConv conv(int cin, int cout, int stride = 1, bool transposed = false) {
  const std::size_t taps = transposed ? 16 : 9;
  return {.transposed = transposed,
          .cin = cin,
          .cout = cout,
          .stride = stride,
          .weights = std::vector<float>(taps * static_cast<std::size_t>(cin * cout), 0.5F),
          .bias = std::vector<float>(static_cast<std::size_t>(cout), 0.0F)};
}

// Les quatre entrées, une descente, une montée par transposée, puis la queue jusqu'à l'image.
GmfssGrid small_grid() {
  GmfssGrid grid;
  grid.tensors = {
      {0, 16}, {0, 128}, {1, 256}, {2, 384}, {1, 64}, {0, 64}, {kGmfssFullLevel, 64}, {kGmfssFullLevel, 8}};
  grid.steps.push_back({.conv = conv(9, 64, 2), .from = 0, .to = 4});
  grid.steps.push_back({.conv = conv(64, 64, 2, true), .from = 4, .to = 5});
  grid.steps.push_back({.conv = conv(64, 256), .from = 5, .to = 6, .shuffle = true});
  grid.steps.push_back({.conv = conv(64, 3), .from = 6, .to = 7});
  return grid;
}

}  // namespace

TEST_CASE("GMFSS natif : le manifeste se relit, un manifeste abîmé est refusé", "[prepare][gmfss]") {
  const GmfssFusedManifest manifest{.height = 384,
                                    .width = 640,
                                    .pieces = {"fusion.onnx", "fusion-ifnet.onnx"},
                                    .sizes = {10, 20},
                                    .digests = {kDigest, kDigest}};
  const std::string text = format_gmfss_manifest(manifest);
  const auto parsed = parse_gmfss_manifest(text);
  REQUIRE(parsed);
  CHECK(parsed->height == 384);
  CHECK(parsed->width == 640);
  CHECK(parsed->pieces == manifest.pieces);
  CHECK(parsed->sizes == manifest.sizes);
  CHECK(parsed->digests == manifest.digests);
  CHECK(is_gmfss_fused_manifest("a/b.gmfssfused"));
  CHECK_FALSE(is_gmfss_fused_manifest("a/b.onnx"));
  GmfssFusedManifest odd = manifest;
  odd.height = 380;  // pas un multiple de 32
  CHECK_FALSE(parse_gmfss_manifest(format_gmfss_manifest(odd)));
  GmfssFusedManifest alone = manifest;
  alone.pieces.pop_back();  // l'IFNet manque
  CHECK_FALSE(parse_gmfss_manifest(format_gmfss_manifest(alone)));
  CHECK_FALSE(parse_gmfss_manifest(text + "piece ../fusion.onnx\n"));
  CHECK_FALSE(parse_gmfss_manifest(text + "inconnu 1\n"));
}

TEST_CASE("GMFSS natif : un réseau qui se tient passe, ses fautes sont refusées", "[prepare][gmfss]") {
  const auto ok = gmfss_grid_check(small_grid());
  INFO((ok ? std::string() : ok.error().detail));
  CHECK(ok);
  GmfssGrid unread = small_grid();
  unread.steps[1].from = 5;  // lit sa propre sortie
  CHECK_FALSE(gmfss_grid_check(unread));
  GmfssGrid early = small_grid();
  early.steps[0].from = 5;  // lit un tenseur pas encore écrit
  CHECK_FALSE(gmfss_grid_check(early));
  GmfssGrid size = small_grid();
  size.steps[0].conv.stride = 1;  // la sortie n'est pas à la moitié
  CHECK_FALSE(gmfss_grid_check(size));
  GmfssGrid phase_skip = small_grid();
  phase_skip.steps[1].skip = 5;
  CHECK_FALSE(gmfss_grid_check(phase_skip));
  GmfssGrid channels = small_grid();
  channels.steps[3].conv = conv(32, 3);  // 32 canaux pour un tenseur de 64
  CHECK_FALSE(gmfss_grid_check(channels));
  CHECK_FALSE(gmfss_grid(std::vector<std::byte>(16, std::byte{0x7f})));
}

TEST_CASE("GMFSS natif : la fusion exportée se lit et son IFNet s'extrait", "[prepare][gmfss]") {
  const std::filesystem::path model =
      std::filesystem::path(KIRARI_RIFE_MODEL_DIR) / "gmfss_fortuna_union_fusion.onnx";
  if (!std::filesystem::exists(model)) SKIP("gmfss_fortuna_union_fusion.onnx absent");
  auto bytes = read_onnx_bytes(model);
  REQUIRE(bytes);
  const auto grid = gmfss_grid(*bytes);
  INFO((grid ? std::string() : grid.error().detail));
  REQUIRE(grid);
  CHECK(grid->steps.size() == 2 * 21 + 3);
  CHECK(grid->tensors.back().level == kGmfssFullLevel);
  CHECK(grid->steps[2].conv.in_slope == 0.25F);  // residual_model_head0.0 (après les deux de head1)
  const auto ifnet = gmfss_ifnet(*bytes, 384, 640);
  REQUIRE(ifnet);
  CHECK(ifnet->size() < bytes->size());
}

}  // namespace kirari::prepare
