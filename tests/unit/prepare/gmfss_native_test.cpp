// La fusion de GMFSS coupée pour le GridNet natif (prepare/gmfss_fused.h, prepare/gmfss_native.h) : le
// manifeste relu et ses refus, la vérification d'un réseau écrit à la main et sa coupe en morceaux, l'IFNet
// trouvé dans une fusion en miniature des deux exports, et sur la fusion exportée quand elle est là
// (KIRARI_RIFE_MODEL_DIR) : le GridNet lu, ses morceaux et l'IFNet extrait.
#include "prepare/gmfss_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/gmfss_fused.h"
#include "prepare/model_source.h"
#include "rife_fake.h"

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

// Une fusion en miniature, aux noms et opérateurs des exports : l'IFNet (half0, half1, time) rend image, le
// Concat des images [image0, image, image1] passe par le PReLU décomposé jusqu'à la première convolution du
// GridNet (head : le nom de son poids). images : les entrées du Concat (3, ou 4 comme la fusion de base).
Bytes mini_fusion(const std::string& image, const std::string& head, int images = 3) {
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  add(node("/ifnet/Sub", "Sub", {"half1", "half0"}, {"/ifnet/Sub_output_0"}));
  add(node("/ifnet/Mul", "Mul", {"/ifnet/Sub_output_0", "time"}, {"/ifnet/Mul_output_0"}));
  add(node("/ifnet/Add_16", "Add", {"half0", "/ifnet/Mul_output_0"}, {image}));
  add(node("/Slice", "Identity", {"warped0_1"}, {"/Slice_output_0"}));
  add(node("/Slice_1", "Identity", {"warped1_1"}, {"/Slice_1_output_0"}));
  std::vector<std::string> concat{"/Slice_output_0", image, "/Slice_1_output_0"};
  if (images == 4) concat.insert(concat.begin(), "half0");
  add(node("/Concat", "Concat", concat, {"/Concat_output_0"}, {Attr{.name = "axis", .num = 1}}));
  const std::string prelu = "/fusionnet/residual_model_head0/residual_model_head0.0/";
  add(node(prelu + "Relu", "Relu", {"/Concat_output_0"}, {prelu + "Relu_output_0"}));
  add(node(prelu + "Neg", "Neg", {"/Concat_output_0"}, {prelu + "Neg_output_0"}));
  add(node(prelu + "Relu_1", "Relu", {prelu + "Neg_output_0"}, {prelu + "Relu_1_output_0"}));
  add(node(prelu + "Mul", "Mul", {"slope", prelu + "Relu_1_output_0"}, {prelu + "Mul_output_0"}));
  add(node(prelu + "Sub", "Sub", {prelu + "Relu_output_0", prelu + "Mul_output_0"},
           {prelu + "Sub_output_0"}));
  add(node("/fusionnet/residual_model_head0/residual_model_head0.1/Conv", "Conv",
           {prelu + "Sub_output_0", head, "gmfss.fusionnet.residual_model_head0.1.bias"}, {"out"}));
  delimited(graph, 5, tensor_dims("slope", {1}, floats({0.25F})));
  Bytes ones;
  for (int i = 0; i < 3 * images; ++i) {
    const Bytes one = floats({1.0F});
    ones.insert(ones.end(), one.begin(), one.end());
  }
  delimited(graph, 5, tensor_dims(head, {1, 3 * images, 1, 1}, ones));
  delimited(graph, 5, tensor_dims("gmfss.fusionnet.residual_model_head0.1.bias", {1}, floats({0.0F})));
  for (const char* input : {"half0", "half1", "warped0_1", "warped1_1"})
    delimited(graph, 11, value_info(input, 10, {1, 3, 32, 32}));
  delimited(graph, 11, value_info("time", 10, {1, 1, 1, 1}));
  delimited(graph, 12, value_info("out", 10, {1, 1, 32, 32}));
  Bytes model;
  integer(model, 1, 8);
  delimited(model, 7, graph);
  return model;
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

TEST_CASE("GMFSS natif : les morceaux du GridNet suivent la structure, une autre forme n'est pas coupée",
          "[prepare][gmfss]") {
  // La descente écrit la ligne 1 ; la queue part de la montée, qui nourrit le PixelShuffle.
  CHECK(gmfss_grid_pieces(small_grid()) ==
        std::vector<int>{1, kGmfssQueuePiece, kGmfssQueuePiece, kGmfssQueuePiece});
  GmfssGrid flat = small_grid();
  flat.steps[2].shuffle = false;  // pas de PixelShuffle
  CHECK(gmfss_grid_pieces(flat).empty());
  GmfssGrid first = small_grid();
  first.steps.erase(first.steps.begin(), first.steps.begin() + 2);  // le PixelShuffle en premier
  CHECK(gmfss_grid_pieces(first).empty());
  GmfssGrid full = small_grid();
  full.tensors[4].level = kGmfssFullLevel;  // une convolution avant la queue écrit au double
  CHECK(gmfss_grid_pieces(full).empty());
  GmfssGrid deep = small_grid();
  deep.tensors[4].level = 3;  // une quatrième ligne
  CHECK(gmfss_grid_pieces(deep).empty());
}

TEST_CASE("GMFSS natif : l'IFNet se trouve et s'extrait dans les fusions des deux exports",
          "[prepare][gmfss]") {
  constexpr std::string_view kHead = "gmfss.fusionnet.residual_model_head0.1.weight";
  // ifnet_rife.py nomme l'image « rife » ; export_gmfss.py (l'IFNet d'union dans la fusion) lui laisse le nom
  // de son dernier nœud.
  for (const std::string image : {"rife", "/ifnet/Add_16_output_0"}) {
    INFO(image);
    const Bytes fusion = mini_fusion(image, std::string(kHead));
    const auto found = gmfss_ifnet_tensor(fusion);
    INFO((found ? std::string() : found.error().detail));
    REQUIRE(found);
    CHECK(*found == image);
    const auto ifnet = gmfss_ifnet(fusion, 32, 32);
    INFO((ifnet ? std::string() : ifnet.error().detail));
    REQUIRE(ifnet);
    const auto graph = read_onnx_graph(*ifnet);
    REQUIRE(graph);
    CHECK(graph->outputs == std::vector<std::string>{image});
    REQUIRE(graph->nodes.size() == 3);  // Sub, Mul, Add : rien du GridNet
    CHECK(graph->nodes.back().op_type == "Add");
  }
  // Sans GridNet d'union, ou devant une fusion de base (quatre images) : refusée, TensorRT garde la fusion.
  const auto headless = gmfss_ifnet_tensor(mini_fusion("rife", "gmfss.fusionnet.autre.weight"));
  REQUIRE_FALSE(headless);
  CHECK(headless.error().code == RefusalCode::FieldMissing);
  const auto base = gmfss_ifnet(mini_fusion("rife", std::string(kHead), 4), 32, 32);
  REQUIRE_FALSE(base);
  CHECK(base.error().code == RefusalCode::FieldMissing);
}

TEST_CASE("GMFSS natif : le GridNet et les quatre blocs de l'IFNet se lisent dans la fusion exportée",
          "[prepare][gmfss]") {
  const std::filesystem::path dir(KIRARI_RIFE_MODEL_DIR);
  auto model = dir / "gmfss_fortuna_union_fusion.onnx";
  // La fusion est commune aux deux échelles. Les anciens exports au demi-flot la renommaient.
  if (!std::filesystem::exists(model)) model = dir / "gmfss_fortuna_union_s05_fusion.onnx";
  if (!std::filesystem::exists(model)) SKIP("fusion GMFSS Fortuna Union absente");
  const auto fusion = read_onnx_bytes(model);
  REQUIRE(fusion);

  SECTION("Le GridNet et son découpage") {
    const auto grid = gmfss_grid(*fusion);
    INFO((grid ? std::string() : grid.error().detail));
    REQUIRE(grid);
    CHECK(grid->steps.size() == 2 * 21 + 3);
    CHECK(grid->tensors.back().level == kGmfssFullLevel);
    CHECK(grid->steps[2].conv.in_slope == 0.25F);  // residual_model_head0.0 (après les deux de head1)
    // 14 convolutions à la ligne 0, 16 à la ligne 1, 12 à la ligne 2. La queue contient le PixelShuffle et
    // les convolutions qui l'encadrent.
    const std::vector<int> pieces = gmfss_grid_pieces(*grid);
    REQUIRE(pieces.size() == grid->steps.size());
    std::vector<int> count(kGmfssGridPieces, 0);
    for (const int p : pieces) ++count[static_cast<std::size_t>(p)];
    CHECK(count == std::vector<int>{14, 16, 12, 3});
    CHECK(std::all_of(pieces.end() - 3, pieces.end(), [](int p) { return p == kGmfssQueuePiece; }));
    CHECK(grid->steps[grid->steps.size() - 2].shuffle);
  }

  SECTION("L'IFNet extrait et ses quatre blocs") {
    const auto derived = gmfss_ifnet(*fusion, 384, 640);
    REQUIRE(derived);
    CHECK(derived->size() < fusion->size());
    const auto blocks = gmfss_ifnet_blocks(*derived, 384, 640);
    INFO((blocks ? std::string() : blocks.error().detail));
    REQUIRE(blocks);
    constexpr std::array<int, 4> sources{7, 12, 12, 12};
    constexpr std::array<int, 4> first{96, 64, 48, 32};
    for (std::size_t b = 0; b < blocks->size(); ++b) {
      INFO("bloc " << b);
      const auto& layers = (*blocks)[b];
      REQUIRE(layers.size() == 11);
      CHECK(layers[0].cin == sources[b]);
      CHECK(layers[0].cout == first[b]);
      CHECK(layers[0].stride == 2);
      CHECK(layers[1].cout == 2 * first[b]);
      CHECK(layers[1].stride == 2);
      const auto channels = static_cast<std::size_t>(2 * first[b]);
      for (std::size_t r = 2; r < 10; ++r) {
        CHECK(layers[r].stride == 1);
        CHECK(layers[r].scale.size() == channels);
        CHECK(layers[r].slope == std::vector<float>(channels, 0.2F));
      }
      const auto& tail = layers.back();
      CHECK(tail.transposed);
      CHECK(tail.cout == 24);
      CHECK(tail.kernel == 4);
      CHECK(tail.stride == 2);
      CHECK(tail.weights.size() == 16 * 24 * channels);
    }
    CHECK_FALSE(gmfss_ifnet_blocks(*derived, 384, 672));
  }
}

}  // namespace kirari::prepare
