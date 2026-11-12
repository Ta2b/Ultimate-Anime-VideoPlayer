#include "kernels/ref/detect.h"

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/color.h"
#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Les n premiers pixels du bloc (bx, by) de b diffèrent de a de delta.
void touch_block(PlaneU16& b, int bx, int by, int n, int delta) {
  for (int i = 0; i < n; ++i) {
    const int x = bx * kHeldBlock + i % kHeldBlock;
    const int y = by * kHeldBlock + i / kHeldBlock;
    b.at(x, y) = static_cast<std::uint16_t>(b.at(x, y) + delta);
  }
}

}  // namespace

TEST_CASE("coupe de scène : normalisation par la crête du format", "[detect]") {
  CHECK(scene_norm(Format::Nv12) == 1.0F / 255.0F);
  CHECK(scene_norm(Format::P010) == 1.0F / 65472.0F);
  CHECK(scene_norm(Format::Yuv444p16) == 1.0F / 65535.0F);
  CHECK(scene_norm(Format::Yuv420p10) == 1.0F / 1023.0F);
}

TEST_CASE("coupe de scène : somme calculée à la main", "[detect][parity]") {
  // 4 × 2 : six pixels identiques, deux qui diffèrent de 51 et de 102 niveaux 8 bits.
  PlaneU16 a(4, 2, 100);
  PlaneU16 b(4, 2, 100);
  b.at(1, 0) = 151;
  b.at(3, 1) = 202;
  const auto sum = ref::scene_diff_sum(a, b, Format::Nv12);
  REQUIRE(sum);
  CHECK_THAT(*sum, WithinAbs(51.0 / 255.0 + 102.0 / 255.0, 1e-7));
  // P010 : mêmes écarts en 10 bits décalés, crête 65472.
  PlaneU16 pa(4, 2, 100 * 64);
  PlaneU16 pb(4, 2, 100 * 64);
  pb.at(0, 0) = 300 * 64;
  const auto psum = ref::scene_diff_sum(pa, pb, Format::P010);
  REQUIRE(psum);
  CHECK_THAT(*psum, WithinRel(200.0 * 64.0 / 65472.0, 1e-6));
}

TEST_CASE("coupe de scène : le score divise par l'aire remplie de RIFE", "[detect]") {
  // 1280 × 720 rempli en 1280 × 768 : un écart uniforme de 32 niveaux (0,125 en moyenne) donne
  // 0,125 × 720 / 768 = 0,118, sous le seuil ; 34 niveaux donnent 0,125, au-dessus.
  const RifeGeometry g = rife_geometry(1280, 720);
  const PlaneU16 a(1280, 720, 16);
  const auto score32 = ref::scene_score(a, PlaneU16(1280, 720, 48), Format::Nv12, g);
  const auto score34 = ref::scene_score(a, PlaneU16(1280, 720, 50), Format::Nv12, g);
  REQUIRE(score32);
  REQUIRE(score34);
  CHECK_THAT(*score32, WithinRel(32.0 / 255.0 * 720.0 / 768.0, tolerance::kSceneScoreRelative));
  CHECK_FALSE(is_scene_cut(*score32));
  CHECK(is_scene_cut(*score34));
  CHECK_FALSE(is_scene_cut(kSceneCutThreshold));  // strictement au-dessus
  CHECK(is_scene_cut(0.1200001));
  const PlaneU16 small(640, 360, 0);
  CHECK(ref::scene_score(small, small, Format::Nv12, g).error().code == ErrorCode::SizeMismatch);
}

TEST_CASE("dessins tenus : pire bloc 16 × 16 calculé à la main", "[detect][parity]") {
  PlaneU16 a(64, 64, 80);
  PlaneU16 b = a;
  // Bloc (1, 0) : 175 pixels sur 256 à un niveau → 175 × 256 / 256 = 175 (0,68 niveau).
  touch_block(b, 1, 0, 175, 1);
  // Bloc (0, 1) : 25 pixels à 3 niveaux → 75 × 256 / 256 = 75.
  touch_block(b, 0, 1, 25, 3);
  const auto worst = ref::block_diff_max(a, b, Format::Nv12);
  REQUIRE(worst);
  CHECK(*worst == 175);
  CHECK(is_held(*worst));                         // 17500 ≤ 70 × 256 = 17920
  CHECK_FALSE(is_held(*worst, kHeldNoiseFloor));  // le seuil de 0,40 le prendrait pour un mouvement
  // 180 pixels : 180, juste au-dessus de 0,70 niveau.
  touch_block(b, 1, 1, 180, 1);
  const auto moved = ref::block_diff_max(a, b, Format::Nv12);
  REQUIRE(moved);
  CHECK(*moved == 180);
  CHECK_FALSE(is_held(*moved));
}

TEST_CASE("dessins tenus : même échelle en 8, 10 et 16 bits", "[detect]") {
  // Un niveau 8 bits partout dans un bloc donne 256 dans chaque format.
  const auto one_level = [](Format f) {
    const auto unit = static_cast<int>(level_unit(f));
    PlaneU16 a(32, 32, static_cast<std::uint16_t>(64 * unit));
    PlaneU16 b(32, 32, static_cast<std::uint16_t>(65 * unit));
    const auto worst = ref::block_diff_max(a, b, f);
    REQUIRE(worst);
    return *worst;
  };
  CHECK(one_level(Format::Nv12) == 256);
  CHECK(one_level(Format::P010) == 256);
  CHECK(one_level(Format::Yuv420p10) == 256);
  CHECK(one_level(Format::Yuv444p16) == 256);
  CHECK(held_scale(Format::Nv12) == 256);
  CHECK(held_scale(Format::P010) == 1);
  CHECK(held_scale(Format::Yuv420p10) == 64);
}

TEST_CASE("dessins tenus : blocs tronqués au bord, division entière", "[detect]") {
  PlaneU16 a(40, 40, 50);
  PlaneU16 b = a;
  b.at(39, 39) = 52;  // coin 8 × 8 : 2 × 256 / 64 = 8
  const auto corner = ref::block_diff_max(a, b, Format::Nv12);
  REQUIRE(corner);
  CHECK(*corner == 8);
  PlaneU16 c(32, 32, 50);
  PlaneU16 d = c;
  const PlaneU16 e(19, 16, 50);
  PlaneU16 f = e;
  f.at(18, 0) = 51;  // bloc 3 × 16 : 256 / 48 = 5,33, tronqué à 5
  const auto tiny = ref::block_diff_max(e, f, Format::Nv12);
  REQUIRE(tiny);
  CHECK(*tiny == 5);
  CHECK(ref::block_diff_max(c, PlaneU16(32, 16), Format::Nv12).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::block_diff_max(c, PlaneU16(32, 32, 300), Format::Nv12).error().code ==
        ErrorCode::SampleOutOfRange);
}

TEST_CASE("détection : une luma hors de son conteneur est refusée", "[detect]") {
  // P010 : les 6 bits bas sont nuls ; 4097 n'est pas un code P010, comme pour la conversion.
  const PlaneU16 a(32, 32, 4096);
  const PlaneU16 b(32, 32, 4097);
  CHECK(ref::block_diff_max(a, b, Format::P010).error().code == ErrorCode::SampleOutOfRange);
  CHECK(ref::scene_diff_sum(b, a, Format::P010).error().code == ErrorCode::SampleOutOfRange);
  CHECK(ref::scene_diff_sum(a, PlaneU16(32, 32, 1024), Format::Yuv420p10).error().code ==
        ErrorCode::SampleOutOfRange);
  // Un plan dont les données ne couvrent pas les dimensions annoncées.
  PlaneU16 hollow = a;
  hollow.data.resize(10);
  CHECK(ref::block_diff_max(a, hollow, Format::P010).error().code == ErrorCode::SizeMismatch);
}

TEST_CASE("dessins tenus : un clignement lointain n'est pas un dessin tenu (P3.23)", "[detect]") {
  // Un œil de 3 × 2 pixels qui se ferme (40 niveaux) : 240 / 1024 = 0,23 niveau en 32 × 32, tenu à tort ;
  // 240 / 256 = 0,94 en 16 × 16, au-dessus du seuil.
  const PlaneU16 a(1920, 1080, 120);
  PlaneU16 b = a;
  for (int y = 500; y < 502; ++y)
    for (int x = 900; x < 903; ++x) b.at(x, y) = 160;
  const auto worst = ref::block_diff_max(a, b, Format::Nv12);
  REQUIRE(worst);
  CHECK_FALSE(is_held(*worst));
}

TEST_CASE("mesure de paire : la plage de la source fixe les niveaux (C50-01)", "[detect]") {
  // Deux aplats gris qui sautent, comme une coupe : le contrat YUV (scene_score) et la mesure du lecteur sur
  // le RGB ingéré doivent s'accorder dans les deux plages. Avec 219 en plage pleine, le score sortait 14,1 %
  // trop bas.
  const int w = 256;
  const int h = 128;
  const RifeGeometry g = rife_geometry(w, h);
  const YuvImage a(Format::Nv12, w, h, 60, 128, 128);
  const YuvImage b(Format::Nv12, w, h, 110, 128, 128);
  const auto contract = ref::scene_score(a.y, b.y, Format::Nv12, g);
  REQUIRE(contract);
  for (const Range range : {Range::Limited, Range::Full}) {
    INFO("plage " << (range == Range::Full ? "pleine" : "limitée"));
    const ref::ColorSpec color{.matrix = Matrix::Bt709, .range = range, .siting = Siting::Left};
    const auto ra = ref::yuv_to_rgb(a, color);
    const auto rb = ref::yuv_to_rgb(b, color);
    REQUIRE(ra);
    REQUIRE(rb);
    // Le RGB ingéré garde l'échelle de la plage : en limitée, 1,0 = 219 pas ; en pleine, 255.
    const auto measure = ref::pair_scores(*ra, *rb, pair_levels(range), g);
    REQUIRE(measure);
    CHECK_THAT(measure->scene_score, WithinRel(*contract, tolerance::kSceneScoreRelative));
    CHECK(is_scene_cut(measure->scene_score) == is_scene_cut(*contract));
  }
  // L'ancienne mesure (219 partout) s'écartait en plage pleine : le test ci-dessus le verrait.
  const ref::ColorSpec full{.matrix = Matrix::Bt709, .range = Range::Full, .siting = Siting::Left};
  const auto fa = ref::yuv_to_rgb(a, full);
  const auto fb = ref::yuv_to_rgb(b, full);
  REQUIRE(fa);
  REQUIRE(fb);
  const auto wrong = ref::pair_scores(*fa, *fb, 219.0F, g);
  REQUIRE(wrong);
  CHECK(std::fabs(wrong->scene_score - *contract) > 0.1 * *contract);
}

TEST_CASE("mesure de paire : blocs, pire bloc et refus", "[detect]") {
  const int w = 100;
  const int h = 70;
  const RifeGeometry g = rife_geometry(w, h);
  RgbImage a(w, h, 0.25F);
  RgbImage b = a;
  for (int y = 64; y < h; ++y) b.planes[1].at(w - 1, y) += 0.5F;  // un bloc du bord, tronqué à 4 × 6
  const auto m = ref::pair_scores(a, b, 255.0F, g);
  REQUIRE(m);
  REQUIRE(m->block_sums.size() == 7U * 5U);
  CHECK(m->block_sums.back() > 0.0);
  const float want_mean = 0.7152F * 0.5F * 6.0F / 24.0F;  // 6 pixels sur les 24 du bloc tronqué
  CHECK(m->worst_block == static_cast<std::uint32_t>(want_mean * 255.0F * 256.0F));
  CHECK(ref::pair_scores(a, a, 255.0F, g)->total == 0.0);
  CHECK(ref::pair_scores(a, RgbImage(w - 2, h, 0.0F), 255.0F, g).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::pair_scores(a, b, 0.0F, g).error().code == ErrorCode::ParameterOutOfRange);
}
