#include "kernels/ref/rife.h"

#include <cstdint>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/color.h"
#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Une image YUV aux lumas aléatoires et à la chroma donnée.
YuvImage random_frame(Format f, int w, int h, std::uint16_t cb, std::uint16_t cr, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint32_t> pick(0, max_code(f) >> info(f).shift);
  YuvImage frame(f, w, h, 0, cb, cr);
  for (std::uint16_t& code : frame.y.data) code = static_cast<std::uint16_t>(pick(rng) << info(f).shift);
  return frame;
}

}  // namespace

TEST_CASE("RIFE : remplissage mod-64 centré, bords pairs", "[rife]") {
  static_assert(rife_geometry(1280, 720).padded_height == 768);
  static_assert(rife_geometry(1280, 720).top == 24);
  const auto check = [](int w, int h, int pw, int ph, int left, int top) {
    const RifeGeometry g = rife_geometry(w, h);
    CHECK(g.padded_width == pw);
    CHECK(g.padded_height == ph);
    CHECK(g.left == left);
    CHECK(g.top == top);
  };
  check(1280, 720, 1280, 768, 0, 24);
  check(1920, 1080, 1920, 1088, 0, 4);
  check(3840, 2160, 3840, 2176, 0, 8);
  check(1440, 1080, 1472, 1088, 16, 4);
  check(720, 480, 768, 512, 24, 16);
  check(854, 480, 896, 512, 20, 16);  // (896 − 854) / 2 = 21, arrondi au pair inférieur
  check(1366, 768, 1408, 768, 20, 0);
  check(64, 64, 64, 64, 0, 0);
}

TEST_CASE("RIFE : multiple de 128 pour 4.25 lite, fenêtre lue sur le tenseur rempli", "[rife]") {
  // 1080p : 1088 lignes au multiple de 64, que 4.25 lite refuse ; 1152 au multiple de 128.
  static_assert(rife_geometry(1920, 1080).padded_height == 1088);
  constexpr RifeGeometry lite = rife_geometry(1920, 1080, kRifeAlignLite);
  static_assert(lite.padded_width == 1920 && lite.padded_height == 1152 && lite.left == 0 && lite.top == 36);
  // 720p tombe juste aux deux multiples.
  static_assert(rife_geometry(1280, 720, kRifeAlignLite).padded_height == 768);
  static_assert(rife_geometry(1280, 720, kRifeAlignLite).top == rife_geometry(1280, 720).top);
  // Un backend relit la même fenêtre sur la taille du tenseur, sans connaître le modèle.
  constexpr RifeGeometry read = rife_window(1920, 1080, 1920, 1152);
  static_assert(read.left == lite.left && read.top == lite.top && read.padded_height == lite.padded_height);
  const int w = GENERATE(1, 63, 640, 960, 1280, 1366, 1920, 3840);
  const int h = GENERATE(1, 64, 360, 540, 720, 1080, 2160);
  for (const int align : {kRifeAlign, kRifeAlignLite}) {
    const RifeGeometry g = rife_geometry(w, h, align);
    CHECK(g.padded_width % align == 0);
    CHECK(g.padded_height % align == 0);
    CHECK(rife_fits(w, h, g.padded_width, g.padded_height));
    const RifeGeometry again = rife_window(w, h, g.padded_width, g.padded_height);
    CHECK(again.left == g.left);
    CHECK(again.top == g.top);
  }
  // Trop petit, ou hors des multiples de 64 : refusé.
  CHECK_FALSE(rife_fits(1920, 1080, 1920, 1024));
  CHECK_FALSE(rife_fits(1920, 1080, 1920, 1100));
  CHECK_FALSE(rife_fits(0, 1080, 64, 1088));
}

TEST_CASE("RIFE : bords noir studio, quelle que soit la plage", "[rife]") {
  CHECK(rife_pad_luma(Format::Nv12) == 16);
  CHECK(rife_pad_chroma(Format::Nv12) == 128);
  CHECK(rife_pad_luma(Format::P010) == 4096);
  CHECK(rife_pad_chroma(Format::P010) == 32768);
  CHECK(rife_pad_luma(Format::Yuv420p10) == 64);
  CHECK(rife_pad_luma(Format::Yuv444p16) == 16 * 256);
  CHECK(rife_pad_chroma(Format::Yuv444p16) == 128 * 256);
  CHECK(kRifeMatrix == Matrix::Bt709);
  CHECK(kRifeChromaFilter == Filter::Bilinear);
}

TEST_CASE("RIFE : remplir puis recadrer rend l'image", "[rife]") {
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::Yuv444p16);
  const YuvImage frame = random_frame(format, 100, 60, rife_pad_chroma(format), rife_pad_chroma(format), 3);
  const auto padded = ref::rife_pad(frame);
  REQUIRE(padded);
  const RifeGeometry g = rife_geometry(100, 60);
  REQUIRE(g.left == 14);
  REQUIRE(g.top == 2);
  CHECK(padded->width() == 128);
  CHECK(padded->height() == 64);
  CHECK(padded->y.at(0, 0) == rife_pad_luma(format));
  CHECK(padded->y.at(127, 63) == rife_pad_luma(format));
  CHECK(padded->y.at(14, 2) == frame.y.at(0, 0));
  CHECK(padded->y.at(113, 61) == frame.y.at(99, 59));
  const auto back = ref::rife_crop(*padded, g);
  REQUIRE(back);
  CHECK(*back == frame);
}

TEST_CASE("RIFE : la préparation suit la conversion BT.709 bilinéaire à l'intérieur", "[rife][parity]") {
  // Loin du bord de l'image, la chroma bilinéaire ne lit que des échantillons de l'image :
  // les valeurs remplies égalent celles de l'image convertie seule, au bit près.
  YuvImage frame = random_frame(Format::Nv12, 40, 20, 0, 0, 5);
  std::mt19937 rng(9);
  std::uniform_int_distribution<int> chroma(16, 240);
  for (std::uint16_t& c : frame.u.data) c = static_cast<std::uint16_t>(chroma(rng));
  for (std::uint16_t& c : frame.v.data) c = static_cast<std::uint16_t>(chroma(rng));
  const auto prepared = ref::rife_prepare(frame, Range::Limited);
  const auto alone = ref::yuv_to_rgb(
      frame, {.matrix = Matrix::Bt709, .range = Range::Limited, .siting = Siting::Left}, Filter::Bilinear);
  REQUIRE(prepared);
  REQUIRE(alone);
  const RifeGeometry g = rife_geometry(40, 20);
  for (int y = 1; y < 19; ++y) {
    for (int x = 0; x < 39; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        REQUIRE(prepared->planes[c].at(g.left + x, g.top + y) == alone->planes[c].at(x, y));
      }
    }
  }
  // Le bord est noir studio : RGB nul en plage limitée.
  CHECK_THAT(prepared->planes[1].at(0, 0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("RIFE : en plage pleine, le bord noir studio vaut 16/255", "[rife]") {
  const auto prepared = ref::rife_prepare(YuvImage(Format::Yuv444p16, 64, 32, 0, 32768, 32768), Range::Full);
  REQUIRE(prepared);
  CHECK_THAT(prepared->planes[0].at(0, 0), WithinAbs(4096.0 / 65535.0, 1e-6));
  CHECK_THAT(prepared->planes[0].at(0, 20), WithinAbs(0.0, 1e-6));  // l'image, noire
}

TEST_CASE("RIFE : préparer puis finir rend l'image, luma exacte", "[rife][roundtrip]") {
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::Yuv420p10);
  const YuvImage frame = random_frame(format, 72, 40, rife_pad_chroma(format), rife_pad_chroma(format), 21);
  const RifeGeometry g = rife_geometry(72, 40);
  const auto back = ref::rife_prepare(frame, Range::Limited).and_then([&](const RgbImage& rgb) {
    return ref::rife_finish(rgb, format, Range::Limited, g);
  });
  REQUIRE(back);
  CHECK(*back == frame);

  const YuvImage full = random_frame(Format::Yuv444p16, 50, 30, 1000, 64000, 1);
  const auto full_back = ref::rife_prepare(full, Range::Full).and_then([&](const RgbImage& rgb) {
    return ref::rife_finish(rgb, Format::Yuv444p16, Range::Full, rife_geometry(50, 30));
  });
  REQUIRE(full_back);
  CHECK(*full_back == full);
}

TEST_CASE("RIFE : les 11 plans de l'entrée", "[rife]") {
  const RgbImage a(128, 64, 0.25F);
  const RgbImage b(128, 64, 0.75F);
  const auto input = ref::rife_input(a, b, 0.5F);
  REQUIRE(input);
  REQUIRE(input->size() == kRifeInputChannels);
  const auto& planes = *input;
  CHECK(planes[kRifeChannelA].at(3, 3) == 0.25F);
  CHECK(planes[kRifeChannelB + 2].at(3, 3) == 0.75F);
  CHECK(planes[kRifeChannelTime].at(100, 50) == 0.5F);
  CHECK(planes[kRifeChannelMeshX].at(0, 7) == -1.0F);
  CHECK(planes[kRifeChannelMeshX].at(127, 7) == 1.0F);
  CHECK_THAT(planes[kRifeChannelMeshY].at(5, 32), WithinAbs(2.0 * 32 / 63 - 1, 1e-6));
  CHECK(planes[kRifeChannelStepX].at(9, 9) == 2.0F / 127.0F);
  CHECK(planes[kRifeChannelStepY].at(9, 9) == 2.0F / 63.0F);
}

TEST_CASE("RIFE : les entrées hors contrat sont refusées", "[rife]") {
  CHECK(ref::rife_input(RgbImage(100, 64), RgbImage(100, 64), 0.5F).error().code ==
        ErrorCode::DimensionOutOfRange);
  CHECK(ref::rife_input(RgbImage(128, 64), RgbImage(64, 64), 0.5F).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::rife_input(RgbImage(64, 64), RgbImage(64, 64), 1.5F).error().code ==
        ErrorCode::ParameterOutOfRange);
  RifeGeometry g = rife_geometry(100, 60);
  CHECK(ref::rife_crop(RgbImage(128, 128), g).error().code == ErrorCode::SizeMismatch);
  g.left = 30;
  CHECK(ref::rife_crop(RgbImage(128, 64), g).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::rife_pad(YuvImage(Format::Nv12, 7, 4, 16, 128, 128)).error().code == ErrorCode::OddDimension);
}
