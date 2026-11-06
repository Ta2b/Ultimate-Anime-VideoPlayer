// Le grade HDR (méthode « HDR anime ») sur le GPU : les surfaces gardent leur niveau d'un plan à
// l'autre et leur grain, seuls les émetteurs montent. Mêmes conditions que grade_gpu_test.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_gpu_support.h"

namespace ajp::render {

using namespace gpu_test;

namespace {

// Une image dont chaque colonne x est un gris de `nits(x)` (les lignes sont identiques).
template <typename F>
std::vector<std::uint16_t> columns(F nits) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (std::size_t y = 0; y < static_cast<std::size_t>(kSide); ++y) {
    for (std::size_t x = 0; x < static_cast<std::size_t>(kSide); ++x) {
      const double e = std::clamp(nits_to_pq(nits(static_cast<int>(x))), 0.0, 1.0);
      const auto code = static_cast<std::uint16_t>(std::lround(e * 65535.0));
      std::uint16_t* px = &pixels[(y * static_cast<std::size_t>(kSide) + x) * 4];
      px[0] = px[1] = px[2] = code;
      px[3] = 65535;
    }
  }
  return pixels;
}

GradeParams anime_params() {
  GradeParams p;
  p.peak = kPeak;
  p.shadows = 0.0;
  return p;
}

double luma(const Rgb& px) {
  return 0.2627 * px[0] + 0.6780 * px[1] + 0.0593 * px[2];
}

}  // namespace

TEST_CASE("HDR anime : une surface garde son niveau et son grain", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  for (const double level : {60.0, 160.0, 190.0}) {
    // Un grain d'un damier de ±3 % de luminance sur une surface uniforme (peau pâle, mur, chemise).
    const auto line =
        graded_rgb(gpu, columns([&](int x) { return level * (x % 2 == 0 ? 1.03 : 0.97); }), anime_params());
    double mean_in = 0.0, mean_out = 0.0, dev_out = 0.0;
    for (int x = 32; x < kSide - 32; ++x) {
      mean_in += level * (x % 2 == 0 ? 1.03 : 0.97);
      mean_out += luma(line[static_cast<std::size_t>(x)]);
    }
    const double n = kSide - 64;
    mean_in /= n;
    mean_out /= n;
    for (int x = 32; x < kSide - 32; ++x)
      dev_out += std::abs(luma(line[static_cast<std::size_t>(x)]) - mean_out);
    dev_out /= n;
    INFO("niveau " << level << " : sortie " << mean_out << " nits");
    // Le relèvement fixe (1,15), sans montée vers les éclats ni décision par pixel.
    CHECK(mean_out == Catch::Approx(1.15 * mean_in).epsilon(0.03));
    // Le grain garde son contraste relatif (3 % de la luminance, à 10 % près).
    CHECK(dev_out / mean_out == Catch::Approx(0.03).epsilon(0.10));
  }
}

TEST_CASE("HDR anime : une lumière isolée monte, son entourage non", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Un halo clair de 48 colonnes (195 nits) sur un fond sombre (3 nits).
  const auto line =
      graded_rgb(gpu, columns([](int x) { return x >= 104 && x < 152 ? 195.0 : 3.0; }), anime_params());
  const double centre = luma(line[128]);
  const double dark = luma(line[16]);
  INFO("halo " << centre << " nits, fond " << dark << " nits");
  CHECK(centre > 1.8 * 195.0);
  CHECK(centre <= kPeak);
  CHECK(dark == Catch::Approx(1.15 * 3.0).epsilon(0.05));
}

TEST_CASE("HDR anime : une peau claire près d'un fond sombre n'est pas une lumière (audit F2)",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une bande de peau claire (255/232/214 en sRGB, ramenée en BT.2020 linéaire à un blanc de 203 nits) de
  // 48 colonnes sur un fond sombre : la même forme que la lumière isolée ci-dessus.
  constexpr std::array<double, 3> kSkin{187.2, 166.2, 140.0};
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (std::size_t y = 0; y < static_cast<std::size_t>(kSide); ++y) {
    for (std::size_t x = 0; x < static_cast<std::size_t>(kSide); ++x) {
      std::uint16_t* px = &pixels[(y * static_cast<std::size_t>(kSide) + x) * 4];
      for (std::size_t c = 0; c < 3; ++c) {
        const double nits = x >= 104 && x < 152 ? kSkin[c] : 3.0;
        px[c] = static_cast<std::uint16_t>(std::lround(std::clamp(nits_to_pq(nits), 0.0, 1.0) * 65535.0));
      }
      px[3] = 65535;
    }
  }
  const auto line = graded_rgb(gpu, pixels, anime_params());
  const double skin_in = 0.2627 * kSkin[0] + 0.6780 * kSkin[1] + 0.0593 * kSkin[2];
  for (const std::size_t x : {std::size_t{106}, std::size_t{128}, std::size_t{149}}) {
    INFO("colonne " << x << " : " << luma(line[x]) << " nits");
    CHECK(luma(line[x]) == Catch::Approx(1.15 * skin_in).epsilon(0.05));  // le relèvement des surfaces seul
  }
}

TEST_CASE("HDR anime : le curseur Éclats agit même sans l'ancien curseur des éclats", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // La configuration de Tatoub : highlights_rtx à 0, donc spec_gain = 1.
  const auto halo = columns([](int x) { return x >= 104 && x < 152 ? 195.0 : 3.0; });
  const auto centre = [&](double an_spec) {
    GradeParams p = anime_params();
    p.spec_gain = 1.0;
    p.an_spec = an_spec;
    return luma(graded_rgb(gpu, halo, p)[128]);
  };
  const double off = centre(0.0);
  const double plain = centre(1.0);
  const double strong = centre(2.0);
  INFO("0 % " << off << ", 100 % " << plain << ", 200 % " << strong);
  CHECK(plain == Catch::Approx(1.15 * 195.0).epsilon(0.05));  // 100 % : la méthode telle quelle
  CHECK(strong > 1.8 * plain);
  CHECK(off < 0.85 * plain);
}

TEST_CASE("HDR anime : route adaptative (paper = demi-pic), la lumière monte aussi", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Les valeurs de decide_grade pour l'aspect adaptatif : blanc papier à la moitié du pic, blanc SDR à 203.
  GradeParams p = anime_params();
  p.paper = 0.5 * kPeak;
  p.adapt = 1.0;
  p.sdr_white = 203.0;
  p.spec_gain = 3.0;
  // Une lumière à 195 nits (presque le blanc SDR) sur un fond sombre.
  const auto line = graded_rgb(gpu, columns([](int x) { return x >= 104 && x < 152 ? 195.0 : 3.0; }), p);
  const double centre = luma(line[128]);
  INFO("lumière " << centre << " nits");
  CHECK(centre > 1.8 * 195.0);
  CHECK(centre <= kPeak);
  // Une surface moyenne (60 nits, un mur) reste au relèvement fixe.
  const auto wall = graded_rgb(gpu, columns([](int) { return 60.0; }), p);
  CHECK(luma(wall[128]) == Catch::Approx(1.15 * 60.0).epsilon(0.03));
}

TEST_CASE("HDR anime : sortie SDR inchangée", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.9, 0.8);
  const std::vector<Rgb> out = graded_codes(gpu, source, sdr_neutral());
  for (std::size_t x = 8; x < out.size(); x += 16) {
    const double expected = 1.0 * (static_cast<double>(x) + 1.0) / kSide;
    CHECK(out[x][0] == Catch::Approx(expected).margin(0.01));
  }
}

TEST_CASE("HDR anime vers un écran SDR : le rendu PQ revient au SDR, identique sous le genou",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Sans relèvement des surfaces ni éclat, l'aller-retour SDR → PQ → SDR rend l'image ; au-dessus du genou
  // (70 % du blanc des surfaces) la courbe compresse, sans jamais dépasser le blanc ni redescendre.
  GradeParams p = anime_params();
  p.an_lift = 1.0;
  p.spec_gain = 1.0;
  sdr_tonemap(p);
  p.sdr_gamma = 2.2;
  const std::vector<Rgb> out = graded_codes(gpu, sdr_ramp(1.0, 1.0, 1.0), p);
  double last = 0.0;
  for (std::size_t x = 0; x < out.size(); ++x) {
    const double in = (static_cast<double>(x) + 1.0) / kSide;
    INFO("x " << x);
    if (std::pow(in, 2.2) < 0.65) CHECK(out[x][0] == Catch::Approx(in).margin(0.02));
    CHECK(out[x][0] <= 1.0);
    CHECK(out[x][0] >= last - 0.01);
    last = out[x][0];
  }
}

TEST_CASE("HDR anime : le peps (contraste, saturation) se règle, identité à 100 %", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une surface sombre (30 nits) et une claire (150 nits) : le contraste écarte les deux autour du pivot.
  const auto grays = columns([](int x) { return x < kSide / 2 ? 30.0 : 150.0; });
  const auto plain = graded_rgb(gpu, grays, anime_params());
  GradeParams punchy = anime_params();
  punchy.an_punch = 0.5;
  const auto lifted = graded_rgb(gpu, grays, punchy);
  const auto dark = static_cast<std::size_t>(kSide / 4);
  const auto bright = static_cast<std::size_t>(3 * kSide / 4);
  CHECK(luma(lifted[dark]) < luma(plain[dark]));
  CHECK(luma(lifted[bright]) > luma(plain[bright]));

  // Une surface colorée : la saturation écarte les canaux de la luminance, sans canal négatif.
  std::vector<std::uint16_t> tinted = columns([](int) { return 80.0; });
  for (std::size_t i = 0; i < tinted.size(); i += 4)
    tinted[i + 2] = static_cast<std::uint16_t>(std::lround(nits_to_pq(40.0) * 65535.0));
  const auto base = graded_rgb(gpu, tinted, anime_params());
  GradeParams vivid = anime_params();
  vivid.an_chroma = 1.5;
  const auto more = graded_rgb(gpu, tinted, vivid);
  const auto mid = static_cast<std::size_t>(kSide / 2);
  CHECK(more[mid][0] - more[mid][2] > 1.2 * (base[mid][0] - base[mid][2]));
  CHECK(more[mid][2] >= 0.0);
}

TEST_CASE("HDR anime : les noirs profonds suivent la scène, nuls en scène très sombre", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const auto dim = columns([](int) { return 2.0; });  // un gris sombre, sous le genou des noirs
  GradeParams night = anime_params();
  night.shadows = 1.0;
  night.fs_apl = 1.0;  // scène très sombre
  GradeParams none = night;
  none.shadows = 0.0;
  GradeParams day = night;
  day.fs_apl = 150.0;  // scène très claire
  const auto mid = static_cast<std::size_t>(kSide / 2);
  const double plain = luma(graded_rgb(gpu, dim, none)[mid]);
  CHECK(luma(graded_rgb(gpu, dim, night)[mid]) == Catch::Approx(plain).epsilon(0.01));
  CHECK(luma(graded_rgb(gpu, dim, day)[mid]) < 0.9 * plain);
}

}  // namespace ajp::render
