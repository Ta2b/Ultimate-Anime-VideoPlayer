// Le grade sur le GPU, ses règles locales : la peau, et les lumières colorées réglées selon leur voisinage.
// Mêmes conditions que grade_gpu_test.cpp (lavapipe en CI, sautés sans appareil Vulkan).
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "grade_gpu_support.h"

namespace ajp::render {

using namespace gpu_test;

TEST_CASE("peau : la courbe HDR ne la pousse pas vers les éclats", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une image uniforme d'une couleur sRGB, en BT.2020 au blanc SDR de 203 nits.
  const auto image = [](double r8, double g8, double b8) {
    const auto lin = [](double v) {
      const double c = v / 255.0;
      return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    const double r = lin(r8), g = lin(g8), b = lin(b8);
    const std::array<double, 3> bt2020{0.6274 * r + 0.3293 * g + 0.0433 * b,
                                       0.0691 * r + 0.9195 * g + 0.0114 * b,
                                       0.0164 * r + 0.0880 * g + 0.8956 * b};
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
      for (std::size_t c = 0; c < 3; ++c)
        pixels[i + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(203.0 * bt2020.at(c)) * 65535.0));
      pixels[i + 3] = 65535;
    }
    return pixels;
  };
  const auto luminance = [](const Rgb& px) { return 0.2627 * px[0] + 0.6780 * px[1] + 0.0593 * px[2]; };
  const Rgb pale = graded_rgb(gpu, image(255, 232, 214), native_look(0.0))[kSide / 2];
  const Rgb mid = graded_rgb(gpu, image(240, 195, 165), native_look(0.0))[kSide / 2];
  const Rgb grey = graded_rgb(gpu, image(236, 236, 236), native_look(0.0))[kSide / 2];
  CHECK(luminance(pale) < 330.0);  // sans la règle : environ 550 nits
  CHECK(luminance(pale) > 220.0);
  CHECK(luminance(mid) < luminance(pale));  // l'ordre des tons reste
  CHECK(luminance(grey) > 450.0);           // un gris clair, lui, monte toujours
  // La teinte de la peau reste la sienne : rouge au-dessus du vert, vert au-dessus du bleu.
  CHECK(pale[0] > pale[1]);
  CHECK(pale[1] > pale[2]);
}

TEST_CASE(
    "lumière colorée : sur un fond clair, plus de renfort de couleur ; sur un fond sombre, tout son éclat",
    "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Un néon magenta au blanc SDR (carré de 16 px au centre) sur un fond gris uniforme.
  const auto neon_on = [](double background) {
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
    for (int y = 0; y < kSide; ++y) {
      for (int x = 0; x < kSide; ++x) {
        const bool neon = std::abs(x - kSide / 2) < 8 && std::abs(y - kSide / 2) < 8;
        const std::array<double, 3> nits = neon ? std::array<double, 3>{203.0, 8.0, 180.0}
                                                : std::array<double, 3>{background, background, background};
        const std::size_t at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
        for (std::size_t c = 0; c < 3; ++c)
          pixels[at + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits.at(c)) * 65535.0));
        pixels[at + 3] = 65535;
      }
    }
    return pixels;
  };
  // Notre HDR adaptatif avec des renforts de couleur francs (vibrance, couleurs étendues, saturation de
  // fin), puis le même sans aucun renfort.
  GradeParams vivid = native_look(0.0);
  vivid.vibrance = 0.6;
  vivid.expand = 0.7;
  vivid.chroma = 1.15;
  const GradeParams plain = native_look(0.0);
  const auto centre = [&](double background, const GradeParams& p) {
    return graded_rgb(gpu, neon_on(background), p)[kSide / 2];
  };
  const Rgb dark_vivid = centre(2.0, vivid);
  const Rgb dark_plain = centre(2.0, plain);
  const Rgb bright_vivid = centre(150.0, vivid);
  const Rgb bright_plain = centre(150.0, plain);
  INFO("sombre " << dark_vivid[0] << " " << saturation(dark_vivid) << " / " << saturation(dark_plain)
                 << " ; clair " << bright_vivid[0] << " / " << bright_plain[0] << " "
                 << saturation(bright_vivid) << " / " << saturation(bright_plain));
  // Sur un fond sombre : les renforts jouent, le néon est plus saturé qu'à nu.
  CHECK(saturation(dark_vivid) > saturation(dark_plain) + 0.01);
  // Sur un fond clair : plus de renfort, le néon reste tel que la courbe le rend.
  CHECK(std::abs(saturation(bright_vivid) - saturation(bright_plain)) < 0.01);
  CHECK(std::abs(bright_vivid[0] - bright_plain[0]) < 0.05 * bright_plain[0]);
  // Il reste lumineux et magenta.
  CHECK(bright_plain[0] > 200.0);
  CHECK(bright_plain[2] > 4.0 * bright_plain[1]);
  // Le fond gris clair, lui, est rendu comme sans néon.
  const Rgb grey_with = graded_rgb(gpu, neon_on(150.0), vivid)[4];
  const Rgb grey_alone = graded_rgb(gpu, uniform(150.0), vivid)[4];
  CHECK(std::abs(grey_with[1] - grey_alone[1]) < 0.02 * grey_alone[1]);
}

TEST_CASE("couleurs : un bleu terne se ravive, un bleu déjà vif ou lumineux n'est presque plus poussé",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une image uniforme d'une couleur sRGB, en BT.2020 au blanc SDR de 203 nits.
  const auto image = [](double r8, double g8, double b8) {
    const auto lin = [](double v) {
      const double c = v / 255.0;
      return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    const double r = lin(r8), g = lin(g8), b = lin(b8);
    const std::array<double, 3> bt2020{0.6274 * r + 0.3293 * g + 0.0433 * b,
                                       0.0691 * r + 0.9195 * g + 0.0114 * b,
                                       0.0164 * r + 0.0880 * g + 0.8956 * b};
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
      for (std::size_t c = 0; c < 3; ++c)
        pixels[i + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(203.0 * bt2020.at(c)) * 65535.0));
      pixels[i + 3] = 65535;
    }
    return pixels;
  };
  GradeParams off = expanded(0.0);
  GradeParams on = off;
  on.vibrance = 1.2;  // le curseur des couleurs poussé fort
  const auto gain = [&](double r8, double g8, double b8) {
    const auto source = image(r8, g8, b8);
    return saturation(graded_rgb(gpu, source, on)[kSide / 2]) -
           saturation(graded_rgb(gpu, source, off)[kSide / 2]);
  };
  const double dull_blue = gain(110, 140, 170);   // ciel pâle, couvert
  const double dull_green = gain(120, 150, 110);  // feuillage terne
  const double vivid_blue = gain(30, 140, 255);   // halo bleu vif, lumineux
  const double vivid_green = gain(40, 230, 150);  // lueur verte
  INFO("terne " << dull_blue << " " << dull_green << " ; vif " << vivid_blue << " " << vivid_green);
  CHECK(dull_blue > 0.04);
  CHECK(dull_green > 0.04);
  CHECK(vivid_blue < 0.04);
  CHECK(vivid_green < 0.04);
  CHECK(vivid_blue < dull_blue / 3.0);
  CHECK(vivid_green < dull_green / 3.0);
}

}  // namespace ajp::render
