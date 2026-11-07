#include "render/cube_lut.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;

namespace ajp::render {
namespace {

std::string cube_text(int size, const std::string& header = "") {
  std::string text = header + std::format("LUT_3D_SIZE {}\n", size);
  const CubeLut identity = identity_lut(size);
  for (std::size_t i = 0; i < identity.rgb.size(); i += 3)
    text += std::format("{} {} {}\n", identity.rgb[i], identity.rgb[i + 1], identity.rgb[i + 2]);
  return text;
}

}  // namespace

TEST_CASE("un .cube d'identité se relit tel quel, R variant le plus vite", "[render][lut]") {
  const auto lut = parse_cube(cube_text(17, "# commentaire\nTITLE \"Neutre\"\n\n"));
  REQUIRE(lut);
  CHECK(lut->size == 17);
  CHECK(lut->title == "Neutre");
  CHECK(lut->rgb.size() == 17U * 17U * 17U * 3U);
  CHECK(lut->rgb[3] == Approx(1.0F / 16.0F));           // le deuxième point : R avance d'un cran
  CHECK(lut->rgb[3 * 17 + 1] == Approx(1.0F / 16.0F));  // G après 17 points
  CHECK(lut->rgb.size() == identity_lut(17).rgb.size());
}

TEST_CASE("DOMAIN_MIN et DOMAIN_MAX bornent l'entrée, pas la sortie (P3.69)", "[render][lut]") {
  // L'identité sur le domaine [0, 2] (la référence d'OpenColorIO : une entrée de 0,5 lit le quart de la
  // grille et rend 0,5) ; les sorties restent celles du fichier.
  std::string text = "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\nLUT_3D_SIZE 17\n";
  for (int b = 0; b < 17; ++b)
    for (int g = 0; g < 17; ++g)
      for (int r = 0; r < 17; ++r) text += std::format("{} {} {}\n", r / 8.0, g / 8.0, b / 8.0);
  const auto lut = parse_cube(text);
  REQUIRE(lut);
  CHECK(lut->rgb[3 * 16] == Approx(2.0F));  // la dernière valeur de R, non ramenée à 1
  const std::array<float, 3> out = sample_lut(*lut, {0.5F, 0.25F, 1.5F});
  CHECK(out[0] == Approx(0.5F));
  CHECK(out[1] == Approx(0.25F));
  CHECK(out[2] == Approx(1.5F));
}

TEST_CASE("refus nommés : taille, lignes, nombres, LUT 1D, taille de fichier", "[render][lut]") {
  CHECK_FALSE(parse_cube(cube_text(16)));
  CHECK_FALSE(parse_cube(cube_text(66)));
  CHECK_FALSE(parse_cube("LUT_1D_SIZE 256\n0 0 0\n"));
  CHECK_FALSE(parse_cube("0 0 0\n"));  // données sans LUT_3D_SIZE
  CHECK_FALSE(parse_cube(""));
  std::string short_text = cube_text(17);
  short_text.resize(short_text.size() - 40);
  CHECK_FALSE(parse_cube(short_text));                 // une ligne de moins
  CHECK_FALSE(parse_cube(cube_text(17) + "0 0 0\n"));  // une ligne de trop
  CHECK_FALSE(parse_cube(cube_text(17, "LUT_BIDON 1\n")));
  std::string bad = cube_text(17);
  bad.replace(bad.rfind('\n', bad.size() - 2), 1, "\nx y z\n");
  CHECK_FALSE(parse_cube(bad));
  std::string nan = "LUT_3D_SIZE 17\n" + std::string(1, 'n') + "an 0 0\n";
  CHECK_FALSE(parse_cube(nan));
  CHECK_FALSE(parse_cube(std::string(kLutMaxBytes + 1, ' ')));
  CHECK_FALSE(parse_cube("DOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\nLUT_3D_SIZE 17\n0 0 0\n"));
}

TEST_CASE("les demi-flottants arrondissent au plus proche", "[render][lut]") {
  CHECK(to_half(0.0F) == 0x0000);
  CHECK(to_half(1.0F) == 0x3c00);
  CHECK(to_half(0.5F) == 0x3800);
  CHECK(to_half(2.0F) == 0x4000);
  CHECK(to_half(1.0F / 3.0F) == 0x3555);
  CHECK(to_half(1.0e-9F) == 0x0000);
}

TEST_CASE("le bloc de texture annonce sa taille et porte 8 octets par point", "[render][lut]") {
  const CubeLut lut = identity_lut(17);
  const std::string block = lut_texture_block(lut, "AJG_LUT");
  CHECK(block.starts_with("//!TEXTURE AJG_LUT\n//!SIZE 17 17 17\n//!FORMAT rgba16hf\n"));
  std::size_t hex = 0;
  const std::size_t data = block.find("CLAMP\n") + 6;
  for (std::size_t i = data; i < block.size(); ++i)
    if (block[i] != '\n') ++hex;
  CHECK(hex == 17U * 17U * 17U * 16U);  // 4 composantes × 2 octets × 2 chiffres
}

}  // namespace ajp::render

#include "render/looks.h"

namespace ajp::render {

TEST_CASE("les looks maison sont des LUT valides, bornées, qui gardent le noir et le blanc",
          "[render][lut][looks]") {
  REQUIRE(builtin_looks().size() == 5);
  for (const BuiltinLook& look : builtin_looks()) {
    const auto lut = builtin_look(look.key);
    REQUIRE(lut);
    CHECK(lut->size == 33);
    for (const float v : lut->rgb) {
      CHECK(v >= 0.0F);
      CHECK(v <= 1.0F);
    }
    // Un look rend ses propres LUT lisibles : elles passent l'analyseur une fois écrites.
    CHECK(lut->rgb.size() == 33U * 33U * 33U * 3U);
    // Le gris du milieu reste un gris proche (pas de dérive de teinte énorme).
    const std::size_t mid = (16U * 33U * 33U + 16U * 33U + 16U) * 3U;
    CHECK(std::abs(lut->rgb[mid] - lut->rgb[mid + 1]) < 0.08F);
  }
  CHECK_FALSE(builtin_look("maison:inconnu"));
}

TEST_CASE("le look explosif (B6) pousse plus loin que le vif, contraste et couleurs",
          "[render][lut][looks]") {
  const auto vivid = builtin_look("maison:vif");
  const auto loud = builtin_look("maison:explosif");
  REQUIRE(vivid);
  REQUIRE(loud);
  const auto at = [](int r, int g, int b) {
    return (static_cast<std::size_t>(b) * 33U * 33U + static_cast<std::size_t>(g) * 33U +
            static_cast<std::size_t>(r)) *
           3U;
  };
  // Un gris sombre plus sombre, un gris clair plus clair.
  CHECK(loud->rgb[at(8, 8, 8)] < vivid->rgb[at(8, 8, 8)]);
  CHECK(loud->rgb[at(24, 24, 24)] > vivid->rgb[at(24, 24, 24)]);
  // Un bleu moyen plus saturé : moins de rouge dedans.
  CHECK(loud->rgb[at(8, 10, 22)] <= vivid->rgb[at(8, 10, 22)]);
  // Le noir et le blanc tiennent.
  CHECK(loud->rgb[at(0, 0, 0)] == 0.0F);
  CHECK(loud->rgb[at(32, 32, 32)] > 0.99F);
}

TEST_CASE("les looks maison changent l'image : le vif sature, le pastel relève les noirs",
          "[render][lut][looks]") {
  const auto vivid = builtin_look("maison:vif");
  const auto pastel = builtin_look("maison:pastel");
  REQUIRE(vivid);
  REQUIRE(pastel);
  const std::size_t red = (0U * 33U * 33U + 0U * 33U + 24U) * 3U;  // un rouge moyen
  CHECK(vivid->rgb[red] > 0.75F);
  CHECK(vivid->rgb[red + 1] < 0.02F);
  CHECK(pastel->rgb[0] > 0.02F);  // le noir relevé
}

TEST_CASE("une LUT d'HDR se reconnaît à son nom ou à son titre", "[render][lut][looks]") {
  CHECK(look_is_pq("HDR_grading_LUT_ver9.1.cube", ""));
  CHECK(look_is_pq("look.cube", "PQ Rec2020"));
  CHECK(look_is_pq("rec2020-warm.cube", ""));
  CHECK_FALSE(look_is_pq("teal-orange.cube", "Teal Orange"));
}

}  // namespace ajp::render

namespace ajp::render {

TEST_CASE("les entrées noires d'un gamut d'entrée inutile sont recopiées du voisin valide", "[render][lut]") {
  CubeLut lut = identity_lut(17);
  const auto index = [&](std::size_t r, std::size_t g, std::size_t b) {
    return (b * 17U * 17U + g * 17U + r) * 3U;
  };
  // Une zone invalide : sortie noire alors que l'entrée ne l'est pas (coin du cube).
  for (std::size_t b = 12; b < 17; ++b)
    for (std::size_t g = 12; g < 17; ++g)
      for (std::size_t r = 12; r < 17; ++r)
        lut.rgb[index(r, g, b)] = lut.rgb[index(r, g, b) + 1] = lut.rgb[index(r, g, b) + 2] = 0.0F;
  const std::size_t filled = fill_unreachable(lut);
  CHECK(filled == 5U * 5U * 5U);
  CHECK(lut.rgb[index(16, 16, 16)] > 0.6F);  // recopié de la périphérie valide, plus noir
  CHECK(lut.rgb[index(14, 14, 14) + 1] > 0.6F);
  // Le noir réel (l'origine) et une LUT sans zone invalide restent intacts.
  CubeLut neutral = identity_lut(17);
  CHECK(fill_unreachable(neutral) == 0);
  CHECK(neutral.rgb == identity_lut(17).rgb);
}

}  // namespace ajp::render

namespace ajp::render {
namespace {

// Une LUT de `size` points d'une fonction couleur f.
template <typename F>
CubeLut lut_of(int size, F f) {
  CubeLut lut;
  lut.size = size;
  const float step = 1.0F / static_cast<float>(size - 1);
  for (int b = 0; b < size; ++b)
    for (int g = 0; g < size; ++g)
      for (int r = 0; r < size; ++r) {
        const std::array<float, 3> c =
            f(static_cast<float>(r) * step, static_cast<float>(g) * step, static_cast<float>(b) * step);
        lut.rgb.insert(lut.rgb.end(), c.begin(), c.end());
      }
  return lut;
}

float at(const CubeLut& lut, int r, int g, int b, int ch) {
  const auto un = [](int v) { return static_cast<std::size_t>(v); };
  return lut.rgb[((((un(b) * un(lut.size)) + un(g)) * un(lut.size) + un(r)) * 3) + un(ch)];
}

TEST_CASE("compose_luts : une LUT de 33 points se refait à 65 sans écart, mélange compris", "[render][lut]") {
  const CubeLut warm = lut_of(33, [](float r, float g, float b) {
    return std::array<float, 3>{std::sqrt(r), g * g, (b * 0.5F) + 0.25F};
  });
  const std::array<const CubeLut*, 2> luts{&warm, nullptr};
  const std::array<double, 2> mixes{0.5, 1.0};
  const CubeLut stack = compose_luts(luts, mixes);
  REQUIRE(stack.size == 65);
  // Les points pairs de la grille de 65 sont ceux de la grille de 33.
  for (const int i : {0, 7, 19, 32}) {
    CHECK(at(stack, 2 * i, 2 * i, 2 * i, 0) ==
          Catch::Approx((at(warm, i, i, i, 0) + (static_cast<float>(i) / 32.0F)) * 0.5F).margin(1e-5));
  }
  // Un point impair : le milieu de deux points de la grosse grille, comme la lecture trilinéaire.
  CHECK(at(stack, 3, 0, 0, 0) ==
        Catch::Approx(((at(warm, 1, 0, 0, 0) + at(warm, 2, 0, 0, 0)) * 0.25F) + (3.0F / 64.0F * 0.5F))
            .margin(1e-5));
}

TEST_CASE("compose_luts : deux LUT s'enchaînent dans l'ordre, une intensité nulle est sautée",
          "[render][lut]") {
  const CubeLut half = lut_of(
      17, [](float r, float g, float b) { return std::array<float, 3>{r * 0.5F, g * 0.5F, b * 0.5F}; });
  const CubeLut lift = lut_of(
      17, [](float r, float g, float b) { return std::array<float, 3>{std::min(r + 0.25F, 1.0F), g, b}; });
  const std::array<const CubeLut*, 3> luts{&half, &lift, &half};
  const CubeLut stack = compose_luts(luts, std::array<double, 3>{1.0, 1.0, 0.0});
  // Blanc : divisé par deux, puis relevé de 0,25 en rouge ; la troisième LUT, à 0, ne joue pas.
  CHECK(at(stack, 64, 64, 64, 0) == Catch::Approx(0.75F));
  CHECK(at(stack, 64, 64, 64, 1) == Catch::Approx(0.5F));
  // L'ordre compte : relever puis diviser donnerait 0,5 au lieu de 0,25 sur un rouge nul.
  CHECK(at(stack, 0, 0, 0, 0) == Catch::Approx(0.25F));
  const CubeLut none = compose_luts(luts, std::array<double, 3>{0.0, 0.0, 0.0});
  CHECK(at(none, 32, 16, 48, 0) == Catch::Approx(0.5F));
  CHECK(at(none, 32, 16, 48, 2) == Catch::Approx(0.75F));
}

TEST_CASE("write_cube : la LUT écrite se relit à 1e-6 près, titre et taille compris", "[render][lut]") {
  CubeLut lut;
  lut.size = 17;
  lut.title = "ajp anime HDR (pile composee)";
  lut.rgb.resize(17 * 17 * 17 * 3);
  for (std::size_t i = 0; i < lut.rgb.size(); ++i) lut.rgb[i] = static_cast<float>((i * 37) % 1000) / 999.0F;
  const auto back = parse_cube(write_cube(lut));
  REQUIRE(back);
  CHECK(back->size == 17);
  CHECK(back->title == lut.title);
  REQUIRE(back->rgb.size() == lut.rgb.size());
  for (std::size_t i = 0; i < lut.rgb.size(); ++i) CHECK(back->rgb[i] == Approx(lut.rgb[i]).margin(1e-6));
}

}  // namespace
}  // namespace ajp::render
