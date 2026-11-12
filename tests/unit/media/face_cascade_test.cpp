// La cascade de visages (media/face_cascade.h) : la cascade embarquée se lit entière, et l'évaluation suit
// l'ordre des bits des motifs LBP du format d'OpenCV (une cascade d'un trait, faite ici). Sa justesse sur de
// vrais visages d'anime a été vérifiée contre OpenCV (detectMultiScale) hors du dépôt : aucune image n'y
// entre.
#include "media/face_cascade.h"

#include <array>
#include <format>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace ajp::media {
namespace {

// Une cascade d'un étage et d'un trait (trois blocs de 8 de côté dans une fenêtre de 24) qui n'accepte que
// le motif `code`.
std::string one_code_cascade(unsigned code) {
  std::array<std::int32_t, 8> subset{};
  subset.at(code >> 5U) = static_cast<std::int32_t>(1U << (code & 31U));
  std::string words;
  for (const std::int32_t w : subset) words += std::format(" {}", w);
  return std::format(
      "<opencv_storage><cascade><stageType>BOOST</stageType><featureType>LBP</featureType>"
      "<height>24</height><width>24</width><stageNum>1</stageNum><stages><_><maxWeakCount>1</maxWeakCount>"
      "<stageThreshold>0.</stageThreshold><weakClassifiers><_><internalNodes>0 -1 0{}</internalNodes>"
      "<leafValues>1. -1.</leafValues></_></weakClassifiers></_></stages>"
      "<features><_><rect>0 0 8 8</rect></_></features></cascade></opencv_storage>",
      words);
}

// Une image de 24 × 24 en neuf blocs de 8 : le centre à 100, les voisins à 50, sauf le bloc (col, row) à 200.
GrayImage blocks(int col, int row) {
  GrayImage image{.width = 24, .height = 24, .pixels = {}};
  image.pixels.resize(24 * 24);
  for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 24; ++x) {
      const int c = x / 8;
      const int r = y / 8;
      const int v = (c == 1 && r == 1) ? 100 : (c == col && r == row) ? 200 : 50;
      image.pixels[static_cast<std::size_t>((y * 24) + x)] = static_cast<std::uint8_t>(v);
    }
  return image;
}

}  // namespace

TEST_CASE("cascade de visages : la cascade embarquée se lit entière", "[media][faces]") {
  const FaceCascade* cascade = FaceCascade::anime();
  REQUIRE(cascade != nullptr);
  CHECK(cascade->stages() == 20);
  CHECK(cascade->window_width() == 24);
  CHECK(cascade->window_height() == 24);
  // Un fichier tronqué, ou une cascade d'un autre type, ne se lit pas.
  CHECK_FALSE(FaceCascade::parse(kAnimeFaceCascade.substr(0, kAnimeFaceCascade.size() / 2)).has_value());
  CHECK_FALSE(FaceCascade::parse("<stageType>BOOST</stageType><featureType>HAAR</featureType>").has_value());
  CHECK_FALSE(FaceCascade::parse("").has_value());
}

TEST_CASE("cascade de visages : les huit voisins dans le sens des aiguilles d'une montre, bit 7 d'abord",
          "[media][faces]") {
  // Le voisin plus clair que le centre, et le bit qu'il pose : coin haut gauche (7), haut (6), haut droit
  // (5), droite (4), bas droit (3), bas (2), bas gauche (1), gauche (0).
  constexpr std::array<std::array<int, 3>, 8> kNeighbours{
      {{0, 0, 7}, {1, 0, 6}, {2, 0, 5}, {2, 1, 4}, {2, 2, 3}, {1, 2, 2}, {0, 2, 1}, {0, 1, 0}}};
  const CascadeOptions one{.min_neighbors = 0};
  for (const auto& [col, row, bit] : kNeighbours) {
    INFO("voisin " << col << ", " << row);
    const unsigned code = 1U << static_cast<unsigned>(bit);
    const auto cascade = FaceCascade::parse(one_code_cascade(code));
    REQUIRE(cascade.has_value());
    // L'égalisation garde l'ordre des niveaux : le motif ne change pas.
    CHECK(cascade->detect(blocks(col, row), one).size() == 1);
    // Un autre motif (le voisin d'à côté) est refusé.
    const auto other = FaceCascade::parse(one_code_cascade(code == 128U ? 1U : code << 1U));
    REQUIRE(other.has_value());
    CHECK(other->detect(blocks(col, row), one).empty());
  }
}

TEST_CASE("cascade de visages : une image unie ou trop petite n'a pas de visage", "[media][faces]") {
  const FaceCascade* cascade = FaceCascade::anime();
  REQUIRE(cascade != nullptr);
  GrayImage flat{.width = 160, .height = 90, .pixels = {}};
  flat.pixels.assign(160 * 90, 128);
  CHECK(cascade->detect(flat).empty());
  GrayImage tiny{.width = 20, .height = 20, .pixels = {}};
  tiny.pixels.assign(20 * 20, 128);
  CHECK(cascade->detect(tiny).empty());
}

TEST_CASE("cascade de visages : l'égalisation étale les niveaux et garde leur ordre", "[media][faces]") {
  GrayImage image{.width = 4, .height = 1, .pixels = {10, 20, 20, 30}};
  equalize(image);
  CHECK(image.pixels == std::vector<std::uint8_t>{0, 170, 170, 255});
}

}  // namespace ajp::media
