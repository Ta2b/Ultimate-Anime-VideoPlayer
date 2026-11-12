// La peau des visages (media/faces.h) : doublons, teinte d'un point, palette d'un visage, palettes d'un plan.
#include "media/faces.h"

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::media {

using Catch::Approx;

TEST_CASE("visages : les doublons se suppriment, le meilleur score reste", "[media][faces]") {
  const FaceBox a{.x0 = 0.1F, .y0 = 0.1F, .x1 = 0.3F, .y1 = 0.3F, .score = 0.6F};
  const FaceBox b{.x0 = 0.11F, .y0 = 0.1F, .x1 = 0.31F, .y1 = 0.3F, .score = 0.9F};
  const FaceBox c{.x0 = 0.6F, .y0 = 0.6F, .x1 = 0.8F, .y1 = 0.8F, .score = 0.7F};
  CHECK(face_iou(a, a) == Approx(1.0));
  CHECK(face_iou(a, c) == 0.0F);
  const auto kept = suppress_faces({a, b, c});
  REQUIRE(kept.size() == 2);
  CHECK(kept[0] == b);
  CHECK(kept[1] == c);
}

TEST_CASE("visages : la teinte d'un point se mesure comme dans le grade", "[media][faces]") {
  // Une peau claire d'anime (BT.2020 linéaire) : orangée, peu saturée.
  const auto skin = skin_tone_of(1.0F, 0.75F, 0.6F);
  REQUIRE(skin.has_value());
  CHECK(skin->hue == Approx(60.0 * 0.15 / 0.4));
  CHECK(skin->sat == Approx(0.4));
  CHECK(skin_wide(*skin) == Approx(1.0));
  // Hors de l'ordre R ≥ V ≥ B (un bleu), ou noir : pas une peau.
  CHECK_FALSE(skin_tone_of(0.2F, 0.3F, 0.9F).has_value());
  CHECK_FALSE(skin_tone_of(0.0F, 0.0F, 0.0F).has_value());
  // Un feu : trop saturé pour la plage large.
  CHECK(skin_wide(*skin_tone_of(1.0F, 0.3F, 0.02F)) == 0.0F);
}

TEST_CASE("visages : la palette d'un visage est la médiane, un reflet ne la tire pas", "[media][faces]") {
  CHECK_FALSE(face_palette({}).has_value());
  const std::vector<SkinTone> samples{{.hue = 22.0F, .sat = 0.30F},
                                      {.hue = 23.0F, .sat = 0.32F},
                                      {.hue = 50.0F, .sat = 0.05F},  // le reflet
                                      {.hue = 22.5F, .sat = 0.31F},
                                      {.hue = 21.0F, .sat = 0.29F}};
  const auto p = face_palette(samples);
  REQUIRE(p.has_value());
  CHECK(p->hue == Approx(22.5));
  CHECK(p->sat == Approx(0.30));
}

TEST_CASE("visages : un plan garde les teintes vues deux fois, pas un faux visage d'une image",
          "[media][faces]") {
  // Deux personnages vus plusieurs fois, un faux visage une fois, une palette hors de la plage large.
  const std::vector<SkinTone> faces{{.hue = 22.0F, .sat = 0.30F}, {.hue = 23.0F, .sat = 0.31F},
                                    {.hue = 22.0F, .sat = 0.29F}, {.hue = 35.0F, .sat = 0.18F},
                                    {.hue = 36.0F, .sat = 0.19F}, {.hue = 12.0F, .sat = 0.55F},
                                    {.hue = 5.0F, .sat = 0.9F},   {.hue = 5.0F, .sat = 0.9F}};
  const PlanSkins skins = plan_skins(faces);
  REQUIRE(skins.count == 2);
  CHECK(skins.tones[0].hue == Approx(22.0));
  CHECK(skins.tones[0].sat == Approx(0.30));
  CHECK(skins.tones[1].hue == Approx(36.0));
  // Sans visage, ou un visage d'une seule image : aucune teinte, le grade garde sa plage large.
  CHECK(plan_skins({}).count == 0);
  CHECK(plan_skins(std::vector<SkinTone>{{.hue = 22.0F, .sat = 0.3F}}).count == 0);
}

}  // namespace ajp::media
