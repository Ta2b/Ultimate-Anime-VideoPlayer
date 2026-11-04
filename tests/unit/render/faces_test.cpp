// Méga plan A4 : les visages sans réseau (render/faces.h) : décodage, doublons, suivi, palette, exclusion.
#include "render/faces.h"

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::render {

using Catch::Approx;

TEST_CASE("visages : la sortie du détecteur se décode en cadres normalisés", "[render][faces]") {
  // Deux lignes de 6 (cx, cy, w, h, score, classe) sur une entrée 640 × 320 ; la seconde sous le seuil.
  const std::vector<float> out{320.0F, 160.0F, 64.0F, 32.0F, 0.9F, 0.0F,
                               10.0F,  10.0F,  8.0F,  8.0F,  0.1F, 0.0F};
  const auto boxes = decode_faces(out, 2, 6, 640, 320);
  REQUIRE(boxes.size() == 1);
  CHECK(boxes[0].x0 == Approx(0.45));
  CHECK(boxes[0].x1 == Approx(0.55));
  CHECK(boxes[0].y0 == Approx(0.45));
  CHECK(boxes[0].y1 == Approx(0.55));
  CHECK(boxes[0].score == Approx(0.9));
  // Rogné à l'image ; une sortie trop courte ou mal formée ne rend rien.
  const std::vector<float> edge{0.0F, 0.0F, 64.0F, 64.0F, 0.8F};
  const auto clipped = decode_faces(edge, 1, 5, 640, 320);
  REQUIRE(clipped.size() == 1);
  CHECK(clipped[0].x0 == 0.0F);
  CHECK(decode_faces(edge, 2, 5, 640, 320).empty());
  CHECK(decode_faces(edge, 1, 4, 640, 320).empty());
}

TEST_CASE("visages : les doublons se suppriment, le meilleur score reste", "[render][faces]") {
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

TEST_CASE("visages : la palette de peau est la médiane, un reflet ne la tire pas", "[render][faces]") {
  CHECK_FALSE(face_palette({}).has_value());
  const std::vector<SkinPalette> samples{{.i = 0.50F, .ct = 0.02F, .cp = 0.05F},
                                         {.i = 0.52F, .ct = 0.03F, .cp = 0.06F},
                                         {.i = 0.95F, .ct = 0.00F, .cp = 0.00F},  // le reflet
                                         {.i = 0.51F, .ct = 0.025F, .cp = 0.055F},
                                         {.i = 0.49F, .ct = 0.02F, .cp = 0.05F}};
  const auto p = face_palette(samples);
  REQUIRE(p.has_value());
  CHECK(p->i == Approx(0.51));
  CHECK(p->ct == Approx(0.02));
  CHECK(p->cp == Approx(0.05));
}

TEST_CASE("visages : le suivi garde l'identité, lisse le cadre, oublie après trois absences",
          "[render][faces]") {
  FaceTracker tracker;
  const FaceBox face{.x0 = 0.4F, .y0 = 0.2F, .x1 = 0.6F, .y1 = 0.5F, .score = 0.9F};
  tracker.update(std::vector{face});
  REQUIRE(tracker.tracks().size() == 1);
  const std::uint32_t id = tracker.tracks()[0].id;
  CHECK_FALSE(tracker.tracks()[0].confirmed());
  // Le visage glisse un peu : la même piste, son cadre à mi-chemin.
  FaceBox moved = face;
  moved.x0 += 0.02F;
  moved.x1 += 0.02F;
  tracker.update(std::vector{moved});
  REQUIRE(tracker.tracks().size() == 1);
  CHECK(tracker.tracks()[0].id == id);
  CHECK(tracker.tracks()[0].confirmed());
  CHECK(tracker.tracks()[0].box.x0 == Approx(0.41));
  tracker.set_palette(id, {.i = 0.5F});
  CHECK(tracker.tracks()[0].palette.has_value());
  // Deux images sans lui : toujours là et confirmé ; la troisième l'efface.
  tracker.update({});
  tracker.update({});
  REQUIRE(tracker.tracks().size() == 1);
  CHECK(tracker.tracks()[0].confirmed());
  tracker.update({});
  CHECK(tracker.tracks().empty());
  // Un autre visage ailleurs : une piste neuve ; une coupe efface tout.
  tracker.update(std::vector{face});
  CHECK(tracker.tracks()[0].id != id);
  tracker.cut();
  CHECK(tracker.tracks().empty());
}

TEST_CASE("visages : l'exclusion des émetteurs couvre le visage confirmé, rampe sur 10 %",
          "[render][faces]") {
  FaceTracker tracker;
  const FaceBox face{.x0 = 0.4F, .y0 = 0.4F, .x1 = 0.6F, .y1 = 0.6F, .score = 0.9F};
  tracker.update(std::vector{face});
  CHECK(face_exclusion(tracker.tracks(), 0.5F, 0.5F) == 0.0F);  // pas encore confirmé
  tracker.update(std::vector{face});
  CHECK(face_exclusion(tracker.tracks(), 0.5F, 0.5F) == 1.0F);
  CHECK(face_exclusion(tracker.tracks(), 0.61F, 0.5F) == Approx(0.5));  // marge de 0,02
  CHECK(face_exclusion(tracker.tracks(), 0.7F, 0.5F) == 0.0F);
}

}  // namespace ajp::render
