// La clé de la composition gardée : elle change avec ce que l'image composée montre, pas avec l'OSD.
#include "app/play/compose_key.h"

#include <memory>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

subs::SubtitleFrame line(std::uint64_t generation, int dst_y) {
  auto atlas = std::make_shared<subs::SubtitleAtlas>();
  atlas->generation = generation;
  subs::SubtitleFrame frame;
  frame.atlas = atlas;
  frame.rects.push_back(subs::SubtitleRect{.src = {.x = 0, .y = 0, .width = 40, .height = 10},
                                           .dst_x = 100,
                                           .dst_y = dst_y,
                                           .color = {255, 255, 255, 255}});
  return frame;
}

TEST_CASE("Clé de composition : change avec les sous-titres, la taille, le grade, pas avec l'OSD seul",
          "[app][compose]") {
  RingFrame frame;
  frame.slot = 2;
  frame.generation = 1;
  frame.ready_value = 10;
  ComposeState state;
  state.grade_serial = 3;
  const ComposeKey base = compose_key(&frame, nullptr, 1920, 1080, 0, state);
  // L'OSD n'est pas dans la demande de la clé : même image, mêmes sous-titres, même clé.
  CHECK(reusable(base, compose_key(&frame, nullptr, 1920, 1080, 0, state)));
  // Sans sous-titres, la barre de lecture ne déplace rien.
  CHECK(reusable(base, compose_key(&frame, nullptr, 1920, 1080, 120, state)));
  CHECK_FALSE(reusable(std::nullopt, base));

  // Image neuve : même slot réécrit, autre slot, autre anneau, image noire.
  RingFrame next = frame;
  next.ready_value = 12;
  CHECK_FALSE(reusable(base, compose_key(&next, nullptr, 1920, 1080, 0, state)));
  next = frame;
  next.slot = 3;
  CHECK_FALSE(reusable(base, compose_key(&next, nullptr, 1920, 1080, 0, state)));
  next = frame;
  next.generation = 2;
  CHECK_FALSE(reusable(base, compose_key(&next, nullptr, 1920, 1080, 0, state)));
  CHECK_FALSE(reusable(base, compose_key(nullptr, nullptr, 1920, 1080, 0, state)));

  // Sous-titres : apparition, nouveau contenu, déplacement, montée au-dessus de la barre.
  const subs::SubtitleFrame a = line(5, 900);
  const ComposeKey with_a = compose_key(&frame, &a, 1920, 1080, 0, state);
  CHECK_FALSE(reusable(base, with_a));
  CHECK(reusable(with_a, compose_key(&frame, &a, 1920, 1080, 0, state)));
  const subs::SubtitleFrame b = line(6, 900);
  CHECK_FALSE(reusable(with_a, compose_key(&frame, &b, 1920, 1080, 0, state)));
  const subs::SubtitleFrame moved = line(5, 880);
  CHECK_FALSE(reusable(with_a, compose_key(&frame, &moved, 1920, 1080, 0, state)));
  CHECK_FALSE(reusable(with_a, compose_key(&frame, &a, 1920, 1080, 120, state)));
  // Un atlas sans rectangle ne dessine rien.
  subs::SubtitleFrame empty = a;
  empty.rects.clear();
  CHECK(reusable(base, compose_key(&frame, &empty, 1920, 1080, 0, state)));

  // Taille de la cible.
  CHECK_FALSE(reusable(base, compose_key(&frame, nullptr, 2560, 1440, 0, state)));

  // Ce que l'écran tient : grade, finition, forme, sortie.
  const auto changed = [&](auto edit) {
    ComposeState other = state;
    edit(other);
    return !reusable(base, compose_key(&frame, nullptr, 1920, 1080, 0, other));
  };
  CHECK(changed([](ComposeState& s) { ++s.grade_serial; }));
  CHECK(changed([](ComposeState& s) { s.final_scaler = FinalScaler::EwaLanczos; }));
  CHECK(changed([](ComposeState& s) { s.deband = true; }));
  CHECK(changed([](ComposeState& s) { s.sar = 4.0 / 3.0; }));
  CHECK(changed([](ComposeState& s) { s.crop = {0.0F, 60.0F, 1920.0F, 1020.0F}; }));
  CHECK(changed([](ComposeState& s) { s.transfer = 16; }));
  CHECK(changed([](ComposeState& s) { s.primaries = 7; }));
  CHECK(changed([](ComposeState& s) { s.max_luma = 1000.0F; }));
}

}  // namespace
}  // namespace ajp::app::play
