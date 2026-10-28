// File des sous-titres rendus : chaque image présentée garde les siens (clignotement du
// 19 octobre, quand les réponses en avance chassaient celle de l'image à l'écran).
#include "app/play/subtitle_queue.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "core/frame.h"

namespace ajp::app::play {
namespace {

subs::SubtitleFrame response(std::int64_t pts_us) {
  return subs::SubtitleFrame{.pts_us = pts_us};
}

}  // namespace

TEST_CASE("Sous-titres : l'image à l'écran garde sa réponse quand le ring se remplit en avance",
          "[app][subtitles]") {
  // Ring de 4 : l'image 0 est à l'écran, les images 41, 83 et 125 ms attendent derrière ; leurs
  // réponses arrivent pendant que 0 est montrée, une vsync après l'autre.
  SubtitleQueue queue(kMaxKeptSubtitles);
  const std::int64_t shown = 0;
  queue.push(response(0), kNoPts);
  for (const std::int64_t ahead : {41'000, 83'000, 125'000}) {
    queue.push(response(ahead), shown);
    const subs::SubtitleFrame* frame = queue.for_pts(shown);
    REQUIRE(frame != nullptr);
    CHECK(frame->pts_us == shown);
  }
  // Une réponse de trop : la plus en avance part, jamais celle de l'image montrée.
  queue.push(response(166'000), shown);
  CHECK(queue.size() == kMaxKeptSubtitles);
  REQUIRE(queue.for_pts(shown) != nullptr);
  CHECK(queue.for_pts(shown)->pts_us == shown);
  // L'image 166 ms reprend la réponse précédente : une image tôt, jamais aucune.
  REQUIRE(queue.for_pts(166'000) != nullptr);
  CHECK(queue.for_pts(166'000)->pts_us == 125'000);
}

TEST_CASE("Sous-titres : une image sans réponse à elle prend la dernière d'avant", "[app][subtitles]") {
  SubtitleQueue queue(kMaxKeptSubtitles);
  queue.push(response(0), kNoPts);
  queue.push(response(83'000), 0);
  // 41 ms n'a pas eu de demande (trop de demandes en vol) : elle garde la réponse de 0.
  REQUIRE(queue.for_pts(41'000) != nullptr);
  CHECK(queue.for_pts(41'000)->pts_us == 0);
  REQUIRE(queue.for_pts(83'000) != nullptr);
  CHECK(queue.for_pts(83'000)->pts_us == 83'000);
  CHECK(queue.size() == 1);  // 0 ne servira plus
}

TEST_CASE("Sous-titres : réponse en retard rangée à sa place, rien avant la première", "[app][subtitles]") {
  SubtitleQueue queue(kMaxKeptSubtitles);
  CHECK(queue.for_pts(0) == nullptr);
  queue.push(response(83'000), kNoPts);
  queue.push(response(41'000), kNoPts);
  CHECK(queue.for_pts(0) == nullptr);  // aucune réponse d'avant 0
  CHECK(queue.for_pts(kNoPts) == nullptr);
  REQUIRE(queue.for_pts(41'000) != nullptr);
  CHECK(queue.for_pts(41'000)->pts_us == 41'000);
  queue.clear();
  CHECK(queue.size() == 0);
}

TEST_CASE("Sous-titres : les réponses gardées couvrent le ring et l'image à l'écran", "[app][subtitles]") {
  // Le ring du lecteur (player.cpp) a 4 slots ; render ne tient jamais plus de
  // kMaxAtlasesInFlight − 1 atlas avec les demandes en vol.
  STATIC_CHECK(kMaxKeptSubtitles >= 4);
  STATIC_CHECK(kMaxKeptSubtitles + kMaxSubtitleRequests < subs::kMaxAtlasesInFlight);
}

}  // namespace ajp::app::play
