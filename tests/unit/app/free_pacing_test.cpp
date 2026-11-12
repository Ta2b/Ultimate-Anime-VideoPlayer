// Les images manquées du mode Rapide se comptent contre l'écart que la grille a planifié pour chaque image
// (RingFrame::planned_gap_us), pas contre l'écart d'une paire calme : à 165 Hz, une scène lourde planifie
// deux vsyncs par image sans rien perdre.
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "app/play/frame_stats.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kVsync = 6'061;  // 165 Hz
constexpr std::int64_t kCalm = kVsync;  // l'écart d'une paire calme, la période de sortie du Rapide

TEST_CASE("Rapide 165 Hz : les écarts planifiés d'une scène lourde ne manquent pas", "[app][free]") {
  FrameStats planned;
  FrameStats by_period;
  std::int64_t pts = 0;
  planned.paced(pts, 0, kCalm);
  by_period.paced(pts, 0, kCalm);
  for (int k = 0; k < 400; ++k) {
    // Deux vsyncs, de temps en temps trois (la paire affamée, la source qui tombe plus loin).
    const std::int64_t gap = k % 7 == 0 ? 3 * kVsync : 2 * kVsync;
    pts += gap;
    planned.paced(pts, 0, kCalm, false, gap);
    by_period.paced(pts, 0, kCalm);
  }
  CHECK(planned.missing == 0);
  CHECK(by_period.missing >= 400);  // l'ancien compte : au moins une « manquée » par image
}

TEST_CASE("Rapide 165 Hz : un instant planifié qui n'a pas paru manque toujours", "[app][free]") {
  FrameStats stats;
  stats.paced(0, 0, kCalm, false, 0);
  stats.paced(2 * kVsync, 0, kCalm, false, 2 * kVsync);
  // L'image à 4 vsyncs, jetée en retard : celle à 6 vsyncs la suivait de deux.
  stats.paced(6 * kVsync, 0, kCalm, false, 2 * kVsync);
  CHECK(stats.missing == 1);
  // Trois de suite perdues (8, 10, 12) avant celle à 14 vsyncs.
  stats.paced(14 * kVsync, 0, kCalm, false, 2 * kVsync);
  CHECK(stats.missing == 4);
  // Une image planifiée plus tôt que la dernière montrée (pts arrondis) : rien.
  stats.paced(16 * kVsync + 300, 0, kCalm, false, 2 * kVsync);
  CHECK(stats.missing == 4);
  // Laissé à dessein (coupe, dessin tenu) : compté à part.
  stats.paced(20 * kVsync, 0, kCalm, true, 2 * kVsync);
  CHECK(stats.missing == 4);
  CHECK(stats.uninterpolated == 1);
  // Après un seek, l'écart n'est pas un manque, planifié ou non.
  stats.restart_pacing();
  stats.paced(1'000'000, 0, kCalm, false, 2 * kVsync);
  CHECK(stats.missing == 4);
}

TEST_CASE("Rapide : sans écart planifié, la période de sortie compte comme avant", "[app][free]") {
  FrameStats stats;
  stats.paced(0, 0, 40'000, false, 0);
  stats.paced(120'000, 0, 40'000, false, 0);
  CHECK(stats.missing == 2);
}

}  // namespace
}  // namespace ajp::app::play
