// Le débogage des paires (pair_marks.h) : les paires notées et retrouvées par l'image montrée, la ligne du
// journal, et la teinte que la session demande à l'écran seulement quand le menu Statistiques l'allume.
#include "app/play/pair_marks.h"

#include <cstdint>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"
#include "gpu/kernel_ops.h"

namespace ajp::app::play {
namespace {

TEST_CASE("Paires notées : retrouvées par l'époque et l'intervalle (a, b], la plus récente d'abord",
          "[app][pairs]") {
  PairMarks marks;
  CHECK_FALSE(marks.at(0, 40'000));
  marks.add({.epoch = 0, .from_us = 40'000, .to_us = 80'000, .verdict = PairVerdict::Held});
  marks.add({.epoch = 0, .from_us = 80'000, .to_us = 120'000, .verdict = PairVerdict::Cut});
  CHECK_FALSE(marks.at(0, 40'000));  // a n'est pas de la paire : elle finit la précédente
  CHECK(marks.at(0, 60'000) == PairVerdict::Held);
  CHECK(marks.at(0, 80'000) == PairVerdict::Held);
  CHECK(marks.at(0, 100'000) == PairVerdict::Cut);
  CHECK_FALSE(marks.at(1, 80'000));
  marks.add({.epoch = 0, .from_us = 40'000, .to_us = 80'000, .verdict = PairVerdict::Cut});  // refaite
  CHECK(marks.at(0, 80'000) == PairVerdict::Cut);
  marks.clear();
  CHECK_FALSE(marks.at(0, 100'000));
}

TEST_CASE("Paires notées : les plus anciennes cèdent la place", "[app][pairs]") {
  PairMarks marks;
  for (std::int64_t k = 0; k < static_cast<std::int64_t>(PairMarks::kKept) + 4; ++k)
    marks.add({.epoch = 0, .from_us = k * 1000, .to_us = (k + 1) * 1000, .verdict = PairVerdict::Held});
  CHECK_FALSE(marks.at(0, 1000));
  CHECK(marks.at(0, static_cast<std::int64_t>(PairMarks::kKept + 4) * 1000) == PairVerdict::Held);
}

TEST_CASE("Paires notées : la ligne du journal dit le verdict, le minutage et les mesures", "[app][pairs]") {
  const PairMark held{.epoch = 0, .from_us = 12'345'000, .to_us = 12'387'000, .verdict = PairVerdict::Held};
  const gpu::PairScores scores{.scene_score = 0.031, .worst_block = 105};  // 105 / 256 = 0,41 niveau
  CHECK(
      pair_mark_line(held, scores, std::nullopt, 1.0) ==
      "paire tenue 12.345 s → 12.387 s, pire bloc 0.41 (tenue ≤ 0.70), score 0.031 (coupe > 0.120), plans : "
      "hors plans");
  const PairMark cut{.epoch = 0, .from_us = 1'000'000, .to_us = 1'042'000, .verdict = PairVerdict::Cut};
  const gpu::PairScores sharp{.scene_score = 0.1, .worst_block = 5000};
  // Le score rapporté à la zone hors bandes noires, comme la coupe le compare.
  const std::string line = pair_mark_line(cut, sharp, true, 0.5);
  CHECK(line.starts_with("paire coupée 1.000 s → 1.042 s"));
  CHECK(line.contains("score 0.200"));
  CHECK(line.ends_with("plans : coupe"));
  CHECK(pair_mark_line(cut, sharp, false, 1.0).ends_with("plans : même plan"));
}

TEST_CASE("Paires notées : bleu pour un dessin tenu, rouge pour une coupe, le cadre plus franc que le voile",
          "[app][pairs]") {
  const PairTint held = pair_tint(PairVerdict::Held);
  const PairTint cut = pair_tint(PairVerdict::Cut);
  CHECK(held.wash[2] > held.wash[0]);
  CHECK(cut.wash[0] > cut.wash[2]);
  CHECK(held.frame[3] > held.wash[3]);
  CHECK(cut.frame[3] > cut.wash[3]);
}

struct Verdicts final : StatsSource {
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() override { return std::nullopt; }
  std::optional<PairVerdict> pair_verdict(std::uint32_t epoch, std::int64_t pts_us) const override {
    if (epoch == 0 && pts_us == 80'000) return PairVerdict::Held;
    return std::nullopt;
  }
};

TEST_CASE("Session : la teinte de la paire va à l'écran seulement quand le débogage est allumé",
          "[app][pairs][session]") {
  Verdicts extra;
  test::FakeStages stages;
  test::FakeScreen screen;
  test::FakeClock clock;
  Session session(SessionConfig{.extra = &extra}, stages, screen, clock);
  session.handle(test::ring_frame(0, 80'000));
  REQUIRE(session.tick(screen.now_us));
  REQUIRE_FALSE(screen.shown.empty());
  CHECK_FALSE(screen.shown.back().tint);  // éteint : rien
  session.set_stats(StatsSettings{.pairs = true});
  const std::size_t before = screen.shown.size();
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.size() == before + 1);  // allumé : l'image tenue se représente avec sa teinte
  CHECK(screen.shown.back().tint == pair_tint(PairVerdict::Held));
  session.set_stats(StatsSettings{});
  REQUIRE(session.tick(screen.now_us));
  CHECK_FALSE(screen.shown.back().tint);
}

}  // namespace
}  // namespace ajp::app::play
