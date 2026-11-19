// La reprise de SPAN sur les dessins tenus (upscale_reuse.h) : la règle seule, puis l'étage en SPAN seul sur
// le banc sans GPU (une série longue contre l'ancre, une image abîmée, --no-span-reuse).
#include "app/play/upscale_reuse.h"

#include <cstdint>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "app/play/upscale_stage.h"
#include "stage_rig.h"
#include "upscale_rig.h"

namespace kirari::app::play {

using namespace upscale_test;

TEST_CASE("Reprise de SPAN : la règle contre l'ancre, et l'étage en SPAN seul", "[app][stage]") {
  const AnchorGap still{.worst_block = 10, .mean_levels = 0.01};   // sous 0,70 niveau
  const AnchorGap fading{.worst_block = 150, .mean_levels = 0.5};  // tenu par bloc, mais partout : fondu
  const AnchorGap moved{.worst_block = 1000, .mean_levels = 0.3};
  const ReuseFacts base{.same_epoch = true, .has_output = true, .pair_held = true};
  const auto with = [&](const AnchorGap& gap) {
    ReuseFacts f = base;
    f.anchor = gap;
    return f;
  };

  SpanReuse r;
  // Première reprise sur la paire seule (la précédente est l'ancre), puis contre l'ancre jusqu'à la limite.
  CHECK(r.decide(base) == ReuseKind::Whole);
  for (int k = 1; k < SpanReuse::kMaxInRow; ++k) CHECK(r.decide(with(still)) == ReuseKind::Whole);
  CHECK(r.decide(with(still)) == ReuseKind::Compute);  // la 25e : rafraîchie
  CHECK(r.in_row() == 0);
  // Un fondu passe deux fois (comme avant), puis s'arrête ; un mouvement s'arrête tout de suite.
  CHECK(r.decide(with(fading)) == ReuseKind::Whole);
  CHECK(r.decide(with(fading)) == ReuseKind::Whole);
  CHECK(r.decide(with(fading)) == ReuseKind::Compute);
  CHECK(r.decide(with(moved)) == ReuseKind::Compute);
  // Sans mesure contre l'ancre : deux de suite au plus.
  CHECK(r.decide(base) == ReuseKind::Whole);
  CHECK(r.decide(base) == ReuseKind::Whole);
  CHECK(r.decide(base) == ReuseKind::Compute);
  // Une image abîmée est calculée et ne sert pas d'ancre ; un saut ou une sortie perdue coupe la série.
  ReuseFacts corrupt = base;
  corrupt.corrupt = true;
  CHECK(r.decide(corrupt) == ReuseKind::Compute);
  CHECK(r.decide(with(still)) == ReuseKind::Compute);
  CHECK(r.decide(with(still)) == ReuseKind::Whole);
  ReuseFacts seek = with(still);
  seek.same_epoch = false;
  CHECK(r.decide(seek) == ReuseKind::Compute);
  ReuseFacts lost = with(still);
  lost.has_output = false;
  CHECK(r.decide(lost) == ReuseKind::Compute);
  CHECK(r.summary().starts_with("SPAN repris : 29 sur 37 images"));
  r.enabled = false;
  CHECK(r.decide(with(still)) == ReuseKind::Compute);

  // L'étage, SPAN seul : 2 à 6 tiennent le dessin de 1 (mesurées contre l'ancre 1, et non paire par paire), 7
  // bouge ; 8 est abîmée et calculée, 9 tient 8 mais ne la reprend pas.
  const auto run = [](bool reuse) {
    Rig rig;
    for (std::uint32_t k = 2; k <= 6; ++k) rig.backend.score(SourceId{1}, SourceId{k}, 0.0, 10);
    rig.backend.score(SourceId{8}, SourceId{9}, 0.0, 10);
    UpscaleSetup s = rig.setup(false);
    s.span_reuse.whole = reuse;
    rig.start(std::move(s));
    rig.feed(1, 5);
    REQUIRE(rig.collect(5));
    rig.feed(6, 7);
    DecodedFrame bad = test::host_frame(8, kWidth, kHeight, 8 * kPeriodUs, 0);
    bad.corrupt = true;
    REQUIRE(rig.push(std::move(bad)));
    REQUIRE(rig.collect(8));
    rig.feed(9, 9);
    REQUIRE(rig.collect(9));
    rig.stage.reset();
    CHECK(shape_of(rig.sink.frames()) ==
          std::vector<std::string>{"1", "2", "3", "4", "5", "6", "7", "8", "9"});
    return rig.backend.count("engine span");
  };
  CHECK(run(true) == 4);  // 1, 7, 8, 9
  CHECK(run(false) == 9);
}

}  // namespace kirari::app::play
