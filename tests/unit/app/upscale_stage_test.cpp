// L'étage gpu (upscale_stage.h) sur le banc sans GPU (stage_rig.h) : une sortie par image sans
// interpolation, les images intermédiaires de RIFE et leurs pts, coupes de scène et dessins tenus
// sans RIFE, le facteur changé en direct, le seek, l'horloge en retard, DistilDRBA qui attend la
// suivante, le protocole des slots, les refus de démarrage, et l'arrêt sur une image refusée.
#include "app/play/upscale_stage.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/live_settings.h"
#include "app/play/options.h"
#include "core/clock.h"
#include "doubles.h"
#include "stage_rig.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

constexpr int kWidth = 320;
constexpr int kHeight = 180;
constexpr std::int64_t kPeriodUs = 40'000;

struct Rig : test::StageRig {
  test::FakeEngine span{"span", {"input"}, {"output"}};
  test::FakeEngine rife{"rife", {"input"}, {"output"}};
  test::FakeEngine drba{"drba", {"pair01", "pair12"}, {"output"}};
  test::FakeEngine motion{"rife_motion", {"input"}, {"flow", "mask"}};
  std::shared_ptr<LiveSettings> live = std::make_shared<LiveSettings>();

  // SPAN toujours ; RIFE ×num/den par les réglages en direct quand with_rife.
  UpscaleSetup setup(bool with_rife, Factor factor = {.num = 2, .den = 1}, std::uint32_t slot_count = 4) {
    live->set_rife_factor(with_rife ? factor : Factor{.num = 1, .den = 1});
    return UpscaleSetup{.backend = &backend,
                        .engine = &span,
                        .rife = with_rife ? &rife : nullptr,
                        .live = live,
                        .source_width = kWidth,
                        .source_height = kHeight,
                        .slots = slots(slot_count, kWidth * kSpanFactor, kHeight * kSpanFactor)};
  }
  void start(UpscaleSetup s, const MasterClock* clock = nullptr) {
    auto started = UpscaleStage::start(init(clock), std::move(s));
    if (!started) FAIL("refus " << name(started.error().code) << " : " << started.error().detail);
    stage = std::move(*started);
  }
  // Pousse les images id = first … last, à kPeriodUs d'écart, dans l'époque donnée.
  void feed(std::uint32_t first, std::uint32_t last, std::uint32_t epoch = 0) {
    for (std::uint32_t id = first; id <= last; ++id)
      REQUIRE(push(test::host_frame(id, kWidth, kHeight, static_cast<std::int64_t>(id) * kPeriodUs, epoch)));
  }
};

// « 3 » : la source 3 telle quelle ; « 2-3 » : une image entre les sources 2 et 3.
std::vector<std::string> shape_of(const std::vector<RingFrame>& frames) {
  std::vector<std::string> out;
  for (const RingFrame& f : frames) {
    out.push_back(f.left == f.right ? std::to_string(f.left.value)
                                    : std::to_string(f.left.value) + "-" + std::to_string(f.right.value));
  }
  return out;
}

// La voie attendue d'un travail du journal (« ingest 3 » : measure) ; vide pour les autres (alloc).
std::string lane_of(std::string_view entry) {
  if (entry.starts_with("ingest") || entry.starts_with("scores_") || entry == "rife_input" ||
      entry == "rife_pair")
    return "measure";
  if (entry == "engine span" || entry.starts_with("write")) return "upscale";
  if (entry.starts_with("engine rife") || entry.starts_with("rife_") || entry.starts_with("warp_blend") ||
      entry.starts_with("flow_max"))
    return "interpolate";
  return "";
}

// Chaque travail du journal sur sa voie ; combien sur chacune.
std::map<std::string, int> check_lanes(const std::vector<std::string>& laned) {
  std::map<std::string, int> per_lane;
  for (const std::string& line : laned) {
    const std::size_t space = line.find(' ');
    const std::string want = lane_of(std::string_view(line).substr(space + 1));
    if (want.empty()) continue;
    INFO(line);
    CHECK(line.substr(0, space) == want);
    ++per_lane[want];
  }
  return per_lane;
}

std::vector<std::int64_t> pts_of(const std::vector<RingFrame>& frames) {
  std::vector<std::int64_t> out;
  for (const RingFrame& f : frames) out.push_back(f.pts_us);
  return out;
}

}  // namespace

TEST_CASE("Étage gpu : sans interpolation, une sortie par image, slots et sémaphores dans l'ordre",
          "[app][stage]") {
  Rig rig;
  rig.start(rig.setup(false));
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "2", "3", "4", "5", "6"});
  CHECK(pts_of(frames) == std::vector<std::int64_t>{40'000, 80'000, 120'000, 160'000, 200'000, 240'000});
  for (std::size_t i = 0; i < frames.size(); ++i) {
    CHECK(frames[i].slot == i % 4);  // les slots rendus dans l'ordre sont repris dans l'ordre
    CHECK(frames[i].generation == 1);
    CHECK(frames[i].ready_value % 2 == 1);
  }
  // Chaque écriture attend la valeur rendue par main (ready + 1) et signale la suivante, impaire.
  const std::vector<test::RingWritten> written = rig.backend.written();
  REQUIRE(written.size() == 6);
  CHECK(written[0].wait_value == 0);
  CHECK(written[0].signal_value == 1);
  CHECK(written[4].slot == 0);
  CHECK(written[4].wait_value == 2);
  CHECK(written[4].signal_value == 3);
  CHECK(rig.backend.count("engine span") == 6);
  CHECK(rig.backend.count("engine rife") == 0);
  CHECK(rig.credits.load() == 6);
  CHECK(rig.reported().empty());
}

TEST_CASE("Étage gpu : RIFE ×2, une image intermédiaire par paire ; coupe et dessin tenu sans RIFE",
          "[app][stage]") {
  Rig rig;
  rig.backend.score(SourceId{3}, SourceId{4}, 0.5, 1000);  // coupe de scène
  rig.backend.score(SourceId{5}, SourceId{6}, 0.0, 10);    // dessin tenu
  rig.start(rig.setup(true));
  rig.feed(1, 6);
  REQUIRE(rig.collect(9));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3", "4", "4-5", "5", "6"});
  CHECK(pts_of(frames) == std::vector<std::int64_t>{40'000, 60'000, 80'000, 100'000, 120'000, 160'000,
                                                    180'000, 200'000, 240'000});
  CHECK(rig.backend.count("engine rife") == 3);
  CHECK(rig.backend.count("rife_input") == 3);
  CHECK(rig.backend.count("rife_crop") == 3);
  // SPAN sur chaque source sauf le dessin tenu, qui reprend l'agrandissement de la précédente.
  CHECK(rig.backend.count("engine span") == 5 + 3);  // + les trois images intermédiaires
  const RifeCounts counts = rig.live->counts();
  CHECK(counts.pairs == 5);
  CHECK(counts.interpolated == 3);
  CHECK(counts.cuts == 1);
  CHECK(counts.held == 1);
  CHECK(counts.late == 0);
  CHECK(rig.reported().empty());
}

TEST_CASE("Étage gpu : RIFE ×3, deux instants par paire, l'entrée remplie une fois puis l'instant seul",
          "[app][stage]") {
  Rig rig;
  rig.start(rig.setup(true, Factor{.num = 3, .den = 1}));
  rig.feed(1, 3);
  REQUIRE(rig.collect(7));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-2", "1-2", "2", "2-3", "2-3", "3"});
  const std::vector<std::int64_t> pts = pts_of(frames);
  CHECK(pts[1] == 53'333);  // 40 ms + 1/3 et 2/3 de période, arrondis
  CHECK(pts[2] == 66'667);
  CHECK(pts[3] == 80'000);
  CHECK(rig.backend.count("engine rife") == 4);
  CHECK(rig.backend.count("rife_input") == 2);  // les images de la paire une fois
  CHECK(rig.backend.count("rife_time") == 2);   // puis l'instant seul
}

TEST_CASE("Étage gpu : RIFE mouvement, une passe au milieu tant que le mouvement est faible",
          "[app][stage]") {
  Rig rig;
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(10));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) ==
        std::vector<std::string>{"1", "1-2", "1-2", "2", "2-3", "2-3", "3", "3-4", "3-4", "4"});
  CHECK(rig.backend.count("engine rife_motion") == 3);  // une par paire, au milieu
  CHECK(rig.backend.count("flow_max_start") == 3);
  CHECK(rig.backend.count("flow_max_take") == 3);
  CHECK(rig.backend.count("warp_blend") == 6);
}

TEST_CASE("Étage gpu : mesures, SPAN et RIFE chacun sur sa voie, SPAN avant le warp aussi", "[app][stage]") {
  SECTION("RIFE image ×3") {
    Rig rig;
    rig.start(rig.setup(true, Factor{.num = 3, .den = 1}));
    rig.feed(1, 3);
    REQUIRE(rig.collect(7));
    rig.stage.reset();
    const std::map<std::string, int> per_lane = check_lanes(rig.backend.laned());
    CHECK(per_lane.size() == 3);
    CHECK(rig.backend.count("engine rife") == 4);
  }
  SECTION("RIFE mouvement, warp après SPAN : SPAN enfilé au milieu du travail de la paire") {
    Rig rig;
    UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
    s.rife = nullptr;
    s.rife_motion = &rig.motion;
    s.warp_after_span = true;
    rig.start(std::move(s));
    rig.feed(1, 4);
    REQUIRE(rig.collect(10));
    rig.stage.reset();
    const std::map<std::string, int> per_lane = check_lanes(rig.backend.laned());
    CHECK(per_lane.size() == 3);
    CHECK(rig.backend.count("engine span") == 4);  // une passe par image source
    CHECK(rig.backend.count("warp_blend") == 6);   // sur la voie de RIFE, reprise après SPAN
  }
}

TEST_CASE("Étage gpu : le warp écrit l'image intermédiaire dans le slot quand le backend le permet",
          "[app][stage]") {
  const bool spanned = GENERATE(true, false);
  INFO((spanned ? "warp après SPAN" : "sans SPAN"));
  struct Run {
    std::vector<RingFrame> frames;
    std::vector<test::RingWritten> written;
    std::size_t blends = 0;
    std::size_t ring_blends = 0;
    std::size_t ring_writes = 0;  // par write_ring
    std::size_t allocs = 0;
  };
  const auto run = [&](bool direct) {
    Rig rig;
    rig.backend.offer_ring_targets(direct);
    UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
    s.rife = nullptr;
    s.rife_motion = &rig.motion;
    s.warp_after_span = spanned;
    if (!spanned) {
      s.engine = nullptr;
      s.slots = test::StageRig::slots(4, kWidth, kHeight);
    }
    rig.start(std::move(s));
    rig.feed(1, 4);
    REQUIRE(rig.collect(10));
    rig.stage.reset();
    Run out{.frames = rig.sink.frames(),
            .written = rig.backend.written(),
            .blends = rig.backend.count("warp_blend"),
            .ring_blends = rig.backend.count("warp_blend_ring"),
            .allocs = rig.backend.count("alloc")};
    for (const std::string& entry : rig.backend.log())
      out.ring_writes += entry.starts_with("write ") ? 1U : 0U;
    return out;
  };
  const Run plain = run(false);
  const Run direct = run(true);
  CHECK(shape_of(direct.frames) == shape_of(plain.frames));
  CHECK(plain.blends == 6);
  CHECK(plain.ring_writes == 10);
  CHECK(direct.blends == 0);
  CHECK(direct.ring_blends == 6);
  CHECK(direct.ring_writes == 4);            // les images sources seules
  CHECK(direct.allocs == plain.allocs - 1);  // une voie : un tampon d'image intermédiaire de moins
  // Chaque slot écrit une fois, par le warp ou par write_ring, au protocole des sémaphores : la
  // sortie attend la libération de l'image précédente du même slot (release_of), 0 au premier tour.
  // Quel slot revient quand dépend du fil main : la valeur se déduit dans chaque passe.
  REQUIRE(direct.written.size() == direct.frames.size());
  for (std::size_t i = 0; i < direct.frames.size(); ++i) {
    INFO("sortie " << i);
    CHECK(direct.written[i].slot == direct.frames[i].slot);
    CHECK(direct.written[i].signal_value == direct.frames[i].ready_value);
    std::uint64_t freed = 0;
    for (std::size_t j = 0; j < i; ++j) {
      if (direct.frames[j].slot == direct.frames[i].slot)
        freed = release_of(direct.frames[j]).semaphore_value;
    }
    CHECK(direct.written[i].wait_value == freed);
  }
}

TEST_CASE(
    "Étage gpu : avec plusieurs voies, entrée de RIFE et image intermédiaire dans deux tampons à tour de "
    "rôle",
    "[app][stage]") {
  const int lanes = GENERATE(1, 3);
  Rig rig;
  rig.backend.offer_lanes(lanes);
  rig.start(rig.setup(true, Factor{.num = 3, .den = 1}));
  rig.feed(1, 4);
  REQUIRE(rig.collect(10));
  rig.stage.reset();
  const std::vector<std::uintptr_t> inputs = rig.backend.targets("rife_input");
  const std::vector<std::uintptr_t> crops = rig.backend.targets("rife_crop");
  REQUIRE(inputs.size() == 3);  // une par paire, au premier instant
  REQUIRE(crops.size() == 6);   // une par instant
  INFO(lanes << " voie(s)");
  for (std::size_t i = 1; i < inputs.size(); ++i) CHECK((inputs[i] != inputs[i - 1]) == (lanes > 1));
  for (std::size_t i = 1; i < crops.size(); ++i) CHECK((crops[i] != crops[i - 1]) == (lanes > 1));
  CHECK(std::set(inputs.begin(), inputs.end()).size() == (lanes > 1 ? 2U : 1U));
  CHECK(rig.backend.count("rife_time") == 3);  // le second instant de chaque paire
}

TEST_CASE("Étage gpu : le maillage de RIFE écrit une fois par tampon, les paires suivantes sans lui",
          "[app][stage]") {
  const int lanes = GENERATE(1, 3);
  Rig rig;
  rig.backend.offer_lanes(lanes);
  rig.backend.offer_rife_pair(true);
  rig.start(rig.setup(true, Factor{.num = 3, .den = 1}));
  rig.feed(1, 5);
  REQUIRE(rig.collect(13));
  rig.stage.reset();
  INFO(lanes << " voie(s)");
  const std::vector<std::uintptr_t> full = rig.backend.targets("rife_input");
  const std::vector<std::uintptr_t> pairs = rig.backend.targets("rife_pair");
  const std::size_t copies = lanes > 1 ? 2 : 1;
  CHECK(full.size() == copies);  // une fois par tampon d'entrée
  CHECK(pairs.size() == 4 - copies);
  CHECK(std::set(full.begin(), full.end()).size() == copies);
  for (const std::uintptr_t target : pairs) CHECK(std::ranges::find(full, target) != full.end());
  CHECK(rig.backend.count("rife_time") == 4);  // le second instant de chaque paire
}

TEST_CASE("Étage gpu : RIFE mouvement fort, sans passe au milieu après la première paire, oublié à la coupe",
          "[app][stage]") {
  Rig rig;
  rig.backend.motion_max(40.0F);  // au-delà du seuil d'une passe (3,75 px à 180 lignes)
  rig.backend.score(SourceId{4}, SourceId{5}, 0.5, 1000);  // coupe de scène
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  rig.start(std::move(s));
  rig.feed(1, 4);  // 1, puis trois paires interpolées
  REQUIRE(rig.collect(1 + 3 * 3));
  rig.feed(5, 7);  // 5 seule (coupe), puis deux paires interpolées
  REQUIRE(rig.collect(1 + 3 * 3 + 1 + 2 * 3));
  rig.stage.reset();
  // (1,2) : milieu puis 1/3 et 2/3, trois passes ; (2,3) et (3,4) : deux passes, sans milieu.
  // Après la coupe, (5,6) repart du milieu (trois passes), puis (6,7) en deux.
  CHECK(rig.backend.count("engine rife_motion") == 3 + 2 + 2 + 3 + 2);
  // Une mesure par paire : relue tout de suite après le milieu, ou à la paire suivante sans lui.
  CHECK(rig.backend.count("flow_max_start") == 5);
  CHECK(rig.backend.count("flow_max_take") == 4);
  CHECK(rig.backend.pending_flow_tickets() == 1);  // celle de (6,7), sans paire après elle
  CHECK(rig.backend.count("warp_blend") == 2 * 5);
}

TEST_CASE("Étage gpu : mesures en avance quand la suivante attend déjà, scores_now sans elles",
          "[app][stage]") {
  SECTION("backend avec scores_start : la paire est mesurée derrière l'image précédente") {
    Rig rig;
    rig.start(rig.setup(true, Factor{.num = 2, .den = 1}, 2));
    // Deux slots pour que les images s'accumulent derrière la tête : la suivante attend.
    rig.feed(1, 4);
    REQUIRE(rig.collect(7));
    rig.stage.reset();
    CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3", "3-4", "4"});
    CHECK(rig.backend.count("scores_take") > 0);
    CHECK(rig.backend.count("ingest 2") == 1);  // ingérée une fois, en avance ou à son tour
  }
  SECTION("backend sans scores_start : scores_now à chaque paire") {
    Rig rig;
    rig.backend.offer_ahead(false);
    rig.start(rig.setup(true));
    rig.feed(1, 4);
    REQUIRE(rig.collect(7));
    rig.stage.reset();
    CHECK(rig.backend.count("scores_take") == 0);
    CHECK(rig.backend.count("scores_now 1 2") == 1);
    CHECK(rig.backend.count("scores_now 2 3") == 1);
    CHECK(rig.backend.count("scores_now 3 4") == 1);
  }
}

TEST_CASE("Étage gpu : facteur changé en direct, la grille repart de l'image suivante", "[app][stage]") {
  Rig rig;
  rig.start(rig.setup(true));
  rig.feed(1, 2);
  REQUIRE(rig.collect(3));
  rig.live->set_rife_factor(Factor{.num = 1, .den = 1});  // plus d'interpolation : l'image seule
  rig.feed(3, 4);
  REQUIRE(rig.collect(5));
  rig.live->set_rife_factor(Factor{.num = 3, .den = 1});  // ×3 : origine, puis deux instants par paire
  rig.feed(5, 6);
  REQUIRE(rig.collect(9));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) ==
        std::vector<std::string>{"1", "1-2", "2", "3", "4", "5", "5-6", "5-6", "6"});
  CHECK(rig.backend.count("engine rife") == 3);
}

TEST_CASE("Étage gpu : seek, les images de l'époque passée sont jetées, la nouvelle repart seule",
          "[app][stage]") {
  Rig rig;
  rig.start(rig.setup(true));
  rig.feed(1, 2);
  REQUIRE(rig.collect(3, false));  // les trois slots restent pris : la source 3 attendra
  rig.feed(3, 3);
  REQUIRE(rig.collect(4, false));  // son image intermédiaire prend le dernier slot, elle attend le sien
  REQUIRE(rig.stage->post(Flush{.epoch = 1}));
  rig.feed(4, 5, 1);
  rig.release_all();
  REQUIRE(rig.collect(7));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-2", "2", "2-3", "4", "4-5", "5"});
  CHECK(frames[3].epoch == 0);  // l'intermédiaire déjà écrite : main la jette, elle rend son slot
  CHECK(frames[4].epoch == 1);
  CHECK(rig.credits.load() == 5);  // la source 3 jetée rend aussi sa place
  CHECK(rig.backend.count("scores_now 3 4") + rig.backend.count("scores_start 3 4") == 0);  // autre époque
  CHECK(rig.reported().empty());
}

TEST_CASE("Étage gpu : en retard sur l'horloge maître, la paire sort sans interpolation", "[app][stage]") {
  Rig rig;
  test::FakeClock clock;
  clock.value =
      ClockSnapshot{.media_us = 10'000'000, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  UpscaleSetup s = rig.setup(true);
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 4);
  REQUIRE(rig.collect(4));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "4"});
  CHECK(rig.backend.count("engine rife") == 0);
  CHECK(rig.live->counts().late == 3);
  CHECK(rig.live->counts().pairs == 0);  // non mesurées : rien à interpoler
}

TEST_CASE("Étage gpu : un instant en retard est sauté, les suivants de la paire sont faits", "[app][stage]") {
  Rig rig;
  test::FakeClock clock;
  // L'horloge reste à 41 ms (vitesse quasi nulle) : prêts une période de sortie plus tard (13,3 ms
  // à ×3), les instants avant 54,3 ms sont en retard, soit le premier de la paire (1, 2) seul.
  clock.value =
      ClockSnapshot{.media_us = 41'000, .host_us = host_now_us(), .rate = 1e-6, .epoch = 0, .valid = true};
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 3);
  REQUIRE(rig.collect(6));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2", "2-3", "2-3", "3"});
  CHECK(pts_of(rig.sink.frames())[1] == 66'667);  // 53 333 sauté
  CHECK(rig.backend.count("engine rife") == 1 + 2);
  CHECK(rig.live->counts().late == 1);
  CHECK(rig.live->counts().interpolated == 2);
}

TEST_CASE("Étage gpu : DistilDRBA attend l'image suivante, la dernière n'est pas montrée", "[app][stage]") {
  Rig rig;
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(5));
  REQUIRE_FALSE(rig.collect(6, true, 200ms));
  CHECK(rig.stage->idle());  // la dernière image attend sa suivante : rien d'autre ne sortira
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3"});
  CHECK(rig.backend.count("engine drba") == 2);
  CHECK(rig.backend.count("rife_input") == 4);  // deux entrées par passe
  CHECK(rig.backend.count("ingest 4") == 1);    // ingérée en avance, derrière la source 3
  CHECK(rig.credits.load() == 3);
}

TEST_CASE("Étage gpu : refus de démarrage nommés", "[app][stage]") {
  Rig rig;
  const auto refused = [&](UpscaleSetup s) {
    auto started = UpscaleStage::start(rig.init(), std::move(s));
    REQUIRE_FALSE(started);
    return started.error().code;
  };
  UpscaleSetup work = rig.setup(false);
  work.work_width = kWidth * 2;
  work.work_height = kHeight;
  CHECK(refused(std::move(work)) == RefusalCode::ShapeInvalid);
  UpscaleSetup slots = rig.setup(false);
  slots.slots = test::StageRig::slots(2, kWidth, kHeight);  // la taille de travail, pas celle de SPAN
  CHECK(refused(std::move(slots)) == RefusalCode::ShapeInvalid);
  UpscaleSetup two = rig.setup(true);
  two.distil_drba = &rig.drba;
  CHECK(refused(std::move(two)) == RefusalCode::FieldOutOfRange);
  UpscaleSetup planes = rig.setup(true);
  planes.rife_planes = 5;
  CHECK(refused(std::move(planes)) == RefusalCode::InputMismatch);
  UpscaleSetup fixed = rig.setup(true);
  fixed.live = nullptr;
  fixed.rife_factor = Factor{.num = 1, .den = 1};  // sans réglages en direct, le facteur doit interpoler
  CHECK(refused(std::move(fixed)) == RefusalCode::FieldOutOfRange);
  UpscaleSetup none = rig.setup(false);
  none.backend = nullptr;
  CHECK(refused(std::move(none)) == RefusalCode::FieldMissing);
}

TEST_CASE("Étage gpu : une image refusée arrête la vidéo, les suivantes rendent leur place", "[app][stage]") {
  Rig rig;
  rig.backend.refuse_ingest(SourceId{3});
  rig.start(rig.setup(false));
  rig.feed(1, 5);
  REQUIRE(rig.collect(2));
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (rig.credits.load() < 5 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2"});
  CHECK(rig.credits.load() == 5);
  const std::vector<Refusal> reported = rig.reported();
  REQUIRE(reported.size() == 1);
  CHECK(reported[0].code == RefusalCode::ThreadFailed);
  CHECK(reported[0].detail.contains("image 3 refusée"));
}

}  // namespace ajp::app::play
