// DistilDRBA dans l'étage gpu (upscale_stage.h) sur le banc sans GPU (upscale_rig.h) : l'image suivante
// attendue, les côtés tenus rendus sans passe, et la variante mouvement (SPAN par source, warp après).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "media/shot_cutter.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace std::chrono_literals;
using namespace upscale_test;

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

TEST_CASE("Étage gpu : DistilDRBA rend b sans passe quand (b, après) est un dessin tenu", "[app][stage]") {
  Rig rig;
  rig.backend.score(SourceId{4}, SourceId{5}, 0.0, 10);  // dessin tenu
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(8));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) ==
        std::vector<std::string>{"1", "1-2", "2", "2-3", "3", "3-4", "4", "5"});
  // 3-4 à ½ est centrée sur 4, dont la paire suivante (4, 5) est tenue : l'image 4 telle quelle.
  CHECK(rig.backend.count("engine drba") == 2);
}

TEST_CASE("Étage gpu : DistilDRBA ne lit pas l'image d'après une coupe que seuls les plans connaissent",
          "[app][stage]") {
  // A09-01 : la coupe entre 3 et 4 est connue de la pré-analyse, et son score (0,10) passe le plancher des
  // plans sans atteindre le seuil brut. (a, b) la voyait déjà ; (b, après) la jugeait sur le score brut et
  // lisait 4.
  for (const bool known : {true, false}) {
    Rig rig;
    rig.backend.score(SourceId{3}, SourceId{4}, 0.10, 255);
    if (known) {
      media::Shot first;
      first.end_us = 140'000;
      first.cut_before = false;
      media::Shot second;
      second.start_us = 140'000;
      second.end_us = 400'000;
      const std::vector<media::Shot> shots{first, second};
      rig.live->cuts().set_shots(shots);
    }
    UpscaleSetup s = rig.setup(true);
    s.rife = nullptr;
    s.distil_drba = &rig.drba;
    rig.start(std::move(s));
    rig.feed(1, 5);
    REQUIRE(rig.collect(5));
    rig.stage.reset();

    const std::vector<std::string> shape = shape_of(rig.sink.frames());
    REQUIRE(shape.size() >= 5);
    CHECK(shape[3] == "2-3");
    // L'image entre 2 et 3, centrée sur 3 : (2, 3) puis (3, après). Après une coupe connue, 3 tient lieu
    // d'après.
    const auto pairs = rig.backend.rife_pairs();
    const bool read_next = std::ranges::count(pairs, std::pair<std::uint32_t, std::uint32_t>{3, 4}) > 0;
    CHECK(read_next == !known);
  }
}

TEST_CASE("Étage gpu : DistilDRBA n'ingère pas une image suivante d'une autre taille", "[app][stage]") {
  // A09-04 : la source rétrécit à l'image 3. L'image 2 sort sans lire la 3, et c'est la 3 qui rapporte
  // SourceResized, le refus qui rouvre la chaîne ; avant, l'ingestion de la 3 comme « image d'après »
  // échouait sur la 2.
  Rig rig;
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  rig.start(std::move(s));
  rig.feed(1, 2);
  REQUIRE(rig.push(test::host_frame(3, kWidth / 2, kHeight / 2, 3 * kPeriodUs, 0)));
  // DistilDRBA attend l'image suivante avant de traiter la 3.
  REQUIRE(rig.push(test::host_frame(4, kWidth / 2, kHeight / 2, 4 * kPeriodUs, 0)));
  std::vector<Refusal> reported;
  for (int tries = 0; tries < 500 && reported.empty(); ++tries) {
    rig.release_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    reported = rig.reported();
  }
  rig.stage.reset();
  REQUIRE(reported.size() == 1);
  CHECK(reported.front().code == RefusalCode::SourceResized);
  CHECK(rig.backend.count("ingest 3") == 0);
  const std::vector<std::string> shape = shape_of(rig.sink.frames());
  REQUIRE_FALSE(shape.empty());
  CHECK(shape.back() == "2");
}

TEST_CASE("Étage gpu : en lecture, DistilDRBA écrit b une fois quand (b, après) est tenu, sans la source",
          "[app][stage]") {
  Rig rig;
  rig.backend.score(SourceId{4}, SourceId{5}, 0.0, 10);  // dessin tenu
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  s.skip_held = true;
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3", "4"});
  CHECK(frames[5].pts_us == 140'000);
  CHECK(rig.backend.count("engine drba") == 2);
}

TEST_CASE("Étage gpu : DistilDRBA mouvement, un SPAN par source et le warp après", "[app][stage]") {
  Rig rig;
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba_motion;
  s.drba_motion = true;
  s.warp_after_span = true;
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(5));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3"});
  CHECK(rig.backend.count("engine drba_motion") == 2);
  // SPAN sur les trois sources montrées seulement : les images créées sont warpées depuis elles.
  CHECK(rig.backend.count("engine span") == 3);
  CHECK(rig.backend.count("warp_blend") + rig.backend.count("warp_blend_ring") == 2);
}

TEST_CASE("Étage gpu : DistilDRBA mouvement estimé plus petit que le travail, sur des copies réduites",
          "[app][stage]") {
  Rig rig;
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba_motion;
  s.drba_motion = true;
  s.warp_after_span = true;
  s.rife_width = kWidth / 2;
  s.rife_height = kHeight / 2;
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(5));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2", "2-3", "3"});
  CHECK(rig.backend.count("engine drba_motion") == 2);
  // Chaque source deux fois : au travail et réduite ; la réduite de l'image d'après resservant comme b.
  for (const char* entry : {"ingest 1", "ingest 2", "ingest 3", "ingest 4"})
    CHECK(rig.backend.count(entry) == 2);
  CHECK(rig.backend.count("warp_blend") + rig.backend.count("warp_blend_ring") == 2);
}

TEST_CASE("Étage gpu : DistilDRBA image refusé plus petit que le travail", "[app][stage]") {
  Rig rig;
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  s.rife_width = kWidth / 2;
  s.rife_height = kHeight / 2;
  auto started = UpscaleStage::start(rig.init(), std::move(s));
  REQUIRE_FALSE(started);
  CHECK(started.error().code == RefusalCode::ShapeInvalid);
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

}  // namespace ajp::app::play
