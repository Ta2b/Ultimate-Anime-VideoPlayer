// Session du thread main avec des doublures : amorce, horloge, images en retard, seek,
// pause, sous-titres, fin de flux.
#include "app/play/session.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "doubles.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;
using test::ring_frame;

struct Rig {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session;

  explicit Rig(SessionConfig config = {}) : session(std::move(config), stages, screen, clock) {}

  // L'horloge maître part de media_us à l'instant hôte courant de l'écran.
  void run_clock(std::int64_t media_us, std::uint32_t epoch = 0) {
    clock.value = {
        .media_us = media_us, .host_us = screen.now_us, .rate = 1.0, .epoch = epoch, .valid = true};
  }
  void tick() {
    auto shown = session.tick(screen.now_us);
    REQUIRE(shown);
  }
  std::int64_t shown_pts() const {
    REQUIRE(!screen.shown.empty());
    REQUIRE(screen.shown.back().frame);
    return screen.shown.back().frame->pts_us;
  }
};

TEST_CASE("Amorce : la première image part tout de suite, puis l'horloge démarre de son pts",
          "[app][session]") {
  Rig rig;
  rig.session.begin(0);
  rig.tick();
  CHECK_FALSE(rig.screen.shown.back().frame);  // rien encore : image noire et OSD
  CHECK(rig.stages.log.empty());

  rig.session.handle(ring_frame(0, 40'000));
  rig.session.handle(ring_frame(1, 80'000));
  rig.tick();
  CHECK(rig.shown_pts() == 40'000);
  CHECK(rig.stages.log == std::vector<std::string>{"play 0 40000"});

  // Horloge pas encore valide : l'image reste.
  rig.tick();
  CHECK(rig.shown_pts() == 40'000);
  rig.run_clock(80'000);
  rig.tick();
  CHECK(rig.shown_pts() == 80'000);
  REQUIRE(rig.stages.freed.size() == 1);  // le slot 0 revient à gpu, valeur suivante
  CHECK(rig.stages.freed[0].slot == 0);
  CHECK(rig.stages.freed[0].semaphore_value == 2);
}

TEST_CASE("Une horloge audio qui ne court jamais cède la main à l'horloge hôte", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 40'000));
  rig.session.handle(ring_frame(1, 80'000));
  rig.tick();  // amorce : la première image part, l'horloge ne démarre pas
  auto lost = [&] { return std::ranges::count(rig.stages.log, "audio perdu 0 40000"); };
  // Avant le délai, rien : un démarrage lent de l'audio est normal.
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(lost() == 0);
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(lost() == 0);  // le délai court depuis le premier constat
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(lost() == 1);
  CHECK(rig.session.refusals().size() == 1);  // l'avis est visible, non fatal
  CHECK_FALSE(rig.session.finished());
}

TEST_CASE("Une horloge qui court ou une pause n'est pas une perte d'audio", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 40'000));
  rig.session.handle(ring_frame(1, 80'000));
  rig.tick();
  for (int i = 0; i < 10; ++i) {
    rig.screen.now_us += 1'000'000;
    rig.run_clock(40'000);
    rig.tick();
  }
  CHECK(std::ranges::count_if(rig.stages.log,
                              [](const std::string& l) { return l.starts_with("audio perdu"); }) == 0);
}

TEST_CASE("Les images doublées par l'horloge sont sautées et leurs slots rendus", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  for (std::uint32_t i = 1; i <= 4; ++i) rig.session.handle(ring_frame(i, i * 41'708));
  rig.run_clock(3 * 41'708 + 1000);  // l'horloge est déjà sur la quatrième image
  rig.tick();
  CHECK(rig.shown_pts() == 3 * 41'708);
  CHECK(rig.session.stats().dropped == 2);
  // Rendus : les deux sautées (1, 2), puis celle qu'on remplace (0).
  REQUIRE(rig.stages.freed.size() == 3);
  CHECK(rig.stages.freed[0].slot == 1);
  CHECK(rig.stages.freed[1].slot == 2);
  CHECK(rig.stages.freed[2].slot == 0);
}

TEST_CASE("Une image pas encore écrite n'est ni montrée ni rendue : l'écran garde la sienne",
          "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  rig.run_clock(0);
  rig.screen.unwritten = {1, 2};
  rig.session.handle(ring_frame(1, 41'708));
  rig.session.handle(ring_frame(2, 83'417));
  rig.session.handle(ring_frame(3, 125'125));
  rig.run_clock(130'000);  // les trois sont dues ; la 3 est écrite, mais derrière la 1
  const std::size_t presented = rig.screen.shown.size();
  rig.tick();
  CHECK(rig.screen.shown.size() == presented);  // rien de neuf à présenter
  CHECK(rig.stages.freed.empty());
  // La 1 écrite : elle seule part, la 2 l'arrête encore.
  rig.screen.write(1);
  rig.tick();
  CHECK(rig.shown_pts() == 41'708);
  REQUIRE(rig.stages.freed.size() == 1);
  CHECK(rig.stages.freed[0].slot == 0);
  // Toutes écrites : la plus récente est montrée, la 2 sautée.
  rig.screen.write(2);
  rig.tick();
  CHECK(rig.shown_pts() == 125'125);
  CHECK(rig.session.stats().dropped == 1);
}

TEST_CASE("Retard des images : de la soumission à l'écriture vue par main", "[app][session]") {
  FrameLatency latency;
  Rig rig(SessionConfig{.latency = &latency});
  rig.screen.now_us = 1'000'000;
  rig.screen.unwritten = {0};
  RingFrame first = ring_frame(0, 40'000);
  first.submitted_us = 1'000'000;
  rig.session.handle(first);
  rig.tick();
  CHECK(latency.us() == 0);  // pas encore écrite
  rig.screen.now_us = 1'030'000;
  rig.screen.write(0);
  rig.tick();
  CHECK(latency.us() == 30'000);
  // Sans instant de soumission : rien à mesurer.
  rig.session.handle(ring_frame(1, 81'708));
  rig.tick();
  CHECK(latency.us() == 30'000);
  // Seek : une image de l'époque passée, rendue, ne se mesure plus, même écrite plus tard.
  rig.screen.unwritten = {2};
  RingFrame stale = ring_frame(2, 123'417);
  stale.submitted_us = rig.screen.now_us;
  rig.session.handle(stale);
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  rig.screen.now_us += 500'000;
  rig.screen.write(2);
  rig.tick();
  CHECK(latency.us() == 30'000);
}

TEST_CASE("Amorce : la première image attend d'être écrite, l'horloge aussi", "[app][session]") {
  Rig rig;
  rig.screen.unwritten = {0};
  rig.session.handle(ring_frame(0, 40'000));
  rig.tick();
  CHECK_FALSE(rig.screen.shown.back().frame);
  CHECK(rig.stages.log.empty());
  rig.screen.write(0);
  rig.tick();
  CHECK(rig.shown_pts() == 40'000);
  CHECK(rig.stages.log == std::vector<std::string>{"play 0 40000"});
}

TEST_CASE("Seek : une image pas encore écrite est rendue une fois écrite, jamais avant", "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.screen.unwritten = {1, 2};
  rig.session.handle(ring_frame(1, 1'041'000));
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  rig.session.handle(ring_frame(2, 1'083'000, 0));  // d'avant le seek, arrivée après
  rig.tick();
  CHECK(rig.stages.freed.empty());
  CHECK(rig.screen.retired.empty());
  rig.screen.write(2);
  rig.tick();
  REQUIRE(rig.stages.freed.size() == 1);
  CHECK(rig.stages.freed[0].slot == 2);
  rig.screen.write(1);
  rig.tick();
  REQUIRE(rig.stages.freed.size() == 2);
  CHECK(rig.stages.freed[1].slot == 1);
  // À l'arrêt, tout est rendu, écrit ou non.
  rig.screen.unwritten = {3};
  rig.session.handle(ring_frame(3, 5'000'000, 1));
  rig.session.end();
  CHECK(rig.stages.freed.size() == 4);
}

TEST_CASE("Seek : pause, Flush puis Seek, et les images de l'époque passée sont rendues", "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.handle(ring_frame(1, 1'041'000));
  rig.stages.log.clear();

  rig.session.command(cmd::SeekRelative{.delta_us = 5'000'000});
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "flush 1", "seek 6000000"});
  CHECK(rig.session.epoch() == 1);
  REQUIRE(rig.stages.freed.size() == 1);
  CHECK(rig.stages.freed[0].slot == 1);  // en file, d'avant le seek
  // Un second appui part de la cible du premier, pas de l'image affichée.
  rig.session.command(cmd::SeekRelative{.delta_us = 5'000'000});
  CHECK(rig.stages.log.back() == "seek 11000000");

  // Une image en retard de l'époque 0 arrive : rendue sans être montrée.
  rig.session.handle(ring_frame(2, 1'083'000, 0));
  CHECK(rig.session.stats().stale == 2);
  rig.tick();
  CHECK(rig.shown_pts() == 1'000'000);  // l'ancienne image reste jusqu'à la nouvelle

  rig.session.handle(ring_frame(3, 11'000'000, 2));
  rig.tick();
  CHECK(rig.shown_pts() == 11'000'000);
  CHECK(rig.stages.log.back() == "play 2 11000000");
}

TEST_CASE("Image par image : pause, puis un seek exact d'une image de la source", "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.stages.log.clear();

  rig.session.command(cmd::StepFrame{.direction = 1});
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "pause", "flush 1", "seek 1041667"});
  CHECK(rig.session.paused());
  // La touche répétée part de la cible, et reculer redescend d'une image.
  rig.session.command(cmd::StepFrame{.direction = 1});
  CHECK(rig.stages.log.back() == "seek 1083334");
  rig.session.command(cmd::StepFrame{.direction = -1});
  CHECK(rig.stages.log.back() == "seek 1041667");
}

TEST_CASE("Seek : un Flush refusé n'empêche pas le Seek, et le tout est rejoué jusqu'à passer",
          "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.stages.log.clear();

  rig.stages.refuse_flushes = 1;
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  // La source qui a reçu son Flush attend le Seek : il part malgré le refus d'une autre boîte.
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "flush 1 refusé", "seek 5000000"});
  CHECK(rig.session.refusals().size() == 1);

  // Le tour suivant, après le délai, rejoue Flush puis Seek à une époque neuve.
  rig.stages.log.clear();
  rig.screen.now_us += 100'000;
  rig.tick();
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "flush 2", "seek 5000000"});
  CHECK(rig.session.epoch() == 2);
  CHECK(rig.session.refusals().size() == 1);  // un seul avis, pas un par essai

  // Réussi, il n'est plus rejoué.
  rig.stages.log.clear();
  rig.screen.now_us += 100'000;
  rig.tick();
  CHECK(rig.stages.log.empty());
}

TEST_CASE("Un seek est borné au fichier", "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 10'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.command(cmd::SeekRelative{.delta_us = -5'000'000});
  CHECK(rig.stages.log.back() == "seek 0");
  rig.session.command(cmd::Seek{.target_us = 99'000'000});
  CHECK(rig.stages.log.back() == "seek 10000000");
}

TEST_CASE("Pause : l'horloge s'arrête, la reprise repart de l'image montrée", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  rig.session.command(cmd::TogglePause{});
  CHECK(rig.session.paused());
  CHECK(rig.stages.log.back() == "pause");
  rig.session.command(cmd::TogglePause{});
  CHECK_FALSE(rig.session.paused());
  CHECK(rig.stages.log.back() == "play 0 0");
}

TEST_CASE("Seek en pause : la nouvelle image s'affiche, l'horloge reste arrêtée", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  rig.session.command(cmd::Pause{});
  rig.session.command(cmd::Seek{.target_us = 2'000'000});
  rig.session.handle(ring_frame(1, 2'000'000, 1));
  rig.stages.log.clear();
  rig.tick();
  CHECK(rig.shown_pts() == 2'000'000);
  CHECK(rig.stages.log.empty());  // pas de Play
}

TEST_CASE("Sous-titres : une demande par image, l'image du même pts est dessinée", "[app][session]") {
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.session.handle(ring_frame(0, 0));
  rig.session.handle(ring_frame(1, 41'000));
  REQUIRE(rig.stages.requests.size() == 2);
  CHECK(rig.stages.requests[1].pts_us == 41'000);
  CHECK(rig.stages.requests[1].target.width == 1920);
  CHECK(rig.stages.requests[1].target.video_height == 1080);

  rig.session.handle(subs::SubtitleFrame{.pts_us = 0});
  rig.session.handle(subs::SubtitleFrame{.pts_us = 41'000});
  rig.tick();
  CHECK(rig.screen.shown.back().subtitles_pts == 0);
  rig.run_clock(41'000);
  rig.tick();
  CHECK(rig.screen.shown.back().subtitles_pts == 41'000);
}

TEST_CASE("Sous-titres : chaque vsync de l'image montrée a ses sous-titres, ring plein en avance",
          "[app][session]") {
  // Le clignotement du 19 octobre : l'image 0 à l'écran, trois images en avance dans le ring ;
  // leurs réponses chassaient celle de 0, qui perdait ses sous-titres jusqu'à l'image suivante.
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  for (std::uint32_t slot = 0; slot < 4; ++slot) rig.session.handle(ring_frame(slot, slot * 41'000));
  REQUIRE(rig.stages.requests.size() == kMaxSubtitleRequests);
  rig.session.handle(subs::SubtitleFrame{.pts_us = 0});
  rig.tick();  // amorce : 0 à l'écran
  CHECK(rig.screen.shown.back().subtitles_pts == 0);
  for (const auto& request : rig.stages.requests) {
    if (request.pts_us == 0) continue;
    rig.session.handle(subs::SubtitleFrame{.pts_us = request.pts_us});
    for (int vsync = 0; vsync < 7; ++vsync) {  // 24 im/s sur un écran de 165 Hz
      rig.tick();
      CHECK(rig.shown_pts() == 0);
      CHECK(rig.screen.shown.back().subtitles_pts == 0);
    }
  }
  CHECK(rig.session.stats().subtitle_gaps == 0);
}

TEST_CASE("Présentation à la demande : rien ne change, rien n'est recomposé", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();  // amorce : 0 à l'écran
  const std::size_t presents = rig.screen.shown.size();
  rig.tick();
  rig.tick();
  CHECK(rig.screen.shown.size() == presents);  // même image, même OSD : l'écran la garde
  rig.screen.now_us += 300'000;                // l'OSD est relu, son texte n'a pas changé
  rig.tick();
  CHECK(rig.screen.shown.size() == presents);
  rig.session.handle(ring_frame(1, 41'000));
  rig.run_clock(41'000);
  rig.tick();  // image neuve : présentée
  CHECK(rig.screen.shown.size() == presents + 1);
  CHECK(rig.shown_pts() == 41'000);
}

TEST_CASE("Cadence fixe : l'image est représentée à chaque période de l'écran", "[app][session]") {
  Rig rig(SessionConfig{.display_period_us = 6'000});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();  // amorce : 0 à l'écran
  const std::size_t presents = rig.screen.shown.size();
  rig.screen.now_us += 6'000;
  rig.tick();
  CHECK(rig.screen.shown.size() == presents);  // à la demande : rien de neuf, rien de présenté
  rig.session.set_steady_cadence(true);
  rig.tick();
  CHECK(rig.screen.shown.size() == presents + 1);  // la période est passée : même image, représentée
  rig.screen.now_us += 2'000;
  rig.tick();
  CHECK(rig.screen.shown.size() == presents + 1);  // pas avant la période suivante
  rig.screen.now_us += 3'600;
  rig.tick();
  CHECK(rig.screen.shown.size() == presents + 2);  // à une demi-milliseconde près
  CHECK(rig.shown_pts() == 0);
}

TEST_CASE("L'interface demande une présentation quand elle change, et une pour s'effacer", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  const std::size_t presents = rig.screen.shown.size();
  const auto* ui = reinterpret_cast<const ImDrawData*>(&rig);  // jamais lue par la doublure
  REQUIRE(rig.session.tick(rig.screen.now_us, UiFrame{.draw = ui, .changed = true}));
  CHECK(rig.screen.shown.size() == presents + 1);
  CHECK(rig.screen.shown.back().ui);
  REQUIRE(rig.session.tick(rig.screen.now_us, UiFrame{.draw = ui, .changed = false}));
  CHECK(rig.screen.shown.size() == presents + 1);  // inchangée : rien
  REQUIRE(rig.session.tick(rig.screen.now_us, UiFrame{}));
  CHECK(rig.screen.shown.size() == presents + 2);  // effacée : une présentation sans elle
  CHECK_FALSE(rig.screen.shown.back().ui);
}

TEST_CASE("Statistiques masquées par défaut ; les refus restent visibles", "[app][session]") {
  Rig rig(SessionConfig{.header = {"SPAN ×2"}});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  CHECK(rig.screen.shown.back().osd.empty());
  CHECK_FALSE(rig.session.view().stats);
  rig.session.command(cmd::ToggleOsd{});
  rig.tick();
  REQUIRE_FALSE(rig.screen.shown.back().osd.empty());
  CHECK(rig.screen.shown.back().osd[0] == "SPAN ×2");
  rig.session.command(cmd::ToggleOsd{});
  rig.session.handle(StageRefusal{.stage = "audio", .refusal = {RefusalCode::AudioDecodeFailed, "paquet"}});
  rig.screen.now_us += 300'000;
  rig.tick();
  REQUIRE(rig.screen.shown.back().osd.size() == 1);
  CHECK(rig.screen.shown.back().osd[0] == "audio : AudioDecodeFailed paquet");
}

TEST_CASE("Les mesures s'affichent une par une, dans l'ordre de l'OSD", "[app][session]") {
  struct Extra final : StatsSource {
    std::optional<std::string> rife() const override { return "RIFE : 3 paires"; }
    std::optional<std::string> memory() const override { return std::nullopt; }
  } extra;
  Rig rig(SessionConfig{
      .header = {"SPAN ×2"}, .tracks = {"audio aac jpn"}, .output = "sortie HDR10", .extra = &extra});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  CHECK(rig.screen.shown.back().osd.empty());
  rig.session.set_stats(StatsSettings{.rife = true, .output = true});
  rig.tick();
  REQUIRE(rig.screen.shown.back().osd.size() == 2);
  CHECK(rig.screen.shown.back().osd[0] == "sortie HDR10");
  CHECK(rig.screen.shown.back().osd[1] == "RIFE : 3 paires");
  rig.session.set_stats(StatsSettings{.all = true});
  rig.session.set_tracks({"audio flac eng"});
  rig.tick();
  const auto& all = rig.screen.shown.back().osd;
  REQUIRE(all.size() == 7);  // chaîne, piste, sortie, images, fluidité, RIFE, position (pas de mémoire)
  CHECK(all[1] == "audio flac eng");
  CHECK(all[4].starts_with("fluidité"));
  CHECK(all[5] == "RIFE : 3 paires");
}

TEST_CASE("La taille des sous-titres va dans les demandes, l'image à l'écran est redemandée",
          "[app][session]") {
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.session.handle(ring_frame(0, 0));
  rig.session.handle(subs::SubtitleFrame{.pts_us = 0, .epoch = 0});  // réponse : rien à dessiner
  rig.tick();
  REQUIRE_FALSE(rig.stages.requests.empty());
  CHECK(rig.stages.requests.back().target.font_scale == 1.0F);
  const std::size_t before = rig.stages.requests.size();
  rig.session.set_subtitle_scale(1.5F);
  REQUIRE(rig.stages.requests.size() == before + 1);
  CHECK(rig.stages.requests.back().pts_us == 0);
  CHECK(rig.stages.requests.back().target.font_scale == 1.5F);
}

TEST_CASE("Amorce : la première image attend ses sous-titres, au plus 100 ms", "[app][session]") {
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  CHECK_FALSE(rig.screen.shown.back().frame);  // pas encore : sa demande de sous-titres est en vol
  rig.session.handle(subs::SubtitleFrame{.pts_us = 0, .epoch = 0});
  rig.tick();
  REQUIRE(rig.screen.shown.back().frame);
  CHECK(rig.shown_pts() == 0);

  // Sans réponse, elle part quand même après 100 ms.
  Rig late(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  late.session.handle(ring_frame(0, 0));
  late.tick();
  CHECK_FALSE(late.screen.shown.back().frame);
  late.screen.now_us += 150'000;
  late.tick();
  REQUIRE(late.screen.shown.back().frame);
}

TEST_CASE("Volume et délai audio vont au thread audio", "[app][session]") {
  Rig rig;
  rig.session.command(cmd::SetVolume{.gain = 0.5F});
  rig.session.command(cmd::SetAudioDelay{.delay_us = -40'000});
  REQUIRE(rig.stages.log.size() == 2);
  CHECK(rig.stages.log[0] == "volume 0.50");
  CHECK(rig.stages.log[1] == "délai audio -40000");
}

TEST_CASE("La barre montre la cible d'un seek en cours, puis l'image", "[app][session]") {
  Rig rig(SessionConfig{.duration_us = 100'000'000});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  CHECK(rig.session.view().position_us == 0);
  CHECK(rig.session.view().duration_us == 100'000'000);
  rig.session.command(cmd::Seek{.target_us = 30'000'000});
  CHECK(rig.session.view().position_us == 30'000'000);  // avant même la première image
  rig.session.handle(ring_frame(1, 29'980'000, 1));
  rig.tick();
  CHECK(rig.session.view().position_us == 29'980'000);
}

TEST_CASE("Fin de flux : la lecture finit quand plus rien n'arrive", "[app][session]") {
  Rig rig;
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  rig.session.handle(StreamEnded{.epoch = 0});
  rig.tick();
  CHECK_FALSE(rig.session.finished());
  rig.screen.now_us += 2'000'000;
  rig.tick();
  CHECK(rig.session.finished());
  rig.session.end();
  CHECK(rig.stages.quit_called);
  CHECK(rig.stages.freed.size() == 1);
}

TEST_CASE("Durée de lecture et plein écran différé", "[app][session]") {
  Rig rig(SessionConfig{.stop_after_us = 1'000'000, .fullscreen_after_us = 500'000});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  rig.screen.now_us += 600'000;
  rig.tick();
  CHECK(rig.screen.is_fullscreen);
  CHECK_FALSE(rig.session.finished());
  rig.screen.now_us += 600'000;
  rig.tick();
  CHECK(rig.session.finished());
}

TEST_CASE("Un refus fatal d'un thread arrête la lecture et s'affiche", "[app][session]") {
  Rig rig;
  rig.session.handle(
      StageRefusal{.stage = "gpu", .refusal = {RefusalCode::ThreadFailed, "boum"}, .fatal = true});
  CHECK(rig.session.finished());
  REQUIRE(rig.session.refusals().size() == 1);
  CHECK(rig.session.refusals()[0] == "gpu : ThreadFailed boum");
}

TEST_CASE("L'OSD dit le temps par image et les images sautées", "[app][session]") {
  Rig rig(SessionConfig{.header = {"SPAN ×2"}, .stats = {.all = true}});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();
  const auto& osd = rig.screen.shown.back().osd;
  REQUIRE(osd.size() >= 4);
  CHECK(osd[0] == "SPAN ×2");
  CHECK(osd[1].find("sautées 0") != std::string::npos);
  // Sans période de sortie connue, ni cible ni manque.
  CHECK(osd[2] == "fluidité 0,0 im/s · manquées 0 · retard p95 0,0 ms");
  CHECK(osd[3].starts_with("00:00:00.000"));
}

TEST_CASE("Fluidité : les images manquées se lisent dans l'écart des pts", "[app][session]") {
  FrameStats stats;
  stats.paced(0, 0, 40'000);
  stats.paced(40'000, 1'000, 40'000);
  stats.paced(120'000, 0, 40'000);  // 80 000 n'a jamais paru
  CHECK(stats.missing == 1);
  stats.paced(150'000, 0, 40'000);  // plus tôt que prévu : rien ne manque
  CHECK(stats.missing == 1);
  stats.restart_pacing();
  stats.paced(1'000'000, 0, 40'000);  // après un seek : l'écart n'est pas un manque
  CHECK(stats.missing == 1);
  stats.paced(1'070'000, 0, 40'000);  // 1,75 période : une manquée
  CHECK(stats.missing == 2);
  stats.paced(1'100'000, -3'000, 0);  // période inconnue : rien de compté ; une avance n'est pas un retard
  CHECK(stats.missing == 2);
  CHECK(stats.lag.percentile(100) == 1'000);
  CHECK(stats.lag.latest() == 0);
  // Une coupe ou un dessin tenu, laissés sans interpolation à dessein : comptés à part.
  stats.paced(1'180'000, 0, 40'000, true);
  CHECK(stats.missing == 2);
  CHECK(stats.uninterpolated == 1);
}

TEST_CASE("Fluidité : les images neuves par seconde, relues chaque seconde", "[app][session]") {
  FrameStats stats;
  for (int k = 0; k <= 60; ++k) stats.presented(100, 1'000'000 + std::int64_t{k} * 16'667, true);
  CHECK(format_fps(stats.shown_fps) == "60,0");
  for (int k = 61; k <= 120; ++k) stats.presented(100, 1'000'000 + std::int64_t{k} * 16'667, k % 2 == 0);
  CHECK(format_fps(stats.shown_fps) == "30,0");  // une présentation sur deux est un redessin
}

// Une chaîne dont la cadence de sortie se règle pendant le test.
struct OutputRate final : StatsSource {
  double fps = 24000.0 / 1001.0 * 2.5;  // 23,976 im/s ×5/2 : 59,94 im/s
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() const override { return std::nullopt; }
  double output_fps() const override { return fps; }
};

TEST_CASE("Synchro sur l'écran : la vitesse suit la cadence de sortie, 1 sans calage", "[app][session]") {
  OutputRate rate;
  Rig rig(SessionConfig{.display_hz = 60.0, .display_sync = true, .extra = &rate});
  rig.tick();
  REQUIRE(rig.stages.speeds.size() == 1);
  CHECK(rig.stages.speeds[0] == Catch::Approx(1.001).epsilon(1e-9));
  CHECK(rig.session.speed() == rig.stages.speeds[0]);
  rig.tick();
  CHECK(rig.stages.speeds.size() == 1);  // inchangée : rien de redemandé
  rate.fps = 24000.0 / 1001.0 * 3.0;     // plus rapide que l'écran : aucun calage, l'audio mène
  rig.tick();
  REQUIRE(rig.stages.speeds.size() == 2);
  CHECK(rig.stages.speeds[1] == 1.0);
  rate.fps = 24000.0 / 1001.0 * 2.5;
  rig.session.set_display_sync(false);
  rig.tick();
  CHECK(rig.stages.speeds.size() == 2);  // coupée : la vitesse reste 1
  // Boîte audio pleine : rien ne change, la vitesse est redemandée au tour suivant.
  rig.session.set_display_sync(true);
  rig.stages.refuse_speed = true;
  rig.tick();
  CHECK(rig.session.speed() == 1.0);
  rig.stages.refuse_speed = false;
  rig.tick();
  CHECK(rig.session.speed() == Catch::Approx(1.001).epsilon(1e-9));
}

TEST_CASE("Latence de l'écran : l'image choisie est celle qui tombe à son apparition", "[app][session]") {
  // 60 Hz, rendu de 1 ms, file libre : l'image paraît 1 + 8,3 ms après son choix en moyenne ; on
  // choisit donc jusqu'à 1 + 8,3 + 8,3 = 17,7 ms d'avance sur l'horloge.
  Rig rig(SessionConfig{.display_period_us = 16'667});
  rig.session.handle(ring_frame(0, 0));
  rig.tick();  // amorce : 0 à l'écran
  rig.session.handle(ring_frame(1, 20'000));
  rig.run_clock(2'000);
  rig.tick();
  CHECK(rig.shown_pts() == 0);
  rig.run_clock(3'000);
  rig.tick();
  CHECK(rig.shown_pts() == 20'000);

  // Latence imposée : 30 ms, plus la demi-période.
  rig.session.set_video_latency(30'000);
  rig.session.handle(ring_frame(2, 55'000));
  rig.run_clock(16'000);
  rig.tick();
  CHECK(rig.shown_pts() == 20'000);
  rig.run_clock(17'000);
  rig.tick();
  CHECK(rig.shown_pts() == 55'000);
}

TEST_CASE("Latence de l'écran : file pleine, l'image passe après celle déjà en file", "[app][session]") {
  Rig rig(SessionConfig{.display_period_us = 16'667, .swapchain_depth = 2});
  rig.screen.queue_wait_us = 10'000;  // chaque présentation attend une place
  rig.session.handle(ring_frame(0, 0));
  rig.tick();  // amorce
  rig.session.set_steady_cadence(true);
  for (int vsync = 0; vsync < 8; ++vsync) {  // représentée à chaque période, attente comprise
    rig.screen.now_us += 16'667;
    rig.tick();
  }
  REQUIRE(rig.screen.shown.size() == 9);
  // Une période en file, une demi-période jusqu'au vsync, une demi-période de visée : 33,3 ms.
  rig.session.handle(ring_frame(1, 33'000));
  rig.run_clock(0);
  rig.tick();
  CHECK(rig.shown_pts() == 33'000);
}

}  // namespace
}  // namespace ajp::app::play
