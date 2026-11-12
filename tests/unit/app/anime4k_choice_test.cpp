// Anime4K dans la liste des modèles (anime4k_choice.h) : son nom dans play.toml, son libellé, quand il
// remplace SPAN, ce qu'il demande au changement et sans moteur de réseau.
#include "app/play/anime4k_choice.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/auto_config.h"
#include "app/play/first_run.h"
#include "app/play/settings.h"
#include "core/settings.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

TEST_CASE("Anime4K : deux entrées de la liste des modèles, à leur nom", "[app][anime4k]") {
  CHECK(anime4k_size("anime4k-fast") == render::Anime4kSize::Fast);
  CHECK(anime4k_size("anime4k-hq") == render::Anime4kSize::Quality);
  CHECK_FALSE(anime4k_size(""));
  CHECK_FALSE(anime4k_size("2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"));
  CHECK_FALSE(anime4k_size("Anime4K-fast"));  // le nom exact de play.toml, pas un à peu près
  CHECK(model_label("anime4k-fast") == "Anime4K Fast");
  CHECK(model_label("anime4k-hq") == "Anime4K HQ");
}

TEST_CASE("Anime4K : il remplace SPAN quand l'agrandissement tourne, au mode de la source",
          "[app][anime4k]") {
  // 1080p réduite en 540p (Économe sur un écran 1080p) : C ; gardée en 1080p : A ; 720p : B.
  const auto reduced = anime4k_choice("anime4k-fast", true, 1080, 540);
  REQUIRE(reduced);
  CHECK(*reduced == render::Anime4kChoice{.size = render::Anime4kSize::Fast, .mode = render::Anime4kMode::C});
  CHECK(anime4k_choice("anime4k-hq", true, 1080, 1080)->mode == render::Anime4kMode::A);
  CHECK(anime4k_choice("anime4k-hq", true, 720, 720)->size == render::Anime4kSize::Quality);
  CHECK(anime4k_choice("anime4k-hq", true, 720, 720)->mode == render::Anime4kMode::B);
  // Sans agrandissement (case décochée, ou source gardée par « restaurer les traits » coupé) : rien.
  CHECK_FALSE(anime4k_choice("anime4k-fast", false, 1080, 540));
  // Un modèle ONNX : SPAN, pas Anime4K.
  CHECK_FALSE(anime4k_choice("", true, 1080, 540));
}

TEST_CASE("Anime4K : se garde dans play.toml", "[app][anime4k][settings]") {
  PlaySettings s;
  s.model = std::string(kAnime4kQuality);
  s.work_height = 540;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->model == kAnime4kQuality);
}

TEST_CASE("Anime4K : passer de SPAN à lui rouvre la chaîne sans rien construire",
          "[app][anime4k][settings]") {
  PlaySettings span;
  PlaySettings shaders = span;
  shaders.model = std::string(kAnime4kFast);
  const Change change = change_between(span, shaders, false);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.contains("Anime4K"));
  CHECK_FALSE(change.reason.contains("built the first time"));
  // Et le retour à SPAN dit qu'il se recharge.
  CHECK(change_between(shaders, span, false).reason.contains("SPAN reloads"));
}

TEST_CASE("Anime4K : sans moteur de réseau, son agrandissement reste", "[app][anime4k][first_run]") {
  const FrenchUi french;
  PlaySettings s;
  s.upscale = true;
  s.model = std::string(kAnime4kFast);
  s.rife_factor = 2;
  core::Settings none;
  none.first_run_done = true;
  none.chosen_engines.clear();
  std::vector<std::string> notes;
  const PlaySettings kept = within_engines(s, none, notes);
  CHECK(kept.upscale);           // des shaders, pas un réseau
  CHECK(kept.rife_factor == 0);  // RIFE, lui, demande un moteur
  CHECK(notes.size() == 1);
  // Un modèle ONNX perd son agrandissement, comme avant.
  s.model.clear();
  CHECK_FALSE(within_engines(s, none, notes).upscale);
}

TEST_CASE("Anime4K : hors de l'Auto et du préréglage Économe tant que le banc n'a pas parlé",
          "[app][anime4k][auto]") {
  // kAnime4kInAuto (auto_config.h) : Économe joue la source telle quelle, sans agrandissement.
  STATIC_CHECK_FALSE(kAnime4kInAuto);
  PlaySettings s;
  s.preset = Preset::Economy;
  const PlaySettings economy =
      apply_preset(s, Machine{.tier = Tier::Economy, .display_height = 1080, .display_hz = 60.0},
                   core::SourceInfo{.width = 1920, .height = 1080, .fps = 23.976});
  CHECK_FALSE(economy.upscale);
  CHECK_FALSE(anime4k_size(economy.model));
}

}  // namespace
}  // namespace ajp::app::play
