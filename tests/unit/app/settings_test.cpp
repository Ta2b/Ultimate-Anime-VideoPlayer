// Réglages du menu des paramètres : plages, play.toml aller-retour, ligne de commande, et ce
// qu'un changement demande (direct ou réouverture de la chaîne).
#include "app/play/settings.h"

#include <array>
#include <filesystem>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("play.toml relit à l'identique ce qu'il a écrit", "[app][settings]") {
  PlaySettings s;
  s.model = "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
  s.work_height = 540;
  s.downscale = Downscale::CatmullRom;
  s.rife_factor = 4;
  s.true_hdr = true;
  s.true_hdr_contrast = 120;
  s.true_hdr_saturation = 90;
  s.true_hdr_gray = 40;
  s.true_hdr_peak = 800;
  s.true_hdr_debanding = false;
  s.audio_delay_ms = -120;
  s.volume = 0.85F;
  s.stats.frames = true;
  s.stats.memory = true;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(*back == s);
}

TEST_CASE("Une clé absente garde son défaut, une clé inconnue est ignorée", "[app][settings]") {
  const auto s = parse_settings("[rtx_hdr]\nenabled = true\nfutur = 3\n[autre]\nx = 1\n");
  REQUIRE(s);
  CHECK(s->true_hdr);
  CHECK(s->true_hdr_contrast == 100);
  CHECK(s->rife_factor == kDefaultRifeFactor);
  CHECK(s->downscale == Downscale::Area);
  const auto empty = parse_settings("");
  REQUIRE(empty);
  CHECK(*empty == PlaySettings{});
}

TEST_CASE("Hors plage, type faux ou TOML invalide : refus nommé", "[app][settings]") {
  const auto out = parse_settings("[rtx_hdr]\ncontrast = 250\n");
  REQUIRE_FALSE(out);
  CHECK(out.error().code == RefusalCode::FieldOutOfRange);
  CHECK(out.error().detail.find("rtx_hdr.contrast") != std::string::npos);
  const auto peak = parse_settings("[rtx_hdr]\npeak = 300\n");
  REQUIRE_FALSE(peak);
  CHECK(peak.error().detail.find("rtx_hdr.peak") != std::string::npos);
  const auto type = parse_settings("[interpolation]\nrife = \"trois\"\n");
  REQUIRE_FALSE(type);
  CHECK(type.error().code == RefusalCode::FieldType);
  CHECK(type.error().detail == "interpolation.rife");
  const auto filter = parse_settings("[image]\ndownscale = \"lanczos\"\n");
  REQUIRE_FALSE(filter);
  CHECK(filter.error().code == RefusalCode::FieldOutOfRange);
  const auto path = parse_settings("[image]\nmodel = \"../x.onnx\"\n");
  REQUIRE_FALSE(path);
  const auto broken = parse_settings("[image\n");
  REQUIRE_FALSE(broken);
  CHECK(broken.error().code == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Les options données remplacent play.toml pour cette lecture", "[app][settings]") {
  PlaySettings saved;
  saved.rife_factor = 2;
  saved.true_hdr = false;
  saved.work_height = 540;
  const std::array<std::string_view, 4> args{"film.mkv", "--rife=4", "--truehdr", "--truehdr-peak=900"};
  const auto options = parse_play_options(args);
  REQUIRE(options);
  const PlaySettings s = settings_for(saved, *options);
  CHECK(s.rife_factor == 4);
  CHECK(s.true_hdr);
  CHECK(s.true_hdr_peak == 900);
  CHECK(s.work_height == 540);  // non donnée : celle de play.toml
  const PlayOptions applied = options_for(*options, s);
  CHECK(applied.work_height == 540);
  CHECK(applied.rife_factor == 4);
}

TEST_CASE("Une option d'essai n'entre pas dans play.toml, un changement du menu oui", "[app][settings]") {
  PlaySettings file;
  file.rife_factor = 3;
  file.true_hdr_contrast = 100;
  const std::array<std::string_view, 3> args{"film.mkv", "--rife=4", "--truehdr-contrast=120"};
  const auto options = parse_play_options(args);
  REQUIRE(options);
  const PlaySettings started = settings_for(file, *options);
  PlaySettings running = started;
  running.true_hdr_saturation = 150;  // changé au menu
  running.true_hdr_contrast = 140;    // donné en option puis changé au menu
  const PlaySettings saved = settings_to_save(running, file, started, *options);
  CHECK(saved.rife_factor == 3);  // --rife=4 n'était que pour cette lecture
  CHECK(saved.true_hdr_contrast == 140);
  CHECK(saved.true_hdr_saturation == 150);
}

TEST_CASE("Ce qu'un changement demande : direct ou réouverture, dit avant d'appliquer", "[app][settings]") {
  const PlaySettings running;
  CHECK(change_between(running, running, true).cost == ChangeCost::None);
  PlaySettings live = running;
  live.true_hdr_contrast = 130;
  live.downscale = Downscale::Mitchell;
  live.volume = 0.5F;
  live.stats.all = true;
  CHECK(change_between(running, live, true).cost == ChangeCost::Live);
  PlaySettings height = running;
  height.work_height = 1080;
  const Change reopen = change_between(running, height, true);
  CHECK(reopen.cost == ChangeCost::Reopen);
  CHECK(reopen.reason.find("SPAN") != std::string::npos);
  PlaySettings hdr = running;
  hdr.true_hdr = true;
  CHECK(change_between(running, hdr, true).reason.find("RTX HDR") != std::string::npos);
  PlaySettings off;
  off.rife_factor = 0;
  PlaySettings on = off;
  on.rife_factor = 3;
  CHECK(change_between(off, on, true).cost == ChangeCost::Live);     // moteur RIFE chargé
  CHECK(change_between(off, on, false).cost == ChangeCost::Reopen);  // à charger
  CHECK(change_between(on, off, false).cost == ChangeCost::Live);
}

TEST_CASE("La part directe d'un changement laisse ce qui demande une réouverture", "[app][settings]") {
  const PlaySettings running;
  PlaySettings wanted = running;
  wanted.work_height = 1080;  // réouverture
  wanted.true_hdr = true;     // réouverture
  wanted.true_hdr_contrast = 150;
  wanted.downscale = Downscale::CatmullRom;
  wanted.stats.frames = true;
  wanted.rife_factor = 2;
  const PlaySettings live = live_part(running, wanted, true);
  CHECK(live.work_height == running.work_height);
  CHECK_FALSE(live.true_hdr);
  CHECK(live.true_hdr_contrast == 150);
  CHECK(live.downscale == Downscale::CatmullRom);
  CHECK(live.stats.frames);
  CHECK(live.rife_factor == 2);
  CHECK(change_between(running, live, true).cost == ChangeCost::Live);
  PlaySettings off = running;
  off.rife_factor = 0;
  PlaySettings on = off;
  on.rife_factor = 3;
  CHECK(live_part(off, on, false).rife_factor == 0);  // RIFE à charger : attend « Appliquer »
}

TEST_CASE("Écrire puis relire play.toml ; absent : les défauts", "[app][settings]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-settings-test";
  std::filesystem::remove_all(dir);
  const auto path = dir / "play.toml";
  const auto missing = load_settings(path);
  REQUIRE(missing);
  CHECK(*missing == PlaySettings{});
  PlaySettings s;
  s.rife_factor = 2;
  REQUIRE(save_settings(path, s));
  const auto back = load_settings(path);
  REQUIRE(back);
  CHECK(back->rife_factor == 2);
  PlaySettings bad;
  bad.true_hdr_gray = 5;
  CHECK_FALSE(save_settings(path, bad));
  CHECK(load_settings(path)->rife_factor == 2);  // l'ancien fichier reste
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace ajp::app::play
