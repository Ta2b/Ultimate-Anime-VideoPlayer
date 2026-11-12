// Les réglages du menu sont appliqués ou annoncés, jamais perdus (C12 : A22-01, A22-05, A22-09, A22-15).
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/settings.h"

namespace ajp::app::play {

// A22-01, A22-15 : chaque réglage que le menu change est appliqué en direct, ou annoncé par « Appliquer »
// (la réouverture) : jamais ni l'un ni l'autre, sinon il se perd à la fermeture du panneau.
TEST_CASE("Réglages du menu : « Restaurer les traits », la carte et les moteurs fusionnés ne se perdent pas",
          "[app][settings]") {
  PlaySettings running;
  running.rife_factor = 0;

  PlaySettings lines = running;
  lines.restore_lines = false;
  CHECK(change_between(running, lines, true).cost == ChangeCost::Reopen);

  // La carte vaut au prochain lancement : prise tout de suite, donc écrite.
  PlaySettings card = running;
  card.gpu = "NVIDIA GeForce RTX 4070 Ti";
  CHECK(live_part(running, card, true).gpu == card.gpu);

  // Sans interpolation, rien à recharger : les deux réglages fusionnés passent en direct.
  PlaySettings fused = running;
  fused.rife_fused = false;
  fused.drba_fused = false;
  const PlaySettings applied = live_part(running, fused, true);
  CHECK_FALSE(applied.rife_fused);
  CHECK_FALSE(applied.drba_fused);
  CHECK(change_between(applied, fused, true).cost != ChangeCost::Reopen);

  // Avec RIFE, le changer rouvre la lecture et le dit.
  running.rife_factor = 3;
  PlaySettings whole = running;
  whole.rife_fused = false;
  CHECK(change_between(running, whole, true).cost == ChangeCost::Reopen);
}

// A22-05 : `--rife` coupe le mode auto pour la lecture ; play.toml garde le mode auto du fichier.
TEST_CASE("L'option --rife n'enregistre pas la coupure du mode auto", "[app][settings]") {
  PlaySettings from_file;
  from_file.rife_auto = true;
  from_file.rife_factor = 4;
  PlayOptions options;
  options.rife_factor = 3;
  options.rife_factor_den = 1;
  options.given.push_back("--rife");
  const PlaySettings started = settings_for(from_file, options);
  REQUIRE_FALSE(started.rife_auto);
  const PlaySettings saved = settings_to_save(started, from_file, started, options);
  CHECK(saved.rife_auto);
  CHECK(saved.rife_factor == 4);
}

}  // namespace ajp::app::play

namespace ajp::app::play {
namespace {

// A22-15 : chaque réglage que le menu change ressort de live_part (appliqué, donc enregistré) ou de
// change_between (« Appliquer » le rouvre) ; aucun ne se perd à la fermeture du panneau. Les exceptions sont
// dites : ce que le menu ne change pas lui-même.
TEST_CASE("Chaque réglage du menu est appliqué ou annoncé", "[app][settings]") {
  PlaySettings base;
  base.rife_factor = 3;
  base.rife_auto = false;
  using Mutation = void (*)(PlaySettings&);
  const std::vector<std::pair<std::string_view, Mutation>> changes{
      {"preset", [](PlaySettings& s) { s.preset = Preset::Custom; }},
      {"upscale", [](PlaySettings& s) { s.upscale = false; }},
      {"restore_lines", [](PlaySettings& s) { s.restore_lines = false; }},
      {"model", [](PlaySettings& s) { s.model = "autre.onnx"; }},
      {"work_height", [](PlaySettings& s) { s.work_height = 540; }},
      {"downscale", [](PlaySettings& s) { s.downscale = Downscale::Spline36; }},
      {"rife_model", [](PlaySettings& s) { s.rife_model = "autre.onnx"; }},
      {"motion_engine", [](PlaySettings& s) { s.motion_engine = MotionEngine::DistilDrba; }},
      {"keep_poses", [](PlaySettings& s) { s.keep_poses = !s.keep_poses; }},
      {"rife_fused", [](PlaySettings& s) { s.rife_fused = !s.rife_fused; }},
      {"smooth_drawings", [](PlaySettings& s) { s.smooth_drawings = !s.smooth_drawings; }},
      {"pair_scale", [](PlaySettings& s) { s.pair_scale = !s.pair_scale; }},
      {"span_precision", [](PlaySettings& s) { s.span_precision = PrecisionTier::Quality; }},
      {"motion_precision", [](PlaySettings& s) { s.motion_precision = PrecisionTier::Quality; }},
      {"span_order", [](PlaySettings& s) { s.span_order = SpanOrder::After; }},
      {"final_scaler", [](PlaySettings& s) { s.final_scaler = FinalScaler::EwaLanczos; }},
      {"deband", [](PlaySettings& s) { s.deband = !s.deband; }},
      {"sharpness", [](PlaySettings& s) { s.sharpness = 40; }},
      {"engines_dir", [](PlaySettings& s) { s.engines_dir = "/srv/moteurs"; }},
      {"gpu", [](PlaySettings& s) { s.gpu = "Une carte"; }},
      {"language", [](PlaySettings& s) { s.language = "fr"; }},
      {"audio_delay_ms", [](PlaySettings& s) { s.audio_delay_ms = 80; }},
      {"audio_device", [](PlaySettings& s) { s.audio_device = "casque"; }},
      {"display_sync", [](PlaySettings& s) { s.display_sync = !s.display_sync; }},
      {"video_latency_ms", [](PlaySettings& s) { s.video_latency_ms = 30; }},
      {"volume", [](PlaySettings& s) { s.volume = 0.5F; }},
      {"subtitle_scale", [](PlaySettings& s) { s.subtitle_scale = 1.5F; }},
      {"subtitle_delay_ms", [](PlaySettings& s) { s.subtitle_delay_ms = 200; }},
      {"grade", [](PlaySettings& s) { s.grade = !s.grade; }},
      {"look_contrast", [](PlaySettings& s) { s.look_contrast = 20; }},
      {"flash_guard", [](PlaySettings& s) { s.flash_guard = !s.flash_guard; }},
      {"surface_white", [](PlaySettings& s) { s.surface_white = 70; }},
      {"oled_care", [](PlaySettings& s) { s.oled_care = !s.oled_care; }},
      {"autoplay_next", [](PlaySettings& s) { s.autoplay_next = !s.autoplay_next; }},
      {"update_mode", [](PlaySettings& s) { s.update_mode = UpdateMode::Never; }},
  };
  for (const auto& [name, change] : changes) {
    CAPTURE(name);
    PlaySettings wanted = base;
    change(wanted);
    REQUIRE(wanted != base);
    const bool applied = live_part(base, wanted, true) != base;
    const bool announced = change_between(base, wanted, true).cost == ChangeCost::Reopen;
    CHECK((applied || announced));
  }
  // Sans interpolation, rien à recharger : les réglages fusionnés passent en direct.
  PlaySettings off = base;
  off.rife_factor = 0;
  PlaySettings off_wanted = off;
  off_wanted.drba_fused = false;
  CHECK(live_part(off, off_wanted, false) != off);
}

// « Lisser les dessins tenus » : recharge la chaîne avec RIFE (l'étage lit cinq images en avance), passe en
// direct sans interpolation, l'emporte sur les poses gardées, et --smooth-drawings ne s'enregistre pas.
TEST_CASE("Réglages : « Lisser les dessins tenus » rouvre la lecture, l'option de séance ne s'écrit pas",
          "[app][settings][smooth]") {
  PlaySettings running;
  running.rife_factor = 3;
  PlaySettings smooth = running;
  smooth.smooth_drawings = true;
  const Change change = change_between(running, smooth, true);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("held drawings are smoothed") != std::string::npos);
  // La chaîne ouverte a chargé le Smooth : la rouvrir pour le garder ne sert à rien.
  OpenedChain opened;
  opened.rife_loaded = true;
  opened.loaded = loaded_by(smooth);
  opened.shape = chain_shape(smooth, opened);
  CHECK(change_between(smooth, smooth, true, &opened).cost != ChangeCost::Reopen);

  PlaySettings off = running;
  off.rife_factor = 0;
  PlaySettings off_smooth = off;
  off_smooth.smooth_drawings = true;
  CHECK(live_part(off, off_smooth, true).smooth_drawings);

  // Les poses gardées cèdent au Smooth, leur réglage reste tel quel.
  CHECK(poses_kept(running));
  CHECK_FALSE(poses_kept(smooth));
  CHECK(smooth.keep_poses);

  PlaySettings from_file;
  PlayOptions options;
  options.smooth_drawings = true;
  const PlaySettings started = settings_for(from_file, options);
  CHECK(started.smooth_drawings);
  CHECK_FALSE(settings_to_save(started, from_file, started, options).smooth_drawings);
  // Sans l'option, le choix du fichier reste.
  from_file.smooth_drawings = true;
  CHECK(settings_to_save(started, from_file, started, PlayOptions{}).smooth_drawings);
}

TEST_CASE("Écriture refusée : save_settings le dit au lieu de se taire", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-settings-refused";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  // Le « dossier » du fichier est un fichier : l'écriture ne peut pas aboutir.
  std::ofstream(dir / "pas-un-dossier") << "x";
  const auto written = save_settings(dir / "pas-un-dossier" / "play.toml", PlaySettings{});
  CHECK_FALSE(written);
  std::filesystem::remove_all(dir);
}

TEST_CASE("Un dossier d'installation absurde est refusé à l'écriture (A22-09)", "[app][settings]") {
  PlaySettings s;
  s.engines_dir = std::string(kMaxEnginesDirBytes + 1, 'x');
  CHECK_FALSE(validate(s));
  s.engines_dir = "/tmp/a\nb";
  CHECK_FALSE(validate(s));
  s.engines_dir = "/srv/moteurs";
  CHECK(validate(s));
}

}  // namespace
}  // namespace ajp::app::play
