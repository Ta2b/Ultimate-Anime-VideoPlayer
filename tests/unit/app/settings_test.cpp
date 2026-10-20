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
  s.subtitle_scale = 1.25F;
  s.stats.frames = true;
  s.stats.memory = true;
  s.hdr_native = true;
  s.grade_preset = render::GradePreset::Custom;
  s.grade_values.shadows = 0.55;
  s.grade_values.guard = 0.25;
  s.grade_values.gamut_p3 = true;
  s.oled_black = OledBlack::On;
  s.oled_black_steps = 3.5;
  s.oled_care = false;
  s.steady_cadence = true;
  s.smooth_motion = true;
  s.motion_engine = MotionEngine::OpticalFlow;
  s.rife_model = "rife_v4.26_heavy.onnx";
  s.keep_poses = true;
  s.grade_values.expand = 0.3;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(*back == s);
  s.motion_engine = MotionEngine::Gmfss;
  CHECK(parse_settings(format_settings(s))->motion_engine == MotionEngine::Gmfss);
}

TEST_CASE("Grade : un préréglage nommé donne ses valeurs, gamut P3 reste celui du fichier",
          "[app][settings]") {
  const PlaySettings defaults;
  CHECK(defaults.grade);
  CHECK(defaults.grade_preset == render::GradePreset::Eclatant);
  const auto natural = parse_settings("[grade]\npreset = \"naturel\"\nshadows = 0.7\ngamut_p3 = true\n");
  REQUIRE(natural);
  CHECK(natural->grade_values.shadows == render::GradeSettings{}.shadows);
  CHECK(natural->grade_values.gamut_p3);
  CHECK_FALSE(parse_settings("[grade]\npreset = \"criard\"\n"));
  CHECK_FALSE(parse_settings("[grade]\noled_black = \"peut-etre\"\n"));
  CHECK_FALSE(parse_settings("[grade]\noled_black_steps = 20.0\n"));
  CHECK(defaults.oled_black == OledBlack::Auto);
  CHECK(parse_settings("[grade]\npreset = \"perso\"\nguard = 2.0\n").error().detail.find("guard") !=
        std::string::npos);
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
  const auto quarter = parse_settings("[interpolation]\nrife = 1.25\n");
  REQUIRE_FALSE(quarter);
  CHECK(quarter.error().code == RefusalCode::FieldOutOfRange);
  const auto half = parse_settings("[interpolation]\nrife = 1.5\n");
  REQUIRE(half);
  CHECK(factor_of(*half) == Factor{.num = 3, .den = 2});
  CHECK(parse_settings(format_settings(*half)).value().rife_factor_den == 2);
  const auto filter = parse_settings("[image]\ndownscale = \"lanczos\"\n");
  REQUIRE_FALSE(filter);
  CHECK(filter.error().code == RefusalCode::FieldOutOfRange);
  const auto path = parse_settings("[image]\nmodel = \"../x.onnx\"\n");
  REQUIRE_FALSE(path);
  CHECK(parse_settings("[subtitles]\nscale = 3.0\n").error().detail.find("subtitles.scale") !=
        std::string::npos);
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
  live.hdr_native = true;
  live.grade_preset = render::GradePreset::Naturel;
  live.grade_values = render::grade_preset(render::GradePreset::Naturel);
  CHECK(change_between(running, live, true).cost == ChangeCost::Live);
  CHECK(live_part(running, live, true) == live);
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
  // Le moteur du mouvement se recharge s'il interpole ; sans interpolation, il attend.
  PlaySettings flow = on;
  flow.motion_engine = MotionEngine::OpticalFlow;
  CHECK(change_between(on, flow, true).reason.find("mouvement") != std::string::npos);
  PlaySettings flow_off = off;
  flow_off.motion_engine = MotionEngine::OpticalFlow;
  CHECK(change_between(off, flow_off, true).cost == ChangeCost::Live);
  CHECK(live_part(off, flow_off, true).motion_engine == MotionEngine::OpticalFlow);
  CHECK(live_part(on, flow, true).motion_engine == MotionEngine::Rife);
  PlaySettings heavy = on;
  heavy.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(change_between(on, heavy, true).reason.find("modèle RIFE") != std::string::npos);
  // Avec GMFSS, le modèle RIFE ne sert pas : le changer ne recharge rien.
  PlaySettings gmfss = on;
  gmfss.motion_engine = MotionEngine::Gmfss;
  PlaySettings gmfss_heavy = gmfss;
  gmfss_heavy.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(change_between(gmfss, gmfss_heavy, true).cost == ChangeCost::Live);
  CHECK(live_part(gmfss, gmfss_heavy, true).rife_model == "rife_v4.26_heavy.onnx");
}

TEST_CASE("Libellé d'un modèle RIFE", "[app][settings]") {
  CHECK(rife_label("rife_v4.26_heavy.onnx") == "RIFE 4.26 heavy");
  CHECK(rife_label("rife_v4.26.onnx") == "RIFE 4.26");
  CHECK(rife_label("autre.onnx") == "autre.onnx");
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
