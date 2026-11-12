// Réglages du menu des paramètres : plages, play.toml aller-retour, ligne de commande, et ce
// qu'un changement demande (direct ou réouverture de la chaîne).
#include "app/play/settings.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/export_job.h"
#include "app/play/free_engines.h"
#include "app/play/host_facts.h"
#include "app/play/live_settings.h"
#include "app/play/settings_sound.h"

namespace ajp::app::play {
namespace {

TEST_CASE("play.toml relit à l'identique ce qu'il a écrit", "[app][settings]") {
  PlaySettings s;
  s.model = "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
  s.work_height = 540;
  s.downscale = Downscale::CatmullRom;
  s.rife_factor = 4;
  s.true_hdr_peak = 800;
  s.audio_delay_ms = -120;
  s.display_sync = false;
  s.video_latency_ms = 40;
  s.volume = 0.85F;
  s.subtitle_scale = 1.25F;
  s.stats.frames = true;
  s.stats.memory = true;
  s.grade_values.shadows = 0.55;
  s.grade_values.area_rtx = 0.25;
  s.oled_black = OledBlack::On;
  s.final_scaler = FinalScaler::EwaLanczosSharp;
  s.deband = true;
  s.oled_black_steps = 3.5;
  s.oled_care = false;
  s.steady_cadence = true;
  s.flash_guard = false;
  s.face_skin = true;
  s.surface_white = 80;
  s.smooth_motion = true;
  s.motion_engine = MotionEngine::DistilDrba;
  s.rife_model = "rife_v4.22_lite.onnx";
  s.keep_poses = true;
  s.span_precision = PrecisionTier::Performance;
  s.motion_precision = PrecisionTier::Quality;
  s.rife_auto = true;
  s.span_order = SpanOrder::Before;
  s.grade_values.highlights_rtx = 0.7;
  s.language_presets = {{"eng", ""}, {"", "fre"}};
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(*back == s);
  // P5.2 : GMFSS est retiré, et un ancien play.toml qui le nomme se lit en RIFE, (A22-18).
  for (const char* old : {"gmfss", "gmfss_union"}) {
    const auto read = parse_settings(std::format("[interpolation]\nengine = \"{}\"\n", old));
    REQUIRE(read);
    CHECK(read->motion_engine == MotionEngine::Rife);
  }
  // L'interpolation d'Apple se garde partout ; elle n'est proposée que sur un Mac.
  s.motion_engine = MotionEngine::Apple;
  CHECK(parse_settings(format_settings(s))->motion_engine == MotionEngine::Apple);
#if defined(__APPLE__)
  CHECK(offered(MotionEngine::Apple));
#else
  CHECK_FALSE(offered(MotionEngine::Apple));
#endif
  CHECK(offered(MotionEngine::Rife));
  // Les niveaux du menu : Très rapide, Rapide, Équilibré, puis Apple.
  CHECK(interp_level(MotionEngine::DistilDrba, "") == InterpLevel::VeryFast);
  CHECK(interp_level(MotionEngine::Rife, "rife_v4.22_lite.onnx") == InterpLevel::Fast);
  CHECK(interp_level(MotionEngine::Rife, "rife_v4.25.onnx") == InterpLevel::Balanced);
  CHECK(interp_level(MotionEngine::Apple, "") == InterpLevel::System);
  CHECK_FALSE(interp_level(MotionEngine::Rife, "rife_v4.6.onnx"));
}

TEST_CASE("La netteté se garde dans play.toml, bornée de 0 à 100 %, en direct", "[app][settings]") {
  PlaySettings s;
  s.sharpness = 40;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->sharpness == 40);
  CHECK_FALSE(parse_settings("[picture]\nsharpness = 150\n"));
  CHECK(parse_settings("")->sharpness == 0);
  const PlaySettings running;
  CHECK(change_between(running, s, true).cost == ChangeCost::Live);
}

TEST_CASE("Le HDR suit le système : un écran HDR trop sombre reste en SDR", "[app][settings]") {
  CHECK_FALSE(system_hdr(std::nullopt));  // écran SDR, ou HDR coupé dans Windows
  CHECK_FALSE(system_hdr(300.0));         // se dit HDR, n'en a pas la lumière
  CHECK_FALSE(system_hdr(std::numeric_limits<double>::quiet_NaN()));
  CHECK(system_hdr(kMinHdrPeakNits));
  CHECK(system_hdr(1000.0));
}

TEST_CASE("Le mode HDR n'est plus un réglage : seul le pic se lit, les fichiers d'avant aussi",
          "[app][settings]") {
  // engine (HDR ou SDR) et contrast (HDR adaptatif retiré) sont ignorés : le HDR suit le système.
  for (const std::string_view engine : {"adaptive", "sdr", "rtx", "maxine"}) {
    const auto s =
        parse_settings(std::format("[hdr]\nengine = \"{}\"\ncontrast = 150\npeak = 800\n", engine));
    REQUIRE(s);
    CHECK_FALSE(s->hdr_native);
    CHECK(s->true_hdr_peak == 800);
    CHECK(*parse_settings(format_settings(*s)) == *s);
  }
  PlaySettings hdr_on;
  hdr_on.hdr_native = true;
  const std::string written = format_settings(hdr_on);
  CHECK(written.find("[rtx_hdr]") == std::string::npos);
  CHECK(written.find("engine = \"adaptive\"") == std::string::npos);
  CHECK_FALSE(parse_settings(written)->hdr_native);  // jamais écrit dans play.toml
  // Un play.toml d'avant : section rtx_hdr, seul son pic compte.
  const auto rtx = parse_settings("[rtx_hdr]\nenabled = true\ncontrast = 200\npeak = 700\n");
  REQUIRE(rtx);
  CHECK_FALSE(rtx->hdr_native);
  CHECK(rtx->true_hdr_peak == 700);
  // La nouvelle section l'emporte sur l'ancienne.
  const auto both = parse_settings("[rtx_hdr]\npeak = 700\n[hdr]\npeak = 900\n");
  REQUIRE(both);
  CHECK(both->true_hdr_peak == 900);
}

TEST_CASE("Warp et SPAN : Auto passe après SPAN au-dessus de 1080p ; les fichiers d'avant",
          "[app][settings]") {
  CHECK(PlaySettings{}.span_order == SpanOrder::Auto);
  CHECK(PlaySettings{}.span_precision == PrecisionTier::Balanced);
  CHECK(PlaySettings{}.autoplay_next);  // P3.84
  CHECK(PlaySettings{}.motion_precision == PrecisionTier::Balanced);
  CHECK_FALSE(warp_after_span(SpanOrder::Auto, 1080));
  CHECK(warp_after_span(SpanOrder::Auto, 1440));
  CHECK(warp_after_span(SpanOrder::Auto, 2160));
  CHECK_FALSE(warp_after_span(SpanOrder::Before, 2160));
  CHECK(warp_after_span(SpanOrder::After, 720));
  // D5 : dès ×3, le warp passe après SPAN quelle que soit la sortie ; jamais en Avant.
  CHECK(warp_after_span(SpanOrder::Auto, 1080, 3));
  CHECK_FALSE(warp_after_span(SpanOrder::Auto, 1080, 2));
  CHECK_FALSE(warp_after_span(SpanOrder::Before, 1080, 6));
  for (const auto order : {SpanOrder::Auto, SpanOrder::Before, SpanOrder::After}) {
    PlaySettings s;
    s.span_order = order;
    CHECK(parse_settings(format_settings(s))->span_order == order);
  }
  // warp_after_span = false, le défaut d'avant écrit partout, laisse Auto ; vrai donne Après.
  CHECK(parse_settings("[interpolation]\nwarp_after_span = false\n")->span_order == SpanOrder::Auto);
  CHECK(parse_settings("[interpolation]\nwarp_after_span = true\n")->span_order == SpanOrder::After);
  CHECK(parse_settings("[interpolation]\nwarp_after_span = true\nspan_order = \"before\"\n")->span_order ==
        SpanOrder::Before);
  CHECK_FALSE(parse_settings("[interpolation]\nspan_order = \"milieu\"\n"));
}

TEST_CASE("Grade : plus de préréglages, les clés retirées sont ignorées", "[app][settings]") {
  const PlaySettings defaults;
  CHECK(defaults.grade);
  CHECK(defaults.grade_values.shadows == 0.50);
  // preset, gamut_p3, vibrance, adaptive : des clés d'avant, lues sans effet.
  const auto old = parse_settings(
      "[grade]\npreset = \"naturel\"\nshadows = 0.7\ngamut_p3 = true\nvibrance = 0.4\nadaptive = true\n");
  REQUIRE(old);
  CHECK(old->grade_values.shadows == 0.7);  // le préréglage ne réécrit plus les valeurs
  CHECK(*parse_settings(format_settings(*old)) == *old);
  CHECK(parse_settings("[grade]\npreset = \"criard\"\n"));
  CHECK_FALSE(parse_settings("[grade]\noled_black = \"peut-etre\"\n"));
  CHECK_FALSE(parse_settings("[grade]\noled_black_steps = 20.0\n"));
  CHECK(defaults.oled_black == OledBlack::Auto);
  CHECK(parse_settings("[grade]\npreset = \"perso\"\nshadows = 2.0\n").error().detail.find("shadows") !=
        std::string::npos);
}

TEST_CASE("Une clé absente garde son défaut, une clé inconnue est ignorée", "[app][settings]") {
  const auto s = parse_settings("[rtx_hdr]\nenabled = true\nfutur = 3\n[autre]\nx = 1\n");
  REQUIRE(s);
  CHECK_FALSE(s->hdr_native);
  CHECK(s->true_hdr_peak == 0);
  CHECK(s->rife_factor == kDefaultRifeFactor);
  CHECK(s->downscale == Downscale::Hermite);
  const auto empty = parse_settings("");
  REQUIRE(empty);
  CHECK(*empty == PlaySettings{});
  // Les clés retirées en 0.10 (P5.5) se relisent sans refus et ne se réécrivent plus.
  const auto old = parse_settings("[span_prepare]\nbackground = true\n[default_player]\ndismissed = true\n");
  REQUIRE(old);
  CHECK(*old == *parse_settings("[autre]\nx = 1\n"));
  CHECK(format_settings(*old).find("dismissed") == std::string::npos);
}

TEST_CASE("Hors plage, type faux ou TOML invalide : refus nommé", "[app][settings]") {
  const auto out = parse_settings("[audio]\ndelay_ms = 99999\n");
  REQUIRE_FALSE(out);
  CHECK(out.error().code == RefusalCode::FieldOutOfRange);
  CHECK(out.error().detail.find("audio.delay_ms") != std::string::npos);
  const auto peak = parse_settings("[rtx_hdr]\npeak = 300\n");
  REQUIRE_FALSE(peak);
  CHECK(peak.error().detail.find("hdr.peak") != std::string::npos);
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
  // L'ancien format (deux listes) devient deux préférences.
  const auto languages =
      parse_settings("[audio]\nlanguages = [\"eng\", \"jpn\"]\n[subtitles]\nlanguages = [\"fre\"]\n");
  REQUIRE(languages);
  CHECK(languages->language_presets == std::vector<ajp::media::LanguagePreset>{{"eng", "fre"}, {"jpn", ""}});
  const auto written =
      parse_settings("[languages]\naudio_1 = \"jpn\"\nsubtitles_1 = \"fre\"\naudio_2 = \"fre\"\n");
  REQUIRE(written);
  CHECK(written->language_presets == std::vector<ajp::media::LanguagePreset>{{"jpn", "fre"}, {"fre", ""}});
  CHECK(parse_settings("[languages]\naudio_1 = 3\n").error().code == RefusalCode::FieldType);
  CHECK(parse_settings("[languages]\naudio_1 = \"fr/../x\"\n").error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse_settings("[subtitles]\nlanguages = \"fre\"\n").error().code == RefusalCode::FieldType);
  const auto broken = parse_settings("[image\n");
  REQUIRE_FALSE(broken);
  CHECK(broken.error().code == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Les options données remplacent play.toml pour cette lecture", "[app][settings]") {
  PlaySettings saved;
  saved.rife_factor = 2;
  saved.work_height = 540;
  const std::array<std::string_view, 3> args{"film.mkv", "--rife=4", "--truehdr-peak=900"};
  const auto options = parse_play_options(args);
  REQUIRE(options);
  const PlaySettings s = settings_for(saved, *options);
  CHECK(s.rife_factor == 4);
  CHECK(s.true_hdr_peak == 900);
  CHECK(s.work_height == 540);  // non donnée : celle de play.toml
  const PlayOptions applied = options_for(*options, s);
  CHECK(applied.work_height == 540);
  CHECK(applied.rife_factor == 4);
}

TEST_CASE("Une option d'essai n'entre pas dans play.toml, un changement du menu oui", "[app][settings]") {
  PlaySettings file;
  file.rife_factor = 3;
  file.true_hdr_peak = 0;
  const std::array<std::string_view, 3> args{"film.mkv", "--rife=4", "--truehdr-peak=900"};
  const auto options = parse_play_options(args);
  REQUIRE(options);
  const PlaySettings started = settings_for(file, *options);
  PlaySettings running = started;
  running.volume = 0.4F;         // changé au menu
  running.true_hdr_peak = 1000;  // donné en option puis changé au menu
  const PlaySettings saved = settings_to_save(running, file, started, *options);
  CHECK(saved.rife_factor == 3);  // --rife=4 n'était que pour cette lecture
  CHECK(saved.true_hdr_peak == 1000);
  CHECK(saved.volume == 0.4F);
}

TEST_CASE(
    "Options de moteur : --rife-model et --motion, le menu montre le moteur chargé, play.toml n'en garde "
    "rien (A18-17, A18-09)",
    "[app][settings]") {
  PlaySettings file;
  file.motion_engine = MotionEngine::DistilDrba;
  file.rife_model = "rife_v4.25.onnx";
  const std::array<std::string_view, 2> rife{"film.mkv", "--rife-model=/opt/ajp/models/rife_v4.22_lite.onnx"};
  const auto options = parse_play_options(rife);
  REQUIRE(options);
  const PlaySettings started = settings_for(file, *options);
  CHECK(started.rife_model == "rife_v4.22_lite.onnx");
  CHECK(started.motion_engine == MotionEngine::Rife);
  const PlaySettings saved = settings_to_save(started, file, started, *options);
  CHECK(saved.rife_model == "rife_v4.25.onnx");
  CHECK(saved.motion_engine == MotionEngine::DistilDrba);

  const std::array<std::string_view, 2> drba{"film.mkv", "--motion=drba"};
  const auto chosen = parse_play_options(drba);
  REQUIRE(chosen);
  PlaySettings rife_file;
  CHECK(settings_for(rife_file, *chosen).motion_engine == MotionEngine::DistilDrba);
  CHECK(
      settings_to_save(settings_for(rife_file, *chosen), rife_file, settings_for(rife_file, *chosen), *chosen)
          .motion_engine == MotionEngine::Rife);
  const std::array<std::string_view, 2> wrong{"film.mkv", "--motion=gmfss"};
  CHECK_FALSE(parse_play_options(wrong));
}

TEST_CASE("Ce qu'un changement demande : direct ou réouverture, dit avant d'appliquer", "[app][settings]") {
  const PlaySettings running;
  CHECK(change_between(running, running, true).cost == ChangeCost::None);
  PlaySettings live = running;
  live.true_hdr_peak = 900;
  live.downscale = Downscale::Mitchell;
  live.volume = 0.5F;
  live.stats.all = true;
  live.grade_values = render::GradeSettings{};
  CHECK(change_between(running, live, true).cost == ChangeCost::Live);
  CHECK(live_part(running, live, true) == live);
  PlaySettings height = running;
  height.work_height = 1080;
  const Change reopen = change_between(running, height, true);
  CHECK(reopen.cost == ChangeCost::Reopen);
  CHECK(reopen.reason.find("SPAN") != std::string::npos);
  PlaySettings off;
  off.rife_factor = 0;
  PlaySettings on = off;
  on.rife_factor = 3;
  CHECK(change_between(off, on, true).cost == ChangeCost::Live);     // moteur RIFE chargé
  CHECK(change_between(off, on, false).cost == ChangeCost::Reopen);  // à charger
  CHECK(change_between(on, off, false).cost == ChangeCost::Live);
  // Le moteur du mouvement se recharge s'il interpole ; sans interpolation, il attend.
  PlaySettings flow = on;
  flow.motion_engine = MotionEngine::DistilDrba;
  CHECK(change_between(on, flow, true).reason.find("interpolation engine") != std::string::npos);
  PlaySettings flow_off = off;
  flow_off.motion_engine = MotionEngine::DistilDrba;
  CHECK(change_between(off, flow_off, true).cost == ChangeCost::Live);
  CHECK(live_part(off, flow_off, true).motion_engine == MotionEngine::DistilDrba);
  CHECK(live_part(on, flow, true).motion_engine == MotionEngine::Rife);
  PlaySettings heavy = on;
  heavy.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(change_between(on, heavy, true).reason.find("RIFE model") != std::string::npos);
  // Avec l'interpolation d'Apple, le modèle RIFE ne sert pas : le changer ne recharge rien.
  PlaySettings apple = on;
  apple.motion_engine = MotionEngine::Apple;
  PlaySettings apple_heavy = apple;
  apple_heavy.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(change_between(apple, apple_heavy, true).cost == ChangeCost::Live);
  CHECK(live_part(apple, apple_heavy, true).rife_model == "rife_v4.26_heavy.onnx");
  // L'ordre du warp et de SPAN recharge la chaîne qui interpole, et attend sans interpolation.
  PlaySettings before = on;
  before.span_order = SpanOrder::Before;
  CHECK(change_between(on, before, true).reason.find("before SPAN") != std::string::npos);
  PlaySettings before_off = off;
  before_off.span_order = SpanOrder::Before;
  CHECK(change_between(off, before_off, true).cost == ChangeCost::Live);
  CHECK(live_part(off, before_off, true).span_order == SpanOrder::Before);
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
  wanted.true_hdr_peak = 900;
  wanted.downscale = Downscale::CatmullRom;
  wanted.stats.frames = true;
  wanted.rife_factor = 2;
  const PlaySettings live = live_part(running, wanted, true);
  CHECK(live.work_height == running.work_height);
  CHECK(live.true_hdr_peak == 900);
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

TEST_CASE("Panneau : Auto choisi sans RIFE chargé reste choisi", "[app][settings]") {
  PlaySettings off;
  off.rife_factor = 0;
  off.rife_auto = false;
  PlaySettings draft = off;  // « Auto » choisi dans le menu
  draft.rife_auto = true;
  draft.rife_factor = 8;
  const PlaySettings running = live_part(off, draft, false);  // rife_auto en direct, ×8 attend
  CHECK(running.rife_factor == 0);
  const PlaySettings followed = follow_running(draft, off, running, false);
  CHECK(followed.rife_auto);
  CHECK(followed.rife_factor == 8);
  PlaySettings tab = running;  // Tab ailleurs : ×2 en direct, le brouillon le suit
  tab.rife_factor = 2;
  CHECK(follow_running(followed, running, tab, true).rife_factor == 2);
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
  bad.true_hdr_peak = 5;
  CHECK_FALSE(save_settings(path, bad));
  CHECK(load_settings(path)->rife_factor == 2);  // l'ancien fichier reste
  std::filesystem::remove_all(dir);
}

TEST_CASE("SPAN, RIFE et DistilDRBA : le cran de précision recharge la chaîne", "[app][settings]") {
  PlaySettings running;
  running.rife_factor = 2;
  PlaySettings wanted = running;
  wanted.span_precision = PrecisionTier::Performance;
  const Change span = change_between(running, wanted, true);
  CHECK(span.cost == ChangeCost::Reopen);
  CHECK(span.reason.find("SPAN") != std::string::npos);
  CHECK(span.reason.find("FP8") != std::string::npos);
  // Sans SPAN, rien à recharger.
  running.upscale = false;
  wanted.upscale = false;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
  // RIFE.
  running = PlaySettings{};
  running.rife_factor = 2;
  wanted = running;
  wanted.motion_precision = PrecisionTier::Quality;
  const Change motion = change_between(running, wanted, true);
  CHECK(motion.cost == ChangeCost::Reopen);
  CHECK(motion.reason.find("FP32") != std::string::npos);
  // Sans interpolation, rien à recharger.
  running.rife_factor = 0;
  wanted.rife_factor = 0;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
}

TEST_CASE("crans de précision : play.toml ; les clés de GMFSS d'avant se lisent, ignorées",
          "[app][settings]") {
  PlaySettings s;
  s.span_precision = PrecisionTier::Performance;
  s.motion_precision = PrecisionTier::Quality;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->span_precision == PrecisionTier::Performance);
  CHECK(back->motion_precision == PrecisionTier::Quality);
  CHECK(format_settings(s).find("gmfss") == std::string::npos);
  const auto legacy = parse_settings(
      "version = 1\n[interpolation]\ngmfss_quality = true\ngmfss_fused = true\ngmfss_precision = "
      "\"quality\"\n");
  REQUIRE(legacy);
  CHECK(legacy->motion_precision == PrecisionTier::Balanced);
  // Un cran inconnu est refusé, sans changer les autres réglages.
  CHECK_FALSE(parse_settings("version = 1\n[image]\nspan_precision = \"fp4\"\n"));
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\nmotion_precision = 3\n"));
}

TEST_CASE("TierSupport : Équilibre toujours, les autres selon les graphes", "[app][settings]") {
  const TierSupport none;
  CHECK(none.has(PrecisionTier::Balanced));
  CHECK_FALSE(none.has(PrecisionTier::Performance));
  CHECK(none.effective(PrecisionTier::Quality) == PrecisionTier::Balanced);
  const TierSupport fp8{.performance = true};
  CHECK(fp8.effective(PrecisionTier::Performance) == PrecisionTier::Performance);
  CHECK(fp8.effective(PrecisionTier::Quality) == PrecisionTier::Balanced);
  CHECK(precision_text(PrecisionTier::Performance) == "FP8");
  CHECK(precision_text(PrecisionTier::Balanced) == "FP16");
  CHECK(precision_text(PrecisionTier::Quality) == "FP32");
  CHECK(label(PrecisionTier::Balanced) == "Balance");
}

TEST_CASE("mises à jour : mode et fréquence lus et écrits par leur nom (P5.29)", "[settings]") {
  // Un play.toml d'avant la simple notification : « auto » se lit « notify », le canal est ignoré.
  const auto read =
      parse_settings("[updates]\nmode = \"auto\"\nchannel = \"preview\"\nfrequency = \"weekly\"\n");
  REQUIRE(read);
  CHECK(read->update_mode == UpdateMode::Notify);
  CHECK(read->update_frequency == UpdateFrequency::Weekly);
  const std::string written = format_settings(*read);
  CHECK(written.find("channel") == std::string::npos);
  CHECK(parse_settings(written)->update_frequency == UpdateFrequency::Weekly);
  CHECK(parse_settings("[updates]\nmode = \"never\"\n")->update_mode == UpdateMode::Never);
  CHECK_FALSE(parse_settings("[updates]\nfrequency = \"hourly\"\n"));
}

}  // namespace
}  // namespace ajp::app::play

TEST_CASE("repères des listes et résolution en français courant", "[settings]") {
  using namespace ajp::app::play;
  CHECK(resolution_text(720) == "720p");
  CHECK(badge_text(Badge::Finest) == "Finest");
  CHECK(badge_of(Downscale::Hermite) == Badge::Recommended);
  CHECK_FALSE(badge_of(Downscale::Mitchell));
  CHECK_FALSE(badge_of(Downscale::Area));
  CHECK(label(MotionEngine::DistilDrba) == "DistilDRBA");
}

TEST_CASE("audio : l'appareil fait l'aller-retour, les anciennes clés du son sont ignorées", "[settings]") {
  using namespace ajp::app::play;
  PlaySettings s;
  s.audio_device = "Casque USB";
  const auto read = parse_settings(format_settings(s));
  REQUIRE(read);
  CHECK(read->audio_device == "Casque USB");
  // Le traitement du son a été retiré : un play.toml d'avant se relit sans perdre le reste.
  const auto old = parse_settings(
      "version = 1\n[audio]\ndevice = \"A\"\nlistening = \"headphones\"\nprofile = \"night\"\n"
      "anime_sound = \"strong\"\nhrir = \"bundled:atmos\"\n");
  REQUIRE(old);
  CHECK(old->audio_device == "A");
  CHECK(change_between(
            PlaySettings{},
            [] {
              PlaySettings x;
              x.audio_device = "A";
              return x;
            }(),
            false)
            .cost == ChangeCost::Reopen);
}

TEST_CASE("Hermite : un filtre de réduction de plus, cubique B = C = 0", "[app][settings]") {
  using namespace ajp::app::play;
  CHECK(name(Downscale::Hermite) == "hermite");
  CHECK(label(Downscale::Hermite) == "Hermite");
  const ajp::gpu::Reduction reduction = reduction_of(Downscale::Hermite);
  CHECK(reduction.kind == ajp::gpu::ReduceKind::Cubic);
  CHECK(reduction.b == 0.0F);
  CHECK(reduction.c == 0.0F);
  PlaySettings s;
  s.downscale = Downscale::Hermite;
  CHECK(parse_settings(format_settings(s))->downscale == Downscale::Hermite);
}

TEST_CASE("play.toml garde le plafond du mode auto et refuse un plafond absurde",
          "[settings][auto][plafond]") {
  using namespace ajp::app::play;
  PlaySettings s;
  CHECK(s.auto_cap == 0);  // maximum : l'écran
  CHECK(parse_settings("version = 1\n")->auto_cap == 0);
  s.auto_cap = 120;
  const auto read = parse_settings(format_settings(s));
  REQUIRE(read);
  CHECK(read->auto_cap == 120);
  CHECK(validate(s).has_value());
  s.auto_cap = 5;
  CHECK_FALSE(validate(s).has_value());
  s.auto_cap = 100000;
  CHECK_FALSE(validate(s).has_value());
  // Un plafond posé dans le menu s'applique en direct, sans rouvrir la lecture.
  PlaySettings running;
  PlaySettings wanted;
  wanted.auto_cap = 60;
  CHECK(live_part(running, wanted, true).auto_cap == 60);
}

TEST_CASE("play.toml garde le mode d'Auto (Rapide par défaut) et refuse une valeur inconnue",
          "[settings][auto][libre]") {
  using namespace ajp::app::play;
  PlaySettings s;
  CHECK(s.auto_free);  // C6 : Rapide est l'Auto ; Stable reste au choix
  CHECK(parse_settings("version = 1\n")->auto_free);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\nauto_mode = \"recommande\"\n")->auto_free);
  const std::string text = format_settings(s);
  CHECK(text.find("auto_mode") != std::string::npos);
  const auto read = parse_settings(text);
  REQUIRE(read);
  CHECK(read->auto_free);
  CHECK(parse_settings(format_settings(PlaySettings{}))->auto_free);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\nauto_mode = \"fou\"\n").has_value());
  // Le mode se change en direct, sans rouvrir la lecture.
  PlaySettings running;
  running.auto_free = false;
  PlaySettings wanted;
  wanted.auto_free = true;
  CHECK(live_part(running, wanted, true).auto_free);
}

TEST_CASE("mode Rapide : instants par paire acceptés par chaque moteur", "[settings][auto][libre]") {
  using namespace ajp::app::play;
  for (const MotionEngine engine : {MotionEngine::Rife, MotionEngine::DistilDrba}) {
    CHECK(free_instants_limit(engine) >= 2);
    CHECK(free_instants_limit(engine) <= ajp::core::kMaxInstantsPerPair);  // la borne du ring
  }
  CHECK(free_instants_limit(MotionEngine::Apple) == 0);  // pas de tau arbitraire : Auto
}

TEST_CASE("une valeur fautive n'invalide que sa clé", "[settings][lenient]") {
  using namespace ajp::app::play;
  const std::string text =
      "version = 1\n[image]\nupscale = false\nwork_height = 99999\n[audio]\nvolume = "
      "0.5\n[sync]\nvideo_latency_ms = "
      "\"oups\"\n";
  CHECK_FALSE(parse_settings(text).has_value());  // le mode strict refuse tout le fichier
  std::vector<std::string> dropped;
  const auto lenient = parse_settings_lenient(text, &dropped);
  if (!lenient) UNSCOPED_INFO(lenient.error().detail);
  REQUIRE(lenient.has_value());
  CHECK_FALSE(lenient->upscale);  // les autres clés restent en vigueur
  CHECK(lenient->volume == 0.5F);
  CHECK(lenient->work_height == PlaySettings{}.work_height);  // la clé fautive retombe au défaut
  CHECK(lenient->video_latency_ms == PlaySettings{}.video_latency_ms);
  CHECK(dropped.size() == 2);
  CHECK_FALSE(parse_settings_lenient("[image\nbroken").has_value());  // un TOML illisible reste refusé
}

TEST_CASE("un refus qui ne nomme pas sa clé n'efface pas le reste, grade compris", "[settings][lenient]") {
  using namespace ajp::app::play;
  // « languages.audio » ne nomme pas audio_1 : avant, tout le fichier était mis de côté.
  const std::string text =
      "version = 1\n[grade]\nenabled = true\nshadows = 0.42\narea_rtx = 0.31\n[languages]\naudio_1 = \"pas "
      "une "
      "langue !\"\n";
  std::vector<std::string> dropped;
  const auto lenient = parse_settings_lenient(text, &dropped);
  if (!lenient) UNSCOPED_INFO(lenient.error().detail);
  REQUIRE(lenient.has_value());
  CHECK(lenient->grade_values.area_rtx == 0.31);
  CHECK(lenient->grade_values.shadows == 0.42);
  CHECK(dropped.size() == 1);
}

TEST_CASE("play.toml : les clés de l'ancienne pile de LUT se lisent et ne sont plus écrites",
          "[app][settings]") {
  const auto old = ajp::app::play::parse_settings(
      "[grade]\nlook = '../x.cube'\nlook_mix = 1.5\nlooks = ['a/b.cube']\nlooks_mix = [-0.1]\nlook_contrast "
      "= 20\n");
  REQUIRE(old.has_value());
  CHECK(old->look_contrast == 20);
  const std::string written = ajp::app::play::format_settings(ajp::app::play::PlaySettings{});
  CHECK(written.find("look_mix") == std::string::npos);
  CHECK(written.find("looks") == std::string::npos);
}

TEST_CASE("play.toml : le limiteur de flashs, activé par défaut et dans un fichier d'avant, se coupe",
          "[app][settings]") {
  CHECK(ajp::app::play::PlaySettings{}.flash_guard);
  const auto old = ajp::app::play::parse_settings("[grade]\nlook_contrast = 20\n");
  REQUIRE(old.has_value());
  CHECK(old->flash_guard);
  const auto off = ajp::app::play::parse_settings("[grade]\nflash_guard = false\n");
  REQUIRE(off.has_value());
  CHECK_FALSE(off->flash_guard);
}

TEST_CASE("play.toml : la peau des visages, éteinte par défaut et dans un fichier d'avant, s'allume",
          "[app][settings]") {
  CHECK_FALSE(ajp::app::play::PlaySettings{}.face_skin);
  const auto old = ajp::app::play::parse_settings("[grade]\nflash_guard = true\n");
  REQUIRE(old.has_value());
  CHECK_FALSE(old->face_skin);
  const auto on = ajp::app::play::parse_settings("[grade]\nface_skin = true\n");
  REQUIRE(on.has_value());
  CHECK(on->face_skin);
}

TEST_CASE("play.toml : le blanc des surfaces, 100 % par défaut et dans un fichier d'avant, borné",
          "[app][settings]") {
  CHECK(ajp::app::play::PlaySettings{}.surface_white == 100);
  const auto old = ajp::app::play::parse_settings("[grade]\nlook_contrast = 20\n");
  REQUIRE(old.has_value());
  CHECK(old->surface_white == 100);
  const auto lower = ajp::app::play::parse_settings("[grade]\nsurface_white = 70\n");
  REQUIRE(lower.has_value());
  CHECK(lower->surface_white == 70);
  ajp::app::play::PlaySettings s;
  s.surface_white = 70;
  const auto again = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(again.has_value());
  CHECK(again->surface_white == 70);
  s.surface_white = 30;
  CHECK_FALSE(ajp::app::play::validate(s).has_value());
}

TEST_CASE("play.toml : une version plus récente garde ses clés à la réécriture (P2.22)", "[app][settings]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-future.toml";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "version = 2\nfuture_option = 7\n[image]\nupscale = false\nfuture_filter = \"x\"\n[nouveau]\na = "
           "1\n";
  }
  auto loaded = ajp::app::play::load_settings(path);
  REQUIRE(loaded);
  CHECK_FALSE(loaded->upscale);
  loaded->upscale = true;
  REQUIRE(ajp::app::play::save_settings(path, *loaded));
  std::string text;
  {
    std::ifstream in(path, std::ios::binary);  // refermé avant remove : Windows refuse un fichier ouvert
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  CHECK(text.find("future_option = 7") != std::string::npos);
  CHECK(text.find("future_filter = 'x'") != std::string::npos);
  CHECK(text.find("[nouveau]") != std::string::npos);
  auto reread = ajp::app::play::load_settings(path);
  REQUIRE(reread);
  CHECK(*reread == *loaded);
  std::filesystem::remove(path);
}

TEST_CASE("Délai audio mémorisé par appareil, relu de play.toml (P2.7)", "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.audio_delay_ms = 20;  // la sortie du système
  s.audio_device = "Casque Bluetooth";
  ajp::app::play::switch_device_delay(s, "");
  CHECK(s.audio_delay_ms == 0);
  s.audio_delay_ms = 240;  // réglé pour le casque
  s.audio_device.clear();
  ajp::app::play::switch_device_delay(s, "Casque Bluetooth");
  CHECK(s.audio_delay_ms == 20);
  // B23-06 : suivre le système et nommer la même carte, c'est le même appareil : le délai reste.
  ajp::app::play::PlaySettings tv;
  tv.audio_delay_ms = 120;
  tv.audio_device = "HDMI";
  ajp::app::play::switch_device_delay(tv, "", "HDMI");
  CHECK(tv.audio_delay_ms == 120);
  tv.audio_device.clear();
  ajp::app::play::switch_device_delay(tv, "HDMI", "HDMI");
  CHECK(tv.audio_delay_ms == 120);
  tv.audio_device = "Casque";  // une autre carte : le délai de la TV est gardé pour elle
  ajp::app::play::switch_device_delay(tv, "", "HDMI");
  CHECK(tv.audio_delay_ms == 0);
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-delays.toml";
  REQUIRE(ajp::app::play::save_settings(path, s));
  auto reread = ajp::app::play::load_settings(path);
  REQUIRE(reread);
  CHECK(reread->audio_device_delays_ms.at("Casque Bluetooth") == 240);
  CHECK(*reread == s);
  std::filesystem::remove(path);
}

TEST_CASE("play.toml : deux lancements, chacun ne réécrit que les clés qu'il a changées (P2.14)",
          "[app][settings]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-two.toml";
  std::filesystem::remove(path);
  REQUIRE(ajp::app::play::save_settings(path, ajp::app::play::PlaySettings{}));
  auto ours = ajp::app::play::load_settings(path);
  REQUIRE(ours);
  // L'autre lancement réécrit le fichier : le volume change.
  ajp::app::play::PlaySettings theirs = *ours;
  theirs.volume = 0.25F;
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << ajp::app::play::format_settings(theirs);
  }
  ours->audio_delay_ms = 120;
  REQUIRE(ajp::app::play::save_settings(path, *ours));
  const auto merged = ajp::app::play::load_settings(path);
  REQUIRE(merged);
  CHECK(merged->audio_delay_ms == 120);
  CHECK(merged->volume == Catch::Approx(0.25F));
  std::filesystem::remove(path);
}
