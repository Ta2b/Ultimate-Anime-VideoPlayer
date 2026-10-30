// Réglages du menu des paramètres : plages, play.toml aller-retour, ligne de commande, et ce
// qu'un changement demande (direct ou réouverture de la chaîne).
#include "app/play/settings.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/free_engines.h"
#include "app/play/live_settings.h"

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
  s.display_sync = false;
  s.video_latency_ms = 40;
  s.volume = 0.85F;
  s.subtitle_scale = 1.25F;
  s.stats.frames = true;
  s.stats.memory = true;
  s.grade_preset = render::GradePreset::Custom;
  s.grade_values.shadows = 0.55;
  s.grade_values.guard = 0.25;
  s.grade_values.gamut_p3 = true;
  s.oled_black = OledBlack::On;
  s.final_scaler = FinalScaler::EwaLanczosSharp;
  s.deband = true;
  s.oled_black_steps = 3.5;
  s.oled_care = false;
  s.steady_cadence = true;
  s.smooth_motion = true;
  s.motion_engine = MotionEngine::DistilDrba;
  s.rife_model = "rife_v4.22_lite.onnx";
  s.keep_poses = true;
  s.gmfss_precision = PrecisionTier::Quality;
  s.span_precision = PrecisionTier::Performance;
  s.motion_precision = PrecisionTier::Quality;
  s.rife_auto = true;
  s.span_order = SpanOrder::Before;
  s.grade_values.expand = 0.3;
  s.language_presets = {{"eng", ""}, {"", "fre"}};
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(*back == s);
  s.motion_engine = MotionEngine::Gmfss;
  CHECK(parse_settings(format_settings(s))->motion_engine == MotionEngine::Gmfss);
  s.motion_engine = MotionEngine::GmfssUnion;
  CHECK(parse_settings(format_settings(s))->motion_engine == MotionEngine::GmfssUnion);
  CHECK(is_gmfss(MotionEngine::GmfssUnion));
  CHECK_FALSE(is_gmfss(MotionEngine::Rife));
  // L'interpolation d'Apple se garde partout ; elle n'est proposée que sur un Mac.
  s.motion_engine = MotionEngine::Apple;
  CHECK(parse_settings(format_settings(s))->motion_engine == MotionEngine::Apple);
#if defined(__APPLE__)
  CHECK(offered(MotionEngine::Apple));
#else
  CHECK_FALSE(offered(MotionEngine::Apple));
#endif
  CHECK(offered(MotionEngine::Rife));
}

TEST_CASE("Mode HDR : moteur RTX HDR, HDR adaptatif ou SDR, et les fichiers d'avant", "[app][settings]") {
  for (const std::string_view engine : {"rtx", "adaptive", "sdr"}) {
    const auto s = parse_settings(std::format("[hdr]\nengine = \"{}\"\ncontrast = 150\n", engine));
    REQUIRE(s);
    CHECK(hdr_engine_key(*s) == engine);
    CHECK(s->true_hdr_contrast == 150);
    CHECK(*parse_settings(format_settings(*s)) == *s);
  }
  CHECK(format_settings(PlaySettings{}).find("[rtx_hdr]") == std::string::npos);
  CHECK_FALSE(parse_settings("[hdr]\nengine = \"maxine\"\n"));
  // Un play.toml d'avant : section rtx_hdr, enabled et native.
  const auto rtx = parse_settings("[rtx_hdr]\nenabled = true\ncontrast = 200\nmiddle_gray = 51\n");
  REQUIRE(rtx);
  CHECK(hdr_engine_key(*rtx) == "rtx");
  CHECK(rtx->true_hdr_contrast == 200);
  CHECK(rtx->true_hdr_gray == 51);
  const auto native = parse_settings("[rtx_hdr]\nenabled = false\nnative = true\n");
  REQUIRE(native);
  CHECK(hdr_engine_key(*native) == "adaptive");
  // La nouvelle section l'emporte sur l'ancienne.
  const auto both = parse_settings("[rtx_hdr]\nenabled = true\n[hdr]\nengine = \"adaptive\"\n");
  REQUIRE(both);
  CHECK(hdr_engine_key(*both) == "adaptive");
}

TEST_CASE("Warp et SPAN : Auto passe après SPAN au-dessus de 1080p ; les fichiers d'avant",
          "[app][settings]") {
  CHECK(PlaySettings{}.span_order == SpanOrder::Auto);
  CHECK(PlaySettings{}.gmfss_precision ==
        PrecisionTier::Balanced);  // GMFSS reste en FP16 : son FP8 dégrade les gros mouvements
  CHECK(PlaySettings{}.span_precision == PrecisionTier::Balanced);
  CHECK(PlaySettings{}.motion_precision == PrecisionTier::Balanced);
  CHECK_FALSE(warp_after_span(SpanOrder::Auto, 1080));
  CHECK(warp_after_span(SpanOrder::Auto, 1440));
  CHECK(warp_after_span(SpanOrder::Auto, 2160));
  CHECK_FALSE(warp_after_span(SpanOrder::Before, 2160));
  CHECK(warp_after_span(SpanOrder::After, 720));
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
  const auto out = parse_settings("[rtx_hdr]\ncontrast = 450\n");
  REQUIRE_FALSE(out);
  CHECK(out.error().code == RefusalCode::FieldOutOfRange);
  CHECK(out.error().detail.find("hdr.contrast") != std::string::npos);
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
  CHECK(written->language_presets == PlaySettings{}.language_presets);
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
  // Avec GMFSS, le modèle RIFE ne sert pas : le changer ne recharge rien.
  PlaySettings gmfss = on;
  gmfss.motion_engine = MotionEngine::Gmfss;
  PlaySettings gmfss_heavy = gmfss;
  gmfss_heavy.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(change_between(gmfss, gmfss_heavy, true).cost == ChangeCost::Live);
  CHECK(live_part(gmfss, gmfss_heavy, true).rife_model == "rife_v4.26_heavy.onnx");
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

TEST_CASE("GMFSS : changer de cran de précision recharge la chaîne, seulement pour GMFSS",
          "[app][settings]") {
  PlaySettings running;
  running.rife_factor = 2;
  running.motion_engine = MotionEngine::Gmfss;
  running.gmfss_precision = PrecisionTier::Balanced;
  PlaySettings wanted = running;
  wanted.gmfss_precision = PrecisionTier::Quality;
  const Change change = change_between(running, wanted, true);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("Quality") != std::string::npos);
  CHECK(change.reason.find("FP32") != std::string::npos);
  // Le nouveau cran Performance (FP8), et le sens inverse.
  wanted.gmfss_precision = PrecisionTier::Performance;
  const Change fast = change_between(running, wanted, true);
  CHECK(fast.cost == ChangeCost::Reopen);
  CHECK(fast.reason.find("FP8") != std::string::npos);
  running.gmfss_precision = PrecisionTier::Quality;
  wanted.gmfss_precision = PrecisionTier::Balanced;
  CHECK(change_between(running, wanted, true).cost == ChangeCost::Reopen);
  // Un autre moteur du mouvement ne recharge rien pour ce réglage.
  running.motion_engine = MotionEngine::Rife;
  wanted.motion_engine = MotionEngine::Rife;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
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

TEST_CASE("crans de précision : play.toml, y compris l'ancien gmfss_quality", "[app][settings]") {
  PlaySettings s;
  s.span_precision = PrecisionTier::Performance;
  s.motion_precision = PrecisionTier::Quality;
  s.gmfss_precision = PrecisionTier::Performance;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->span_precision == PrecisionTier::Performance);
  CHECK(back->motion_precision == PrecisionTier::Quality);
  CHECK(back->gmfss_precision == PrecisionTier::Performance);
  // Les fichiers d'avant : gmfss_quality vrai donne Qualité, faux laisse l'Équilibre (l'ancien Performance).
  const auto quality = parse_settings("version = 1\n[interpolation]\ngmfss_quality = true\n");
  REQUIRE(quality);
  CHECK(quality->gmfss_precision == PrecisionTier::Quality);
  const auto legacy_off = parse_settings("version = 1\n[interpolation]\ngmfss_quality = false\n");
  REQUIRE(legacy_off);
  CHECK(legacy_off->gmfss_precision == PrecisionTier::Balanced);
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

}  // namespace
}  // namespace ajp::app::play

TEST_CASE("repères des listes et résolution en français courant", "[settings]") {
  using namespace ajp::app::play;
  CHECK(resolution_text(720) == "720p");
  CHECK(badge_text(Badge::Finest) == "Finest");
  CHECK(badge_of(MotionEngine::Gmfss) == Badge::Finest);
  CHECK(badge_of(MotionEngine::GmfssUnion) == Badge::Heaviest);
  CHECK(badge_of(MotionEngine::Rife) == Badge::Recommended);
  CHECK(badge_of(Downscale::Mitchell) == Badge::Recommended);
  CHECK_FALSE(badge_of(Downscale::Area));
  CHECK(label(MotionEngine::Gmfss) == "GMFSS Fortuna");
}

TEST_CASE("audio : appareil, écoute et profils par appareil font l'aller-retour", "[settings]") {
  using namespace ajp::app::play;
  PlaySettings s;
  s.audio_device = "Casque USB";
  s.listening = "speakers";
  s.audio_faithful = true;
  s.device_listening = {{"Casque USB", "headphones"}, {"Barre de son", "cinema"}};
  const auto read = parse_settings(format_settings(s));
  REQUIRE(read);
  CHECK(read->audio_device == "Casque USB");
  CHECK(read->audio_faithful);
  auto sorted = [](auto list) {
    std::ranges::sort(list);
    return list;
  };
  CHECK(sorted(read->device_listening) == sorted(s.device_listening));  // la table TOML trie ses clés
  // Le profil propre prime, un appareil inconnu reprend le global.
  CHECK(effective_listening(*read, "Casque USB").listen == 0);
  CHECK(effective_listening(*read, "Barre de son").listen == 2);
  CHECK(effective_listening(*read, "Autre").listen == 1);
  CHECK(effective_listening(*read, "Autre").faithful);
  CHECK_FALSE(parse_settings("[audio]\nlistening = \"stade\"\n"));
  CHECK(change_between(
            PlaySettings{},
            [] {
              PlaySettings x;
              x.audio_device = "A";
              return x;
            }(),
            false)
            .cost == ChangeCost::Reopen);
  CHECK(change_between(
            PlaySettings{},
            [] {
              PlaySettings x;
              x.listening = "headphones";
              return x;
            }(),
            false)
            .cost == ChangeCost::Live);
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

TEST_CASE("play.toml garde le profil audio et ses réglages personnalisés", "[settings][audio]") {
  ajp::app::play::PlaySettings s;
  s.audio_profile = "custom";
  s.audio_custom.crossfeed_gain = 0.45F;
  s.audio_custom.center_lift = 1.7F;
  const auto read = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(read);
  CHECK(read->audio_profile == "custom");
  CHECK(read->audio_custom.crossfeed_gain == Catch::Approx(0.45F));
  CHECK(read->audio_custom.center_lift == Catch::Approx(1.7F));
  CHECK(ajp::app::play::parse_settings("version = 1\n")->audio_profile == "natural");
  CHECK_FALSE(ajp::app::play::parse_settings("version = 1\n[audio]\nprofile = \"zzz\"\n"));
  // Un profil qui ne va pas à l'écoute retombe sur Naturel.
  s.audio_profile = "night";
  CHECK(ajp::app::play::resolved_tuning(s, 0) == ajp::audio::MixTuning{});
  CHECK(ajp::app::play::resolved_tuning(s, 1) == ajp::audio::find_preset("night")->tuning);
}

TEST_CASE("un appareil garde son profil de virtualisation, les autres le global", "[settings][audio]") {
  ajp::app::play::PlaySettings s;
  s.listening = "headphones";
  s.audio_profile = "soft";
  s.device_listening = {{"Casque BT", "headphones"}};
  s.device_profile = {{"Casque BT", "room"}};
  CHECK(ajp::app::play::effective_listening(s, "Casque BT").tuning ==
        ajp::audio::find_preset("room")->tuning);
  CHECK(ajp::app::play::effective_listening(s, "Autre").tuning == ajp::audio::find_preset("soft")->tuning);
  const auto read = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(read);
  CHECK(read->device_profile == s.device_profile);
  CHECK_FALSE(ajp::app::play::parse_settings("version = 1\n[audio.device_profiles]\nX = \"zzz\"\n"));
  // Les nouveaux réglages (égaliseur, nuit) font l'aller-retour de play.toml.
  s.audio_profile = "custom";
  s.audio_custom.eq_voice_db = 4.5F;
  s.audio_custom.night = 0.6F;
  s.audio_custom.room = 0.25F;
  const auto again = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(again);
  const auto got = ajp::audio::to_array(again->audio_custom);
  const auto want = ajp::audio::to_array(s.audio_custom);
  for (std::size_t i = 0; i < got.size(); ++i) {
    INFO("champ " << i);
    CHECK(got[i] == want[i]);
  }
}

TEST_CASE("les réponses impulsionnelles : chemin global et propre à un appareil", "[settings][audio]") {
  ajp::app::play::PlaySettings s;
  CHECK(s.hrir_path.empty());  // défaut : aucune
  CHECK(ajp::app::play::effective_listening(s, "X").hrir_path.empty());
  s.hrir_path = "/home/moi/hesuvi/a.wav";
  s.device_hrir = {{"Casque BT", "/home/moi/hesuvi/b.wav"}};
  CHECK(ajp::app::play::effective_listening(s, "Casque BT").hrir_path == "/home/moi/hesuvi/b.wav");
  CHECK(ajp::app::play::effective_listening(s, "Autre").hrir_path == "/home/moi/hesuvi/a.wav");
  const auto read = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(read);
  CHECK(read->hrir_path == s.hrir_path);
  CHECK(read->device_hrir == s.device_hrir);
  CHECK_FALSE(ajp::app::play::parse_settings("version = 1\n[audio.device_hrir]\nX = 3\n"));
  // Un chemin vide : pas de fichier, pas de refus.
  CHECK(*ajp::app::play::load_listening_hrir("") == nullptr);
  CHECK_FALSE(ajp::app::play::load_listening_hrir("/nonexistent/ajp/a.wav"));
}

TEST_CASE("play.toml garde le Son anime : niveau, réglages personnalisés, niveau par appareil",
          "[settings][audio][anime]") {
  using namespace ajp::app::play;
  PlaySettings s;
  // Défaut : « Anime » actif.
  CHECK(s.anime_sound == "anime");
  CHECK(effective_listening(s, "X").anime == ajp::audio::find_anime_level("anime")->tuning);
  CHECK(parse_settings("version = 1\n")->anime_sound == "anime");
  s.anime_sound = "custom";
  s.anime_custom.voice_db = 3.5F;
  s.anime_custom.target_lufs = -17.0F;
  s.anime_custom.loudness = false;
  s.anime_custom.voice_forward = 0.6F;
  s.device_anime = {{"Casque BT", "strong"}, {"Enceintes", "natural"}};
  const auto read = parse_settings(format_settings(s));
  REQUIRE(read);
  CHECK(read->anime_sound == "custom");
  CHECK(read->anime_custom.voice_db == Catch::Approx(3.5F));
  CHECK(read->anime_custom.target_lufs == Catch::Approx(-17.0F));
  CHECK_FALSE(read->anime_custom.loudness);
  CHECK(read->device_anime == s.device_anime);
  CHECK(effective_listening(s, "Casque BT").anime == ajp::audio::find_anime_level("strong")->tuning);
  CHECK_FALSE(effective_listening(s, "Enceintes").anime.enabled);
  CHECK(effective_listening(s, "Autre").anime.voice_db == Catch::Approx(3.5F));
  // Valeurs refusées en nommant la clé.
  CHECK_FALSE(parse_settings("version = 1\n[audio.anime]\nlevel = \"zzz\"\n"));
  CHECK_FALSE(parse_settings("version = 1\n[audio.anime.custom]\nvoice_db = 9.0\n"));
  CHECK_FALSE(parse_settings("version = 1\n[audio.anime.custom]\ntarget_lufs = -30.0\n"));
  CHECK_FALSE(parse_settings("version = 1\n[audio.device_anime]\nX = \"zzz\"\n"));
  // Un changement de niveau est un changement de l'écoute, appliqué en direct.
  PlaySettings other = PlaySettings{};
  other.anime_sound = "strong";
  CHECK_FALSE(effective_listening(other, "X") == effective_listening(PlaySettings{}, "X"));
  CHECK(live_part(PlaySettings{}, other, false).anime_sound == "strong");
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

TEST_CASE("play.toml garde le mode d'Auto (stable par défaut) et refuse une valeur inconnue",
          "[settings][auto][libre]") {
  using namespace ajp::app::play;
  PlaySettings s;
  CHECK_FALSE(s.auto_free);  // le défaut reste Stable
  CHECK_FALSE(parse_settings("version = 1\n")->auto_free);
  s.auto_free = true;
  const std::string text = format_settings(s);
  CHECK(text.find("auto_mode") != std::string::npos);
  const auto read = parse_settings(text);
  REQUIRE(read);
  CHECK(read->auto_free);
  CHECK_FALSE(parse_settings(format_settings(PlaySettings{}))->auto_free);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\nauto_mode = \"fou\"\n").has_value());
  // Le mode se change en direct, sans rouvrir la lecture.
  PlaySettings running;
  PlaySettings wanted;
  wanted.auto_free = true;
  CHECK(live_part(running, wanted, true).auto_free);
}

TEST_CASE("mode Rapide : instants par paire acceptés par chaque moteur", "[settings][auto][libre]") {
  using namespace ajp::app::play;
  for (const MotionEngine engine :
       {MotionEngine::Rife, MotionEngine::Gmfss, MotionEngine::GmfssUnion, MotionEngine::DistilDrba}) {
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

TEST_CASE("un play.toml de la rc.3 (400 %, vibrance 1,2) se relit au maximum d'aujourd'hui",
          "[app][settings]") {
  const auto s = ajp::app::play::parse_settings(
      "[hdr]\nengine = 'rtx'\ncontrast = 350\nsaturation = 400\n[grade]\npreset = 'perso'\nvibrance = "
      "1.1\nvibrance_rtx = 1.2\n");
  REQUIRE(s.has_value());
  CHECK(s->true_hdr_contrast == 200);
  CHECK(s->true_hdr_saturation == 200);
  CHECK(s->grade_values.vibrance == 0.8);
  CHECK(s->grade_values.vibrance_rtx == 0.8);
}

TEST_CASE("le look (LUT) et son intensité se relisent à l'identique, un nom de chemin est refusé",
          "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.look = "maison:shinkai";
  s.look_mix = 0.65;
  s.look_layers = {{"HDR_base.cube", 0.4}, {"maison:pastel", 0.25}, {"x.cube", 1.0}};
  const auto back = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(back.has_value());
  CHECK(back->look == "maison:shinkai");
  CHECK(back->look_mix == 0.65);
  CHECK(back->look_layers == s.look_layers);
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlooks = ['a/b.cube']\n").has_value());
  CHECK_FALSE(
      ajp::app::play::parse_settings("[grade]\nlooks = ['a.cube']\nlooks_mix = [-0.1]\n").has_value());
  CHECK_FALSE(
      ajp::app::play::parse_settings("[grade]\nlooks = ['a','b','c','d','e','f','g','h']\n").has_value());
  const auto no_mix = ajp::app::play::parse_settings("[grade]\nlooks = ['a.cube']\n");
  REQUIRE(no_mix.has_value());
  CHECK(no_mix->look_layers.at(0).mix == 1.0);
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlook = '../x.cube'\n").has_value());
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlook_mix = 1.5\n").has_value());
  CHECK(ajp::app::play::PlaySettings{}.look.empty());  // aucun look par défaut
}
