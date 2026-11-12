#include "app/play/machine_calibration.h"

#include <filesystem>
#include <map>
#include <stop_token>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {

// Joue la recherche contre une machine qui tient ce que holds accepte ; rend les essais faits.
template <typename Holds>
std::vector<CandidateTrial> search(Holds holds, int start_height = 0) {
  std::vector<CandidateTrial> tried;
  while (const auto chain = next_candidate(tried, start_height)) {
    REQUIRE(tried.size() < 20);
    tried.push_back({.chain = *chain, .holds = holds(*chain)});
  }
  return tried;
}

}  // namespace

TEST_CASE("les paliers montent jusqu'à la 1080p de la séquence, sur tout écran (A49-02)",
          "[calibration-machine]") {
  CHECK(calibration_heights() == std::vector<int>{1080, 720, 540, 480, 360});
}

TEST_CASE("les modèles de la calibration sont ceux du niveau, jamais en dessous (A49-01, A38-02)",
          "[calibration-machine]") {
  for (const Tier tier : {Tier::Quality, Tier::Ultra}) {
    CHECK(calibration_models(tier).span.find("Balanced_SPANF3_b8f64") != std::string::npos);
    CHECK(calibration_models(tier).rife == "rife_v4.26.onnx");
  }
  for (const Tier tier : {Tier::Economy, Tier::Fast, Tier::Normal}) {
    CHECK(calibration_models(tier).span.empty());  // le SPAN par défaut, Performance Sharp
    CHECK(calibration_models(tier).rife == "rife_v4.22_lite.onnx");
  }
}

TEST_CASE("la calibration monte SPAN seul, puis descend SPAN + RIFE depuis son plus haut palier",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // Tout tient : SPAN seul 360 → 1080, puis SPAN + RIFE à 1080 d'emblée ; la source 1080p est gardée.
  auto tried = search([](const ChainCandidate&) { return true; });
  REQUIRE(tried.size() == 6);
  CHECK(tried[0].chain == C{.span = true, .rife = false, .height = 360});
  CHECK(tried[4].chain == C{.span = true, .rife = false, .height = 1080});
  CHECK(tried[5].chain == C{.span = true, .rife = true, .height = 1080});
  CHECK(best_candidate(tried) == tried[5].chain);
  // 540p tient au plus : SPAN seul s'arrête au premier échec (720), SPAN + RIFE part de 540.
  tried = search([](const ChainCandidate& c) { return c.height <= 540; });
  REQUIRE(tried.size() == 5);
  CHECK(tried[3].chain == C{.span = true, .rife = false, .height = 720});
  CHECK(tried[4].chain == C{.span = true, .rife = true, .height = 540});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 540});
}

TEST_CASE("la calibration part du palier du niveau estimé, monte ou descend (B07-14)",
          "[calibration-machine]") {
  using C = ChainCandidate;
  CHECK(calibration_start_height(Tier::Ultra) == 1080);
  CHECK(calibration_start_height(Tier::Quality) == 1080);
  CHECK(calibration_start_height(Tier::Normal) == 720);
  // Une 4090 qui tient tout : SPAN seul 1080, puis SPAN + RIFE 1080 ; deux essais au lieu de six.
  auto tried = search([](const ChainCandidate&) { return true; }, 1080);
  REQUIRE(tried.size() == 2);
  CHECK(tried[0].chain == C{.span = true, .rife = false, .height = 1080});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 1080});
  // Estimée Qualité mais 720p au plus : SPAN seul descend de 1080 à 720, SPAN + RIFE part de 720.
  tried = search([](const ChainCandidate& c) { return c.height <= 720; }, 1080);
  REQUIRE(tried.size() == 3);
  CHECK(tried[1].chain == C{.span = true, .rife = false, .height = 720});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 720});
  // Estimée Normal (720) et la carte tient 1080 : SPAN seul monte de 720 à 1080.
  tried = search([](const ChainCandidate&) { return true; }, 720);
  REQUIRE(tried.size() == 3);
  CHECK(tried[1].chain == C{.span = true, .rife = false, .height = 1080});
  // SPAN nulle part : la famille descend jusqu'au bout, puis RIFE seul.
  tried = search([](const ChainCandidate& c) { return !c.span; }, 720);
  CHECK(tried.back().chain == C{.span = false, .rife = true, .height = 0});
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
}

TEST_CASE("la calibration préfère RIFE seul à SPAN seul, SPAN seul quand RIFE ne tient pas",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // RIFE ne tient nulle part : SPAN + RIFE descend jusqu'au bout, RIFE seul est essayé, SPAN seul est gardé.
  auto tried = search([](const ChainCandidate& c) { return c.span && !c.rife && c.height <= 540; });
  CHECK(tried.size() == 4 + 3 + 1);
  CHECK(best_candidate(tried) == C{.span = true, .rife = false, .height = 540});
  // SPAN + RIFE nulle part mais RIFE seul tient : la fluidité passe avant SPAN seul (P3.11).
  tried = search([](const ChainCandidate& c) { return !(c.span && c.rife); });
  CHECK(tried.back().chain == C{.span = false, .rife = true, .height = 0});
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // La fluidité d'abord (P3.11) : 480p + RIFE passe avant 1080p seul, qui reste au tableau.
  tried = search([](const ChainCandidate& c) { return !c.rife || c.height <= 480; });
  CHECK(tried.back().chain == C{.span = true, .rife = true, .height = 480});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 480});
  // SPAN nulle part : RIFE seul, sans SPAN + RIFE.
  tried = search([](const ChainCandidate& c) { return !c.span; });
  REQUIRE(tried.size() == 2);
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // Rien ne tient : SPAN seul au plus léger, puis RIFE seul ; ni l'un ni l'autre.
  tried = search([](const ChainCandidate&) { return false; });
  CHECK(tried.size() == 2);
  CHECK_FALSE(best_candidate(tried).span);
  CHECK_FALSE(best_candidate(tried).rife);
}

TEST_CASE("les réglages d'une chaîne : Personnalisé, Hermite, modèles du niveau en FP16, auto Rapide",
          "[calibration-machine]") {
  PlaySettings base;
  base.true_hdr_peak = 800;
  const PlaySettings s = candidate_settings({.span = true, .rife = true, .height = 540}, base);
  CHECK(s.preset == Preset::Custom);
  CHECK(s.upscale);
  CHECK(s.work_height == 540);
  CHECK(s.downscale == Downscale::Hermite);
  CHECK(s.rife_model == "rife_v4.22_lite.onnx");
  CHECK(s.rife_auto);
  CHECK(s.auto_free);
  CHECK(s.rife_factor == 2);
  CHECK(s.true_hdr_peak == 800);  // le reste des réglages de l'utilisateur
  CHECK(s.model.empty());         // SPAN par défaut (Performance Sharp) en FP16
  CHECK(s.span_precision == PrecisionTier::Balanced);
  // Niveau Qualité : le b8f64 et RIFE 4.25, toujours en FP16 (A49-03 : plus de FP8 posé par la calibration).
  const PlaySettings fine = candidate_settings({.span = true, .rife = true, .height = 1080}, base,
                                               calibration_models(Tier::Quality));
  CHECK(fine.model == kQualitySpanModel);
  CHECK(fine.rife_model == "rife_v4.26.onnx");
  CHECK(fine.span_precision == PrecisionTier::Balanced);
  CHECK(fine.work_height == 1080);
  const PlaySettings plain = candidate_settings({.span = false, .rife = false, .height = 0}, base);
  CHECK_FALSE(plain.upscale);
  CHECK_FALSE(plain.rife_auto);
  CHECK(plain.rife_factor == 0);
}

TEST_CASE("un essai joue la chaîne que l'Auto calibré écrira, pas les réglages de l'utilisateur (A07-01)",
          "[calibration-machine]") {
  PlaySettings base;
  base.span_order = SpanOrder::Before;
  base.keep_poses = false;
  base.motion_precision = PrecisionTier::Performance;
  base.restore_lines = false;  // A07-06 : sans lui un essai SPAN sous 1080p jouerait la source sans SPAN
  const ChainCandidate chain{.span = true, .rife = true, .height = 720};
  const CalibrationModels models = calibration_models(Tier::Quality);
  const PlaySettings trial = candidate_settings(chain, base, models);
  CHECK(span_runs(trial, 1080));
  PlaySettings automatic;
  automatic.preset = Preset::Auto;
  const PlaySettings played = apply_preset(automatic,
                                           Machine{.tier = Tier::Quality,
                                                   .display_height = 1440,
                                                   .display_hz = 60.0,
                                                   .calibrated = CalibratedChain{.span = true,
                                                                                 .rife = true,
                                                                                 .height = chain.height,
                                                                                 .span_model = models.span,
                                                                                 .rife_model = models.rife}},
                                           ajp::core::SourceInfo{.width = 1920, .height = 1080, .fps = 24.0});
  CHECK(trial.upscale == played.upscale);
  CHECK(trial.work_height == played.work_height);
  CHECK(trial.downscale == played.downscale);
  CHECK(trial.model == played.model);
  CHECK(trial.span_precision == played.span_precision);
  CHECK(trial.span_order == played.span_order);
  CHECK(trial.motion_engine == played.motion_engine);
  CHECK(trial.motion_precision == played.motion_precision);
  CHECK(trial.rife_model == played.rife_model);
  CHECK(trial.keep_poses == played.keep_poses);
  CHECK(trial.auto_free == played.auto_free);
  CHECK(trial.rife_factor == played.rife_factor);  // 60 Hz sur 24 i/s : ×2, comme l'essai
}

TEST_CASE("calibration.toml relit les essais et la recommandation", "[calibration-machine]") {
  MachineCalibration c{.chosen = {.span = true, .rife = true, .height = 540},
                       .tried = {{.chain = {.span = true, .rife = true, .height = 720},
                                  .holds = false,
                                  .lost_percent = 12.5,
                                  .fps = 33.0,
                                  .target_fps = 48.0},
                                 {.chain = {.span = true, .rife = true, .height = 540},
                                  .holds = true,
                                  .lost_percent = 0.4,
                                  .fps = 47.9,
                                  .low_fps = 41.0,
                                  .target_fps = 48.0}},
                       .gpu = "RTX 4070 Ti",
                       .date = "2026-11-01",
                       .display_height = 1440,
                       .models = calibration_models(Tier::Quality),
                       .engine = "tensorrt",
                       .driver = "581"};
  const auto read = parse_calibration(format_calibration(c));
  REQUIRE(read);
  CHECK(read->chosen == c.chosen);
  REQUIRE(read->tried.size() == 2);
  CHECK(read->tried[1].holds);
  CHECK(read->tried[0].lost_percent == 12.5);
  CHECK(read->tried[1].fps == 47.9);
  CHECK(read->tried[1].low_fps == 41.0);
  CHECK(read->tried[1].target_fps == 48.0);
  CHECK(read->gpu == "RTX 4070 Ti");
  CHECK(read->display_height == 1440);
  CHECK(read->version == kCalibrationVersion);
  CHECK(read->models == c.models);
  CHECK(read->engine == "tensorrt");
  CHECK(read->driver == "581");
  CHECK_FALSE(read->on_battery);
  CHECK_FALSE(parse_calibration("version = 1\n"));
  // Rejoués dans les deux sens (P3.12) : l'échec, et le succès de justesse ; pas le succès net.
  CHECK(needs_second_run(c.tried[0]));
  CHECK_FALSE(needs_second_run(c.tried[1]));
  CandidateTrial close = c.tried[1];
  close.lost_percent = 1.6;
  CHECK(needs_second_run(close));
  // Sur secteur, à cette version : l'Auto la suit, avec ses modèles.
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-battery";
  std::filesystem::create_directories(dir);
  REQUIRE(save_calibration(calibration_file(dir), c));
  const CalibrationHost host{.gpu = "RTX 4070 Ti", .display_height = 1440, .tensorrt = true, .driver = "581"};
  const auto followed = calibrated_chain(dir, host);
  REQUIRE(followed);
  CHECK(followed->height == 540);
  CHECK(followed->span_model == kQualitySpanModel);
  CHECK(followed->rife_model == "rife_v4.26.onnx");
  CHECK_FALSE(calibrated_chain(dir, {.gpu = "RTX 4090", .display_height = 1440, .tensorrt = true}));
  CHECK(calibration_use(c, {.gpu = "RTX 4070 Ti", .display_height = 2160, .tensorrt = true}) ==
        CalibrationUse::OtherMachine);
  // Mesurée sous TensorRT, jouée sous Vulkan (TensorRT refusé ou désinstallé) : pas suivie (A07-05).
  CalibrationHost vulkan = host;
  vulkan.tensorrt = false;
  CHECK(calibration_use(c, vulkan) == CalibrationUse::OtherEngine);
  CHECK_FALSE(calibrated_chain(dir, vulkan));
  // Un autre pilote : suivie, mais le tableau le dit (A36-02). Un pilote inconnu ne dit rien.
  CHECK_FALSE(calibration_driver_changed(c, host));
  CalibrationHost updated = host;
  updated.driver = std::to_string((590U << 22U) | (10U << 14U));
  CHECK(calibration_use(c, updated) == CalibrationUse::Followed);
  CHECK(calibration_driver_changed(c, updated));
  updated.driver.clear();
  CHECK_FALSE(calibration_driver_changed(c, updated));
  // Mesurée sur batterie : relue, mais l'Auto ne la suit pas.
  c.on_battery = true;
  REQUIRE(save_calibration(calibration_file(dir), c));
  CHECK(load_calibration(calibration_file(dir))->on_battery);
  CHECK(calibration_use(c, host) == CalibrationUse::OnBattery);
  CHECK_FALSE(calibrated_chain(dir, host));
  std::filesystem::remove_all(dir);
}

TEST_CASE("une calibration par hauteur d'écran, l'ancien fichier unique relu (A37-10)",
          "[calibration-machine]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-ecrans";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  MachineCalibration desk{.chosen = {.span = true, .rife = true, .height = 720},
                          .tried = {{.chain = {.span = true, .rife = true, .height = 720}, .holds = true}},
                          .gpu = "RTX 4070 Ti",
                          .display_height = 1440,
                          .models = calibration_models(Tier::Quality),
                          .engine = "tensorrt"};
  const CalibrationHost at_desk{.gpu = "RTX 4070 Ti", .display_height = 1440, .tensorrt = true};
  const CalibrationHost at_tv{.gpu = "RTX 4070 Ti", .display_height = 2160, .tensorrt = true};
  // L'ancien calibration.toml (1440p) sert encore au bureau, pas à la télé.
  REQUIRE(save_calibration(calibration_file(dir), desk));
  CHECK(calibrated_chain(dir, at_desk));
  CHECK_FALSE(calibrated_chain(dir, at_tv));
  // Calibrer la télé écrit son propre fichier : le bureau garde le sien.
  MachineCalibration tv = desk;
  tv.display_height = 2160;
  tv.chosen.height = 1080;
  REQUIRE(save_calibration(calibration_file(dir, 2160), tv));
  CHECK(calibration_file(dir, 2160).filename() == "calibration-2160p.toml");
  REQUIRE(calibrated_chain(dir, at_tv));
  CHECK(calibrated_chain(dir, at_tv)->height == 1080);
  REQUIRE(calibrated_chain(dir, at_desk));
  CHECK(calibrated_chain(dir, at_desk)->height == 720);
  CHECK(load_calibration_at(dir, 2160)->display_height == 2160);
  CHECK(load_calibration_at(dir, 1440)->display_height == 1440);
  std::filesystem::remove_all(dir);
}

TEST_CASE("un calibration.toml d'avant la version 2 se relit mais l'Auto ne le suit plus",
          "[calibration-machine]") {
  // Tel qu'écrit par la 0.10 : essais bornés à la moitié de l'écran, SPAN FP8, RIFE 4.22 lite.
  constexpr std::string_view kOld = R"(version = 1
gpu = "RTX 4070 Ti"
date = "2026-11-01"
display_height = 1440
span_fp8 = true
on_battery = false

[chosen]
span = true
rife = true
height = 720

[[trials]]
span = true
rife = true
height = 720
holds = true
lost_percent = 0.4
fps = 47.9
low_fps = 41.0
target_fps = 48.0
)";
  const auto read = parse_calibration(kOld);
  REQUIRE(read);
  CHECK(read->version == 1);
  CHECK(read->chosen == ChainCandidate{.span = true, .rife = true, .height = 720});
  CHECK(read->models.span.empty());
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-v1";
  std::filesystem::create_directories(dir);
  REQUIRE(save_calibration(calibration_file(dir), *read));  // garde sa version : toujours ancienne
  CHECK(load_calibration(calibration_file(dir))->version == 1);
  const CalibrationHost host{.gpu = "RTX 4070 Ti", .display_height = 1440, .tensorrt = true};
  CHECK(calibration_use(*read, host) == CalibrationUse::OldVersion);
  CHECK_FALSE(calibrated_chain(dir, host));
  std::filesystem::remove_all(dir);
}

TEST_CASE("la durée annoncée et la barre qui avance pendant une construction (P3.13)",
          "[calibration-machine]") {
  const int plain = calibration_minutes(false, false);
  CHECK(plain >= 3);
  CHECK(plain <= 8);
  CHECK(calibration_minutes(true, false) > plain + 10);  // TensorRT construit ses moteurs
  CHECK(calibration_minutes(true, true) > calibration_minutes(true, false));
  CHECK(creeping_fraction(0.05F, 0.0) == 0.05F);
  CHECK(creeping_fraction(0.05F, 300.0) > 0.1F);   // cinq minutes de construction : la barre a bougé
  CHECK(creeping_fraction(0.05F, 1e6) <= 0.131F);  // sans rattraper l'essai suivant
  CHECK(creeping_fraction(0.95F, 1e6) <= 0.97F);
}

namespace {

// Une lecture de dix secondes sur un écran à 60 Hz : sans perte, fps images neuves par seconde.
BenchReading reading_at(double fps, std::int64_t uninterpolated = 0) {
  BenchReading reading{.gpu = "RTX 4070 Ti", .driver = "581", .engine = "tensorrt"};
  reading.run = {.presents = 600, .lag_p95_us = 4000, .period_us = 16'667};
  reading.interval_mean_us = static_cast<std::int64_t>(1e6 / fps);
  reading.interval_p99_us = reading.interval_mean_us * 2;
  reading.new_frames = static_cast<std::int64_t>(fps * 10.0);
  reading.uninterpolated = uninterpolated;
  return reading;
}

}  // namespace

TEST_CASE("un essai avec RIFE ne tient que si RIFE double vraiment la cadence (A07-02)",
          "[calibration-machine]") {
  const ChainCandidate both{.span = true, .rife = true, .height = 540};
  const ChainCandidate alone{.span = true, .rife = false, .height = 540};
  // Le Rapide a baissé le facteur faute de temps : 26 i/s pour 48, aucune image perdue. Avant, « tient ».
  const BenchReading cut = reading_at(26.0);
  CHECK(holds(cut.run));
  CHECK_FALSE(trial_holds(both, cut, 48.0));
  CHECK(trial_holds(both, reading_at(47.5), 48.0));
  // Les coupes et les dessins tenus de la séquence ne comptent pas contre RIFE : 43 i/s et 4,5 instants
  // par seconde laissés sans image à dessein valent 47,5.
  CHECK_FALSE(trial_holds(both, reading_at(43.0), 48.0));
  CHECK(trial_holds(both, reading_at(43.0, 45), 48.0));
  // Sans RIFE, la cadence obtenue n'entre pas dans le verdict ; des pertes, toujours.
  CHECK(trial_holds(alone, reading_at(24.0), 24.0));
  BenchReading lossy = reading_at(47.9);
  lossy.run.dropped = 60;
  CHECK_FALSE(trial_holds(both, lossy, 48.0));
  // Les pertes se comptent par image visée, pas par présentation (B07-02) : 12 pertes en dix secondes à
  // 60 Hz font 2 % des présentations, mais 5 % des 240 images d'une séquence à 24 i/s.
  BenchReading jerky = reading_at(24.0);
  jerky.run.dropped = 12;
  CHECK(holds(jerky.run));
  CHECK_FALSE(trial_holds(alone, jerky, 24.0));
  jerky.run.dropped = 1;  // une image sur 240 : tient
  CHECK(trial_holds(alone, jerky, 24.0));
  jerky.run.dropped = 2;  // deux sur 240 : plus de 0,5 %
  CHECK_FALSE(trial_holds(alone, jerky, 24.0));
  // Une lecture sans décompte (lecteur plus ancien) : la cadence brute.
  BenchReading old = reading_at(47.0);
  old.new_frames = 0;
  CHECK(credited_fps(old) > 46.9);
  CHECK(credited_fps(BenchReading{}) == 0.0);
  // La cadence visée : doublée, sans dépasser l'écran ; un écran inconnu ou plus lent que la source ne
  // borne rien.
  CHECK(trial_target_fps(alone, 16'667) == 24.0);
  CHECK(trial_target_fps(both, 16'667) == 48.0);
  CHECK(trial_target_fps(both, 0) == 48.0);
  CHECK(trial_target_fps(both, 25'000) == 40.0);
}

TEST_CASE("la recherche descend SPAN + RIFE jusqu'au palier où RIFE tourne vraiment (A07-02, A07-11)",
          "[calibration-machine]") {
  // SPAN seul tient jusqu'à 720p ; avec RIFE, 720p et 540p retombent à 26 i/s sans perte, 480p tient.
  std::map<int, int> played;  // par hauteur, les essais SPAN + RIFE joués
  const TrialPlayer play = [&](const ChainCandidate& chain, int,
                               bool) -> std::expected<BenchReading, ajp::Refusal> {
    if (!chain.rife) {
      BenchReading reading = reading_at(24.0);
      if (chain.height > 720) reading.run.dropped = 200;
      return reading;
    }
    ++played[chain.height];
    return reading_at(chain.height > 480 ? 26.0 : 47.8);
  };
  const auto trials = run_calibration_trials(play, std::stop_token{}, {});
  REQUIRE(trials);
  CHECK(trials->gpu == "RTX 4070 Ti");
  CHECK(trials->driver == "581");
  CHECK(trials->engine == "tensorrt");
  CHECK(best_candidate(trials->tried) == ChainCandidate{.span = true, .rife = true, .height = 480});
  // Un essai qui ne tient pas se rejoue une fois ; celui qui tient avec marge, non.
  CHECK(played[720] == 2);
  CHECK(played[540] == 2);
  CHECK(played[480] == 1);
  CHECK(played.count(1080) == 0);  // au-dessus du plus haut palier de SPAN seul : pas essayé
  for (const CandidateTrial& trial : trials->tried)
    if (trial.chain.rife && trial.chain.height > 480) CHECK_FALSE(trial.holds);
}

TEST_CASE("la recherche : arrêt demandé, premier essai vide, délai dépassé rejoué (A07-11)",
          "[calibration-machine]") {
  std::stop_source stop;
  const TrialPlayer cancelled = [&](const ChainCandidate&, int,
                                    bool) -> std::expected<BenchReading, ajp::Refusal> {
    stop.request_stop();
    return reading_at(24.0);
  };
  const auto stopped = run_calibration_trials(cancelled, stop.get_token(), {});
  REQUIRE_FALSE(stopped);
  CHECK(stopped.error().code == ajp::RefusalCode::Cancelled);

  const TrialPlayer nothing = [](const ChainCandidate&, int,
                                 bool) -> std::expected<BenchReading, ajp::Refusal> {
    return std::unexpected(ajp::Refusal{ajp::RefusalCode::DecodeFailed, "rien"});
  };
  const auto empty = run_calibration_trials(nothing, std::stop_token{}, {});
  REQUIRE_FALSE(empty);
  CHECK(empty.error().code == ajp::RefusalCode::DecodeFailed);

  // Un délai dépassé (une construction de moteur) : le même essai, une fois, avec le long délai.
  int calls = 0;
  bool long_second = false;
  const TrialPlayer slow_build = [&](const ChainCandidate& chain, int,
                                     bool first) -> std::expected<BenchReading, ajp::Refusal> {
    ++calls;
    if (calls == 2) {
      ajp::Refusal refusal{ajp::RefusalCode::DecodeFailed, "délai"};
      refusal.key = std::string(kTrialTimedOutKey);
      return std::unexpected(std::move(refusal));
    }
    if (calls == 3) long_second = first && chain.height == 480;
    return reading_at(chain.rife ? 47.9 : 24.0);
  };
  float last = 0.0F;
  const auto built = run_calibration_trials(slow_build, std::stop_token{}, [&](float f) { last = f; });
  REQUIRE(built);
  CHECK(long_second);
  CHECK(last > 0.0F);
  CHECK(best_candidate(built->tried) == ChainCandidate{.span = true, .rife = true, .height = 1080});
}

TEST_CASE("un calibration.toml de version 2 (sans moteur noté) se relit mais n'est plus suivi",
          "[calibration-machine]") {
  constexpr std::string_view kTwo = R"(version = 2
gpu = "RTX 4070 Ti"
date = "2026-11-09"
display_height = 1440
span_model = "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"
rife_model = "rife_v4.25.onnx"
on_battery = false

[chosen]
span = true
rife = true
height = 1080

[[trials]]
span = true
rife = true
height = 1080
holds = true
lost_percent = 0.0
fps = 26.0
low_fps = 22.0
target_fps = 48.0
)";
  const auto read = parse_calibration(kTwo);
  REQUIRE(read);
  CHECK(read->version == 2);
  CHECK(read->engine.empty());
  CHECK(read->models.rife == "rife_v4.25.onnx");
  CHECK(calibration_use(*read, {.gpu = "RTX 4070 Ti", .display_height = 1440, .tensorrt = true}) ==
        CalibrationUse::OldVersion);
}

TEST_CASE("calibration : le dossier des essais est propre au lancement, sous les données (A25-11)",
          "[app][calibration]") {
  const std::filesystem::path data = std::filesystem::temp_directory_path() / "ajp-donnees";
  const std::filesystem::path first = calibration_work_dir(data);
  CHECK(first.parent_path() == data);
  CHECK(first.is_absolute());
  CHECK(calibration_work_dir({}).empty());
  // Sans dossier, ou avec un chemin relatif au dossier courant : refusé avant tout essai.
  for (const std::filesystem::path& bad :
       {std::filesystem::path{}, std::filesystem::path{"ajp-calibration"}}) {
    const BenchPlan plan{.executable = "/x/ajp", .video = "/v/a.mkv", .work_dir = bad};
    const auto refused = calibrate_machine(plan, PlaySettings{}, 1080, Tier::Normal, std::stop_token{}, {});
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == ajp::RefusalCode::FileUnreadable);
    CHECK_FALSE(std::filesystem::exists("ajp-calibration"));
  }
}

TEST_CASE("la fin de la calibration dit la chaîne retenue et ce qui a été abaissé (C25-02)",
          "[calibration-machine]") {
  const auto trial = [](ChainCandidate chain, bool holds, double lost, double fps, double target) {
    return CandidateTrial{.chain = chain,
                          .holds = holds,
                          .lost_percent = lost,
                          .fps = fps,
                          .low_fps = fps,
                          .target_fps = target};
  };
  const ChainCandidate full{.span = true, .rife = true, .height = 1080};
  const ChainCandidate spans{.span = true, .rife = false, .height = 1080};
  const ChainCandidate low{.span = true, .rife = false, .height = 720};
  MachineCalibration c;
  // La chaîne la plus lourde tient : rien d'abaissé, mais la mesure est dite.
  c.chosen = full;
  c.tried = {trial(full, true, 0.2, 47.5, 48.0)};
  const std::string whole = calibration_notice(c);
  CHECK(whole.contains("full quality kept"));
  CHECK(whole.contains("47.5 of 48"));
  // Une chaîne abaissée dit de quoi, et la perte d'images du réglage juste au-dessus qui ne tenait pas.
  c.chosen = low;
  c.tried = {trial(full, false, 12.5, 30.0, 48.0), trial(spans, false, 8.0, 20.0, 24.0),
             trial(low, true, 0.0, 24.0, 24.0)};
  const std::string lowered = calibration_notice(c);
  CHECK(lowered.contains("working resolution 720p instead of 1080p"));
  CHECK(lowered.contains("no interpolation"));
  CHECK(lowered.contains("24.0 of 24"));
  CHECK(lowered.contains("8.0 %"));  // le plus léger des essais plus lourds, pas le plus lourd
  CHECK_FALSE(lowered.contains("12.5 %"));
  // Rien n'a tenu : le plus léger est gardé, et c'est dit.
  c.tried = {trial(full, false, 12.5, 30.0, 48.0)};
  CHECK(calibration_notice(c).contains("no setting held the target"));
}
