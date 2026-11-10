#include "app/play/machine_calibration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>

#include <SDL3/SDL_power.h>

#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "core/log.h"
#include "core/text_file.h"
#include "core/utf8_path.h"
#include "media/bench_clip.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {

namespace {

const CandidateTrial* find(const std::vector<CandidateTrial>& tried, const ChainCandidate& chain) {
  const auto it = std::ranges::find(tried, chain, &CandidateTrial::chain);
  return it == tried.end() ? nullptr : &*it;
}

// Le prochain palier d'une famille SPAN, du plus lourd au plus léger, jusqu'au premier qui tient ; rien si
// elle est finie (held : un palier a tenu).
std::optional<int> next_height(const std::vector<CandidateTrial>& tried, ChainCandidate family,
                               const std::vector<int>& heights, bool& held) {
  held = false;
  for (const int height : heights) {
    family.height = height;
    const CandidateTrial* trial = find(tried, family);
    if (trial == nullptr) return height;
    if (trial->holds) {
      held = true;
      return std::nullopt;
    }
  }
  return std::nullopt;
}

constexpr ChainCandidate kRifeOnly{.span = false, .rife = true, .height = 0};

}  // namespace

float creeping_fraction(float reported, double seconds_since_change) {
  constexpr double kCreep = 0.08;
  constexpr double kSeconds = 120.0;
  const double extra = kCreep * (1.0 - std::exp(-std::max(seconds_since_change, 0.0) / kSeconds));
  return std::min(0.97F, reported + static_cast<float>(extra));
}

int calibration_minutes(bool builds_engines, bool laptop) {
  const auto sizes = static_cast<double>(calibration_heights().size());
  // Les essais : chaque palier de SPAN seul, deux ou trois de SPAN + RIFE, un rejeu sur deux ; chacun la
  // durée de l'essai et une quinzaine de secondes d'ouverture.
  const double trials = (sizes + 3.0) * 1.5;
  double seconds = trials * (kBenchSeconds + 15.0);
  // Une construction de TensorRT par taille, SPAN puis RIFE : 2,5 min sur une RTX 4070 Ti, le double sur un
  // portable.
  if (builds_engines) seconds += (sizes + 2.0) * (laptop ? 300.0 : 150.0);
  return std::max(1, static_cast<int>(std::ceil(seconds / 60.0)));
}

bool needs_second_run(const CandidateTrial& trial) {
  if (!trial.holds) return true;
  // De justesse : plus de la moitié des pertes permises, ou une cadence moyenne sous 97 % de la visée.
  const bool lossy = trial.lost_percent > 50.0 * kMaxLossRate;
  const bool slow = trial.target_fps > 0.0 && trial.fps > 0.0 && trial.fps < 0.97 * trial.target_fps;
  return lossy || slow;
}

double credited_fps(const BenchReading& reading) {
  if (reading.interval_mean_us <= 0) return 0.0;
  const double fps = 1e6 / static_cast<double>(reading.interval_mean_us);
  if (reading.new_frames <= 0) return fps;
  return fps * static_cast<double>(reading.new_frames + reading.uninterpolated) /
         static_cast<double>(reading.new_frames);
}

double trial_target_fps(const ChainCandidate& chain, std::int64_t period_us) {
  const auto source = static_cast<double>(media::BenchClipSpec{}.fps);
  if (!chain.rife) return source;
  const double display = period_us > 0 ? 1e6 / static_cast<double>(period_us) : 0.0;
  return display > source ? std::min(2.0 * source, display) : 2.0 * source;
}

bool trial_holds(const ChainCandidate& chain, const BenchReading& reading, double target_fps) {
  if (!holds(reading.run)) return false;
  return !chain.rife || credited_fps(reading) >= kMinRifeShare * target_fps;
}

std::vector<int> calibration_heights() {
  return {kCalibrationHeights.rbegin(), kCalibrationHeights.rend()};
}

CalibrationModels calibration_models(Tier tier) {
  return {.span = std::string(level_span_model(tier)), .rife = std::string(level_rife_model(tier))};
}

std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried) {
  const std::vector<int> heavy_first = calibration_heights();
  // 1. SPAN seul, en montant, jusqu'au premier palier qui ne tient pas.
  int span_top = 0;
  for (auto it = heavy_first.rbegin(); it != heavy_first.rend(); ++it) {
    const ChainCandidate chain{.span = true, .rife = false, .height = *it};
    const CandidateTrial* trial = find(tried, chain);
    if (trial == nullptr) return chain;
    if (!trial->holds) break;
    span_top = *it;
  }
  if (span_top > 0) {
    // 2. SPAN + RIFE, du plus haut palier de SPAN seul vers le bas, jusqu'au premier qui tient.
    std::vector<int> below;
    for (const int height : heavy_first)
      if (height <= span_top) below.push_back(height);
    bool held = false;
    if (const auto height = next_height(tried, ChainCandidate{.span = true, .rife = true}, below, held))
      return ChainCandidate{.span = true, .rife = true, .height = *height};
    if (held) return std::nullopt;
  }
  // SPAN + RIFE ne tient nulle part, ou SPAN nulle part : RIFE seul (SPAN coupé), le recours de la fluidité.
  if (find(tried, kRifeOnly) == nullptr) return kRifeOnly;
  return std::nullopt;
}

ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried) {
  // La fluidité d'abord (P3.11) : le plus haut palier où SPAN et RIFE tiennent ensemble,
  // même plus bas qu'un SPAN seul ; sinon RIFE seul s'il tient ; SPAN seul en dernier.
  // Un rang entier (palier, puis RIFE) plutôt qu'un std::optional comparé champ à champ : MSVC 14.51 en /O2
  // perdait le départage par RIFE à palier égal (test de la calibration, release Windows 0.9.0).
  const CandidateTrial* rife = find(tried, kRifeOnly);
  const bool rife_holds = rife != nullptr && rife->holds;
  const CandidateTrial* best = nullptr;
  int best_rank = -1;
  for (const CandidateTrial& trial : tried) {
    if (!trial.holds || !trial.chain.span) continue;
    if (!trial.chain.rife && rife_holds) continue;
    const int rank = trial.chain.height + (trial.chain.rife ? 100'000 : 0);
    if (rank > best_rank) {
      best_rank = rank;
      best = &trial;
    }
  }
  if (best != nullptr) return best->chain;
  if (rife_holds) return kRifeOnly;
  return ChainCandidate{.span = false, .rife = false, .height = 0};
}

PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base,
                                const CalibrationModels& models) {
  base.preset = Preset::Custom;
  base.upscale = chain.span;
  base.restore_lines = true;
  base.model = models.span;
  base.span_precision = PrecisionTier::Balanced;
  base.span_order = SpanOrder::Auto;
  base.work_height = chain.span ? chain.height : 0;
  base.downscale = Downscale::Hermite;
  base.motion_engine = MotionEngine::Rife;
  base.motion_precision = PrecisionTier::Balanced;
  base.rife_model = models.rife.empty() ? std::string(kLightRifeModel) : models.rife;
  base.rife_auto = chain.rife;
  base.auto_free = chain.rife;  // le mode auto Rapide, le seul que la calibration essaie
  base.rife_factor = chain.rife ? 2 : 0;
  base.rife_factor_den = 1;
  base.keep_poses = true;
  return base;
}

std::string describe(const ChainCandidate& chain) {
  std::string out = chain.span ? std::format("{}p · SPAN", chain.height) : std::string(i18n::tr("SPAN off"));
  out += chain.rife ? i18n::tr(" · RIFE (auto Instant)") : i18n::tr(" · no interpolation");
  return out;
}

std::string format_calibration(const MachineCalibration& c) {
  toml::table root;
  root.insert("version", static_cast<std::int64_t>(c.version));
  root.insert("gpu", c.gpu);
  root.insert("date", c.date);
  root.insert("display_height", static_cast<std::int64_t>(c.display_height));
  root.insert("span_model", c.models.span);
  root.insert("rife_model", c.models.rife);
  root.insert("engine", c.engine);
  root.insert("driver", c.driver);
  root.insert("on_battery", c.on_battery);
  const auto chain_table = [](const ChainCandidate& chain) {
    toml::table t;
    t.insert("span", chain.span);
    t.insert("rife", chain.rife);
    t.insert("height", static_cast<std::int64_t>(chain.height));
    return t;
  };
  root.insert("chosen", chain_table(c.chosen));
  toml::array trials;
  for (const CandidateTrial& trial : c.tried) {
    toml::table t = chain_table(trial.chain);
    t.insert("holds", trial.holds);
    t.insert("lost_percent", trial.lost_percent);
    t.insert("fps", trial.fps);
    t.insert("low_fps", trial.low_fps);
    t.insert("target_fps", trial.target_fps);
    trials.push_back(std::move(t));
  }
  root.insert("trials", std::move(trials));
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<MachineCalibration, Refusal> parse_calibration(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed)
    return std::unexpected(Refusal{RefusalCode::ProfileUnreadable,
                                   std::format("calibration.toml : {}", parsed.error().description())});
  const toml::table& root = parsed.table();
  const auto chain_of = [](const toml::table& t) {
    const auto height = t["height"].value_or(std::int64_t{0});
    return ChainCandidate{.span = t["span"].value_or(false),
                          .rife = t["rife"].value_or(false),
                          .height = static_cast<int>(std::clamp<std::int64_t>(height, 0, 4320))};
  };
  MachineCalibration out;
  out.gpu = root["gpu"].value_or(std::string());
  out.date = root["date"].value_or(std::string());
  out.display_height = static_cast<int>(root["display_height"].value_or(std::int64_t{0}));
  // Sans version : un fichier d'avant elle (1). « span_fp8 » des fichiers de version 1 est ignoré.
  out.version =
      static_cast<int>(std::clamp<std::int64_t>(root["version"].value_or(std::int64_t{1}), 1, 1000));
  out.models = {.span = root["span_model"].value_or(std::string()),
                .rife = root["rife_model"].value_or(std::string())};
  out.engine = root["engine"].value_or(std::string());
  out.driver = root["driver"].value_or(std::string());
  out.on_battery = root["on_battery"].value_or(false);
  if (const toml::table* chosen = root["chosen"].as_table()) out.chosen = chain_of(*chosen);
  if (const toml::array* trials = root["trials"].as_array()) {
    for (const toml::node& node : *trials) {
      const toml::table* t = node.as_table();
      if (t == nullptr) continue;
      out.tried.push_back({.chain = chain_of(*t),
                           .holds = (*t)["holds"].value_or(false),
                           .lost_percent = (*t)["lost_percent"].value_or(-1.0),
                           .fps = (*t)["fps"].value_or(0.0),
                           .low_fps = (*t)["low_fps"].value_or(0.0),
                           .target_fps = (*t)["target_fps"].value_or(0.0)});
    }
  }
  if (out.tried.empty())
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "calibration.toml : aucun essai"});
  return out;
}

std::filesystem::path calibration_file(const std::filesystem::path& data_dir, int display_height) {
  if (display_height <= 0) return data_dir / "calibration.toml";
  return data_dir / std::format("calibration-{}p.toml", display_height);
}

std::filesystem::path calibration_work_dir(const std::filesystem::path& data_dir) {
  if (data_dir.empty()) return {};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return data_dir / std::format("calibration-essais-{:x}", static_cast<unsigned long long>(stamp));
}

std::expected<MachineCalibration, Refusal> load_calibration(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::unexpected(Refusal{RefusalCode::FileUnreadable, core::to_utf8(file)});
  std::ostringstream text;
  text << in.rdbuf();
  return parse_calibration(text.str());
}

std::expected<MachineCalibration, Refusal> load_calibration_at(const std::filesystem::path& data_dir,
                                                               int display_height) {
  if (display_height > 0) {
    auto here = load_calibration(calibration_file(data_dir, display_height));
    if (here || here.error().code != RefusalCode::FileUnreadable) return here;
  }
  return load_calibration(calibration_file(data_dir));
}

CalibrationUse calibration_use(const MachineCalibration& calibration, const CalibrationHost& host) {
  // D'une version plus ancienne : ses bornes (moitié de l'écran, RIFE 4.22 lite, RIFE coupé par le Rapide
  // compté comme tenu) sont sous ou à côté de ce que l'Auto estimé donne aujourd'hui (A49-02, A07-02).
  if (calibration.version < kCalibrationVersion) return CalibrationUse::OldVersion;
  // Sur batterie : l'Auto estimé vaut mieux que les bornes d'une carte bridée (P3.12).
  if (calibration.on_battery) return CalibrationUse::OnBattery;
  if (calibration.gpu != host.gpu ||
      (host.display_height > 0 && calibration.display_height != host.display_height))
    return CalibrationUse::OtherMachine;
  // Les bornes de Vulkan sous TensorRT brident la carte, celles de TensorRT sous Vulkan la débordent
  // (A25-02, A07-05).
  if ((calibration.engine == "tensorrt") != host.tensorrt) return CalibrationUse::OtherEngine;
  return CalibrationUse::Followed;
}

bool calibration_driver_changed(const MachineCalibration& calibration, const CalibrationHost& host) {
  return driver_branch_changed(calibration.driver, host.driver);
}

std::optional<CalibratedChain> calibrated_chain(const std::filesystem::path& data_dir,
                                                const CalibrationHost& host) {
  auto calibration = load_calibration_at(data_dir, host.display_height);
  if (!calibration || calibration_use(*calibration, host) != CalibrationUse::Followed) return std::nullopt;
  const ChainCandidate& chosen = calibration->chosen;
  return CalibratedChain{.span = chosen.span,
                         .rife = chosen.rife,
                         .height = chosen.height,
                         .span_model = chosen.span ? calibration->models.span : std::string(),
                         .rife_model = calibration->models.rife};
}

std::expected<void, Refusal> save_calibration(const std::filesystem::path& file,
                                              const MachineCalibration& calibration) {
  return core::write_text_atomically(file, format_calibration(calibration));
}

std::expected<CalibrationTrials, Refusal> run_calibration_trials(const TrialPlayer& play,
                                                                 const std::stop_token& stop,
                                                                 const std::function<void(float)>& fraction) {
  const auto log = log_channel("app");
  CalibrationTrials out;
  // Au plus : les paliers des deux familles SPAN, puis RIFE seul.
  const auto most = static_cast<float>(2 * calibration_heights().size() + 1);
  int index = 0;
  std::optional<ChainCandidate> retried;  // la chaîne déjà rejouée une fois
  std::optional<ChainCandidate> rebuilt;  // la chaîne rejouée avec le délai d'une construction
  while (const auto chain = next_candidate(out.tried)) {
    if (fraction) fraction(std::min(0.95F, 0.05F + 0.9F * static_cast<float>(index) / most));
    // Le premier essai gardé charge, et la première fois construit, les moteurs : il a tout son temps.
    auto reading = play(*chain, index, out.tried.empty() || rebuilt == *chain);
    if (stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, i18n::tr("evaluation cancelled")});
    // Délai dépassé (P3.12) : un moteur long à construire (un portable) n'est pas une machine lente ; l'essai
    // se refait une fois avec le délai d'une construction, le moteur fini reste en cache.
    if (!reading && reading.error().key == kTrialTimedOutKey && rebuilt != *chain) {
      log->info("calibration de la machine : {} a dépassé son délai, rejoué avec celui d'une construction",
                describe(*chain));
      rebuilt = *chain;
      ++index;
      continue;
    }
    if (!reading && out.tried.empty()) return std::unexpected(std::move(reading.error()));
    CandidateTrial trial{.chain = *chain,
                         .target_fps = trial_target_fps(*chain, reading ? reading->run.period_us : 0)};
    if (reading) {
      out.gpu = reading->gpu;
      out.driver = reading->driver;
      out.engine = reading->engine;
      trial.holds = trial_holds(*chain, *reading, trial.target_fps);
      const auto presents = std::max<std::int64_t>(reading->run.presents, 1);
      trial.lost_percent = 100.0 * static_cast<double>(reading->run.dropped + reading->run.missing) /
                           static_cast<double>(presents);
      if (reading->interval_mean_us > 0) trial.fps = 1e6 / static_cast<double>(reading->interval_mean_us);
      if (reading->interval_p99_us > 0) trial.low_fps = 1e6 / static_cast<double>(reading->interval_p99_us);
    }
    // Rejoué dans les deux sens (P3.12) : un essai qui ne tient pas (le premier d'une taille construit son
    // moteur à froid et perd quelques images au démarrage), et un essai qui tient de justesse (une mesure
    // chanceuse) ; le second passage décide.
    if (reading && needs_second_run(trial) && retried != *chain) {
      log->info("calibration de la machine : {} {} ({:.1f} % perdues, {:.1f} i/s pour {:.0f}), rejoué",
                describe(*chain), trial.holds ? "tient de justesse" : "ne tient pas", trial.lost_percent,
                trial.fps, trial.target_fps);
      retried = *chain;
      ++index;
      continue;
    }
    log->info("calibration de la machine : {} {} ({:.1f} i/s pour {:.0f})", describe(*chain),
              trial.holds ? "tient" : "ne tient pas", trial.fps, trial.target_fps);
    out.tried.push_back(trial);
    ++index;
  }
  return out;
}

std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             int display_height, Tier tier,
                                                             const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction) {
  const auto log = log_channel("app");
  MachineCalibration out;
  out.display_height = display_height;
  out.models = calibration_models(tier);
  out.date = std::format("{:%F}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
  std::error_code error;
  // Jamais dans le dossier courant (A25-11) ; le dossier part à la sortie, essais finis ou refusés.
  if (plan.work_dir.empty() || !plan.work_dir.is_absolute())
    return std::unexpected(
        Refusal{RefusalCode::FileUnreadable, "calibration : pas de dossier pour les essais"});
  std::filesystem::create_directories(plan.work_dir, error);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(dir, ignored);
    }
  } const cleanup{plan.work_dir};
  if (fraction) fraction(0.02F);
  if (!std::filesystem::is_regular_file(plan.video, error)) {
    if (auto written = media::write_bench_clip(plan.video); !written)
      return std::unexpected(std::move(written.error()));
  }
  log->info("calibration de la machine : SPAN {}, RIFE {}, en FP16",
            out.models.span.empty() ? std::string(kDefaultModelFile) : out.models.span, out.models.rife);
  out.on_battery = on_battery();
  if (out.on_battery)
    log->warn("calibration de la machine : sur batterie, mesure sous-estimée (à refaire sur secteur)");
  const TrialPlayer play = [&](const ChainCandidate& chain, int index,
                               bool first) -> std::expected<BenchReading, Refusal> {
    // Relue à chaque essai (A07-09) : un portable débranché en cours de route bride sa carte.
    if (!out.on_battery && on_battery()) {
      out.on_battery = true;
      log->warn("calibration de la machine : passée sur batterie, mesure sous-estimée");
    }
    const std::filesystem::path settings = plan.work_dir / std::format("chaine-{}.toml", index);
    if (auto written = save_settings(settings, candidate_settings(chain, base, out.models)); !written)
      return std::unexpected(std::move(written.error()));
    return play_variant(plan, Variant{.name = describe(chain), .settings = settings}, index, stop,
                        first ? plan.first_allowance : plan.build_allowance);
  };
  auto trials = run_calibration_trials(play, stop, fraction);
  if (!trials) return std::unexpected(std::move(trials.error()));
  out.tried = std::move(trials->tried);
  out.gpu = std::move(trials->gpu);
  out.driver = std::move(trials->driver);
  out.engine = std::move(trials->engine);
  out.chosen = best_candidate(out.tried);
  log->info("calibration de la machine : recommandé {} (moteur {})", describe(out.chosen), out.engine);
  if (!plan.data.empty()) {
    if (auto saved = save_calibration(calibration_file(plan.data, out.display_height), out); !saved)
      log->warn("calibration.toml : {}", saved.error().detail);
  }
  if (fraction) fraction(1.0F);
  return out;
}

}  // namespace ajp::app::play
