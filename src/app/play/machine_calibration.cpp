#include "app/play/machine_calibration.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <map>
#include <sstream>
#include <system_error>

#include "app/play/i18n.h"
#include "core/log.h"
#include "core/user_overlay.h"
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

std::vector<int> calibration_heights(int display_height) {
  const int top = std::max(display_height / 2, kCalibrationHeights.front());
  std::vector<int> out;
  for (const int height : kCalibrationHeights)
    if (height <= top) out.insert(out.begin(), height);
  return out;
}

std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried, int display_height) {
  const std::vector<int> heights = calibration_heights(display_height);
  bool held = false;
  for (const bool rife : {true, false}) {
    const ChainCandidate family{.span = true, .rife = rife, .height = 0};
    if (const auto height = next_height(tried, family, heights, held))
      return ChainCandidate{.span = true, .rife = rife, .height = *height};
    if (held) break;  // la meilleure famille qui tient : SPAN seul n'est pas essayé
  }
  // RIFE seul (SPAN coupé) : de quoi comparer, et le recours quand SPAN ne tient nulle part.
  if (find(tried, kRifeOnly) == nullptr) return kRifeOnly;
  return std::nullopt;
}

ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried) {
  for (const ChainCandidate family :
       {ChainCandidate{.span = true, .rife = true}, ChainCandidate{.span = true, .rife = false}, kRifeOnly}) {
    std::optional<ChainCandidate> best;
    for (const CandidateTrial& trial : tried) {
      if (!trial.holds || trial.chain.span != family.span || trial.chain.rife != family.rife) continue;
      if (!best || trial.chain.height > best->height) best = trial.chain;
    }
    if (best) return *best;
  }
  return ChainCandidate{.span = false, .rife = false, .height = 0};
}

PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base) {
  base.preset = Preset::Custom;
  base.upscale = chain.span;
  base.work_height = chain.span ? chain.height : 0;
  base.downscale = Downscale::Hermite;
  base.motion_engine = MotionEngine::Rife;
  base.rife_model = std::string(kCalibrationRife);
  base.rife_auto = chain.rife;
  base.auto_free = chain.rife;  // le mode auto Rapide, le seul que la calibration essaie
  base.rife_factor = chain.rife ? 2 : 0;
  base.rife_factor_den = 1;
  return base;
}

PlaySettings trial_settings(const CandidateTrial& trial, PlaySettings base) {
  base = candidate_settings(trial.chain, std::move(base));
  if (trial.fp8 && trial.chain.span) base.span_precision = PrecisionTier::Performance;
  return base;
}

PlaySettings chosen_settings(const MachineCalibration& calibration, PlaySettings base) {
  const CandidateTrial* trial = find(calibration.tried, calibration.chosen);
  if (trial != nullptr) return trial_settings(*trial, std::move(base));
  return candidate_settings(calibration.chosen, std::move(base));
}

std::string describe(const ChainCandidate& chain) {
  std::string out = chain.span ? std::format("{}p · SPAN", chain.height) : std::string(i18n::tr("SPAN off"));
  out += chain.rife ? i18n::tr(" · RIFE 4.22 lite (auto Fast)") : i18n::tr(" · no interpolation");
  return out;
}

std::string format_calibration(const MachineCalibration& c) {
  toml::table root;
  root.insert("version", 1);
  root.insert("gpu", c.gpu);
  root.insert("date", c.date);
  root.insert("display_height", static_cast<std::int64_t>(c.display_height));
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
    if (trial.fp8) t.insert("fp8", true);
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
                           .target_fps = (*t)["target_fps"].value_or(0.0),
                           .fp8 = (*t)["fp8"].value_or(false)});
    }
  }
  if (out.tried.empty())
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "calibration.toml : aucun essai"});
  return out;
}

std::filesystem::path calibration_file(const std::filesystem::path& data_dir) {
  return data_dir / "calibration.toml";
}

std::expected<MachineCalibration, Refusal> load_calibration(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::unexpected(Refusal{RefusalCode::FileUnreadable, file.string()});
  std::ostringstream text;
  text << in.rdbuf();
  return parse_calibration(text.str());
}

std::expected<void, Refusal> save_calibration(const std::filesystem::path& file,
                                              const MachineCalibration& calibration) {
  return core::write_text_atomically(file, format_calibration(calibration));
}

std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             int display_height, const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction) {
  const auto log = log_channel("app");
  MachineCalibration out;
  out.display_height = display_height;
  out.date = std::format("{:%F}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
  std::error_code error;
  std::filesystem::create_directories(plan.work_dir, error);
  if (fraction) fraction(0.02F);
  if (!std::filesystem::is_regular_file(plan.video, error)) {
    if (auto written = media::write_bench_clip(plan.video); !written)
      return std::unexpected(std::move(written.error()));
  }
  // Au plus : les paliers des deux familles SPAN, puis RIFE seul.
  const auto most = static_cast<float>(2 * calibration_heights(display_height).size() + 1);
  int index = 0;
  bool fp8_possible = !plan.span_fp8_video.empty();
  std::map<int, bool> fp8_ready;  // par hauteur de travail : le paquet FP8 prêt (préparé une fois)
  while (const auto chain = next_candidate(out.tried, display_height)) {
    if (fraction) fraction(std::min(0.95F, 0.05F + 0.9F * static_cast<float>(index) / most));
    const std::filesystem::path settings = plan.work_dir / std::format("chaine-{}.toml", index);
    CandidateTrial trial{
        .chain = *chain,
        .target_fps = static_cast<double>(media::BenchClipSpec{}.fps) * (chain->rife ? 2.0 : 1.0)};
    if (chain->span && fp8_possible && !fp8_ready.contains(chain->height)) {
      // La préparation lit les réglages FP16 de l'essai (sa taille de travail) ; la première construit aussi
      // les moteurs.
      if (auto written = save_settings(settings, candidate_settings(*chain, base)); !written)
        return std::unexpected(std::move(written.error()));
      const bool ready =
          prepare_span_variant(plan, {.name = describe(*chain), .settings = settings}, index, stop,
                               fp8_ready.empty() ? plan.first_allowance : plan.build_allowance);
      if (stop.stop_requested())
        return std::unexpected(Refusal{RefusalCode::Cancelled, i18n::tr("evaluation cancelled")});
      if (!ready && fp8_ready.empty()) fp8_possible = false;
      fp8_ready[chain->height] = ready;
    }
    trial.fp8 = chain->span && fp8_ready.contains(chain->height) && fp8_ready.at(chain->height);
    if (auto written = save_settings(settings, trial_settings(trial, base)); !written)
      return std::unexpected(std::move(written.error()));
    const Variant variant{.name = describe(*chain), .settings = settings};
    // Le premier essai charge, et la première fois construit, les moteurs : il a tout son temps.
    auto reading =
        play_variant(plan, variant, index, stop, index == 0 ? plan.first_allowance : plan.build_allowance);
    if (stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, i18n::tr("evaluation cancelled")});
    if (!reading && index == 0) return std::unexpected(std::move(reading.error()));
    if (reading) {
      out.gpu = reading->gpu;
      trial.holds = holds(reading->run);
      const auto presents = std::max<std::int64_t>(reading->run.presents, 1);
      trial.lost_percent = 100.0 * static_cast<double>(reading->run.dropped + reading->run.missing) /
                           static_cast<double>(presents);
      if (reading->interval_mean_us > 0) trial.fps = 1e6 / static_cast<double>(reading->interval_mean_us);
      if (reading->interval_p99_us > 0) trial.low_fps = 1e6 / static_cast<double>(reading->interval_p99_us);
    }
    log->info("calibration de la machine : {}{} {}", describe(*chain), trial.fp8 ? " (FP8)" : "",
              trial.holds ? "tient" : "ne tient pas");
    out.tried.push_back(trial);
    ++index;
  }
  out.chosen = best_candidate(out.tried);
  log->info("calibration de la machine : recommandé {}", describe(out.chosen));
  std::filesystem::remove_all(plan.work_dir, error);
  if (!plan.data.empty()) {
    if (auto saved = save_calibration(calibration_file(plan.data), out); !saved)
      log->warn("calibration.toml : {}", saved.error().detail);
  }
  if (fraction) fraction(1.0F);
  return out;
}

}  // namespace ajp::app::play
