#include "app/play/machine_calibration.h"

#include <algorithm>
#include <array>
#include <format>
#include <fstream>
#include <system_error>

#include "app/play/i18n.h"
#include "core/log.h"

namespace ajp::app::play {

namespace {

// Les familles, de la meilleure à la plus légère.
constexpr std::array<ChainCandidate, 3> kFamilies{{
    {.span = true, .rife = true, .height = kCenterHeight},
    {.span = true, .rife = false, .height = kCenterHeight},
    {.span = false, .rife = true, .height = 0},
}};

bool same_family(const ChainCandidate& a, const ChainCandidate& b) {
  return a.span == b.span && a.rife == b.rife;
}

const CandidateTrial* find(const std::vector<CandidateTrial>& tried, const ChainCandidate& chain) {
  const auto it = std::ranges::find(tried, chain, &CandidateTrial::chain);
  return it == tried.end() ? nullptr : &*it;
}

// Le prochain palier d'une famille SPAN, ou rien (finie) ; best reçoit son plus haut palier qui tient.
std::optional<int> next_height(const std::vector<CandidateTrial>& tried, const ChainCandidate& family,
                               std::optional<int>& best) {
  best.reset();
  const auto at = [&](int height) {
    ChainCandidate chain = family;
    chain.height = height;
    return find(tried, chain);
  };
  const CandidateTrial* center = at(kCenterHeight);
  if (center == nullptr) return kCenterHeight;
  const auto index =
      static_cast<int>(std::ranges::find(kCalibrationHeights, kCenterHeight) - kCalibrationHeights.begin());
  if (center->holds) {
    best = kCenterHeight;
    for (int i = index + 1; i < static_cast<int>(kCalibrationHeights.size()); ++i) {
      const int height = kCalibrationHeights[static_cast<std::size_t>(i)];
      const CandidateTrial* trial = at(height);
      if (trial == nullptr) return height;
      if (!trial->holds) return std::nullopt;
      best = height;
    }
    return std::nullopt;
  }
  for (int i = index - 1; i >= 0; --i) {
    const int height = kCalibrationHeights[static_cast<std::size_t>(i)];
    const CandidateTrial* trial = at(height);
    if (trial == nullptr) return height;
    if (trial->holds) {
      best = height;
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried) {
  for (const ChainCandidate& family : kFamilies) {
    if (family.span) {
      std::optional<int> best;
      if (const auto height = next_height(tried, family, best)) {
        ChainCandidate chain = family;
        chain.height = *height;
        return chain;
      }
      if (best) return std::nullopt;  // la meilleure famille qui tient : fini
      continue;
    }
    const CandidateTrial* trial = find(tried, family);
    if (trial == nullptr) return family;
    if (trial->holds) return std::nullopt;
  }
  return std::nullopt;
}

ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried) {
  for (const ChainCandidate& family : kFamilies) {
    std::optional<ChainCandidate> best;
    for (const CandidateTrial& trial : tried) {
      if (!trial.holds || !same_family(trial.chain, family)) continue;
      if (!best || trial.chain.height > best->height) best = trial.chain;
    }
    if (best) return *best;
  }
  return ChainCandidate{.span = false, .rife = false, .height = 0};
}

PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base) {
  base.preset = Preset::Custom;
  base.upscale = chain.span;
  if (chain.span) base.work_height = chain.height;
  base.downscale = Downscale::Hermite;
  base.motion_engine = MotionEngine::Rife;
  base.rife_model = std::string(kCalibrationRife);
  base.rife_auto = chain.rife;
  base.auto_free = chain.rife;  // le mode auto Rapide
  base.rife_factor = chain.rife ? 2 : 0;
  base.rife_factor_den = 1;
  return base;
}

std::string describe(const ChainCandidate& chain) {
  std::string out = chain.span ? std::format("{}p · SPAN", chain.height) : std::string("SPAN off");
  out += chain.rife ? " · RIFE 4.22 lite" : " · RIFE off";
  return out;
}

std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction) {
  const auto log = log_channel("app");
  MachineCalibration out;
  std::error_code error;
  std::filesystem::create_directories(plan.work_dir, error);
  int index = 0;
  while (const auto chain = next_candidate(out.tried)) {
    if (fraction) fraction(std::min(0.95F, 0.05F + 0.15F * static_cast<float>(index)));
    const std::filesystem::path settings = plan.work_dir / std::format("chaine-{}.toml", index);
    if (auto written = save_settings(settings, candidate_settings(*chain, base)); !written)
      return std::unexpected(std::move(written.error()));
    const Variant variant{.name = describe(*chain), .settings = settings};
    // Le premier essai charge, et la première fois construit, les moteurs : il a tout son temps.
    auto reading =
        play_variant(plan, variant, index, stop, index == 0 ? plan.first_allowance : plan.build_allowance);
    if (stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, i18n::tr("evaluation cancelled")});
    if (!reading && index == 0) return std::unexpected(std::move(reading.error()));
    const bool held = reading && holds(reading->run);
    log->info("calibration de la machine : {} {}", describe(*chain), held ? "tient" : "ne tient pas");
    out.tried.push_back({.chain = *chain, .holds = held});
    ++index;
  }
  out.chosen = best_candidate(out.tried);
  log->info("calibration de la machine : gardé {}", describe(out.chosen));
  if (fraction) fraction(1.0F);
  return out;
}

}  // namespace ajp::app::play
