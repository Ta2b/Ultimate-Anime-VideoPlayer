#include "app/play/auto_config.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace ajp::app::play {
namespace {

struct PresetName {
  Preset preset;
  std::string_view key;
  std::string_view label;
};
constexpr std::array<PresetName, kPresetCount> kPresets{{
    {Preset::Auto, "auto", "Auto"},
    {Preset::Economy, "economy", "Économe"},
    {Preset::Fast, "fast", "Rapide"},
    {Preset::Normal, "normal", "Normal"},
    {Preset::Quality, "quality", "Qualité"},
    {Preset::Ultra, "ultra", "Ultra"},
    {Preset::Custom, "custom", "Personnalisé"},
}};

struct PriorityName {
  Priority priority;
  std::string_view key;
  std::string_view label;
};
constexpr std::array<PriorityName, kPriorityCount> kPriorities{{
    {Priority::Fluidity, "fluidity", "Fluidité"},
    {Priority::Balance, "balance", "Équilibre"},
    {Priority::Image, "image", "Image"},
}};

// Les résolutions de travail proposées par le menu, de la plus petite à la plus grande.
constexpr std::array<int, 5> kWorkHeights{480, 540, 720, 900, 1080};

int step_of(int height) {
  const auto at = std::ranges::lower_bound(kWorkHeights, height);
  return static_cast<int>(at - kWorkHeights.begin());
}

// La plus grande résolution de travail que la tranche tient.
constexpr std::array<int, kTierCount> kWorkCap{720, 540, 720, 720, 1080};
// Le plus grand multiplicateur ×k du mode auto.
constexpr std::array<int, kTierCount> kFactorCap{0, 2, 3, 4, 4};

}  // namespace

std::string_view key(Preset preset) {
  return kPresets[static_cast<std::size_t>(preset)].key;
}
std::string_view label(Preset preset) {
  return kPresets[static_cast<std::size_t>(preset)].label;
}
std::optional<Preset> preset_from_key(std::string_view text) {
  for (const PresetName& entry : kPresets) {
    if (entry.key == text) return entry.preset;
  }
  return std::nullopt;
}
std::string_view key(Priority priority) {
  return kPriorities[static_cast<std::size_t>(priority)].key;
}
std::string_view label(Priority priority) {
  return kPriorities[static_cast<std::size_t>(priority)].label;
}
std::optional<Priority> priority_from_key(std::string_view text) {
  for (const PriorityName& entry : kPriorities) {
    if (entry.key == text) return entry.priority;
  }
  return std::nullopt;
}

std::optional<Tier> chain_level(Preset preset, Tier detected) {
  switch (preset) {
    case Preset::Auto:
      return detected;
    case Preset::Economy:
      return Tier::Economy;
    case Preset::Fast:
      return Tier::Light;
    case Preset::Normal:
      return Tier::Normal;
    case Preset::Quality:
      return Tier::Fast;
    case Preset::Ultra:
      return Tier::Ultra;
    case Preset::Custom:
      return std::nullopt;
  }
  return std::nullopt;
}

PlaySettings apply_preset(const PlaySettings& settings, const Machine& machine,
                          const core::SourceInfo& source) {
  const std::optional<Tier> level = chain_level(settings.preset, machine.tier);
  if (!level) return settings;
  const auto at = static_cast<std::size_t>(*level);
  PlaySettings out = settings;
  const int display_height = machine.display_height > 0 ? machine.display_height : 1080;
  const double display_hz = machine.display_hz > 0.0 ? machine.display_hz : 60.0;

  // Résolution : SPAN double, la sortie vise l'écran. Écran 1080p : 540p ; 1440p : 720p ; 4K : 720p
  // (puis la mise à l'échelle finale), jamais plus que la tranche ne tient, jamais plus que la source.
  const int ideal = std::clamp(display_height / 2, kWorkHeights.front(), kWorkHeights.back());
  int work = std::min(ideal, kWorkCap[at]);
  int step = step_of(work);
  if (settings.priority == Priority::Fluidity) step = std::max(0, step - 1);
  if (settings.priority == Priority::Image) step = std::min(step + 1, step_of(kWorkCap[at]) + 1);
  step = std::min(step, static_cast<int>(kWorkHeights.size()) - 1);
  work = kWorkHeights[static_cast<std::size_t>(step)];
  if (source.height > 0) work = std::min(work, std::max(source.height, kWorkHeights.front()));

  // Tranche Économe : on ne réduit ni n'agrandit, la chaîne est la source telle quelle.
  out.upscale = *level != Tier::Economy;
  out.work_height = *level == Tier::Economy ? kWorkHeights.back() : work;
  out.downscale = Downscale::Mitchell;
  out.model.clear();  // SPAN Sharp (performance), le modèle du lecteur
  out.span_order = SpanOrder::Auto;

  // Interpolation : multiplicateur en Auto ; une source déjà à la cadence de l'écran n'est pas interpolée.
  const double fps = source.fps > 0.0 ? source.fps : 24.0;
  int cap = kFactorCap[at];
  if (cap > 0 && settings.priority == Priority::Fluidity) cap = std::min(cap + 1, 6);
  if (cap > 0 && settings.priority == Priority::Image) cap = std::max(2, cap - 1);
  const int wanted = static_cast<int>(std::floor(display_hz / fps + 1e-6));
  const int factor = std::min(cap, wanted);
  out.rife_auto = factor >= 2;
  out.rife_factor = factor >= 2 ? factor : 0;
  out.rife_factor_den = 1;
  out.keep_poses = true;  // mesuré A/B (4070 Ti, ×3) : tient mieux, qualité équivalente
  out.gmfss_precision = PrecisionTier::Balanced;
  out.span_precision = PrecisionTier::Balanced;
  out.motion_precision = PrecisionTier::Balanced;
  out.rife_model.clear();
  switch (*level) {
    case Tier::Economy:
      out.rife_factor = 0;
      out.rife_auto = false;
      break;
    case Tier::Light:
    case Tier::Normal:
      out.motion_engine = MotionEngine::Rife;
      out.rife_model = "rife_v4.22_lite.onnx";
      break;
    case Tier::Fast:
      out.motion_engine = MotionEngine::DistilDrba;
      break;
    case Tier::Ultra:
      out.motion_engine = MotionEngine::GmfssUnion;
      out.gmfss_precision =
          settings.priority == Priority::Image ? PrecisionTier::Quality : PrecisionTier::Balanced;
      break;
  }
  return out;
}

bool preset_chain_differs(const PlaySettings& a, const PlaySettings& b) {
  return a.upscale != b.upscale || a.work_height != b.work_height || a.downscale != b.downscale ||
         a.model != b.model || a.span_order != b.span_order || a.rife_auto != b.rife_auto ||
         a.rife_factor != b.rife_factor || a.rife_factor_den != b.rife_factor_den ||
         a.keep_poses != b.keep_poses || a.gmfss_precision != b.gmfss_precision ||
         a.span_precision != b.span_precision || a.motion_precision != b.motion_precision ||
         a.rife_model != b.rife_model || a.motion_engine != b.motion_engine;
}

}  // namespace ajp::app::play
