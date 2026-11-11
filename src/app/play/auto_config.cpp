#include "app/play/auto_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "app/play/i18n.h"
#include "app/play/model_choice.h"

namespace ajp::app::play {
namespace {

struct PresetName {
  Preset preset;
  std::string_view key;
  std::string_view label;
};
constexpr std::array<PresetName, kPresetCount> kPresets{{
    {Preset::Auto, "auto", "Auto"},
    {Preset::Economy, "economy", "Economy"},
    {Preset::Fast, "fast", "Fast"},
    {Preset::Normal, "normal", "Normal"},
    {Preset::Quality, "quality", "Quality"},
    {Preset::Ultra, "ultra", "Ultra"},
    {Preset::Custom, "custom", "Custom"},
}};

struct PriorityName {
  Priority priority;
  std::string_view key;
  std::string_view label;
};
constexpr std::array<PriorityName, kPriorityCount> kPriorities{{
    {Priority::Fluidity, "fluidity", "Smoothness"},
    {Priority::Balance, "balance", "Balance"},
    {Priority::Image, "image", "Image"},
}};

// Les résolutions de travail proposées par le menu, de la plus petite à la plus grande.
constexpr std::array<int, 5> kWorkHeights{480, 540, 720, 900, 1080};

int step_of(int height) {
  const auto at = std::ranges::lower_bound(kWorkHeights, height);
  return static_cast<int>(at - kWorkHeights.begin());
}

// Le palier au plus égal à height (le plus petit si height est en dessous) : la calibration arrondit vers le
// bas, l'Auto estimé aussi (A37-06).
int step_below(int height) {
  const auto at = std::ranges::upper_bound(kWorkHeights, height);
  return std::max(0, static_cast<int>(at - kWorkHeights.begin()) - 1);
}

// La hauteur de l'image dans l'écran, pas celle de l'écran : une vidéo 16:9 sur un écran 1920 × 1200 n'en
// occupe que 1080 lignes (A37-06). Largeur ou source inconnues : toute la hauteur.
int image_height(int display_height, int display_width, const core::SourceInfo& source) {
  if (display_height <= 0 || display_width <= 0 || source.width <= 0 || source.height <= 0)
    return display_height;
  const auto fitted =
      static_cast<int>(static_cast<std::int64_t>(display_width) * source.height / source.width);
  return std::min(display_height, fitted);
}

// La plus grande résolution de travail que la tranche tient. Qualité et Ultra tiennent la 1080p (A49-02) :
// le b8f64 FP16 y prend 14,25 ms sur une RTX 4070 Ti pour 41,7 ms par image source.
constexpr std::array<int, kTierCount> kWorkCap{720, 540, 720, 1080, 1080};
// Pas de résolution par rapport à l'idéal de l'écran (la moitié de sa hauteur), pour Léger et Normal : chaque
// niveau travaille plus haut que le précédent (480p, 540p sur un écran 1080p). Qualité et Ultra ne suivent
// pas l'écran : voir keeps_source.
constexpr std::array<int, kTierCount> kWorkOffset{0, -1, 0, 0, 0};
// Qualité et Ultra gardent la source jusqu'à 1080p sur tout écran (A49-02, compromis C2 de la synthèse des
// audits) : réduire une source 1080p à la moitié de l'écran jette plus de la moitié de ses pixels avant le
// réseau, sans mesure qui le justifie. Sur un écran plus petit que la sortie de SPAN, libplacebo réduit.
constexpr bool keeps_source(Tier level) {
  return level == Tier::Fast || level == Tier::Ultra;
}
// Le plus grand multiplicateur ×k du mode auto.
constexpr std::array<int, kTierCount> kFactorCap{0, 2, 3, 4, 4};

// Le modèle du niveau (RIFE 4.22 lite, ou 4.25 pour Qualité et Ultra), ou en Auto le modèle plus fin
// que la machine a exploré quand il tient aussi.
void use_rife(PlaySettings& out, const PlaySettings& settings, const Machine& machine,
              std::string_view model) {
  out.motion_engine = MotionEngine::Rife;
  out.rife_model = model;
  if (settings.preset == Preset::Auto && !machine.explored_rife_model.empty())
    out.rife_model = machine.explored_rife_model;
}

// Les modèles légers des deux tranches du bas, une fois admis (kLightSpanInAuto) : le plus léger (C4F16, 16
// canaux) pour la plus faible, C4F32 au-dessus. Le banc peut les échanger ou les égaler s'il mesure autre
// chose.
constexpr std::string_view kEconomySpanModel = kLightestSpanModel;
constexpr std::string_view kLightTierSpanModel = kLightSpanModel;

// SPAN de la tranche : rien en Économe (sauf le modèle léger s'il est admis, P6.7), le modèle du niveau
// au-dessus (level_span_model) ; les modèles légers, s'ils sont admis, dans les deux tranches du bas.
void set_span(PlaySettings& out, Tier level, int work) {
  const bool economy = level == Tier::Economy;
  out.upscale = !economy || kLightSpanInAuto;
  out.work_height = economy && !kLightSpanInAuto ? 0 : work;
  out.model = std::string(level_span_model(level));
  if (kLightSpanInAuto && economy) out.model = std::string(kEconomySpanModel);
  if (kLightSpanInAuto && level == Tier::Light) out.model = std::string(kLightTierSpanModel);
}

// La résolution de travail du niveau, avant l'exploration et la source. Qualité et Ultra : la source telle
// quelle jusqu'à 1080p, sur tout écran (keeps_source) ; la priorité Fluidité descend à 720p. Léger et Normal
// : SPAN double, la sortie vise l'écran (écran 1080p : 540p ; 1440p et 4K : 720p), jamais plus que la tranche
// ne tient ; la calibration, qui mesure, monte plus haut quand la carte tient. Écran inconnu (0) : 1080p.
int level_work_height(Tier level, Priority priority, int display_height) {
  const auto at = static_cast<std::size_t>(level);
  if (keeps_source(level)) return priority == Priority::Fluidity ? 720 : kWorkCap[at];
  const int ideal =
      std::clamp((display_height > 0 ? display_height : 1080) / 2, kWorkHeights.front(), kWorkHeights.back());
  int step = std::clamp(step_below(ideal) + kWorkOffset[at], 0, step_of(kWorkCap[at]));
  if (priority == Priority::Fluidity) step = std::max(0, step - 1);
  if (priority == Priority::Image) step = std::min(step + 1, step_of(kWorkCap[at]) + 1);
  step = std::min(step, static_cast<int>(kWorkHeights.size()) - 1);
  return kWorkHeights[static_cast<std::size_t>(step)];
}

// Le multiplicateur que la fréquence demande, arrondi avec 1 % de tolérance (P3.14) : 59,94 Hz sur une source
// à 30 i/s vaut ×2, pas ×1.
int wanted_factor(double display_hz, double fps) {
  return static_cast<int>(std::floor((display_hz / fps) + 0.01));
}

}  // namespace

std::string_view level_span_model(Tier level) {
  return keeps_source(level) ? kQualitySpanModel : std::string_view{};
}
std::string_view level_rife_model(Tier level) {
  return keeps_source(level) ? kQualityRifeModel : kLightRifeModel;
}

std::string_view key(Preset preset) {
  return kPresets[static_cast<std::size_t>(preset)].key;
}
std::string_view label(Preset preset) {
  return i18n::tr_view(kPresets[static_cast<std::size_t>(preset)].label);
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
  return i18n::tr_view(kPriorities[static_cast<std::size_t>(priority)].label);
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
  std::optional<Tier> level = chain_level(settings.preset, machine.tier);
  if (!level) return settings;
  // Une carte estimée Économe que la calibration a vue tenir SPAN suit la mesure, pas l'estimation.
  if (settings.preset == Preset::Auto && machine.calibrated && *level == Tier::Economy) level = Tier::Light;
  const auto at = static_cast<std::size_t>(*level);
  PlaySettings out = settings;
  const double display_hz = machine.display_hz > 0.0 ? machine.display_hz : 60.0;

  int work = level_work_height(*level, settings.priority,
                               image_height(machine.display_height, machine.display_width, source));
  if (settings.preset == Preset::Auto && machine.explored_work_height > 0)
    work = std::max(work,
                    machine.explored_work_height);  // l'exploration a trouvé que la machine tient plus haut
  if (source.height > 0) work = std::min(work, std::max(source.height, kWorkHeights.front()));

  // Tranche Économe : on ne réduit ni n'agrandit, la chaîne est la source telle quelle (P3.20 : une 4K reste
  // en 4K, 0 ne réduit rien).
  set_span(out, *level, work);
  out.downscale = Downscale::Hermite;  // net sur les traits, sans rebond, quatre points comme Mitchell
  out.span_order = SpanOrder::Auto;

  // Interpolation : multiplicateur en Auto ; une source déjà à la cadence de l'écran n'est pas interpolée.
  const double fps = source.fps > 0.0 ? source.fps : 24.0;
  int cap = kFactorCap[at];
  if (cap > 0 && settings.priority == Priority::Fluidity) cap = std::min(cap + 1, 6);
  if (cap > 0 && settings.priority == Priority::Image) cap = std::max(2, cap - 1);
  const int factor = std::min(cap, wanted_factor(display_hz, fps));
  out.rife_auto = factor >= 2;
  out.rife_factor = factor >= 2 ? factor : 0;
  out.rife_factor_den = 1;
  out.keep_poses = true;  // mesuré A/B (4070 Ti, ×3) : tient mieux, qualité équivalente
  // SPAN en FP16 : le FP8 se choisit à la main dans la liste des modèles (« … · FP8 ») tant que sa qualité et
  // son gain sur le FP16 ne sont pas jugés à l'œil et mesurés ; les préréglages ne le posent plus.
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
      use_rife(out, settings, machine, level_rife_model(*level));
      break;
    case Tier::Fast:
    case Tier::Ultra:
      // DistilDRBA y rentre quand sa porte de qualité est passée (kDistilDrbaInAuto).
      if (kDistilDrbaInAuto) {
        out.motion_engine = MotionEngine::DistilDrba;
      } else {
        use_rife(out, settings, machine, level_rife_model(*level));
      }
      break;
  }
  // Auto calibré : la chaîne mesurée (SPAN, sa hauteur, son modèle en FP16 ; RIFE en mode Rapide ou rien,
  // avec le modèle mesuré), adaptée à la source et à la cadence de cette vidéo comme le reste d'Auto. Le FP8
  // n'y est plus posé (A49-03, ADR 0007 suspendue).
  if (settings.preset == Preset::Auto && machine.calibrated) {
    const CalibratedChain& cal = *machine.calibrated;
    out.upscale = cal.span;
    if (cal.span && cal.height > 0)
      out.work_height = source.height > 0
                            ? std::min(cal.height, std::max(source.height, kWorkHeights.front()))
                            : cal.height;
    out.model = cal.span_model;
    out.span_precision = PrecisionTier::Balanced;
    out.motion_engine = MotionEngine::Rife;
    out.rife_model = cal.rife_model.empty() ? std::string(kLightRifeModel) : cal.rife_model;
    out.auto_free = cal.rife;
    if (!cal.rife) {
      out.rife_auto = false;
      out.rife_factor = 0;
    }
  }
  return out;
}

bool preset_chain_differs(const PlaySettings& a, const PlaySettings& b, bool calibrated) {
  if (calibrated && a.auto_free != b.auto_free) return true;
  return a.upscale != b.upscale || a.work_height != b.work_height || a.downscale != b.downscale ||
         a.model != b.model || a.span_order != b.span_order || a.rife_auto != b.rife_auto ||
         a.rife_factor != b.rife_factor || a.rife_factor_den != b.rife_factor_den ||
         a.keep_poses != b.keep_poses || a.span_precision != b.span_precision ||
         a.motion_precision != b.motion_precision || a.rife_model != b.rife_model ||
         a.motion_engine != b.motion_engine;
}

bool display_moves_chain(Preset preset, const DisplayMove& move) {
  if (preset == Preset::Custom) return false;
  if (move.opened_height > 0 && move.now_height > 0 && move.opened_height != move.now_height) return true;
  if (move.opened_hz <= 0.0 || move.now_hz <= 0.0 || move.source_fps <= 0.0) return false;
  return wanted_factor(move.opened_hz, move.source_fps) != wanted_factor(move.now_hz, move.source_fps);
}

}  // namespace ajp::app::play
