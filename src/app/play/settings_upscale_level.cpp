#include "app/play/settings_upscale_level.h"

#include <algorithm>
#include <array>
#include <string>

#include <imgui.h>

#include "app/play/anime4k_choice.h"
#include "app/play/card_advice.h"
#include "app/play/i18n.h"
#include "app/play/model_choice.h"
#include "app/play/panel_style.h"

namespace kirari::app::play {

namespace {

struct LevelEntry {
  UpscaleLevel level;
  std::string_view model;
  std::string_view sharp;  // la variante Sharp1, ou le même modèle
  const char* name;
  const char* description;
};

constexpr std::string_view kPerformance = kSpanCatalogue[1];
constexpr std::string_view kBalanced = kSpanCatalogue[0];
constexpr std::string_view kSharpPerformance = kSpanCatalogue[3];
constexpr std::string_view kSharpBalanced = kSpanCatalogue[2];

constexpr std::array<LevelEntry, 4> kLevels{{
    {UpscaleLevel::VeryLow, kAnime4kFast, kAnime4kFast, "Very low",
     "Anime4K, shaders without a network: runs on any graphics card, even integrated. Quality impact: "
     "modest gain over plain scaling."},
    {UpscaleLevel::Low, kLightSpanModel, kLightSpanModel, "Low",
     "ArtCNN, a light network for small graphics cards."},
    {UpscaleLevel::Medium, kPerformance, kSharpPerformance, "Medium",
     "AnimeJaNai Performance (SPAN): firm, clean lines; keeps up on most dedicated cards."},
    {UpscaleLevel::High, kBalanced, kSharpBalanced, "High",
     "AnimeJaNai Balanced (SPAN): finer detail, for strong graphics cards."},
}};

bool present(const LevelEntry& entry, const SettingsChoices& choices) {
  if (anime4k_size(entry.model)) return true;
  return std::ranges::find(choices.models, entry.model) != choices.models.end();
}

}  // namespace

bool is_sharp(std::string_view model) {
  return model == kSharpPerformance || model == kSharpBalanced;
}

std::optional<UpscaleLevel> upscale_level(std::string_view model) {
  if (model == kAnime4kQuality) return UpscaleLevel::VeryLow;
  if (model == kLightestSpanModel) return UpscaleLevel::Low;
  for (const LevelEntry& entry : kLevels) {
    if (model == entry.model || model == entry.sharp) return entry.level;
  }
  return std::nullopt;
}

std::string_view upscale_level_model(UpscaleLevel level, bool sharp) {
  const LevelEntry& entry = kLevels.at(static_cast<std::size_t>(level));
  return sharp ? entry.sharp : entry.model;
}

void upscale_combo(PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  const std::optional<UpscaleLevel> now = upscale_level(current);
  const auto text = [&](const LevelEntry& entry) {
    std::string shown = i18n::tr(entry.name);
    if (entry.model == choices.default_model) shown += i18n::tr(" (recommended)");
    return shown;
  };
  std::string shown = model_label(current);
  for (const LevelEntry& entry : kLevels) {
    if (now == entry.level) shown = text(entry);
  }
  // Sans modèle (rien d'installé), l'agrandissement ne tourne pas : Désactivé, en attendant un niveau.
  if (!d.upscale || current.empty()) shown = i18n::tr("Off");
  if (panel::combo(i18n::tr("Upscaling"), shown.c_str())) {
    if (ImGui::Selectable(i18n::tr("Off"), !d.upscale)) d.upscale = false;
    for (const LevelEntry& entry : kLevels) {
      if (!present(entry, choices)) continue;
      if (ImGui::Selectable(text(entry).c_str(), d.upscale && now == entry.level)) {
        d.upscale = true;
        // Le niveau garde la saveur Sharp1 du modèle en cours, quand il en a une.
        const std::string model(is_sharp(current) ? entry.sharp : entry.model);
        d.model = model == choices.default_model ? std::string() : model;
        // Le FP8 n'existe que pour SPAN d'AnimeJaNai (model_choice) : ailleurs, Équilibre.
        if (d.span_precision == PrecisionTier::Performance && entry.level < UpscaleLevel::Medium)
          d.span_precision = PrecisionTier::Balanced;
      }
      panel::tip(i18n::tr(entry.name), i18n::tr(entry.description));
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Upscaling"),
             std::string(i18n::tr(
                 "Redraws the video at twice its size with a model trained on anime: sharper lines and less "
                 "blur on a big screen. Higher levels keep finer detail but need a stronger card; it is the "
                 "heaviest step, lower it first if the video stutters. Very low is Anime4K, Low is ArtCNN, "
                 "Medium and High are AnimeJaNai Performance and Balanced (SPAN).")) +
                 " " + advice_line(choices, model_advice),
             panel::Apply::Reload);
}

bool is_animejanai(std::string_view model) {
  return model == kPerformance || model == kBalanced || is_sharp(model);
}

void sharp_checkbox(PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  const std::optional<UpscaleLevel> level = upscale_level(current);
  ImGui::BeginDisabled(!upscale_on(d, choices) || !is_animejanai(current));
  bool sharp = is_sharp(current);
  if (ImGui::Checkbox(i18n::tr("Sharp lines (Sharp1)"), &sharp) && level) {
    const std::string model(upscale_level_model(*level, sharp));
    d.model = model == choices.default_model ? std::string() : model;
  }
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Sharp lines (Sharp1)"),
             i18n::tr("The Sharp1 version of AnimeJaNai draws crisper lines, a little harsher. Its engine is "
                      "built once the first time it is used, which takes a few minutes. Quality impact: a "
                      "matter of taste. Recommended: off. Medium and High only."),
             panel::Apply::Reload);
}

bool upscale_on(const PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  return d.upscale && !current.empty();
}

bool fp8_in_use(const PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  return d.upscale && is_animejanai(current) && choices.span_tiers.performance &&
         d.span_precision == PrecisionTier::Performance;
}

void fp8_checkbox(PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  const bool offered = choices.span_tiers.performance && is_animejanai(current);
  ImGui::BeginDisabled(!upscale_on(d, choices) || !offered);
  bool fp8 = offered && d.span_precision == PrecisionTier::Performance;
  if (ImGui::Checkbox(i18n::tr("FP8"), &fp8))
    d.span_precision = fp8 ? PrecisionTier::Performance : PrecisionTier::Balanced;
  ImGui::EndDisabled();
  panel::tip(i18n::tr("FP8"),
             i18n::tr("AnimeJaNai computes in 8-bit precision: prepared once the first time, then faster. "
                      "Speed: about +37 %. Quality impact: imperceptible. Recommended: on when offered. "
                      "AnimeJaNai on an NVIDIA RTX 40 or newer only."),
             panel::Apply::Reload);
}

}  // namespace kirari::app::play
