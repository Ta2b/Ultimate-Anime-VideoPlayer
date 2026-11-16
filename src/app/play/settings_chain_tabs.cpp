#include "app/play/settings_chain_tabs.h"

#include <algorithm>
#include <array>
#include <format>
#include <string>

#include <imgui.h>

#include "app/play/anime4k_choice.h"
#include "app/play/card_advice.h"
#include "app/play/model_choice.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

using panel::note;
using panel::section;

namespace {

struct HeightChoice {
  int lines;
  const char* label;
};
constexpr std::array<HeightChoice, 6> kHeights{{
    {0, "Source (no downscale)"},  // clé i18n
    {1080, "1080p"},               // les niveaux Qualité et Ultra, et la calibration, la posent (B23-03)
    {900, "900p"},
    {720, "720p"},
    {540, "540p"},
    {480, "480p"},
}};

// Le modèle SPAN, en FP16 et, quand la carte le permet, en FP8.
void model_combo(PlaySettings& d, const SettingsChoices& choices) {
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  // Chaque modèle en FP16 et, sur une carte qui calcule le SPAN FP8 (RTX 40 et plus), en FP8 : le FP8 se
  // prépare à la prochaine ouverture d'une vidéo, sur l'écran de chargement, comme un moteur (chain_setup).
  const bool fp8_offered = choices.span_tiers.performance;
  const bool fp8_now = fp8_offered && d.span_precision == PrecisionTier::Performance;
  const auto entry = [](const std::string& model, bool fp8) {
    return fp8 ? model_label(model) + " · FP8" : model_label(model);
  };
  if (ImGui::BeginCombo(panel::left(i18n::tr("Model")).c_str(), entry(current, fp8_now).c_str())) {
    for (const std::string& model : choices.models) {
      for (const bool fp8 : {false, true}) {
        if (fp8 && !fp8_offered) continue;
        const bool chosen = model == current && fp8 == fp8_now;
        ImGui::PushID(fp8 ? 1 : 0);
        if (ImGui::Selectable(entry(model, fp8).c_str(), chosen)) {
          d.model = model == choices.default_model ? std::string() : model;
          if (fp8) {
            d.span_precision = PrecisionTier::Performance;
          } else if (d.span_precision == PrecisionTier::Performance) {
            d.span_precision = PrecisionTier::Balanced;
          }
        }
        ImGui::PopID();
        if (chosen) ImGui::SetItemDefaultFocus();
      }
    }
    // Anime4K, sans fichier ni moteur : toujours proposé, après les modèles ONNX (anime4k_choice.h).
    for (const std::string_view shader : kAnime4kModels) {
      const bool chosen = current == shader;
      if (ImGui::Selectable(model_label(std::string(shader)).c_str(), chosen)) {
        d.model = std::string(shader);
        if (d.span_precision == PrecisionTier::Performance) d.span_precision = PrecisionTier::Balanced;
      }
      if (chosen) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Model"),
             fp8_offered ? std::string(i18n::tr(
                               "Which network redraws the picture. Performance keeps up on most cards. "
                               "Balanced keeps finer detail if your card can afford it. Sharp1 draws crisper "
                               "lines, a little harsher. ArtCNN and Anime4K are for weak graphics cards. "
                               "FP8: prepared once, then over +30 % speed for an imperceptible loss.")) +
                               " " + advice_line(choices, model_advice)
                         : std::string(i18n::tr(
                               "Which network redraws the picture. Performance keeps up on most cards. "
                               "Balanced keeps finer detail if your card can afford it. Sharp1 draws crisper "
                               "lines, a little harsher. ArtCNN and Anime4K are for weak graphics cards.")) +
                               " " + advice_line(choices, model_advice),
             panel::Apply::Reload);
  if (anime4k_size(current))
    note(i18n::tr("Anime4K: SDR sources only; an HDR source is upscaled by the display."));
}

// La sortie de la chaîne sous le choix de la résolution de travail : doublée par SPAN, ou telle quelle, et la
// réduction faite pour rester en 4K (chain_work_size).
void output_notes(const PlaySettings& d, const SettingsChoices& choices, int work) {
  const bool doubles = span_runs(d, choices.source_height);
  note(doubles ? i18n::tr_format("Output: {} (SPAN ×2).", resolution_text(work * 2))
               : i18n::tr_format("Output: {} (no upscaling).", resolution_text(work)));
  const int asked =
      d.work_height > 0 ? std::min(d.work_height, choices.source_height) : choices.source_height;
  if (doubles && work < asked)
    note(i18n::tr_format("Reduced to {} first, so the output stays within 4K.", resolution_text(work)));
}

}  // namespace

// Le sélecteur de cran de précision : « Performance · FP8 », « Équilibre · FP16 », « Qualité · FP32 ». Il ne
// montre que les crans que ce poste rend pour ce réseau (Équilibre l'est toujours) et rien du tout quand
// Équilibre est le seul : un cran qui n'est pas une option n'apparaît pas, grisé ni « absent ». Un cran
// choisi dans play.toml puis devenu indisponible se lit « Équilibre » : la chaîne le prend.
bool precision_combo(const char* title, PrecisionTier& tier, const TierSupport& support) {
  if (!support.performance && !support.quality) return false;
  const auto text = [](PrecisionTier t) {
    return std::format("{} · {}", i18n::tr_view(label(t)), precision_text(t));
  };
  const std::string shown = text(support.effective(tier));
  bool changed = false;
  if (ImGui::BeginCombo(panel::left(title).c_str(), shown.c_str())) {
    for (int i = 0; i < kPrecisionTierCount; ++i) {
      const auto candidate = static_cast<PrecisionTier>(i);
      if (!support.has(candidate)) continue;
      if (ImGui::Selectable(text(candidate).c_str(), candidate == support.effective(tier)) &&
          candidate != tier) {
        tier = candidate;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(title,
             i18n::tr("Quality computes in full precision (FP32) when the model has such a version; "
                      "otherwise it runs like Balanced, as with the current SPAN models. Only useful to "
                      "compare pictures. Speed: up to 2 to 3× slower, no change with the current SPAN "
                      "models. Quality impact: imperceptible. Recommended: Balanced."),
             panel::Apply::Reload);
  return changed;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void image_tab(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Upscaling"));
  const bool anime4k = anime4k_size(d.model).has_value();
  const char* toggle = anime4k ? i18n::tr("Anime4K ×2") : i18n::tr("SPAN ×2 (AnimeJaNai)");
  ImGui::Checkbox(toggle, &d.upscale);
  panel::tip(toggle,
             std::string(i18n::tr(
                 "Redraws the video at twice its size with a model trained on anime: sharper lines and less "
                 "blur on a big screen. It is the heaviest step: turn it off first if the video stutters. "
                 "Speed: the heaviest step of the chain. Quality impact: high gain.")) +
                 " " + advice_line(choices, upscale_advice),
             panel::Apply::Reload);
  ImGui::BeginDisabled(!d.upscale);
  model_combo(d, choices);
  ImGui::Checkbox(i18n::tr("Restore line art"), &d.restore_lines);
  panel::tip(
      i18n::tr("Restore line art"),
      i18n::tr("Videos sharper than the working resolution (a 1080p video with 720p work) get cleaner, "
               "firmer lines, redrawn by the upscaler. Off: such a video is shown as is, with its grain and "
               "textures, and the card rests. Speed: about 2× less work than upscaling such a video at full "
               "size. Quality impact: modest (cleaner lines, less grain). Recommended: on."),
      panel::Apply::Reload);
  ImGui::EndDisabled();

  section(i18n::tr("Resolution"));
  // La source sans réduction, avec sa hauteur quand on la connaît : « 1080p », pas « 4K », que la sortie
  // n'atteint qu'agrandie par SPAN.
  const auto label_of = [&](const HeightChoice& choice) {
    if (choice.lines == 0 && choices.source_height > 0)
      return i18n::tr_format("Source, no downscale ({})", resolution_text(choices.source_height));
    return std::string(i18n::tr(choice.label));
  };
  const auto height = std::ranges::find(kHeights, d.work_height, &HeightChoice::lines);
  const std::string height_label =
      height != kHeights.end() ? label_of(*height) : resolution_text(d.work_height);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Working resolution")).c_str(), height_label.c_str())) {
    for (const HeightChoice& choice : kHeights) {
      if (ImGui::Selectable(label_of(choice).c_str(), choice.lines == d.work_height))
        d.work_height = choice.lines;
    }
    // Une hauteur de play.toml hors de la liste reste visible et choisie, au lieu de disparaître (B23-03).
    if (height == kHeights.end()) ImGui::Selectable(height_label.c_str(), true);
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("Working resolution"),
      std::string(i18n::tr("The size the upscaler works at, then doubles (720p gives 1440p). Higher keeps "
                           "more detail but asks much more of the card: lower it first if the video "
                           "stutters. Speed: 1080p costs about 2× 720p. Quality impact: modest.")) +
          " " + advice_line(choices, work_height_advice),
      panel::Apply::Reload);
  // La taille que la chaîne joue vraiment (chain_work_size) : la hauteur choisie, ou moins pour rester en 4K.
  const int work = chain_work_size(d, choices.source_width, choices.source_height).height;
  if (choices.source_height > 0) output_notes(d, choices, work);
  ImGui::SliderInt(panel::left(i18n::tr("Sharpness")).c_str(), &d.sharpness, limits::kSharpness.lo,
                   limits::kSharpness.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  panel::live_preview();
  panel::tip(i18n::tr("Sharpness"),
             i18n::tr("Makes lines crisper after upscaling, without halos. Too high makes the picture look "
                      "harsh. Speed: imperceptible cost. Recommended: 0 %, to taste."));
}

void upscale_advanced(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Filter and precision"));
  const bool reduces =
      choices.source_height <= 0 ||
      chain_work_size(d, choices.source_width, choices.source_height).height < choices.source_height;
  ImGui::BeginDisabled(!reduces);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Downscaler")).c_str(),
                        std::string(label(d.downscale)).c_str())) {
    for (int i = 0; i < kDownscaleCount; ++i) {
      const auto filter = static_cast<Downscale>(i);
      std::string text(label(filter));
      if (const auto mark = badge_of(filter)) text += " · " + std::string(badge_text(*mark));
      if (ImGui::Selectable(text.c_str(), filter == d.downscale)) d.downscale = filter;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Downscaler"),
             i18n::tr("How the video is shrunk before upscaling. Hermite keeps lines clean, with no halos; "
                      "Spline36 keeps a little more detail. Quality impact: minor. Recommended: Hermite."));
  if (choices.span_tiers.quality && !anime4k_size(d.model)) {
    TierSupport tiers = choices.span_tiers;
    // Le FP8 se choisit dans la liste des modèles ; une fois choisi il se lit ici aussi, au lieu d'un faux
    // « Équilibre · FP16 » (B23-02).
    tiers.performance = tiers.performance && d.span_precision == PrecisionTier::Performance;
    ImGui::BeginDisabled(!d.upscale);
    precision_combo(i18n::tr("SPAN precision"), d.span_precision, tiers);
    ImGui::EndDisabled();
  }
}

}  // namespace ajp::app::play
