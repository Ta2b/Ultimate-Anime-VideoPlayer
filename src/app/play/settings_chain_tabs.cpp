#include "app/play/settings_chain_tabs.h"

#include <algorithm>
#include <array>
#include <format>
#include <string>

#include <imgui.h>

#include "app/play/anime4k_choice.h"
#include "app/play/drba_model.h"
#include "app/play/model_choice.h"
#include "app/play/panel_style.h"
#include "app/play/visible_error.h"
#include "core/auto_rate.h"

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
  if (ImGui::BeginCombo(title, shown.c_str())) {
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
  return changed;
}

std::string motion_description(const InterpChoice& choice) {
  switch (choice.level) {
    case InterpLevel::VeryFast:
      return i18n::tr(
          "DistilDRBA: keeps characters animated on twos on their rhythm. One frame of delay. Its speed and "
          "image quality against RIFE are not compared yet, so Auto does not use it.");
    case InterpLevel::Fast:
      return i18n::tr("RIFE 4.22 lite: light and clean, for most graphics cards.");
    case InterpLevel::Balanced:
      return i18n::tr("RIFE 4.25: cleaner in-between frames, for recent graphics cards.");
    case InterpLevel::System:
      return i18n::tr(
          "Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend "
          "only; RIFE is used elsewhere.");
  }
  return {};
}

// L'état réel de DistilDRBA dans la chaîne ouverte (A23-07) : fusionné ou entier, variante mouvement, et
// pourquoi il n'est pas fusionné quand la case est cochée.
void drba_state_notes(const PlaySettings& d, const SettingsChoices& choices) {
  const OpenedChain* chain = choices.chain();
  if (chain == nullptr || !chain->drba) return;
  note(chain->drba_fused ? i18n::tr("DistilDRBA is running: fused engines.")
                         : i18n::tr("DistilDRBA is running: whole network."));
  if (chain->drba_motion) note(i18n::tr("Motion variant: one SPAN pass per source frame."));
  if (chain->drba_fused || !d.drba_fused) return;
  if (!chain->drba_fused_capable) {
    note(i18n::tr("Not fused: this engine cannot run it fused."));
  } else if (d.motion_precision != PrecisionTier::Balanced) {
    note(i18n::tr("Not fused: fused engines need Balanced precision."));
  } else if (chain->drba_fused_refused) {
    note(i18n::tr_format("Not fused: the fused engines were refused ({}); details in the log.",
                         visible_error(Refusal{*chain->drba_fused_refused, ""})));
  } else {
    note(i18n::tr("Not fused: the fused engines were refused (see the log)."));
  }
}

// RIFE fusionné demandé puis refusé (A03-21) : la case reste cochée, le panneau dit ce qui tourne.
void rife_state_notes(const PlaySettings& d, const SettingsChoices& choices) {
  const OpenedChain* chain = choices.chain();
  if (chain == nullptr || !d.rife_fused || !chain->rife_fused_refused) return;
  note(i18n::tr_format("RIFE is not fused: the fused engines were refused ({}); details in the log.",
                       visible_error(Refusal{*chain->rife_fused_refused, ""})));
}

void fused_state_notes(const PlaySettings& d, const SettingsChoices& choices) {
  drba_state_notes(d, choices);
  rife_state_notes(d, choices);
}

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
  if (ImGui::BeginCombo(i18n::tr("Model"), entry(current, fp8_now).c_str())) {
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
  panel::tip(fp8_offered
                 ? i18n::tr("Models: Performance is fast, Balanced finer, Sharp1 crisp, ArtCNN the lightest "
                            "(C4F16 more than C4F32). FP8: prepared once on this PC, then about 37 % faster "
                            "per pass (14.7 to 9.2 ms at 1080p on an RTX 4070 Ti), at about 49 to 52 dB PSNR "
                            "against FP16.")
                 : i18n::tr("Models: Performance is fast, Balanced finer, Sharp1 crisp, ArtCNN the lightest "
                            "(C4F16 more than C4F32)."));
  if (anime4k_size(current))
    note(
        i18n::tr("Anime4K: shaders drawn by the display, with no engine to build, for small GPUs. Mode A, B "
                 "or C follows the source and the downscale."));
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void image_tab(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Upscaling"));
  const bool anime4k = anime4k_size(d.model).has_value();
  ImGui::Checkbox(anime4k ? i18n::tr("Anime4K ×2") : i18n::tr("SPAN ×2 (AnimeJaNai)"), &d.upscale);
  ImGui::BeginDisabled(!d.upscale);
  ImGui::Checkbox(i18n::tr("Restore line art (reduce, then upscale)"), &d.restore_lines);
  panel::tip(i18n::tr(
      "A source sharper than the working resolution is reduced, then SPAN redraws its lines: cleaner "
      "lines for a quarter of the work. Off: such a source keeps all its detail (grain, textures), "
      "without SPAN."));
  model_combo(d, choices);
  if (choices.span_tiers.quality && !anime4k) {
    TierSupport tiers = choices.span_tiers;
    // Le FP8 se choisit dans la liste des modèles ; une fois choisi il se lit ici aussi, au lieu d'un faux
    // « Équilibre · FP16 » (B23-02).
    tiers.performance = tiers.performance && d.span_precision == PrecisionTier::Performance;
    precision_combo(i18n::tr("SPAN precision"), d.span_precision, tiers);
    panel::tip(i18n::tr("Balanced is FP16; Quality is FP32. Reloads playback."));
  }
  ImGui::EndDisabled();

  section(i18n::tr("Resolution"));
  // La source sans réduction, avec sa hauteur quand on la connaît : « 1080 lignes », pas « 4K », que
  // la sortie n'atteint qu'agrandie par SPAN.
  const auto label_of = [&](const HeightChoice& choice) {
    if (choice.lines == 0 && choices.source_height > 0)
      return i18n::tr_format("Source, no downscale ({})", resolution_text(choices.source_height));
    return std::string(i18n::tr(choice.label));
  };
  const auto height = std::ranges::find(kHeights, d.work_height, &HeightChoice::lines);
  const std::string height_label =
      height != kHeights.end() ? label_of(*height) : resolution_text(d.work_height);
  if (ImGui::BeginCombo(i18n::tr("Working resolution"), height_label.c_str())) {
    for (const HeightChoice& choice : kHeights) {
      if (ImGui::Selectable(label_of(choice).c_str(), choice.lines == d.work_height))
        d.work_height = choice.lines;
    }
    // Une hauteur de play.toml hors de la liste reste visible et choisie, au lieu de disparaître (B23-03).
    if (height == kHeights.end()) ImGui::Selectable(height_label.c_str(), true);
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("The size the video is reduced to before SPAN, which doubles it: 720p gives 1440p."));
  const bool reduces = d.work_height > 0 && d.work_height < choices.source_height;
  if (choices.source_height > 0) {
    const int work = reduces ? d.work_height : choices.source_height;
    note(d.upscale ? i18n::tr_format("Output: {}, twice the working resolution (SPAN ×2).",
                                     resolution_text(work * 2))
                   : i18n::tr_format("Without upscaling, the output stays at the working resolution: {}.",
                                     resolution_text(work)));
  }
  if (reduces && panel::advanced()) {
    if (ImGui::BeginCombo(i18n::tr("Downscaler"), std::string(label(d.downscale)).c_str())) {
      for (int i = 0; i < kDownscaleCount; ++i) {
        const auto filter = static_cast<Downscale>(i);
        std::string text(label(filter));
        if (const auto mark = badge_of(filter)) text += " · " + std::string(badge_text(*mark));
        if (ImGui::Selectable(text.c_str(), filter == d.downscale)) d.downscale = filter;
      }
      ImGui::EndCombo();
    }
    panel::tip(
        i18n::tr("Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live."));
  } else if (!reduces && choices.source_height > 0) {
    note(i18n::tr("No downscale: the source goes into SPAN as is."));
  }
}

namespace {

// Le niveau d'interpolation : liste déroulante et note.
void level_combo(PlaySettings& d, const SettingsChoices& choices) {
  // Le niveau : un nom pour tous (Très rapide, Rapide, Équilibré), le modèle derrière nommé en mode avancé.
  const std::string rife_now = d.rife_model.empty() ? choices.default_rife_model : d.rife_model;
  const std::optional<InterpLevel> level_now = interp_level(d.motion_engine, rife_now);
  const auto level_name = [](InterpLevel level) -> std::string {
    switch (level) {
      case InterpLevel::VeryFast:
        return i18n::tr("DistilDRBA");
      case InterpLevel::Fast:
        return i18n::tr("Fast");
      case InterpLevel::Balanced:
        return i18n::tr("Balanced");
      case InterpLevel::System:
        return i18n::tr("System (macOS)");
    }
    return {};
  };
  const auto model_name = [](const InterpChoice& choice) {
    return choice.engine == MotionEngine::Rife ? rife_label(std::string(choice.rife_model))
                                               : std::string(label(choice.engine));
  };
  const auto level_text = [&](const InterpChoice& choice) {
    std::string text = level_name(choice.level);
    if (panel::advanced()) text += " · " + model_name(choice);
    if (choice.engine == MotionEngine::Rife && choice.rife_model == choices.default_rife_model)
      text += i18n::tr(" (recommended)");
    return text;
  };
  const auto present = [&](const InterpChoice& choice) {
    if (!offered(choice.engine)) return false;
    if (choice.engine == MotionEngine::DistilDrba) return choices.distil_drba_available;
    if (choice.engine == MotionEngine::Rife)
      return std::ranges::find(choices.rife_models, choice.rife_model) != choices.rife_models.end();
    return true;
  };
  std::string shown = level_now ? std::string() : rife_label(rife_now);
  if (d.motion_engine != MotionEngine::Rife && !level_now) shown = std::string(label(d.motion_engine));
  for (const InterpChoice& choice : kInterpLevels) {
    if (level_now == choice.level) shown = level_text(choice);
  }
  ImGui::BeginDisabled(!choices.rife_available);
  if (ImGui::BeginCombo(i18n::tr("Interpolation level"), shown.c_str())) {
    for (const InterpChoice& choice : kInterpLevels) {
      if (!present(choice)) continue;
      if (ImGui::Selectable(level_text(choice).c_str(), level_now == choice.level)) {
        d.motion_engine = choice.engine;
        if (choice.engine == MotionEngine::Rife)
          d.rife_model = choice.rife_model == choices.default_rife_model ? std::string()
                                                                         : std::string(choice.rife_model);
      }
      panel::tip(motion_description(choice));
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Fast and Balanced are RIFE models; DistilDRBA keeps poses. Their speed and image quality "
               "are not compared in this version. Applies after a short reload."));
}

// Smooth Motion est une couche de l'écran, absente du fichier, et du seul pilote NVIDIA. Le lecteur ne sait
// la demander qu'au pilote Linux (NVPRESENT_ENABLE_SMOOTH_MOTION) : sous Windows, elle se règle par profil
// dans l'application NVIDIA, et la case n'y ferait rien (B34-06).
void smooth_motion_section([[maybe_unused]] PlaySettings& d,
                           [[maybe_unused]] const SettingsChoices& choices) {
#if defined(__linux__)
  section(i18n::tr("Smooth Motion (NVIDIA driver)"));
  // Sans RIFE la lecture cale (16 images neuves en 8 s contre 192) : l'option reste éteinte.
  // Le choix reste tel quel, grisé : l'interpolation éteinte puis rallumée ne le perd plus, et le lancement
  // ne le prend que si RIFE tourne (player.cpp). Sans modèle d'interpolation, la case est grisée aussi
  // (B23-12).
  const bool without_rife = d.rife_factor < 2 || !choices.rife_available;
  ImGui::BeginDisabled(without_rife);
  ImGui::Checkbox(i18n::tr("Double the frames (NVIDIA driver)"), &d.smooth_motion);
  panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  ImGui::EndDisabled();
  note(i18n::tr(
      "The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames. Applies at next launch."));
  if (without_rife) note(i18n::tr("Greyed out: turn on RIFE at ×2 or more, or playback stalls."));
#endif
}

// La liste du plafond de la cadence Auto et la phrase qui dit où il s'arrête.
void auto_cap_row(PlaySettings& d, const SettingsChoices& choices) {
  // Le plafond de la cadence : le maximum de l'écran par défaut, ou une valeur plus basse pour ménager le
  // GPU.
  const auto cap_label = [](int cap) {
    return cap == 0 ? std::string(i18n::tr("Maximum (display)")) : std::format("{} i/s", cap);
  };
  if (ImGui::BeginCombo(i18n::tr("Auto limit"), cap_label(d.auto_cap).c_str())) {
    for (const int cap : {0, 120, 60}) {
      if (ImGui::Selectable(cap_label(cap).c_str(), d.auto_cap == cap)) d.auto_cap = cap;
    }
    // Un plafond de play.toml hors de cette liste reste visible et choisi (B23-05).
    if (d.auto_cap != 0 && d.auto_cap != 120 && d.auto_cap != 60)
      ImGui::Selectable(cap_label(d.auto_cap).c_str(), true);
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("The highest frame rate Auto aims for. It still drops by itself on heavy scenes."));
  // Le plafond réel (A23-09) : il dépend de l'écran, de la source et du mode, pas du seul choix ci-dessus.
  if (choices.display_hz > 0.0 && choices.source_fps > 0.0) {
    // Le plafond choisi (60 ou 120) abaisse celui de l'écran : c'est lui que la phrase cite (B23-05).
    note(i18n::tr_format("Here Auto goes up to ×{} ({:.0f} fps limit).",
                         top_factor(d, choices.display_hz, choices.source_fps),
                         core::auto_limit_hz(choices.display_hz, d.auto_cap)));
  }
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  // Sortie 4K : une source 1080p (ou plus) sans réduction, agrandie par SPAN.
  const bool four_k = d.upscale && (d.work_height == 0 || d.work_height >= choices.source_height) &&
                      choices.source_height >= 1080;
  section(i18n::tr("Interpolation"));
  ImGui::BeginDisabled(!choices.rife_available);
  // Une liste déroulante plutôt qu'un bouton par facteur : ×1,5 puis ×2 à ×8 tiennent sur une ligne.
  // Jusqu'à ×8 : au-delà, le profil (core::limits::kInterpFactor) refuserait la règle, et le choix
  // ne survivrait pas au fichier suivant.
  constexpr std::array<Factor, 9> kFactors{{{.num = 0, .den = 1},
                                            {.num = 3, .den = 2},
                                            {.num = 2, .den = 1},
                                            {.num = 3, .den = 1},
                                            {.num = 4, .den = 1},
                                            {.num = 5, .den = 1},
                                            {.num = 6, .den = 1},
                                            {.num = 7, .den = 1},
                                            {.num = 8, .den = 1}}};
  const auto factor_label = [](Factor factor) {
    if (factor.num <= factor.den) return std::string(i18n::tr("Off"));
    std::string text = factor_text(factor);
    if (factor == Factor{.num = kDefaultRifeFactor, .den = 1}) text += i18n::tr(" (recommended)");
    return text;
  };
  // Grisé : plus d'images par seconde que l'écran n'en affiche (le choix en cours reste visible).
  const auto too_fast = [&](Factor factor) {
    return choices.source_fps > 0.0 && choices.display_hz > 0.0 && factor.num > factor.den &&
           choices.source_fps * static_cast<double>(factor.num) / static_cast<double>(factor.den) >
               1.01 * choices.display_hz;
  };
  const Factor now = factor_of(d);
  const std::string factor_shown =
      d.rife_auto && now.num > now.den ? std::string(i18n::tr("Auto")) : factor_label(now);
  if (ImGui::BeginCombo(i18n::tr("Multiplier"), factor_shown.c_str())) {
    if (!exporting && ImGui::Selectable(i18n::tr("Auto"), d.rife_auto && now.num > now.den)) {
      d.rife_auto = true;
      if (now.num <= now.den) d.rife_factor = 8;  // le plafond du mode auto
      d.rife_factor_den = 1;
    }
    for (const Factor factor : kFactors) {
      const bool chosen = !d.rife_auto && (now == factor || (factor.num == 0 && now.num <= now.den));
      // Les facteurs fixes servent à l'export (et à --rife=N) : la lecture garde Auto ou rien.
      if (!exporting && factor.num > factor.den && !chosen) continue;
      const ImGuiSelectableFlags flags = too_fast(factor) && !chosen ? ImGuiSelectableFlags_Disabled : 0;
      if (ImGui::Selectable(factor_label(factor).c_str(), chosen, flags)) {
        d.rife_factor = factor.num;
        d.rife_factor_den = factor.den;
        d.rife_auto = false;
      }
    }
    ImGui::EndCombo();
  }
  if (d.rife_auto && now.num > now.den && !exporting) auto_cap_row(d, choices);
  if (d.rife_auto && now.num > now.den && !exporting) {
    // Mode Rapide : sur un écran VRR armé, chaque image part dès qu'elle est prête ; sans VRR, les instants
    // sont calés sur les vsyncs de l'écran (comme un jeu en vsync). Dans les deux cas la lecture repasse sur
    // Stable si trop d'images sont perdues.
    const bool vrr_armed = choices.display_vrr && choices.display_vrr_armed;
    const char* const shown = d.auto_free ? i18n::tr("Instant") : i18n::tr("Stable");
    if (ImGui::BeginCombo(i18n::tr("Auto mode"), shown)) {
      if (ImGui::Selectable(i18n::tr("Stable"), !d.auto_free)) d.auto_free = false;
      if (ImGui::Selectable(i18n::tr("Instant"), d.auto_free)) d.auto_free = true;
      if (vrr_armed) {
        panel::tip(i18n::tr("No steps: each frame is computed for the exact moment it is shown."));
      } else {
        panel::tip(i18n::tr(
            "Without VRR, frames follow the display refresh; a late frame repeats the previous one."));
      }
      ImGui::EndCombo();
    }
    panel::tip(i18n::tr(
        "Stable follows what the GPU can hold in steps (×1 to ×8). Instant has no steps and falls back "
        "to Stable if too many frames are lost. Playback only."));
  }
  if (d.rife_auto && now.num > now.den) {
    if (!choices.display_vrr) {
      note(i18n::tr("Without VRR, motion may judder at a fixed refresh rate."));
      panel::tip(i18n::tr("VRR unavailable: turn on G-Sync / FreeSync / Adaptive Sync to avoid judder."));
    } else if (choices.window_fullscreen && !choices.display_vrr_armed) {
      note(i18n::tr("VRR is turned off by the system: motion may judder."));
      panel::tip(i18n::tr("VRR is off in the system: turn on adaptive sync (KDE: Display > Adaptive sync)."));
    }
  }
  // En lecture, les facteurs fixes sont masqués : rien n'y est grisé. L'export les montre, grisés (B23-05).
  if (exporting && choices.source_fps > 0.0 && choices.display_hz > 0.0) {
    note(i18n::tr_format("Greyed out: above the display ({:.0f} Hz for {:.3f} fps).", choices.display_hz,
                         choices.source_fps));
  }
  if (exporting && now.num > now.den) {
    note(i18n::tr("Export skips no frames: each created frame costs a pass."));
  } else if (now.num > 4 * now.den) {
    note(
        i18n::tr("Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are "
                 "skipped."));
  }
  ImGui::EndDisabled();
  if (!choices.rife_available)
    note(i18n::tr("No interpolation model found: put a RIFE model in the models folder."));
  level_combo(d, choices);
  if (d.motion_engine == MotionEngine::DistilDrba && !choices.distil_drba_available) {
    note(i18n::tr_format("DistilDRBA model not found in {} ({}); RIFE is used instead.", choices.rife_folder,
                         kDistilDrbaFile));
  }
  fused_state_notes(d, choices);
  if (d.motion_engine == MotionEngine::DistilDrba && now.num > 5 * now.den) {
    note(i18n::tr("Above ×5 DistilDRBA computes more than RIFE: Fast (RIFE 4.22 lite) is lighter."));
  }
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(i18n::tr(
        "4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything."));
  } else if (four_k) {
    note(
        i18n::tr("4K output: RIFE estimates motion on a lighter 720p copy (less precise motion; the loss "
                 "is not measured)."));
  }
  note(i18n::tr("Scene cuts and held drawings are not interpolated."));
  if (!panel::advanced()) return;  // poses, ordre du traitement et Smooth Motion : réglages avancés
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Keep poses (DRBA) with RIFE"), &d.keep_poses);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Characters animated on twos keep their rhythm while the background interpolates. Applies "
               "live; DistilDRBA does this itself."));
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Fused RIFE"), &d.rife_fused);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Splits RIFE into small TensorRT engines: 1.4 ms instead of 4.5 ms per pass on an RTX 4070 "
               "Ti, the image within 53 dB of the whole network. TensorRT only. Reloads playback; built on "
               "first use for this size."));
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::DistilDrba);
  ImGui::Checkbox(i18n::tr("Fused DistilDRBA"), &d.drba_fused);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Splits DistilDRBA into small TensorRT engines: 0.92 ms instead of 2.10 ms per pass at "
               "1088×1920 on an RTX 4070 Ti. TensorRT only, Balanced precision. Reloads playback; built on "
               "first use for this size."));
  // Le cran de précision du mouvement (A23-06) : hors Équilibre, la chaîne fusionnée ne tourne pas.
  const TierSupport& motion_tiers =
      d.motion_engine == MotionEngine::DistilDrba ? choices.drba_tiers : choices.rife_tiers;
  if (d.motion_engine != MotionEngine::Apple &&
      (motion_tiers.performance || motion_tiers.quality || d.motion_precision != PrecisionTier::Balanced)) {
    TierSupport offered = motion_tiers;
    offered.performance = offered.performance || d.motion_precision == PrecisionTier::Performance;
    offered.quality = offered.quality || d.motion_precision == PrecisionTier::Quality;
    precision_combo(i18n::tr("Interpolation precision"), d.motion_precision, offered);
    panel::tip(i18n::tr("Balanced is FP16; Quality is FP32. Reloads playback."));
    if (d.motion_precision != PrecisionTier::Balanced && (d.rife_fused || d.drba_fused))
      note(i18n::tr("Fused engines only run at Balanced precision: the whole network is used instead."));
  }
  // DistilDRBA aussi choisit sa variante (mouvement ou non) d'après l'ordre du warp : seuls les moteurs sans
  // warp à ordonner grisent la liste (A23-02).
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && d.motion_engine != MotionEngine::DistilDrba);
  if (ImGui::BeginCombo(i18n::tr("Warp and SPAN"), std::string(label(d.span_order)).c_str())) {
    for (int i = 0; i < kSpanOrderCount; ++i) {
      const auto order = static_cast<SpanOrder>(i);
      if (ImGui::Selectable(std::string(label(order)).c_str(), order == d.span_order)) d.span_order = order;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("After SPAN upscales each source frame once, then warps it: faster (about 8 ms saved per "
               "frame from 1080p to 4K on an RTX 4070 Ti), but SPAN no longer cleans the warp defects of the "
               "in-between frames; the image quality of the two orders is not compared. Before SPAN also "
               "cleans each in-between frame. Auto picks After above 1080p, and from ×3 for a fixed "
               "multiplier. Reloads playback."));
  if (!exporting && choices.nvidia) smooth_motion_section(d, choices);
}

}  // namespace ajp::app::play
