// L'onglet Vidéo > Interpolation du menu des paramètres : le multiplicateur, l'Auto, le niveau et ses options
// ; en avancé, les moteurs, la précision, l'ordre du warp, GMFSS et les essais. L'onglet Export le reprend
// (exporting) pour la chaîne de l'export.
#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <utility>

#include <imgui.h>

#include "app/play/card_advice.h"
#include "app/play/drba_model.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/source_fit.h"
#include "app/play/visible_error.h"
#include "core/auto_rate.h"

namespace ajp::app::play {

using panel::note;
using panel::section;

namespace {

// Le warp de Catmull-Rom (PlaySettings::cubic_warp) : allumé par défaut.
void cubic_warp_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Sharper interpolation warp (Catmull-Rom)"), &d.cubic_warp);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Sharper interpolation warp (Catmull-Rom)"),
             i18n::tr("In-between frames keep finer lines, for a few percent of speed. Speed: a slight cost. "
                      "Quality impact: minor gain. Recommended: on. RIFE only."),
             panel::Apply::Reload);
}

// Moins de passes RIFE sur les paires fortes (PlaySettings::strong_middle) : un essai, éteint par défaut.
void strong_middle_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Fewer RIFE passes on fast motion"), &d.strong_middle);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Fewer RIFE passes on fast motion"),
             i18n::tr("Long pans are made faster by reusing the middle in-between frame for its neighbors, "
                      "which are then a little less accurate. A trial. Speed: up to +30 % on long pans. "
                      "Quality impact: minor loss. Recommended: off. RIFE with Keep poses only."));
}

// L'échelle par paire (PlaySettings::pair_scale) : un essai, éteint par défaut.
void pair_scale_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Lighter RIFE on fast motion (half size)"), &d.pair_scale);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Lighter RIFE on fast motion (half size)"),
             i18n::tr("Fast action is made faster by estimating motion at half size; fast motion can look "
                      "less precise. A trial, for cards at their limit. Speed: up to +50 % on fast motion. "
                      "Quality impact: moderate loss. Recommended: off. RIFE only."),
             panel::Apply::Reload);
}

std::string motion_description(const InterpChoice& choice) {
  switch (choice.level) {
    case InterpLevel::VeryFast:
      return i18n::tr(
          "RIFE 4.25 lite, the lightest level. Speed: about 2× Fast. Quality impact: moderate loss (softer "
          "in-between frames).");
    case InterpLevel::Fast:
      return i18n::tr(
          "RIFE 4.22 lite: light and clean, the reference for the other levels. Recommended on graphics "
          "cards other than NVIDIA RTX.");
    case InterpLevel::Balanced:
      return i18n::tr(
          "RIFE 4.26: cleaner in-between frames. Speed: slightly slower than Fast. Quality impact: minor "
          "gain. Recommended on NVIDIA RTX.");
    case InterpLevel::High:
      return i18n::tr(
          "RIFE guided by 1 to 3 GMFSS key frames on fast motion only. Speed: about −20 to −40 % vs "
          "Balanced, depending on the scene. Quality impact: moderate gain on fast turns and pans. NVIDIA "
          "RTX only.");
    case InterpLevel::GmfssUnion:
      return i18n::tr(
          "RIFE guided by 1 to 3 GMFSS key frames on every moving pair. Speed: about half the speed of "
          "Balanced. Quality impact: moderate gain, the steadiest motion. NVIDIA RTX only.");
    case InterpLevel::Rife425:
      return i18n::tr(
          "RIFE 4.25: the former Balanced level. Speed: slightly slower than Balanced. Quality impact: "
          "imperceptible.");
    case InterpLevel::DistilDrba:
      return i18n::tr(
          "DistilDRBA: characters animated on twos keep their rhythm; one frame of delay. Speed: about Fast "
          "up to ×5, slower above. Quality impact: none measured over RIFE. Auto does not use it.");
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

// Le nom du niveau dans la liste : Très rapide, Rapide, Équilibré, Élevé, Très élevé pour tous ; le nom du
// modèle pour les entrées avancées, rangées sous un filet « Avancé ».
std::string level_name(InterpLevel level) {
  switch (level) {
    case InterpLevel::VeryFast:
      return i18n::tr("Very fast");
    case InterpLevel::Fast:
      return i18n::tr("Fast");
    case InterpLevel::Balanced:
      return i18n::tr("Balanced");
    case InterpLevel::High:
      return i18n::tr("High");
    case InterpLevel::GmfssUnion:
      return i18n::tr("Very high");
    case InterpLevel::Rife425:
      return std::string("RIFE 4.25");
    case InterpLevel::DistilDrba:
      return std::string("DistilDRBA");
    case InterpLevel::System:
      return i18n::tr("System (macOS)");
  }
  return {};
}

// Le niveau se montre : son moteur est proposé ici, son modèle est là (GMFSS : ses deux graphes, sous
// TensorRT).
bool level_present(const InterpChoice& choice, const SettingsChoices& choices) {
  if (!offered(choice.engine)) return false;
  if ((choice.engine == MotionEngine::Gmfss || choice.gmfss_middle) && !choices.gmfss_available) return false;
  if (choice.engine == MotionEngine::DistilDrba) return choices.distil_drba_available;
  if (choice.engine == MotionEngine::Rife)
    return std::ranges::find(choices.rife_models, choice.rife_model) != choices.rife_models.end();
  return true;
}

// Le niveau d'interpolation : liste déroulante et note.
void level_combo(PlaySettings& d, const SettingsChoices& choices) {
  // Le niveau : un nom pour tous (Très rapide, Rapide, Équilibré), le modèle derrière dans l'infobulle ;
  // RIFE 4.25 et DistilDRBA (advanced) sous un filet « Avancé ».
  const std::string rife_now = d.rife_model.empty() ? choices.default_rife_model : d.rife_model;
  const std::optional<InterpLevel> level_now = interp_level(d.motion_engine, rife_now, d.gmfss_middle);
  const auto level_text = [&](const InterpChoice& choice) {
    std::string text = level_name(choice.level);
    if (choice.engine == MotionEngine::Rife && !choice.gmfss_middle &&
        choice.rife_model == choices.default_rife_model)
      text += i18n::tr(" (recommended)");
    if (choice.level == InterpLevel::High || choice.level == InterpLevel::GmfssUnion)
      text += i18n::tr(" (experimental)");
    return text;
  };
  std::string shown = level_now ? std::string() : rife_label(rife_now);
  if (d.motion_engine != MotionEngine::Rife && !level_now) shown = std::string(label(d.motion_engine));
  for (const InterpChoice& choice : kInterpLevels) {
    if (level_now == choice.level) shown = level_text(choice);
  }
  ImGui::BeginDisabled(!choices.rife_available);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Interpolation level")).c_str(), shown.c_str())) {
    bool separated = false;
    for (const InterpChoice& choice : kInterpLevels) {
      if (!level_present(choice, choices)) continue;
      if (choice.advanced && !std::exchange(separated, true)) ImGui::SeparatorText(i18n::tr("Advanced"));
      if (ImGui::Selectable(level_text(choice).c_str(), level_now == choice.level)) {
        d.motion_engine = choice.engine;
        d.gmfss_middle = choice.gmfss_middle;
        if (choice.engine == MotionEngine::Rife)
          d.rife_model = choice.rife_model == choices.default_rife_model ? std::string()
                                                                         : std::string(choice.rife_model);
      }
      panel::tip(level_name(choice.level).c_str(), motion_description(choice));
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  // GMFSS (Élevé, Très élevé) reste un essai : il n'est recommandé à aucune carte.
  if (level_now == InterpLevel::High || level_now == InterpLevel::GmfssUnion)
    panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  panel::tip(i18n::tr("Interpolation level"),
             std::string(i18n::tr(
                 "How clean the in-between frames are. Higher levels fix more artifacts on fast motion but "
                 "need a stronger card; High and Very high add GMFSS, are still experimental and need an "
                 "NVIDIA RTX.")) +
                 " " + advice_line(choices, interpolation_level_advice),
             panel::Apply::Reload);
}

// Les options de GMFSS (Élevé et Très élevé), dans l'Avancé de l'onglet.
void gmfss_options(PlaySettings& d, const SettingsChoices& choices) {
  if (d.motion_engine != MotionEngine::Gmfss && !d.gmfss_middle) return;
  section(i18n::tr("GMFSS"));
  ImGui::BeginDisabled(!choices.gmfss_half_flow_available);
  ImGui::Checkbox(i18n::tr("GMFSS motion at half size"), &d.gmfss_half_flow);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("GMFSS motion at half size"),
             i18n::tr("GMFSS runs much faster; very small movements can be slightly less precise. Speed: "
                      "GMFSS over 1.5× faster. Quality impact: minor loss on small motion. Recommended: on."),
             panel::Apply::Reload);
  const bool very_high = d.motion_engine == MotionEngine::Gmfss;
  if (very_high) {
    ImGui::Checkbox(i18n::tr("GMFSS at every instant, without RIFE"), &d.gmfss_every_instant);
    panel::tip(i18n::tr("GMFSS at every instant, without RIFE"),
               i18n::tr("Every in-between frame comes from GMFSS, the cleanest on hard motion, but the card "
                        "works far more and frames drop when it can't keep up. Speed: about a third of the "
                        "frames of Very high. Quality impact: minor change. Recommended: off."),
               panel::Apply::Reload);
  }
  if (d.gmfss_middle || (very_high && !d.gmfss_every_instant)) {
    ImGui::Checkbox(i18n::tr("GMFSS and RIFE at 540p"), &d.gmfss_light);
    panel::tip(i18n::tr("GMFSS and RIFE at 540p"),
               std::string(i18n::tr(
                   "GMFSS and RIFE work on a smaller picture, with slightly less precise motion. The time "
                   "saved does not add frames per second: it lets GMFSS draw more of the in-between frames "
                   "(over +50 %). Turn it on if the card can't keep up at High or Very high. Speed: same "
                   "frames per second, over +50 % frames drawn by GMFSS. Quality impact: minor loss.")) +
                   " " + advice_line(choices, light_motion_advice),
               panel::Apply::Reload);
    ImGui::Checkbox(i18n::tr("GMFSS keeps the source detail"), &d.gmfss_compose);
    panel::tip(
        i18n::tr("GMFSS keeps the source detail"),
        i18n::tr("Sharper GMFSS frames above 720p of working resolution, for a little speed. Speed: slightly "
                 "slower (around −10 %). Quality impact: minor gain, not yet judged. Recommended: off."),
        panel::Apply::Reload);
  }
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
  // C37-03 : hors RTX 40 et 50, ou sous le pilote 580.82.07, le pilote n'ajoute rien ; la case le dit au
  // lieu d'être cochée pour rien.
  const SmoothMotionBlock block = smooth_motion_block(choices.gpu_name, choices.driver_version);
  ImGui::BeginDisabled(without_rife || block != SmoothMotionBlock::None);
  ImGui::Checkbox(i18n::tr("Double the frames (NVIDIA driver)"), &d.smooth_motion);
  panel::tip(i18n::tr("Double the frames (NVIDIA driver)"),
             i18n::tr("The NVIDIA driver (RTX 40 and 50) adds a frame between each displayed frame, for even "
                      "smoother motion on a high-refresh display. Its frames are rougher than the player's. "
                      "Quality impact: moderate loss (driver frames are rougher). Recommended: off."),
             panel::Apply::Restart);
  panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  ImGui::EndDisabled();
  if (block == SmoothMotionBlock::Gpu)
    note(i18n::tr("Greyed out: this graphics card is not an RTX 40 or 50."));
  else if (block == SmoothMotionBlock::Driver)
    note(i18n::tr("Greyed out: the NVIDIA driver is older than 580.82.07, update it."));
  else if (without_rife)
    note(i18n::tr("Greyed out: turn on RIFE at ×2 or more, or playback stalls."));
#endif
}

// La liste du plafond de la cadence Auto et la phrase qui dit où il s'arrête.
void auto_cap_row(PlaySettings& d, const SettingsChoices& choices) {
  // Le plafond de la cadence : le maximum de l'écran par défaut, ou une valeur plus basse pour ménager le
  // GPU.
  const auto cap_label = [](int cap) {
    return cap == 0 ? std::string(i18n::tr("Maximum (display)")) : i18n::tr_format("{} fps", cap);
  };
  if (ImGui::BeginCombo(panel::left(i18n::tr("Auto limit")).c_str(), cap_label(d.auto_cap).c_str())) {
    for (const int cap : {0, 120, 60}) {
      if (ImGui::Selectable(cap_label(cap).c_str(), d.auto_cap == cap)) d.auto_cap = cap;
    }
    // Un plafond de play.toml hors de cette liste reste visible et choisi (B23-05).
    if (d.auto_cap != 0 && d.auto_cap != 120 && d.auto_cap != 60)
      ImGui::Selectable(cap_label(d.auto_cap).c_str(), true);
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("Auto limit"),
      i18n::tr("The highest frame rate Auto aims for; a lower limit saves power and heat. Recommended: "
               "Maximum."));
  // Le plafond réel (A23-09) : il dépend de l'écran, de la source et du mode, pas du seul choix ci-dessus.
  if (choices.display_hz > 0.0 && choices.source_fps > 0.0) {
    // Le plafond choisi (60 ou 120) abaisse celui de l'écran : c'est lui que la phrase cite (B23-05).
    note(i18n::tr_format("Here Auto goes up to ×{} ({:.0f} fps limit).",
                         top_factor(d, choices.display_hz, choices.source_fps),
                         core::auto_limit_hz(choices.display_hz, d.auto_cap)));
  }
}

// L'Auto au-dessus de ×1 : ce que le VRR de l'écran lui permet, et comment l'armer.
void vrr_notes(const SettingsChoices& choices) {
  if (!choices.display_vrr) {
    note(i18n::tr("Without VRR, motion may judder at a fixed refresh rate."));
    panel::tip(i18n::tr("VRR unavailable: turn on G-Sync / FreeSync / Adaptive Sync to avoid judder."));
  } else if (choices.window_fullscreen && !choices.display_vrr_armed) {
    note(i18n::tr("VRR is turned off by the system: motion may judder."));
#ifdef _WIN32
    panel::tip(
        i18n::tr("VRR is off in the system: turn on Variable refresh rate in Settings > System > "
                 "Display > Graphics."));
#else
    panel::tip(i18n::tr("VRR is off in the system: turn on adaptive sync (KDE: Display > Adaptive sync)."));
#endif
  }
#ifndef _WIN32
  else if (!choices.window_fullscreen && !choices.display_vrr_armed) {
    // Hors plein écran, KDE n'arme pas le VRR (C23-09).
    note(i18n::tr("VRR only works in full screen here: motion may judder in a window."));
  }
#endif
}

// Lecture en Auto : le mode Stable ou Instantané, et en Instantané la coupure de l'interpolation quand la
// carte ne suit pas (motion_cut.h).
void auto_mode_rows(PlaySettings& d, const SettingsChoices& choices) {
  // Mode Rapide : sur un écran VRR armé, chaque image part dès qu'elle est prête ; sans VRR, les instants
  // sont calés sur les vsyncs de l'écran (comme un jeu en vsync). Une image en retard laisse la source.
  const bool vrr_armed = choices.display_vrr && choices.display_vrr_armed;
  const char* const shown = d.auto_free ? i18n::tr("Instant") : i18n::tr("Stable");
  if (ImGui::BeginCombo(panel::left(i18n::tr("Auto mode")).c_str(), shown)) {
    if (ImGui::Selectable(i18n::tr("Stable"), !d.auto_free)) d.auto_free = false;
    if (ImGui::Selectable(i18n::tr("Instant"), d.auto_free)) d.auto_free = true;
    if (vrr_armed) {
      panel::tip(i18n::tr("No steps: each frame is computed for the exact moment it is shown."));
    } else {
      panel::tip(
          i18n::tr("Without VRR, frames follow the display refresh; a late frame repeats the previous one."));
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Auto mode"),
             i18n::tr("Stable: fixed steps (×1 to ×8), the most regular motion. Instant: each frame is made "
                      "for the exact moment it is shown, for more frames and smoother motion; it can stutter "
                      "slightly at the card's limit. Speed: Instant +50 % frames or more. Quality impact: "
                      "minor (Instant can stutter slightly at the card's limit). Recommended: Instant."));
  if (d.auto_free) {
    ImGui::Checkbox(i18n::tr("Turn interpolation off if the graphics card can't keep up"), &d.auto_off);
    panel::tip(i18n::tr("Turn interpolation off if the graphics card can't keep up"),
               i18n::tr("If the card can't keep up for about 10 s, interpolation stops for this video and "
                        "upscaling keeps running, instead of stuttering. Recommended: on."));
  }
}

// Le multiplicateur : Auto ou rien en lecture, les facteurs fixes à l'export.
void multiplier_combo(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
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
  if (ImGui::BeginCombo(panel::left(i18n::tr("Multiplier")).c_str(), factor_shown.c_str())) {
    if (!exporting && ImGui::Selectable(i18n::tr("Auto"), d.rife_auto && now.num > now.den)) {
      d.rife_auto = true;
      if (now.num <= now.den) d.rife_factor = core::kAutoMaxStep;  // ×10, le plafond de l'Auto (C12-06)
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
  panel::tip(i18n::tr("Multiplier"),
             exporting ? i18n::tr("How many frames per source frame the export makes.")
                       : i18n::tr("Auto: as many frames as the display and the graphics card allow. Off: no "
                                  "interpolation. Recommended: Auto."));
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
namespace {

// Ce que la carte aura du mal à tenir : DistilDRBA au-delà de ×5, la sortie 4K, la copie 720p de RIFE.
void load_notes(const PlaySettings& d, const SettingsChoices& choices, Factor now, bool exporting) {
  // Sortie 4K : une source 1080p (ou plus) sans réduction, agrandie par SPAN.
  const bool four_k = d.upscale && (d.work_height == 0 || d.work_height >= choices.source_height) &&
                      choices.source_height >= 1080;
  if (d.motion_engine == MotionEngine::DistilDrba && now.num > 5 * now.den) {
    note(i18n::tr("Above ×5 DistilDRBA computes more than RIFE: Fast (RIFE 4.22 lite) is lighter."));
  }
  // La copie 720p de RIFE : la même condition que la chaîne, sur la taille de travail réellement jouée
  // (C23-04).
  const bool reduced_motion =
      now.num > now.den && choices.source_height > 0 &&
      rife_motion_reduced(chain_work_size(d, choices.source_width, choices.source_height));
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(i18n::tr(
        "4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything."));
  } else if (reduced_motion) {
    // Le gain de vitesse n'est pas mesuré : la note le dit, sans chiffre inventé (C23-03).
    note(
        i18n::tr("Above 720p of working resolution, RIFE estimates motion on a lighter 720p copy (less "
                 "precise motion; the speed gained and the quality lost are not measured)."));
  }
}

}  // namespace

void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  section(i18n::tr("Interpolation"));
  // La chaîne ne l'allume qu'à une cadence d'anime (source_fit.h) : le choix reste, pour la vidéo suivante.
  if (d.rife_factor >= 2 && !anime_rate(choices.source_fps))
    note(i18n::tr_format(
        "Off for this video: {:.2f} fps is not an anime frame rate (23.976, 24, 25, 29.97 or 30).",
        choices.source_fps));
  ImGui::BeginDisabled(!choices.rife_available);
  multiplier_combo(d, choices, exporting);
  const Factor now = factor_of(d);
  if (d.rife_auto && now.num > now.den && !exporting) auto_cap_row(d, choices);
  if (d.rife_auto && now.num > now.den && !exporting) auto_mode_rows(d, choices);
  if (d.rife_auto && now.num > now.den) vrr_notes(choices);
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
  load_notes(d, choices, now, exporting);
  // « Lisser les dessins tenus » (upscale_smooth.cpp) : RIFE seulement, à la place des poses gardées.
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Smooth held drawings"), &d.smooth_drawings);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Smooth held drawings"),
             i18n::tr("A drawing held 2 to 6 frames glides to the next one instead of freezing, then "
                      "jumping. Speed: no loss. Quality impact: changes the look of the animation (a matter "
                      "of taste). Recommended: off. RIFE only; replaces Keep poses."),
             panel::Apply::Reload);
  cubic_warp_checkbox(d);
}

void motion_advanced(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  section(i18n::tr("Engines and precision"));
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Fused RIFE"), &d.rife_fused);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Fused RIFE"),
             i18n::tr("RIFE about 3 times faster on NVIDIA, with no visible difference. Built once on first "
                      "use. Speed: about 3× faster per pass. Quality impact: imperceptible. Recommended: on. "
                      "NVIDIA only; built on first use."),
             panel::Apply::Reload);
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::DistilDrba);
  ImGui::Checkbox(i18n::tr("Fused DistilDRBA"), &d.drba_fused);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Fused DistilDRBA"),
             i18n::tr("DistilDRBA about 3 times faster on NVIDIA, with no visible difference. Built once on "
                      "first use. Speed: about 3× faster per pass. Quality impact: imperceptible. "
                      "Recommended: on. NVIDIA only; built on first use."),
             panel::Apply::Reload);
  // Le cran de précision du mouvement (A23-06) : hors Équilibre, la chaîne fusionnée ne tourne pas.
  const TierSupport& motion_tiers =
      d.motion_engine == MotionEngine::DistilDrba ? choices.drba_tiers : choices.rife_tiers;
  if (d.motion_engine != MotionEngine::Apple &&
      (motion_tiers.performance || motion_tiers.quality || d.motion_precision != PrecisionTier::Balanced)) {
    TierSupport offered = motion_tiers;
    offered.performance = offered.performance || d.motion_precision == PrecisionTier::Performance;
    offered.quality = offered.quality || d.motion_precision == PrecisionTier::Quality;
    precision_combo(i18n::tr("Interpolation precision"), d.motion_precision, offered);
    if (d.motion_precision != PrecisionTier::Balanced && (d.rife_fused || d.drba_fused))
      note(i18n::tr("Fused engines only run at Balanced precision: the whole network is used instead."));
  }
  // DistilDRBA aussi choisit sa variante (mouvement ou non) d'après l'ordre du warp : seuls les moteurs sans
  // warp à ordonner grisent la liste (A23-02).
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && d.motion_engine != MotionEngine::DistilDrba);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Warp and SPAN")).c_str(),
                        std::string(label(d.span_order)).c_str())) {
    for (int i = 0; i < kSpanOrderCount; ++i) {
      const auto order = static_cast<SpanOrder>(i);
      if (ImGui::Selectable(std::string(label(order)).c_str(), order == d.span_order)) d.span_order = order;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Warp and SPAN"),
             i18n::tr("After: each source frame is upscaled once, then moved; much lighter at high "
                      "multipliers. Before: every shown frame is upscaled. No visible difference. Speed: "
                      "After about −50 to −85 % of upscaling time from ×2 to ×8. Quality impact: "
                      "imperceptible. Recommended: Auto (After above 1080p and from ×3)."),
             panel::Apply::Reload);
  gmfss_options(d, choices);

  section(i18n::tr("Trials"));
  const bool smoothed = d.smooth_drawings && d.motion_engine == MotionEngine::Rife;
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife || smoothed);
  ImGui::Checkbox(i18n::tr("Keep poses (DRBA) with RIFE"), &d.keep_poses);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Keep poses (DRBA) with RIFE"),
      i18n::tr("Characters animated on twos keep their rhythm while the background moves. Speed: "
               "slightly faster (held poses skip a pass). Quality impact: minor gain. Recommended: on. "
               "RIFE only; Smooth held drawings replaces it."));
  pair_scale_checkbox(d);
  strong_middle_checkbox(d);
  if (!exporting && choices.nvidia) smooth_motion_section(d, choices);
}

}  // namespace ajp::app::play
