// L'onglet Vidéo > Interpolation du menu des paramètres : le multiplicateur, l'Auto, le niveau et ses options
// ; en avancé, les moteurs, la précision, l'ordre du warp, GMFSS et les essais. L'onglet Export le reprend
// (exporting) pour la chaîne de l'export.
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string>
#include <utility>

#include <imgui.h>

#include "app/play/card_advice.h"
#include "app/play/drba_model.h"
#include "app/play/export_job.h"
#include "app/play/gmfss_model.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/interp_retired.h"
#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/source_fit.h"
#include "app/play/visible_error.h"
#include "core/auto_rate.h"

namespace kirari::app::play {

using panel::note;
using panel::section;

namespace {

// Le warp de Catmull-Rom (PlaySettings::cubic_warp) : allumé par défaut.
void cubic_warp_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox(i18n::tr("Sharper interpolation warp (Catmull-Rom)"), &d.cubic_warp);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Sharper interpolation warp (Catmull-Rom)"),
             i18n::tr("In-between frames keep finer lines, for a few percent of speed. Speed: about −3 %. "
                      "Quality impact: minor gain. Recommended: on. RIFE only."),
             panel::Apply::Reload);
}

// Le mouvement d'ensemble (PlaySettings::pan_lock) : éteint par défaut.
void pan_lock_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && d.motion_engine != MotionEngine::DistilDrba);
  ImGui::Checkbox(i18n::tr("Steady slow pans"), &d.pan_lock);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Steady slow pans"),
             i18n::tr("On slow pans, the motion of the whole picture, measured to a hundredth of a pixel, "
                      "replaces small uneven motion. Available for DistilDRBA Motion and for RIFE frames "
                      "that use the regular warp. RIFE frames using Keep poses keep their own warp."),
             panel::Apply::Reload);
}

// Le mouvement estimé sur une copie 720p (PlaySettings::motion_720p) : éteint par défaut, la taille de
// travail.
void motion_720p_checkbox(PlaySettings& d, const SettingsChoices& choices) {
  const bool already_small =
      choices.source_height > 0 &&
      !rife_motion_reduced(chain_work_size(d, choices.source_width, choices.source_height), true);
  const bool motion_warp =
      d.motion_engine == MotionEngine::Rife || d.motion_engine == MotionEngine::DistilDrba;
  ImGui::BeginDisabled(!motion_warp || already_small);
  ImGui::Checkbox(i18n::tr("Estimate motion at 720p"), &d.motion_720p);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Estimate motion at 720p"),
      i18n::tr("RIFE and the DistilDRBA Motion variant can estimate motion on a 720p copy. This saves GPU "
               "time above 720p, with less precise fine motion. The warp keeps the working image size."),
      panel::Apply::Reload);
}

// Moins de passes RIFE sur les paires fortes (PlaySettings::strong_middle) : un essai, éteint par défaut.
void strong_middle_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife || !poses_kept(d));
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
             i18n::tr("On fast pairs, some RIFE passes estimate motion at half size. This saves time at the "
                      "cost of less precise motion. It does not speed up calm pairs. Recommended: off."),
             panel::Apply::Reload);
}

std::string motion_description(const InterpChoice& choice) {
  switch (choice.level) {
    case InterpLevel::DistilDrba:
      return i18n::tr(
          "Keeps the rhythm of the animation: characters drawn on twos hold their poses while the background "
          "moves. Solid on fine repeating patterns (bars, grids). Lighter than RIFE up to ×5, heavier above. "
          "One frame of delay.");
    case InterpLevel::Rife422Lite:
      return i18n::tr(
          "The lightest RIFE, with fine lines. Recommended on graphics cards other than NVIDIA RTX.");
    case InterpLevel::Rife425:
      return i18n::tr(
          "The most precise RIFE, for about 15 % more computation than RIFE 4.22 lite. Recommended on NVIDIA "
          "RTX.");
    case InterpLevel::GmfssMiddle:
      return i18n::tr(
          "RIFE 4.25 for most frames, GMFSS union for the middle of fast motion: holds fast pans and "
          "rotations better. Heavier than RIFE. NVIDIA RTX (TensorRT) only; a trial.");
    case InterpLevel::GmfssUnion:
      return i18n::tr(
          "Prioritizes GMFSS keys on every moving pair, with RIFE when time is short. The GMFSS options "
          "also offer a mode without RIFE. NVIDIA RTX (TensorRT) only.");
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

// Le nom du modèle dans la liste : plus de niveaux, les modèles par leur nom.
std::string level_name(InterpLevel level) {
  switch (level) {
    case InterpLevel::DistilDrba:
      return std::string("DistilDRBA v2 lite");
    case InterpLevel::Rife422Lite:
      return std::string("RIFE 4.22 lite");
    case InterpLevel::Rife425:
      return std::string("RIFE 4.25");
    case InterpLevel::GmfssMiddle:
      return std::string("RIFE 4.25 + GMFSS Fortuna Union");
    case InterpLevel::GmfssUnion:
      return std::string("GMFSS Fortuna Union");
    case InterpLevel::System:
      return i18n::tr("System (macOS)");
  }
  return {};
}

bool gmfss_available(const SettingsChoices& choices) {
  return choices.gmfss_available && gmfss_kept();
}

bool interpolation_available(const SettingsChoices& choices) {
  return choices.rife_available || gmfss_available(choices) || choices.rife_loaded;
}

void gmfss_state_note(const PlaySettings& d, const SettingsChoices& choices) {
  const auto* chain = choices.chain();
  if (chain == nullptr || !chain->loaded || choices.gmfss_loaded || !choices.rife_loaded) return;
  const auto& loaded = *chain->loaded;
  if (d.motion_engine != loaded.motion_engine || d.gmfss_middle != loaded.gmfss_middle ||
      d.gmfss_every_instant != loaded.gmfss_every_instant)
    return;
  if (loaded.motion_engine == MotionEngine::Gmfss ||
      (loaded.motion_engine == MotionEngine::Rife && loaded.gmfss_middle))
    note(i18n::tr("GMFSS could not load for this video. RIFE is running instead."));
}

// Le choix se montre : son moteur est proposé ici, son modèle est là. GMFSS union, seul ou au milieu des
// paires fortes, seulement avec le natif (interp_retired.h, sauf --no-gmfss-native) et ses deux graphes.
bool level_present(const InterpChoice& choice, const SettingsChoices& choices) {
  if (!offered(choice.engine)) return false;
  if ((choice.engine == MotionEngine::Gmfss || choice.gmfss_middle) && !gmfss_available(choices))
    return false;
  if (choice.engine == MotionEngine::DistilDrba) return choices.distil_drba_available;
  if (choice.engine == MotionEngine::Rife)
    return std::ranges::find(choices.rife_models, choice.rife_model) != choices.rife_models.end();
  return true;
}

// Les niveaux de la liste Interpolation ; choisir un niveau quand elle est désactivée l'allume.
template <class Text>
void level_items(PlaySettings& d, const SettingsChoices& choices, bool exporting, bool on,
                 std::optional<InterpLevel> level_now, const Text& level_text) {
  for (const InterpChoice& choice : kInterpLevels) {
    if (!level_present(choice, choices)) continue;
    if (ImGui::Selectable(level_text(choice).c_str(), on && level_now == choice.level)) {
      d.motion_engine = choice.engine;
      d.gmfss_middle = choice.gmfss_middle;
      if (choice.engine == MotionEngine::Gmfss && choices.default_rife_model.empty())
        d.gmfss_every_instant = true;
      if (choice.engine == MotionEngine::Rife)
        d.rife_model =
            choice.rife_model == choices.default_rife_model ? std::string() : std::string(choice.rife_model);
      if (!on) {
        // En lecture, l'Auto jusqu'à ×10 (C12-06) ; à l'export, ×2 en attendant le multiplicateur.
        d.rife_auto = !exporting;
        d.rife_factor = exporting ? 2 : core::kAutoMaxStep;
        d.rife_factor_den = 1;
      }
    }
    panel::tip(level_name(choice.level).c_str(), motion_description(choice));
  }
}

// L'interpolation : Désactivée ou un niveau, en tête de l'onglet. Un niveau allume l'Auto en lecture (un
// facteur fixe à l'export) ; Désactivée garde le niveau pour la fois suivante.
void interpolation_combo(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  const bool chosen_here =
      choices.default_rife_model.empty() || std::ranges::contains(choices.rife_models, d.rife_model);
  const std::string& rife_now = chosen_here ? d.rife_model : choices.default_rife_model;
  const std::optional<InterpLevel> level_now = interp_level(d.motion_engine, rife_now, d.gmfss_middle);
  const Factor now = exporting ? export_factor(d, choices.source_fps) : factor_of(d);
  const bool on = now.num > now.den;
  const auto level_text = [&](const InterpChoice& choice) {
    std::string text = level_name(choice.level);
    if (choice.engine == MotionEngine::Rife && !choice.gmfss_middle &&
        choice.rife_model == choices.default_rife_model)
      text += i18n::tr(" (recommended)");
    return text;
  };
  std::string shown = level_now ? std::string() : rife_label(rife_now);
  if (d.motion_engine != MotionEngine::Rife && !level_now) shown = std::string(label(d.motion_engine));
  for (const InterpChoice& choice : kInterpLevels) {
    if (level_now == choice.level) shown = level_text(choice);
  }
  const bool available = interpolation_available(choices) && (!exporting || anime_rate(choices.source_fps));
  if (!on || !available ||
      (!choices.rife_available && !choices.rife_loaded && d.motion_engine != MotionEngine::Gmfss))
    shown = i18n::tr("Off");
  ImGui::BeginDisabled(!available);
  if (panel::combo(i18n::tr("Interpolation"), shown.c_str())) {
    if (ImGui::Selectable(i18n::tr("Off"), !on)) {
      d.rife_auto = false;
      d.rife_factor = 0;
      d.rife_factor_den = 1;
    }
    level_items(d, choices, exporting, on, level_now, level_text);
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Interpolation"),
             std::string(i18n::tr(
                 "Adds frames between the video's own frames for smoother motion, as many as the display "
                 "and the graphics card allow. DistilDRBA v2 lite keeps the rhythm of the animation; RIFE "
                 "4.22 lite is the lightest; RIFE 4.25 is the most precise RIFE.")) +
                 " " + advice_line(choices, interpolation_level_advice),
             panel::Apply::Reload);
}

// Les options de GMFSS (Élevé et Très élevé), dans l'Avancé de l'onglet.
void gmfss_options(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("GMFSS"));
  ImGui::BeginDisabled(!choices.gmfss_half_flow_available && !d.gmfss_half_flow);
  ImGui::Checkbox(i18n::tr("GMFSS motion at half size"), &d.gmfss_half_flow);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("GMFSS motion at half size"),
      i18n::tr(
          "GMFlow measures motion at half size to save GPU time. Very small movements can be less precise."),
      panel::Apply::Reload);
  const bool very_high = d.motion_engine == MotionEngine::Gmfss;
  if (very_high) {
    ImGui::BeginDisabled(choices.default_rife_model.empty() && d.gmfss_every_instant);
    ImGui::Checkbox(i18n::tr("GMFSS at every instant, without RIFE"), &d.gmfss_every_instant);
    ImGui::EndDisabled();
    panel::tip(i18n::tr("GMFSS at every instant, without RIFE"),
               i18n::tr("Every interpolated instant uses GMFSS. It costs more than sharing the work with "
                        "RIFE, and playback can miss instants when the GPU falls behind."),
               panel::Apply::Reload);
  }
  if (!very_high || !d.gmfss_every_instant) {
    ImGui::Checkbox(i18n::tr("GMFSS and RIFE at 540p"), &d.gmfss_light);
    panel::tip(i18n::tr("GMFSS and RIFE at 540p"),
               std::string(i18n::tr(
                   "Limits the working size of the hybrid motion models to 540p. Saves GPU time, which "
                   "can allow more GMFSS keys, with less precise motion and softer keys.")) +
                   " " + advice_line(choices, light_motion_advice),
               panel::Apply::Reload);
    ImGui::Checkbox(i18n::tr("GMFSS keeps the source detail"), &d.gmfss_compose);
    panel::tip(i18n::tr("GMFSS keeps the source detail"),
               i18n::tr("When the hybrid models work below the working image size, add source detail to "
                        "the GMFSS keys. This costs two extra splats per input image."),
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
  const bool without_rife =
      !smooth_motion_rife(d) || choices.rife_models.empty() || motion_off_for_rate(d, choices.source_fps);
  // C37-03 : hors RTX 40 et 50, ou sous le pilote 580.82.07, le pilote n'ajoute rien ; la case le dit au
  // lieu d'être cochée pour rien.
  const SmoothMotionBlock block = smooth_motion_block(choices.gpu_name, choices.driver_version);
  ImGui::BeginDisabled(!d.smooth_motion && (without_rife || block != SmoothMotionBlock::None));
  ImGui::Checkbox(i18n::tr("Frame generation (NVIDIA driver)"), &d.smooth_motion);
  panel::tip(i18n::tr("Frame generation (NVIDIA driver)"),
             i18n::tr("The NVIDIA driver can add intermediate frames on compatible GPUs. The gain depends "
                      "on the display refresh rate and frames can look rougher. Recommended: off."),
             panel::Apply::Restart);
  panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  ImGui::EndDisabled();
  if (block == SmoothMotionBlock::Gpu)
    note(i18n::tr("Greyed out: this graphics card is not an RTX 40 or 50."));
  else if (block == SmoothMotionBlock::Driver)
    note(i18n::tr("Update the NVIDIA driver to use Smooth Motion on this GPU."));
  else if (without_rife)
    note(i18n::tr("Smooth Motion needs RIFE at ×2 or more in Kirari."));
#endif
}

// La limite d'images par seconde de l'interpolation : un curseur jusqu'à la fréquence de l'écran, comme dans
// les jeux, et la phrase qui dit où l'Auto s'arrête.
void fps_limit_row(PlaySettings& d, const SettingsChoices& choices) {
  // Tout en haut, le maximum de l'écran : auto_cap vaut 0 et suit l'écran s'il change.
  const int top = choices.display_hz > 0.0 ? static_cast<int>(std::lround(choices.display_hz)) : 240;
  const int bottom = std::min(48, top);
  int shown = d.auto_cap == 0 ? top : std::clamp(d.auto_cap, bottom, top);
  const std::string value =
      shown >= top ? i18n::tr_format("Max ({} fps)", top) : i18n::tr_format("{} fps", shown);
  // La valeur est écrite au bout du rail : le panneau tient la colonne du libellé (left), sans champ.
  const std::string id = panel::left(i18n::tr("FPS limit"));
  ImGui::SetNextItemWidth(-1.0F);
  // Le texte le plus long, réservé au bout, pour que le rail ne bouge pas quand la valeur change.
  if (panel::slider_int(id.c_str(), &shown, bottom, top, value)) d.auto_cap = shown >= top ? 0 : shown;
  panel::tip(i18n::tr("FPS limit"),
             i18n::tr("The highest frame rate interpolation aims for; a lower limit saves power and heat. "
                      "Recommended: the display maximum."));
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

// Le mode Stable ou Instantané de l'interpolation en lecture.
void mode_combo(PlaySettings& d, const SettingsChoices& choices) {
  // Mode Rapide : sur un écran VRR armé, chaque image part dès qu'elle est prête ; sans VRR, les instants
  // sont calés sur les vsyncs de l'écran (comme un jeu en vsync). Une image en retard laisse la source.
  const bool vrr_armed = choices.display_vrr && choices.display_vrr_armed;
  const std::string shown = !d.rife_auto  ? i18n::tr_format("Fixed ({})", factor_text(factor_of(d)))
                            : d.auto_free ? i18n::tr("Instant")
                                          : i18n::tr("Stable");
  if (ImGui::BeginCombo(panel::left(i18n::tr("Mode")).c_str(), shown.c_str())) {
    const auto select_auto = [&](bool instant) {
      if (!d.rife_auto) {
        d.rife_auto = true;
        d.rife_factor = core::kAutoMaxStep;
        d.rife_factor_den = 1;
      }
      d.auto_free = instant;
    };
    if (ImGui::Selectable(i18n::tr("Stable"), d.rife_auto && !d.auto_free)) select_auto(false);
    if (ImGui::Selectable(i18n::tr("Instant"), d.rife_auto && d.auto_free)) select_auto(true);
    if (vrr_armed) {
      panel::tip(i18n::tr("No steps: each frame is computed for the exact moment it is shown."));
    } else {
      panel::tip(
          i18n::tr("Without VRR, frames follow the display refresh; a late frame repeats the previous one."));
    }
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("Mode"),
      i18n::tr("Stable chooses a sustainable cadence from fixed steps. Instant renders at the intended "
               "presentation times and may repeat a frame when the GPU falls behind. Recommended: Instant."));
}

// En Instantané, la coupure de l'interpolation quand la carte ne suit pas (motion_cut.h) ; grisée en
// Stable.
void keep_up_checkbox(PlaySettings& d) {
  ImGui::BeginDisabled(!d.rife_auto || !d.auto_free);
  ImGui::Checkbox(i18n::tr("Turn off if the graphics card can't keep up"), &d.auto_off);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Turn off if the graphics card can't keep up"),
             i18n::tr("If the card can't keep up for about 10 s, interpolation stops for this video and "
                      "upscaling keeps running, instead of stuttering. Recommended: on. Instant mode only."));
}

// Le multiplicateur de l'export : toujours fixe, indépendant de l'écran.
void multiplier_combo(PlaySettings& d, double source_fps) {
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
  const Factor now = export_factor(d, source_fps);
  const std::string factor_shown = factor_label(now);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Multiplier")).c_str(), factor_shown.c_str())) {
    for (const Factor factor : kFactors) {
      const bool chosen = now == factor || (factor.num == 0 && now.num <= now.den);
      if (ImGui::Selectable(factor_label(factor).c_str(), chosen)) {
        d.rife_factor = factor.num;
        d.rife_factor_den = factor.den;
        d.rife_auto = false;
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Multiplier"), i18n::tr("How many frames per source frame the export makes."));
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
    note(i18n::tr("Above ×5 DistilDRBA computes more than RIFE: RIFE 4.22 lite is lighter."));
  }
  // La copie 720p de RIFE : la même condition que la chaîne, sur la taille de travail réellement jouée
  // (C23-04).
  const bool reduced_motion =
      d.motion_engine == MotionEngine::Rife && now.num > now.den && choices.source_height > 0 &&
      rife_motion_reduced(chain_work_size(d, choices.source_width, choices.source_height), d.motion_720p);
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(i18n::tr(
        "4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything."));
  } else if (reduced_motion) {
    // Le gain de vitesse n'est pas mesuré : la note le dit, sans chiffre inventé (C23-03).
    note(
        i18n::tr("Above 720p of working resolution, RIFE estimates motion on a lighter 720p copy (Estimate "
                 "motion at 720p): slow pans can ripple."));
  }
}

}  // namespace

void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  section(i18n::tr("Interpolation"));
  // La chaîne ne l'allume qu'à une cadence d'anime (source_fit.h) : le choix reste, pour la vidéo suivante.
  if (choices.source_fps > 0.0 && motion_off_for_rate(d, choices.source_fps))
    note(i18n::tr_format(
        "Off for this video: {:.2f} fps is not an anime frame rate (23.976, 24, 25, 29.97 or 30).",
        choices.source_fps));
  const Factor now = exporting ? export_factor(d, choices.source_fps) : factor_of(d);
  const bool on = now.num > now.den && interpolation_available(choices);
  // Deux réglages par ligne, comme chez Dolphin : l'interpolation et sa coupure, le mode et la limite
  // d'images par seconde, puis les deux options de rendu. Désactivée, tout le reste est grisé.
  if (panel::begin_grid("##interpolation")) {
    panel::cell();
    interpolation_combo(d, choices, exporting);
    panel::cell();
    ImGui::BeginDisabled(!on);
    if (exporting) {
      multiplier_combo(d, choices.source_fps);
    } else {
      keep_up_checkbox(d);
      panel::cell();
      mode_combo(d, choices);
      panel::cell();
      ImGui::BeginDisabled(!d.rife_auto);
      fps_limit_row(d, choices);
      ImGui::EndDisabled();
    }
    // « Lisser les dessins tenus » (upscale_smooth.cpp) : RIFE seulement, à la place des poses gardées.
    panel::cell();
    ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
    ImGui::Checkbox(i18n::tr("Smooth held drawings"), &d.smooth_drawings);
    ImGui::EndDisabled();
    panel::tip(
        i18n::tr("Smooth held drawings"),
        i18n::tr("RIFE can make a drawing held for 2 to 6 frames glide into the next one. This changes "
                 "the timing of the animation and needs more frames read ahead. It replaces Keep poses."),
        panel::Apply::Reload);
    panel::cell();
    cubic_warp_checkbox(d);
    panel::cell();
    pan_lock_checkbox(d);
    panel::cell();
    motion_720p_checkbox(d, choices);
    ImGui::EndDisabled();
    panel::end_grid();
  }
  // Le plafond réel (A23-09) : il dépend de l'écran, de la source et du mode, pas du seul choix ci-dessus.
  if (on && d.rife_auto && !exporting && choices.display_hz > 0.0 && choices.source_fps > 0.0) {
    const double limit_hz = core::auto_limit_hz(choices.display_hz, d.auto_cap);
    int gmfss_cap = 0;
    if (d.motion_engine == MotionEngine::Gmfss && d.gmfss_every_instant) {
      gmfss_cap = choices.gmfss_only_auto_cap;
      if (choices.chain() == nullptr && gmfss_available(choices))
        gmfss_cap = gmfss_frame_limit(choices.source_fps, d.gmfss_half_flow && choices.gmfss_half_flow_available);
    }
    if (gmfss_cap > 0) {
      note(i18n::tr_format("GMFSS frame rate limit: {:.0f} fps.", core::auto_limit_hz(limit_hz, gmfss_cap)));
    } else {
      note(i18n::tr_format("Here interpolation goes up to ×{} ({:.0f} fps limit).",
                           top_factor(d, choices.display_hz, choices.source_fps), limit_hz));
    }
  }
  if (on && !exporting) vrr_notes(choices);
  if (exporting && on) {
    note(i18n::tr("Export skips no frames: each created frame costs a pass."));
  } else if (now.num > 4 * now.den && !d.rife_auto) {
    note(
        i18n::tr("Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are "
                 "skipped."));
  }
  if (!interpolation_available(choices))
    note(i18n::tr("No interpolation model found: put a RIFE model in the models folder."));
  if (d.motion_engine == MotionEngine::DistilDrba && !choices.distil_drba_available) {
    note(i18n::tr_format("DistilDRBA model not found in {} ({}); RIFE is used instead.", choices.rife_folder,
                         kDistilDrbaFile));
  }
  fused_state_notes(d, choices);
  if (on && !exporting) gmfss_state_note(d, choices);
  load_notes(d, choices, now, exporting);
}

// L'Avancé de l'onglet : GMFSS et les essais côte à côte, puis Smooth Motion. Les moteurs fusionnés, la
// précision et l'ordre du warp ne sont plus au menu : leur réglage par défaut est le bon partout, et
// play.toml les garde (fused, motion_precision, span_order).
void motion_advanced(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  if (!panel::begin_boxes()) return;
  const Factor now = exporting ? export_factor(d, choices.source_fps) : factor_of(d);
  const bool on = now.num > now.den && interpolation_available(choices);
  // Les options de GMFSS ne se montrent que si ce moteur peut être choisi.
  const bool gmfss = (d.motion_engine == MotionEngine::Gmfss || d.gmfss_middle) && gmfss_available(choices);
  if (gmfss) {
    panel::next_box();
    ImGui::BeginDisabled(!on);
    gmfss_options(d, choices);
    panel::end_group();
    ImGui::EndDisabled();
  }
  ImGui::BeginDisabled(!on);
  panel::next_box();
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
  panel::end_group();
  ImGui::EndDisabled();
  if (!exporting && choices.nvidia) {
    panel::next_box();
    smooth_motion_section(d, choices);
  }
  panel::end_boxes();
}

}  // namespace kirari::app::play
