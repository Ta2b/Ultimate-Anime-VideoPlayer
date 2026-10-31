#include "app/play/settings_panel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_export.h"
#include "app/play/settings_sound.h"
#include "app/play/text_fit.h"
#include "app/play/tracks.h"
#include "core/engines.h"
#include "render/looks.h"

namespace ajp::app::play {
namespace {

using panel::kWarn;
using panel::note;
using panel::section;

// Le moteur du mode HDR : RTX HDR (TrueHDR, NVIDIA), HDR adaptatif (tous GPU : libplacebo, puis
// l'aspect TrueHDR émulé qui suit la scène), ou SDR.
constexpr std::array<const char*, 3> kRoutes{"RTX HDR (NVIDIA)", "Adaptive HDR (any GPU)",
                                             "SDR"};  // clés i18n

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
  // Le choix de la méthode reste libre, même sur un écran SDR ; ce qui règle le HDR de l'écran (pic, gris
  // moyen) est grisé tant que l'écran n'en reçoit pas.
  section(i18n::tr("Output"));
  const int route = d.true_hdr ? 0 : (d.hdr_native ? 1 : 2);
  if (ImGui::BeginCombo(i18n::tr("HDR method"), i18n::tr(kRoutes[static_cast<std::size_t>(route)]))) {
    for (int i = 0; i < static_cast<int>(kRoutes.size()); ++i) {
      if (i == 0 && !(choices.nvidia && core::kHostOs == core::Os::Linux))
        continue;  // TrueHDR : GPU NVIDIA, sous Linux seulement
      ImGui::BeginDisabled(i == 0 && !choices.true_hdr_available);
      if (ImGui::Selectable(i18n::tr(kRoutes[static_cast<std::size_t>(i)]), i == route)) {
        d.true_hdr = i == 0;
        d.hdr_native = i == 1;
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }
  if (choices.nvidia && !choices.true_hdr_backend)
    note(i18n::tr("RTX HDR needs the TensorRT engine; Adaptive HDR is used instead."));
  else if (choices.nvidia && !choices.true_hdr_available)
    note(i18n::tr("RTX HDR: the Maxine VideoFX SDK is not installed (see Engines)."));
  if (!choices.hdr_output)
    note(i18n::tr("SDR display: HDR is unavailable. The Color page's grade imitates the HDR look."));
  panel::end_group();  // l'encadré Sortie se ferme avant la zone grisée
  ImGui::BeginDisabled(!choices.hdr_output);
  section(choices.nvidia ? i18n::tr("HDR settings") : i18n::tr("Adaptive HDR settings"));
  bool automatic = d.true_hdr_peak == 0;
  if (ImGui::Checkbox(i18n::tr_format("Auto peak ({} nits)", choices.auto_peak).c_str(), &automatic))
    d.true_hdr_peak =
        automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
  ImGui::BeginDisabled(automatic);
  int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
  if (ImGui::SliderInt(i18n::tr("Peak (nits)"), &peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi, "%d",
                       ImGuiSliderFlags_AlwaysClamp) &&
      !automatic)
    d.true_hdr_peak = peak;
  ImGui::EndDisabled();
  if (panel::advanced())
    ImGui::SliderInt(i18n::tr("Mid gray"), &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
  if (choices.nvidia && panel::advanced()) {
    ImGui::BeginDisabled(!d.true_hdr);
    ImGui::Checkbox(i18n::tr("Debanding (RTX HDR)"), &d.true_hdr_debanding);
    ImGui::EndDisabled();
  }
  if (panel::advanced() && ImGui::Button(i18n::tr("Defaults"))) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.true_hdr_gray = defaults.true_hdr_gray;
    d.true_hdr_debanding = defaults.true_hdr_debanding;
  }
  note(i18n::tr("Applied live. Adaptive HDR follows the scene: less expansion on bright scenes."));
  panel::end_group();  // le dernier encadré s'ouvre dans la zone grisée : il se ferme avant d'en sortir
  ImGui::EndDisabled();
}

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au
// centième et bornée : le float du curseur à 0,8 vaut 0,800000012 en double, au-dessus de la borne
// de check_grade_settings, et une valeur saisie au clavier (Ctrl + clic) peut sortir de la plage ;
// l'une comme l'autre faisait refuser tout le grade.
bool grade_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(label, &shown, static_cast<float>(range.lo), static_cast<float>(range.hi), "%.2f",
                          ImGuiSliderFlags_AlwaysClamp))
    return false;
  value = std::clamp(std::round(static_cast<double>(shown) * 100.0) / 100.0, range.lo, range.hi);
  return true;
}

// Contraste et saturation : ceux de TrueHDR, que le HDR adaptatif émule, et que la sortie SDR
// gradée reprend autour du gris moyen (render::sdr_params) ; un seul réglage pour les trois.
void look_section(PlaySettings& d) {
  section(i18n::tr("Contrast and saturation"));
  ImGui::SliderInt(i18n::tr("Contrast"), &d.true_hdr_contrast, limits::kTrueHdrContrast.lo,
                   limits::kTrueHdrContrast.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt(i18n::tr("Saturation"), &d.true_hdr_saturation, limits::kTrueHdrSaturation.lo,
                   limits::kTrueHdrSaturation.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button(i18n::tr("Reset contrast and saturation"))) {
    const PlaySettings defaults;
    d.true_hdr_contrast = defaults.true_hdr_contrast;
    d.true_hdr_saturation = defaults.true_hdr_saturation;
  }
  note(i18n::tr("Live, for RTX HDR, Adaptive HDR and SDR output."));
  if (!d.grade && !d.true_hdr) note(i18n::tr("Without RTX HDR, turn on \"Color grading\" below."));
}

// Le look : jusqu'à quinze LUT 3D .cube enchaînées dans l'ordre, comme des nœuds de Resolve
// (render/cube_lut.h) ; ceux du lecteur, puis les fichiers du dossier luts. Chaque intensité va de 0 à 1.
void look_section(PlaySettings& d, const SettingsChoices& choices) {
  const auto label_of = [&](const std::string& key) -> std::string {
    if (key.empty()) return i18n::tr("None");
    for (const render::BuiltinLook& look : render::builtin_looks())
      if (look.key == key) return i18n::tr(look.label.data());
    return key;
  };
  const std::vector<LookEntry> entries = list_looks(choices.luts_dir);
  // Sur une sortie SDR, une pile qui contient une LUT d'HDR est coupée en entier (chain_hdr.cpp) : ses
  // réglages n'agissent pas, ils sont grisés.
  bool has_hdr_look = render::look_is_pq(d.look, "");
  for (const LookLayer& layer : d.look_layers)
    has_hdr_look = has_hdr_look || render::look_is_pq(layer.key, "");
  const bool stack_off = !choices.hdr_output && has_hdr_look;
  ImGui::BeginDisabled(stack_off);
  const auto slot = [&](int number, std::string& name, double& mix) {
    ImGui::PushID(number);
    if (ImGui::BeginCombo(i18n::tr_format("Look {} (LUT)", number).c_str(), label_of(name).c_str())) {
      if (ImGui::Selectable(i18n::tr("None"), name.empty())) name.clear();
      for (const LookEntry& entry : entries)
        if (ImGui::Selectable(entry.label.c_str(), entry.key == name)) name = entry.key;
      ImGui::EndCombo();
    }
    ImGui::BeginDisabled(name.empty());
    grade_slider(i18n::tr_format("Look {} intensity", number).c_str(), mix, core::Range<double>{0.0, 1.0});
    ImGui::EndDisabled();
    ImGui::PopID();
  };
  slot(1, d.look, d.look_mix);
  for (std::size_t i = 0; i < d.look_layers.size(); ++i)
    slot(static_cast<int>(i) + 2, d.look_layers[i].key, d.look_layers[i].mix);
  // + ajoute une LUT à la fin de la pile, − retire la dernière.
  ImGui::BeginDisabled(d.look_layers.size() + 1 >= render::kLookSlots);
  if (ImGui::Button("+")) d.look_layers.push_back({});
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(d.look_layers.empty());
  if (ImGui::Button("-")) d.look_layers.pop_back();
  ImGui::EndDisabled();
  ImGui::EndDisabled();  // stack_off
  if (stack_off)
    note(i18n::tr("SDR display: a stack with an HDR LUT is turned off (the LUTs are made for HDR)."));
  note(
      i18n::tr("Looks are LUT files (.cube) placed in the luts folder of the player's data, applied in order "
               "like Resolve nodes; names containing PQ, HDR or 2020 are read as HDR LUTs."));
}

// Grade adaptatif : les curseurs du grade et le gris moyen de RTX HDR sont la scène sombre ; ces valeurs, la
// scène claire, mélangées selon la luminance moyenne lissée de l'image source (moteur TensorRT). Hors du bloc
// désactivé du grade : section() ouvre un encadré, et un BeginDisabled ne peut pas le traverser.
void adaptive_section(PlaySettings& d, const SettingsChoices& choices) {
  using namespace render::grade_limits;
  render::GradeSettings& g = d.grade_values;
  // Grade adaptatif : les curseurs ci-dessus et le gris moyen de RTX HDR sont la scène sombre ; ces
  // valeurs, la scène claire, mélangées selon la luminance moyenne lissée de l'image source (moteur
  // TensorRT).
  section(i18n::tr("Adaptive to scene brightness"));
  ImGui::Checkbox(i18n::tr("Adapt to scene brightness"), &g.adaptive);
  ImGui::BeginDisabled(!g.adaptive);
  const auto percent_slider = [](const char* label, double& value) {
    auto shown = static_cast<float>(value);
    if (ImGui::SliderFloat(label, &shown, static_cast<float>(kAdaptiveLuma.lo),
                           static_cast<float>(kAdaptiveLuma.hi), "%.0f %%", ImGuiSliderFlags_AlwaysClamp))
      value = std::clamp(std::round(static_cast<double>(shown)), kAdaptiveLuma.lo, kAdaptiveLuma.hi);
  };
  percent_slider(i18n::tr("Dark scene up to"), g.adaptive_lo);
  percent_slider(i18n::tr("Bright scene from"), g.adaptive_hi);
  if (g.adaptive_hi <= g.adaptive_lo) g.adaptive_hi = std::min(g.adaptive_lo + 1.0, kAdaptiveLuma.hi);
  grade_slider(i18n::tr("Deep blacks, bright scene"), g.adaptive_shadows, kShadows);
  ImGui::BeginDisabled(!choices.hdr_output);  // la vibrance adaptative ne joue pas en sortie SDR
  grade_slider(i18n::tr("Colors, saturated blue scene"), g.adaptive_vibrance, kVibrance);
  ImGui::EndDisabled();
  auto gray = static_cast<float>(g.adaptive_gray);
  if (ImGui::SliderFloat(i18n::tr("Middle gray, bright scene"), &gray, static_cast<float>(kMiddleGray.lo),
                         static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp))
    g.adaptive_gray = std::clamp(std::round(static_cast<double>(gray)), kMiddleGray.lo, kMiddleGray.hi);
  ImGui::EndDisabled();
  ImGui::Checkbox(i18n::tr("Record scene statistics (learning)"), &d.adaptive_record);
  note(
      i18n::tr("The sliders above and the middle gray are for dark scenes; the values here apply to bright "
               "scenes, blended with the source's smoothed average brightness. Works with RTX HDR and with "
               "Adaptive HDR (any GPU)."));
}

void grade_tab(PlaySettings& d, const SettingsChoices& choices) {
  // Le grade s'applique sur toutes les routes, le mode SDR compris : il imite le rendu HDR (contraste,
  // couleurs riches, hautes lumières) pour ceux qui n'ont pas d'écran HDR. L'onglet reste actif.
  look_section(d);
  section(i18n::tr("Color grading"));
  ImGui::Checkbox(i18n::tr("Color grading"), &d.grade);
  ImGui::BeginDisabled(!d.grade);
  if (ImGui::BeginCombo(i18n::tr("Preset"), std::string(grade_label(d.grade_preset)).c_str())) {
    for (int i = 0; i < render::kGradePresetCount; ++i) {
      const auto preset = static_cast<render::GradePreset>(i);
      if (ImGui::Selectable(std::string(grade_label(preset)).c_str(), preset == d.grade_preset)) {
        d.grade_preset = preset;
        d.grade_values = with_preset(preset, d.grade_values);
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Vivid pushes highlights, colors and blacks; Natural is the default look. Live."));
  if (panel::advanced()) look_section(d, choices);  // la pile de LUT : réglages avancés seulement
  render::GradeSettings& g = d.grade_values;
  using namespace render::grade_limits;
  bool moved = false;
  if (panel::advanced()) {
    // Sur une sortie SDR, ni les éclats, ni les zones claires, ni le gamut large n'ont d'effet (sdr_params).
    ImGui::BeginDisabled(!choices.hdr_output);
    moved |= grade_slider(i18n::tr("Highlights"), g.highlights_rtx, kHighlights);
    moved |= grade_slider(i18n::tr("Bright areas"), g.area_rtx, kAreaRtx);
    ImGui::EndDisabled();
    moved |= grade_slider(i18n::tr("Colors"), g.vibrance_rtx, kVibrance);
    ImGui::BeginDisabled(!choices.hdr_output);
    moved |= grade_slider(i18n::tr("Wide colors"), g.expand, kExpand);
    ImGui::EndDisabled();
    moved |= grade_slider(i18n::tr("Deep blacks"), g.shadows, kShadows);
    moved |= grade_slider(i18n::tr("Color limiter"), g.guard, kGuard);
    if (moved) d.grade_preset = render::GradePreset::Custom;
    ImGui::BeginDisabled(!choices.hdr_output);
    ImGui::Checkbox(i18n::tr("Display P3 gamut"), &g.gamut_p3);
    ImGui::EndDisabled();
    if (!choices.hdr_output)
      note(
          i18n::tr("SDR display: Highlights, Bright areas, Wide colors and Display P3 gamut have no effect "
                   "and are greyed out."));
  }
  ImGui::EndDisabled();
  if (panel::advanced()) adaptive_section(d, choices);
  if (d.true_hdr) {
    note(i18n::tr("RTX HDR path: TrueHDR, then the grade on the PQ output."));
  } else if (d.hdr_native) {
    note(i18n::tr("Adaptive HDR path: libplacebo in PQ with a scene-based TrueHDR look, then the grade."));
  } else {
    note(
        choices.hdr_output
            ? i18n::tr("SDR mode: the grade imitates the HDR look in the display's HDR10 container.")
            : i18n::tr(
                  "SDR mode: the grade imitates the HDR look without an HDR display. Vivid strengthens it."));
  }
}

// Composition finale (Image IA > Agrandissement, réglages avancés) : le filtre vers la fenêtre et le
// débandage.
void composition_section(PlaySettings& d) {
  section(i18n::tr("Final scaling"));
  if (ImGui::BeginCombo(i18n::tr("Scaler"), std::string(label(d.final_scaler)).c_str())) {
    for (int i = 0; i < kFinalScalerCount; ++i) {
      const auto scaler = static_cast<FinalScaler>(i);
      if (ImGui::Selectable(std::string(label(scaler)).c_str(), scaler == d.final_scaler))
        d.final_scaler = scaler;
    }
    ImGui::EndCombo();
  }
  ImGui::Checkbox(i18n::tr("Debanding"), &d.deband);
  note(
      i18n::tr("The scaler stretches the output to the window; EWA Lanczos is sharpest but heaviest. No "
               "effect on export."));
}

// Noir absolu et soins OLED (HDR et couleurs > HDR, réglages avancés).
void oled_sections(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("OLED black"));
  if (ImGui::BeginCombo(i18n::tr("True black"), std::string(label(d.oled_black)).c_str())) {
    for (int i = 0; i < kOledBlackCount; ++i) {
      const auto mode = static_cast<OledBlack>(i);
      if (ImGui::Selectable(std::string(label(mode)).c_str(), mode == d.oled_black)) d.oled_black = mode;
    }
    ImGui::EndCombo();
  }
  ImGui::BeginDisabled(d.oled_black == OledBlack::Off);
  grade_slider(i18n::tr("Threshold (code steps)"), d.oled_black_steps, render::grade_limits::kBlackSteps);
  ImGui::EndDisabled();
  note((i18n::tr("Display: ") + choices.display_text).c_str());
  section(i18n::tr("OLED care"));
  ImGui::Checkbox(i18n::tr("Hide the bar when paused (burn-in)"), &d.oled_care);
  ImGui::Checkbox(i18n::tr("Steady frame rate (VRR flicker)"), &d.steady_cadence);
  note(i18n::tr("Steady frame rate keeps VRR at a constant refresh so dark scenes do not flicker on OLED."));
  if (!choices.black_status.empty()) note(choices.black_status.c_str());
  note(i18n::tr("Near-blacks become true black (0), up to the threshold. Live."));
}

std::string language_label(std::string_view tag) {
  std::string name = language_name(tag);
  return name.empty() ? std::string(tag) : name;  // une balise écrite à la main dans play.toml
}

// Une langue au choix : `none_label` pour la valeur vide (audio : « Selon le fichier », sous-titres :
// « Aucun »), sinon une des langues du menu.
void language_combo(const char* label, const char* none_label, std::string& tag) {
  const std::string current = tag.empty() ? none_label : language_label(tag);
  if (!ImGui::BeginCombo(label, current.c_str())) return;
  if (ImGui::Selectable(none_label, tag.empty())) tag.clear();
  for (const Language& language : menu_languages()) {
    if (ImGui::Selectable(i18n::tr(language.name), same_language(tag, language.code)))
      tag = std::string(language.code);
  }
  ImGui::EndCombo();
}

// Deux préférences par ordre : l'audio et les sous-titres voulus ensemble ; la 2e sert quand la 1re
// n'est pas entière dans le fichier.
void languages_tab(PlaySettings& d) {
  d.language_presets.resize(limits::kLanguagePresets);
  for (std::size_t rank = 0; rank < d.language_presets.size(); ++rank) {
    ImGui::PushID(static_cast<int>(rank));
    section(rank == 0 ? i18n::tr("Preferred languages") : i18n::tr("Fallback"));
    language_combo(i18n::tr("Audio language"), i18n::tr("From the file"), d.language_presets[rank].audio);
    language_combo(i18n::tr("Subtitle language"), i18n::tr("None"), d.language_presets[rank].subtitles);
    panel::end_group();  // l'encadré de la section ouvre sa propre fenêtre fille : il se ferme avant le PopID
    ImGui::PopID();
  }
  note(i18n::tr(
      "The first preference whose tracks exist is used when a file opens. Applies to the next file."));
}

void subtitles_tab(PlaySettings& d) {
  section(i18n::tr("Subtitle size"));
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt(i18n::tr("Size"), &percent, static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F), "%d %%",
                       ImGuiSliderFlags_AlwaysClamp)) {
    d.subtitle_scale = static_cast<float>(percent) / 100.0F;
  }
  if (ImGui::Button(i18n::tr("Reset size"))) d.subtitle_scale = 1.0F;
  panel::tip(i18n::tr("Scales all text, panels included. Live."));
}

// L'épisode suivant : lu tout seul à la fin du fichier ouvert.
void episodes_section(PlaySettings& d) {
  section(i18n::tr("Episodes"));
  ImGui::Checkbox(i18n::tr("Autoplay next episode"), &d.autoplay_next);
  panel::tip(i18n::tr("Plays the next file in the folder. N: next, Shift+N: previous, O: skip credits."));
}

void stats_tab(PlaySettings& d) {
  section(i18n::tr("On-screen statistics"));
  ImGui::Checkbox(i18n::tr("All statistics (Tab)"), &d.stats.all);
  ImGui::Spacing();
  ImGui::BeginDisabled(d.stats.all);
  ImGui::Checkbox(i18n::tr("Pipeline"), &d.stats.chain);
  ImGui::Checkbox(i18n::tr("Frames"), &d.stats.frames);
  ImGui::Checkbox(i18n::tr("Interpolation"), &d.stats.rife);
  ImGui::Checkbox(i18n::tr("Tracks"), &d.stats.tracks);
  ImGui::Checkbox(i18n::tr("HDR output"), &d.stats.output);
  ImGui::Checkbox(i18n::tr("Exact position"), &d.stats.position);
  ImGui::Checkbox(i18n::tr("Video memory"), &d.stats.memory);
  ImGui::EndDisabled();
}

// Général > Configuration : « Optimiser pour ma machine » d'abord (un bouton, le reste suit), puis le
// préréglage, la priorité et ce que le lecteur sait de la machine.
BenchEvent config_page(PlaySettings& d, const SettingsChoices& choices, BenchNote& bench,
                       std::vector<Input>& actions) {
  const BenchContext context{.display_height = choices.display_height > 0 ? choices.display_height : 1080,
                             .display_hz = choices.display_hz > 0.0 ? choices.display_hz : 60.0,
                             .source_height = choices.source_height > 0 ? choices.source_height : 1080,
                             .source_fps = choices.source_fps > 0.0 ? choices.source_fps : 23.976,
                             .span = choices.span_prepare,
                             .fp8_reason = choices.fp8_reason,
                             .data_dir = choices.settings_dir};
  const BenchEvent event = bench_page(choices.place, choices.source_file, context, bench, d);
  if (event == BenchEvent::Started) actions.emplace_back(Command{cmd::Pause{}});

  section(i18n::tr("This machine"));
  ImGui::TextWrapped("%s",
                     choices.gpu_name.empty() ? i18n::tr("Unknown graphics card") : choices.gpu_name.c_str());
  if (choices.display_hz > 0.0) note(i18n::tr_format("Display {:.0f} Hz", choices.display_hz));
  note(choices.bench_text.empty()
           ? std::string(i18n::tr("Not optimized yet: settings are an estimate from the card."))
           : choices.bench_text);

  section(i18n::tr("Automatic setup"));
  if (ImGui::BeginCombo(i18n::tr("Preset"), std::string(label(d.preset)).c_str())) {
    for (int i = 0; i < kPresetCount; ++i) {
      const auto preset = static_cast<Preset>(i);
      if (ImGui::Selectable(std::string(label(preset)).c_str(), preset == d.preset)) {
        d.preset = preset;
        // La chaîne du préréglage s'écrit tout de suite (Personnalisé : rien ne bouge).
        d = apply_preset(
            d,
            Machine{.tier = static_cast<Tier>(choices.machine_tier),
                    .display_height = choices.display_height,
                    .display_hz = choices.display_hz},
            core::SourceInfo{
                .width = choices.source_width, .height = choices.source_height, .fps = choices.source_fps});
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr(
      "Auto picks the settings for your card, display and video. Changing one by hand switches to Custom."));
  if (ImGui::BeginCombo(i18n::tr("Priority"), std::string(label(d.priority)).c_str())) {
    for (int i = 0; i < kPriorityCount; ++i) {
      const auto priority = static_cast<Priority>(i);
      if (ImGui::Selectable(std::string(label(priority)).c_str(), priority == d.priority))
        d.priority = priority;
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("What Auto favors when the machine cannot do everything."));

  section(i18n::tr("Setup wizard"));
  if (ImGui::Button(i18n::tr("Run the wizard again"))) actions.emplace_back(OpenEngineScreen{.wizard = true});
  panel::tip(i18n::tr("Nothing changes until \"Finish\"."));
  return event;
}

// Général > Réinitialiser.
void reset_page(std::vector<Input>& actions, const ProfilesView& view) {
  section(i18n::tr("Profile defaults"));
  note(i18n::tr("Resets the active profile (Ctrl+1 to Ctrl+9); other profiles are untouched."));
  if (ImGui::Button(i18n::tr("Reset profile")))
    actions.emplace_back(EditProfile{.edit = core::edit::ResetProfile{.slot = view.active}});
}

}  // namespace

// L'arbre : l'ordre est celui de l'écran. Les catégories se déduisent du changement de `category`.
constexpr std::array<SettingsPage, 12> kPages{{
    {"General", "Setup", false},
    {"General", "Playback", false},     // langues préférées, sous-titres, épisode suivant
    {"General", "Application", false},  // lecteur par défaut et mises à jour
    {"General", "Statistics", false},
    {"General", "Reset", false},
    {"AI image", "Upscaling", false},
    {"AI image", "Smoothness", false},
    {"HDR and color", "HDR", false},
    {"HDR and color", "Colors", false},
    {"Audio", "Output", false},
    {"Export and engines", "Export", false},
    {"Export and engines", "Engines", false},
}};

std::span<const SettingsPage> settings_pages() {
  return kPages;
}

void language_choice(const char* label, const char* none_label, std::string& tag) {
  language_combo(label, none_label, tag);
}

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
SettingsPanel::~SettingsPanel() {
  if (prepare_.stop) prepare_.stop->request_stop();  // la tâche dépend du lecteur qui ouvre ce panneau
  bench_.stop();
}

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
  if (!open_) return result;
  // Réglages changés ailleurs (Tab, volume) : le panneau les reprend, sans perdre ce qui attend.
  if (running != seen_) {
    draft_ = live_part(draft_, running, choices.rife_loaded);
    seen_ = running;
  }
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 18.0F * scale;
  const float width = std::min(760.0F * scale, io.DisplaySize.x - 2.0F * pad);
  const float bottom = 120.0F * scale;  // la barre de lecture reste visible dessous
  ImGui::SetNextWindowPos({io.DisplaySize.x - pad, pad + 40.0F * scale}, ImGuiCond_Always, {1.0F, 0.0F});
  ImGui::SetNextWindowSize({width, std::max(240.0F * scale, io.DisplaySize.y - pad - 40.0F * scale - bottom)},
                           ImGuiCond_Always);
  bool keep = true;
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  if (ImGui::Begin((std::string(i18n::tr("Settings")) + "###settings").c_str(), &keep, kFlags)) {
    ImGui::PushItemWidth(-width * 0.42F);
    panel::advanced() = advanced_;
    bool delay_held = false;
    const PlaySettings before = draft_;
    // Sous les deux colonnes : la case « Réglages avancés » (et le bandeau « Appliquer » quand il est là).
    const float footer = (pending_shown_ ? 84.0F * scale : 0.0F) + ImGui::GetFrameHeightWithSpacing() +
                         2.0F * ImGui::GetStyle().ItemSpacing.y;
    const float tree_width = std::min(200.0F * scale, width * 0.3F);
    if (!advanced_ && kPages[static_cast<std::size_t>(page_)].advanced) page_ = 0;
    // Une page masquée : un réglage avancé sans la case « Réglages avancés ».
    const auto visible = [&](std::size_t i) { return advanced_ || !kPages[i].advanced; };
    // Barre de défilement fine, collée au bord extérieur des deux zones.
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0F * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0F * scale);
    // Arbre collé à gauche : peu de marge intérieure.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0F * scale, 4.0F * scale));
    ImGui::BeginChild("##arbre", ImVec2(tree_width, -footer), ImGuiChildFlags_Borders);
    ImGui::PopStyleVar();
    // La molette sur l'arbre passe d'une page à la suivante ou à la précédente.
    if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0F) {
      const int step = io.MouseWheel < 0.0F ? 1 : -1;
      for (int next = page_ + step; next >= 0 && next < static_cast<int>(kPages.size()); next += step) {
        if (visible(static_cast<std::size_t>(next))) {
          page_ = next;
          break;
        }
      }
    }
    // L'arbre, dessiné à la main : une bande par catégorie (chevron, titre en couleur d'accent), les pages en
    // retrait dessous, la page choisie marquée d'une barre d'accent et d'un fond.
    {
      ImDrawList* draw = ImGui::GetWindowDrawList();
      const ImGuiStyle& style = ImGui::GetStyle();
      const float row_height = ImGui::GetFontSize() + 12.0F * scale;
      const float head_height = ImGui::GetFontSize() + 14.0F * scale;
      const float indent = 26.0F * scale;
      const ImU32 accent = ImGui::ColorConvertFloat4ToU32(panel::kAccent);
      ImVec4 soft = panel::kAccent;
      soft.w = 0.20F;
      ImVec4 band = style.Colors[ImGuiCol_Header];
      band.w = 0.14F;
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {style.ItemSpacing.x, 2.0F * scale});
      std::string_view category;
      const std::string_view page_category = kPages[static_cast<std::size_t>(page_)].category;
      std::string_view shown_category;
      if (shown_page_ >= 0) shown_category = kPages[static_cast<std::size_t>(shown_page_)].category;
      int category_index = -1;
      bool category_open = false;
      for (std::size_t i = 0; i < kPages.size(); ++i) {
        const SettingsPage& page = kPages[i];
        if (!visible(i)) continue;
        if (std::string_view(page.category) != category) {
          category = page.category;
          ++category_index;
          const auto slot = static_cast<std::size_t>(category_index);
          // Quand la page change de catégorie (clic, molette), la catégorie de la page s'ouvre et les autres
          // se replient : l'arbre tient dans une fenêtre basse. Ensuite chaque catégorie garde l'état choisi
          // à la main.
          if (page_ != shown_page_ && shown_category != page_category)
            category_open_[slot] = category == page_category;
          if (category_index > 0) ImGui::Dummy({0.0F, 6.0F * scale});
          const ImVec2 at = ImGui::GetCursorScreenPos();
          const float row_width = ImGui::GetContentRegionAvail().x;
          ImGui::PushID(category_index);
          if (ImGui::InvisibleButton("##categorie", {row_width, head_height}))
            category_open_[slot] = !category_open_[slot];
          ImGui::PopID();
          const bool hovered = ImGui::IsItemHovered();
          ImVec4 fill = band;
          if (hovered) fill.w = 0.30F;
          draw->AddRectFilled(at, {at.x + row_width, at.y + head_height},
                              ImGui::ColorConvertFloat4ToU32(fill), 5.0F * scale);
          // Le chevron : un triangle plein, à droite quand c'est replié, vers le bas quand c'est ouvert.
          const float cx = at.x + 12.0F * scale;
          const float cy = at.y + head_height * 0.5F;
          const float r = 4.0F * scale;
          if (category_open_[slot])
            draw->AddTriangleFilled({cx - r, cy - r * 0.6F}, {cx + r, cy - r * 0.6F}, {cx, cy + r * 0.9F},
                                    accent);
          else
            draw->AddTriangleFilled({cx - r * 0.6F, cy - r}, {cx - r * 0.6F, cy + r}, {cx + r * 0.9F, cy},
                                    accent);
          draw->AddText({at.x + 26.0F * scale, at.y + (head_height - ImGui::GetFontSize()) * 0.5F}, accent,
                        i18n::tr(page.category));
          category_open = category_open_[slot];
        }
        if (!category_open) continue;
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float row_width = ImGui::GetContentRegionAvail().x;
        ImGui::PushID(static_cast<int>(i));
        const bool picked = ImGui::InvisibleButton("##page", {row_width, row_height});
        ImGui::PopID();
        if (picked) page_ = static_cast<int>(i);
        const bool selected = static_cast<int>(i) == page_;
        if (selected) {
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y}, {at.x + row_width, at.y + row_height},
                              ImGui::ColorConvertFloat4ToU32(soft), 5.0F * scale);
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y + 4.0F * scale},
                              {at.x + indent - 5.0F * scale, at.y + row_height - 4.0F * scale}, accent,
                              2.0F * scale);
        } else if (ImGui::IsItemHovered()) {
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y}, {at.x + row_width, at.y + row_height},
                              ImGui::ColorConvertFloat4ToU32(band), 5.0F * scale);
        }
        draw->AddText({at.x + indent, at.y + (row_height - ImGui::GetFontSize()) * 0.5F},
                      ImGui::GetColorU32(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                      i18n::tr(page.name));
      }
      ImGui::PopStyleVar();
    }
    shown_page_ = page_;
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##page", ImVec2(0, -footer));
    panel::begin_page();
    {
      const std::string_view name = kPages[static_cast<std::size_t>(page_)].name;
      if (name == "Setup") {
        // L'optimisation (ou son annulation) s'applique tout de suite, sans « Appliquer » : c'est ce qu'on a
        // demandé.
        const BenchEvent event = config_page(draft_, choices, bench_, actions);
        if ((event == BenchEvent::Applied || event == BenchEvent::Undone) && draft_ != running)
          result.reopen = draft_;
      } else if (name == i18n::tr("Application")) {
        language_section(draft_);
        default_player_page(choices.place, system_, choices.default_player_is_ajp);
        updates_page(draft_, choices.place, updates_);
      } else if (name == "Upscaling") {
        image_tab(draft_, choices);
        if (advanced_) {
          span_prepare_section(choices.span_prepare, choices.span_job ? *choices.span_job : prepare_);
          composition_section(draft_);
        }
      } else if (name == "Smoothness") {
        fluidity_tab(draft_, choices);
      } else if (name == "HDR") {
        hdr_tab(draft_, choices);
        if (advanced_) oled_sections(draft_, choices);
      } else if (name == "Colors") {
        grade_tab(draft_, choices);
      } else if (name == "Output") {
        delay_held = sound_tab(draft_, choices);
      } else if (name == "Playback") {
        languages_tab(draft_);
        subtitles_tab(draft_);
        episodes_section(draft_);
      } else if (name == i18n::tr("Export")) {
        export_tab(*export_, running, choices, actions);
      } else if (name == "Engines") {
        engines_tab(choices, engine_draft_, actions, draft_.engines_dir);
      } else if (name == "Statistics") {
        stats_tab(draft_);
      } else if (name == "Reset") {
        reset_page(actions, choices.profiles);
      }
    }
    panel::end_page();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::Checkbox(i18n::tr("Advanced settings"), &advanced_);
    // Un réglage de la chaîne fait à la main : le préréglage devient Personnalisé (les valeurs restent).
    if (draft_.preset != Preset::Custom && draft_.preset == before.preset &&
        preset_chain_differs(before, draft_))
      draft_.preset = Preset::Custom;
    ImGui::PopItemWidth();

    // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
    PlaySettings live = live_part(running, draft_, choices.rife_loaded);
    // Chaque délai recale le son (il reprend les blocs en file) : un glissement hacherait le son à
    // chaque image, le délai s'applique donc au relâchement (saisie au clavier comprise).
    if (delay_held) live.audio_delay_ms = running.audio_delay_ms;
    if (live != running) result.live = live;
    const Change pending = change_between(live, draft_, choices.rife_loaded);
    pending_shown_ = pending.cost == ChangeCost::Reopen;
    if (pending.cost == ChangeCost::Reopen) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped(
          "%s",
          i18n::tr_format("To apply: {}. Closing the panel discards these changes.", pending.reason).c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button(i18n::tr("Apply (reopens the video)"))) result.reopen = draft_;
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Discard changes"))) draft_ = live;
    }
  }
  ImGui::End();
  if (!keep) open_ = false;
  return result;
}

}  // namespace ajp::app::play
