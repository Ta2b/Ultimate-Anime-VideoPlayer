#include "app/play/settings_chain_tabs.h"

#include <algorithm>
#include <array>
#include <format>
#include <string>

#include <imgui.h>

#include "app/play/drba_model.h"
#include "app/play/gmfss_model.h"
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
constexpr std::array<HeightChoice, 5> kHeights{{
    {0, "Source (no downscale)"},  // clé i18n
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
    return std::format("{} · {}", i18n::tr(label(t).data()), precision_text(t));
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

std::string motion_description(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
      return i18n::tr("RIFE: the default engine, fast and good everywhere.");
    case MotionEngine::Gmfss:
      return i18n::tr(
          "GMFSS Fortuna: built for animation, the cleanest in-between frames at several times RIFE's cost. "
          "Built on first use (several minutes). Experimental.");
    case MotionEngine::GmfssUnion:
      return i18n::tr(
          "GMFSS Union: GMFSS blended with a built-in RIFE, fewer artifacts on fast motion, a bit slower. "
          "Built on first use.");
    case MotionEngine::DistilDrba:
      return i18n::tr(
          "DistilDRBA: a three-frame RIFE trained to imitate DRBA; it holds poses by itself, at about RIFE "
          "4.22 lite's cost. One frame of delay.");
    case MotionEngine::Apple:
      return i18n::tr(
          "Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend "
          "only; RIFE is used elsewhere.");
  }
  return {};
}

}  // namespace

void image_tab(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Upscaling"));
  ImGui::Checkbox(i18n::tr("SPAN ×2 (AnimeJaNai)"), &d.upscale);
  ImGui::BeginDisabled(!d.upscale);
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  if (ImGui::BeginCombo(i18n::tr("Model"), model_label(current).c_str())) {
    for (const std::string& model : choices.models) {
      const bool chosen = model == current;
      if (ImGui::Selectable(model_label(model).c_str(), chosen))
        d.model = model == choices.default_model ? std::string() : model;
      if (chosen) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Performance: fastest. Balanced: finer, slower. Sharp1: crisper edges."));
  if (choices.span_tiers.performance || choices.span_tiers.quality) {
    precision_combo(i18n::tr("SPAN precision"), d.span_precision, choices.span_tiers);
    panel::tip(i18n::tr(
        "Balanced is FP16; Performance is FP8 on RTX 40 and newer; Quality is FP32. Reloads playback."));
  }
  ImGui::EndDisabled();
  if (panel::advanced() && !choices.span_tiers.performance) {
    panel::note(
        i18n::tr("Performance (FP8) is built on your PC (see \"Prepare Performance mode\" below), for RTX 40 "
                 "and newer."));
  }

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
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("The size the video is reduced to before SPAN, which doubles it: 720p gives 1440p."));
  if (d.work_height == 720) panel::badge(std::string(badge_text(Badge::Recommended)).c_str(), panel::kGood);
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
      ImGui::BeginDisabled();
      ImGui::Selectable(i18n::tr("SSimDownscaler (coming soon)"), false);
      ImGui::EndDisabled();
      ImGui::EndCombo();
    }
    panel::tip(
        i18n::tr("Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live."));
  } else if (!reduces) {
    note(i18n::tr("No downscale: the source goes into SPAN as is."));
  }
}

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
      const ImGuiSelectableFlags flags = too_fast(factor) && !chosen ? ImGuiSelectableFlags_Disabled : 0;
      if (ImGui::Selectable(factor_label(factor).c_str(), chosen, flags)) {
        d.rife_factor = factor.num;
        d.rife_factor_den = factor.den;
        d.rife_auto = false;
      }
    }
    ImGui::EndCombo();
  }
  if (d.rife_auto && now.num > now.den && !exporting) {
    // Le plafond de la cadence : le maximum de l'écran par défaut, ou une valeur plus basse pour ménager le
    // GPU.
    const auto cap_label = [](int cap) {
      return cap == 0 ? std::string(i18n::tr("Maximum (display)")) : std::format("{} i/s", cap);
    };
    if (ImGui::BeginCombo(i18n::tr("Auto limit"), cap_label(d.auto_cap).c_str())) {
      for (const int cap : {0, 120, 60}) {
        if (ImGui::Selectable(cap_label(cap).c_str(), d.auto_cap == cap)) d.auto_cap = cap;
      }
      ImGui::EndCombo();
    }
    panel::tip(i18n::tr("The highest frame rate Auto aims for. It still drops by itself on heavy scenes."));
  }
  if (d.rife_auto && now.num > now.den && !exporting) {
    // Mode Rapide : sur un écran VRR armé, chaque image part dès qu'elle est prête ; sans VRR, les instants
    // sont calés sur les vsyncs de l'écran (comme un jeu en vsync). Dans les deux cas la lecture repasse sur
    // Stable si trop d'images sont perdues.
    const bool vrr_armed = choices.display_vrr && choices.display_vrr_armed;
    const char* const shown = d.auto_free ? i18n::tr("Fast (experimental)") : i18n::tr("Stable");
    if (ImGui::BeginCombo(i18n::tr("Auto mode"), shown)) {
      if (ImGui::Selectable(i18n::tr("Stable"), !d.auto_free)) d.auto_free = false;
      if (ImGui::Selectable(i18n::tr("Fast (experimental)"), d.auto_free)) d.auto_free = true;
      if (vrr_armed) {
        panel::tip(i18n::tr("No steps: each frame is computed for the exact moment it is shown."));
      } else {
        panel::tip(i18n::tr(
            "Without VRR, frames follow the display refresh; a late frame repeats the previous one."));
      }
      ImGui::EndCombo();
    }
    panel::tip(
        i18n::tr("Stable follows what the GPU can hold in steps (×1 to ×6). Fast has no steps and falls back "
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
  if (choices.source_fps > 0.0 && choices.display_hz > 0.0) {
    std::string fps = std::format("{:.3f}", choices.source_fps);
    std::ranges::replace(fps, '.', ',');  // la virgule décimale du français : « 23,976 »
    note(i18n::tr_format("Greyed out: above the display ({:.0f} Hz for {} fps).", choices.display_hz, fps));
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
    note(
        i18n::tr("No interpolation model found: put a RIFE model or the GMFSS graphs in the models folder."));
  section(i18n::tr("Interpolation engine"));
  // Les moteurs se rangent sur une ligne tant qu'ils y tiennent (pastille comprise), sinon à la ligne.
  const float row_left = ImGui::GetCursorScreenPos().x;
  const float row_right = row_left + ImGui::GetContentRegionAvail().x;
  bool first_engine = true;
  for (int i = 0; i < kMotionEngineCount; ++i) {
    const auto engine = static_cast<MotionEngine>(i);
    if (!offered(engine) && d.motion_engine != engine) continue;
    if (!first_engine) {
      const ImGuiStyle& style = ImGui::GetStyle();
      const std::string name(label(engine));
      float need = ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize(name.c_str()).x;
      if (const auto mark = badge_of(engine))
        need += style.ItemSpacing.x + ImGui::CalcTextSize(std::string(badge_text(*mark)).c_str()).x + 12.0F;
      if (ImGui::GetItemRectMax().x + style.ItemSpacing.x * 2.0F + need <= row_right) ImGui::SameLine();
    }
    first_engine = false;
    if (ImGui::RadioButton(std::string(label(engine)).c_str(), d.motion_engine == engine))
      d.motion_engine = engine;
    panel::tip(motion_description(engine));
    if (const auto mark = badge_of(engine)) {
      // Une chaîne de « ? : » se met en page autrement d'une version de clang-format à l'autre : un if.
      ImVec4 color = panel::kExperimental;
      if (*mark == Badge::Recommended) {
        color = panel::kGood;
      } else if (*mark == Badge::Heaviest) {
        color = panel::kWarn;
      } else if (*mark == Badge::Finest) {
        color = panel::kFine;
      }
      panel::badge(std::string(badge_text(*mark)).c_str(), color);
    }
  }
  if (!choices.rife_models.empty() && panel::advanced()) {
    ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
    const std::string current = d.rife_model.empty() ? choices.default_rife_model : d.rife_model;
    if (ImGui::BeginCombo(i18n::tr("RIFE model"), rife_label(current).c_str())) {
      for (const RifeEntry& entry : rife_menu(choices.rife_models)) {
        const bool chosen = entry.file == current;
        if (!entry.present) continue;  // un modèle absent n'est pas une option
        std::string text = rife_label(entry.file);
        if (entry.file == choices.default_rife_model) text += i18n::tr(" · suggested for this card");
        if (ImGui::Selectable(text.c_str(), chosen))
          d.rife_model = entry.file == choices.default_rife_model ? std::string() : entry.file;
        if (chosen) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
  }
  if (is_gmfss(d.motion_engine) && panel::advanced()) {
    const bool union_model = d.motion_engine == MotionEngine::GmfssUnion;
    const TierSupport& tiers = union_model ? choices.gmfss_union_tiers : choices.gmfss_tiers;
    precision_combo(i18n::tr("GMFSS"), d.gmfss_precision, tiers);
    panel::tip(
        i18n::tr("GMFSS precision. Performance is FP8 on RTX 40 and newer (15% faster); Quality is FP32. "
                 "Reloads playback."));
    if (!tiers.performance) {
      note(i18n::tr_format("Performance (FP8): add {} (from the models release) to {}.",
                           gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true,
                                       PrecisionTier::Performance)
                               .pair,
                           choices.rife_folder));
    }
    if (union_model) {
      note(i18n::tr("Union adds a built-in RIFE to GMFSS: fewer artifacts on fast motion, a bit slower."));
    }
    if (!(union_model ? choices.gmfss_union_available : choices.gmfss_available)) {
      const GmfssFiles files = gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true);
      note(i18n::tr_format("GMFSS models not found in {} ({} and {}); RIFE is used instead.",
                           choices.rife_folder, files.pair, files.fusion));
    }
  }
  if (d.motion_engine == MotionEngine::DistilDrba && !choices.distil_drba_available) {
    note(i18n::tr_format("DistilDRBA model not found in {} ({}); RIFE is used instead.", choices.rife_folder,
                         kDistilDrbaFile));
  }
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(i18n::tr(
        "4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything."));
  } else if (four_k) {
    note(i18n::tr("4K output: RIFE estimates motion on a lighter 720p copy."));
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
      i18n::tr("Splits RIFE into small TensorRT engines: about three times faster. TensorRT only. Reloads "
               "playback; built on first use for this size."));
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::DistilDrba);
  ImGui::Checkbox(i18n::tr("Fused DistilDRBA"), &d.drba_fused);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Splits DistilDRBA into small TensorRT engines: more than twice as fast. TensorRT only, "
               "Balanced precision. Reloads playback; built on first use for this size."));
  ImGui::BeginDisabled(!is_gmfss(d.motion_engine));
  ImGui::Checkbox(i18n::tr("Fused GMFSS"), &d.gmfss_fused);
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Splits the GMFSS pair into two TensorRT engines so each frame's features are computed once. "
               "TensorRT only, Balanced precision. Reloads playback."));
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && !is_gmfss(d.motion_engine));
  if (ImGui::BeginCombo(i18n::tr("Warp and SPAN"), std::string(label(d.span_order)).c_str())) {
    for (int i = 0; i < kSpanOrderCount; ++i) {
      const auto order = static_cast<SpanOrder>(i);
      if (ImGui::Selectable(std::string(label(order)).c_str(), order == d.span_order)) d.span_order = order;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("After SPAN upscales each source frame once, then warps it: faster. Before SPAN also cleans "
               "each in-between frame. Auto picks After above 1080p. Reloads playback."));
  // Smooth Motion est une couche de l'écran, absente du fichier, et du seul pilote NVIDIA.
  if (exporting || !choices.nvidia) return;
  section(i18n::tr("Smooth Motion (NVIDIA driver)"));
  // Sans RIFE la lecture cale (16 images neuves en 8 s contre 192) : l'option reste éteinte.
  const bool without_rife = d.rife_factor < 2;
  if (without_rife) d.smooth_motion = false;
  ImGui::BeginDisabled(without_rife);
  ImGui::Checkbox(i18n::tr("Double the frames (NVIDIA driver)"), &d.smooth_motion);
  panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  ImGui::EndDisabled();
  note(i18n::tr(
      "The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames. Applies at next launch."));
  if (without_rife) note(i18n::tr("Greyed out: turn on RIFE at ×2 or more, or playback stalls."));
}

}  // namespace ajp::app::play
