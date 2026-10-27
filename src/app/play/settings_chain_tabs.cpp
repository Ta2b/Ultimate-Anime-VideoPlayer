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
    {0, "Source (sans réduction)"},
    {900, "900 lignes"},
    {720, "720 lignes (recommandé)"},
    {540, "540 lignes"},
    {480, "480 lignes"},
}};

}  // namespace

void image_tab(PlaySettings& d, const SettingsChoices& choices) {
  section("Agrandissement");
  ImGui::Checkbox("SPAN ×2 (AnimeJaNai)", &d.upscale);
  ImGui::BeginDisabled(!d.upscale);
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  if (ImGui::BeginCombo("Modèle", model_label(current).c_str())) {
    for (const std::string& model : choices.models) {
      const bool chosen = model == current;
      if (ImGui::Selectable(model_label(model).c_str(), chosen))
        d.model = model == choices.default_model ? std::string() : model;
      if (chosen) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  note("Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets.");

  section("Réduction avant l'agrandissement");
  // La source sans réduction, avec sa hauteur quand on la connaît : « 1080 lignes », pas « 4K », que
  // la sortie n'atteint qu'agrandie par SPAN.
  const auto label_of = [&](const HeightChoice& choice) {
    if (choice.lines == 0 && choices.source_height > 0)
      return std::format("Source, sans réduction ({} lignes)", choices.source_height);
    return std::string(choice.label);
  };
  const auto height = std::ranges::find(kHeights, d.work_height, &HeightChoice::lines);
  const std::string height_label =
      height != kHeights.end() ? label_of(*height) : std::format("{} lignes", d.work_height);
  if (ImGui::BeginCombo("Hauteur de travail", height_label.c_str())) {
    for (const HeightChoice& choice : kHeights) {
      if (ImGui::Selectable(label_of(choice).c_str(), choice.lines == d.work_height))
        d.work_height = choice.lines;
    }
    ImGui::EndCombo();
  }
  const bool reduces = d.work_height > 0 && d.work_height < choices.source_height;
  if (choices.source_height > 0) {
    const int work = reduces ? d.work_height : choices.source_height;
    note(d.upscale
             ? std::format("Sortie : {} lignes, le double de la hauteur de travail (SPAN ×2).", work * 2)
             : std::format("Sans agrandissement, la sortie garde la hauteur de travail : {} lignes.", work));
  }
  if (reduces) {
    if (ImGui::BeginCombo("Algorithme", std::string(label(d.downscale)).c_str())) {
      for (int i = 0; i < kDownscaleCount; ++i) {
        const auto filter = static_cast<Downscale>(i);
        if (ImGui::Selectable(std::string(label(filter)).c_str(), filter == d.downscale))
          d.downscale = filter;
      }
      ImGui::BeginDisabled();
      ImGui::Selectable("SSimDownscaler (à venir)", false);
      ImGui::EndDisabled();
      ImGui::EndCombo();
    }
    note(
        "Area : moyenne exacte. Spline36 : celui d'AnimeJaNai, net. Catmull-Rom et Mitchell gardent du "
        "détail ; Bilinear et Bicubic adoucissent. Changement en direct.");
  } else {
    note("Pas de réduction : la source entre telle quelle dans SPAN.");
  }
}

void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting) {
  // Sortie 4K : une source 1080p (ou plus) sans réduction, agrandie par SPAN.
  const bool four_k = d.upscale && (d.work_height == 0 || d.work_height >= choices.source_height) &&
                      choices.source_height >= 1080;
  section("Interpolation");
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
    if (factor.num <= factor.den) return std::string("Désactivée");
    std::string text = factor_text(factor);
    if (factor == Factor{.num = kDefaultRifeFactor, .den = 1}) text += " (recommandé)";
    return text;
  };
  // Grisé : plus d'images par seconde que l'écran n'en affiche (le choix en cours reste visible).
  const auto too_fast = [&](Factor factor) {
    return choices.source_fps > 0.0 && choices.display_hz > 0.0 && factor.num > factor.den &&
           choices.source_fps * static_cast<double>(factor.num) / static_cast<double>(factor.den) >
               1.01 * choices.display_hz;
  };
  const Factor now = factor_of(d);
  const std::string factor_shown = d.rife_auto && now.num > now.den ? std::string("Auto") : factor_label(now);
  if (ImGui::BeginCombo("Multiplicateur", factor_shown.c_str())) {
    if (!exporting && ImGui::Selectable("Auto", d.rife_auto && now.num > now.den)) {
      d.rife_auto = true;
      if (now.num <= now.den) d.rife_factor = 6;  // le plafond du mode auto
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
  if (d.rife_auto && now.num > now.den) {
    ImGui::Checkbox("Écran à fréquence variable (G-Sync, FreeSync)", &d.rife_auto_vrr);
    note(
        "Auto choisit seul le multiplicateur de ×1 à ×6 (24 à 144 i/s pour 23,976) : il baisse dès que le "
        "GPU prend du retard, remonte après quelques secondes sans retard, et réessaie de plus en plus "
        "tard un palier qui vient d'échouer. Sans écran à fréquence variable, seuls les paliers qui tiennent "
        "un nombre entier de rafraîchissements sont permis (sinon du judder) : sur 165 Hz fixe, Auto reste "
        "à ×1. Lecture seulement ; l'export garde un multiplicateur fixe.");
  }
  if (choices.source_fps > 0.0 && choices.display_hz > 0.0)
    note(std::format("Grisés : au-delà de l'écran ({:.0f} Hz pour {:.3f} i/s).", choices.display_hz,
                     choices.source_fps)
             .c_str());
  if (exporting && now.num > now.den) {
    note(
        "À l'export, aucune image n'est sautée : chaque image créée coûte une passe, le temps de l'export "
        "suit.");
  } else if (now.num > 4 * now.den) {
    note(
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un écran assez "
        "rapide (×8 en 24 i/s donne 192 Hz) et un GPU qui suit, sinon des images sont sautées.");
  }
  ImGui::EndDisabled();
  if (!choices.rife_available)
    note(
        "Ni modèle RIFE (rife_v4.25.onnx, rife_v4.26.onnx, rife_v4.22_lite.onnx…) ni graphes GMFSS trouvés.");
  section("Moteur de l'interpolation");
  for (int i = 0; i < kMotionEngineCount; ++i) {
    const auto engine = static_cast<MotionEngine>(i);
    if (!offered(engine) && d.motion_engine != engine) continue;
    if (ImGui::RadioButton(std::string(label(engine)).c_str(), d.motion_engine == engine))
      d.motion_engine = engine;
  }
  if (!choices.rife_models.empty()) {
    ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
    const std::string current = d.rife_model.empty() ? choices.default_rife_model : d.rife_model;
    std::string missing;
    if (ImGui::BeginCombo("Modèle RIFE", rife_label(current).c_str())) {
      for (const RifeEntry& entry : rife_menu(choices.rife_models)) {
        const bool chosen = entry.file == current;
        std::string text = rife_label(entry.file);
        if (!entry.present) text += " · absent";
        if (entry.file == choices.default_rife_model) text += " · conseillé pour cette carte";
        if (ImGui::Selectable(text.c_str(), chosen, entry.present ? 0 : ImGuiSelectableFlags_Disabled))
          d.rife_model = entry.file == choices.default_rife_model ? std::string() : entry.file;
        if (chosen) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    for (const RifeEntry& entry : rife_menu(choices.rife_models)) {
      if (!entry.present) missing += std::format("{}{}", missing.empty() ? "" : ", ", entry.file);
    }
    if (!missing.empty() && d.motion_engine == MotionEngine::Rife) {
      note(std::format("Absents de {} : {} (exports ONNX de vs-mlrt convertis en FP16, docs/handoff-pc.md).",
                       choices.rife_folder, missing));
    }
  }
  if (is_gmfss(d.motion_engine)) {
    const bool union_model = d.motion_engine == MotionEngine::GmfssUnion;
    note(
        "Expérimental. GMFSS Fortuna (réseau fait pour l'animation) donne les images intermédiaires les plus "
        "propres, pour plusieurs fois le coût de RIFE. Il prend toujours ses graphes au flot à demi-échelle "
        "(_s05) : mesuré, ils valent le flot entier ou le dépassent pour moitié moins de calcul, et 720 "
        "lignes de travail tiennent en ×2 sur une 4070 Ti. Ses deux moteurs se construisent la première fois "
        "(plusieurs minutes).");
    const bool quality_present =
        union_model ? choices.gmfss_union_quality_available : choices.gmfss_quality_available;
    ImGui::BeginDisabled(!quality_present);
    if (ImGui::BeginCombo("GMFSS", d.gmfss_quality && quality_present ? "Qualité" : "Performance")) {
      if (ImGui::Selectable("Performance", !d.gmfss_quality)) d.gmfss_quality = false;
      if (ImGui::Selectable("Qualité", d.gmfss_quality)) d.gmfss_quality = true;
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    note(
        "Performance : le flot de GMFlow se calcule en FP16 (base 26 ms et union 37 ms par image "
        "intermédiaire "
        "à 720p sur une 4070 Ti). Qualité : GMFlow en FP32, un écart à l'original divisé par 4 environ, pour "
        "+4 ms avec base et +20 ms avec union. Recharge la lecture.");
    if (!quality_present) {
      note(
          std::format("Mode qualité grisé : {} manque dans {} (export_gmfss.py --scale 0.5 --precision "
                      "flownet32).",
                      gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true, true).pair,
                      choices.rife_folder));
    }
    if (union_model) {
      note(
          "Union mêle à GMFSS l'image d'un RIFE intégré au réseau : moins de défauts sur les mouvements "
          "rapides, pour un peu plus de calcul à chaque image intermédiaire.");
    }
    if (!(union_model ? choices.gmfss_union_available : choices.gmfss_available)) {
      const GmfssFiles files = gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true);
      note(
          std::format("Graphes introuvables : {} et {}, à exporter dans {} (tools/gmfss/export_gmfss.py{} "
                      "--scale 0.5 --width 256 --height 256) ; RIFE prend la place.",
                      files.pair, files.fusion, choices.rife_folder, union_model ? " --model union" : ""));
    }
  }
  if (d.motion_engine == MotionEngine::DistilDrba) {
    note(
        "DistilDRBA : un RIFE à trois images entraîné à imiter DRBA. Le réseau garde lui-même les poses d'un "
        "personnage animé en deux pendant que le fond s'interpole, pour à peu près le coût de RIFE 4.22 "
        "lite par image créée. Il lit l'image d'après la paire : une image de retard.");
    if (!choices.distil_drba_available) {
      note(
          std::format("Graphe introuvable : {}, à exporter dans {} (tools/drba/export_distil_drba.py) ; RIFE "
                      "prend la place.",
                      kDistilDrbaFile, choices.rife_folder));
    }
  }
  if (d.motion_engine == MotionEngine::Apple) {
    note(
        "L'interpolation d'Apple : le réseau de flot optique de macOS (VideoToolbox, macOS 15.4 ou plus), "
        "sur le Neural Engine et le GPU de la puce, sans modèle à télécharger. Sous le backend Core ML "
        "seulement ; ailleurs, ou sur un Mac qui ne l'offre pas, RIFE prend la place. Essayée la première "
        "fois pour cette taille.");
  }
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(
        "Sortie 4K (source sans réduction agrandie par SPAN) : ×3 et plus demandent un GPU haut de gamme ; "
        "SPAN 1080p → 2160p est la passe la plus lourde (11 ms par image sur une RTX 4070 Ti). Une hauteur "
        "de travail de 720 lignes allège tout.");
  } else if (four_k) {
    note("Sortie 4K : RIFE estime le mouvement sur une copie en 720 lignes, plus légère.");
  }
  note("Les coupes de scène et les dessins tenus ne sont pas interpolés.");
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox("Garder les poses (DRBA) avec RIFE", &d.keep_poses);
  ImGui::EndDisabled();
  note(
      "Un personnage animé en deux sur un fond qui défile garde son rythme (il tient sa pose puis change "
      "d'un coup) pendant que le fond s'interpole, au lieu de se déformer entre deux dessins. Avec RIFE 4.25 "
      "ou 4.22 lite, une seule passe par paire. En direct. DistilDRBA le fait de lui-même.");
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && !is_gmfss(d.motion_engine));
  if (ImGui::BeginCombo("Warp et SPAN", std::string(label(d.span_order)).c_str())) {
    for (int i = 0; i < kSpanOrderCount; ++i) {
      const auto order = static_cast<SpanOrder>(i);
      if (ImGui::Selectable(std::string(label(order)).c_str(), order == d.span_order)) d.span_order = order;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  note(
      "Après SPAN : SPAN agrandit chaque image source une seule fois, puis RIFE déforme les images "
      "agrandies : en ×3, une passe de SPAN au lieu de trois. Avec GMFSS, les images agrandies sont "
      "splattées par son flot et corrigées du résidu de sa fusion : le détail de SPAN reste. Avant SPAN, "
      "SPAN nettoie aussi chaque image intermédiaire. Auto choisit Après dès que la sortie dépasse 1080p. "
      "Recharge la lecture.");
  // Smooth Motion est une couche de l'écran, absente du fichier, et du seul pilote NVIDIA.
  if (exporting || !choices.nvidia) return;
  section("Smooth Motion (pilote NVIDIA)");
  // Sans RIFE la lecture cale (16 images neuves en 8 s contre 192) : l'option reste éteinte.
  const bool without_rife = d.rife_factor < 2;
  if (without_rife) d.smooth_motion = false;
  ImGui::BeginDisabled(without_rife);
  ImGui::Checkbox("Doubler les images après RIFE", &d.smooth_motion);
  ImGui::EndDisabled();
  note(
      "Expérimental. La couche du pilote NVIDIA (RTX 40 et 50) génère une image entre deux images "
      "présentées : RIFE ×3 puis Smooth Motion donnent environ 144 images par seconde. Vaut au prochain "
      "lancement. Sans RIFE, la lecture peut caler : gardez RIFE à ×2 ou plus.");
  if (without_rife) note("Grisé : activez RIFE à ×2 ou plus, sans quoi la lecture cale.");
}

}  // namespace ajp::app::play
