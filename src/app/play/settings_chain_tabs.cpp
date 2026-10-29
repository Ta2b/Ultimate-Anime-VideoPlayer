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
  const auto text = [](PrecisionTier t) { return std::format("{} · {}", label(t), precision_text(t)); };
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
      return "RIFE : le moteur de base, rapide, bon partout.";
    case MotionEngine::Gmfss:
      return "GMFSS Fortuna : réseau fait pour l'animation, les images intermédiaires les plus propres, pour "
             "plusieurs "
             "fois le coût de RIFE. Il prend ses graphes au flot à demi-échelle (_s05), qui valent le flot "
             "entier pour "
             "moitié moins de calcul. Construit la première fois (plusieurs minutes). Expérimental.";
    case MotionEngine::GmfssUnion:
      return "GMFSS Union : GMFSS mêlé à l'image d'un RIFE intégré au réseau, moins de défauts sur les "
             "mouvements "
             "rapides, pour un peu plus de calcul. Construit la première fois (plusieurs minutes).";
    case MotionEngine::DistilDrba:
      return "DistilDRBA : un RIFE à trois images entraîné à imiter DRBA ; le réseau garde lui-même les "
             "poses d'un "
             "personnage animé en deux, pour à peu près le coût de RIFE 4.22 lite. Une image de retard.";
    case MotionEngine::Apple:
      return "Apple : le flot optique de macOS (VideoToolbox, macOS 15.4 ou plus), sur le Neural Engine et "
             "le GPU, sans "
             "modèle à télécharger. Sous le backend Core ML seulement ; ailleurs RIFE prend la place.";
  }
  return {};
}

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
  panel::tip("Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets.");
  if (choices.span_tiers.performance || choices.span_tiers.quality) {
    precision_combo("Précision de SPAN", d.span_precision, choices.span_tiers);
    panel::tip(
        "Équilibre : FP16, le réseau tel que livré. Performance : le plus rapide (calcul en FP8 sur les RTX "
        "40 "
        "et après), fabriqué sur votre PC par le bouton « Préparer le mode Performance » des réglages "
        "avancés ; il n'apparaît dans la liste qu'une fois prêt. Qualité : FP32, si sa variante est "
        "installée. Recharge la lecture.");
  }
  ImGui::EndDisabled();
  if (panel::advanced() && !choices.span_tiers.performance) {
    panel::note(
        "Performance (FP8) : fabriqué sur votre PC (bouton « Préparer le mode Performance » plus bas), pour "
        "les "
        "RTX 40 et après.");
  }

  section("Résolution");
  // La source sans réduction, avec sa hauteur quand on la connaît : « 1080 lignes », pas « 4K », que
  // la sortie n'atteint qu'agrandie par SPAN.
  const auto label_of = [&](const HeightChoice& choice) {
    if (choice.lines == 0 && choices.source_height > 0)
      return std::format("Source, sans réduction ({})", resolution_text(choices.source_height));
    return std::string(choice.label);
  };
  const auto height = std::ranges::find(kHeights, d.work_height, &HeightChoice::lines);
  const std::string height_label =
      height != kHeights.end() ? label_of(*height) : resolution_text(d.work_height);
  if (ImGui::BeginCombo("Résolution de travail", height_label.c_str())) {
    for (const HeightChoice& choice : kHeights) {
      if (ImGui::Selectable(label_of(choice).c_str(), choice.lines == d.work_height))
        d.work_height = choice.lines;
    }
    ImGui::EndCombo();
  }
  panel::tip(
      "Taille à laquelle la vidéo est ramenée avant SPAN. SPAN la double : 720p donne une sortie en 1440p.");
  if (d.work_height == 720) panel::badge(std::string(badge_text(Badge::Recommended)).c_str(), panel::kGood);
  const bool reduces = d.work_height > 0 && d.work_height < choices.source_height;
  if (choices.source_height > 0) {
    const int work = reduces ? d.work_height : choices.source_height;
    note(d.upscale ? std::format("Sortie : {}, le double de la résolution de travail (SPAN ×2).",
                                 resolution_text(work * 2))
                   : std::format("Sans agrandissement, la sortie garde la résolution de travail : {}.",
                                 resolution_text(work)));
  }
  if (reduces && panel::advanced()) {
    if (ImGui::BeginCombo("Algorithme", std::string(label(d.downscale)).c_str())) {
      for (int i = 0; i < kDownscaleCount; ++i) {
        const auto filter = static_cast<Downscale>(i);
        std::string text(label(filter));
        if (const auto mark = badge_of(filter)) text += " · " + std::string(badge_text(*mark));
        if (ImGui::Selectable(text.c_str(), filter == d.downscale)) d.downscale = filter;
      }
      ImGui::BeginDisabled();
      ImGui::Selectable("SSimDownscaler (à venir)", false);
      ImGui::EndDisabled();
      ImGui::EndCombo();
    }
    panel::tip(
        "Area : moyenne exacte. Spline36 : celui d'AnimeJaNai, net. Hermite : net et sans halo, pensé pour "
        "l'animation. "
        "Catmull-Rom et Mitchell gardent du détail ; Bilinear et Bicubic adoucissent. Changement en direct.");
  } else if (!reduces) {
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
      return cap == 0 ? std::string("Maximum (écran)") : std::format("{} i/s", cap);
    };
    if (ImGui::BeginCombo("Plafond d'Auto", cap_label(d.auto_cap).c_str())) {
      for (const int cap : {0, 120, 60}) {
        if (ImGui::Selectable(cap_label(cap).c_str(), d.auto_cap == cap)) d.auto_cap = cap;
      }
      ImGui::EndCombo();
    }
    panel::tip(
        "La cadence de sortie la plus haute que le mode Auto vise : le maximum de l'écran par défaut, ou 120 "
        "/ 60 "
        "i/s pour ménager le GPU. Il baisse toujours de lui-même quand une scène est lourde.");
  }
  if (d.rife_auto && now.num > now.den && !exporting) {
    // Mode Rapide : sur un écran VRR armé, chaque image part dès qu'elle est prête ; sans VRR, les instants
    // sont calés sur les vsyncs de l'écran (comme un jeu en vsync). Dans les deux cas la lecture repasse sur
    // Stable si trop d'images sont perdues.
    const bool vrr_armed = choices.display_vrr && choices.display_vrr_armed;
    const char* const shown = d.auto_free ? "Rapide (expérimental)" : "Stable";
    if (ImGui::BeginCombo("Mode d'Auto", shown)) {
      if (ImGui::Selectable("Stable", !d.auto_free)) d.auto_free = false;
      if (ImGui::Selectable("Rapide (expérimental)", d.auto_free)) d.auto_free = true;
      panel::tip(vrr_armed ? "Aucun palier : chaque image est calculée à l'instant exact où elle sera vue."
                           : "Sans VRR, la cadence est calée sur le rafraîchissement de l'écran : une image "
                             "trop tardive laisse la précédente un vsync de plus.");
      ImGui::EndCombo();
    }
    panel::tip(
        "Stable : paliers de multiplicateur qui suivent ce que le GPU tient (de ×1 à ×6), avec 15 % de marge "
        "; "
        "le palier baisse en moins d'une seconde quand une scène est lourde et remonte dès qu'elle s'allège. "
        "Rapide (expérimental) : aucun palier, chaque image est calculée à l'instant exact où elle sera "
        "affichée, dès que le GPU est libre, sans dépasser l'écran ni le plafond ; la vitesse de la vidéo "
        "reste constante. Si trop d'images sont perdues (3 % sur 3 s), la lecture repasse sur Stable avec un "
        "message. Lecture seulement : l'export garde un multiplicateur fixe.");
  }
  if (d.rife_auto && now.num > now.den) {
    if (!choices.display_vrr) {
      note("Sans VRR, le mouvement peut vibrer à fréquence fixe.");
      panel::tip(
          "VRR indisponible : à fréquence fixe, un multiplicateur qui ne tombe pas sur un nombre entier de "
          "rafraîchissements (×k de 23,976 sur 165 Hz) fait vibrer le mouvement (judder). Active G-Sync / "
          "FreeSync / Adaptive Sync pour l'éviter.");
    } else if (choices.window_fullscreen && !choices.display_vrr_armed) {
      note("VRR coupé par le système : le mouvement peut vibrer.");
      panel::tip(
          "VRR présent mais coupé par le système : active la synchronisation adaptative (KDE : Affichage et "
          "écran > Synchronisation adaptative), sinon le mouvement vibre à fréquence fixe.");
    }
  }
  if (choices.source_fps > 0.0 && choices.display_hz > 0.0) {
    std::string fps = std::format("{:.3f}", choices.source_fps);
    std::ranges::replace(fps, '.', ',');  // la virgule décimale du français : « 23,976 »
    note(std::format("Grisé : au-delà de l'écran ({:.0f} Hz pour {} i/s).", choices.display_hz, fps));
  }
  if (exporting && now.num > now.den) {
    note(
        "À l'export, aucune image n'est sautée : chaque image créée coûte une passe, le temps de l'export "
        "suit.");
  } else if (now.num > 4 * now.den) {
    note(
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un écran rapide et un "
        "GPU qui suit, sinon des images sont sautées.");
  }
  ImGui::EndDisabled();
  if (!choices.rife_available)
    note(
        "Aucun modèle d'interpolation trouvé : placez un modèle RIFE ou les graphes GMFSS dans le dossier "
        "des modèles.");
  section("Moteur de l'interpolation");
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
    if (ImGui::BeginCombo("Modèle RIFE", rife_label(current).c_str())) {
      for (const RifeEntry& entry : rife_menu(choices.rife_models)) {
        const bool chosen = entry.file == current;
        if (!entry.present) continue;  // un modèle absent n'est pas une option
        std::string text = rife_label(entry.file);
        if (entry.file == choices.default_rife_model) text += " · conseillé pour cette carte";
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
    precision_combo("GMFSS", d.gmfss_precision, tiers);
    panel::tip(
        "Précision de GMFSS. Performance : la paire en FP8 hors du squelette de GMFlow (15 convolutions sur "
        "37), −15 % sur la paire, 47 dB au pire cas, RTX 40 et après. Équilibre : FP16. Qualité : FP32, un "
        "écart à l'original divisé par 4 environ, pour +7 ms avec base et +10 ms avec union. Recharge la "
        "lecture.");
    if (!tiers.performance) {
      note(std::format("Cran Performance (FP8) : ajouter {} (release des modèles) dans {}.",
                       gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true,
                                   PrecisionTier::Performance)
                           .pair,
                       choices.rife_folder));
    }
    if (union_model) {
      note(
          "Union mêle à GMFSS l'image d'un RIFE intégré au réseau : moins de défauts sur les mouvements "
          "rapides, pour un peu plus de calcul à chaque image intermédiaire.");
    }
    if (!(union_model ? choices.gmfss_union_available : choices.gmfss_available)) {
      const GmfssFiles files = gmfss_files(union_model ? GmfssVariant::Union : GmfssVariant::Base, true);
      note(std::format("Modèles GMFSS introuvables dans {} ({} et {}) ; RIFE prend la place.",
                       choices.rife_folder, files.pair, files.fusion));
    }
  }
  if (d.motion_engine == MotionEngine::DistilDrba && !choices.distil_drba_available) {
    note(std::format("Modèle DistilDRBA introuvable dans {} ({}) ; RIFE prend la place.", choices.rife_folder,
                     kDistilDrbaFile));
  }
  if (exporting) {
    // Les limites de la lecture en direct ne valent pas à l'export.
  } else if (four_k && now.num >= 3 * now.den) {
    note(
        "Sortie 4K (source sans réduction agrandie par SPAN) : ×3 et plus demandent un GPU haut de gamme ; "
        "SPAN 1080p → 2160p est la passe la plus lourde. Une "
        "résolution "
        "de travail de 720p allège tout.");
  } else if (four_k) {
    note("Sortie 4K : RIFE estime le mouvement sur une copie en 720p, plus légère.");
  }
  note("Les coupes de scène et les dessins tenus ne sont pas interpolés.");
  if (!panel::advanced()) return;  // poses, ordre du traitement et Smooth Motion : réglages avancés
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox("Garder les poses (DRBA) avec RIFE", &d.keep_poses);
  ImGui::EndDisabled();
  panel::tip(
      "Un personnage animé en deux sur un fond qui défile garde son rythme (il tient sa pose puis change "
      "d'un coup) pendant que le fond s'interpole, au lieu de se déformer entre deux dessins. Avec RIFE 4.25 "
      "ou 4.22 lite, une seule passe par paire. En direct. DistilDRBA le fait de lui-même.");
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
  ImGui::Checkbox("RIFE fusionné", &d.rife_fused);
  ImGui::EndDisabled();
  panel::tip(
      "Coupe RIFE 4.25 ou 4.22 lite en six petits moteurs TensorRT (les convolutions à basse résolution) "
      "reliés par "
      "des noyaux CUDA pour tout ce qui travaille en pleine résolution : une passe de RIFE prend un tiers du "
      "temps "
      "(2,2 ms au lieu de 6,5 ms à 768×1280 sur une RTX 4070 Ti), avec la même image à moins d'un dB près. "
      "Sous "
      "TensorRT seulement ; le RIFE entier reprend si le modèle ne se coupe pas ou si un moteur échoue. "
      "Recharge la "
      "lecture ; la première fois, six petits moteurs se construisent pour cette taille.");
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::DistilDrba);
  ImGui::Checkbox("DistilDRBA fusionné", &d.drba_fused);
  ImGui::EndDisabled();
  panel::tip(
      "Coupe DistilDRBA en quatre petits moteurs TensorRT (les convolutions à basse résolution) reliés par "
      "des "
      "noyaux CUDA pour tout le reste (réductions d'images, warps, mélange) : une passe de DistilDRBA prend "
      "moins de la moitié du temps (0,92 ms au lieu de 2,10 ms à 1088×1920 sur une RTX 4070 Ti), avec la "
      "même "
      "image à moins d'un dB près. Sous TensorRT seulement, au cran Équilibre ; le DistilDRBA entier reprend "
      "si "
      "le modèle ne se coupe pas ou si un moteur échoue. Recharge la lecture ; la première fois, quatre "
      "petits "
      "moteurs se construisent pour cette taille.");
  ImGui::BeginDisabled(!is_gmfss(d.motion_engine));
  ImGui::Checkbox("GMFSS fusionné", &d.gmfss_fused);
  ImGui::EndDisabled();
  panel::tip(
      "Coupe la paire de GMFSS en deux moteurs TensorRT : les caractéristiques d'une image et le flot. Les "
      "caractéristiques d'une image servent à deux paires de suite : elles ne sont calculées qu'une fois, un "
      "FeatureNet de moins par paire (docs/gmfss-fusionne.md). Sous TensorRT seulement, au cran Équilibre ; "
      "la "
      "paire entière reprend si le graphe ne se coupe pas ou si un moteur échoue. Recharge la lecture ; la "
      "première "
      "fois, deux moteurs se construisent pour cette taille.");
  ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife && !is_gmfss(d.motion_engine));
  if (ImGui::BeginCombo("Warp et SPAN", std::string(label(d.span_order)).c_str())) {
    for (int i = 0; i < kSpanOrderCount; ++i) {
      const auto order = static_cast<SpanOrder>(i);
      if (ImGui::Selectable(std::string(label(order)).c_str(), order == d.span_order)) d.span_order = order;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(
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
  ImGui::Checkbox("Doubler les images (pilote NVIDIA)", &d.smooth_motion);
  panel::badge(std::string(badge_text(Badge::Experimental)).c_str(), panel::kExperimental);
  ImGui::EndDisabled();
  note(
      "La couche du pilote NVIDIA (RTX 40 et 50) génère une image entre deux images présentées : RIFE ×3 "
      "puis "
      "Smooth Motion donnent environ 144 images par seconde. Vaut au prochain lancement.");
  if (without_rife) note("Grisé : activez RIFE à ×2 ou plus, sans quoi la lecture cale.");
}

}  // namespace ajp::app::play
