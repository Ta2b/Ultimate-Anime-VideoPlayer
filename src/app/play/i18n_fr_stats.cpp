#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Statistiques : (clé anglaise, français).
constexpr std::array kStatsTexts{
    Entry{" (+ {} cuts and held drawings)", " (+ {} coupes et dessins tenus)"},
    Entry{" fused", " fusionné"},
    Entry{" · heavy scene", " · scène lourde"},
    Entry{" · locked to the display {:+.3f} %", " · calée sur l'écran {:+.3f} %"},
    Entry{" · predicted ×{}", " · prédit ×{}"},
    Entry{" · service gap ", " · écart service "},
    Entry{"AJP is in \"Open with\" ({}). To make it the default: \"Make this the default player\" in "
          "Settings > General > Application, or xdg-mime default {}{}",
          "AJP figure dans « Ouvrir avec » ({}). Pour en faire le lecteur par défaut : « Faire de ce lecteur "
          "le lecteur par défaut » dans Paramètres > Général > Application, ou xdg-mime default {}{}"},
    Entry{"AJP is in \"Open with\". To make it the default: \"Make this the default player\" in Settings > "
          "General > Application, or Settings > Apps > Default apps > {}.",
          "AJP figure dans « Ouvrir avec ». Pour en faire le lecteur par défaut : « Faire de ce lecteur le "
          "lecteur par défaut » dans Paramètres > Général > Application, ou Paramètres > Applications > "
          "Applications par défaut > {}."},
    Entry{"All files", "Tous les fichiers"},
    Entry{"Auto ×{} ({:.0f} fps) · display limit ×{}", "Auto ×{} ({:.0f} i/s) · plafond écran ×{}"},
    Entry{"Exporting · Esc to cancel", "Export en cours · Échap pour annuler"},
    Entry{"Last optimization: {}", "Dernière optimisation : {}"},
    Entry{"Light", "Léger"},
    Entry{"Open a video: the player prepares Performance mode on its frames.",
          "Ouvrez une vidéo : le lecteur prépare le mode Performance sur ses images."},
    Entry{"Preparing RTX HDR", "Préparation de RTX HDR"},
    Entry{"True black: 0 below {:.4f} nit, toe up to {:.4f} nit (source black was {:.4f} nit).",
          "Vrai noir : 0 sous {:.4f} nit, pied jusqu'à {:.4f} nit (noir de la source à {:.4f} nit avant)."},
    Entry{"True black: no effect here (grading off, or HDR output without a path).",
          "Vrai noir : sans effet ici (étalonnage coupé, ou sortie HDR sans voie)."},
    Entry{"True black: off.", "Vrai noir : coupé."},
    Entry{"True black: the ramp does not rise, nothing changed.",
          "Vrai noir : la rampe ne monte pas, rien n'est changé."},
    Entry{"Video ({})", "Vidéo ({})"},
    Entry{"Videos", "Vidéos"},
    Entry{"audio missing: the display clock leads", "audio absent : l'horloge de l'écran mène"},
    Entry{"display refresh rate unknown", "fréquence de l'écran inconnue"},
    Entry{"frame {} (σ {}) · render {} · skipped {}{}", "image {} (σ {}) · rendu {} · sautées {}{}"},
    Entry{"input {}x{} · built in {:.0f} s", "entrée {}x{} · construit en {:.0f} s"},
    Entry{"no RIFE", "pas de RIFE"},
    Entry{"no upscaling · {}", "sans agrandissement · {}"},
    Entry{"smoothness {}{} fps · missed {}{} · p95 delay {}{}",
          "fluidité {}{} im/s · manquées {}{} · retard p95 {}{}"},
    Entry{"video memory {:.0f} / {:.0f} MiB", "mémoire vidéo {:.0f} / {:.0f} Mio"},
    Entry{"{} cannot compute at a free instant", "{} ne calcule pas à un instant libre"},
    Entry{"{} frames · {:.1f} fps · elapsed {} · left ~{}", "{} images · {:.1f} i/s · écoulé {} · reste ~{}"},
    Entry{"{} {}: {} pairs, {} interpolated, {} held drawings, {} cuts, {} late",
          "{} {} : {} paires, {} interpolées, {} dessins tenus, {} coupes, {} en retard"},
    Entry{"{}x{} at {} fps · {}", "{}x{} à {} i/s · {}"},
};

}  // namespace

std::span<const Entry> french_stats() {
  return kStatsTexts;
}

}  // namespace ajp::app::play::i18n
