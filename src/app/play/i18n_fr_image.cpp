#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Image IA : (clé anglaise, français).
constexpr std::array kImageTabs{
    Entry{" (recommended)", " (recommandé)"},
    Entry{" · suggested for this card", " · conseillé pour cette carte"},
    Entry{"4K output: RIFE estimates motion on a lighter 720p copy.",
          "Sortie 4K : RIFE estime le mouvement sur une copie 720p, plus légère."},
    Entry{"4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything.",
          "Sortie 4K : ×3 et plus demandent un GPU haut de gamme. Une résolution de travail de 720p allège "
          "tout."},
    Entry{
        "Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are skipped.",
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un GPU rapide, sinon "
        "des images sont sautées."},
    Entry{
        "After SPAN upscales each source frame once, then warps it: faster. Before SPAN also cleans each "
        "in-between frame. Auto picks After above 1080p. Reloads playback.",
        "Après SPAN agrandit chaque image source une fois, puis la déforme : plus rapide. Avant SPAN nettoie "
        "aussi chaque image intermédiaire. Auto choisit Après au-dessus de 1080p. Recharge la lecture."},
    Entry{"Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend "
          "only; RIFE is used elsewhere.",
          "Apple : le flot optique de macOS (VideoToolbox, macOS 15.4+), sans modèle à télécharger. Moteur "
          "Core ML seulement ; ailleurs RIFE prend la place."},
    Entry{"Auto", "Auto"},
    Entry{"Auto limit", "Plafond d'Auto"},
    Entry{"Auto mode", "Mode d'Auto"},
    Entry{"Balanced is FP16; Quality is FP32. Reloads playback.",
          "Équilibre : FP16 ; Qualité : FP32. Recharge la lecture."},
    Entry{"Characters animated on twos keep their rhythm while the background interpolates. Applies live; "
          "DistilDRBA does this itself.",
          "Un personnage animé « en deux » garde son rythme pendant que le fond s'interpole. En direct ; "
          "DistilDRBA le fait seul."},
    Entry{"DistilDRBA model not found in {} ({}); RIFE is used instead.",
          "Modèle DistilDRBA introuvable dans {} ({}) ; RIFE prend la place."},
    Entry{"Double the frames (NVIDIA driver)", "Doubler les images (pilote NVIDIA)"},
    Entry{"Downscaler", "Réduction"},
    Entry{"Export skips no frames: each created frame costs a pass.",
          "L'export ne saute aucune image : chaque image créée coûte une passe."},
    Entry{"Fused DistilDRBA", "DistilDRBA fusionné"},
    Entry{"Fused RIFE", "RIFE fusionné"},
    Entry{"Very fast", "Très rapide"},
    Entry{"Balanced", "Équilibré"},
    Entry{"System (macOS)", "Système (macOS)"},
    Entry{"DistilDRBA: the lightest; it keeps characters animated on twos on their rhythm. One frame of "
          "delay.",
          "DistilDRBA : le plus léger ; les personnages animés à une image sur deux gardent leur rythme. Une "
          "image de retard."},
    Entry{"RIFE 4.22 lite: light and clean, for most graphics cards.",
          "RIFE 4.22 lite : léger et propre, pour la plupart des cartes graphiques."},
    Entry{"RIFE 4.25: cleaner in-between frames, for recent graphics cards.",
          "RIFE 4.25 : des images intermédiaires plus propres, pour les cartes graphiques récentes."},
    Entry{"Very fast is the lightest, Balanced the cleanest. Applies after a short reload.",
          "Très rapide est le plus léger, Équilibré le plus propre. S'applique après un court rechargement."},
    Entry{"Greyed out: above the display ({:.0f} Hz for {} fps).",
          "Grisé : au-delà de l'écran ({:.0f} Hz pour {} i/s)."},
    Entry{"Greyed out: turn on RIFE at ×2 or more, or playback stalls.",
          "Grisé : activez RIFE à ×2 ou plus, sans quoi la lecture cale."},
    Entry{"Keep poses (DRBA) with RIFE", "Garder les poses (DRBA) avec RIFE"},
    Entry{"Maximum (display)", "Maximum (écran)"},
    Entry{"Model", "Modèle"},
    Entry{"Multiplier", "Multiplicateur"},
    Entry{"No downscale: the source goes into SPAN as is.",
          "Pas de réduction : la source entre telle quelle dans SPAN."},
    Entry{"No interpolation model found: put a RIFE model in the models folder.",
          "Aucun modèle d'interpolation trouvé : placez un modèle RIFE dans le dossier des modèles."},
    Entry{"No steps: each frame is computed for the exact moment it is shown.",
          "Aucun palier : chaque image est calculée pour l'instant exact où elle sera vue."},
    Entry{"Output: {}, twice the working resolution (SPAN ×2).",
          "Sortie : {}, le double de la résolution de travail (SPAN ×2)."},
    Entry{"Performance: fastest. Balanced: finer, slower. Sharp1: crisper edges.",
          "Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets."},
    Entry{
        "Performance: fastest. Balanced: finer, slower. Sharp1: crisper edges. FP8: prepared once on this PC "
        "at the next video, then faster.",
        "Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets. FP8 : "
        "préparé une fois sur ce PC à la prochaine vidéo, puis plus rapide."},
    Entry{"RIFE model", "Modèle RIFE"},
    Entry{"Resolution", "Résolution"},
    Entry{"SPAN precision", "Précision de SPAN"},
    Entry{"SPAN ×2 (AnimeJaNai)", "SPAN ×2 (AnimeJaNai)"},
    Entry{"SSimDownscaler (coming soon)", "SSimDownscaler (à venir)"},
    Entry{"Scene cuts and held drawings are not interpolated.",
          "Les coupes de scène et les dessins tenus ne sont pas interpolés."},
    Entry{"Smooth Motion (NVIDIA driver)", "Smooth Motion (pilote NVIDIA)"},
    Entry{"Source (no downscale)", "Source (sans réduction)"},
    Entry{"Source, no downscale ({})", "Source, sans réduction ({})"},
    Entry{"Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live.",
          "Spline36 est le choix net d'AnimeJaNai ; Hermite est net sans halo. En direct."},
    Entry{
        "Splits DistilDRBA into small TensorRT engines: more than twice as fast. TensorRT only, Balanced "
        "precision. Reloads playback; built on first use for this size.",
        "Coupe DistilDRBA en petits moteurs TensorRT : plus de deux fois plus rapide. TensorRT seulement, "
        "précision Équilibre. Recharge la lecture ; construit à la première utilisation pour cette taille."},
    Entry{"Splits RIFE into small TensorRT engines: about three times faster. TensorRT only. Reloads "
          "playback; built on first use for this size.",
          "Coupe RIFE en petits moteurs TensorRT : environ trois fois plus rapide. TensorRT seulement. "
          "Recharge la lecture ; construit à la première utilisation pour cette taille."},
    Entry{"Stable", "Stable"},
    Entry{"Stable follows what the GPU can hold in steps (×1 to ×6). Fast has no steps and falls back to "
          "Stable if too many frames are lost. Playback only.",
          "Stable suit ce que le GPU tient par paliers (×1 à ×6). Rapide n'a pas de palier et repasse sur "
          "Stable si trop d'images sont perdues. Lecture seulement."},
    Entry{"The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames. Applies at next launch.",
          "Le pilote NVIDIA (RTX 40 et 50) ajoute une image entre deux images affichées. Vaut au prochain "
          "lancement."},
    Entry{"The highest frame rate Auto aims for. It still drops by itself on heavy scenes.",
          "La cadence la plus haute visée par Auto. Elle baisse seule sur les scènes lourdes."},
    Entry{"The size the video is reduced to before SPAN, which doubles it: 720p gives 1440p.",
          "Taille à laquelle la vidéo est ramenée avant SPAN, qui la double : 720p donne du 1440p."},
    Entry{"VRR is off in the system: turn on adaptive sync (KDE: Display > Adaptive sync).",
          "VRR coupé par le système : activez la synchronisation adaptative (KDE : Affichage > "
          "Synchronisation adaptative)."},
    Entry{"VRR is turned off by the system: motion may judder.",
          "VRR coupé par le système : le mouvement peut vibrer."},
    Entry{"VRR unavailable: turn on G-Sync / FreeSync / Adaptive Sync to avoid judder.",
          "VRR indisponible : activez G-Sync / FreeSync / Adaptive Sync pour éviter les saccades."},
    Entry{"Warp and SPAN", "Warp et SPAN"},
    Entry{"Without VRR, frames follow the display refresh; a late frame repeats the previous one.",
          "Sans VRR, la cadence suit le rafraîchissement ; une image tardive répète la précédente."},
    Entry{"Without VRR, motion may judder at a fixed refresh rate.",
          "Sans VRR, le mouvement peut vibrer à fréquence fixe."},
    Entry{"Without upscaling, the output stays at the working resolution: {}.",
          "Sans agrandissement, la sortie garde la résolution de travail : {}."},
    Entry{"Working resolution", "Résolution de travail"},
};

}  // namespace

std::span<const Entry> french_image() {
  return kImageTabs;
}

}  // namespace ajp::app::play::i18n
