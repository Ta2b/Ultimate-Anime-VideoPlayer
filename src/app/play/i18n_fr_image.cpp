#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Image IA : (clé anglaise, français).
constexpr std::array kImageTabs{
    Entry{"Restore line art (reduce, then upscale)", "Restaurer les traits (réduire, puis agrandir)"},
    Entry{
        "A source sharper than the working resolution is reduced, then SPAN redraws its lines: cleaner lines "
        "for a quarter of the work. Off: such a source keeps all its detail (grain, textures), without SPAN.",
        "Une source plus nette que la résolution de travail est réduite, puis SPAN redessine ses traits : "
        "des "
        "traits plus propres pour quatre fois moins de calcul. Coupé : une telle source garde tout son "
        "détail "
        "(grain, textures), sans SPAN."},
    Entry{" (recommended)", " (recommandé)"},
    Entry{"4K output: RIFE estimates motion on a lighter 720p copy (less precise motion; the loss is not "
          "measured).",
          "Sortie 4K : RIFE estime le mouvement sur une copie 720p, plus légère (mouvement moins précis ; la "
          "perte n'est pas mesurée)."},
    Entry{"4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything.",
          "Sortie 4K : ×3 et plus demandent un GPU haut de gamme. Une résolution de travail de 720p allège "
          "tout."},
    Entry{
        "Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are skipped.",
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un GPU rapide, sinon "
        "des images sont sautées."},
    Entry{
        "After SPAN upscales each source frame once, then warps it: faster (about 8 ms saved per "
        "frame from 1080p to 4K on an RTX 4070 Ti), but SPAN no longer cleans the warp defects of the "
        "in-between frames; the image quality of the two orders is not compared. Before SPAN also "
        "cleans each in-between frame. Auto picks After above 1080p, and from ×3 for a fixed "
        "multiplier. Reloads playback.",
        "Après SPAN agrandit chaque image source une fois, puis la déforme : plus rapide (environ 8 ms "
        "gagnées par image de 1080p à 4K sur une RTX 4070 Ti), mais SPAN ne nettoie plus les défauts du warp "
        "des images intermédiaires ; la qualité d'image des deux ordres n'est pas comparée. Avant SPAN "
        "nettoie aussi chaque image intermédiaire. Auto choisit Après au-dessus de 1080p, et dès ×3 pour un "
        "multiplicateur fixe. Recharge la lecture."},
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
    Entry{"DistilDRBA", "DistilDRBA"},
    Entry{"Balanced", "Équilibré"},
    Entry{"System (macOS)", "Système (macOS)"},
    Entry{"DistilDRBA: keeps characters animated on twos on their rhythm. One frame of delay. Its speed and "
          "image quality against RIFE are not compared yet, so Auto does not use it.",
          "DistilDRBA : les personnages animés à une image sur deux gardent leur rythme. Une image de "
          "retard. Sa vitesse et sa qualité d'image face à RIFE ne sont pas encore comparées : Auto ne "
          "l'utilise pas."},
    Entry{"RIFE 4.22 lite: light and clean, for most graphics cards.",
          "RIFE 4.22 lite : léger et propre, pour la plupart des cartes graphiques."},
    Entry{"RIFE 4.25: cleaner in-between frames, for recent graphics cards.",
          "RIFE 4.25 : des images intermédiaires plus propres, pour les cartes graphiques récentes."},
    Entry{"Fast and Balanced are RIFE models; DistilDRBA keeps poses. Their speed and image quality are not "
          "compared in this version. Applies after a short reload.",
          "Rapide et Équilibré sont des modèles RIFE ; DistilDRBA garde les poses. Leur vitesse et leur "
          "qualité d'image ne sont pas comparées dans cette version. S'applique après un court "
          "rechargement."},
    Entry{"Greyed out: above the display ({:.0f} Hz for {:.3f} fps).",
          "Grisé : au-delà de l'écran ({:.0f} Hz pour {:.3f} i/s)."},
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
    Entry{"Models: Performance is fast, Balanced finer, Sharp1 crisp, ArtCNN the lightest (C4F16 more "
          "than C4F32).",
          "Modèles : Performance est rapide, Balanced plus fin, Sharp1 plus net, ArtCNN le plus léger "
          "(C4F16 plus que C4F32)."},
    Entry{"Models: Performance is fast, Balanced finer, Sharp1 crisp, ArtCNN the lightest (C4F16 more "
          "than C4F32). FP8: prepared once on this PC, then about 37 % faster per pass (14.7 to 9.2 ms "
          "at 1080p on an RTX 4070 Ti), at about 49 to 52 dB PSNR against FP16.",
          "Modèles : Performance est rapide, Balanced plus fin, Sharp1 plus net, ArtCNN le plus léger "
          "(C4F16 plus que C4F32). FP8 : préparé une fois sur ce PC, puis environ 37 % plus rapide par "
          "passe (14,7 à 9,2 ms en 1080p sur une RTX 4070 Ti), à environ 49 à 52 dB de PSNR face au FP16."},
    Entry{"DistilDRBA is running: fused engines.", "DistilDRBA tourne : moteurs fusionnés."},
    Entry{"DistilDRBA is running: whole network.", "DistilDRBA tourne : réseau entier."},
    Entry{"Motion variant: one SPAN pass per source frame.",
          "Variante mouvement : une passe de SPAN par image source."},
    Entry{"Not fused: this engine cannot run it fused.",
          "Non fusionné : ce moteur ne sait pas le fusionner."},
    Entry{"Not fused: fused engines need Balanced precision.",
          "Non fusionné : les moteurs fusionnés demandent le cran Équilibre."},
    Entry{"Not fused: the fused engines were refused (see the log).",
          "Non fusionné : les moteurs fusionnés ont été refusés (voir le journal)."},
    Entry{"Not fused: the fused engines were refused ({}); details in the log.",
          "Non fusionné : les moteurs fusionnés ont été refusés ({}) ; le détail est au journal."},
    Entry{"RIFE is not fused: the fused engines were refused ({}); details in the log.",
          "RIFE n'est pas fusionné : les moteurs fusionnés ont été refusés ({}) ; le détail est au journal."},
    Entry{"Fused DistilDRBA unavailable ({}): the whole network runs, slower",
          "DistilDRBA fusionné indisponible ({}) : le réseau entier tourne, plus lent"},
    Entry{"Fused RIFE unavailable ({}): the whole network runs, slower",
          "RIFE fusionné indisponible ({}) : le réseau entier tourne, plus lent"},
    Entry{"Above ×5 DistilDRBA computes more than RIFE: Fast (RIFE 4.22 lite) is lighter.",
          "Au-delà de ×5, DistilDRBA calcule plus que RIFE : Rapide (RIFE 4.22 lite) est plus léger."},
    Entry{"Here Auto goes up to ×{} ({:.0f} fps limit).",
          "Ici Auto monte jusqu'à ×{} (plafond de {:.0f} i/s)."},
    Entry{"Interpolation precision", "Précision de l'interpolation"},
    Entry{"Fused engines only run at Balanced precision: the whole network is used instead.",
          "Les moteurs fusionnés ne tournent qu'au cran Équilibre : le réseau entier sert à la place."},
    Entry{"RIFE model", "Modèle RIFE"},
    Entry{"Resolution", "Résolution"},
    Entry{"SPAN precision", "Précision de SPAN"},
    Entry{"SPAN ×2 (AnimeJaNai)", "SPAN ×2 (AnimeJaNai)"},
    Entry{"Scene cuts and held drawings are not interpolated.",
          "Les coupes de scène et les dessins tenus ne sont pas interpolés."},
    Entry{"Smooth Motion (NVIDIA driver)", "Smooth Motion (pilote NVIDIA)"},
    Entry{"Source (no downscale)", "Source (sans réduction)"},
    Entry{"Source, no downscale ({})", "Source, sans réduction ({})"},
    Entry{"Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live.",
          "Spline36 est le choix net d'AnimeJaNai ; Hermite est net sans halo. En direct."},
    Entry{
        "Splits DistilDRBA into small TensorRT engines: 0.92 ms instead of 2.10 ms per pass at 1088×1920 on "
        "an RTX 4070 Ti. TensorRT only, Balanced precision. Reloads playback; built on first use for this "
        "size.",
        "Coupe DistilDRBA en petits moteurs TensorRT : 0,92 ms au lieu de 2,10 ms par passe en 1088×1920 "
        "sur une RTX 4070 Ti. TensorRT seulement, précision Équilibre. Recharge la lecture ; construit à la "
        "première utilisation pour cette taille."},
    Entry{"Splits RIFE into small TensorRT engines: 1.4 ms instead of 4.5 ms per pass on an RTX 4070 Ti, "
          "the image within 53 dB of the whole network. TensorRT only. Reloads playback; built on first use "
          "for this size.",
          "Coupe RIFE en petits moteurs TensorRT : 1,4 ms au lieu de 4,5 ms par passe sur une RTX 4070 Ti, "
          "l'image à 53 dB du réseau entier. TensorRT seulement. Recharge la lecture ; construit à la "
          "première utilisation pour cette taille."},
    Entry{"Stable", "Stable"},
    Entry{
        "Stable follows what the GPU can hold in steps (×1 to ×8). Instant has no steps and falls back to "
        "Stable if too many frames are lost. Playback only.",
        "Stable suit ce que le GPU tient par paliers (×1 à ×8). Instantané n'a pas de palier et repasse sur "
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
    Entry{"Anime4K ×2", "Anime4K ×2"},
    Entry{
        "Anime4K: shaders drawn by the display, with no engine to build, for small GPUs. Mode A, B or C "
        "follows the source and the downscale.",
        "Anime4K : des shaders dessinés par l'affichage, sans moteur à construire, pour les petites cartes. "
        "Le mode A, B ou C suit la source et la réduction."},
    Entry{"the upscaler changes to Anime4K", "l'agrandissement passe à Anime4K"},
};

}  // namespace

std::span<const Entry> french_image() {
  return kImageTabs;
}

}  // namespace ajp::app::play::i18n
