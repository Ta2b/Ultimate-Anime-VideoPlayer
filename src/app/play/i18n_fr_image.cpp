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
    Entry{"{} fps", "{} i/s"},
    Entry{"Above 720p of working resolution, RIFE estimates motion on a lighter 720p copy (less precise "
          "motion; the speed gained and the quality lost are not measured).",
          "Au-dessus de 720p de résolution de travail, RIFE estime le mouvement sur une copie 720p, plus "
          "légère (mouvement moins précis ; le gain de vitesse et la perte de qualité ne sont pas mesurés)."},
    Entry{"4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything.",
          "Sortie 4K : ×3 et plus demandent un GPU haut de gamme. Une résolution de travail de 720p allège "
          "tout."},
    Entry{
        "Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are skipped.",
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un GPU rapide, sinon "
        "des images sont sautées."},
    Entry{"After: SPAN upscales each source frame once, then frames are moved. Before: SPAN on every shown "
          "frame. Speed: After about −50 to −85 % of SPAN time from ×2 to ×8. Quality impact: imperceptible. "
          "Recommended: Auto (After above 1080p and from ×3).",
          "Après : SPAN agrandit chaque image source une fois, puis les images sont déplacées. Avant : SPAN "
          "sur chaque image affichée. Vitesse : Après, environ −50 à −85 % du temps de SPAN de ×2 à ×8. "
          "Impact sur la qualité : imperceptible. Recommandé : Auto (Après au-delà de 1080p et dès ×3)."},
    Entry{"Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend "
          "only; RIFE is used elsewhere.",
          "Apple : le flot optique de macOS (VideoToolbox, macOS 15.4+), sans modèle à télécharger. Moteur "
          "Core ML seulement ; ailleurs RIFE prend la place."},
    Entry{"Auto", "Auto"},
    Entry{"Auto limit", "Plafond d'Auto"},
    Entry{"Auto mode", "Mode d'Auto"},
    Entry{"Balanced is FP16, Quality FP32. Speed: Quality about 2 to 3× slower. Quality impact: "
          "imperceptible. Recommended: Balanced. Reloads playback.",
          "Équilibre est en FP16, Qualité en FP32. Vitesse : Qualité environ 2 à 3× plus lent. Impact sur la "
          "qualité : imperceptible. Recommandé : Équilibre. Recharge la lecture."},
    Entry{"Balanced is FP16; Quality is FP32. Reloads playback.",
          "Équilibre : FP16 ; Qualité : FP32. Recharge la lecture."},
    Entry{"Characters animated on twos keep their rhythm while the background moves. Speed: slightly faster "
          "(held poses skip a pass). Quality impact: minor gain. Recommended: on. RIFE only.",
          "Les personnages animés à une image sur deux gardent leur rythme pendant que le fond bouge. "
          "Vitesse : un peu plus rapide (une pose tenue saute une passe). Impact sur la qualité : gain "
          "mineur. Recommandé : activé. RIFE seulement."},
    Entry{"DistilDRBA model not found in {} ({}); RIFE is used instead.",
          "Modèle DistilDRBA introuvable dans {} ({}) ; RIFE prend la place."},
    Entry{"Double the frames (NVIDIA driver)", "Doubler les images (pilote NVIDIA)"},
    Entry{"Downscaler", "Réduction"},
    Entry{"Export skips no frames: each created frame costs a pass.",
          "L'export ne saute aucune image : chaque image créée coûte une passe."},
    Entry{"Fused DistilDRBA", "DistilDRBA fusionné"},
    Entry{"Fused RIFE", "RIFE fusionné"},
    Entry{"Sharper interpolation warp (Catmull-Rom)", "Warp de l'interpolation plus net (Catmull-Rom)"},
    Entry{"In-between frames read the source with a sharper filter (Catmull-Rom), for finer lines. Speed: "
          "about −3 %. Quality impact: minor gain. Recommended: on. RIFE only.",
          "Les images intermédiaires lisent la source avec un filtre plus net (Catmull-Rom), pour des traits "
          "plus fins. Vitesse : environ −3 %. Impact sur la qualité : gain mineur. Recommandé : activé. RIFE "
          "seulement."},
    Entry{"Lighter RIFE on fast motion (half size)", "RIFE allégé sur les mouvements rapides (demi-taille)"},
    Entry{"Fewer RIFE passes on fast motion", "Moins de passes RIFE sur les mouvements rapides"},
    Entry{"Trial: on fast motion, frames near the middle reuse the middle RIFE pass. Speed: up to +30 % on "
          "long pans. Quality impact: minor loss. Recommended: off. RIFE with Keep poses only.",
          "Essai : sur un mouvement rapide, les images proches du milieu reprennent la passe RIFE du milieu. "
          "Vitesse : jusqu'à +30 % sur les longs panoramiques. Impact sur la qualité : perte mineure. "
          "Recommandé : désactivé. RIFE avec Garder les poses seulement."},
    Entry{
        "Trial: fast motion is estimated on half-size frames. Speed: up to +50 % on fast motion. Quality "
        "impact: moderate loss. Recommended: off. RIFE only.",
        "Essai : le mouvement rapide s'estime sur des images de demi-taille. Vitesse : jusqu'à +50 % sur les "
        "mouvements rapides. Impact sur la qualité : perte modérée. Recommandé : désactivé. RIFE seulement."},
    Entry{"Very fast", "Très rapide"},
    Entry{"Balanced", "Équilibré"},
    Entry{"System (macOS)", "Système (macOS)"},
    Entry{"RIFE 4.25 lite, the lightest level. Speed: about 2× Fast. Quality impact: moderate loss (softer "
          "in-between frames).",
          "RIFE 4.25 lite, le niveau le plus léger. Vitesse : environ 2× Rapide. Impact sur la qualité : "
          "perte modérée (images intermédiaires plus douces)."},
    Entry{"RIFE 4.22 lite: light and clean, the reference for the other levels. Recommended on graphics "
          "cards other than NVIDIA RTX.",
          "RIFE 4.22 lite : léger et propre, la référence des autres niveaux. Recommandé sur les cartes "
          "graphiques autres que NVIDIA RTX."},
    Entry{"RIFE 4.26: cleaner in-between frames. Speed: about −10 % vs Fast. Quality impact: minor gain. "
          "Recommended on NVIDIA RTX.",
          "RIFE 4.26 : des images intermédiaires plus propres. Vitesse : environ −10 % face à Rapide. Impact "
          "sur la qualité : gain mineur. Recommandé sur NVIDIA RTX."},
    Entry{"RIFE 4.25: the former Balanced level. Speed: about −10 % vs Balanced. Quality impact: "
          "imperceptible.",
          "RIFE 4.25 : l'ancien niveau Équilibré. Vitesse : environ −10 % face à Équilibré. Impact sur la "
          "qualité : imperceptible."},
    Entry{"DistilDRBA: characters animated on twos keep their rhythm; one frame of delay. Speed: about Fast "
          "up to ×5, slower above. Quality impact: none measured over RIFE. Auto does not use it.",
          "DistilDRBA : les personnages animés à une image sur deux gardent leur rythme ; une image de "
          "retard. Vitesse : environ Rapide jusqu'à ×5, plus lent au-delà. Impact sur la qualité : aucun "
          "mesuré face à RIFE. Auto ne l'utilise pas."},
    Entry{"Very fast, Fast and Balanced use RIFE alone; High and Very high add GMFSS key frames (NVIDIA "
          "RTX). Recommended: Balanced on NVIDIA RTX, Fast elsewhere. Applies after a short reload.",
          "Très rapide, Rapide et Équilibré n'utilisent que RIFE ; Élevé et Très élevé y ajoutent des images "
          "clés GMFSS (NVIDIA RTX). Recommandé : Équilibré sur NVIDIA RTX, Rapide ailleurs. S'applique après "
          "un court rechargement."},
    Entry{"High", "Élevé"},
    Entry{"Very high", "Très élevé"},
    Entry{"GMFSS motion at half size", "Mouvement de GMFSS à demi-taille"},
    Entry{"GMFSS keeps the source detail", "GMFSS garde le détail de la source"},
    Entry{"GMFSS and RIFE at 540p", "GMFSS et RIFE en 540p"},
    Entry{"RIFE and GMFSS estimate motion at 540p instead of 720p. Speed: about +35 %. Quality impact: minor "
          "loss. Recommended: off; on if the card can't keep up. Reloads playback.",
          "RIFE et GMFSS estiment le mouvement en 540p au lieu de 720p. Vitesse : environ +35 %. Impact sur "
          "la qualité : perte mineure. Recommandé : désactivé ; activé si la carte ne suit pas. Recharge la "
          "lecture."},
    Entry{"GMFSS at every instant, without RIFE", "GMFSS à chaque instant, sans RIFE"},
    Entry{
        "Every in-between frame comes from GMFSS, without RIFE. Speed: about −70 % vs Very high; frames drop "
        "when the card can't keep up. Quality impact: minor change. Recommended: off. Reloads playback.",
        "Chaque image intermédiaire vient de GMFSS, sans RIFE. Vitesse : environ −70 % face à Très élevé ; "
        "des images sautent si la carte ne suit pas. Impact sur la qualité : changement mineur. Recommandé : "
        "désactivé. Recharge la lecture."},
    Entry{"Above 720p of working resolution, GMFSS key frames are rebuilt from the full-size source instead "
          "of enlarged. Speed: about −5 to −15 %. Quality impact: minor gain, not yet judged. Recommended: "
          "off. Reloads playback.",
          "Au-delà de 720p de résolution de travail, les images clés GMFSS sont refaites depuis la source en "
          "pleine taille au lieu d'être agrandies. Vitesse : environ −5 à −15 %. Impact sur la qualité : "
          "gain mineur, pas encore jugé. Recommandé : désactivé. Recharge la lecture."},
    Entry{"GMFSS estimates motion on half-size frames. Speed: GMFSS about 1.7× faster. Quality impact: minor "
          "loss on small motion. Recommended: on. Reloads playback.",
          "GMFSS estime le mouvement sur des images de demi-taille. Vitesse : GMFSS environ 1,7× plus "
          "rapide. Impact sur la qualité : perte mineure sur les petits mouvements. Recommandé : activé. "
          "Recharge la lecture."},
    Entry{
        "RIFE guided by 1 to 3 GMFSS key frames on fast motion only. Speed: about −20 to −40 % vs Balanced, "
        "depending on the scene. Quality impact: moderate gain on fast turns and pans. NVIDIA RTX only.",
        "RIFE guidé par 1 à 3 images clés GMFSS, sur les mouvements rapides seulement. Vitesse : environ −20 "
        "à −40 % face à Équilibré, selon la scène. Impact sur la qualité : gain modéré sur les rotations et "
        "panoramiques rapides. NVIDIA RTX seulement."},
    Entry{
        "RIFE guided by 1 to 3 GMFSS key frames on every moving pair. Speed: about −45 % vs Balanced. "
        "Quality impact: moderate gain, the steadiest motion. NVIDIA RTX only.",
        "RIFE guidé par 1 à 3 images clés GMFSS, sur chaque paire qui bouge. Vitesse : environ −45 % face à "
        "Équilibré. Impact sur la qualité : gain modéré, le mouvement le plus stable. NVIDIA RTX seulement."},
    Entry{" + GMFSS union on fast motion", " + GMFSS union sur les mouvements rapides"},
    Entry{" + GMFSS union on every pair", " + GMFSS union sur chaque paire"},
    Entry{"GMFSS (pair)", "GMFSS (paire)"},
    Entry{"GMFSS (fusion)", "GMFSS (fusion)"},
    Entry{"Greyed out: above the display ({:.0f} Hz for {:.3f} fps).",
          "Grisé : au-delà de l'écran ({:.0f} Hz pour {:.3f} i/s)."},
    Entry{"Greyed out: the NVIDIA driver is older than 580.82.07, update it.",
          "Grisé : le pilote NVIDIA est antérieur à 580.82.07, mettez-le à jour."},
    Entry{"Greyed out: this graphics card is not an RTX 40 or 50.",
          "Grisé : cette carte graphique n'est pas une RTX 40 ou 50."},
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
    Entry{"Reduced to {} first, so the output stays within 4K.",
          "Réduite d'abord à {}, pour que la sortie reste en 4K."},
    Entry{
        "Off for this video: {:.2f} fps is not an anime frame rate (23.976, 24, 25, 29.97 or 30).",
        "Coupée pour cette vidéo : {:.2f} i/s n'est pas une cadence d'anime (23,976, 24, 25, 29,97 ou 30)."},
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
    Entry{"{} engine: reduced precision on {} layer(s) for speed ({}, {})",
          "Moteur {} : précision réduite sur {} couche(s) pour gagner en vitesse ({}, {})"},
    Entry{"{:.1f} dB against float", "{:.1f} dB contre le calcul en float"},
    Entry{"loss not measured", "perte non mesurée"},
    Entry{"speed {:+.0f} %", "vitesse {:+.0f} %"},
    Entry{"speed gain not measured", "gain de vitesse non mesuré"},
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
    Entry{"Smooth held drawings", "Lisser les dessins tenus"},
    Entry{"A drawing held 2 to 6 frames glides to the next one instead of freezing, then jumping. Speed: "
          "about −10 %. Quality impact: changes the look of the animation (a matter of taste). Recommended: "
          "off. RIFE only; replaces Keep poses.",
          "Un dessin tenu 2 à 6 images glisse vers le suivant au lieu de se figer, puis de sauter. Vitesse : "
          "environ −10 %. Impact sur la qualité : change le rendu de l'animation (affaire de goût). "
          "Recommandé : désactivé. RIFE seulement ; remplace Garder les poses."},
    Entry{"Scene cuts are not interpolated; drawings held 2 to 6 frames are smoothed.",
          "Les coupes de scène ne sont pas interpolées ; les dessins tenus de 2 à 6 images sont lissés."},
    Entry{"Scene cuts and held drawings are not interpolated.",
          "Les coupes de scène et les dessins tenus ne sont pas interpolés."},
    Entry{"Smooth Motion (NVIDIA driver)", "Smooth Motion (pilote NVIDIA)"},
    Entry{"Source (no downscale)", "Source (sans réduction)"},
    Entry{"Source, no downscale ({})", "Source, sans réduction ({})"},
    Entry{"Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live.",
          "Spline36 est le choix net d'AnimeJaNai ; Hermite est net sans halo. En direct."},
    Entry{"Runs DistilDRBA as small TensorRT engines. Speed: about 2× faster per pass. Quality impact: "
          "imperceptible. Recommended: on. NVIDIA only; built on first use.",
          "Fait tourner DistilDRBA en petits moteurs TensorRT. Vitesse : environ 2× plus rapide par passe. "
          "Impact sur la qualité : imperceptible. Recommandé : activé. NVIDIA seulement ; construit à la "
          "première utilisation."},
    Entry{"Runs RIFE as small TensorRT engines. Speed: about 3× faster per pass. Quality impact: "
          "imperceptible. Recommended: on. NVIDIA only; built on first use.",
          "Fait tourner RIFE en petits moteurs TensorRT. Vitesse : environ 3× plus rapide par passe. Impact "
          "sur la qualité : imperceptible. Recommandé : activé. NVIDIA seulement ; construit à la première "
          "utilisation."},
    Entry{"Stable", "Stable"},
    Entry{"Stable: fixed steps (×1 to ×8), the most regular motion. Instant: no steps, each frame made for "
          "the moment it is shown. Speed: Instant about +20 to +40 % frames. Quality impact: minor (Instant "
          "can stutter slightly at the card's limit). Recommended: Instant.",
          "Stable : des paliers fixes (×1 à ×8), le mouvement le plus régulier. Instantané : sans palier, "
          "chaque image faite pour l'instant où elle s'affiche. Vitesse : Instantané environ +20 à +40 % "
          "d'images. Impact sur la qualité : mineur (Instantané peut saccader un peu à la limite de la "
          "carte). Recommandé : Instantané."},
    Entry{"The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames. Applies at next launch.",
          "Le pilote NVIDIA (RTX 40 et 50) ajoute une image entre deux images affichées. Vaut au prochain "
          "lancement."},
    Entry{"The highest frame rate Auto aims for; a lower limit saves power and heat. Recommended: Maximum.",
          "La cadence la plus haute visée par Auto ; une limite plus basse économise l'énergie et la "
          "chauffe. Recommandé : Maximum."},
    Entry{"The size the video is reduced to before SPAN, which doubles it: 720p gives 1440p.",
          "Taille à laquelle la vidéo est ramenée avant SPAN, qui la double : 720p donne du 1440p."},
    Entry{"VRR is off in the system: turn on adaptive sync (KDE: Display > Adaptive sync).",
          "VRR coupé par le système : activez la synchronisation adaptative (KDE : Affichage > "
          "Synchronisation adaptative)."},
    Entry{"VRR is off in the system: turn on Variable refresh rate in Settings > System > Display > "
          "Graphics.",
          "VRR coupé par le système : activez la fréquence d'actualisation variable dans Paramètres > "
          "Système > Affichage > Graphiques."},
    Entry{"VRR only works in full screen here: motion may judder in a window.",
          "Ici, le VRR ne marche qu'en plein écran : le mouvement peut saccader en fenêtre."},
    Entry{"VRR is turned off by the system: motion may judder.",
          "VRR coupé par le système : le mouvement peut vibrer."},
    Entry{"VRR unavailable: turn on G-Sync / FreeSync / Adaptive Sync to avoid judder.",
          "VRR indisponible : activez G-Sync / FreeSync / Adaptive Sync pour éviter les saccades."},
    Entry{"Turn interpolation off if the graphics card can't keep up",
          "Couper l'interpolation si la carte ne suit pas"},
    Entry{"Stops interpolation for this video when it makes less than one frame per source pair for about 10 "
          "s; the card goes back to upscaling. Recommended: on.",
          "Arrête l'interpolation pour cette vidéo quand elle fait moins d'une image par paire de sources "
          "pendant environ 10 s ; la carte revient à l'agrandissement. Recommandé : activé."},
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
        "follows the source and the downscale. SDR sources only: an HDR source is upscaled by the display.",
        "Anime4K : des shaders dessinés par l'affichage, sans moteur à construire, pour les petites cartes. "
        "Le mode A, B ou C suit la source et la réduction. Sources SDR seulement : une source HDR est "
        "agrandie "
        "par l'affichage."},
    Entry{"the upscaler changes to Anime4K", "l'agrandissement passe à Anime4K"},
};

}  // namespace

std::span<const Entry> french_image() {
  return kImageTabs;
}

}  // namespace ajp::app::play::i18n
