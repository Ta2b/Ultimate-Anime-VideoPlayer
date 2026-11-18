#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Image IA : (clé anglaise, français).
constexpr std::array kImageTabs{
    Entry{" (recommended)", " (recommandé)"},
    Entry{" (experimental)", " (expérimental)"},
    Entry{"{} fps", "{} i/s"},
    Entry{"Above 720p of working resolution, RIFE estimates motion on a lighter 720p copy (Estimate motion "
          "at 720p): slow pans can ripple.",
          "Au-dessus de 720p de résolution de travail, RIFE estime le mouvement sur une copie 720p plus "
          "légère (Estimer le mouvement en 720p) : les panoramiques lents peuvent onduler."},
    Entry{"4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything.",
          "Sortie 4K : ×3 et plus demandent un GPU haut de gamme. Une résolution de travail de 720p allège "
          "tout."},
    Entry{
        "Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are skipped.",
        "Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un GPU rapide, sinon "
        "des images sont sautées."},
    Entry{"Low", "Bas"},
    Entry{"Medium", "Moyen"},
    Entry{"High+", "Élevé+"},
    Entry{"Mode", "Mode"},
    Entry{"Max ({} fps)", "Max ({} i/s)"},
    Entry{"Turn off if the graphics card can't keep up", "Couper si la carte graphique ne suit pas"},
    Entry{"If the card can't keep up for about 10 s, interpolation stops for this video and upscaling "
          "keeps running, instead of stuttering. Recommended: on. Instant mode only.",
          "Si la carte ne suit pas pendant environ 10 s, l'interpolation s'arrête pour cette vidéo et "
          "l'agrandissement continue, au lieu de saccader. Recommandé : activé. Mode Instantané seulement."},
    Entry{"RIFE 4.25 lite, the lightest level. Speed: about 2× Medium. Quality impact: moderate loss "
          "(softer in-between frames).",
          "RIFE 4.25 lite, le niveau le plus léger. Vitesse : environ 2× Moyen. Impact sur la qualité : "
          "perte modérée (images intermédiaires plus douces)."},
    Entry{"RIFE 4.26 guided by 1 to 3 GMFSS key frames on fast motion only. Experimental. Speed: about −20"
          " to −40 % vs High, depending on the scene. Quality impact: moderate gain on fast turns and "
          "pans. NVIDIA RTX only.",
          "RIFE 4.26 guidé par 1 à 3 images clés GMFSS, sur les mouvements rapides seulement. "
          "Expérimental. Vitesse : environ −20 à −40 % face à Élevé, selon la scène. Impact sur la qualité"
          " : gain modéré sur les rotations et panoramiques rapides. NVIDIA RTX seulement."},
    Entry{"RIFE 4.26: cleaner in-between frames. Speed: about −10 % vs Medium. Quality impact: minor gain."
          " Recommended on NVIDIA RTX.",
          "RIFE 4.26 : des images intermédiaires plus propres. Vitesse : environ −10 % face à Moyen. "
          "Impact sur la qualité : gain mineur. Recommandé sur NVIDIA RTX."},
    Entry{"The highest frame rate interpolation aims for; a lower limit saves power and heat. Recommended:"
          " the display maximum.",
          "La cadence la plus haute visée par l'interpolation ; une limite plus basse économise l'énergie "
          "et la chauffe. Recommandé : le maximum de l'écran."},
    Entry{
        "Adds frames between the video's own frames for smoother motion, as many as the display and the "
        "graphics card allow. Higher levels make cleaner in-between frames on fast motion but need a "
        "stronger card. Low, Medium and High are RIFE 4.25 lite, 4.22 lite and 4.26; High+ adds GMFSS to "
        "High on fast motion, is still experimental and needs an NVIDIA RTX.",
        "Ajoute des images entre celles de la vidéo pour un mouvement plus fluide, autant que l'écran et la "
        "carte graphique le permettent. Les niveaux plus hauts font des images intermédiaires plus propres "
        "sur les mouvements rapides, mais demandent une carte plus puissante. Bas, Moyen et Élevé sont "
        "RIFE 4.25 lite, 4.22 lite et 4.26 ; Élevé+ ajoute GMFSS à Élevé sur les mouvements rapides, est "
        "encore expérimental et demande une NVIDIA RTX."},
    Entry{"FPS limit", "Limite d'i/s"},
    Entry{"Here interpolation goes up to ×{} ({:.0f} fps limit).",
          "Ici, l'interpolation va jusqu'à ×{} (limite de {:.0f} i/s)."},
    Entry{"Very low", "Très bas"},
    Entry{"Anime4K, shaders without a network: runs on any graphics card, even integrated. Quality impact:"
          " modest gain over plain scaling.",
          "Anime4K, des shaders sans réseau : tourne sur toute carte graphique, même intégrée. Impact sur "
          "la qualité : gain modeste face à un simple agrandissement."},
    Entry{"AnimeJaNai Performance (SPAN): firm, clean lines; keeps up on most dedicated cards.",
          "AnimeJaNai Performance (SPAN) : des traits fermes et propres ; suit sur la plupart des cartes "
          "dédiées."},
    Entry{"AnimeJaNai Balanced (SPAN): finer detail, for strong graphics cards.",
          "AnimeJaNai Balanced (SPAN) : des détails plus fins, pour les cartes graphiques puissantes."},
    Entry{"FP8", "FP8"},
    Entry{"Sharp lines (Sharp1)", "Traits nets (Sharp1)"},
    Entry{"ArtCNN, a light network for small graphics cards.",
          "ArtCNN, un réseau léger pour les petites cartes graphiques."},
    Entry{"AnimeJaNai computes in 8-bit precision: prepared once the first time, then faster. Speed: about"
          " +37 %. Quality impact: imperceptible. Recommended: on when offered. AnimeJaNai on an NVIDIA "
          "RTX 40 or newer only.",
          "AnimeJaNai calcule en précision 8 bits : préparé une fois la première fois, puis plus rapide. "
          "Vitesse : environ +37 %. Impact sur la qualité : imperceptible. Recommandé : activé quand il "
          "est proposé. AnimeJaNai sur une NVIDIA RTX 40 ou plus récente seulement."},
    Entry{"Redraws the video at twice its size with a model trained on anime: sharper lines and less blur "
          "on a big screen. Higher levels keep finer detail but need a stronger card; it is the heaviest "
          "step, lower it first if the video stutters. Very low is Anime4K, Low is ArtCNN, Medium and High"
          " are AnimeJaNai Performance and Balanced (SPAN).",
          "Redessine la vidéo au double de sa taille avec un modèle entraîné sur l'anime : des traits plus"
          " nets et moins de flou sur un grand écran. Les niveaux plus hauts gardent plus de détails mais "
          "demandent une carte plus puissante ; c'est l'étape la plus lourde, à baisser d'abord si la "
          "vidéo saccade. Très bas est Anime4K, Bas est ArtCNN, Moyen et Élevé sont AnimeJaNai Performance"
          " et Balanced (SPAN)."},
    Entry{"The Sharp1 version of AnimeJaNai draws crisper lines, a little harsher. Its engine is built "
          "once the first time it is used, which takes a few minutes. Quality impact: a matter of taste. "
          "Recommended: off. Medium and High only.",
          "La version Sharp1 d'AnimeJaNai dessine des traits plus nets, un peu plus durs. Son moteur se "
          "construit une fois, à la première utilisation, ce qui prend quelques minutes. Impact sur la "
          "qualité : une affaire de goût. Recommandé : désactivé. Moyen et Élevé seulement."},
    Entry{"Greyed out: the video is already HDR, it is shown without the anime grade.",
          "Grisé : la vidéo est déjà en HDR, elle s'affiche sans le grade anime."},
    Entry{"Needs the TensorRT engine (NVIDIA RTX card).", "Demande le moteur TensorRT (carte NVIDIA RTX)."},
    Entry{"Needs the two GMFSS models in the models folder.",
          "Demande les deux modèles GMFSS dans le dossier des modèles."},
    Entry{"Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend "
          "only; RIFE is used elsewhere.",
          "Apple : le flot optique de macOS (VideoToolbox, macOS 15.4+), sans modèle à télécharger. Moteur "
          "Core ML seulement ; ailleurs RIFE prend la place."},
    Entry{"Auto", "Auto"},
    Entry{"DistilDRBA model not found in {} ({}); RIFE is used instead.",
          "Modèle DistilDRBA introuvable dans {} ({}) ; RIFE prend la place."},
    Entry{"Double the frames (NVIDIA driver)", "Doubler les images (pilote NVIDIA)"},
    Entry{"Downscaler", "Réduction"},
    Entry{"Export skips no frames: each created frame costs a pass.",
          "L'export ne saute aucune image : chaque image créée coûte une passe."},
    Entry{"Fused DistilDRBA", "DistilDRBA fusionné"},
    Entry{"Fused RIFE", "RIFE fusionné"},
    Entry{"Sharper interpolation warp (Catmull-Rom)", "Warp de l'interpolation plus net (Catmull-Rom)"},
    Entry{"Steady slow pans", "Panoramiques lents stables"},
    Entry{
        "On slow pans, the motion of the whole picture, measured to a hundredth of a pixel, replaces small "
        "uneven motion: backgrounds stop rippling and juddering. Speed: no noticeable loss. Quality impact: "
        "clear gain on slow pans. Recommended: on. RIFE only.",
        "Sur les panoramiques lents, le mouvement de l'image entière, mesuré au centième de pixel, remplace "
        "les petits mouvements irréguliers : les décors n'ondulent plus et ne saccadent plus. Vitesse : pas "
        "de "
        "perte sensible. Effet sur la qualité : gain net sur les panoramiques lents. Conseillé : activé. "
        "RIFE "
        "seulement."},
    Entry{"Estimate motion at 720p", "Estimer le mouvement en 720p"},
    Entry{
        "Above 720p of working resolution, RIFE measures motion on a lighter 720p copy. Faster, but slow "
        "pans can ripple instead of moving in even steps (RIFE 4.25 lite and 4.26 especially). Speed: "
        "about 15 % more frames in Auto (RIFE pass about 2× faster). Quality impact: moderate loss on slow "
        "pans. Recommended: off. RIFE only.",
        "Au-delà de 720p de résolution de travail, RIFE mesure le mouvement sur une copie 720p plus "
        "légère. Plus rapide, mais les panoramiques lents peuvent onduler au lieu d'avancer à pas réguliers "
        "(surtout RIFE 4.25 lite et 4.26). Vitesse : environ 15 % d'images en plus en Auto (passe RIFE "
        "environ 2× plus rapide). Impact sur la qualité : perte modérée sur "
        "les panoramiques lents. Recommandé : désactivé. RIFE seulement."},
    Entry{"Lighter RIFE on fast motion (half size)", "RIFE allégé sur les mouvements rapides (demi-taille)"},
    Entry{"Fewer RIFE passes on fast motion", "Moins de passes RIFE sur les mouvements rapides"},
    Entry{"Long pans are made faster by reusing the middle in-between frame for its neighbors, which are "
          "then a little less accurate. A trial. Speed: up to +30 % on long pans. Quality impact: minor "
          "loss. Recommended: off. RIFE with Keep poses only.",
          "Les longs panoramiques vont plus vite en reprenant l'image intermédiaire du milieu pour ses "
          "voisines, alors un peu moins justes. Un essai. Vitesse : jusqu'à +30 % sur les longs "
          "panoramiques. Impact sur la qualité : perte mineure. Recommandé : désactivé. RIFE avec Garder les "
          "poses seulement."},
    Entry{"Fast action is made faster by estimating motion at half size; fast motion can look less precise. "
          "A trial, for cards at their limit. Speed: up to +50 % on fast motion. Quality impact: moderate "
          "loss. Recommended: off. RIFE only.",
          "Les scènes d'action vont plus vite en estimant le mouvement à demi-taille ; les mouvements "
          "rapides peuvent paraître moins précis. Un essai, pour les cartes à leur limite. Vitesse : jusqu'à "
          "+50 % sur les mouvements rapides. Impact sur la qualité : perte modérée. Recommandé : désactivé. "
          "RIFE seulement."},
    Entry{"Balanced", "Équilibré"},
    Entry{"System (macOS)", "Système (macOS)"},
    Entry{"RIFE 4.22 lite: light and clean, the reference for the other levels. Recommended on graphics "
          "cards other than NVIDIA RTX.",
          "RIFE 4.22 lite : léger et propre, la référence des autres niveaux. Recommandé sur les cartes "
          "graphiques autres que NVIDIA RTX."},
    Entry{"RIFE 4.25: the former Balanced level. Speed: about −10 % vs Balanced. Quality impact: "
          "imperceptible.",
          "RIFE 4.25 : l'ancien niveau Équilibré. Vitesse : environ −10 % face à Équilibré. Impact sur la "
          "qualité : imperceptible."},
    Entry{"DistilDRBA: characters animated on twos keep their rhythm; one frame of delay. Speed: about Fast "
          "up to ×5, slower above. Quality impact: none measured over RIFE. Auto does not use it.",
          "DistilDRBA : les personnages animés à une image sur deux gardent leur rythme ; une image de "
          "retard. Vitesse : environ Rapide jusqu'à ×5, plus lent au-delà. Impact sur la qualité : aucun "
          "mesuré face à RIFE. Auto ne l'utilise pas."},
    Entry{"High", "Élevé"},
    Entry{"GMFSS motion at half size", "Mouvement de GMFSS à demi-taille"},
    Entry{"GMFSS keeps the source detail", "GMFSS garde le détail de la source"},
    Entry{"GMFSS and RIFE at 540p", "GMFSS et RIFE en 540p"},
    Entry{"GMFSS at every instant, without RIFE", "GMFSS à chaque instant, sans RIFE"},
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
    Entry{"Multiplier", "Multiplicateur"},
    Entry{"No interpolation model found: put a RIFE model in the models folder.",
          "Aucun modèle d'interpolation trouvé : placez un modèle RIFE dans le dossier des modèles."},
    Entry{"No steps: each frame is computed for the exact moment it is shown.",
          "Aucun palier : chaque image est calculée pour l'instant exact où elle sera vue."},
    Entry{"Reduced to {} first, so the output stays within 4K.",
          "Réduite d'abord à {}, pour que la sortie reste en 4K."},
    Entry{
        "Off for this video: {:.2f} fps is not an anime frame rate (23.976, 24, 25, 29.97 or 30).",
        "Coupée pour cette vidéo : {:.2f} i/s n'est pas une cadence d'anime (23,976, 24, 25, 29,97 ou 30)."},
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
    Entry{"RIFE model", "Modèle RIFE"},
    Entry{"Resolution", "Résolution"},
    Entry{"SPAN precision", "Précision de SPAN"},
    Entry{"Smooth held drawings", "Lisser les dessins tenus"},
    Entry{"A drawing held 2 to 6 frames glides to the next one instead of freezing, then jumping. Speed: no "
          "loss. Quality impact: changes the look of the animation (a matter of taste). Recommended: off. "
          "RIFE only; replaces Keep poses.",
          "Un dessin tenu 2 à 6 images glisse vers le suivant au lieu de se figer, puis de sauter. Vitesse : "
          "aucune perte. Impact sur la qualité : change le rendu de l'animation (affaire de goût). "
          "Recommandé : désactivé. RIFE seulement ; remplace Garder les poses."},
    Entry{"Smooth Motion (NVIDIA driver)", "Smooth Motion (pilote NVIDIA)"},
    Entry{"Source (no downscale)", "Source (sans réduction)"},
    Entry{"Source, no downscale ({})", "Source, sans réduction ({})"},
    Entry{"Stable", "Stable"},
    Entry{"Stable: fixed steps (×1 to ×8), the most regular motion. Instant: each frame is made for the "
          "exact moment it is shown, for more frames and smoother motion; it can stutter slightly at the "
          "card's limit. Speed: Instant about +50 to +85 % frames. Quality impact: minor (Instant can "
          "stutter slightly at the card's limit). Recommended: Instant.",
          "Stable : des paliers fixes (×1 à ×8), le mouvement le plus régulier. Instantané : chaque image "
          "est faite pour l'instant exact où elle s'affiche, pour plus d'images et un mouvement plus fluide "
          "; il peut saccader un peu à la limite de la carte. Vitesse : Instantané environ +50 à +85 % "
          "d'images. Impact sur la qualité : mineur (Instantané peut saccader un peu à la limite de la "
          "carte). Recommandé : Instantané."},
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
    Entry{"Without VRR, frames follow the display refresh; a late frame repeats the previous one.",
          "Sans VRR, la cadence suit le rafraîchissement ; une image tardive répète la précédente."},
    Entry{"Without VRR, motion may judder at a fixed refresh rate.",
          "Sans VRR, le mouvement peut vibrer à fréquence fixe."},
    Entry{"Light", "Légère"},
    Entry{"Strong", "Forte"},
    Entry{"Stability", "Stabilité"},
    Entry{"Temporal stabilization", "Stabilisation temporelle"},
    Entry{
        "Removes the shimmer from one frame to the next (fizzling lines, crawling textures, flickering "
        "grain) by comparing each frame with the three before and the three after. Upscaling and "
        "interpolation then work on a steady picture. Stronger also smooths very small movements. Speed: "
        "about half the cost of upscaling, as it works at half size and keeps the full detail. TensorRT "
        "only for now. Recommended: Off, or Normal on an RTX card.",
        "Retire le fourmillement d'une image à l'autre (traits qui grésillent, textures qui grouillent, "
        "grain qui scintille) en comparant chaque image aux trois d'avant et aux trois d'après. "
        "L'agrandissement et l'interpolation travaillent ensuite sur une image stable. Plus fort lisse "
        "aussi les tout petits mouvements. Vitesse : environ la moitié de l'agrandissement, car elle "
        "travaille à demi-taille en gardant tout le détail. TensorRT seulement pour l'instant. Recommandé : "
        "Désactivée, ou Normale sur une carte RTX."},
    Entry{"Full-size stabilization", "Stabilisation à pleine taille"},
    Entry{"Runs the temporal stabilization on the full working picture instead of half size: it also "
          "steadies the finest grain and lines, for about four times the cost (about twice upscaling). "
          "Recommended: off, except on a high-end RTX card with interpolation low.",
          "Fait tourner la stabilisation temporelle sur toute l'image de travail au lieu de la demi-taille : "
          "elle stabilise aussi le grain et les traits les plus fins, pour environ quatre fois le coût (à "
          "peu près deux fois l'agrandissement). Recommandé : désactivée, sauf sur une carte RTX haut de "
          "gamme avec peu d'interpolation."},
    Entry{
        "Temporal stabilization runs on TensorRT only for now: this engine plays without it.",
        "La stabilisation temporelle ne tourne que sous TensorRT pour l'instant : ce moteur lit sans elle."},
    Entry{"temporal stabilization turns off", "la stabilisation temporelle s'arrête"},
    Entry{"temporal stabilization loads (built the first time for this size)",
          "la stabilisation temporelle se charge (construite la première fois pour cette taille)"},
    Entry{"temporal stabilization changes size (built the first time for this size)",
          "la stabilisation temporelle change de taille (construite la première fois pour cette taille)"},
    Entry{"Working resolution", "Résolution de travail"},
    Entry{"the upscaler changes to Anime4K", "l'agrandissement passe à Anime4K"},
};

}  // namespace

std::span<const Entry> french_image() {
  return kImageTabs;
}

}  // namespace ajp::app::play::i18n
