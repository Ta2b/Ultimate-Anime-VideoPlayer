// Le menu des paramètres, à la manière de Dolphin : infobulles, encadrés, onglets
// Vidéo > Général, Agrandissement, Interpolation et HDR, et ce que le pied du panneau dit.
#include <array>
#include <span>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Menu des paramètres : (clé anglaise, français).
constexpr std::array kMenuTexts{
    Entry{"Advanced", "Avancé"},
    Entry{"Speed:", "Vitesse :"},
    Entry{"Quality impact:", "Impact sur la qualité :"},
    Entry{"Recommended: {}", "Recommandé : {}"},
    Entry{"Applies by reloading the video.", "S'applique en rechargeant la vidéo."},
    Entry{"Applies by restarting the player.", "S'applique en relançant le lecteur."},
    Entry{"Auto (follows the system)", "Auto (suit le système)"},
    Entry{"the graphics card changes", "la carte graphique change"},
    Entry{"the engines folder changes", "le dossier des moteurs change"},
    Entry{"Vulkan engines are built again with the new precision",
          "les moteurs Vulkan se reconstruisent avec la nouvelle précision"},
    Entry{"NVIDIA Smooth Motion turns on", "NVIDIA Smooth Motion s'active"},
    Entry{"NVIDIA Smooth Motion turns off", "NVIDIA Smooth Motion se coupe"},
    Entry{"the video is analysed again for faces", "la vidéo est de nouveau analysée pour les visages"},
    Entry{" ; the player restarts on this video, at the same spot",
          " ; le lecteur se relance sur cette vidéo, au même endroit"},
    Entry{"Calibration", "Calibration"},
    Entry{"Measures which settings this machine holds; Auto then follows them. Playback closes meanwhile, "
          "then resumes at the same spot.",
          "Mesure les réglages que cette machine tient ; Auto les suit ensuite. La lecture se ferme pendant "
          "la mesure, puis reprend au même endroit."},
    Entry{"Performance: fastest. Balanced: finer. Sharp1: crisper lines. ArtCNN: lightest. Anime4K: shaders, "
          "for small graphics cards. FP8: prepared once, then about +37 % speed for an imperceptible loss. "
          "Recommended: the preset's choice.",
          "Performance : le plus rapide. Équilibre : plus fin. Sharp1 : traits plus nets. ArtCNN : le plus "
          "léger. Anime4K : shaders, pour les petites cartes graphiques. FP8 : préparé une fois, puis "
          "environ +37 % de vitesse pour une perte imperceptible. Recommandé : le choix du préréglage."},
    Entry{
        "Performance: fastest. Balanced: finer. Sharp1: crisper lines. ArtCNN: lightest. Anime4K: shaders, "
        "for small graphics cards. Recommended: the preset's choice.",
        "Performance : le plus rapide. Équilibre : plus fin. Sharp1 : traits plus nets. ArtCNN : le plus "
        "léger. Anime4K : shaders, pour les petites cartes graphiques. Recommandé : le choix du préréglage."},
    Entry{"Anime4K: SDR sources only; an HDR source is upscaled by the display.",
          "Anime4K : sources SDR seulement ; une source HDR est agrandie par l'affichage."},
    Entry{"Output: {} (SPAN ×2).", "Sortie : {} (SPAN ×2)."},
    Entry{"Output: {} (no upscaling).", "Sortie : {} (sans agrandissement)."},
    Entry{"Balanced is FP16, Quality FP32. Speed: Quality about 2 to 3× slower. Quality impact: "
          "imperceptible. Recommended: Balanced.",
          "Équilibre est en FP16, Qualité en FP32. Vitesse : Qualité environ 2 à 3× plus lent. Impact sur la "
          "qualité : imperceptible. Recommandé : Équilibre."},
    Entry{"Doubles the resolution with a model trained on anime. Speed: the heaviest step of the chain. "
          "Quality impact: high gain. Recommended: on.",
          "Double la résolution avec un modèle entraîné sur l'anime. Vitesse : l'étape la plus lourde de la "
          "chaîne. Impact sur la qualité : gain élevé. Recommandé : activé."},
    Entry{"Restore line art", "Restaurer les traits"},
    Entry{"A source above the working resolution is reduced, then SPAN redraws its lines. Off: such a source "
          "keeps its grain and textures, without SPAN. Speed: about 4× less SPAN work on such a source. "
          "Quality impact: modest (cleaner lines, less grain). Recommended: on.",
          "Une source plus grande que la résolution de travail est réduite, puis SPAN redessine ses traits. "
          "Désactivé : une telle source garde son grain et ses textures, sans SPAN. Vitesse : environ 4× "
          "moins de travail pour SPAN sur une telle source. Impact sur la qualité : modéré (traits plus "
          "propres, moins de grain). Recommandé : activé."},
    Entry{"The size the video is reduced to before SPAN doubles it: 720p gives 1440p. Speed: 1080p costs "
          "about 2.3× 720p. Quality impact: modest. Recommended: the preset's choice.",
          "La taille à laquelle la vidéo est réduite avant que SPAN la double : 720p donne 1440p. Vitesse : "
          "1080p coûte environ 2,3× 720p. Impact sur la qualité : modéré. Recommandé : le choix du "
          "préréglage."},
    Entry{"Crisper lines after upscaling, without halos. 0: untouched. Speed: imperceptible cost. "
          "Recommended: 0 %, to taste.",
          "Des traits plus nets après l'agrandissement, sans halos. 0 : intact. Vitesse : coût "
          "imperceptible. Recommandé : 0 %, selon le goût."},
    Entry{"Filter and precision", "Filtre et précision"},
    Entry{"The filter that reduces the source before SPAN. Hermite: sharp, no halos. Spline36: sharper. "
          "Quality impact: minor. Recommended: Hermite.",
          "Le filtre qui réduit la source avant SPAN. Hermite : net, sans halos. Spline36 : plus net. Impact "
          "sur la qualité : mineur. Recommandé : Hermite."},
    Entry{"HDR", "HDR"},
    Entry{"Auto: HDR when the system is in HDR and the display is bright enough (350 nits). On: HDR whenever "
          "the system is in HDR. Off: SDR. HDR needs HDR turned on in the system. Recommended: Auto.",
          "Auto : HDR quand le système est en HDR et que l'écran est assez lumineux (350 nits). Activé : HDR "
          "dès que le système est en HDR. Désactivé : SDR. Le HDR demande d'activer le HDR dans le système. "
          "Recommandé : Auto."},
    Entry{"Now: HDR10.", "En ce moment : HDR10."},
    Entry{"Now: SDR.", "En ce moment : SDR."},
    Entry{"Color intensity. Recommended: 100 %.", "L'intensité des couleurs. Recommandé : 100 %."},
    Entry{"Gap between dark and bright areas. Recommended: 100 %.",
          "L'écart entre les zones sombres et claires. Recommandé : 100 %."},
    Entry{"Brightness of lights and glints. Recommended: 100 %.",
          "La luminosité des lumières et des reflets. Recommandé : 100 %."},
    Entry{"How much dark scenes are deepened, at most; less in dark episodes. Recommended: 0.50.",
          "De combien les scènes sombres sont approfondies, au plus ; moins dans un épisode sombre. "
          "Recommandé : 0,50."},
    Entry{"LUT contrast", "Contraste des LUT"},
    Entry{"Strength of the contrast LUT. 0: the reference look; −100: removed; 100: doubled. Recommended: 0.",
          "La force de la LUT de contraste. 0 : le rendu de référence ; −100 : retirée ; 100 : doublée. "
          "Recommandé : 0."},
    Entry{"LUT highlights", "Éclats des LUT"},
    Entry{
        "Strength of the highlight LUT. 0: the reference look; −100: removed; 100: doubled. Recommended: 0.",
        "La force de la LUT des éclats. 0 : le rendu de référence ; −100 : retirée ; 100 : doublée. "
        "Recommandé : 0."},
    Entry{"Defaults##couleurs", "Par défaut##couleurs"},
    Entry{"The output is SDR: these settings apply when it is HDR (Video > General).",
          "La sortie est en SDR : ces réglages s'appliquent quand elle est en HDR (Vidéo > Général)."},
    Entry{"Brightness", "Luminosité"},
    Entry{"Display peak: auto ({} nits)", "Pic de l'écran : auto ({} nits)"},
    Entry{"The brightest light the grade aims for, read from the display. Recommended: auto.",
          "La lumière la plus forte que vise le grade, lue sur l'écran. Recommandé : auto."},
    Entry{"In content nits: KDE shows them brighter (× its SDR brightness / 203).",
          "En nits du contenu : KDE les montre plus lumineux (× sa luminosité SDR / 203)."},
    Entry{"Brightness of walls and skin. Lower leaves more room to the lights, which still reach the peak. "
          "Recommended: 100 %.",
          "La luminosité des murs et de la peau. Plus bas laisse plus de place aux lumières, qui atteignent "
          "toujours le pic. Recommandé : 100 %."},
    Entry{"Dims bursts of more than three flashes a second for one second, for photosensitive viewers. "
          "Recommended: on.",
          "Atténue pendant une seconde les rafales de plus de trois flashs par seconde, pour les personnes "
          "photosensibles. Recommandé : activé."},
    Entry{"Grade", "Grade"},
    Entry{"Grade before interpolation", "Grade avant l'interpolation"},
    Entry{"HDR, interpolation after SPAN: each source frame is graded once instead of every shown frame. "
          "Speed: up to 8× less grading at ×8. Quality impact: minor, on moving edges. Recommended: off.",
          "En HDR, interpolation après SPAN : chaque image de la source est gradée une fois, au lieu de "
          "chaque image affichée. Vitesse : jusqu'à 8× moins de grade à ×8. Impact sur la qualité : mineur, "
          "sur les bords en mouvement. Recommandé : désactivé."},
    Entry{"LUTs in one table", "LUT en une seule table"},
    Entry{"The LUT stack is read once per pixel instead of LUT by LUT. Speed: grading about 4× faster. "
          "Quality impact: imperceptible. Recommended: on.",
          "La pile de LUT est lue une fois par pixel au lieu de LUT par LUT. Vitesse : grade environ 4× plus "
          "rapide. Impact sur la qualité : imperceptible. Recommandé : activé."},
    Entry{"Faces are found ahead of playback and only their tones are treated as skin, so warm lamps light "
          "up again. Quality impact: minor gain, not yet judged. Recommended: off.",
          "Les visages sont cherchés avant la lecture et seuls leurs tons sont traités comme de la peau : "
          "les lampes chaudes se rallument. Impact sur la qualité : gain mineur, pas encore jugé. Recommandé "
          ": désactivé."},
    Entry{"Defaults##hdr", "Par défaut##hdr"},
    Entry{"The card that plays the video. Recommended: Automatic.",
          "La carte qui lit la vidéo. Recommandé : Automatique."},
    Entry{"Where downloaded engines go. Installed engines are not moved.",
          "Là où vont les moteurs téléchargés. Les moteurs installés ne sont pas déplacés."},
    Entry{"Default: {}. Free: {}.", "Par défaut : {}. Libre : {}."},
    Entry{"Backend", "Moteur de calcul"},
    Entry{"Runs SPAN and the interpolation. TensorRT: NVIDIA RTX, fastest, engines built once per model and "
          "size. Vulkan: every graphics card. Recommended: TensorRT on NVIDIA RTX, Vulkan elsewhere.",
          "Fait tourner SPAN et l'interpolation. TensorRT : NVIDIA RTX, le plus rapide, moteurs construits "
          "une fois par modèle et par taille. Vulkan : toutes les cartes graphiques. Recommandé : TensorRT "
          "sur NVIDIA RTX, Vulkan ailleurs."},
    Entry{"Opens the engines screen. Playback resumes at the same spot.",
          "Ouvre l'écran des moteurs. La lecture reprend au même endroit."},
    Entry{"Vulkan engines keep half precision when the difference is invisible. Speed: about 2× on "
          "upscaling. Quality impact: imperceptible. Recommended: on.",
          "Les moteurs Vulkan gardent la demi-précision quand la différence est invisible. Vitesse : environ "
          "2× sur l'agrandissement. Impact sur la qualité : imperceptible. Recommandé : activé."},
    Entry{"Prepares these engines again.", "Prépare de nouveau ces moteurs."},
    Entry{
        "In-between frames read the source with a sharper filter, for finer lines. Speed: about −3 %. "
        "Quality impact: minor gain. Recommended: on. RIFE only.",
        "Les images intermédiaires lisent la source avec un filtre plus net, pour des traits plus fins. "
        "Vitesse : environ −3 %. Impact sur la qualité : gain mineur. Recommandé : activé. RIFE seulement."},
    Entry{"Very fast, Fast and Balanced use RIFE alone; High and Very high add GMFSS key frames (NVIDIA "
          "RTX). Recommended: Balanced on NVIDIA RTX, Fast elsewhere.",
          "Très rapide, Rapide et Équilibré n'utilisent que RIFE ; Élevé et Très élevé ajoutent des images "
          "clés GMFSS (NVIDIA RTX). Recommandé : Équilibré sur NVIDIA RTX, Rapide ailleurs."},
    Entry{"GMFSS", "GMFSS"},
    Entry{"GMFSS estimates motion on half-size frames. Speed: GMFSS about 1.7× faster. Quality impact: minor "
          "loss on small motion. Recommended: on.",
          "GMFSS estime le mouvement sur des images de demi-taille. Vitesse : GMFSS environ 1,7× plus "
          "rapide. Impact sur la qualité : perte mineure sur les petits mouvements. Recommandé : activé."},
    Entry{"Every in-between frame comes from GMFSS, without RIFE. Speed: about −70 % vs Very high; frames "
          "drop when the card can't keep up. Quality impact: minor change. Recommended: off.",
          "Chaque image intermédiaire vient de GMFSS, sans RIFE. Vitesse : environ −70 % face à Très élevé ; "
          "des images sautent quand la carte ne suit pas. Impact sur la qualité : changement mineur. "
          "Recommandé : désactivé."},
    Entry{"RIFE and GMFSS estimate motion at 540p instead of 720p. Speed: about +35 %. Quality impact: minor "
          "loss. Recommended: off; on if the card can't keep up.",
          "RIFE et GMFSS estiment le mouvement en 540p au lieu de 720p. Vitesse : environ +35 %. Impact sur "
          "la qualité : perte mineure. Recommandé : désactivé ; activé si la carte ne suit pas."},
    Entry{
        "Above 720p of working resolution, GMFSS key frames are rebuilt from the full-size source instead of "
        "enlarged. Speed: about −5 to −15 %. Quality impact: minor gain, not yet judged. Recommended: off.",
        "Au-dessus de 720p de résolution de travail, les images clés GMFSS sont refaites à partir de la "
        "source en pleine taille au lieu d'être agrandies. Vitesse : environ −5 à −15 %. Impact sur la "
        "qualité : gain mineur, pas encore jugé. Recommandé : désactivé."},
    Entry{"The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames, on top of interpolation. "
          "Quality impact: moderate loss (driver frames are rougher). Recommended: off.",
          "Le pilote NVIDIA (RTX 40 et 50) ajoute une image entre les images affichées, en plus de "
          "l'interpolation. Impact sur la qualité : perte modérée (les images du pilote sont plus "
          "grossières). Recommandé : désactivé."},
    Entry{"How many frames per source frame the export makes.",
          "Combien d'images l'export fait pour chaque image de la source."},
    Entry{"Auto: as many frames as the display and the graphics card allow. Off: no interpolation. "
          "Recommended: Auto.",
          "Auto : autant d'images que l'écran et la carte graphique le permettent. Désactivé : pas "
          "d'interpolation. Recommandé : Auto."},
    Entry{"Engines and precision", "Moteurs et précision"},
    Entry{"Trials", "Essais"},
    Entry{"Characters animated on twos keep their rhythm while the background moves. Speed: slightly faster "
          "(held poses skip a pass). Quality impact: minor gain. Recommended: on. RIFE only; Smooth held "
          "drawings replaces it.",
          "Les personnages animés en deux images gardent leur rythme pendant que le décor bouge. Vitesse : "
          "un peu plus rapide (les poses tenues sautent une passe). Impact sur la qualité : gain mineur. "
          "Recommandé : activé. RIFE seulement ; « Lisser les dessins tenus » le remplace."},
    Entry{"Quality preset", "Préréglage de qualité"},
    Entry{"Sets upscaling and interpolation together, from Economy (light) to Ultra. Auto picks the level of "
          "this machine. Changing a setting by hand switches to Custom. Recommended: Auto.",
          "Règle ensemble l'agrandissement et l'interpolation, d'Économe (léger) à Ultra. Auto prend le "
          "niveau de cette machine. Changer un réglage à la main passe en Personnalisé. Recommandé : Auto."},
    Entry{"On screen", "À l'écran"},
    Entry{"Show FPS", "Afficher les FPS"},
    Entry{"The frame rate on screen, top left. Tab shows every statistic. Recommended: off.",
          "La cadence à l'écran, en haut à gauche. Tab montre toutes les statistiques. Recommandé : "
          "désactivé."},
    Entry{"Plays the next file of the folder at the end. N: next, Shift+N: previous, O: skip credits. "
          "Recommended: on.",
          "Lit le fichier suivant du dossier à la fin. N : suivant, Maj+N : précédent, O : passer le "
          "générique. Recommandé : activé."},
    Entry{"Every setting goes back to its default. The engines folder and the language are kept.",
          "Chaque réglage revient à sa valeur par défaut. Le dossier des moteurs et la langue sont gardés."},
    Entry{"Each pair judged a held drawing (blue) or a scene cut (red) tints the screen for a moment and "
          "goes to ajp.log. This session only.",
          "Chaque paire jugée dessin tenu (bleu) ou changement de plan (rouge) teinte l'écran un instant et "
          "va dans ajp.log. Cette séance seulement."},
    Entry{"Stretches the picture to the window. EWA Lanczos is the sharpest and the heaviest. Recommended: "
          "the default.",
          "Étire l'image à la fenêtre. EWA Lanczos est le plus net et le plus lourd. Recommandé : celui par "
          "défaut."},
    Entry{"Smooths steps in flat areas and gradients. Speed: minor cost. Quality impact: minor; can soften "
          "fine grain. Recommended: off.",
          "Lisse les marches dans les aplats et les dégradés. Vitesse : coût mineur. Impact sur la qualité : "
          "mineur ; peut adoucir le grain fin. Recommandé : désactivé."},
    Entry{"Near-blacks become true black (0), up to the threshold. Auto: on an OLED display. Recommended: "
          "Auto.",
          "Les quasi-noirs deviennent du vrai noir (0), jusqu'au seuil. Auto : sur un écran OLED. Recommandé "
          ": Auto."},
    Entry{"On an OLED display, the bar fades after 20 s of pause. Recommended: on.",
          "Sur un écran OLED, la barre s'efface après 20 s de pause. Recommandé : activé."},
    Entry{"Keeps VRR at a constant refresh so dark scenes do not flicker on OLED. Speed: one more "
          "composition per refresh. Recommended: off; on if dark scenes flicker.",
          "Garde le VRR à une fréquence constante pour que les scènes sombres ne scintillent pas sur OLED. "
          "Vitesse : une composition de plus par rafraîchissement. Recommandé : désactivé ; activé si les "
          "scènes sombres scintillent."},
    Entry{"Picked when a file opens. Change the current track from the bar.",
          "Choisies à l'ouverture d'un fichier. La piste en cours se change depuis la barre."},
    Entry{"Scales all subtitle text. Recommended: 100 %.",
          "Agrandit tout le texte des sous-titres. Recommandé : 100 %."},
    Entry{"Positive: subtitles later. Keys z and x: 100 ms. Ctrl+click to type a value (up to ±30,000 ms). "
          "This session only.",
          "Positif : sous-titres plus tard. Touches z et x : 100 ms. Ctrl+clic pour taper une valeur "
          "(jusqu'à ±30 000 ms). Cette séance seulement."},
    Entry{"Reset size and delay", "Remettre la taille et le délai"},
    Entry{"the engine changes to {} (built the first time for each model and size)",
          "le moteur passe à {} (construit la première fois pour chaque modèle et chaque taille)"},
    Entry{"Apply (restarts the player)", "Appliquer (relance le lecteur)"},
    Entry{"Apply (reloads the video)", "Appliquer (recharge la vidéo)"},
    Entry{"Recommended: follow the system.", "Recommandé : suivre le système."},
    Entry{"Positive delays the sound. Kept for this output device. Ctrl+click to type a value (up to ±5,000 "
          "ms). Recommended: 0.",
          "Positif : le son plus tard. Gardé pour cet appareil de sortie. Ctrl+clic pour taper une valeur "
          "(jusqu'à ±5 000 ms). Recommandé : 0."},
    Entry{
        "On a fixed-rate display, playback speed changes by at most 0.125 % so each frame fits whole "
        "refreshes. Sound follows, inaudibly. Recommended: on.",
        "Sur un écran à fréquence fixe, la vitesse de lecture change d'au plus 0,125 % pour que chaque image "
        "tienne en rafraîchissements entiers. Le son suit, sans que cela s'entende. Recommandé : activé."},
    Entry{"The time between choosing a frame and seeing it. Set it if the picture seems early or late. "
          "Recommended: estimated (0).",
          "Le temps entre le choix d'une image et son affichage. À régler si l'image semble en avance ou en "
          "retard. Recommandé : estimée (0)."},
    Entry{"Interface", "Interface"},
    Entry{"Installed: {}", "Installés : {}"},
};

}  // namespace

std::span<const Entry> french_menu() {
  return kMenuTexts;
}

}  // namespace ajp::app::play::i18n
