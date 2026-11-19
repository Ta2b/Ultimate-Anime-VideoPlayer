// Le menu des paramètres, à la manière de Dolphin : infobulles, encadrés, onglets
// Vidéo > Général, Agrandissement, Interpolation et HDR, et ce que le pied du panneau dit.
#include <array>
#include <span>

#include "app/play/i18n_fr.h"

namespace kirari::app::play::i18n {
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
    Entry{"Anime4K: SDR sources only; an HDR source is upscaled by the display.",
          "Anime4K : sources SDR seulement ; une source HDR est agrandie par l'affichage."},
    Entry{"Output: {} (SPAN ×2).", "Sortie : {} (SPAN ×2)."},
    Entry{"Output: {} (no upscaling).", "Sortie : {} (sans agrandissement)."},
    Entry{"Quality computes in full precision (FP32) when the model has such a version; otherwise it runs "
          "like Balanced, as with the current SPAN models. Only useful to compare pictures. Speed: up to 2 "
          "to 3× slower, no change with the current SPAN models. Quality impact: imperceptible. Recommended: "
          "Balanced.",
          "Qualité calcule en pleine précision (FP32) quand le modèle en a une version ; sinon il tourne "
          "comme Équilibré, ce qui est le cas des modèles SPAN actuels. Utile seulement pour comparer des "
          "images. Vitesse : jusqu'à 2 à 3× plus lent, aucun changement avec les modèles SPAN actuels. "
          "Impact sur la qualité : imperceptible. Recommandé : Équilibré."},
    Entry{"Restore line art", "Restaurer les traits"},
    Entry{"Videos sharper than the working resolution (a 1080p video with 720p work) get cleaner, firmer "
          "lines, redrawn by the upscaler. Off: such a video is shown as is, with its grain and textures, "
          "and the card rests. Speed: about 2× less work than upscaling such a video at full size. Quality "
          "impact: modest (cleaner lines, less grain). Recommended: on.",
          "Les vidéos plus fines que la résolution de travail (une vidéo 1080p pour un travail en 720p) "
          "gagnent des traits plus propres et plus fermes, redessinés par l'agrandissement. Désactivé : une "
          "telle vidéo s'affiche telle quelle, avec son grain et ses textures, et la carte se repose. "
          "Vitesse : environ 2× moins de travail que d'agrandir une telle vidéo à pleine taille. Impact sur "
          "la qualité : modeste (traits plus propres, moins de grain). Recommandé : activé."},
    Entry{"The size the upscaler works at, then doubles (720p gives 1440p). Higher keeps more detail but "
          "asks much more of the card: lower it first if the video stutters. Speed: 1080p costs about 2× "
          "720p. Quality impact: modest.",
          "La taille à laquelle l'agrandissement travaille, puis qu'il double (720p donne 1440p). Plus haut "
          "garde plus de détails mais demande beaucoup plus à la carte : à baisser en premier si la vidéo "
          "saccade. Vitesse : 1080p coûte environ 2× 720p. Impact sur la qualité : modeste."},
    Entry{"Makes lines crisper after upscaling, without halos. Too high makes the picture look harsh. Speed: "
          "imperceptible cost. Recommended: 0 %, to taste.",
          "Rend les traits plus nets après l'agrandissement, sans halos. Trop haut, l'image devient dure. "
          "Vitesse : coût imperceptible. Recommandé : 0 %, selon le goût."},
    Entry{"Filter and precision", "Filtre et précision"},
    Entry{"How the video is shrunk before upscaling. Hermite keeps lines clean, with no halos; Spline36 "
          "keeps a little more detail. Quality impact: minor. Recommended: Hermite.",
          "La façon de réduire la vidéo avant l'agrandissement. Hermite garde des traits propres, sans halos "
          "; Spline36 garde un peu plus de détails. Impact sur la qualité : mineur. Recommandé : Hermite."},
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
    Entry{"Makes dark scenes deeper and richer, less so in episodes that are dark throughout. Lower it if "
          "shadows lose detail. Recommended: 0.50.",
          "Rend les scènes sombres plus profondes et plus riches, moins dans les épisodes sombres de bout en "
          "bout. À baisser si les ombres perdent des détails. Recommandé : 0,50."},
    Entry{"LUT contrast", "Contraste des LUT"},
    Entry{"More or less contrast than the reference look. Higher makes colors punchier; lower is softer and "
          "closer to the source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0.",
          "Plus ou moins de contraste que le rendu de référence. Plus haut, les couleurs claquent davantage "
          "; plus bas, l'image est plus douce et plus proche de la source. 0 : le rendu de référence ; −100 "
          ": retirée ; 100 : doublée. Recommandé : 0."},
    Entry{"LUT highlights", "Éclats des LUT"},
    Entry{"How much lights, sparks and skies shine. Higher makes them brighter; lower stays closer to the "
          "source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0.",
          "À quel point les lumières, les étincelles et les ciels brillent. Plus haut, ils éclatent "
          "davantage ; plus bas, ils restent proches de la source. 0 : le rendu de référence ; −100 : "
          "retirée ; 100 : doublée. Recommandé : 0."},
    Entry{"Defaults##couleurs", "Par défaut##couleurs"},
    Entry{"The output is SDR: these settings apply when it is HDR (Video > General).",
          "La sortie est en SDR : ces réglages s'appliquent quand elle est en HDR (Vidéo > Général)."},
    Entry{"Brightness", "Luminosité"},
    Entry{"Display peak: auto ({} nits)", "Pic de l'écran : auto ({} nits)"},
    Entry{"How bright the brightest lights get. Read from the display; set it by hand if highlights look "
          "clipped or dull. Recommended: auto.",
          "La luminosité des lumières les plus vives. Lue sur l'écran ; à régler à la main si les éclats "
          "paraissent écrêtés ou ternes. Recommandé : auto."},
    Entry{"In content nits: KDE shows them brighter (× its SDR brightness / 203).",
          "En nits du contenu : KDE les montre plus lumineux (× sa luminosité SDR / 203)."},
    Entry{"How bright walls and skin are. Lower makes the picture calmer and lets lights stand out more; "
          "they still reach the peak. Recommended: 100 %.",
          "La luminosité des murs et de la peau. Plus bas, l'image est plus calme et les lumières ressortent "
          "davantage ; elles atteignent toujours le pic. Recommandé : 100 %."},
    Entry{"Dims bursts of more than three flashes a second for one second, for photosensitive viewers. "
          "Recommended: on.",
          "Atténue pendant une seconde les rafales de plus de trois flashs par seconde, pour les personnes "
          "photosensibles. Recommandé : activé."},
    Entry{"Grade", "Grade"},
    Entry{"Grade before interpolation", "Grade avant l'interpolation"},
    Entry{"Grades each source frame once instead of every displayed frame: frees the card at high "
          "multipliers. Moving edges can show slight color differences. Speed: up to 8× less grading at ×8. "
          "Quality impact: minor, on moving edges. Recommended: off.",
          "Le grade passe une fois par image source au lieu de chaque image affichée : soulage la carte aux "
          "grands multiplicateurs. Les bords en mouvement peuvent montrer de légers écarts de couleur. "
          "Vitesse : jusqu'à 8× moins de grade à ×8. Impact sur la qualité : mineur, sur les bords en "
          "mouvement. Recommandé : désactivé."},
    Entry{"LUTs in one table", "LUT en une seule table"},
    Entry{"Makes the HDR grade about 4 times lighter for the card, with no visible difference. Turn it off "
          "only to compare. Speed: grading about 4× faster. Quality impact: imperceptible. Recommended: on.",
          "Rend le grade HDR environ 4 fois plus léger pour la carte, sans différence visible. À couper "
          "seulement pour comparer. Vitesse : grade environ 4× plus rapide. Impact sur la qualité : "
          "imperceptible. Recommandé : activé."},
    Entry{"Warm lamps in scenes with faces light up again, instead of being toned down like skin. A trial: "
          "some faces can be missed. Quality impact: minor gain, not yet judged. Recommended: off.",
          "Les lampes chaudes des scènes avec des visages se rallument, au lieu d'être adoucies comme de la "
          "peau. Un essai : certains visages peuvent être manqués. Impact sur la qualité : gain mineur, pas "
          "encore jugé. Recommandé : désactivé."},
    Entry{"Defaults##hdr", "Par défaut##hdr"},
    Entry{"The card that plays the video. Recommended: Automatic.",
          "La carte qui lit la vidéo. Recommandé : Automatique."},
    Entry{"Where downloaded engines go. Installed engines are not moved.",
          "Là où vont les moteurs téléchargés. Les moteurs installés ne sont pas déplacés."},
    Entry{"Default: {}. Free: {}.", "Par défaut : {}. Libre : {}."},
    Entry{"Backend", "Moteur de calcul"},
    Entry{"What runs upscaling and interpolation. TensorRT is the fastest on NVIDIA RTX cards, but builds "
          "its engines once per model and size (a few minutes the first time). Vulkan works on every "
          "graphics card. Recommended: TensorRT on NVIDIA RTX, Vulkan elsewhere.",
          "Ce qui fait tourner l'agrandissement et l'interpolation. TensorRT est le plus rapide sur les "
          "cartes NVIDIA RTX, mais construit ses moteurs une fois par modèle et par taille (quelques minutes "
          "la première fois). Vulkan marche sur toutes les cartes graphiques. Recommandé : TensorRT sur "
          "NVIDIA RTX, Vulkan ailleurs."},
    Entry{"Opens the engines screen. Playback resumes at the same spot.",
          "Ouvre l'écran des moteurs. La lecture reprend au même endroit."},
    Entry{"Lets the Vulkan upscaler use reduced precision where the player finds no visible difference; "
          "elsewhere it keeps full precision and nothing changes, as with the current SPAN models. Turn it "
          "off only if you see banding or artifacts. Speed: up to about 2× on upscaling, none with the "
          "current SPAN models. Quality impact: imperceptible. Recommended: on.",
          "Laisse l'agrandissement Vulkan calculer en précision réduite là où le lecteur ne trouve pas de "
          "différence visible ; ailleurs il garde la pleine précision et rien ne change, ce qui est le cas "
          "des modèles SPAN actuels. À couper seulement si des aplats en escalier ou des défauts "
          "apparaissent. Vitesse : jusqu'à environ 2× sur l'agrandissement, aucun gain avec les modèles SPAN "
          "actuels. Impact sur la qualité : imperceptible. Recommandé : activé."},
    Entry{"Prepares these engines again.", "Prépare de nouveau ces moteurs."},
    Entry{
        "In-between frames keep finer lines, for a few percent of speed. Speed: about −3 %. Quality impact: "
        "minor gain. Recommended: on. RIFE only.",
        "Les images intermédiaires gardent des traits plus fins, pour quelques pour cent de vitesse. Vitesse "
        ": environ −3 %. Impact sur la qualité : gain mineur. Recommandé : activé. RIFE seulement."},
    Entry{"GMFSS", "GMFSS"},
    Entry{"GMFSS runs much faster; very small movements can be slightly less precise. Speed: GMFSS about "
          "1.7× faster. Quality impact: minor loss on small motion. Recommended: on.",
          "GMFSS va beaucoup plus vite ; les très petits mouvements peuvent être un peu moins précis. "
          "Vitesse : GMFSS environ 1,7× plus rapide. Impact sur la qualité : perte mineure sur les petits "
          "mouvements. Recommandé : activé."},
    Entry{"Every in-between frame comes from GMFSS, the cleanest on hard motion, but the card works far more "
          "and frames drop when it can't keep up. Speed: about −70 % vs Very high. Quality impact: minor "
          "change. Recommended: off.",
          "Chaque image intermédiaire vient de GMFSS, le plus propre sur les mouvements difficiles, mais la "
          "carte travaille bien plus et des images tombent quand elle ne suit pas. Vitesse : environ −70 % "
          "face à Très élevé. Impact sur la qualité : changement mineur. Recommandé : désactivé."},
    Entry{"GMFSS and RIFE work on a smaller picture, with slightly less precise motion. The time saved does "
          "not add frames per second: it lets GMFSS draw more of the in-between frames (about +60 %). Turn "
          "it on if the card can't keep up at High or Very high. Speed: same frames per second, about +60 % "
          "frames drawn by GMFSS. Quality impact: minor loss.",
          "GMFSS et RIFE travaillent sur une image plus petite, avec un mouvement un peu moins précis. Le "
          "temps gagné n'ajoute pas d'images par seconde : il laisse GMFSS dessiner plus d'images "
          "intermédiaires (environ +60 %). À activer si la carte ne suit pas en Élevé ou Très élevé. Vitesse "
          ": autant d'images par seconde, environ +60 % d'images dessinées par GMFSS. Impact sur la qualité "
          ": perte mineure."},
    Entry{"Sharper GMFSS frames above 720p of working resolution, for a little speed. Speed: about −5 to −15 "
          "%. Quality impact: minor gain, not yet judged. Recommended: off.",
          "Des images GMFSS plus fines au-dessus de 720p de résolution de travail, pour un peu de vitesse. "
          "Vitesse : environ −5 à −15 %. Impact sur la qualité : gain mineur, pas encore jugé. Recommandé : "
          "désactivé."},
    Entry{"The NVIDIA driver (RTX 40 and 50) adds a frame between each displayed frame, for even smoother "
          "motion on a high-refresh display. Its frames are rougher than the player's. Quality impact: "
          "moderate loss (driver frames are rougher). Recommended: off.",
          "Le pilote NVIDIA (RTX 40 et 50) ajoute une image entre chaque image affichée, pour un mouvement "
          "encore plus fluide sur un écran à haute fréquence. Ses images sont plus grossières que celles du "
          "lecteur. Impact sur la qualité : perte modérée (images du pilote plus grossières). Recommandé : "
          "désactivé."},
    Entry{"How many frames per source frame the export makes.",
          "Combien d'images l'export fait pour chaque image de la source."},
    Entry{"Auto: as many frames as the display and the graphics card allow. Off: no interpolation. "
          "Recommended: Auto.",
          "Auto : autant d'images que l'écran et la carte graphique le permettent. Désactivé : pas "
          "d'interpolation. Recommandé : Auto."},
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
    Entry{
        "Size of the menus and the playback bar. Automatic follows the scale of the desktop, and enlarges a "
        "large screen left at 100 % (1440p and up) on Linux. Recommended: Automatic.",
        "Taille des menus et de la barre de lecture. Automatique suit l'échelle du bureau, et agrandit un "
        "grand écran laissé à 100 % (1440p et plus) sous Linux. Recommandé : automatique."},
    Entry{"Plays the next file of the folder at the end. N: next, Shift+N: previous, O: skip credits. "
          "Recommended: on.",
          "Lit le fichier suivant du dossier à la fin. N : suivant, Maj+N : précédent, O : passer le "
          "générique. Recommandé : activé."},
    Entry{"Every setting goes back to its default. The engines folder and the language are kept.",
          "Chaque réglage revient à sa valeur par défaut. Le dossier des moteurs et la langue sont gardés."},
    Entry{"Each pair judged a held drawing (blue) or a scene cut (red) tints the screen for a moment and "
          "goes to kirari.log. This session only.",
          "Chaque paire jugée dessin tenu (bleu) ou changement de plan (rouge) teinte l'écran un instant et "
          "va dans kirari.log. Cette séance seulement."},
    Entry{"How the picture is stretched to the window. EWA Lanczos is the sharpest but the heaviest for the "
          "card. Recommended: the default.",
          "La façon d'étirer l'image à la fenêtre. EWA Lanczos est le plus net mais le plus lourd pour la "
          "carte. Recommandé : celui par défaut."},
    Entry{"Smooths visible steps in skies and dark gradients. Can soften fine grain. Speed: minor cost. "
          "Quality impact: minor; can soften fine grain. Recommended: off.",
          "Lisse les marches visibles dans les ciels et les dégradés sombres. Peut adoucir un grain fin. "
          "Vitesse : coût mineur. Impact sur la qualité : mineur ; peut adoucir un grain fin. Recommandé : "
          "désactivé."},
    Entry{"On an OLED, near-black areas become truly black: deeper blacks, no grey glow. The very darkest "
          "details can merge into black. Auto: on an OLED display. Recommended: Auto.",
          "Sur un OLED, les zones presque noires deviennent vraiment noires : des noirs plus profonds, sans "
          "voile gris. Les détails les plus sombres peuvent se fondre dans le noir. Auto : sur un écran "
          "OLED. Recommandé : Auto."},
    Entry{"On an OLED display, the bar fades after 20 s of pause. Recommended: on.",
          "Sur un écran OLED, la barre s'efface après 20 s de pause. Recommandé : activé."},
    Entry{
        "On an OLED with VRR, keeps the refresh rate constant so dark scenes stop flickering. Costs a little "
        "card work. Speed: one more composition per refresh. Recommended: off; on if dark scenes flicker.",
        "Sur un OLED avec VRR, garde une fréquence constante pour que les scènes sombres ne scintillent "
        "plus. Coûte un peu de travail à la carte. Vitesse : une composition de plus par rafraîchissement. "
        "Recommandé : désactivé ; activé si les scènes sombres scintillent."},
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
    Entry{"On a fixed-rate display, every frame lasts the same time: no regular judder. Playback speed "
          "changes by at most 0.125 %, and sound follows inaudibly. Recommended: on.",
          "Sur un écran à fréquence fixe, chaque image dure le même temps : pas de saccade régulière. La "
          "vitesse de lecture change de 0,125 % au plus, et le son suit sans que ça s'entende. Recommandé : "
          "activé."},
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

}  // namespace kirari::app::play::i18n
