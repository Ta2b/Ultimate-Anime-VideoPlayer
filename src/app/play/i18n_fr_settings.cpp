#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Pages de réglages : (clé anglaise, français).
constexpr std::array kSettingsPages{
    Entry{" · RIFE (auto Instant)", " · RIFE (auto Instantané)"},
    Entry{" · no interpolation", " · sans interpolation"},
    Entry{"Display peak", "Pic de l'écran"},
    Entry{"SPAN off", "SPAN coupé"},
    Entry{"Apply", "Appliquer"},
    Entry{"Calibration results", "Résultats de la calibration"},
    Entry{"Calibration done: no setting held the target, the lightest is kept ({}).",
          "Calibration terminée : aucun réglage ne tient la cadence visée, le plus léger est gardé ({})."},
    Entry{"working resolution {}p instead of {}p", "résolution de travail {}p au lieu de {}p"},
    Entry{"no interpolation", "sans interpolation"},
    Entry{"Calibration done: full quality kept, {:.1f} of {:.0f} frames per second measured.",
          "Calibration terminée : pleine qualité gardée, {:.1f} images par seconde mesurées sur {:.0f} "
          "visées."},
    Entry{"Calibration done: {} to hold {:.1f} of {:.0f} frames per second.",
          "Calibration terminée : {} pour tenir {:.1f} images par seconde sur {:.0f} visées."},
    Entry{"The next heavier setting lost {:.1f} % of its frames in this test.",
          "Le réglage juste au-dessus perdait {:.1f} % de ses images à ce banc."},
    Entry{
        "Calibration skipped: the engine did not install. Install it in Settings > General > Engines, then "
        "calibrate again (Video > Post-processing).",
        "Calibration non lancée : le moteur ne s'est pas installé. Installez-le dans Paramètres > Général > "
        "Moteurs, puis refaites la calibration (Vidéo > Post-traitement)."},
    Entry{"Cancel##calibration", "Annuler##calibration"},
    Entry{"Interpolation runs in Auto Instant mode. The working resolution goes up to the 1080p of the test "
          "sequence; SPAN doubles it.",
          "L'interpolation tourne en mode auto Instantané. La résolution de travail monte jusqu'à 1080p, la "
          "hauteur de la séquence d'essai ; SPAN la double."},
    Entry{"Measured by an older version: Auto keeps its estimate. Run the calibration again.",
          "Mesuré par une version plus ancienne : Auto garde son estimation. Refaites la calibration."},
    Entry{"Measured with another engine: Auto keeps its estimate. Run the calibration again.",
          "Mesuré avec un autre moteur : Auto garde son estimation. Refaites la calibration."},
    Entry{
        "Measured on another card or display: Auto keeps its estimate. Run the calibration again.",
        "Mesuré sur une autre carte ou un autre écran : Auto garde son estimation. Refaites la calibration."},
    Entry{"The graphics driver changed since this measure: run the calibration again to refresh it.",
          "Le pilote graphique a changé depuis cette mesure : refaites la calibration pour la rafraîchir."},
    Entry{"No calibration yet: the settings are an estimate from the card.",
          "Pas encore de calibration : les réglages sont estimés d'après la carte."},
    Entry{"Measured on battery: Auto keeps its estimate. Run the calibration again plugged in.",
          "Mesuré sur batterie : Auto garde son estimation. Refaites la calibration sur secteur."},
    Entry{"Playback closes during the calibration, then resumes at the same spot.",
          "La lecture se ferme pendant la calibration, puis reprend au même endroit."},
    Entry{"Run the calibration", "Lancer la calibration"},
    Entry{"Run the calibration again", "Refaire la calibration"},
    Entry{"SPAN: {} in FP16 · RIFE: {}", "SPAN : {} en FP16 · RIFE : {}"},
    Entry{"Testing settings on a very demanding sequence: 5 minutes, up to 40 with TensorRT (it builds its "
          "engines).",
          "Essai des réglages sur une séquence très éprouvante : 5 minutes, jusqu'à 40 avec TensorRT (il "
          "construit "
          "ses moteurs)."},
    Entry{"The calibration failed: {}", "La calibration a échoué : {}"},
    Entry{"The maximum, reached in a very bright episode; it fades out in a very dark one.",
          "Le maximum, atteint dans un épisode très clair ; l'effet s'efface dans un épisode très sombre."},
    Entry{"holds", "tient"},
    Entry{"unknown", "inconnu"},
    Entry{"Downscale", "Réduction"},
    Entry{"Upscale", "Agrandissement"},
    Entry{"1% low", "1 % low"},
    Entry{"{:.1f} % lost", "{:.1f} % d'images perdues"},
    Entry{"Green: recommended, with margin. Orange: holds just barely, or misses a little. Red: this "
          "machine cannot keep up.",
          "Vert : recommandé, avec de la marge. Orange : tient de justesse, ou rate de peu. Rouge : la "
          "machine ne suit pas."},
    Entry{"{} · {} · display {}p", "{} · {} · écran {}p"},
    Entry{" ; playback resumes at the same spot", " ; la lecture reprend au même endroit"},
    Entry{"Advanced settings", "Réglages avancés"},
    Entry{"After SPAN (fastest)", "Après SPAN (le plus rapide)"},
    Entry{"All statistics (Tab)", "Toutes les statistiques (Tab)"},
    Entry{"Application", "Application"},
    Entry{"Apply (reopens the video)", "Appliquer (rouvre la vidéo)"},
    Entry{"Audio language", "Langue audio"},
    Entry{"Auto (after SPAN above 1080p)", "Auto (après SPAN au-dessus de 1080p)"},
    Entry{"Auto (from the display)", "Auto (selon l'écran)"},
    Entry{"Auto peak ({} nits)", "Pic automatique ({} nits)"},
    Entry{"In content nits: KDE shows them brighter (× its SDR brightness / 203), so a 1000-nit screen reads "
          "lower here.",
          "En nits du contenu : KDE les affiche plus haut (× sa luminosité SDR / 203), un écran de 1000 nits "
          "lit donc moins ici."},
    Entry{"Autoplay next episode", "Lire l'épisode suivant"},
    Entry{"Balance", "Équilibre"},
    Entry{"Before SPAN (best quality)", "Avant SPAN (qualité maximale)"},
    Entry{"Color grading", "Étalonnage"},
    Entry{"Colors", "Couleurs"},
    Entry{"Look unreadable: {}", "Look illisible : {}"},
    Entry{"All settings", "Tous les réglages"},
    Entry{"Puts every setting back to its default value: chain, HDR, grade and colors, sound, playback. The "
          "engines folder and the language are kept.",
          "Remet tous les réglages à leur valeur par défaut : chaîne, HDR, grade et couleurs, son, lecture. "
          "Le dossier des moteurs et la langue sont conservés."},
    Entry{"Every setting goes back to its default. This cannot be undone.",
          "Tous les réglages reviennent à leur valeur par défaut. Impossible à annuler."},
    Entry{"Yes, reset everything", "Oui, tout réinitialiser"},
    Entry{"Keep my settings", "Garder mes réglages"},
    Entry{"Reset all settings", "Tout réinitialiser"},
    Entry{"Undo", "Annuler"},
    Entry{"Redo", "Rétablir"},
    Entry{"LUT contrast intensity", "Intensité LUT contraste"},
    Entry{"Contrast", "Contraste"},
    Entry{"Custom", "Personnalisé"},
    Entry{"Debanding", "Débandage"},
    Entry{"Deep blacks", "Noirs profonds"},
    Entry{"Defaults", "Valeurs par défaut"},
    Entry{"Discard changes", "Abandonner les changements"},
    Entry{"Display {:.0f} Hz", "Écran {:.0f} Hz"},
    Entry{"Display: ", "Écran : "},
    Entry{"EWA Lanczos", "EWA Lanczos"},
    Entry{"EWA Lanczos sharp", "EWA Lanczos net"},
    Entry{"Economy", "Économe"},
    Entry{"Engines", "Moteurs"},
    Entry{"Episodes", "Épisodes"},
    Entry{"Exact position", "Position exacte"},
    Entry{"Experimental", "Expérimental"},
    Entry{"Export", "Export"},
    Entry{"Replace the file and export", "Remplacer le fichier et exporter"},
    Entry{"Fallback", "Sinon"},
    Entry{"Fast", "Rapide"},
    Entry{"Instant", "Instantané"},
    Entry{"Interpolation level", "Niveau d'interpolation"},
    Entry{"Fastest", "Le plus rapide"},
    Entry{"Final scaling", "Mise à l'échelle finale"},
    Entry{"Finest", "Le plus fin"},
    Entry{"Frames", "Images"},
    Entry{"From the file", "Selon le fichier"},
    Entry{"Smoothness", "Fluidité"},
    Entry{"Sharpness", "Netteté"},
    Entry{"Sharpness: {} %", "Netteté : {} %"},
    Entry{"Crisper line art after upscaling, without halos. 0: untouched. Live, no effect on export.",
          "Des traits plus nets après l'agrandissement, sans halo. 0 : image intacte. En direct, sans effet "
          "sur "
          "l'export."},
    Entry{"Video", "Vidéo"},
    Entry{"Post-processing", "Post-traitement"},
    Entry{"Display", "Affichage"},
    Entry{"Audio and playback", "Audio et lecture"},
    Entry{"Sound", "Son"},
    Entry{"About", "À propos"},
    Entry{"Auto ({})", "Auto ({})"},
    Entry{"Auto (calibrated)", "Auto (calibré)"},
    Entry{"Processing level", "Niveau de traitement"},
    Entry{"From Economy (light on the graphics card) to Ultra. Auto picks the level of this machine. "
          "Changing a "
          "setting below switches to Custom.",
          "D'Économe (léger pour la carte graphique) à Ultra. Auto prend le niveau de cette machine. Changer "
          "un "
          "réglage plus bas passe en Personnalisé."},
    Entry{"General", "Général"},
    Entry{"HDR: on. The player follows your system's HDR setting.",
          "HDR : actif. Le lecteur suit le réglage HDR de votre système."},
    Entry{"HDR output", "Sortie HDR"},
    Entry{"Hide the bar when paused (burn-in)", "Masquer la barre en pause (marquage)"},
    Entry{"LUT highlight intensity", "Intensité LUT éclats"},
    Entry{"Interpolation", "Interpolation"},
    Entry{"Near-blacks become true black (0), up to the threshold. Live.",
          "Les quasi-noirs deviennent un vrai noir (0), jusqu'au seuil. En direct."},
    Entry{"None", "Aucun"},
    Entry{"Normal", "Normal"},
    Entry{"Nothing changes until \"Start\".", "Rien ne change avant « Commencer »."},
    Entry{"OLED black", "Noir OLED"},
    Entry{"OLED care", "Soin OLED"},
    Entry{"On", "Activé"},
    Entry{"On-screen statistics", "Statistiques à l'écran"},
    Entry{"Output", "Sortie"},
    Entry{"Peak (nits)", "Pic (nits)"},
    Entry{"Pipeline", "Chaîne"},
    Entry{"Playback", "Lecture"},
    Entry{"Plays the next file in the folder. N: next, Shift+N: previous, O: skip credits.",
          "Lit le fichier suivant du dossier. N : suivant, Maj+N : précédent, O : passer le générique."},
    Entry{"Preferred languages", "Langues préférées"},
    Entry{"Preset", "Préréglage"},
    Entry{"Quality", "Qualité"},
    Entry{"Recommended", "Recommandé"},
    Entry{"Reset size", "Taille d'origine"},
    Entry{"Run the wizard again", "Relancer l'assistant"},
    Entry{"Report a problem", "Signaler un problème"},
    Entry{"Copies the version, the system, the graphics card, the preset and the end of the log (and the one "
          "before it after a crash), with only the file names of your videos, then opens the report page. "
          "Paste the report there.",
          "Copie la version, le système, la carte graphique, le préréglage et la fin du journal (et celui "
          "d'avant après un plantage), avec le seul nom des fichiers de vos vidéos, puis ouvre la page de "
          "signalement. Collez-y le rapport."},
    Entry{"Report copied. Paste it into the page that opened.",
          "Rapport copié. Collez-le dans la page qui s'est ouverte."},
    Entry{"Report copied. Open {} and paste it there.", "Rapport copié. Ouvrez {} et collez-le."},
    Entry{"SDR", "SDR"},
    Entry{"SPAN reloads (built the first time for this size)",
          "SPAN est rechargé (construit la première fois pour cette taille)"},
    Entry{"SPAN switches to {} (built the first time for this size)",
          "SPAN passe en {} (construit la première fois pour cette taille)"},
    Entry{"Highlights", "Éclats"},
    Entry{"Limit rapid flashes", "Limiter les flashs rapides"},
    Entry{"Skin from faces (test)", "Peau des visages (essai)"},
    Entry{
        "Faces are found ahead of playback and the grade only treats their skin tones as skin: warm lamps of "
        "another tone light up again. A shot without a found face keeps the wide range. Takes effect with "
        "the next video.",
        "Les visages sont cherchés en avance sur la lecture, et le grade ne traite plus comme peau que leurs "
        "teintes : les lampes chaudes d'une autre teinte se rallument. Un plan sans visage trouvé garde la "
        "plage large. Prise en compte à la vidéo suivante."},
    Entry{"Surface white", "Blanc des surfaces"},
    Entry{"HDR: the white of walls and skin, in % of where the LUTs put it (about 70 % of the screen "
          "peak). Lower leaves more room to the lights, which still reach the peak.",
          "HDR : le blanc des murs et des peaux, en % de celui que donnent les LUT (environ 70 % du pic de "
          "l'écran). Plus bas, les lumières gagnent de l'écart ; elles montent toujours jusqu'au pic."},
    Entry{"HDR only: a burst of rapid flashes (more than three a second) is dimmed for a second, for "
          "photosensitive viewers. Off: flashes keep their full light.",
          "HDR seulement : une suite de flashs rapides (plus de trois par seconde) est atténuée pendant une "
          "seconde, pour les personnes photosensibles. Coupé : les flashs gardent toute leur lumière."},
    Entry{"SDR: HDR is off in your system, or the display is too dim for it. The Color page's grade still "
          "applies.",
          "SDR : le HDR est coupé dans votre système, ou l'écran n'a pas assez de lumière pour lui. Le grade "
          "de "
          "la page Couleurs s'applique toujours."},
    Entry{"Saturation", "Saturation"},
    Entry{"Scaler", "Filtre"},
    Entry{"Scales all text, panels included. Live.",
          "Agrandit ou réduit tous les textes, panneaux compris. En direct."},
    Entry{"Settings", "Paramètres"},
    Entry{"Setup wizard", "Assistant"},
    Entry{"Size", "Taille"},
    Entry{"Statistics", "Statistiques"},
    Entry{"Steady frame rate (VRR flicker)", "Cadence fixe (scintillement VRR)"},
    Entry{"Steady frame rate keeps VRR at a constant refresh so dark scenes do not flicker on OLED.",
          "La cadence fixe garde un rafraîchissement VRR constant : pas de scintillement des scènes sombres "
          "sur OLED."},
    Entry{"Subtitle language", "Langue des sous-titres"},
    Entry{"Subtitle delay", "Décalage des sous-titres"},
    Entry{"Delay (ms)", "Décalage (ms)"},
    Entry{"Reset delay", "Décalage à 0"},
    Entry{
        "Positive: subtitles later. Keys z and x: 100 ms. Ctrl+click the slider to type a value (up to "
        "±30,000 ms). This session only.",
        "Positif : les sous-titres plus tard. Touches z et x : 100 ms. Ctrl+clic sur le curseur pour saisir "
        "une valeur (jusqu'à ±30 000 ms). Pour cette séance seulement."},
    Entry{"Subtitle size", "Taille des sous-titres"},
    Entry{
        "The first preference whose tracks exist is used when a file opens. Applies to the next file.",
        "La première préférence dont les pistes existent est prise à l'ouverture. Vaut au prochain fichier."},
    Entry{"The scaler stretches the output to the window; EWA Lanczos is sharpest but heaviest. No effect on "
          "export.",
          "Le filtre étire la sortie vers la fenêtre ; EWA Lanczos est le plus net mais le plus lourd. Sans "
          "effet sur l'export."},
    Entry{"This machine", "Cette machine"},
    Entry{"Threshold (code steps)", "Seuil (pas de code)"},
    Entry{"To apply: {}. Closing the panel discards these changes.",
          "À appliquer : {}. Fermer le panneau abandonne ces changements."},
    Entry{"Tracks", "Pistes"},
    Entry{"True black", "Noir absolu"},
    Entry{"Ultra", "Ultra"},
    Entry{"Unknown graphics card", "Carte graphique inconnue"},
    Entry{"Upscaling", "Agrandissement"},
    Entry{"Video memory", "Mémoire vidéo"},
    Entry{"Debugging", "Débogage"},
    Entry{"Show held and cut pairs", "Montrer les paires tenues et coupées"},
    Entry{"While interpolation runs, each pair judged a held drawing (blue) or a scene cut (red) tints the "
          "screen for a moment, and goes to ajp.log with its time and scores. This session only.",
          "Pendant l'interpolation, chaque paire jugée dessin tenu (bleu) ou coupe de scène (rouge) teinte "
          "l'écran un instant, et va dans ajp.log avec son minutage et ses scores. Pour cette session "
          "seulement."},
    Entry{"fused DistilDRBA (four engines, built the first time for this size)",
          "DistilDRBA fusionné (quatre moteurs, construits la première fois pour cette taille)"},
    Entry{"fused RIFE (six engines, built the first time for this size)",
          "RIFE fusionné (six moteurs, construits la première fois pour cette taille)"},
    Entry{"libplacebo default", "Défaut de libplacebo"},
    Entry{"the RIFE model loads (built the first time for this size)",
          "le modèle RIFE est chargé (construit la première fois pour cette taille)"},
    Entry{"the interpolation engine changes", "le moteur d'interpolation change"},
    Entry{"the interpolation warp turns Catmull-Rom", "le warp de l'interpolation passe en Catmull-Rom"},
    Entry{"the interpolation warp turns bilinear", "le warp de l'interpolation repasse en bilinéaire"},
    Entry{"held drawings are smoothed (frames read ahead)",
          "les dessins tenus sont lissés (images lues en avance)"},
    Entry{"held drawings keep their rhythm", "les dessins tenus gardent leur rythme"},
    Entry{"fast motion is estimated at half size (second engine)",
          "les mouvements rapides sont estimés à demi-taille (second moteur)"},
    Entry{"fast motion is estimated at full size", "les mouvements rapides sont estimés en pleine taille"},
    Entry{"the frame ring grows for this interpolation",
          "l'anneau d'images s'agrandit pour cette interpolation"},
    Entry{"the interpolation engine loads (built the first time for this size)",
          "le moteur d'interpolation est chargé (construit la première fois pour cette taille)"},
    Entry{"the output device changes", "l'appareil de sortie change"},
    Entry{"warp order follows the output", "l'ordre du warp suit la sortie"},
    Entry{"warp runs after SPAN", "le warp passe après SPAN"},
    Entry{"warp runs before SPAN", "le warp passe avant SPAN"},
    Entry{"whole DistilDRBA instead of fused DistilDRBA",
          "DistilDRBA entier à la place du DistilDRBA fusionné"},
    Entry{"whole RIFE instead of fused RIFE", "RIFE entier à la place du RIFE fusionné"},
    Entry{"{} switches to {} (built the first time for this size)",
          "{} passe en {} (construit la première fois pour cette taille)"},
    Entry{"{} cannot be read: this session starts from the default settings and does not change the file",
          "{} ne peut pas être lu : la séance part des réglages par défaut et ne touche pas au fichier"},
    Entry{"Invalid settings in play.toml went back to their default: {}",
          "Réglages invalides de play.toml remis à leur défaut : {}"},
    Entry{"Settings cannot be saved: {}", "Les réglages ne peuvent pas être enregistrés : {}"},
};

}  // namespace

std::span<const Entry> french_settings() {
  return kSettingsPages;
}

}  // namespace ajp::app::play::i18n
