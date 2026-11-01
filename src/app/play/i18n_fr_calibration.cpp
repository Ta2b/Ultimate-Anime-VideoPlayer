#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// La calibration du grade adaptatif (settings_calibration.cpp) et ses images d'exemple
// (calibration_images.cpp) : (clé anglaise, français).
constexpr std::array kCalibration{
    Entry{"Adaptive grade calibration", "Calibration du grade adaptatif"},
    Entry{"The display receives no HDR: the calibration needs an HDR output.",
          "L'écran ne reçoit pas de HDR : la calibration demande une sortie HDR."},
    Entry{
        "Sets the two ends of the adaptive grade on this display: a dark scene, then a bright scene. The "
        "result is kept for this display and drives the grade.",
        "Règle les deux bouts du grade adaptatif sur cet écran : une scène sombre, puis une scène claire. Le "
        "résultat est gardé pour cet écran et pilote le grade."},
    Entry{"This display has a calibrated adaptive grade.", "Cet écran a un grade adaptatif calibré."},
    Entry{"Calibrate the adaptive grade", "Calibrer le grade adaptatif"},
    Entry{"Forget the grade calibration", "Oublier la calibration du grade"},
    Entry{"1/2 · Dark scene. Set the middle gray, the deep blacks and the colors while looking at the "
          "examples on the left (or at a dark scene of your video).",
          "1/2 · Scène sombre. Réglez le gris moyen, les noirs profonds et les couleurs en regardant les "
          "exemples à gauche (ou une scène sombre de votre vidéo)."},
    Entry{"When in doubt, keep the colors low (0.10 or less).",
          "Dans le doute, gardez les couleurs basses (0,10 ou moins)."},
    Entry{"2/2 · Bright scene. Set the middle gray, the deep blacks and the colors cut while looking at the "
          "examples on the left (or at a bright scene of your video).",
          "2/2 · Scène claire. Réglez le gris moyen, les noirs profonds et la réduction des couleurs en "
          "regardant les exemples à gauche (ou une scène claire de votre vidéo)."},
    Entry{"Example {}/{} · {}", "Exemple {}/{} · {}"},
    Entry{"Previous example", "Exemple précédent"},
    Entry{"Next example", "Exemple suivant"},
    // Les images d'exemple : leur titre, puis ce qu'il faut y voir.
    Entry{"Night alley", "Ruelle de nuit"},
    Entry{
        "The six squares at the bottom: the first one stays black like the strip around it, the second is "
        "barely visible, the other four are clearly apart. The walls of the buildings stand out from the "
        "sky. Second square gone: lower Deep blacks. First square grey: raise them.",
        "Les six carrés du bas : le premier reste noir comme la bande autour, le deuxième se devine à peine, "
        "les quatre autres se distinguent nettement. Les murs des immeubles se détachent du ciel. Deuxième "
        "carré disparu : baissez Noirs profonds. Premier carré gris : montez-les."},
    Entry{"Face in the shadow", "Visage dans l'ombre"},
    Entry{"The face reads without effort: both eyes, their small highlights and the line between the lit "
          "cheek and the shadowed one. The room must still feel like night. Face hard to read: raise Mid "
          "gray. The room looks lit or greyish: lower it.",
          "Le visage se lit sans effort : les deux yeux, leurs petits reflets et la limite entre la joue "
          "éclairée et celle dans l'ombre. La pièce doit rester nocturne. Visage difficile à lire : montez "
          "Gris moyen. Pièce qui paraît éclairée ou grisâtre : baissez-le."},
    Entry{"Moonlit forest", "Forêt sous la lune"},
    Entry{
        "The lantern stays an orange red with a lighter yellow core and a soft halo. The trees stay a "
        "deep, dark green and the sky a night blue. Flat bright red lantern or trees turning neon green: "
        "lower Colors.",
        "La lanterne reste d'un rouge orangé, avec un cœur jaune plus clair et un halo doux. Les arbres "
        "restent d'un vert sombre et profond, le ciel d'un bleu de nuit. Lanterne d'un rouge vif uniforme ou "
        "arbres vert fluo : baissez Couleurs."},
    Entry{"Candlelit room", "Chambre à la bougie"},
    Entry{
        "The halo around the flame fades smoothly from yellow to orange then brown, without a red ring or "
        "bands. The book and the edge of the table stay readable, the corners of the room stay dark. Red "
        "ring: lower Colors. Corners crushed into one black: lower Deep blacks.",
        "Le halo autour de la flamme passe en douceur du jaune à l'orange puis au brun, sans anneau rouge "
        "ni bandes. Le livre et le bord de la table restent lisibles, les coins de la pièce restent sombres. "
        "Anneau rouge : baissez Couleurs. Coins écrasés en un seul noir : baissez Noirs profonds."},
    Entry{"Summer sky", "Ciel d'été"},
    Entry{"The clouds keep their relief: grey shading underneath, bright tops, and the sky stays blue up to "
          "the horizon. Clouds turned into flat white blobs or a dazzling picture: lower Middle gray, bright "
          "scene. Dull picture: raise it.",
          "Les nuages gardent leur relief : ombre grise dessous, sommets lumineux, et le ciel reste bleu "
          "jusqu'à l'horizon. Nuages devenus des taches blanches plates ou image éblouissante : baissez Gris "
          "moyen, scène claire. Image terne : montez-le."},
    Entry{"Desert in the sun", "Désert au soleil"},
    Entry{
        "The sand stays ochre and orange, the shaded slope brown; only the small flag is truly red. Sand "
        "or sky near the horizon turning red or pink: raise Colors cut, bright scene.",
        "Le sable reste ocre et orangé, la pente à l'ombre brune ; seul le petit drapeau est vraiment rouge. "
        "Sable ou ciel près de l'horizon qui vire au rouge ou au rose : montez Couleurs réduites, scène "
        "claire."},
    Entry{"Daylight portrait", "Portrait en plein jour"},
    Entry{"The skin stays a natural peach, with a light blush on the cheeks only; the white shirt keeps its "
          "grey folds. Red or orange skin: raise Colors cut, bright scene. Folds gone in the white: lower "
          "Middle gray, bright scene.",
          "La peau reste d'un pêche naturel, avec un léger rose sur les joues seulement ; la chemise blanche "
          "garde ses plis gris. Peau rouge ou orange : montez Couleurs réduites, scène claire. Plis "
          "perdus dans le blanc : baissez Gris moyen, scène claire."},
    Entry{"Snow and outlines", "Neige et contours"},
    Entry{"The black outlines of the house and the snowman stay pure black and sharp. In the bottom strip, "
          "the first square is black and the next five are distinct dark greys. Squares merged together: "
          "lower Deep blacks, bright scene. Grey outlines: raise them.",
          "Les contours noirs de la maison et du bonhomme de neige restent d'un noir pur et nets. Dans la "
          "bande "
          "du bas, le premier carré est noir et les cinq suivants des gris sombres distincts. Carrés fondus "
          "ensemble : baissez Noirs profonds, scène claire. Contours gris : montez-les."},
};

}  // namespace

std::span<const Entry> french_calibration() {
  return kCalibration;
}

}  // namespace ajp::app::play::i18n
