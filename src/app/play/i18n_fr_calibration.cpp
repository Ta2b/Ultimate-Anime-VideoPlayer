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
    Entry{"This display has a calibrated adaptive grade.", "Cet écran a un grade adaptatif calibré."},
    Entry{"Grade profile", "Profil de grade"},
    Entry{"Delete this profile", "Supprimer ce profil"},
    Entry{"New profile name", "Nom du nouveau profil"},
    Entry{"Calibrate a new profile", "Calibrer un nouveau profil"},
    Entry{"Recalibrate this profile", "Recalibrer ce profil"},
    Entry{"Profile {}", "Profil {}"},
    Entry{"This display does not give its identity (EDID): the calibration cannot be kept.",
          "Cet écran ne donne pas son identité (EDID) : la calibration ne peut pas être gardée."},
    Entry{"The calibration video is missing (calibration/calibration.mkv next to the player): reinstall "
          "the player.",
          "La vidéo de calibration manque (calibration/calibration.mkv à côté du lecteur) : réinstallez le "
          "lecteur."},
    Entry{"Sets the adaptive grade on this display from eight anime shots, dark, with a glow and bright: on "
          "each one, set the deep blacks, the colors and the middle gray for that scene. The grade then fits "
          "its curve to all of them at once, each scene at its own brightness. The result is kept for this "
          "display.",
          "Règle le grade adaptatif sur cet écran à partir de huit plans d'anime, sombres, avec éclat et "
          "clairs : sur chacun, réglez les noirs profonds, les couleurs et le gris moyen de cette scène. Le "
          "grade ajuste ensuite sa courbe sur tous à la fois, chaque scène à sa propre luminosité. Le "
          "résultat est gardé pour cet écran."},
    Entry{
        "Set each shot so it looks right, as it will be played in your episodes. Each one counts at its own "
        "brightness; Finish fits the grade to all of them.",
        "Réglez chaque plan pour qu'il soit juste, tel qu'il sera lu dans vos épisodes. Chacun compte à sa "
        "propre luminosité ; Terminer ajuste le grade sur tous."},
    Entry{"Shot {}/{} · {}", "Plan {}/{} · {}"},
    Entry{"Previous shot", "Plan précédent"},
    Entry{"Next shot", "Plan suivant"},
    Entry{"Measuring the shot…", "Mesure du plan…"},
    Entry{"{} of {} shots set.", "{} plans réglés sur {}."},
    Entry{"Opening the calibration video…", "Ouverture de la vidéo de calibration…"},
    // Les images d'exemple : leur titre, puis ce qu'il faut y voir.
    Entry{"Re:Zero · the staircase in the dark", "Re:Zero · l'escalier dans le noir"},
    Entry{"The stone blocks of the walls and every step of the staircase stay visible, and the area under "
          "the stairs on the left stays almost black. Ram's black tights stand out from her boots, and the "
          "small figure sitting at the top left can be made out. Steps or wall joints lost in black: lower "
          "Deep blacks or raise Middle gray. Grey, washed-out walls: raise Deep blacks.",
          "Les blocs de pierre des murs et chaque marche de l'escalier restent visibles, et le dessous de "
          "l'escalier à gauche reste presque noir. Les collants noirs de Ram se détachent de ses bottes, et"
          " on distingue la petite silhouette assise en haut à gauche. Marches ou joints des murs perdus "
          "dans le noir : baissez Noirs profonds ou montez Gris moyen. Murs gris et délavés : montez Noirs "
          "profonds."},
    Entry{"Re:Zero · Ram in the shadow", "Re:Zero · Ram dans l'ombre"},
    Entry{"Ram's hair stays a soft, dusty pink, neither bright nor flat; her red eye, its white highlight "
          "and her eyelashes are sharp. The dark background on the right stays almost black, without "
          "turning grey. Face too dark to read: raise Middle gray. Greyish background or glowing hair: lower "
          "it. Hair turning fuchsia: lower Colors.",
          "Les cheveux de Ram restent d'un rose doux et poudré, ni vif ni plat ; son œil rouge, son reflet "
          "blanc et ses cils sont nets. Le fond sombre à droite reste presque noir, sans virer au gris. "
          "Visage trop sombre pour se lire : montez Gris moyen. Fond grisâtre ou cheveux qui brillent : "
          "baissez-le. Cheveux qui virent au fuchsia : baissez Couleurs."},
    Entry{"Re:Zero · the green magic circle", "Re:Zero · le cercle magique vert"},
    Entry{"The swirl stays a bright green with lighter edges, and its glow on the steps on the right is a "
          "soft, dark green. Ram's blouse stays white with a slight green tint, her skirt a pale turquoise."
          " The bricks of the wall on the left can still be seen in the dark. Neon or flat green: lower "
          "Colors. Bricks lost in black: lower Deep blacks.",
          "Le tourbillon reste d'un vert vif aux bords plus clairs, et sa lueur sur les marches à droite "
          "est d'un vert sombre et doux. Le chemisier de Ram reste blanc à peine teinté de vert, sa jupe "
          "d'un turquoise pâle. Les briques du mur à gauche se devinent encore dans le noir. Vert fluo ou "
          "plat : baissez Couleurs. Briques perdues dans le noir : baissez Noirs profonds."},
    Entry{"Re:Zero · Rem's glowing horn", "Re:Zero · la corne lumineuse de Rem"},
    Entry{"The horn glows white at its core, with a violet halo that stays around the head. Rem's blue hair"
          " keeps its strands, her collar stays a soft pink, and the cloak she lies on stays a dark reddish"
          " brown with visible folds. The wall behind stays a dark grey blue. Halo spreading as a flat "
          "magenta, or a glaring pink collar: lower Colors. Folds of the cloak lost in black: "
          "lower Deep blacks, or raise Middle gray. Glaring white core: lower "
          "Middle gray.",
          "La corne brille de blanc en son cœur, avec un halo violet qui reste autour de la tête. Les "
          "cheveux bleus de Rem gardent leurs mèches, son col reste d'un rose doux, et la cape sur laquelle"
          " elle est couchée reste d'un brun rouge sombre aux plis visibles. Le mur derrière reste d'un "
          "gris bleu sombre. Halo étalé en magenta plat, ou col d'un rose criard : baissez Couleurs. Plis de "
          "la cape perdus dans le noir : baissez Noirs profonds, ou "
          "montez Gris moyen. Cœur blanc éblouissant : baissez Gris moyen."},
    Entry{"Re:Zero · the dragon under the blue sky", "Re:Zero · le dragon sous le ciel bleu"},
    Entry{"The sky stays a deep blue with light, wispy clouds you can tell apart; the white pillar keeps "
          "its grooves and the grey shading on its right side; the dragon's scales and the dark gaps of its"
          " wing stay visible. Pillar turned into a flat white bar, or a dazzling sky: lower Middle gray. "
          " Dark parts of the dragon lost in black: lower Deep blacks.",
          "Le ciel reste d'un bleu profond, avec des nuages légers et effilés qu'on distingue ; le pilier "
          "blanc garde ses cannelures et l'ombre grise de son côté droit ; les écailles du dragon et les "
          "creux sombres de son aile restent visibles. Pilier devenu une barre blanche plate, ou ciel "
          "éblouissant : baissez Gris moyen. Parties sombres du dragon perdues dans le noir :"
          " baissez Noirs profonds."},
    Entry{"Slime · the valley and the river", "Slime · la vallée et la rivière"},
    Entry{"The river shines but keeps its banks and its small ripples; the leaves of the tree stay readable"
          " against the light, and the trunk and its shadow on the grass stay a dark brown, not black. The "
          "sky over the mountains stays a pastel pink. River burnt into a white strip: lower Middle gray. "
          " Trunk and foliage crushed into black: lower Deep blacks.",
          "La rivière brille mais garde ses berges et ses petites vagues ; les feuilles de l'arbre restent "
          "lisibles à contre-jour, et le tronc et son ombre sur l'herbe restent d'un brun sombre, pas "
          "noirs. Le ciel au-dessus des montagnes reste d'un rose pastel. Rivière brûlée en bande blanche :"
          " baissez Gris moyen. Tronc et feuillage écrasés en noir : baissez Noirs profonds."},
    Entry{"Slime · the pink sky", "Slime · le ciel rose"},
    Entry{"The clouds stay a pastel pink against a lavender sky, the faces keep a natural, warm skin, and "
          "the man's tie is the only real red. Clouds turning red or orange, or reddened faces: lower "
          "Colors. The man's dark collar and the shadows on his face keep their detail: "
          "if they go black, lower Deep blacks.",
          "Les nuages restent d'un rose pastel sur un ciel lavande, les visages gardent une peau naturelle "
          "et chaude, et la cravate de l'homme est le seul vrai rouge. Nuages qui virent au rouge ou à "
          "l'orange, ou visages rougis : baissez Couleurs. Le col sombre de l'homme "
          "et les ombres de son visage gardent leurs détails : s'ils deviennent noirs, baissez Noirs "
          "profonds."},
    Entry{"Re:Zero · Emilia in daylight", "Re:Zero · Emilia en plein jour"},
    Entry{"Emilia's hair stays white with lilac shading, the blush on her cheeks stays a light pink, and "
          "the paving stones keep their joints. The violet collar and the gold brooch stay rich without "
          "glowing. Hair turned flat white or dazzling stones: lower Middle gray. Cheeks "
          "turning red: lower Colors.",
          "Les cheveux d'Emilia restent blancs avec des ombres lilas, le rose de ses joues reste léger, et "
          "les pavés gardent leurs joints. Le col violet et la broche dorée restent riches sans briller. "
          "Cheveux devenus d'un blanc plat ou pavés éblouissants : baissez Gris moyen. Joues "
          "qui virent au rouge : baissez Couleurs."},
    // Grade adaptatif : « Cette scène » et l'apprentissage (grade_learning.h).
    Entry{"This scene", "Cette scène"},
    Entry{"The scene on screen is {:.0f} % dark, {:.0f} % bright and {:.0f} % glow: a change here goes to "
          "those "
          "scene kinds in that share, and is remembered for learning.",
          "La scène à l'écran est sombre à {:.0f} %, claire à {:.0f} % et avec éclat à {:.0f} % : un réglage "
          "ici "
          "va à ces genres de scène dans cette part, et il est retenu pour l'apprentissage."},
    Entry{"Deep blacks, this scene", "Noirs profonds, cette scène"},
    Entry{"Colors, this scene", "Couleurs, cette scène"},
    Entry{"Middle gray, this scene", "Gris moyen, cette scène"},
    Entry{"Learn from my corrections", "Apprendre de mes corrections"},
    Entry{"Forget my corrections", "Oublier mes corrections"},
    Entry{
        "{} corrections and {} approved scenes kept. Learning fits the scene thresholds and the dark, glow, "
        "bright and blue-sky values to all of them at once, recent ones first; what they do not contradict "
        "stays as calibrated.",
        "{} corrections et {} scènes approuvées retenues. L'apprentissage ajuste les seuils des scènes et "
        "les "
        "valeurs sombre, avec éclat, claire et ciel bleu sur toutes à la fois, les plus récentes d'abord ; "
        "ce "
        "qu'elles ne contredisent pas reste tel que calibré."},
    Entry{"At least two corrections are needed: set a few scenes with \"This scene\" first.",
          "Il faut au moins deux corrections : réglez d'abord quelques scènes avec « Cette scène »."},
    Entry{"Grade learned from {} corrections: gap {:.0f} % → {:.0f} %",
          "Grade appris de {} corrections : écart {:.0f} % → {:.0f} %"},
    Entry{"Grade corrections forgotten", "Corrections du grade oubliées"},
};

}  // namespace

std::span<const Entry> french_calibration() {
  return kCalibration;
}

}  // namespace ajp::app::play::i18n
