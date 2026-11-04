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
    Entry{
        "Sets the adaptive grade on this display from five required anime shots (night, half-light, "
        "daylight, sky, bright lamps) and thirteen bonus ones: on each one, set the deep blacks, the colors "
        "and "
        "the middle gray for that scene. The grade then fits its curve to all of them at once, each scene "
        "at its own brightness. The result is kept for this display.",
        "Règle le grade adaptatif sur cet écran à partir de cinq plans d'anime obligatoires (nuit, pénombre, "
        "plein jour, ciel, lampes en scène claire) et de treize plans bonus : sur chacun, réglez les noirs "
        "profonds, "
        "les couleurs et le gris moyen de cette scène. Le grade ajuste ensuite sa courbe sur tous à la fois, "
        "chaque scène à sa propre luminosité. Le résultat est gardé pour cet écran."},
    Entry{
        "Set each shot so it looks right, as it will be played in your episodes. Each one counts at its own "
        "brightness; Finish fits the grade to all of them.",
        "Réglez chaque plan pour qu'il soit juste, tel qu'il sera lu dans vos épisodes. Chacun compte à sa "
        "propre luminosité ; Terminer ajuste le grade sur tous."},
    Entry{"Shot {}/{} · {}", "Plan {}/{} · {}"},
    Entry{"Previous shot", "Plan précédent"},
    Entry{"Next shot", "Plan suivant"},
    Entry{"Measuring the shot…", "Mesure du plan…"},
    Entry{"{} of {} required shots set, {} bonus.", "{} plans obligatoires réglés sur {}, {} bonus."},
    Entry{"Bonus shot {}/{} · {}", "Plan bonus {}/{} · {}"},
    Entry{"Go further: bonus shots", "Aller plus loin : plans bonus"},
    Entry{"Opening the calibration video…", "Ouverture de la vidéo de calibration…"},
    // Les images d'exemple : leur titre, puis ce qu'il faut y voir.
    Entry{"Akudama Drive · the cell and its shaft of light",
          "Akudama Drive · la cellule et son rai de lumière"},
    Entry{"The shaft of light through the door stays bright with a soft pink glow; the bars and the "
          "walls of the cell stay a dark, readable grey blue, and the floor at the bottom stays "
          "almost black. Bars lost in black: lower Deep blacks. Shaft spreading into a white flare: "
          "lower Middle gray. Greyish, washed-out cell: raise Deep blacks.",
          "Le rai de lumière de la porte reste vif, avec un halo rose doux ; les barreaux et les murs"
          " de la cellule restent d'un gris bleu sombre mais lisible, et le sol en bas reste presque "
          "noir. Barreaux perdus dans le noir : baissez Noirs profonds. Rai qui s'étale en un éclat "
          "blanc : baissez Gris moyen. Cellule grisâtre, délavée : montez Noirs profonds."},
    Entry{"Weathering with You · Hina in the half-light", "Les Enfants du Temps · Hina dans la pénombre"},
    Entry{"Hina's face, lit from the side, keeps a natural, warm skin; her hair stays a dark brown "
          "with visible strands, and the room behind her stays a deep blue grey without turning "
          "black. Face too dark to read: raise Middle gray. Pale, glowing face: lower it. Greyish "
          "background: raise Deep blacks.",
          "Le visage d'Hina, éclairé de côté, garde une peau naturelle et chaude ; ses cheveux "
          "restent d'un brun sombre aux mèches visibles, et la pièce derrière elle reste d'un gris "
          "bleu profond sans devenir noire. Visage trop sombre pour être lu : montez Gris moyen. "
          "Visage pâle qui brille : baissez-le. Fond grisâtre : montez Noirs profonds."},
    Entry{"Weathering with You · Hina in the rain", "Les Enfants du Temps · Hina sous la pluie"},
    Entry{"Hina's skin stays a natural pinkish beige, neither pale nor orange; her white top keeps "
          "its folds, and the wet paving stones keep their reflections. Pale, washed-out skin or a "
          "glaring top: lower Middle gray. Orange or reddened skin: lower Colors.",
          "La peau d'Hina reste d'un beige rosé naturel, ni pâle ni orangé ; son haut blanc garde ses"
          " plis, et les pavés mouillés gardent leurs reflets. Peau pâle, délavée, ou haut "
          "éblouissant : baissez Gris moyen. Peau orangée ou rougie : baissez Couleurs."},
    Entry{"Violet Evergarden · the town under the sun", "Violet Evergarden · la ville sous le soleil"},
    Entry{"The sun shines but stays a small disc; the sky stays a deep blue that lightens toward the "
          "horizon, the clouds stay white with soft shading, and the roofs of the town keep their "
          "colors. Sky burnt into white around the sun: lower Middle gray. Sea and roofs crushed into"
          " black: lower Deep blacks. Electric blue sky: lower Colors.",
          "Le soleil brille mais reste un petit disque ; le ciel reste d'un bleu profond qui "
          "s'éclaircit vers l'horizon, les nuages restent blancs avec des ombres douces, et les toits"
          " de la ville gardent leurs couleurs. Ciel brûlé en blanc autour du soleil : baissez Gris "
          "moyen. Mer et toits écrasés dans le noir : baissez Noirs profonds. Ciel d'un bleu "
          "électrique : baissez Couleurs."},
    Entry{"Akudama Drive · the hall of lamps", "Akudama Drive · la salle aux lampes"},
    Entry{"The lamps on the pillars and the ceiling glow, but the gold of the pillars and the walls "
          "between them stay readable, and the small figures on the floor keep their colors. Lamps "
          "bleeding into the walls, or a dazzling ceiling: lower Middle gray. Gold turning orange: "
          "lower Colors.",
          "Les lampes des piliers et le plafond brillent, mais l'or des piliers et les murs entre eux"
          " restent lisibles, et les petits personnages au sol gardent leurs couleurs. Lampes qui "
          "débordent sur les murs, ou plafond éblouissant : baissez Gris moyen. Or qui vire à "
          "l'orange : baissez Couleurs."},
    Entry{"Akudama Drive · spotlights in the fog", "Akudama Drive · projecteurs dans la brume"},
    Entry{"The spotlights stay small, sharp points; the fog stays a smooth pink and violet, without "
          "banding, and the dark shapes in front keep their edges. Fog turning a flat fuchsia: lower "
          "Colors. Spotlights spreading into large white blobs: lower Middle gray.",
          "Les projecteurs restent des points petits et nets ; la brume reste d'un rose et d'un "
          "violet lisses, sans bandes, et les formes sombres devant gardent leurs contours. Brume qui"
          " vire au fuchsia plat : baissez Couleurs. Projecteurs qui s'étalent en grosses taches "
          "blanches : baissez Gris moyen."},
    Entry{"Frieren · the heroes in the forest", "Frieren · les héros dans la forêt"},
    Entry{"The foliage stays a natural green, with patches of light and shade; the white cloak keeps "
          "its folds, and the faces keep a natural skin. Fluorescent green leaves: lower Colors. "
          "Cloak turned into a flat white: lower Middle gray. Shade under the trees lost in black: "
          "lower Deep blacks.",
          "Le feuillage reste d'un vert naturel, avec des taches de lumière et d'ombre ; la cape "
          "blanche garde ses plis, et les visages gardent une peau naturelle. Feuilles d'un vert fluo"
          " : baissez Couleurs. Cape devenue d'un blanc plat : baissez Gris moyen. Ombre sous les "
          "arbres perdue dans le noir : baissez Noirs profonds."},
    Entry{"Violet Evergarden · Violet in white", "Violet Evergarden · Violet dans le blanc"},
    Entry{"The sheets stay white with grey folds, Violet's skin stays pale but warm, and her blue "
          "eyes stay the most vivid color. Sheets turning into a glaring white: lower Middle gray. "
          "Grey, dull sheets: raise it. Greyish skin: raise Colors.",
          "Les draps restent blancs avec des plis gris, la peau de Violet reste pâle mais chaude, et "
          "ses yeux bleus restent la couleur la plus vive. Draps d'un blanc éblouissant : baissez "
          "Gris moyen. Draps gris et ternes : montez-le. Peau grisâtre : montez Couleurs."},
    Entry{"Weathering with You · the fireworks", "Les Enfants du Temps · le feu d'artifice"},
    Entry{"The sparks stay sharp gold and green streaks, the brightest ones white at their core, over"
          " a dark city. Sparks merging into a glowing haze: lower Middle gray. Neon green or orange "
          "sparks: lower Colors. City lost in black: lower Deep blacks.",
          "Les étincelles restent des traits dorés et verts bien nets, les plus vives blanches en "
          "leur cœur, au-dessus d'une ville sombre. Étincelles fondues en une brume lumineuse : "
          "baissez Gris moyen. Étincelles vert fluo ou orange : baissez Couleurs. Ville perdue dans "
          "le noir : baissez Noirs profonds."},
    Entry{"Frieren · the old priest in the shadow", "Frieren · le vieux prêtre dans l'ombre"},
    Entry{"The old man's hair stays a pale beige, his cloak a warm brown with visible folds, and the "
          "brick wall behind him stays readable in the dark. Wall lost in black: lower Deep blacks. "
          "Orange, glowing hair: lower Colors. Face too dark to read: raise Middle gray.",
          "Les cheveux du vieil homme restent d'un beige pâle, sa cape d'un brun chaud aux plis "
          "visibles, et le mur de briques derrière lui reste lisible dans l'ombre. Mur perdu dans le "
          "noir : baissez Noirs profonds. Cheveux orangés qui brillent : baissez Couleurs. Visage "
          "trop sombre pour être lu : montez Gris moyen."},
    Entry{"Your Name · twilight above the clouds", "Your Name · le crépuscule au-dessus des nuages"},
    Entry{"The beam of light and the low sun shine without burning the sea of clouds, which keeps its"
          " gold and pink relief; Taki's face stays warm without turning orange, and his dark jacket "
          "keeps its folds. Clouds burnt into white: lower Middle gray. Orange skin or red sky: lower"
          " Colors. Jacket lost in black: lower Deep blacks.",
          "Le rayon de lumière et le soleil bas brillent sans brûler la mer de nuages, qui garde son "
          "relief doré et rose ; le visage de Taki reste chaud sans virer à l'orange, et sa veste "
          "sombre garde ses plis. Nuages brûlés en blanc : baissez Gris moyen. Peau orangée ou ciel "
          "rouge : baissez Couleurs. Veste perdue dans le noir : baissez Noirs profonds."},
    Entry{"Re:Zero · the blue dragon up close", "Re:Zero · le dragon bleu en gros plan"},
    Entry{"The dragon's scales stay a deep, even blue with visible shading, its yellow eye glows "
          "without blooming, and its white teeth stay sharp. Blue banding or a neon, flat blue: "
          "lower Colors. Scales lost in black: lower Deep blacks. Teeth glaring white: lower Middle "
          "gray.",
          "Les écailles du dragon restent d'un bleu profond et régulier, avec leur modelé, son œil "
          "jaune brille sans baver et ses dents blanches restent nettes. Bandes dans le bleu ou "
          "bleu fluo et plat : baissez Couleurs. Écailles perdues dans le noir : baissez Noirs "
          "profonds. Dents d'un blanc éblouissant : baissez Gris moyen."},
    Entry{"Re:Zero · Ram on the dark stairs", "Re:Zero · Ram dans l'escalier sombre"},
    Entry{"The stone stairs and the walls stay readable in a dark green grey, Ram's pink hair and "
          "her dress keep their color, and the old man at the back stays visible. Stairs lost in "
          "black: lower Deep blacks. Whole scene too dark: raise Middle gray. Greyish, foggy stone: "
          "raise Deep blacks.",
          "L'escalier de pierre et les murs restent lisibles dans un gris vert sombre, les cheveux "
          "roses de Ram et sa robe gardent leur couleur, et le vieil homme au fond reste visible. "
          "Escalier perdu dans le noir : baissez Noirs profonds. Scène entière trop sombre : montez "
          "Gris moyen. Pierre grisâtre, voilée : montez Noirs profonds."},
    Entry{"Re:Zero · Rem and her glowing horn", "Re:Zero · Rem et sa corne lumineuse"},
    Entry{"The horn glows pink and white with a soft halo, Rem's blue hair stays smooth without "
          "bands or blotches, and her face stays pale. Bands or blotches in the blue hair or around "
          "the halo: lower Colors. Halo spreading into a white flare: lower Middle gray. Hood lost "
          "in black: lower Deep blacks.",
          "La corne brille de rose et de blanc avec un halo doux, les cheveux bleus de Rem restent "
          "lisses, sans bandes ni taches, et son visage reste pâle. Bandes ou taches dans le bleu "
          "des cheveux ou autour du halo : baissez Couleurs. Halo qui s'étale en éclat blanc : "
          "baissez Gris moyen. Capuche perdue dans le noir : baissez Noirs profonds."},
    Entry{"Slime · the pink sky", "Slime · le ciel rose"},
    Entry{"The pink and violet sky keeps its soft clouds and light rays, the two faces stay "
          "natural, and the dark jacket keeps its folds. Neon pink or violet sky: lower Colors. "
          "Clouds burnt into white: lower Middle gray. Jacket lost in black: lower Deep blacks.",
          "Le ciel rose et violet garde ses nuages doux et ses rayons, les deux visages restent "
          "naturels et la veste sombre garde ses plis. Ciel rose ou violet fluo : baissez Couleurs. "
          "Nuages brûlés en blanc : baissez Gris moyen. Veste perdue dans le noir : baissez Noirs "
          "profonds."},
    Entry{"Slime · the valley in the light", "Slime · la vallée dans la lumière"},
    Entry{"The river stays a bright silver ribbon without burning, the grass a warm green and the "
          "tree a deep, readable green. Glaring white river: lower Middle gray. Neon green grass: "
          "lower Colors. Tree lost in black: lower Deep blacks.",
          "La rivière reste un ruban argenté et lumineux sans brûler, l'herbe d'un vert chaud et "
          "l'arbre d'un vert profond et lisible. Rivière d'un blanc éblouissant : baissez Gris "
          "moyen. Herbe vert fluo : baissez Couleurs. Arbre perdu dans le noir : baissez Noirs "
          "profonds."},
    Entry{"Black Clover · the wall of fire", "Black Clover · le mur de feu"},
    Entry{"The wall of fire stays a deep orange with visible flames and windows, the blue lightning "
          "on the left stays a thin, sharp line, and the stone floor stays readable in the dark. "
          "Fire burning into yellow or white: lower Middle gray. Neon blue lightning or flat "
          "orange: lower Colors. Floor lost in black: lower Deep blacks.",
          "Le mur de feu reste d'un orange profond, flammes et fenêtres visibles, l'éclair bleu à "
          "gauche reste un trait fin et net, et le sol de pierre reste lisible dans le noir. Feu "
          "qui brûle en jaune ou en blanc : baissez Gris moyen. Éclair bleu fluo ou orange plat : "
          "baissez Couleurs. Sol perdu dans le noir : baissez Noirs profonds."},
    Entry{"Black Clover · the chain of light", "Black Clover · la chaîne de lumière"},
    Entry{"The chain and the magic circle glow a pale blue without bands, the red glove stays a "
          "vivid red, and the ruins behind stay a dark, readable blue green. Bands around the chain "
          "or a neon blue glow: lower Colors. Chain spreading into a white flare: lower Middle "
          "gray. Ruins lost in black: lower Deep blacks.",
          "La chaîne et le cercle magique brillent d'un bleu pâle sans bandes, le gant rouge reste "
          "d'un rouge vif et les ruines derrière restent d'un bleu vert sombre et lisible. Bandes "
          "autour de la chaîne ou lueur bleu fluo : baissez Couleurs. Chaîne qui s'étale en éclat "
          "blanc : baissez Gris moyen. Ruines perdues dans le noir : baissez Noirs profonds."},
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
    Entry{"Grade learned from {} corrections", "Grade appris de {} corrections"},
    Entry{"Grade corrections forgotten", "Corrections du grade oubliées"},
};

}  // namespace

std::span<const Entry> french_calibration() {
  return kCalibration;
}

}  // namespace ajp::app::play::i18n
