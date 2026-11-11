# Méthode « HDR anime » (--hdr-anime) : les essais du méga plan

La méthode garde les surfaces à un relèvement fixe et ne fait monter que les émetteurs
(`src/render/shaders/anime-hdr-anime/3-anime.hook`). Chaque étape ci-dessous est un essai en ligne de
commande ; sans son drapeau, rien ne change.

## Niveaux à l'écran (11/2026)

Ce que la chaîne complète rend sur une sortie HDR10 : la passe anime, la pile de LUT (`docs/luts.md`), puis
la dernière passe (`anime-hdr-black.hook`, réglée par `src/render/output_tune.h`).
`tests/unit/render/grade_chain_gpu_test.cpp` vérifie ces niveaux sur les LUT livrées.

- **Blanc des surfaces : là où la pile le porte, vers 70 % du pic.** La dernière passe applique la courbe de
  KWin sous laquelle la pile a été réglée à l'œil, de 10 000 nits vers le pic de l'écran. Les curseurs « LUT
  contrast intensity » et « LUT highlight intensity » déplacent ce blanc (de 483 à 818 nits et de 410 à 863
  nits sur un écran de 1 000 nits).
- **Éclats avant la pile.** La passe anime décide quels pixels sont des lumières et les monte jusqu'au pic ;
  la pile les amplifie ensuite.
- **Luminosité stable d'un plan à l'autre en HDR (Tatoub, 11/11/2026, essai de la rc.4).** Les surfaces
  gardent le relèvement fixe : plus d'exposition par plan (jusqu'à ±19 %, que la pile double à peu près),
  qui faisait sauter un même décor entre un gros plan sombre et un plan large clair. Plus de limiteur de
  scène : la marge des éclats ne tombe plus à la moitié du pic dans une scène claire, la dalle limite
  elle-même. Les deux se déclenchaient bien plus souvent depuis que les coupes de l'animation tenue sont
  trouvées (#676). La sortie SDR garde sa courbe par plan (`steady_hdr`, `render/grade_params.h`). Restent
  adaptatifs : les lumières décidées zone par zone, et les noirs profonds selon la série. Piste si une scène
  paraît trop sombre ou trop claire : une exposition par décor (une suite de plans du même lieu), glissée
  lentement d'un décor à l'autre.
- **MaxFALL annoncé** : le blanc des surfaces réellement rendu (une image toute blanche), ou la moyenne
  plein écran du contrat d'écran quand il la connaît.
- **Noirs (décision 17 du 09/11/2026).** Le noir pur reste un vrai noir : 0 sous RVB 4. Au-dessus, la LUT de
  base rendait presque noir tout ce qui était sous RVB 16 ; l'image suit un pied plus doux (pente 1,3 en log
  au lieu de 2), et la pile reprend toute sa part entre RVB 32 et RVB 48.
- **Sans pile** (fichier `.cube` illisible, ou les premières images avant sa lecture) : la dernière passe ne
  compresse pas une image que la pile n'a pas montée (A41-04). Le blanc reste à 233 nits au lieu de 119.

La 0.11.0-rc.1 et la rc.2 posaient le blanc des surfaces à la moitié du pic et appliquaient les éclats après
la pile (décision 16 du 09/11/2026). Sous ce rendu, la dernière passe ramenait le blanc à sa cible quelle que
soit la pile : les deux curseurs ne changeaient plus les surfaces claires, et les éclats perdaient leur
« peps ». Tatoub a demandé le 11/11/2026 de revenir au blanc et aux éclats d'avant, en gardant les noirs.

Mesuré sous lavapipe sur les shaders du lecteur (`ajp_render_tests "[.mesure]"`), source SDR au gamma 2,4,
écran de 1 000 nits, nits affichés :

| Code RVB 8 bits | 0.10 | rc.1 et rc.2 | Maintenant |
|---|---|---|---|
| 4 | 0 | 0 | 0 |
| 8 | 0,015 | 0,016 | 0,016 |
| 16 | 0,022 | 0,137 | 0,137 |
| 24 | 0,158 | 0,484 | 0,484 |
| 32 | 0,74 | 1,18 | 1,18 |
| 48 | 3,97 | 3,97 | 3,98 |
| 128 | 44,0 | 43,6 | 43,9 |
| 160 | 106 | 102 | 105 |
| 192 | 305 | 258 | 308 |
| 224 | 538 | 403 | 538 |
| 255 (blanc d'une surface) | 690 | 496 | 690 |
| Émetteur blanc, Éclats 100 %, scène sombre | 793 | 690 | 804 |
| Émetteur blanc, Éclats 200 %, scène sombre | 899 | 975 | 899 |

Blanc des surfaces selon le pic : 320 nits à 400, 690 à 1 000, 1 304 à 2 000 (rc.1 et rc.2 : 278, 496, 992).

### Pistes ouvertes (décisions de Tatoub du 11/11/2026)

- **Scènes très claires sur OLED** : le lecteur laisse la dalle baisser l'image (son limiteur plein écran).
  Si une scène très claire (ciel, neige) fait « respirer » l'image sur l'OLED de Tatoub, la piste est que le
  lecteur baisse lui-même les surfaces seules, en gardant les éclats au pic et les tons moyens, à la coupe de
  plan. Rien n'est fait avant ce constat à l'œil.
- **Rendu universel** : le même rendu quel que soit le système et l'écran. La pile fondue en une seule LUT
  est écartée : l'audit A41-01 a mesuré qu'elle change l'image (18 codes PQ au pire, une couleur sur cinq à
  plus d'un code). Restent : une pile de taille fixe (19 emplacements, les curseurs ne changent que des
  intensités, sans refaire le hook), la passe de sortie allumée sur Mac (la pile y demande ~6 000 nits et
  macOS compresse à sa façon), et le temps de la passe du look mesuré sur la carte de Tatoub.

Ce que le cloud ne peut pas mesurer (la dalle, son limiteur, le compositeur) reste dans
`docs/audits/A-CONFIRMER.md`.

| Étape | Drapeau | Ce qui change | État |
| --- | --- | --- | --- |
| A1 | `--hdr-anime` | La carte des émetteurs, portes tirées des gains réels de `decide_grade` | fait |
| A2 | `--anime-shots` | Exposition des surfaces et marge des éclats fixées par plan (pré-analyse), posées à l'image de coupe, sans glissé (un changement à la coupe ne se voit pas ; `app/play/anime_shots.h`) | codé, à mesurer |
| A3 | `--anime-guided` | Carte locale lue par un sur-échantillonnage bilatéral joint guidé par le pixel : le halo ne déborde plus | codé, à mesurer |
| A4 | aucun | Visages : décodage d'un détecteur, doublons, suivi, palette de peau, exclusion des émetteurs (`render/faces.h`) | partie sans réseau codée |
| A6 | aucun | Réseau des émetteurs : étiquettes (port numpy de A5), jeu de données, entraînement et export ONNX (`tools/grade/emitters/`) | scripts écrits, entraînement plus tard |
| A5 | aucun | Mesures d'émetteurs v2, référence CPU (`render/emitters.h`) : pixel, bloc, couronne ; néon admis | codé, juge et étiquettes |

## A2 : décisions par plan
L'exposition suit la règle de la courbe par plan (`render/curve.h`) : 0,25 diaphragme par diaphragme
d'APL du plan rapportée à la médiane de la série, bornée à ±0,25. Sans plan analysé, le relèvement reste
fixe (1,15).

## A3 : écarts au plan
Le lissage temporel existait déjà sur la carte de zone (`2-area.hook`). Tout le grade se fait à l'écran.

## A4 : ce qui manque
Aucun modèle de détection de visages n'est livré ni téléchargé : la règle « aucun fichier NVIDIA dans les
releases » et l'absence d'un modèle libre validé sur l'anime l'excluent pour l'instant. Avec un modèle, le
détecteur tournerait dans l'étage TensorRT à basse résolution, ses sorties passeraient par
`decode_faces`, `suppress_faces` et `FaceTracker` (remis à zéro aux coupes), et `face_exclusion`
multiplierait la carte des émetteurs par 1 − sa part.

## A5 : mesures d'émetteurs v2
Trois échelles : le pixel, le bloc 8 × 8 et une couronne de blocs à une distance de 2 à 4 (les voisins
immédiats, souvent la même lumière, n'y entrent pas). Une lumière domine sa couronne d'un diaphragme au
moins ; la porte de niveau admet le blanc (55 à 95 % du blanc de la source) ou une lumière saturée dès 30 %
(néon, écran) ; l'étincelle est un pixel qui domine son bloc d'un diaphragme et demi. Testée sur le halo,
l'étincelle, la chemise blanche, le visage et le néon. Elle juge les essais du shader et étiquette le jeu de
données du réseau (A6) ; le shader garde ses portes tant que la mesure sur ton PC n'a pas tranché.

## A6 : le réseau appris
`tools/grade/emitters/` : `labels.py` (la carte v2 en numpy, `--self-test` contre le C++ à 1e-4),
`dataset.py` (des images tirées des vidéos par ffmpeg, en tuiles 256 × 256 avec leur étiquette au 1/8 ;
corrections à la main dans `corrections.tsv`), `train.py` (un petit réseau convolutif, perte BCE, export
ONNX). Le contrat : entrée `pq` 1 × 3 × H × W en codes PQ BT.2020, sortie `emitters` 1 × 1 × H/8 × W/8.
L'entraînement demande PyTorch sur un PC ; l'intégration TensorRT attend un modèle entraîné et mesuré.
Aucun modèle n'entre dans une release sans ton accord.

## Peps de la méthode (04/11)

Retour de Tatoub : `--hdr-anime`, `--anime-shots` et `--anime-guided` sont plus propres que le grade actuel mais
manquent de contraste, de saturation et de peps. Trois réglages en ligne de commande, en %, 100 = la méthode telle
quelle : `--anime-contrast=N` (puissance de la luminance autour de 0,4 × le blanc, `an_punch` = (N − 100)/100),
`--anime-saturation=N` (écart à la luminance × N/100, atténué vers le pic, sans canal négatif, `an_chroma`) et
`--anime-highlights=N` (part de la montée des éclats, `an_spec`). Jamais écrits dans play.toml : les valeurs
retenues à l'œil deviendront le défaut de la méthode.

## Noirs profonds selon l'épisode (04/11, texte corrigé le 09/11 : A38-16)

Sous la méthode HDR anime, le curseur « Noirs profonds » est le maximum, atteint dans un épisode très clair :
la clé est la médiane des luminances moyennes des plans déjà analysés de l'épisode (`series_apl_of`,
`src/app/play/anime_shots.cpp`), ≥ 40 % du blanc SDR. L'effet descend en douceur jusqu'à 0 dans un épisode
très sombre (≤ 3 %). La clé est celle de l'épisode, pas celle du plan : elle ne bouge pas aux coupes, pour
que les mêmes cheveux sombres sortent au même niveau d'un plan à l'autre (P3.52). Une scène de nuit dans un
épisode clair reçoit donc l'effet entier ; ses détails sont gardés par le pied adouci de la pile (voir
« Niveaux à l'écran »), pas par ce curseur. Tant que rien n'est analysé, la clé vaut 25 % du blanc SDR et
l'effet 64 % du curseur.

## Sortie HDR ou SDR : elle suit le système (11/2026)

Demande de Tatoub : plus de bouton HDR. La « Méthode HDR » du menu, l'écran HDR de l'assistant et la clé
`hdr.engine` de `play.toml` sont retirés (un ancien fichier se lit encore, la clé est ignorée). À chaque
ouverture, `player_run.cpp` décide avec `system_hdr` (`src/app/play/host_facts.h`) :

| Situation | Sortie |
|---|---|
| HDR actif dans Windows ou le compositeur, crête ≥ 350 nits (`kMinHdrPeakNits`) | HDR10, grade anime guidé |
| Écran HDR, mais HDR coupé dans le système | SDR gradée |
| Écran qui se dit HDR, crête < 350 nits (pas de lumière pour les éclats) | SDR gradée |
| `--sdr` | SDR gradée |

Pendant la lecture, l'état est relu toutes les 2 s : allumer ou couper le HDR du système bascule la sortie en
direct, sans rouvrir la chaîne. Le pic visé (`hdr.peak`, avancé) reste un réglage.

## Plans et flashs (pré-analyse)

La pré-analyse (`media/shot_probe.h`) découpe la vidéo en plans devant la lecture ; le grade y lit la lumière de
chaque plan, l'interpolation ses coupes (`--shot-cuts`, donc aussi sans grade).

- **Découpe** (`media/shot_cutter.cpp`). Une coupe est un saut plus grand que 0,28, ou que 2,5 fois le mouvement
  récent. Ce mouvement se mesure entre dessins distincts : en animation tenue sur deux ou trois images, un
  panoramique reste un plan. Un saut attend trois images avant d'être jugé : un éclair de une à trois images
  reste dans son plan.
- **Flashs** (`media/flash_guard.h`, UIT-R BT.1702-2). Plus de trois flashs par seconde, chacun d'au moins 20 cd/m²
  sur un quart de l'écran, l'image sombre sous 160 cd/m², en lumière de sortie du grade : le pic du grade tombe à
  250 nits, jusqu'à une seconde après le dernier. La pré-analyse marque ces séquences d'avance, le plafond les
  prend dès leur première transition ; sans elle, l'analyse à l'ingestion le pose après ce qu'elle a vu
  (875 ms à 4 Hz). Rien en sortie SDR. Le plafond retire l'amplification du grade, il ne rend pas sûre une
  source déjà dangereuse.
- **Cache** (`media/shot_cache.h`). Un fichier `.shots` par vidéo, avec la version de l'analyse et le chemin de
  décodage (logiciel ou NVDEC réduit) : un fichier d'une autre version se refait.
