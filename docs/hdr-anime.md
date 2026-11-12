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
  nits sur un écran de 1 000 nits). Le réglage avancé « Blanc des surfaces » (50 à 100 %, 100 par défaut) le
  baisse : la courbe est alors résolue pour rendre ce blanc (à 70 %, vers la moitié du pic, comme la rc.1 et
  la rc.2), et les éclats montent toujours jusqu'au pic.
- **Éclats après la pile, jusqu'au pic de l'écran (A41-02).** La passe anime décide quels pixels sont des
  lumières et transmet leur gain dans l'alpha ; la dernière passe l'applique après la pile et la courbe, sous
  le pic de l'écran (`emitter_level`, `src/render/output_tune.h`). Avant, la pile et la courbe divisaient leur
  écart sur les surfaces par deux (un émetteur à 804 nits pour des surfaces à 690). Leur niveau ne dépend ni
  de la scène ni des curseurs de la pile.
- **Luminosité stable d'un plan à l'autre en HDR (Tatoub, 11/11/2026, essai de la rc.4).** Les surfaces
  gardent le relèvement fixe : plus d'exposition par plan (jusqu'à ±19 %, que la pile double à peu près),
  qui faisait sauter un même décor entre un gros plan sombre et un plan large clair. Plus de limiteur de
  scène : la marge des éclats ne tombe plus à la moitié du pic dans une scène claire, la dalle limite
  elle-même. Les deux se déclenchaient bien plus souvent depuis que les coupes de l'animation tenue sont
  trouvées (#676). La sortie SDR garde sa marge des éclats par plan (`steady_hdr`, `render/grade_params.h`). Restent
  adaptatifs : les lumières décidées zone par zone, et les noirs profonds selon la série. Piste si une scène
  paraît trop sombre ou trop claire : une exposition par décor (une suite de plans du même lieu), glissée
  lentement d'un décor à l'autre.
- **Exposition par plan retirée aussi en SDR (Tatoub, 12/11/2026, audit C41-01).** En sortie SDR, la courbe
  vers le SDR ramenait le blanc des surfaces (`tm_white`) avec l'exposition du plan et l'annulait : elle ne
  changeait rien. Plutôt que de la rendre visible (les mêmes sauts qu'en HDR), elle est retirée partout. Les
  surfaces gardent le relèvement fixe (`kAnimeLift`, 1,15) ; la marge des éclats (`tm_top`) suit toujours le
  plan.
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
| Émetteur blanc, Éclats 50 % | | | 699 |
| Émetteur blanc, Éclats 100 %, toute scène | 793 | 690 | 860 |
| Émetteur blanc, Éclats 200 %, toute scène | 899 | 975 | 997 |

Avant les éclats après la pile (rc.4 et rc.5), l'émetteur sortait à 804 nits (Éclats 100 %) et 899 (200 %) en
scène sombre, moins en scène claire. Sur un écran de 400 nits : surfaces à 320, émetteur à 381 (Éclats 100 %).

Blanc des surfaces selon le pic : 320 nits à 400, 690 à 1 000, 1 304 à 2 000 (rc.1 et rc.2 : 278, 496, 992).

### Pistes ouvertes (décisions de Tatoub du 11/11/2026)

- **Scènes très claires sur OLED** : le lecteur laisse la dalle baisser l'image (son limiteur plein écran).
  Si une scène très claire (ciel, neige) fait « respirer » l'image sur l'OLED de Tatoub, la piste est que le
  lecteur baisse lui-même les surfaces seules, en gardant les éclats au pic et les tons moyens, à la coupe de
  plan. Rien n'est fait avant ce constat à l'œil.
- **Rendu universel** : le même rendu quel que soit le système et l'écran. La pile fondue en une seule LUT
  est écartée : l'audit A41-01 a mesuré qu'elle change l'image (18 codes PQ au pire, une couleur sur cinq à
  plus d'un code). La pile a maintenant 19 emplacements fixes : les curseurs ne changent que des intensités,
  sans refaire le hook, et l'image est la même octet pour octet. Restent : la passe de sortie allumée sur Mac (la pile y demande ~6 000 nits et
  macOS compresse à sa façon), et le temps de la passe du look mesuré sur la carte de Tatoub.

Ce que le cloud ne peut pas mesurer (la dalle, son limiteur, le compositeur) reste dans
`docs/audits/A-CONFIRMER.md`.

| Étape | Drapeau | Ce qui change | État |
| --- | --- | --- | --- |
| A1 | `--hdr-anime` | La carte des émetteurs, portes tirées des gains réels de `decide_grade` | fait |
| A2 | `--anime-shots` | Marge des éclats fixée par plan (pré-analyse, sortie SDR), posée à l'image de coupe, sans glissé (un changement à la coupe ne se voit pas ; `app/play/anime_shots.h`) | codé, à mesurer |
| A3 | `--anime-guided` | Carte locale lue par un sur-échantillonnage bilatéral joint guidé par le pixel : le halo ne déborde plus | codé, à mesurer |
| A4 | aucun | Visages : cascade LBP (lbpcascade_animeface), doublons, palette de peau par plan (`media/faces.h`, `media/face_cascade.h`) | branché derrière `face_skin`, éteint par défaut (« Peau des visages ») |
| A6 | aucun | Réseau des émetteurs : étiquettes (port numpy de A5), jeu de données, entraînement et export ONNX (`tools/grade/emitters/`) | scripts écrits, entraînement plus tard |
| A5 | aucun | Mesures d'émetteurs v2, référence CPU (`render/emitters.h`) : pixel, bloc, couronne ; néon admis | codé, juge et étiquettes |

## Les passes et leur coût (12/11)

Le grade tourne sur OUTPUT, à la taille de la fenêtre, pour chaque image affichée : avec RIFE ×2, deux fois par
source. Relevé sur la RTX 4070 Ti (fenêtre 1280×720, sortie HDR sous KWin, vrai noir, pile de quatre LUT) :
médiane 1,61 ms, p95 3,96 ms par image affichée. Sur lavapipe en 4K, la passe anime et le look pèsent chacun
environ 40 % du grade, la fin du grade 12 %, les cartes 1 %. Le journal de fin donne maintenant le temps de chaque
passe (« profil GPU · grade HDR · <passe> »).

Les passes, dans l'ordre : carte locale 1/8 et zone 1/32 (`anime-hdr-anime/1-local`, `2-area`), passe anime
(`3-anime`), look (`anime-hdr-look.hook`, passe de calcul en tuiles de 16 × 16), max local 8 × 8, puis une seule
passe finale pour le vrai noir, la compression vers l'écran et la sortie SDR (`anime-hdr-black.hook`). Ce qui a
changé, sans rien changer à l'image en sortie HDR (au bit près, à un trentième de code 10 bits près avec le vrai
noir) :

- vrai noir, compression et sortie SDR : trois passes pleine taille fondues en une ;
- sortie SDR : la mise en PQ de l'image SDR n'est plus une passe, `1-local` et `3-anime` la font à la lecture
  (l'image SDR gagne en précision : le tampon FP16 intermédiaire disparaît, écart moyen de 0,3 code 10 bits,
  plus proche de l'image exacte) ;
- passe anime : les quatre voisins du trait encré ne sont lus que pour une étincelle possible, la montée des
  émetteurs n'est calculée que sur un émetteur, la teinte de peau que là où une porte est ouverte ;
- look : la luminance du voisinage (3 × 3 points tous les 6 pixels) est décodée une fois par tuile en mémoire
  partagée, au lieu de neuf fois par pixel.

## A2 : décisions par plan
La lumière du plan (sa luminance moyenne) fixe la marge des éclats de la sortie SDR. Les surfaces gardent le
relèvement fixe (1,15) dans tous les plans : l'exposition par plan est retirée (C41-01).

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
`--anime-saturation=N` (chroma d'Oklab × N/100 à luminance gardée, atténuée vers le pic, sans canal négatif,
`an_chroma`) et
`--anime-highlights=N` (part de la montée des éclats, `an_spec`). Jamais écrits dans play.toml : les valeurs
retenues à l'œil deviendront le défaut de la méthode.

## Saturation en Oklab (12/11)

Le curseur Saturation écartait les canaux de la luminance en RVB linéaire : poussé, un bleu tournait vers le
violet (6 à 13° de teinte d'Oklab à 150 et 200 %), un orange vers le rouge. Il multiplie désormais la chroma
(a, b) d'Oklab par le facteur, puis remet la luminance BT.2020 d'origine par un facteur commun : la teinte
d'Oklab ne bouge plus (écart < 0,01° sur une simulation en Python de la passe), la luminance non plus. Un canal
qui passerait sous zéro fait reculer le facteur par dichotomie. À 100 %, la branche ne s'exécute pas : l'image
est identique bit à bit.

## Peau des visages (12/11, essai, éteint par défaut)

La carte des émetteurs exclut la peau (`skin_like`, audit F2) par sa couleur : teinte orangée de 10 à 52°,
saturation de 0,08 à 0,70. Une lampe à incandescence, une bougie ou un coucher de soleil tombent dans la même
plage et restaient éteints. L'option « Peau des visages (essai) » (Vidéo, réglages avancés du grade ; clé
`face_skin` de `[grade]` dans play.toml) resserre cette plage sur la peau des personnages du plan :

- la pré-analyse des plans (`media/shot_probe.h`) passe une image sur dix par la cascade de visages
  lbpcascade_animeface (`third_party/lbpcascade_animeface`, MIT), évaluée par `media/face_cascade.cpp` sans
  OpenCV, sur la luma réduite à 288 lignes et égalisée (fenêtre de 24 × 24, échelles de 1,1, cinq voisines) ;
  sur 75 images d'un clip, elle trouve les mêmes 18 visages qu'OpenCV ;
- la palette d'un visage est la médiane de 48 points sous ses yeux (30 à 70 % de la largeur du cadre, 55 à 85
  % de sa hauteur), mesurés comme le grade les voit (lumière BT.2020) ; le quart au moins doit avoir l'ordre
  R ≥ V ≥ B. Sur le même clip, les palettes d'un visage suivi sur neuf images restent entre 22 et 34° de
  teinte et 0,25 à 0,34 de saturation, et le plan garde 29,6° et 0,31 ;
- un plan garde jusqu'à trois teintes, chacune vue au moins deux fois (`media/faces.h`, `plan_skins`) et dans la
  plage large : la palette resserre la plage, elle ne l'élargit jamais. Le cache des plans les garde (version 6
  du fichier ; une autre variante quand l'option est allumée) ;
- le grade reçoit les teintes à la coupe, en sortie HDR comme SDR, et `skin_like` ne garde que 6° et 19 % de
  saturation autour d'elles (rien au-delà de 12° et 37 %).

Garde-fou : un plan sans visage trouvé garde la plage large. Risque à juger sur de vrais épisodes : un
personnage que la cascade ne voit pas (de profil, de dos, plus petit que 8 % de la hauteur) et dont la peau a
une autre teinte que celle des visages trouvés n'est plus protégé ; si sa peau est claire et isolée, elle peut
monter comme une lumière. Sous NVDEC réduit, la recherche décode à 512 × 288 au lieu de 256 × 144.

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
  source déjà dangereuse. La case « Limiter les flashs rapides » (page Couleurs, `grade.flash_guard`, allumée
  par défaut) le coupe : les flashs gardent alors toute leur lumière. La version 3 de l'analyse refait les
  fichiers marqués sous l'ancien blanc du grade, plus bas, qui manquaient des flashs.
- **Cache** (`media/shot_cache.h`). Un fichier `.shots` par vidéo, avec la version de l'analyse et le chemin de
  décodage (logiciel ou NVDEC réduit) : un fichier d'une autre version se refait.
