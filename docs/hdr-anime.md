# Méthode « HDR anime » (--hdr-anime) : les essais du méga plan

La méthode garde les surfaces à un relèvement fixe et ne fait monter que les émetteurs
(`src/render/shaders/anime-hdr-anime/3-anime.hook`). Chaque étape ci-dessous est un essai en ligne de
commande ; sans son drapeau, rien ne change.

## Niveaux à l'écran (11/2026, décisions 16 et 17 de Tatoub)

Ce que la chaîne complète rend sur une sortie HDR10 : la passe anime, la pile de LUT (`docs/luts.md`), puis
la dernière passe (`anime-hdr-black.hook`, réglée par `src/render/output_tune.h`). Les audits 41 et 38 avaient
mesuré un écart entre ce que le grade décidait et ce qui sortait (A41-02, A41-03) ; les règles sont
maintenant celles-ci, et `tests/unit/render/grade_chain_gpu_test.cpp` les vérifie sur les LUT livrées.

- **Blanc des surfaces : la moitié du pic de l'écran.** Il sortait vers 70 % du pic, si haut que les vrais
  éclats ne s'en détachaient plus. La compression vers l'écran garde sa courbe (celle sous laquelle la pile a
  été réglée à l'œil), mais son sommet est calculé pour poser le blanc d'une surface à la moitié du pic. Sur un
  écran peu lumineux, le blanc ne descend pas sous 300 nits, sans dépasser 70 % du pic. Le gris moyen ne
  bouge pas (RVB 128 reste vers 44 nits).
- **Éclats appliqués en dernier.** La passe anime décide toujours quels pixels sont des lumières et de
  combien elles montent, mais la montée n'est plus appliquée avant la pile, qui la divisait par deux : elle
  voyage dans le canal alpha et la dernière passe l'applique après la pile. Un émetteur blanc sort ainsi un
  demi-diaphragme au-dessus d'une surface blanche au réglage 100 % (0,2 avant), et le curseur Éclats agit
  sur toute sa course.
- **Limiteur de la scène.** Le plafond des éclats est le pic de l'écran en scène sombre ; il descend à la
  moitié du pic en scène très claire (luminance moyenne de 30 à 70 % du blanc SDR), sans passer sous 1,25 fois
  le blanc des surfaces. Les surfaces, elles, ne bougent pas avec la scène.
- **MaxFALL annoncé** : le blanc des surfaces réellement rendu (une image toute blanche), ou la moyenne
  plein écran du contrat d'écran quand il la connaît. 233 nits étaient annoncés pour un blanc rendu à 695.
- **Noirs.** Le noir pur reste un vrai noir : 0 sous RVB 4. Au-dessus, la LUT de base rendait presque noir
  tout ce qui était sous RVB 16 ; l'image suit maintenant un pied plus doux (pente 1,3 en log au lieu de 2),
  et la pile reprend toute sa part entre RVB 32 et RVB 48.
- **Sans pile** (fichier `.cube` illisible, ou les premières images avant sa lecture) : la dernière passe ne
  compresse plus une image que la pile n'a pas montée (A41-04). Le blanc reste à 233 nits au lieu de 119.

Mesuré sous lavapipe sur les shaders du lecteur (`ajp_render_tests "[.mesure]"`), source SDR au gamma 2,4,
écran de 1 000 nits, nits affichés :

| Code RVB 8 bits | Avant | Maintenant |
|---|---|---|
| 4 | 0 | 0 |
| 8 | 0,015 | 0,016 |
| 16 | 0,022 | 0,137 |
| 24 | 0,158 | 0,484 |
| 32 | 0,74 | 1,18 |
| 48 | 3,97 | 3,97 |
| 128 | 44,0 | 43,6 |
| 160 | 106 | 102 |
| 192 | 305 | 258 |
| 224 | 538 | 403 |
| 255 (blanc d'une surface) | 690 | 496 |
| Émetteur blanc, Éclats 100 %, scène sombre | 793 | 690 |
| Émetteur blanc, Éclats 100 %, scène très claire | 793 | 592 |
| Émetteur blanc, Éclats 200 %, scène sombre | 899 | 975 |

Blanc des surfaces selon le pic : 278 nits à 400 (320 avant), 496 à 1 000 (690), 992 à 2 000 (1 310).

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
