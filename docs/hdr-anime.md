# Méthode « HDR anime » (--hdr-anime) : les essais du méga plan

La méthode garde les surfaces à un relèvement fixe et ne fait monter que les émetteurs
(`src/render/shaders/anime-hdr-anime/3-anime.hook`). Chaque étape ci-dessous est un essai en ligne de
commande ; sans son drapeau, rien ne change.

| Étape | Drapeau | Ce qui change | État |
| --- | --- | --- | --- |
| A1 | `--hdr-anime` | La carte des émetteurs, portes tirées des gains réels de `decide_grade` | fait |
| A2 | `--anime-shots` | Exposition des surfaces et marge des éclats fixées par plan (pré-analyse), glissées en 250 ms aux coupes (`app/play/anime_shots.h`) | codé, à mesurer |
| A3 | `--anime-guided` | Carte locale lue par un sur-échantillonnage bilatéral joint guidé par le pixel : le halo ne déborde plus | codé, à mesurer |
| A4 | aucun | Visages : décodage d'un détecteur, doublons, suivi, palette de peau, exclusion des émetteurs (`render/faces.h`) | partie sans réseau codée |
| A6 | aucun | Réseau des émetteurs : étiquettes (port numpy de A5), jeu de données, entraînement et export ONNX (`tools/grade/emitters/`) | scripts écrits, entraînement plus tard |
| A5 | aucun | Mesures d'émetteurs v2, référence CPU (`render/emitters.h`) : pixel, bloc, couronne ; néon admis | codé, juge et étiquettes |

## A2 : décisions par plan
L'exposition suit la règle de la courbe par plan (`render/curve.h`) : 0,25 diaphragme par diaphragme
d'APL du plan rapportée à la médiane de la série, bornée à ±0,25. Sans plan analysé, le relèvement reste
fixe (1,15).

## A3 : écarts au plan
Le lissage temporel existait déjà sur la carte de zone (`2-area.hook`). Les cartes transportées dans
l'anneau sont celles du grade actuel (`--grade-ring`) ; la méthode anime se fait à l'écran.

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

## Noirs profonds selon la scène (04/11)

Sous la méthode HDR anime, le curseur « Noirs profonds » est le maximum, atteint en scène très claire
(luminance moyenne ≥ 40 % du blanc SDR) ; l'effet descend en douceur jusqu'à 0 en scène très sombre (≤ 3 %),
pour garder lisibles les détails des nuits. Sans mesure de la scène, le curseur s'applique tel quel.
