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
