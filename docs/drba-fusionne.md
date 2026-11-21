# DistilDRBA fusionné (quatre moteurs TensorRT et noyaux CUDA)

Sous TensorRT, DistilDRBA v2 lite peut tourner coupé en quatre petits moteurs (l'encodeur, en lot de trois images, et les
trois blocs, des convolutions à basse résolution) reliés par des noyaux CUDA maison pour tout le reste : réductions
d'images, choix du côté de la paire selon tau, Resize, DepthToSpace, flot, grille, GridSample, Concat, masque sigmoïde,
mélange. Par défaut quand la découpe est valide ; le DistilDRBA entier reprend sinon. Réglage :
`interpolation.drba_fused` de `play.toml` (le menu ne le montre plus ; l'onglet Interpolation dit l'état du fusionné).

Prototype et banc : `tools/bench/fp8/drba_fusion/` (même méthode que `rife_fusion` de la branche
`worktree-agent-a68dd8d8bff2ff77e`). Ce document dit ce qui en est devenu produit. Les convolutions ne font que 28 % du
temps du réseau entier : le reste (casts et élémentaires fusionnés 25 %, GridSample 22 %, Resize 15 %, reformatages 8 %)
tient dans six noyaux.

## Fonctionnement

- `prepare/drba_split.h` reconnaît la structure du graphe (`drba_expected_wiring` : nom, opérateur, entrées exactes et
  constantes de chaque nœud de la colle ; échelles des Resize, GridSample aux coins alignés et bords répétés,
  DepthToSpace CRD de bloc 2, nombre des opérateurs) et refuse tout le reste. Réductions des blocs : 16, 8, 4 (graphe
  normal) ou 32, 16, 8 (flot à l'échelle 0,5, 4K).
- `derive_drba_fused` écrit dans le dossier `modeles` du lecteur quatre ONNX à forme fixe (par taille remplie) et un
  manifeste `.drbafused`, sous des noms qui portent la version de la dérivation, la taille et l'empreinte du modèle.
  Morceaux : l'encodeur en lot de 3 (poids partagés par les trois images), puis X_k → Y_k (52 canaux, avant DepthToSpace)
  pour chaque bloc.
- Le backend construit les quatre moteurs (trtexec, cache et timing cache habituels) et charge un `DrbaFusedEngine`
  (`BackendCaps::drba_fused`, ABI 13) qui enfile moteurs et noyaux sur le flux du backend, dans un seul graphe CUDA. Les
  tampons intermédiaires (≈ 41 Mo à 1088×1920) suivent les activations dans la zone partagée de la voie. L'interface est
  celle du moteur entier : `pair01`, `pair12` en entrées, `out` en sortie.
- L'entrée et la sortie de l'encodeur sont adressées dans l'entrée X_2 du bloc 2 (plans 0-8 et 9-56) : ces 57 canaux ne
  sont jamais copiés. Le côté choisi (image 0 ou 2) est lu sur le périphérique, dans le plan du temps de `pair01` : il
  change à chaque image.
- Tant que les trois images restent en place dans les entrées, les moteurs fusionnés peuvent reprendre leurs réductions
  et leurs encodages. Cela vaut aussi entre la seconde moitié de `(A, B)` et la première moitié de `(B, C)` : lorsque
  l'arrivée de `C` a repris les trois mêmes tampons sans réingestion du trio, seuls les plans du temps sont mis à jour.
- Le moteur fusionné ne lit l'instant que dans `pair01`. Sur un trio réutilisé, le backend n'écrit qu'une valeur
  FP16 de `pair01` au lieu des deux plans denses, puis distingue ce format dans la clé du graphe CUDA. Le test
  des réductions compare les deux chemins au bit près. Sur 478 passes en 1920×1088, la médiane est de 0,91 ms
  et le total de 400 ms, contre 0,92 ms et 404 ms après la seule reprise du trio (455 ms au témoin initial).
- Les arrondis FP16 du graphe sont reproduits aux mêmes endroits (sorties de Resize, sigmoïde, mélange) ; les Resize et
  GridSample du graphe sont en FP32 entre deux Cast. La grille des warps fait exception : la colle lit la position en
  pixels `x + flot` en FP32 (`pixel_tap`, `tap_of`), égale à la grille normalisée `base + flot × pas` du graphe ramenée en
  pixels, sans l'arrondi FP16 de la base ni de la somme (A46-04). Le graphe livré (`models-interp-1`, réexporté avec
  `DRBA_GRID_OPS`) garde lui aussi sa grille en FP32 depuis le 10 novembre 2026 (A03-18).
- Sur Ada en 1920×1088, le dernier bloc officiel est candidat au calcul CUDA natif. Ses huit résidus utilisent un
  Winograd vertical et la projection calcule les vingt canaux lus par le warp. Pour les autres instants d'un même
  trio, les sommes FP32 des 48 canaux fixes de la première convolution sont reprises dans le même ordre. Le
  chargement compare flot et masque à TensorRT, vérifie au bit près la reprise du cache et chronomètre les deux
  voies sous graphe CUDA. Un échec de ces contrôles garde le bloc TensorRT.
- Un essai sur la même forme groupe le rangement de X2 en HWC8 et les deux réductions dans une tuile
  partagée. Les trois sorties sont identiques au bit près. Le premier ABBA passe de 52,08 à 48,75 µs, puis
  le banc qui ordonne aussi les sentinelles sur le même flux CUDA mesure 63,66 à 64,20 µs. Le lecteur
  garde les deux opérations séparées.
- Le bloc 1 natif avec entrée et sortie planaires évite deux transpositions. Le banc 1088p donne
  186,42 µs contre 174,12 µs pour le même bloc CUDA avec deux transpositions et 123,69 µs sous TensorRT.
  Les deux voies CUDA donnent les mêmes codes FP16. Face à TensorRT, le SNR du masque varie de
  43,37 à 51,13 dB sur les trois mires. Le bloc reste au banc.
- Sur ce même bloc natif, les lectures planaires vectorisées gardent les codes FP16 et abaissent
  la colle de 41,83 à 31,93 µs et le bloc de 174,71 à 164,49 µs. Une géométrie réutilisant les
  fragments de poids sur deux groupes de pixels réduit aussi le corps natif de 117,72 à 100,01 µs
  dans son essai apparié. Le bloc TensorRT reste plus rapide que ces deux candidats.
- En lecture Wayland 1080p, une capture Nsight Systems à ×2 mesure 70 relectures du même graphe de
  51 nœuds, à 1,029 ms de médiane. La somme des durées des noyaux atteint 1,028 ms, avec 1,3 µs d'écart
  médian entre la durée du graphe et celle de ses noyaux. La colle d'entrée (0,158 ms) et le rangement
  de X2 (0,098 ms) comptent encore dans le coût de la passe de cette version de travail. La capture
  comprend l'essai de regroupement de X2.
- La préparation des entrées prend désormais une grille CUDA par résolution, avec la ligne et le plan
  dans les axes de la grille. À 1920×1088, les réductions et les plans du temps passent de 128,24 à
  15,68 µs en ABBA, ou de 13,63 à 3,20 µs pour les plans du temps seuls. Le banc vérifie les codes FP16
  écrits et les zones laissées intactes, avec temps scalaire ou dense, également à 3840×2176.
  En lecture à ×2, les médianes du moteur passent de 2,45–2,64 à 2,43–2,58 ms. Le p95 varie entre
  les passages. Wayland confirme environ 48 i/s dans les quatre essais en fenêtre.
- Après l'encodeur, les deux réductions de ses 48 plans utilisent elles aussi la ligne et le plan
  dans la grille CUDA pour les tailles usuelles jusqu'en 1080p. En ABBA, l'encodage de 480×272
  passe de 25,664 à 13,165 µs, celui de 320×192 de 12,618 à 8,592 µs. Les deux sorties et les
  sentinelles restent identiques. Les petits encodages et les grandes formes gardent le lancement
  unique, qui coûte moins au premier cas et reste compétitif au second.
- Un essai relit le petit halo des treize plans de contrôle de X2 en mémoire partagée.
  Le microbanc synthétique passe de 240,8 à 212,2 µs à 480×272 avec les mêmes bits.
  En lecture 1080p ×8, les médianes du moteur passent de 0,668 ms à 0,679–0,680 ms
  sur deux tours par voie. Le compositeur confirme 165,00 i/s sur les quatre passages.
  La variante reste au banc.
- Repli sur le DistilDRBA entier : réglage éteint, autre backend, cran Qualité avec son graphe `_fp32` (un cran dont
  le graphe manque garde la chaîne, et Performance la préfère au graphe `_fp8` : `drba_tier`), structure inconnue,
  taille refusée, échec de construction ou de chargement, essai raté. À chaque chargement la chaîne passe trois fois sur
  une paire de zéros (exécution à froid, capture du graphe, rejeu) avec synchronisation : une erreur CUDA se voit là, avant
  la première image. Une erreur plus tard en lecture n'est pas rattrapée (pas de bascule à chaud : reconstruire le moteur
  entier prendrait plus d'une minute).

## Tests

- Sans GPU : `tests/unit/prepare/drba_split_test.cpp` (reconnaissance sur un graphe fabriqué depuis le câblage attendu,
  pour les deux réductions, refus, manifeste, dérivation et cache, vrais modèles si `KIRARI_DRBA_MODEL_DIR`),
  `tests/unit/kernels/drba_fused_test.cpp`, `tests/unit/app/drba_fused_choice_test.cpp` (raisons de renoncement, réglage
  de `play.toml`, coût du changement).
- GPU : `kirari_backend_tensorrt_tests "[drba_fused]"` (noyaux de réduction contre l'hôte, bit à bit) ; test caché
  `"[.drba_fused]"` (chaîne contre le moteur entier de TensorRT, 384×640, PSNR ≥ 50 dB des deux côtés de l'image du milieu :
  54,97 dB à tau 0,5 et 59,09 dB à tau 1,25 sur un dessin synthétique à traits d'un pixel ; échelle 0,5 à 2176×3840 avec une
  vraie image : 65,7 et 71,5 dB — les traits d'un pixel du dessin synthétique bougent d'un pixel au moindre pas de la grille
  FP16 à cette taille, le test saute alors sans image `KIRARI_DRBA_IMAGE`).

## Mesures

Banc (`tools/bench/fp8/drba_fusion/resultats.txt`), RTX 4070 Ti, TensorRT 11.3, 1088×1920, graphe CUDA, vraies données :
chaîne 0,917 ms contre 2,099 ms pour le réseau entier (ABAB 3 tours : 0,9155 / 0,9175 / 0,9185 contre 2,098 / 2,099 /
2,099 ms), soit −56 %. Moteurs seuls : encodeur 0,106, blocs 0,085 / 0,125 / 0,379 ms ; noyaux : 0,152 ms au total (fin 0,085,
borne mémoire 0,075).

Qualité de `out` contre ORT CPU FP16 du réseau entier (3 images d'anime réelles, décalages de 6 et 20 px, tau 0,5, plus un
cas à tau 1,25 de l'autre côté) : 57,9 à 67,1 dB (SSIM 0,99986 à 0,99996), le TensorRT FP16 entier faisant 59,2 à 68,7 dB
(à 1,3 dB près de la chaîne). Le plancher de bruit du réseau était là : la grille de warp en FP16 (un ulp = 0,5 pixel à
1920 de large), deux réalisations du même graphe s'écartent de 60 à 67 dB (ORT enchaîné contre ORT entier : 66,8 dB).
Ces chiffres datent d'avant la grille en FP32 de la colle et sont à refaire sur la RTX. Le graphe livré, grille en FP32
(`DRBA_GRID_OPS`) : 77,5 dB de fidélité à son FP32 au banc de qualité, contre 57,0 dB pour le graphe d'avant
(`docs/banc-qualite.md`).

Dans le lecteur : voir la section suivante.

Une variante termine les huit résidus Winograd du bloc fin en relisant les lignes centrales du
halo partagé. Au banc, les huit sorties gardent les mêmes bits sur deux scènes réelles et les
médianes passent d'environ 114–115 à 110–111 µs. En lecture ×8 avec composition, le p95 du
moteur passe de 0,975–0,984 à 1,009–1,012 ms, et la cadence Wayland reste à 165,00 i/s. Le
candidat reste au banc.

Un recalage final sépare les deux axes du Resize et partage les sommes horizontales FP32 dans
la tuile. Les sorties image, flot et masque gardent tous les codes sur deux scènes à 1920×1088
et deux recadrages, à ×4 et ×8. En mode mouvement 1080p, les deux scènes passent d'environ
29–32 à 36–45 µs au banc. La tuile précédente reste en service.

Dans la colle de X2, huit fils par pixel peuvent relire les canaux fixes rangés en HWC8 et
écrire les canaux variables au même pas. En gardant la contraction FP32 de la référence, les
sorties, sentinelles et données d'entrée restent identiques. Le banc synthétique 1080p gagne
environ 34 %. Avec les vrais tenseurs de deux scènes 1080p, le témoin prend 24,8 à 38,8 µs et
la variante 93,3 à 107,1 µs. Elle reste au banc.

La lecture de ces mêmes échantillons par texture CUDA, au point et sans filtrage, conserve les
codes FP16 avec la même contraction. La texture linéaire ralentit de 16 à 25 % sur les deux
scènes. La texture 2D gagne 38 % sur la mire et ralentit de 7 à 66 % sur les scènes. Le noyau
du lecteur garde ses lectures globales.

Le banc groupé de la première convolution de X2 conserve les sommes FP32 fixes en registres
pour deux instants du même trio. Sur deux scènes, 3 133 440 codes par scène restent identiques.
La variante à tuile de 16 colonnes et poids partagés passe de 204–221 à 182–188 µs pour une
paire dont le premier instant est froid. Deux instants déjà chauds avec reprise du cache coûtent
142–148 µs. Le groupement reste au banc.

Avancer les mêmes sommes sur la voie auxiliaire garde toutes les sorties FP16 sur les deux
scènes. En mode mouvement, le p95 de la chaîne à froid passe de 1,017 à 1,049 ms et de 1,046
à 1,061 ms. En mode image, les écarts de médiane sont inférieurs à 10 µs. Le lecteur garde le
calcul des sommes dans la première convolution à froid.

## Dans le lecteur

`kirari play` (build Debug, le GPU fait le même travail), Re:Zero 18 de 16:00 (`--start=960`, 30 s), 1080p sans agrandissement
(`--no-upscale --work-height=1080`, DistilDRBA en 1920×1088), ×3, `--one-lane`, SDR. ABAB 3 tours, médiane « moteur » du
profil GPU (min entre parenthèses) :

| Tour | A fusionné | B entier |
|---|---|---|
| 1 | 0,97 ms (0,94) | 3,88 ms (2,99) |
| 2 | 0,96 ms (0,94) | 3,65 ms (2,99) |
| 3 | 0,96 ms (0,94) | 3,61 ms (2,99) |

Soit −73 % sur la médiane et −69 % sur le minimum (le profil du lecteur compte aussi l'enfilage du moteur entier, que le banc
ne compte pas : 2,10 ms en graphe CUDA). Environ 2500 passes par essai côté A, sans erreur. Construction des quatre moteurs :
38 s la première fois à 1088×1920 (le moteur entier : 115 s prévues).
