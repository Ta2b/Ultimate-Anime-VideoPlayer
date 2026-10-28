# DistilDRBA fusionné (quatre moteurs TensorRT et noyaux CUDA)

Sous TensorRT, DistilDRBA v2 lite peut tourner coupé en quatre petits moteurs (l'encodeur, en lot de trois images, et les
trois blocs, des convolutions à basse résolution) reliés par des noyaux CUDA maison pour tout le reste : réductions
d'images, choix du côté de la paire selon tau, Resize, DepthToSpace, flot, grille, GridSample, Concat, masque sigmoïde,
mélange. Par défaut quand la découpe est valide ; le DistilDRBA entier reprend sinon. Réglage : Réglages avancés >
Fluidité > « DistilDRBA fusionné » (`interpolation.drba_fused` de `play.toml`).

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
- Les arrondis FP16 du graphe sont reproduits aux mêmes endroits (sorties de Resize, grille FP16 = base + flot × pas,
  sigmoïde, mélange) ; les Resize et GridSample du graphe sont en FP32 entre deux Cast.
- Repli sur le DistilDRBA entier : réglage éteint, autre backend, cran de précision hors Équilibre, structure inconnue,
  taille refusée, échec de construction ou de chargement, essai raté. À chaque chargement la chaîne passe trois fois sur
  une paire de zéros (exécution à froid, capture du graphe, rejeu) avec synchronisation : une erreur CUDA se voit là, avant
  la première image. Une erreur plus tard en lecture n'est pas rattrapée (pas de bascule à chaud : reconstruire le moteur
  entier prendrait plus d'une minute).

## Tests

- Sans GPU : `tests/unit/prepare/drba_split_test.cpp` (reconnaissance sur un graphe fabriqué depuis le câblage attendu,
  pour les deux réductions, refus, manifeste, dérivation et cache, vrais modèles si `AJP_DRBA_MODEL_DIR`),
  `tests/unit/kernels/drba_fused_test.cpp`, `tests/unit/app/drba_fused_choice_test.cpp` (raisons de renoncement, réglage
  de `play.toml`, coût du changement).
- GPU : `ajp_backend_tensorrt_tests "[drba_fused]"` (noyaux de réduction contre l'hôte, bit à bit) ; test caché
  `"[.drba_fused]"` (chaîne contre le moteur entier de TensorRT, 384×640, PSNR ≥ 50 dB des deux côtés de l'image du milieu :
  54,97 dB à tau 0,5 et 59,09 dB à tau 1,25 sur un dessin synthétique à traits d'un pixel ; échelle 0,5 à 2176×3840 avec une
  vraie image : 65,7 et 71,5 dB — les traits d'un pixel du dessin synthétique bougent d'un pixel au moindre pas de la grille
  FP16 à cette taille, le test saute alors sans image `AJP_DRBA_IMAGE`).

## Mesures

Banc (`tools/bench/fp8/drba_fusion/resultats.txt`), RTX 4070 Ti, TensorRT 11.3, 1088×1920, graphe CUDA, vraies données :
chaîne 0,917 ms contre 2,099 ms pour le réseau entier (ABAB 3 tours : 0,9155 / 0,9175 / 0,9185 contre 2,098 / 2,099 /
2,099 ms), soit −56 %. Moteurs seuls : encodeur 0,106, blocs 0,085 / 0,125 / 0,379 ms ; noyaux : 0,152 ms au total (fin 0,085,
borne mémoire 0,075).

Qualité de `out` contre ORT CPU FP16 du réseau entier (3 images d'anime réelles, décalages de 6 et 20 px, tau 0,5, plus un
cas à tau 1,25 de l'autre côté) : 57,9 à 67,1 dB (SSIM 0,99986 à 0,99996), le TensorRT FP16 entier faisant 59,2 à 68,7 dB
(à 1,3 dB près de la chaîne). Le plancher de bruit du réseau est là : la grille de warp est en FP16 (un ulp = 0,5 pixel à
1920 de large), deux réalisations du même graphe s'écartent de 60 à 67 dB (ORT enchaîné contre ORT entier : 66,8 dB).

Dans le lecteur : voir la section suivante.

## Dans le lecteur

`ajp play` (build Debug, le GPU fait le même travail), Re:Zero 18 de 16:00 (`--start=960`, 30 s), 1080p sans agrandissement
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
