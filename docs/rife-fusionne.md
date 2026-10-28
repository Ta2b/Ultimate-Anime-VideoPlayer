# RIFE fusionné (six moteurs TensorRT et noyaux CUDA)

Sous TensorRT, RIFE 4.25 et 4.22 lite peuvent tourner coupés en six petits moteurs (l'encodeur et les blocs, des
convolutions à basse résolution) reliés par des noyaux CUDA maison pour tout ce qui travaille en pleine résolution
(DepthToSpace, Resize, flot, grille, GridSample, Concat, mélange à masque). Par défaut quand la découpe est valide ;
le RIFE entier reprend sinon. Réglage : Réglages avancés > Fluidité > « RIFE fusionné » (`interpolation.rife_fused`
de `play.toml`).

Prototype et banc : `tools/bench/fp8/rife_fusion/` de la branche `worktree-agent-a68dd8d8bff2ff77e` (768×1280 fixe,
4.25 seul : 2,20 ms contre 6,46 ms pour le réseau entier). Ce document dit ce qui en est devenu produit.

## Fonctionnement

- `prepare/rife_split.h` reconnaît la structure du graphe (noms des tenseurs aux frontières, opérateurs, échelles des
  Resize, constantes des Mul et des Split, attributs de Resize, GridSample, DepthToSpace) et refuse tout le reste.
  Bloc k : entrée X_k réduite de d_k (8, 8, 4, 2, 1 pour 4.25 ; 8, 4, 2, 1 pour 4.22 lite), sortie Y_k à 52 canaux.
- `derive_rife_fused` écrit dans le dossier `modeles` du lecteur six ONNX à forme fixe (par taille remplie) et un
  manifeste `.rifefused`, sous des noms qui portent la version de la dérivation, la taille et l'empreinte du modèle.
- Le backend construit les six moteurs (trtexec, cache et timing cache habituels) et charge un `RifeFusedEngine`
  (`BackendCaps::rife_fused`, ABI 12) qui enfile moteurs et noyaux sur le flux du backend, dans un seul graphe CUDA.
  Les tampons intermédiaires (≈ 120 Mo à 768×1280) suivent les activations dans la zone partagée de la voie.
- Deux sorties : l'image (RIFE image) ou le flot et le masque (variante mouvement, le chemin par défaut du lecteur).
- Repli sur le RIFE entier : réglage éteint, autre backend, cran de précision hors Équilibre, entrée v2 (7 plans),
  structure inconnue, taille refusée, échec de construction ou de chargement, essai raté. À chaque chargement la chaîne
  passe trois fois sur une image de zéros (exécution à froid, capture du graphe, rejeu) avec synchronisation : une
  erreur CUDA se voit là, avant la première image. Une erreur plus tard en lecture n'est pas rattrapée (pas de bascule
  à chaud : reconstruire le moteur entier prendrait plus d'une minute).

## Tests

- Sans GPU : `tests/unit/prepare/rife_split_test.cpp` (extraction, reconnaissance sur un graphe copiant la structure,
  refus, manifeste, dérivation et cache, vrais modèles si `AJP_RIFE_MODEL_DIR`), `tests/unit/kernels/rife_fused_test.cpp`,
  `tests/unit/app/rife_fused_choice_test.cpp`.
- GPU : `ajp_backend_tensorrt_tests "[rife_fused]"` (noyaux contre la référence CPU en double, 65 à 83 dB) ; test caché
  `"[.rife_fused]"` (chaîne contre le moteur entier de TensorRT, 384×640, PSNR ≥ 50 dB : 4.25 image 53,2 dB, 4.22 lite
  image 64,0 dB, mouvement 54 à 64 dB).

## Mesures

RTX 4070 Ti, TensorRT 11.3, `ajp play` (build optimisé) sur Re:Zero 18 de 16:00 (`--start=960`, 30 s), 1080p, travail
540 (RIFE en 960×576), ×3 (`--rife=3`), variante mouvement, `--one-lane` (coût propre du moteur), SPAN Performance, SDR.
ABAB 3 tours (A = fusionné, B = RIFE entier ; médiane « moteur » du profil GPU, min entre parenthèses) :

| Tour | A fusionné | B entier |
|---|---|---|
| 1 | 1,32 ms (1,24) | 4,54 ms (3,23) |
| 2 | 1,43 ms (1,19) | 4,53 ms (3,23) |
| 3 | 1,69 ms (1,24) | 4,54 ms (3,23) |

Soit −63 à −71 % sur la médiane (−62 % sur le minimum). Le médian de A varie avec les horloges (24 i/s : le GPU
dort). Images en retard sur l'horloge : A 3, 2, 2 ; B 4, 4, 4 ; manquées 23 à 25 des deux côtés ; temps par image
médian 6,1 à 6,8 ms des deux côtés. `--rife-image` (variante image) : 1,23 ms. Le prototype, à 768×1280 : 2,20 ms
contre 6,46 ms. Construction des six moteurs : 47 s la première fois à 960×576 (le moteur entier : plus d'une minute).
