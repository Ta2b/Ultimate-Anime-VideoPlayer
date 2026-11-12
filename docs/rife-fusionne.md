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
- Sur le backend Vulkan (appareil à stockage 16 bits, sans runtime de fabricant), les six morceaux deviennent des
  artefacts `.ajpvk` et `gpu/vulkan/backend/rife_fused_engine.h` les relie par les mêmes noyaux portés en GLSL
  (`kernels/glsl/rife_fused_*.comp`), enregistrés une fois puis rejoués. Tests : noyaux contre la référence CPU
  (`tests/unit/vk_backend/rife_fused_test.cpp`), et en test caché `[.rife_fused]` la chaîne contre le RIFE entier
  (4.22 lite en 128×64 sur lavapipe : 74,6 dB).
  Les tampons intermédiaires (≈ 120 Mo à 768×1280) suivent les activations dans la zone partagée de la voie.
- Deux sorties : l'image (RIFE image) ou le flot et le masque (variante mouvement, le chemin par défaut du lecteur).
- Grille des warps (A03-20) : la colle, CUDA comme Vulkan, ne lit plus le maillage FP16 des plans 7 à 10. Elle prend la
  position en pixels `x + flot` en FP32 (`pixel_tap`), ce que vaut la grille normalisée du graphe une fois ramenée en
  pixels, sans l'arrondi FP16 de la base (jusqu'à 0,234 pixel à 1920 de large). Mesuré sur le graphe 4.22 lite en ONNX
  Runtime, la grille seule changée : un plan fixe sort à 78 à 86 dB de sa source au lieu de 51 à 60, un panoramique de
  6 pixels gagne 2 à 5 dB. Le RIFE entier v1 lit toujours ses plans 7 à 10 : seul le format v2 (grille calculée dans le
  graphe, en FP32) le corrige.
- Entrée par pointeurs (étape 5.2, backend TensorRT) : les noyaux CUDA lisent l'instant à l'élément 0 du plan 6.
  Quand le dernier moteur à lire l'entrée est la chaîne fusionnée, `rife_time` n'écrit plus qu'un demi-flottant au lieu
  d'un plan entier ; un moteur TensorRT entier ou DistilDRBA fusionné qui lit la même adresse ramène l'écriture du plan
  entier. Les plans 0 à 5 restent copiés (l'encodeur les veut remplis au multiple de 64). Le chemin Vulkan ne change pas.
- Encodage gardé (fusion3, novembre 2026) : l'encodeur prend une image à la fois (morceau `encode` en
  1 × 3 × H × W, dérivation `fusion3` : les morceaux `fusion1` et `fusion2`, encodeur en lot de 2, sont refaits et l'ancien moteur
  d'encodage refusé au chargement). L'encodage de la paire (4 plans pour A, 4 pour B) reste dans les tampons du
  moteur d'un appel à l'autre. Le lecteur (`UpscaleStage::enqueue_rife`) dit au moteur ce que sa passe précédente a
  déjà encodé, d'après le numéro d'ingestion de chaque image : un autre instant des mêmes images dans le même tampon
  (`Engine::next_times_only` : l'encodeur est sauté), ou la paire qui suit, A étant la B de la passe précédente
  (`Engine::next_pair_follows` : l'encodage de B est copié à la place de A, seule la nouvelle B est encodée). Rien
  n'est annoncé après une coupe ou un saut (`forget_motion`), pour une B lissée (« dessins lissés »), quand un autre
  moteur a fait la passe d'avant (variante des paires calmes) ou quand elle a échoué. Le moteur vérifie de son côté
  que son dernier appel est allé au bout et, sous TensorRT, que la zone d'activation de la voie est la même et
  qu'aucun autre moteur n'y a écrit depuis (`SharedActivations::last_user`) ; sinon il encode les deux images.
  Encodages par paire : 2 au lieu de 2 × instants à froid, 1 en régime (paire qui suit). La sortie ne change pas d'un
  bit (tests ci-dessous). Le journal de fin compte les passes : « RIFE : encodage repris sur N passes, B seule sur M,
  A et B sur G après une paire sans passe et F à froid » (G : la paire d'avant était un dessin tenu ou sans
  instant ; F : début, coupe, saut, autre moteur, B lissée, ou `--rife-encode-all`).
  `--debug --rife-encode-all` coupe les annonces, pour mesurer.
- Repli sur le RIFE entier : réglage éteint, autre backend, graphe `_fp8` ou `_fp32` chargé (un cran dont le graphe manque garde la chaîne), entrée v2 (7 plans),
  structure inconnue, taille refusée, échec de construction ou de chargement, essai raté. À chaque chargement la chaîne
  passe trois fois sur une image de zéros (exécution à froid, capture du graphe, rejeu) avec synchronisation : une
  erreur CUDA se voit là, avant la première image. Une erreur plus tard en lecture n'est pas rattrapée (pas de bascule
  à chaud : reconstruire le moteur entier prendrait plus d'une minute).

## Tests

- Sans GPU : `tests/unit/prepare/rife_split_test.cpp` (extraction, reconnaissance sur un graphe copiant la structure,
  refus, manifeste, dérivation et cache, vrais modèles si `AJP_RIFE_MODEL_DIR`), `tests/unit/kernels/rife_fused_test.cpp`,
  `tests/unit/app/rife_fused_choice_test.cpp`.
- Encodage gardé, bit à bit contre la chaîne qui encode tout : `ajp_vk_backend_tests "[.rife_fused]"` (Vulkan,
  lavapipe compris, `rife_fused_chain_test.cpp`) et `ajp_backend_tensorrt_tests "[.rife_fused]"`
  (`rife_fused_encode_test.cpp`, parcours rejoué trois fois pour passer par la capture et le rejeu du graphe CUDA,
  puis un autre moteur fusionné dans la même zone entre deux appels). Les annonces du lecteur :
  `tests/unit/app/upscale_stage_passes_test.cpp`.
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

### Encodage gardé : protocole ABAB (à faire sur la RTX)

Rien n'est encore mesuré sur carte : Vulkan logiciel ne dit rien du temps. Le gain attendu est la part de
l'encodeur dans la passe : une image encodée au lieu de deux par paire en ×2, et au lieu de 2 × (facteur − 1) en ×3
et plus. Binaire livré (`linux-release`, `windows-release`), même poste, même séance, rien d'autre d'ouvert.

1. Première ouverture avec ce code : les six morceaux `fusion3` se dérivent et leurs moteurs se construisent (environ
   une minute). Faire une lecture de chauffe de 30 s qui ne compte pas.
2. Conditions des mesures ci-dessus : Re:Zero 18 (`--start=960`, 30 s), 1080p, travail 540, variante mouvement,
   `--one-lane`, SPAN Performance, SDR. Trois facteurs : `--rife=2`, `--rife=3`, puis `--rife=5` (24 → 120 i/s, la
   cadence que le poste tient dans les scènes qui bougent le plus : quatre instants par paire).
3. A = `ajp --debug play …` (encodage gardé), B = `ajp --debug play --rife-encode-all …`. Ordre A, B, A, B, A, B
   (trois tours), sans changer autre chose entre deux lectures.
4. Relever pour chaque lecture la médiane et le minimum de la ligne « moteur » du profil GPU (le RIFE fusionné),
   les images en retard et manquées, et la ligne « RIFE : encodage repris… ». En A, en ×2, « B seule » doit couvrir
   presque toutes les passes, en ×3 une passe sur deux est sans encodeur, en ×5 quatre sur cinq ; G dit combien de
   fois l'encodeur reprend les deux images après un dessin tenu. En B, tout est « à froid ».
5. Gardé si A est plus rapide que B sur les trois tours aux trois facteurs. L'image est la même par construction ; un
   coup d'œil côte à côte sur une capture (`--snapshot`) au même instant suffit à confirmer que rien n'a bougé.

6. Le cas qui compte sur le poste de référence (écran 165 Hz, aujourd'hui 100 à 120 i/s dans les scènes qui bougent
   le plus, le GPU en limite) : la même scène d'action, réglages habituels de lecture (mode Auto, sans
   `--one-lane`), ABAB 3 tours, et relever les images par seconde réellement produites (compteur à l'écran ou
   journal de fin) en plus du temps « moteur ». La cible est d'environ 7 instants par paire depuis du 23,976.

Reporter les lectures de chaque facteur dans un tableau comme celui du dessus, et pour le point 6 les i/s atteintes
en A et en B.
