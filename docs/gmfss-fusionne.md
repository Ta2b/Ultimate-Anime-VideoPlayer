# GMFSS fusionné (la paire en deux moteurs, caractéristiques gardées)

Sous TensorRT, la paire de GMFSS Fortuna (demi-flot, cran Équilibre) peut tourner coupée en deux moteurs : les
caractéristiques d'une seule image (FeatureNet, 36 nœuds de convolutions) et tout le reste (flot de GMFlow dans les deux
sens, MetricNet, images réduites de moitié, 12 266 nœuds). FeatureNet ne lit ni l'autre image ni le flot, et ses deux côtés
sont le même réseau : les caractéristiques de l'image b d'une paire sont celles du côté a de la paire suivante. Le lecteur
les garde dans deux tranches (celle du tampon d'image qui porte l'image) et ne calcule qu'un FeatureNet par paire au lieu
de deux. **Éteint par défaut** : le gain est réel mais sous le seuil de 15 % qui fait un défaut. Réglage : Réglages
avancés > Fluidité > « GMFSS fusionné » (`interpolation.gmfss_fused` de `play.toml`).

## Ce qui a été essayé, et ce qui n'a rien donné

Mesures `trtexec` FP16 (niveau 5, graphe CUDA, `--noDataTransfers`), 1280×768, graphes demi-flot v2 du lecteur, sous le
verrou GPU (le bureau garde ~35 % du GPU, d'où l'écart médiane/minimum) :

| Essai | Résultat |
|---|---|
| Fusion : les 43 PReLU (Relu, Neg, Relu, Mul, Sub, à pente scalaire) réécrits en `LeakyRelu` | 10,23 / 10,19 / 10,21 ms contre 10,30 / 10,74 / 10,28 (minimums, 3 tours ABAB) : rien. TensorRT quitte Myelin (moteur de 17 Mo au lieu de 915 Mo) pour le même temps |
| Fusion sans les quatre `Slice` et `Concat` d'entrée (les splats écriraient directement dans les tenseurs concaténés) | médianes 11,33 à 11,88 ms contre 11,37 à 11,38 : rien, Myelin les absorbait déjà |
| Pair : 16 PReLU en `LeakyRelu` | −1,3 % (mesure de la branche `worktree-agent-a115b2ac88aee76b2`) |
| Paire coupée : FeatureNet seul / flot seul | voir ci-dessous |

Les éléments d'élémentaires (35 % du temps de la fusion) sont déjà à 85 % de la borne mémoire ; les convolutions font 80 %.
Un noyau CUDA maison ne les bat pas sans écrire des convolutions fusionnées avec leur épilogue (cela, ou le FP8 : étape 2,
non faite). Les splats sont déjà un noyau maison tuilé en mémoire partagée. La découpe en moteurs + noyaux de colle qui a
donné −56 à −70 % sur RIFE et DistilDRBA n'a donc pas d'équivalent ici : GMFSS n'a pas d'opérateur en pleine résolution que
TensorRT traite mal. Ce qui reste, c'est du calcul répété : FeatureNet.

## Fonctionnement

- `prepare/gmfss_split.h` reconnaît la paire (entrée « frames » 1 × 6 × H × W FP16, deux `Slice` aux plans 0-2 et 3-5, sorties
  d'un export version 2, **mêmes poids et mêmes constantes aux deux côtés de FeatureNet**, comparés nœud par nœud) et refuse
  tout le reste. `derive_gmfss_fused` écrit dans le dossier `modeles` deux ONNX à forme fixe (par taille remplie), sous des
  noms qui portent la version de la dérivation, la taille et l'empreinte du modèle : « features » (une image, 3 sorties
  `feature0_1`, `source0_2`, `source0_3`) et « flow » (« frames » → `motion{k}_{l}`, `half{k}`).
- Aucun nouveau moteur du backend : ce sont deux moteurs TensorRT ordinaires (`prepare_engine`, trtexec, cache et timing cache
  habituels). `BackendCaps::gmfss_fused` (ABI 15, vrai pour TensorRT) dit que le backend les enfile avec des tenseurs décalés
  dans leur tampon (l'entrée des caractéristiques est une tranche de « frames », plans 0-2 ou 3-5).
- L'étage (`upscale_gmfss.cpp`) tient deux jeux de caractéristiques, un par tampon d'image RGB (`current_`). `GmfssFeatureSlots`
  (`gmfss_features.h`) retient de quelle image (identité + époque) chaque tranche est ; une paire calcule seulement celles qui
  manquent. Les splats lisent la tranche de leur côté ; selon la parité, deux listes de splats (côtés échangés). Après une
  coupe de scène, un dessin tenu, un seek ou une image sautée, la tranche n'est pas reconnue et l'image est refaite.
- Repli sur la paire entière : réglage éteint, autre backend, cran de précision hors Équilibre, structure inconnue (export
  version 1, autre graphe), taille refusée, échec de dérivation, de construction, de chargement ou d'essai. À chaque
  chargement les deux moteurs passent trois fois sur des tenseurs quelconques aux formes de la lecture (cold, capture du
  graphe, rejeu), l'entrée des caractéristiques à son décalage de plans. Une erreur CUDA asynchrone plus tard en lecture n'est
  pas rattrapée. La fusion n'est pas touchée (GMFSS Union a la même paire : essayé aussi, la reconnaissance tranche ; sa fusion
  reste la sienne).

## Tests

- Sans GPU : `tests/unit/prepare/gmfss_split_test.cpp` (reconnaissance sur un graphe copiant la structure, refus : plage des
  Slice, export v1, poids différents, nœud en trop, entrée mal nommée/mal formée ; dérivation et cache ; vraie paire demi-flot si
  `AJP_GMFSS_MODEL_DIR`), `tests/unit/app/gmfss_fused_choice_test.cpp` (raisons de renoncement, `play.toml`, coût du changement,
  séquences de tranches de caractéristiques : reprise, saut, époque).
- GPU : test caché `"[.gmfss_fused]"` (`AJP_GMFSS_MODEL_DIR`, `AJP_GMFSS_IMAGES` : dix PPM 640×384 de la vidéo d'essai, cinq
  images consécutives à 960 s et 1010 s). Chaîne entière contre chaîne coupée, rejouée comme la lecture ((i0,i1) puis (i1,i2),
  la seconde reprend i1), pas d'une et de deux images, t = 0,25 / 0,5 / 0,75 : 24 comparaisons, **PSNR 73,4 à 75,1 dB, SSIM
  0,999992 à 0,999993** (la paire entière contre elle-même : 76,6 dB, le bruit des additions atomiques).

## Mesures

RTX 4070 Ti, TensorRT 11.3. Banc `trtexec`, 1280×768, ABAB 3 tours (minimum / médiane en ms) :

| Tour | paire entière | FeatureNet seul | flot seul |
|---|---|---|---|
| 1 | 8,05 / 8,47 | 0,950 / 1,024 | 5,85 / 6,24 |
| 2 | 8,08 / 8,48 | 0,952 / 1,020 | 5,85 / 6,78 |
| 3 | 8,55 / 9,28 | 0,954 / 1,037 | 5,71 / 6,96 |

Une paire reprenant un FeatureNet : 0,95 + 5,85 = 6,80 ms contre 8,05 (−15,5 % sur le minimum), −8 à −14 % sur la médiane
(bruit des autres usagers du GPU).

Dans le lecteur (`ajp play`, build Debug, Re:Zero 18 de 16:00, `--start=960`, 30 s, 1080p, `--work-height=720` (moteurs en
1280×768), `--rife=N` fixe, `--sdr`, `--one-lane`, dossier de données à part, sous `ajp-gpu`), ABAB 3 tours, somme des profils GPU
des moteurs GMFSS (paire ou flot + caractéristiques + fusion + splats) rapportée au nombre de paires :

| Facteur | entier (médiane / minimum) | fusionné (médiane / minimum) |
|---|---|---|
| ×2 | 33,4 / 24,9 ; 33,1 / 24,4 ; 32,3 / 23,6 | 30,8 / 23,2 ; 31,6 / 25,0 ; 30,9 / 23,0 |
| ×3 | 35,7 / 25,4 ; 35,5 / 25,1 ; 33,6 / 24,2 | 24,3 / 22,1 ; 33,2 / 23,5 ; 30,2 / 21,8 |

Soit environ −6 % (×2) à −9 % (×3, très bruité : le nombre de paires par essai varie de 340 à 650) du temps GMFSS d'une paire.
Le moteur de la paire seul : médiane 9,9-10,5 ms contre 7,4 ms (flot) + 1,35 ms (FeatureNet) × 1,10 (FeatureNet refait après
une coupe ou un dessin tenu : 590 reprises sur 718 images neuves à ×2). Les essais de la première série à 540 lignes de travail
(1024×640) donnent la même chose (7,3 ms contre 5,4 + 0,5) mais sont écartés (moins de 720 lignes).

## Pourquoi éteint, et la suite

Seuil : ≥ −15 % du temps GMFSS par paire **et** ≥ 50 dB. La qualité est là (73 dB et plus). Le temps de la paire entière
baisse de 15 % au banc, 10 à 13 % dans le lecteur ; celui de toute la chaîne d'une paire (paire, fusion, splats) de 6 à 9 %.
Le réglage est prêt à être allumé par défaut (`PlaySettings::gmfss_fused`) si l'on juge le seuil sur la paire seule.

Pistes non faites : calculer les caractéristiques de l'image qui arrive sur la voie de mesure, en avance, pour les cacher
derrière la fusion de la paire d'avant (gain de latence, pas de coût GPU) ; une convolution fusionnée avec son épilogue pour la
fusion (GridNet : 10,5 ms, 80 % de convolutions) ; l'étape 2 (FP8 du cran Performance), non entreprise : l'étape 1 ne justifie
pas ce chantier, et la fusion, où est le temps, est un réseau de convolutions que le FP8 ne rendrait pas à 42 dB sans
calibration propre.
