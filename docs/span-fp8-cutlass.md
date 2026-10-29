# SPAN entier en FP8 par noyaux CUTLASS (cran Performance)

Le cran **Performance** de la précision de SPAN peut calculer le réseau entier par des noyaux CUTLASS en FP8
(E4M3) à la place du moteur TensorRT FP16, sur une RTX 40 (sm 8.9). Jamais par défaut : le cran Équilibre
(TensorRT FP16) reste le réseau tel que livré, et tout ce qui manque ou échoue ramène le lecteur dessus.

Prototype et banc : `tools/bench/fp8/cutlass/` de la branche `worktree-agent-a2f39568384b17d35` (`span_net.cu`,
`span_net_kernels.h`, `span_net.py`, `resultats_net_*.txt`). Ce document dit ce qui en est devenu produit.

## Mesures

RTX 4070 Ti (sm 8.9), SPAN Balanced HD V3.1 (`2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx`),
TensorRT 11.3, CUDA 13, ABAB sous le verrou GPU.

| Mesure | TensorRT FP16 | Noyaux FP8 | Écart |
|---|---|---|---|
| Prototype, 1×3×1088×1920, graphe CUDA, séance du 28/10 | 14,25 ms | 9,34 ms (acc. f32) ; 9,09 ms (acc. f16 hors blocs) | −34 à −36 % |
| Lecteur, 1×3×1080×1920, 3 tours ABAB (`ajp_span_fp8_run --bench` contre `trtexec --loadEngine` du moteur du lecteur) | 14,70 / 14,74 / 14,74 ms | 9,22 / 9,22 / 9,21 ms | **−37 %** |
| Lecteur, `ajp play --stats`, 24 i/s, 1080p → 2160p, SPAN seul, médiane « moteur » | 15,53 ms (min 14,3) | 14,5 ms (min 8,9) | −6,5 % (voir ci-dessous) |

Pourquoi la médiane en lecture baisse si peu : à 24 i/s le GPU dort les deux tiers du temps et le pilote baisse
ses horloges. Échantillonnées toutes les 500 ms pendant 30 s de lecture : **2825 MHz et 149 W avec TensorRT FP16,
1790 MHz et 72 W avec les noyaux FP8** (même utilisation). Le temps d'une image en lecture est donc celui d'un
GPU à demi-régime ; le gain tient dans la puissance (moitié moins) et la marge sur les contenus à 60 i/s ou les
chaînes lourdes (RIFE, RTX HDR sur la même carte), pas dans la médiane d'une lecture 24 i/s.

Qualité (image finale ×2, FP16 contre FP8) :

| Contre | PSNR | SSIM | Images |
|---|---|---|---|
| Prototype, SPAN d'origine en FP32 (ONNX Runtime CPU) | 49,8 dB en moyenne (48,9 au pire) | 0,9962 | 3, calibration sur 5 autres images |
| Reçu de `derive_span_fp8.py`, SPAN d'origine en FP32 | 52,0 dB en moyenne (49,3 au pire) | 0,9967 au pire | 4 de validation 1080×1920, calibration sur 32 découpes de 576² de 40 autres |
| Lecteur (moteur chargé par le backend), contre le moteur TensorRT FP16, une image réelle 1080×1920 | 49,15 dB | 0,9966 (SSIM par blocs) | test caché `[.span_fp8]` |

## Architecture

### Ce que les noyaux calculent

Entrée 1×3×H×W FP16 (NCHW), sortie 1×3×2H×2W FP16, comme le SPAN en ONNX :

1. dégroupage 2×2 (3 → 12 canaux, complétés à 16) ;
2. `conv_1` 3×3 12→64, FP16 ; sa sortie sort en FP16 (tranche 0 de la concaténation) et en FP8 (entrée du bloc 1) ;
3. 8 blocs SPAB, en FP8 : `c1`+Relu, `c2`+Relu, puis une fin de bloc fusionnée (`c3`, convolution 1×1 d'attention,
   tanh, saut, produit, requantification E4M3) ;
4. `Concat` de 4 tenseurs FP16 (sortie de `conv_1`, sortie du bloc 8, sortie du bloc 1, premier Relu du bloc 8),
   écrits directement par les noyaux dans leur tranche d'un tampon de 256 canaux ;
5. `conv_cat` 1×1 256→64, `to_pixel` 3×3 64→48 (FP16), DepthToSpace(4, mode CRD).

Une boucle principale CUTLASS (convolution implicit GEMM, `mma.sync` m16n8k32 E4M3) et un épilogue écrit à la
main (`span_fp8_kernels.cuh`) : biais, Relu, requantification, tanh. 29 lancements par image, sans allocation ni
synchronisation : ils se capturent dans un graphe CUDA tels quels.

### Choix : (B) exécuteur direct dans le backend, pas (A) plugin TensorRT

| | (A) plugin `IPluginV3` | (B) moteur `gpu::Engine` du backend (retenu) |
|---|---|---|
| Où | un nœud « tout le réseau » dans un moteur TensorRT | une seconde sorte d'`Engine` à côté de `TrtEngine` |
| Taille | profil d'optimisation, formes dynamiques à déclarer | fixée à la construction, comme les moteurs actuels |
| Construction | `trtexec` doit connaître le plugin : bibliothèque du plugin à charger dans le processus de `trtexec` (un exécutable livré par NVIDIA), ONNX à greffer d'un nœud personnalisé, sérialisation | aucun : lecture et contrôle du paquet |
| Espace de travail | `getWorkspaceSize` | la zone d'activation de la voie (`SharedActivations`), comme TensorRT |
| Risque | un échec de plugin peut casser la construction du moteur de repli | un refus se voit à `build()` ou `load()` ; le moteur FP16 n'est pas touché |
| Graphes CUDA | capturés par `TrtEngine` | lancements directs sur le flux du backend, capturables |

La chaîne ne passe que par `Backend::build / load / enqueue` : le moteur FP8 se glisse derrière sans toucher à
l'appelant. (A) ajoutait un chargement de bibliothèque dans un processus étranger et un greffon dans l'ONNX pour
le même résultat ; (B) garde le plus simple, et la seule chose à écrire pour l'appelant est de choisir la variante.

### Où est le code

| Fichier | Rôle |
|---|---|
| `src/gpu/span_fp8/package.{h,cpp}` | format du paquet, lecture, vérifications, tailles prises (C++ pur, testé sans GPU) |
| `src/gpu/cuda/backend/span_fp8_kernels.cuh` | noyaux CUTLASS et épilogue (repris du banc) |
| `src/gpu/cuda/backend/span_fp8_net.{h,cu}` | le réseau : poids FP8 sur l'appareil, zone de travail, lancements. Sans `std::expected` ni `<format>` dans l'interface : nvcc ne les lit pas |
| `src/gpu/cuda/backend/span_fp8_net_none.cpp` | version vide quand CUTLASS n'est pas construit |
| `src/gpu/cuda/backend/span_fp8_engine.{h,cpp}` | `SpanFp8Engine : gpu::Engine`, clé d'artefact, essai au chargement |
| `src/gpu/cuda/backend/tensorrt_span_fp8.cpp` | `TensorRtBackend::build / load / enqueue` pour un paquet |
| `src/gpu/cuda/backend/span_fp8_run.cpp` | `ajp_span_fp8_run`, l'outil de contrôle de `derive_span_fp8.py` |
| `src/app/play/span_int8.{h,cpp}` | variante `-fp8c-` : reçu, garde-fous |
| `src/app/play/chain_setup.cpp` | l'ordre des variantes et le repli |
| `tools/span/derive_span_fp8.py` | calibration, paquet, reçu |
| `third_party/cutlass/` | CUTLASS 4.8.1, BSD-3-Clause, en-têtes seulement (259 fichiers, avec `LICENSE.txt`) |

### Zone de travail

768 octets par pixel de caractéristiques (400 Mo à 1080p) : la concaténation FP16 de 256 canaux (512 o) et quatre
tampons FP8 de 64 canaux (256 o). Le dégroupage, la sortie de `conv_cat` et celle de `to_pixel` recouvrent des
tampons FP8 alors inactifs. La zone est celle de la voie (`SharedActivations::reserve`) : elle sert aussi aux
moteurs TensorRT de la voie, qui s'exécutent l'un après l'autre.

## Le paquet « .span8 »

Ce que `parse` accepte (`src/gpu/span_fp8/package.h`), petit-boutiste :

```
en-tête, 40 octets : "AJPSPAN8" · version u32 (1) · blocs u32 (8) · canaux u32 (64) · réservé u32 (0)
                     · taille de la charge u64 · CRC-32 de la charge u32 · réservé u32 (0)
charge :  échelles f32 × 25      [0] sortie de conv_1 ; bloc b (0..7) : [1+3b] relu1, [2+3b] relu2, [3+3b] sortie
          conv_1   : poids f16 [64][3][3][16] (4 canaux nuls), biais f32 [64]
          bloc 1..8 : c1, c2, c3 : poids f16 [64][3][3][64], biais f32 [64] ;
                      attention : poids f16 [64][64], biais f32 [64]
          conv_cat : poids f16 [64][256], biais f32 [64]
          to_pixel : poids f16 [48][3][3][64], biais f32 [48]
```

Les poids sont ceux du modèle, en FP16 : le paquet ne contient rien que le modèle n'ait déjà. Les convolutions de
bloc passent en E4M3 au chargement, à l'échelle de leur tenseur (maximum / 448), arrondi au plus proche, saturé.
Les échelles d'activation sont « maximum observé / 448 » sur le corpus de calibration (le percentile 99,99 donne
1,1 dB de moins sur le prototype : maximum retenu). Un paquet tronqué, un bit changé, une version inconnue, une
échelle nulle ou un poids infini sont refusés.

Le paquet est dérivé des poids d'AnimeJaNai (CC BY-NC-SA 4.0) : il se construit chez l'utilisateur, ne se publie
pas, et vit dans le dossier « modeles » des données du lecteur à côté des variantes INT8 et FP8.

## Fabriquer la variante

```
~/.animejanai-tools/venv-gpu/bin/python3 -I tools/span/derive_span_fp8.py MODELE.onnx \
    --corpus images/ --out ~/.local/share/ajp/ajp/modeles \
    --runner build/linux-prefix-cuda/src/gpu/cuda/backend/ajp_span_fp8_run \
    --gpu-prefix ~/ajp-tools/bin/ajp-gpu --trtexec ~/ajp-sdk/tensorrt/bin/trtexec
```

Le corpus est celui de `quantize_span.py` (images d'anime SDR 8 bits, aucune sortie RTX HDR/TrueHDR/RTX Video ;
un cinquième des images, choisies par SHA-256, valide et ne calibre jamais). Le script : contrôle le graphe
(refuse le b5f48 à 48 canaux avant tout calcul), calibre en FP32 sur ONNX Runtime CPU, écrit le paquet, fait
tourner les noyaux du lecteur sur les images de validation (`ajp_span_fp8_run`) et les compare au SPAN d'origine
en FP32, mesure (avec `--trtexec`) le gain sur TensorRT FP16 et exige 1,05. Il écrit
`<nom>-fp8c-<16 chiffres du SHA-256 du modèle>.span8` et son reçu `.toml` (format `ajp-span-fp8c-v1`) ; si un
garde-fou échoue, seul le reçu est écrit (`passed = false`) et un paquet d'avant est retiré.

Garde-fous du reçu, ceux de l'INT8 et non ceux, lâches, du FP8 par Q/DQ : PSNR moyen ≥ 42 dB, pire ≥ 38 dB, pire
SSIM ≥ 0,985. Le lecteur les relit (`span_int8.cpp`) : un reçu écrit avec des seuils plus lâches ne suffit pas, ni
un reçu d'un autre modèle, ni un paquet modifié depuis son reçu.

## Repli sur TensorRT FP16

À Performance, l'ordre des variantes est : paquet CUTLASS, FP8 par Q/DQ, INT8, puis le SPAN tel que livré en FP16
par TensorRT. Chacune refusée passe à la suivante.

| Cas | Où | Effet |
|---|---|---|
| GPU autre que sm 8.9 (Turing, Ampere, Blackwell sm 12.0) | `BackendCaps::span_fp8_cutlass` faux ; le lecteur n'essaie pas la variante | variante suivante |
| Backend construit sans CUTLASS (hors Linux, `third_party/cutlass` absent, `AJP_WITH_SPAN_FP8=OFF`, pas de CUDA) | `span_fp8_net_none.cpp` : capacité fausse | variante suivante |
| Reçu absent, refusé, d'un autre modèle, paquet modifié | `find_span_variant` | variante suivante |
| Taille que les noyaux ne prennent pas : hauteur ou largeur impaire, `(H/2)(W/2)` non multiple de 128 (540p = 960×540 par exemple), forme libre | `build` refuse (`FormatUnsupported`) | variante suivante (vérifié en lecture : 960×540) |
| Canaux ≠ 64 ou ≠ 8 blocs (b5f48) | `derive_span_fp8.py` n'écrit rien ; `parse` refuse un autre en-tête | pas de variante |
| Paquet illisible, tronqué, CRC faux | `build` | variante suivante |
| Mémoire insuffisante, noyau qui ne tourne pas (CUDA), sortie non finie | `load` : poids envoyés, zone réservée, une image de zéros passée par les noyaux | variante suivante |
| Erreur CUDA en lecture après un chargement réussi | `enqueue` rend un refus, comme un moteur TensorRT | traité comme toute erreur d'exécution du moteur (arrêt de la chaîne) ; pas de bascule à chaud |

Dernier point : une erreur CUDA persistante (Xid) tue le contexte quel que soit le moteur ; une bascule à chaud
vers TensorRT n'y changerait rien, et rien de ce qui est dans le chemin chaud n'alloue ni ne synchronise.

## Dérivation native, dans le lecteur

Réglages avancés > Image IA > Agrandissement : **« Préparer le mode Performance (SPAN FP8) »** (`src/app/play/span_derive.h`).
Visible quand le moteur annonce `span_fp8_cutlass` (sm 8.9) et qu'une vidéo est ouverte ; sinon une note dit pourquoi.
Tâche de fond (`std::async`, barre de progression, bouton Annuler), comme « Évaluer ma machine ». La case
« Préparer en tâche de fond » (éteinte par défaut, `[span_prepare] background` de `play.toml`) lance la même tâche à la fin de
« Évaluer ma machine » quand une vidéo est ouverte, que rien n'est déjà prêt et que le niveau gardé est Rapide ou plus.

| Étape | Où | Définition |
|---|---|---|
| Poids et graphe | `gpu/span_fp8/onnx_weights.{h,cpp}` | lecture ONNX en C++ ; refuse le b5f48 (48 canaux), les 8 blocs, les noms des 25 tenseurs, le dégroupage (Transpose 0,1,3,5,2,4) et DepthToSpace(4, CRD) |
| Images | `media/stills.{h,cpp}` | 24 instants de 8 % à 92 % de la durée, décodage logiciel, découpe centrale ≤ 1920×1088 alignée sur 128×4, RGB 8 bits ; noires et aplats écartés, les 12 plus contrastées gardées ; HDR (PQ, HLG, BT.2020) refusé ; rien n'est conservé |
| Calibration | `Backend::derive_span_fp8` (`tensorrt_span_derive.cpp`, `span_fp8_ref.cu`) | 8 images : le SPAN en FP16 par les mêmes convolutions CUTLASS (accumulation FP32) relève le maximum de \|valeur\| des 25 tenseurs ; échelle = maximum / 448, comme l'outil Python |
| Validation | idem | 4 images mises de côté (une sur trois) : sortie du réseau FP8 contre la même référence FP16, PSNR et SSIM de `quantize_span.py` (`gpu/span_fp8/metrics.cpp`) ; garde-fous du reçu : 42 dB moyen, 38 dB pire, SSIM pire 0,985 |
| Écriture | `commit_derivation` | `.span8` et `.toml` sous `.part`, renommés ensuite, l'ancien restauré si un renommage échoue ; garde-fous manqués : reçu `passed = false` seul, paquet d'avant retiré ; échec technique : rien n'est écrit |

Choix de la référence : le SPAN FP16 des noyaux CUTLASS, pas TensorRT (un moteur TensorRT coûte une minute de construction et ses
tenseurs intermédiaires sont fondus). Elle colle à ONNX Runtime FP32 : maxima à 0,13 % près et 78,8 dB sur une image 1920×1088
(`ajp_span_fp8_run --reference`). Le gain sur TensorRT FP16 n'est pas exigé à la dérivation (le reçu porte `speedup = 0`) ; le test
caché `[.span_derive]` le vérifie. Le reçu a les clés de l'outil Python, plus `reference` et `derived_by` ; un paquet de l'un se
lit chez l'autre (tests `[.span_compat]` : poids identiques octet pour octet, reçu Python accepté par `find_span_variant`).
Interface de backend : `kBackendAbi` passe à 12 (`derive_span_fp8`), les backends installés se réinstallent.

Mesures (RTX 4070 Ti, build Debug) : douze images 1080p de la vidéo d'essai, dérivation de 36 s (essentiellement le SSIM en Debug),
PSNR moyen 51,5 dB, pire 50,7 dB, SSIM 0,9969, réseau FP8 9,23 ms. Test `[.span_derive]` (12 découpes 1920×1088 de PPM) : 49,5 dB,
SSIM 0,997 ; réseau FP8 9,35 ms contre 17,3 ms pour TensorRT FP16 (−46 %). Lecture `ajp play --stats`, 24 i/s, ABAB 3 tours, médiane
« moteur » : 14,60 / 14,54 / 14,58 ms avec le paquet natif contre 15,55 / 15,55 / 15,52 ms en FP16 (−6 %, même effet d'horloges que ci-dessus).

## Ce qui est connu et pas fait

- **sm 12.0 (Blackwell, RTX 50)** : non pris. Le module est compilé pour 89 seulement (`CUDA_ARCHITECTURES
  89` de la bibliothèque des noyaux) ; un portage demande de mesurer les mêmes `mma` E4M3 et les tuiles, sur
  une carte que je n'ai pas. Le repli est automatique.
- **Tailles dynamiques** : le moteur est à une taille fixe, comme les moteurs TensorRT du lecteur (un changement
  de taille de travail recharge la chaîne). Les noyaux n'ont pas de garde de bord : multiple de 128 exigé.
- **Modèle Performance b5f48 (48 canaux, 5 blocs)** : refusé, reste sur TensorRT FP16. Le prototype et ces
  noyaux supposent 64 canaux.
- **Windows** : le module n'est pas construit (`AJP_WITH_SPAN_FP8` vaut OFF hors Linux), CUTLASS n'y a pas été
  essayé avec MSVC. Le lecteur se comporte comme avant.
- **Batch** 1 seulement.
- **Mémoire** : 400 Mo de zone à 1080p (768 o/pixel), plus 2 Mo de poids ; à comparer aux activations de
  TensorRT, qui s'additionnent à celles de RIFE dans la même zone. Sur une carte de 6 Go c'est à surveiller.
- **Horloges** : voir les mesures ; à 24 i/s le gain se lit en puissance plus qu'en temps d'image.

## Construire et tester

```
export PATH=$HOME/ajp-tools/venv/bin:$HOME/ajp-tools/bin:$PATH AJP_DEPS_PREFIX=$HOME/ajp-deps
cmake --preset linux-prefix-cuda -DAJP_TENSORRT_ROOT=$HOME/ajp-sdk/tensorrt
ajp-build cmake --build build/linux-prefix-cuda --target ajp_backend_tensorrt ajp_span_fp8_run \
    ajp_gpu_span_fp8_tests ajp_app_tests ajp_backend_tensorrt_tests
build/linux-prefix-cuda/tests/unit/gpu_span_fp8/ajp_gpu_span_fp8_tests          # paquet et tailles, sans GPU
build/linux-prefix-cuda/tests/unit/app/ajp_app_tests "[fp8c]"                   # reçu, garde-fous, crans, sans GPU
ajp-gpu build/linux-prefix-cuda/tests/unit/backend_tensorrt/ajp_backend_tensorrt_tests "[span_fp8]"
```

Le test caché `[.span_fp8]` compare la sortie du moteur FP8 à celle du moteur TensorRT FP16 du même modèle :
`-DAJP_SPAN_FP8_PACKAGE=…span8 -DAJP_SPAN_FP8_MODEL=…onnx` (et `-DAJP_SPAN_FP8_IMAGE=image.ppm` pour une image
d'anime au lieu du dessin synthétique), puis `ajp_backend_tensorrt_tests "[.span_fp8]"` sous `ajp-gpu` ; sans ces
variables CMake, il se SKIP.

`BackendCaps::span_fp8_cutlass` ajoute un champ à la frontière du backend : `kBackendAbi` passe à 11, les backends
installés par une version précédente se réinstallent depuis le paquet (même mécanisme que pour les ABI d'avant).
