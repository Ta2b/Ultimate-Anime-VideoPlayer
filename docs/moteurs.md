# Moteurs des réseaux

Le lecteur fait tourner ses réseaux (SPAN pour l'agrandissement, RIFE et DistilDRBA pour
l'interpolation) sur l'un de ces moteurs. Le choix se fait à l'écran des moteurs (premier lancement, ou
Paramètres > Général > Moteurs) et s'enregistre dans `settings.toml`.

## Ce qui est branché

| Moteur | GPU | Livraison | Code |
|---|---|---|---|
| **TensorRT** (CUDA, NVDEC) | NVIDIA RTX | téléchargé : notre module depuis la release, les bibliothèques NVIDIA depuis pypi.nvidia.com | `src/gpu/cuda` |
| **Vulkan** | tous, et le seul pour AMD et Intel | livré avec le lecteur | `src/gpu/vulkan` |
| **Core ML** (Metal) | Apple | livré avec le lecteur macOS | `src/gpu/coreml` |

Règle : une carte NVIDIA RTX prend TensorRT, toute autre carte prend Vulkan. Vulkan est aussi le repli de
TensorRT quand son module ne s'ouvre pas.

## Ce qui est retiré

**MIGraphX** (ROCm sous Linux, Windows ML sous Windows) et **OpenVINO** sont retirés des sources (0.10.0) :
aucune mesure ne montrait qu'ils battent Vulkan sur une carte AMD ou Intel, et Windows ML n'a jamais démarré
une vidéo sur la seule machine AMD essayée (Ryzen AI 7 350).

Ce qui en reste, pour les anciens fichiers :
- les noms `migraphx`, `openvino` et `directml` se lisent encore (`EngineId`, `core::engine_plugged()` dit non) ;
- au lancement, un `settings.toml` qui les avait choisis les perd (journal : « débranché de ce lecteur »),
  Vulkan prend leur place ; un paquet installé par une version précédente ne s'ouvre plus.

Pour en rebrancher un : l'historique git garde `src/gpu/migraphx`, `src/gpu/openvino` et `src/gpu/winml`
(avant 0.10.0). Le faire seulement après une mesure qui le justifie, sur la carte visée, face à Vulkan
(`ajp bench`).

DirectML n'a jamais eu de backend.

## Ce que Vulkan offre face à TensorRT (novembre 2026)

Le mode Performance de SPAN (FP8) existe aussi sous Vulkan, sur une carte à matrices coopératives FP8 (RDNA 4,
RX 9000) : même paquet, même reçu, même porte que sous TensorRT (`docs/span-fp8-cutlass.md`, « Sous
Vulkan »). Les graphes Q/DQ FP8 de RIFE et DistilDRBA (cran Performance) y passent par les mêmes matrices.

La variante INT8 du SPAN (`tools/span/quantize_span.py`, reçu `ajp-span-int8-v1`) se lit aussi sous Vulkan :
l'import plie ses paires Q/DQ INT8 en `FakeQuantInt8` et ses poids INT8 en q × échelle, les passes fondent
l'arrondi dans les Conv (`ConvAttrs::quant_grid`), le noyau direct arrondit à la grille INT8 et la variante
`coopmat_int8` multiplie des entiers sur les tuiles s8 × s8 → s32 16 × 16 × 16 (`net_conv_coopmat_int8.glsl`).
Le lecteur ne la propose au cran Performance (`BackendCaps::int8`) que là où l'INT8 va deux fois plus vite que
le float16 : RDNA 4 et GeForce Turing à Ada (`int8_graphs_allowed`, `family.h`). Les RDNA 3 font l'INT8 au
rythme du float16 (122,8 TOPS INT8 comme TFLOPS FP16 sur RX 7900 XTX) : elles y perdraient de la qualité sans
rien gagner, et gardent le FP16. Une RX 9000 prend d'abord le FP8, plus juste ; l'INT8 ne lui sert que sans
paquet FP8. Les Arc annoncent leurs tuiles INT8 en 8 × 16 × 32, que ce noyau ne sait pas tuiler.
**Écrit sans carte AMD ni Intel pour l'essayer** : les tests tournent sous lavapipe ; les retours
d'utilisateurs diront ce qui manque.

### Optimisations : audit face à TensorRT (17 novembre 2026, sans carte AMD)

À parité dans le code : autotuner et cache par carte et pilote, FP16, FP8 (RDNA 4) et INT8, formes fixes, Conv
fondue avec biais, activation, résiduel et porte du SPAN, DepthToSpace fondu, enregistrements rejoués
(l'équivalent des graphes CUDA), voies sur plusieurs files ordonnées par ressource, décodage sans copie (Vulkan
Video), anneau partagé avec libplacebo, tailles de travail, choix et portes de l'interpolation.

Comblé ce jour-là :
- le Concat final du SPAN est rangé en place même quand ses entrées ont d'autres lecteurs, comme dans le moteur
  FP8 CUDA : plus de copie en pleine taille à chaque image. Une Conv à matrices coopératives qui lit une
  tranche compte K sur ses propres canaux (`ckp`) au lieu de calculer sur des zéros ;
- warp : le mouvement lu une fois par point, comme `sample_n` sous CUDA ;
- débandage fait à l'ingestion, en une passe ; tampons de travail de l'ingestion partagés entre les images du
  décodeur ;
- mesures du flux : 200 µs d'interrogation avant l'attente bloquante, comme `TensorRtBackend::wait_event` ;
- une puce AMD hors des tables (pilote Adrenalin) se range d'après ses tuiles (`family_from_tiles`) au lieu de
  perdre les matrices.

Reste en retrait, à mesurer sur une carte avant d'y toucher :
- entrées et sorties des morceaux fusionnés de RIFE en Nchw : l'équivalent du HWC8 de TensorRT (−29 % sous
  TensorRT) serait le Nhwc16 ;
- fin de bloc du SPAN (c3, puis la Conv 1 × 1 d'attention et la porte) en deux noyaux, là où le moteur FP8 CUDA
  en fait un ; activations FP8 rangées en FP16 ; le mode Performance FP8 n'est le défaut que sous TensorRT (Ada) ;
- ni découpage pour le cache L2, ni zone d'activations partagée entre les voies (mémoire, pas vitesse) ;
- priorités de file : libplacebo crée toutes les files à la même priorité, sans réglage pour la changer ;
- ingestion en trois passes float et copie de l'encodage de RIFE à chaque paire : de l'ordre d'un dixième de
  milliseconde par image en 1080p.

### Parité avec TensorRT (19 novembre 2026, RTX 4070 Ti)

Mesuré dans le lecteur (Kill Blue, 720p, SPAN b8f64 en FP16, sortie 1440p) :

| | Vulkan avant | Vulkan après | TensorRT |
|---|---|---|---|
| SPAN, par image | 27,5 ms | 13,4–14,7 ms (banc : 7,6–9,4 ms) | 10,6–11,2 ms |
| RIFE 4.22 lite, images affichées | 24–28 i/s | 51–68 i/s | 63–82 i/s |
| DistilDRBA, images affichées | 30–32 i/s | 59–70 i/s | 78–99 i/s |

Trois causes, corrigées :
- les boucles des noyaux à matrices coopératives (`net_conv_coopmat*.glsl`, float16, FP8, INT8) sont déroulées
  (`[[unroll]]`) : le pilote NVIDIA ne les déroulait pas et sortait les sommes partielles des registres (une
  Conv 3 × 3 64 → 64 : 0,77 → 0,25 ms) ;
- l'autotuner fait 24 passes de chauffe avant de mesurer (`TunerOptions::warmup`, 2 avant) : la carte, au repos
  pendant la compilation du candidat, mesurait à basse fréquence et choisissait des variantes plus lentes ;
- le bruit ne donne plus qu'au plus 10 % d'avance aux sommes en float (`kNoiseTieCap`).

Les candidats passent à `c9` : les choix en cache se remesurent au premier lancement. En lecture, SPAN partage
la carte avec le rendu de libplacebo (file de calcul à côté) : ses noyaux s'allongent, mais tiennent dans les
42 ms d'une image. Reste l'écart de RIFE et DistilDRBA fusionnés (6,5 et 3,8 ms par passe contre 1,45 et
0,47 ms), surtout dans les convolutions de leurs réseaux.

## Vulkan : les matrices coopératives d'Intel (tuiles 8×16×16)

Le noyau à matrices coopératives float16 (`net_conv_coopmat.glsl`) tuile en 16 × 16 × 16, ce qu'annoncent les
WMMA d'AMD (RDNA 3 et 4) et les tensor cores de NVIDIA. Les XMX d'Intel Xe2 (Lunar Lake, Battlemage ; Panther
Lake aussi) annoncent sous ANV des tuiles de M = 8 : 8 × 16 × 16, f16 × f16 → f32 (et → f16), par sous-groupe.
La variante `coopmat_m8` (`net_conv_m8.cpp`, noyau `net_conv_coopmat_m8_f16`, novembre 2026) les sert : même
noyau compilé en tuiles de 8 pixels (`AJP_COOPMAT_M8`), mêmes poids que `coopmat` (N et K restent 16), sommes en
float, B lue dans le tampon ou partagée, sans les tuiles à halo, l'épilogue en registres ni les canaux découpés.
Des sous-groupes de 16 sont exigés à chaque dispatch, la largeur SIMD des Xe2 (leur `minSubgroupSize`).

Règles tenues :

- **rien ne change ailleurs** : `Target::matrix_m8` n'est vrai que si le pilote annonce la tuile 8 × 16 × 16
  f16 → f32 *sans* la tuile 16 × 16 × 16, sur une famille permise (`matrix_units_excluded`), avec des sous-groupes
  de 16 exigeables. Les modules SPIR-V des autres variantes sont restés identiques à l'octet près ;
- **jamais en tête** : écrite sans carte Intel pour l'essayer, la variante est toujours proposée derrière le noyau
  direct, quel que soit l'appareil. L'autotuner seul la retient, si sa sortie concorde avec celle du noyau direct
  et qu'elle va plus vite ; la porte de bout en bout juge ensuite le réseau entier. Sans autotuner, le noyau direct
  (ou Winograd) reste le choix ;
- **mesure neuve sur ces appareils seuls** : la version des candidats de leur cache prend le suffixe « m8 »
  (`candidate_set`) et leurs artefacts une clé à part (`artifact_key`) ; les autres cartes gardent leurs mesures
  et leurs artefacts.

Ce qui reste : les Arc A (Alchemist, DG2) annoncent sous ANV 8 × 8 × 16 (N = 8), qu'aucun noyau ne tuile ; ce
que le pilote Windows d'Intel annonce n'est pas relevé. Les tuiles du pilote sont au journal d'ouverture du
backend (« tuiles du pilote : … »). `ajp_vk_backend_tests "[m8]"` exécute la variante contre l'interpréteur sur un
appareil qui l'a ; sous lavapipe, seuls son abaissement et sa mémoire partagée (évaluée dans le module SPIR-V)
sont vérifiés.

## Vulkan : la précision des noyaux

Le moteur Vulkan range ses tenseurs en float16. Chaque couche a plusieurs noyaux candidats, que l'autotuner
mesure à la première construction (`src/gpu/vulkan/backend/net_autotune.cpp`). Certains font aussi leurs
**sommes** en float16 (`direct_f16p`, `winograd_f16p`, `coopmat_acc16`) : plus rapides sur certaines cartes,
moins justes. Trois règles tiennent la qualité (C05, A45-01, A45-02) :

- à moins de 5 % de temps d'écart, les sommes en float gagnent (`kPrecisionTie`) ; le bruit de mesure (deux
  écarts types) peut élargir cette marge, au plus de 10 % (`kNoiseTieCap`, novembre 2026) ;
- un choix à précision réduite ne se reprend jamais sans mesure, ni d'une table pré-réglée ni d'un autre
  pilote. La table `NvidiaAda`, relevée avant la règle des 5 % et qui donnait 28 couches sur 29 en sommes
  float16, est retirée ; `tools/vk/presets_from_cache.py` refuse déjà les relevés d'avant `c7` ;
- la porte de bout en bout (`net_gate.h`) rejoue le réseau choisi contre son jumeau calculé tout en float, sur
  un dessin (aplats sombres, traits fins ; deux images décalées et leur grille pour un réseau d'interpolation).
  Les choix passent s'ils perdent au plus 0,5 dB sous les noyaux de référence (`kGateLossDb`), ou s'ils restent
  au-dessus de 60 dB contre le calcul en float (`kGateInvisibleDb`, un écart sous le quart d'un niveau sur 255) ; sous des noyaux de
  référence exacts (200 dB, graphe en float32), le plancher vaut 90 dB (`kGateExactFloorDb`) pour qu'une variante
  juste à l'arrondi près ne soit pas rejetée. Sinon les sommes repassent en float, puis tout le réseau reprend les
  noyaux de référence. Le journal donne les deux PSNR. Un réseau dont le calcul de référence sort une valeur non
  finie est refusé (« Réessayer » dans le lecteur), pas gardé faute de juge.

Avant, la porte comparait à un autre réseau en float16, sur du bruit, à 1 % d'écart quadratique (2 % en
précision réduite, soit 39 dB). Sur du bruit un réseau d'interpolation diverge : RIFE 4.25 entier y montrait
49 % d'écart pour des sommes float16 qui ne lui coûtent que 0,15 dB sur un dessin. La porte refusait donc ce
qui était juste, sans mesurer ce qui se voit : l'écart au calcul exact sur une image.

Mesure (`ajp_vk_backend_tests "[.sommes]"`, lavapipe, PSNR contre le calcul en float, dessin de la porte) :

| réseau | noyaux de référence | `direct_f16p` | `winograd_f16p` |
|---|---|---|---|
| SPAN b5f48, 640 × 360 | 81,6 dB | 74,3 dB (−7,3) | 71,0 dB (−10,6) |
| RIFE 4.25, 640 × 384 | 77,9 dB | 77,8 dB (−0,15) | 77,2 dB (−0,75) |
| RIFE 4.22 lite, 640 × 384 | 77,7 dB | 76,6 dB (−1,1) | 71,4 dB (−6,3) |

Sur une vraie image (plan de `calibration.mkv`, sorties comparées à ONNX Runtime en FP32) : SPAN 75,2 dB aux
noyaux de référence et 70,3 dB en sommes float16 (−4,8 dB, pire écart 1,0/255 contre 0,4/255) ; RIFE 4.25
72,6 et 72,5 dB. Le moteur Vulkan en float rend ONNX Runtime FP32 à 107 dB (RIFE) et 133 dB (SPAN) : le jumeau
est une vraie référence. Avec la seule règle des 0,5 dB, SPAN repassait en sommes float : sur une GeForce Ada,
ses Tensor Cores allaient alors deux fois moins vite, et comme SPAN tourne à chaque image affichée,
l'interpolation sous Vulkan perdait près de la moitié de ses images par seconde face à la 0.10.3 (novembre
2026). Ses 70 dB ne se voient pas : au-dessus de `kGateInvisibleDb`, SPAN et RIFE gardent leurs sommes float16.
La case « Vitesse avant précision (Vulkan) » (Général > Moteurs, `engines.fast_precision` dans `play.toml`, cochée
par défaut) le décide : décochée, seule la règle des 0,5 dB tient (`gpu::kStrictPrecision`, porté par
`BackendInit::trials` sans monter `kBackendAbi`). Elle entre dans la clé des artefacts : chaque réglage a ses
moteurs, construits au prochain lancement. `coopmat_acc16` ne se mesure que sur une carte à Tensor Cores (`docs/audits/A-CONFIRMER.md`).

Quand le jumeau en float ne peut pas être joué (mémoire), la porte d'avant juge et le journal le dit.

**Ce que l'utilisateur voit.** Une perte consentie à la vitesse se dit à l'écran, pas au journal seul : quand
l'autotuner garde des couches à précision réduite, le lecteur affiche au lancement un avis (« Moteur SPAN :
précision réduite sur N couche(s) pour gagner en vitesse (X dB contre le calcul en float, vitesse +Y %) »). La
perte est celle de la porte, le gain celui que l'autotuner a mesuré ; l'un et l'autre disent « non mesuré » quand
ils ne le sont pas (choix repris du cache, jumeau en float injouable). L'avis est rangé dans l'artefact
(`ArtifactData::notice`) et revient à chaque lancement. Gain sur une carte réelle : non mesuré (RTX à mesurer).

**Limite des sommes en float16.** Les trois chemins à sommes float16 (`coopmat_acc16`, `direct_f16p`,
`winograd_f16p`) ne plafonnent qu'en fin de série de sommes (fin de K, paquet de 16 canaux). Une somme qui
déborde 65 504 donne ±Inf, puis NaN dans le même accumulateur ; le plafond de ±60 000 ne rattrape pas un NaN déjà
formé. Les activations de SPAN et de RIFE sont loin de cette borne (sous 100) ; la porte de fidélité ne juge
qu'un dessin et ne verrait pas un débordement sur une image extrême. Limite acceptée et décrite ici (C49-02) ;
la variante en sommes float reste proposée à côté de chacune.

## TensorRT : comment le module arrive

- Chaque release a un paquet TensorRT pour Linux et pour Windows (décision de Tatoub) : `release.yml` échoue sur une
  release finale qui n'en a pas. Le paquet vit dans une release fixe, `trt-<empreinte>` (Linux) ou
  `trt-win-<empreinte>` (Windows), où l'empreinte est celle de ses entrées (`tools/release/trt-fingerprint.sh`,
  `--windows`). Toute modification de `src/core`, `src/gpu`, `src/kernels`, `src/prepare`, de CUTLASS ou des
  définitions CMake change l'empreinte : la CI de release construit alors le paquet sur un runner sans carte
  (`trt-package.yml` pour Linux, job `tensorrt` de `release-windows.yml`), le publie, puis le reprend dans
  l'`engines.lock` du lecteur. Les tests sur GPU restent au poste NVIDIA.
- Un paquet joint **après** la release sert aussi : le lecteur relit `engines-nvidia.lock` de la release de sa
  version au lancement (au plus une fois par jour, `src/app/play/engine_online.h`). Il ne le croit que signé
  (`engines-nvidia.lock.sig`, clé des mises à jour, A26-02) : le job « Publier » de `release.yml` l'écrit
  (`tools/release/nvidia-index.sh`) et le signe (`tools/release/sign-file.sh`). Après un `attach-nvidia-*`, relancer
  Release sur l'étiquette pour le signer.
- Un moteur de l'index n'est cru que pour une archive du dépôt des releases et des fichiers de `pypi.nvidia.com`.
  Aucune version du lecteur n'est inscrite dans l'index signé : un ancien index signé rejoué ne peut proposer qu'un
  module d'une autre version, que le contrôle d'ABI refuse (Vulkan prend la suite).
- Aucun fichier NVIDIA n'est dans nos archives : `engines.lock` décrit chaque bibliothèque (`[[vendor_file]]`,
  plage HTTP dans la roue NVIDIA, SHA-256) et le lecteur la télécharge après l'accord de licence.
- Un module d'une version précédente (empreinte binaire différente, `gpu::check_backend_abi`) se réinstalle
  seul depuis le paquet de la version courante, au lancement ou quand on choisit TensorRT, puis le lecteur se
  relance (`update_stale_engines`).

Détails NVIDIA par système : `docs/nvidia-linux.md`, `docs/nvidia-windows.md`.
