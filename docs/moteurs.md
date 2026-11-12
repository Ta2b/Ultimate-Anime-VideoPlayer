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

## Vulkan : la précision des noyaux

Le moteur Vulkan range ses tenseurs en float16. Chaque couche a plusieurs noyaux candidats, que l'autotuner
mesure à la première construction (`src/gpu/vulkan/backend/net_autotune.cpp`). Certains font aussi leurs
**sommes** en float16 (`direct_f16p`, `winograd_f16p`, `coopmat_acc16`) : plus rapides sur certaines cartes,
moins justes. Trois règles tiennent la qualité (C05, A45-01, A45-02) :

- à moins de 5 % de temps d'écart, les sommes en float gagnent (`kPrecisionTie`) ;
- un choix à précision réduite ne se reprend jamais sans mesure, ni d'une table pré-réglée ni d'un autre
  pilote. La table `NvidiaAda`, relevée avant la règle des 5 % et qui donnait 28 couches sur 29 en sommes
  float16, est retirée ; `tools/vk/presets_from_cache.py` refuse déjà les relevés d'avant `c7` ;
- la porte de bout en bout (`net_gate.h`) rejoue le réseau choisi contre son jumeau calculé tout en float, sur
  un dessin (aplats sombres, traits fins ; deux images décalées et leur grille pour un réseau d'interpolation).
  Les choix passent s'ils perdent au plus 0,5 dB sous les noyaux de référence (`kGateLossDb`) ; sous des noyaux de
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
est une vraie référence. SPAN repasse donc en sommes float quand tout son réseau était en float16 ; RIFE garde
ses choix. `coopmat_acc16` ne se mesure que sur une carte à Tensor Cores (`docs/audits/A-CONFIRMER.md`).

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
