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

## TensorRT : comment le module arrive

- Chaque release a un paquet TensorRT pour Linux et pour Windows (décision de Tatoub) : `release.yml` échoue sur une
  release finale qui n'en a pas. Le paquet vit dans une release fixe, `trt-<empreinte>` (Linux) ou
  `trt-win-<empreinte>` (Windows), où l'empreinte est celle de ses entrées (`tools/release/trt-fingerprint.sh`,
  `--windows`). Toute modification de `src/core`, `src/gpu`, `src/kernels`, `src/prepare`, de CUTLASS ou des
  définitions CMake change l'empreinte : la CI de release construit alors le paquet sur un runner sans carte
  (`trt-package.yml` pour Linux, job `tensorrt` de `release-windows.yml`), le publie, puis le reprend dans
  l'`engines.lock` du lecteur. Les tests sur GPU restent au poste NVIDIA.
- Un paquet joint **après** la release sert aussi : le lecteur relit `engines-nvidia.lock` de la release de sa
  version au lancement (au plus une fois par jour, `src/app/play/engine_online.h`).
- Aucun fichier NVIDIA n'est dans nos archives : `engines.lock` décrit chaque bibliothèque (`[[vendor_file]]`,
  plage HTTP dans la roue NVIDIA, SHA-256) et le lecteur la télécharge après l'accord de licence.
- Un module d'une version précédente (empreinte binaire différente, `gpu::check_backend_abi`) se réinstalle
  seul depuis le paquet de la version courante, au lancement ou quand on choisit TensorRT, puis le lecteur se
  relance (`update_stale_engines`).

Détails NVIDIA par système : `docs/nvidia-linux.md`, `docs/nvidia-windows.md`.
