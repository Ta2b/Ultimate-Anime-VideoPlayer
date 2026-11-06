# Moteurs des réseaux

Le lecteur fait tourner ses réseaux (SPAN pour l'agrandissement, RIFE, GMFSS et DistilDRBA pour
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

## Ce qui est débranché

**MIGraphX** (ROCm sous Linux, Windows ML sous Windows) et **OpenVINO** restent dans les sources
(`src/gpu/migraphx`, `src/gpu/winml`, `src/gpu/openvino`) et la CI les construit encore (`ci.yml`), mais le
lecteur ne les propose plus, ne les installe plus et ne les ouvre plus. La raison : aucune mesure ne montre
qu'ils battent Vulkan sur une carte AMD ou Intel, et Windows ML n'a jamais démarré une vidéo sur la seule
machine AMD essayée (Ryzen AI 7 350).

Concrètement :
- `core::engine_plugged()` (`src/core/engines.h`) dit non pour MIGraphX, OpenVINO et DirectML ;
- `recommend_engines()` ne les propose pas, `installed_backends()` et `backend_candidates()` ne les ouvrent
  pas, même installés par une version précédente ;
- au lancement, un `settings.toml` qui les avait choisis les perd (journal : « débranché de ce lecteur »),
  Vulkan prend leur place ;
- les workflows de release ne les construisent plus : aucune archive `ajp-engine-migraphx-*` ni
  `ajp-engine-openvino-*` n'est jointe.

Pour les rebrancher : rendre `engine_plugged()` vrai pour le moteur, remettre ses offres dans
`recommend_engines()` (l'historique git de `src/core/engines.cpp` les garde) et ses options `AJP_WITH_*` dans
`release-linux.yml` et `release-windows.yml`. Le faire seulement après une mesure qui le justifie, sur la
carte visée, face à Vulkan (`ajp bench`).

DirectML n'a jamais eu de backend.

## TensorRT : comment le module arrive

- Windows : `release-windows.yml` avec `tensorrt=true` construit `ajp_backend_tensorrt.dll` et `trtexec.exe`,
  les joint à la release (`ajp-engine-tensorrt-windows-x64.tar.gz`) avec leur index `engines-nvidia.lock`. La
  construction `tensorrt=false` de la même étiquette reprend cet index dans l'`engines.lock` de l'installeur :
  **elle doit tourner après**, sinon l'installeur ne connaît pas TensorRT.
- Linux : le paquet se construit sur un poste Linux NVIDIA (`tools/release/trt-package.sh`) et vit dans une
  release fixe `trt-<empreinte>` (`tools/release/trt-fingerprint.sh`). Toute modification de `src/core`,
  `src/gpu`, `src/kernels` ou `src/prepare` change l'empreinte et demande un nouveau paquet.
- Aucun fichier NVIDIA n'est dans nos archives : `engines.lock` décrit chaque bibliothèque (`[[vendor_file]]`,
  plage HTTP dans la roue NVIDIA, SHA-256) et le lecteur la télécharge après l'accord de licence.
- Un module d'une version précédente (empreinte binaire différente, `gpu::check_backend_abi`) se réinstalle
  seul depuis le paquet de la version courante, au lancement ou quand on choisit TensorRT, puis le lecteur se
  relance (`update_stale_engines`).

Détails NVIDIA par système : `docs/nvidia-linux.md`, `docs/nvidia-windows.md`.
