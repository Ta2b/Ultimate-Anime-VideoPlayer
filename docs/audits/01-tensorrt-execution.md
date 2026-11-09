# Audit 01. TensorRT : exécution, voies, anneau et mémoire

**Audit par zone. Priorité : très haute.** 30 fichiers, 4 186 lignes ; tests rattachés : 2 fichiers, 244 lignes.

## Mission

Le moteur TensorRT au moment de l'exécution : contexte CUDA, voies et événements, chargement des moteurs, envoi des images, anneau partagé avec Vulkan, noyaux de pixels, de statistiques et de débandage.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Gestion d'erreur CUDA : chaque appel vérifié (18 appels non vérifiés relevés).
- Mémoire : pool CUDA jamais rendu, fuites si la création échoue (TRT-06, TRT-13).
- Anneau partagé : emplacements retirés barrés sur toutes les voies, poignées d'import libérées (TRT-12).
- Envoi des images logicielles : copie synchrone sur le thread qui soumet (`host_upload.cpp:117-128`).
- Voies : nombre selon la mémoire vidéo (une seule sous 7,5 Gio), attente des événements, priorités.
- Graphes CUDA : recapture au changement de taille, invalidation.
- Perte du GPU (TDR sous Windows, Xid sous Linux) : détection et message.
- WDDM sous Windows contre Linux : mêmes temps, même stabilité.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (9 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (9 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (7 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (4 inclusions)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (3 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (1 inclusion)

Utilisé par :

- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (18 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (8 inclusions)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-06 | Mineur | Le pool CUDA par défaut ne rend jamais la mémoire vidéo. Sur une carte de 4 à 6 Go, Vulkan et libplacebo peuvent en manquer. |
| TRT-12 | Mineur | Les emplacements retirés de l'anneau ne sont barrés que sur la voie courante. Une mémoire externe peut être détruite pendant qu'une autre voie écrit. |
| TRT-13 | Mineur | Si `create` échoue à mi-chemin, files et événements CUDA fuient. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/cuda/CMakeLists.txt` | 3 |
| `src/gpu/cuda/backend/CMakeLists.txt` | 120 |
| `src/gpu/cuda/backend/bilinear.cuh` | 64 |
| `src/gpu/cuda/backend/cuda_status.h` | 54 |
| `src/gpu/cuda/backend/cuda_warn.h` | 35 |
| `src/gpu/cuda/backend/deband_kernels.cu` | 54 |
| `src/gpu/cuda/backend/deband_kernels.cuh` | 39 |
| `src/gpu/cuda/backend/deband_kernels.h` | 25 |
| `src/gpu/cuda/backend/entry.cpp` | 101 |
| `src/gpu/cuda/backend/export_kernels.cu` | 139 |
| `src/gpu/cuda/backend/export_kernels.h` | 53 |
| `src/gpu/cuda/backend/gpu_timing.cpp` | 155 |
| `src/gpu/cuda/backend/gpu_timing.h` | 89 |
| `src/gpu/cuda/backend/host_upload.cpp` | 162 |
| `src/gpu/cuda/backend/host_upload.h` | 79 |
| `src/gpu/cuda/backend/pixel_kernels.cuh` | 100 |
| `src/gpu/cuda/backend/ring_slot.cpp` | 174 |
| `src/gpu/cuda/backend/ring_slot.h` | 51 |
| `src/gpu/cuda/backend/ring_surface.h` | 21 |
| `src/gpu/cuda/backend/stats_kernels.cu` | 261 |
| `src/gpu/cuda/backend/stats_kernels.h` | 22 |
| `src/gpu/cuda/backend/tensorrt_backend.cpp` | 510 |
| `src/gpu/cuda/backend/tensorrt_backend.h` | 404 |
| `src/gpu/cuda/backend/tensorrt_export.cpp` | 270 |
| `src/gpu/cuda/backend/tensorrt_kernels.cpp` | 360 |
| `src/gpu/cuda/backend/tensorrt_lanes.cpp` | 152 |
| `src/gpu/cuda/backend/tensorrt_stats.cpp` | 77 |
| `src/gpu/cuda/backend/tensorrt_textures.cpp` | 108 |
| `src/gpu/cuda/backend/trt_engine.cpp` | 347 |
| `src/gpu/cuda/backend/trt_engine.h` | 157 |

## Tests rattachés

- `tests/unit/backend_tensorrt/cuda_status_test.cpp`
- `tests/unit/backend_tensorrt/export_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
