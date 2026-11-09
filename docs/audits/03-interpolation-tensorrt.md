# Audit 03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)

**Audit par zone. Priorité : très haute.** 18 fichiers, 3 452 lignes ; tests rattachés : 0 fichiers, 0 lignes.

## Mission

RIFE et DistilDRBA découpés en moteurs TensorRT et reliés par des noyaux CUDA de colle, le warp final, la reprise à chaud des blocs grossiers.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Enchaînement des morceaux : formes, dépassements de bornes dans les noyaux de colle.
- Reprise à chaud (`next_warm`) : invalidée à chaque coupe et à chaque seek ?
- Warp : bords d'image, flux hors image, instants t différents de 0,5.
- Coût de l'entrée de RIFE (0,4 à 0,8 ms) et pistes pour la fondre.
- Concordance avec le réseau entier (tests existants, seuils).

## Frontières

Dépend de :

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (18 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (7 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (2 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (2 inclusions)

Utilisé par :

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (3 inclusions)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/cuda/backend/drba_fused_engine.cpp` | 407 |
| `src/gpu/cuda/backend/drba_fused_engine.h` | 130 |
| `src/gpu/cuda/backend/drba_fused_kernels.cu` | 355 |
| `src/gpu/cuda/backend/drba_fused_kernels.h` | 83 |
| `src/gpu/cuda/backend/rife_fused_engine.cpp` | 383 |
| `src/gpu/cuda/backend/rife_fused_engine.h` | 108 |
| `src/gpu/cuda/backend/rife_fused_kernels.cu` | 294 |
| `src/gpu/cuda/backend/rife_fused_kernels.h` | 71 |
| `src/gpu/cuda/backend/rife_kernels.cu` | 135 |
| `src/gpu/cuda/backend/rife_kernels.h` | 64 |
| `src/gpu/cuda/backend/slice_kernels.cu` | 402 |
| `src/gpu/cuda/backend/slice_kernels.h` | 85 |
| `src/gpu/cuda/backend/tensorrt_drba_fused.cpp` | 108 |
| `src/gpu/cuda/backend/tensorrt_rife.cpp` | 226 |
| `src/gpu/cuda/backend/tensorrt_rife_fused.cpp` | 110 |
| `src/gpu/cuda/backend/tensorrt_warp.cpp` | 203 |
| `src/gpu/cuda/backend/warp_kernels.cu` | 220 |
| `src/gpu/cuda/backend/warp_kernels.h` | 68 |

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
