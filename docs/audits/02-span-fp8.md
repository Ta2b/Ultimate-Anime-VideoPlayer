# Audit 02. SPAN FP8 : noyaux CUTLASS et dérivation

**Audit par zone. Priorité : très haute.** 25 fichiers, 3 526 lignes ; tests rattachés : 9 fichiers, 1 496 lignes.

## Mission

Le SPAN en FP8 sur Ada et Blackwell : préparation du paquet `.span8` (poids, échelles, cas difficiles), noyaux CUTLASS, référence, essai au chargement et porte de qualité.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Échelles : maximum exact sans marge, saturation silencieuse (TRT-15).
- Essai au chargement : 8 Ko lus sur une entrée nulle, NaN non détectés (TRT-16).
- Porte de qualité : seuils, images de test, résultat gardé par carte et par pilote.
- Noyaux : mémoire partagée réservée pour tous les modes (`span_fp8_kernels.cuh:280`), `__restrict__`, occupation.
- Architectures : sm_89 et sm_100 et plus, refus propre ailleurs.
- Concordance avec le SPAN FP16 sur des images réelles d'anime.

## Frontières

Dépend de :

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (8 inclusions)
- [20. Socle commun (core)](20-socle-core.md) (3 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (2 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (1 inclusion)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (1 inclusion)

Utilisé par :

- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (6 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-15 | Mineur | Les échelles FP8 prennent le maximum exact, sans marge. Une image plus claire que la calibration est écrêtée sans signal. |
| TRT-16 | Mineur | L'essai au chargement ne lit que 8 Ko de sortie, sur une entrée nulle. Un mauvais résultat ou des NaN passent. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/cuda/backend/span_fp8_engine.cpp` | 269 |
| `src/gpu/cuda/backend/span_fp8_engine.h` | 96 |
| `src/gpu/cuda/backend/span_fp8_kernels.cuh` | 295 |
| `src/gpu/cuda/backend/span_fp8_layout.cuh` | 78 |
| `src/gpu/cuda/backend/span_fp8_net.cu` | 370 |
| `src/gpu/cuda/backend/span_fp8_net.h` | 93 |
| `src/gpu/cuda/backend/span_fp8_net_none.cpp` | 72 |
| `src/gpu/cuda/backend/span_fp8_ref.cu` | 367 |
| `src/gpu/cuda/backend/span_fp8_ref.h` | 50 |
| `src/gpu/cuda/backend/span_fp8_run.cpp` | 178 |
| `src/gpu/cuda/backend/tensorrt_span_derive.cpp` | 310 |
| `src/gpu/cuda/backend/tensorrt_span_fp8.cpp` | 108 |
| `src/gpu/span_fp8/CMakeLists.txt` | 8 |
| `src/gpu/span_fp8/derive.cpp` | 175 |
| `src/gpu/span_fp8/derive.h` | 128 |
| `src/gpu/span_fp8/hard_cases.cpp` | 88 |
| `src/gpu/span_fp8/hard_cases.h` | 25 |
| `src/gpu/span_fp8/metrics.cpp` | 134 |
| `src/gpu/span_fp8/metrics.h` | 26 |
| `src/gpu/span_fp8/onnx_weights.cpp` | 250 |
| `src/gpu/span_fp8/onnx_weights.h` | 25 |
| `src/gpu/span_fp8/package.cpp` | 216 |
| `src/gpu/span_fp8/package.h` | 91 |
| `src/gpu/span_fp8/quantize.cpp` | 47 |
| `src/gpu/span_fp8/quantize.h` | 27 |

## Tests rattachés

- `tests/unit/app/span_fp8_cutlass_test.cpp`
- `tests/unit/backend_span_fp8_kernels/span_fp8_kernel_test.cpp`
- `tests/unit/backend_tensorrt/span_fp8_test.cpp`
- `tests/unit/gpu_span_fp8/derive_test.cpp`
- `tests/unit/gpu_span_fp8/hard_cases_test.cpp`
- `tests/unit/gpu_span_fp8/metrics_test.cpp`
- `tests/unit/gpu_span_fp8/onnx_weights_test.cpp`
- `tests/unit/gpu_span_fp8/package_test.cpp`
- `tests/unit/gpu_span_fp8/quantize_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
