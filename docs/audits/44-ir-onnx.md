# Audit 44. Représentation intermédiaire et import ONNX

**Audit par zone. Priorité : réduit : utile au repli Vulkan.** 50 fichiers, 8 037 lignes ; tests rattachés : 16 fichiers, 4 087 lignes.

## Mission

Lecture des modèles ONNX, graphe interne, passes de fusion et de conversion de types, utilisé par le moteur Vulkan. Dans la cible seulement parce qu'il sert au repli Vulkan sur une RTX : auditer l'exactitude et la robustesse, pas la performance.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Fichiers ONNX hostiles : lecteur protobuf, tailles, récursion (SEC-21).
- Passes de fusion : chaque motif vérifié par un test numérique.
- Conversion FP32 vers FP16 des modèles (ArtCNN en FP32, TRT-07).
- Opérateurs non gérés : refus clair.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (14 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (5 inclusions)

Utilisé par :

- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (39 inclusions)
- [48. Moteur Core ML (macOS)](48-coreml-macos.md) (4 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/ir/CMakeLists.txt` | 21 |
| `src/gpu/ir/describe.cpp` | 133 |
| `src/gpu/ir/describe.h` | 27 |
| `src/gpu/ir/fp8.h` | 61 |
| `src/gpu/ir/fp8_weights.cpp` | 60 |
| `src/gpu/ir/fp8_weights.h` | 25 |
| `src/gpu/ir/graph.cpp` | 216 |
| `src/gpu/ir/graph.h` | 62 |
| `src/gpu/ir/import.cpp` | 253 |
| `src/gpu/ir/import.h` | 27 |
| `src/gpu/ir/import_attrs.cpp` | 418 |
| `src/gpu/ir/import_attrs.h` | 30 |
| `src/gpu/ir/interpreter.cpp` | 103 |
| `src/gpu/ir/interpreter.h` | 27 |
| `src/gpu/ir/ir.h` | 267 |
| `src/gpu/ir/log.h` | 16 |
| `src/gpu/ir/memory_plan.cpp` | 121 |
| `src/gpu/ir/memory_plan.h` | 57 |
| `src/gpu/ir/onnx_fp8.cpp` | 151 |
| `src/gpu/ir/onnx_fp8.h` | 19 |
| `src/gpu/ir/onnx_info_main.cpp` | 185 |
| `src/gpu/ir/onnx_io_type.cpp` | 666 |
| `src/gpu/ir/onnx_io_type.h` | 64 |
| `src/gpu/ir/onnx_model.h` | 110 |
| `src/gpu/ir/onnx_reader.cpp` | 502 |
| `src/gpu/ir/onnx_unshuffle.cpp` | 298 |
| `src/gpu/ir/onnx_unshuffle.h` | 28 |
| `src/gpu/ir/op_elementwise.cpp` | 269 |
| `src/gpu/ir/op_impl.h` | 49 |
| `src/gpu/ir/op_pixel.cpp` | 166 |
| `src/gpu/ir/op_sample.cpp` | 257 |
| `src/gpu/ir/op_shape.cpp` | 287 |
| `src/gpu/ir/op_slice.cpp` | 339 |
| `src/gpu/ir/op_util.cpp` | 128 |
| `src/gpu/ir/op_util.h` | 89 |
| `src/gpu/ir/op_window.cpp` | 389 |
| `src/gpu/ir/ops.cpp` | 128 |
| `src/gpu/ir/ops.h` | 62 |
| `src/gpu/ir/pass_cast.cpp` | 81 |
| `src/gpu/ir/pass_fold.cpp` | 199 |
| `src/gpu/ir/pass_fuse.cpp` | 341 |
| `src/gpu/ir/pass_layout.cpp` | 159 |
| `src/gpu/ir/pass_pixel.cpp` | 142 |
| `src/gpu/ir/pass_quant.cpp` | 83 |
| `src/gpu/ir/pass_transpose.cpp` | 321 |
| `src/gpu/ir/passes.h` | 105 |
| `src/gpu/ir/protobuf.cpp` | 245 |
| `src/gpu/ir/protobuf.h` | 99 |
| `src/gpu/ir/shape.cpp` | 127 |
| `src/gpu/ir/shape.h` | 25 |

## Tests rattachés

- `tests/unit/gpu_ir/fp8_test.cpp`
- `tests/unit/gpu_ir/fuzz_test.cpp`
- `tests/unit/gpu_ir/import_test.cpp`
- `tests/unit/gpu_ir/onnx_builder.h`
- `tests/unit/gpu_ir/onnx_io_type_test.cpp`
- `tests/unit/gpu_ir/onnx_reader_test.cpp`
- `tests/unit/gpu_ir/onnx_unshuffle_test.cpp`
- `tests/unit/gpu_ir/ops_conv_test.cpp`
- `tests/unit/gpu_ir/ops_sample_test.cpp`
- `tests/unit/gpu_ir/ops_tensor_test.cpp`
- `tests/unit/gpu_ir/passes_test.cpp`
- `tests/unit/gpu_ir/property_test.cpp`
- `tests/unit/gpu_ir/protobuf_test.cpp`
- `tests/unit/gpu_ir/robustness_test.cpp`
- `tests/unit/gpu_ir/test_util.h`
- `tests/unit/gpu_ir/transpose_pass_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
