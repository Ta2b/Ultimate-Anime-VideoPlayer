# Audit 48. Moteur Core ML (macOS)

**Audit par zone. Priorité : hors cible.** 23 fichiers, 4 008 lignes ; tests rattachés : 4 fichiers, 521 lignes.

**Hors cible pour l'instant : ne pas auditer.** Ce code reste dans le lecteur tel quel, ni supprimé ni caché ; ses bugs et son optimisation viendront plus tard.

## Mission

Le moteur de calcul de macOS. Hors de la cible actuelle (Windows et CachyOS sur NVIDIA).

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- **Hors cible pour l'instant : ne pas auditer.**

## Frontières

Dépend de :

- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (12 inclusions)
- [20. Socle commun (core)](20-socle-core.md) (7 inclusions)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (4 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (2 inclusions)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/coreml/CMakeLists.txt` | 43 |
| `src/gpu/coreml/coreml_compiled.cpp` | 38 |
| `src/gpu/coreml/coreml_compiled.h` | 21 |
| `src/gpu/coreml/coreml_detail.h` | 101 |
| `src/gpu/coreml/coreml_direct.h` | 69 |
| `src/gpu/coreml/coreml_direct.mm` | 612 |
| `src/gpu/coreml/coreml_direct_none.cpp` | 29 |
| `src/gpu/coreml/coreml_measure.cpp` | 301 |
| `src/gpu/coreml/coreml_runtime.cpp` | 692 |
| `src/gpu/coreml/coreml_runtime.h` | 21 |
| `src/gpu/coreml/coreml_steps.h` | 19 |
| `src/gpu/coreml/coreml_surfaces.h` | 40 |
| `src/gpu/coreml/coreml_units.h` | 20 |
| `src/gpu/coreml/coreml_vt.h` | 41 |
| `src/gpu/coreml/coreml_vt.mm` | 521 |
| `src/gpu/coreml/coreml_vt_none.cpp` | 28 |
| `src/gpu/coreml/entry.cpp` | 14 |
| `src/gpu/coreml/vt_buffers.h` | 103 |
| `src/gpu/coreml/vt_buffers.mm` | 266 |
| `src/gpu/coreml/vt_gpu.h` | 57 |
| `src/gpu/coreml/vt_gpu.mm` | 576 |
| `src/gpu/coreml/vt_pixels.cpp` | 319 |
| `src/gpu/coreml/vt_pixels.h` | 77 |

## Tests rattachés

- `tests/unit/coreml_rt/coreml_rt_test.cpp`
- `tests/unit/gpu/coreml_compiled_test.cpp`
- `tests/unit/gpu/vt_pixels_test.cpp`
- `tests/unit/vk_backend/entry_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
