# Audit 21. Frontière des moteurs (interface et ABI)

**Audit par zone. Priorité : haute.** 13 fichiers, 1 606 lignes ; tests rattachés : 13 fichiers, 2 595 lignes.

## Mission

Le contrat entre le lecteur et les modules de calcul chargés dynamiquement (Vulkan, TensorRT, Core ML) : interface `Backend`, tenseurs, voies, mandataire `GuardedBackend`, empreinte d'ABI.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Empreinte d'ABI : couvre-t-elle tout ce qui change la frontière ?
- `GuardedBackend` : toutes les méthodes protégées, aucune exception ne traverse.
- Capacités annoncées (`BackendCaps`) : chaque champ est-il vrai pour chaque moteur (FP8 Vulkan absent, par exemple) ?
- Chargement : chemin absolu, message quand le module manque ou ne correspond pas (TRT-03).
- Durée de vie des tenseurs et des voies à la fermeture d'un moteur.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (11 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (1 inclusion)

Utilisé par :

- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (16 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (9 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (9 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (4 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (4 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (4 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (2 inclusions)
- [48. Moteur Core ML (macOS)](48-coreml-macos.md) (2 inclusions)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (2 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (2 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (1 inclusion)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `cmake/BoundaryHash.cmake` | 24 |
| `src/app/play/backend_library.cpp` | 134 |
| `src/app/play/backend_library.h` | 69 |
| `src/gpu/CMakeLists.txt` | 28 |
| `src/gpu/backend.h` | 417 |
| `src/gpu/gpu_profile.h` | 48 |
| `src/gpu/graph_book.h` | 132 |
| `src/gpu/guarded_backend.h` | 230 |
| `src/gpu/kernel_ops.h` | 291 |
| `src/gpu/lanes.h` | 118 |
| `src/gpu/module_logs.h` | 41 |
| `src/gpu/system_interpolator.h` | 14 |
| `src/gpu/tensor.h` | 60 |

## Tests rattachés

- `tests/unit/backend_tensorrt/backend_fixture.cpp`
- `tests/unit/backend_tensorrt/backend_fixture.h`
- `tests/unit/backend_tensorrt/backend_frames.h`
- `tests/unit/backend_tensorrt/backend_motion_test.cpp`
- `tests/unit/backend_tensorrt/backend_test.cpp`
- `tests/unit/gpu/backend_test.cpp`
- `tests/unit/gpu/gpu_profile_test.cpp`
- `tests/unit/gpu/graph_book_test.cpp`
- `tests/unit/gpu/lanes_test.cpp`
- `tests/unit/vk_backend/backend_rig.cpp`
- `tests/unit/vk_backend/backend_rig.h`
- `tests/unit/vk_backend/backend_test.cpp`
- `tests/unit/vk_backend/kernel_ops_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
