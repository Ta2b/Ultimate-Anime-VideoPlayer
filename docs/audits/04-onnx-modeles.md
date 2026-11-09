# Audit 04. Analyse et découpe des modèles ONNX

**Audit par zone. Priorité : très haute.** 21 fichiers, 2 978 lignes ; tests rattachés : 8 fichiers, 1 453 lignes.

## Mission

Lecture des fichiers ONNX livrés ou fournis, extraction des dimensions, des sorties et des métadonnées, découpe de RIFE et de DistilDRBA en morceaux.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Fichiers ONNX hostiles ou tronqués : lecteur protobuf, tailles, récursion (SEC-21).
- Découpe de RIFE et DRBA : graphe produit identique au réseau entier, tests numériques.
- Modèle fourni par `--model` : vérifications avant de lancer une construction.
- Chemins UTF-8 sous Windows.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (11 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (5 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (2 inclusions)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (2 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)

Utilisé par :

- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (9 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (7 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (5 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (4 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (3 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (2 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (2 inclusions)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/prepare/drba_split.cpp` | 535 |
| `src/prepare/drba_split.h` | 114 |
| `src/prepare/model_source.cpp` | 153 |
| `src/prepare/model_source.h` | 59 |
| `src/prepare/onnx_blend.cpp` | 135 |
| `src/prepare/onnx_blend.h` | 17 |
| `src/prepare/onnx_dims.cpp` | 175 |
| `src/prepare/onnx_dims.h` | 30 |
| `src/prepare/onnx_metadata.cpp` | 76 |
| `src/prepare/onnx_metadata.h` | 23 |
| `src/prepare/onnx_outputs.cpp` | 149 |
| `src/prepare/onnx_outputs.h` | 40 |
| `src/prepare/onnx_scan.cpp` | 304 |
| `src/prepare/onnx_scan.h` | 53 |
| `src/prepare/onnx_split.cpp` | 392 |
| `src/prepare/onnx_split.h` | 72 |
| `src/prepare/protobuf_fields.h` | 70 |
| `src/prepare/rife_split.cpp` | 419 |
| `src/prepare/rife_split.h` | 88 |
| `src/prepare/utf8.cpp` | 50 |
| `src/prepare/utf8.h` | 24 |

## Tests rattachés

- `tests/unit/gpu_ir/onnx_blend_test.cpp`
- `tests/unit/prepare/drba_split_test.cpp`
- `tests/unit/prepare/onnx_dims_test.cpp`
- `tests/unit/prepare/onnx_metadata_test.cpp`
- `tests/unit/prepare/onnx_outputs_test.cpp`
- `tests/unit/prepare/onnx_scan_test.cpp`
- `tests/unit/prepare/rife_split_test.cpp`
- `tests/unit/prepare/utf8_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
