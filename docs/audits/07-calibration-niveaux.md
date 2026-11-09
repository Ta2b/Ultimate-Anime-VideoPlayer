# Audit 07. Calibration, niveaux et mode Auto

**Audit par zone. Priorité : très haute.** 31 fichiers, 4 509 lignes ; tests rattachés : 12 fichiers, 2 308 lignes.

## Mission

Le choix automatique de la chaîne : table des cartes, niveaux Économe à Ultra, calibration de la machine, préparation FP8 et INT8 de SPAN, surveillance en lecture et bornes de l'Auto.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Chaîne appliquée identique à la chaîne mesurée, dans tous les résultats de calibration (TRT-01).
- FP8 jugé avant d'être activé, sur tous les écrans (TRT-08).
- Quatre niveaux distincts pour toutes les définitions de source (TRT-09).
- Table des cartes : portables, RTX A, cartes inconnues.
- Banc relancé pendant une lecture (TRT-19).
- Fiabilité de la mesure : durée, températures, horloges, batterie.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (22 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (8 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (6 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (5 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (4 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (4 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (4 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (3 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (1 inclusion)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (11 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (6 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (6 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (4 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-03 | Majeur | Passer la fenêtre sur un écran d'une autre hauteur rouvre toute la chaîne, sans hystérésis. Sous TensorRT, cela peut lancer une construction de plusieurs minutes. |
| TRT-01 | Majeur | L'Auto calibré passe d'Économe à Léger sans vérifier que SPAN tient. Une source 1080p est réduite à 480p ou 540p sans agrandissement, et RIFE tourne à une autre taille que celle mesurée. |
| TRT-08 | Mineur | Sur un écran 1080p, la calibration écrit `span_fp8 = true` sans l'avoir jugé et change le modèle. Elle peut aussi mesurer un FP8 non validé. |
| TRT-09 | Mineur | Les quatre niveaux ne donnent pas quatre chaînes pour une source de 720p ou moins : Qualité égale Ultra. |
| TRT-19 | Mineur | « Refaire le banc » mesure pendant que la chaîne garde mémoire et GPU. Les bornes de l'Auto calibré sortent trop basses. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `data/gpu_tiers.toml` | 104 |
| `src/app/play/auto_bench.cpp` | 211 |
| `src/app/play/auto_bench.h` | 100 |
| `src/app/play/auto_bench_run.cpp` | 431 |
| `src/app/play/auto_bench_run.h` | 140 |
| `src/app/play/auto_config.cpp` | 227 |
| `src/app/play/auto_config.h` | 77 |
| `src/app/play/auto_watch.cpp` | 40 |
| `src/app/play/auto_watch.h` | 52 |
| `src/app/play/bench_events.cpp` | 146 |
| `src/app/play/bench_events.h` | 32 |
| `src/app/play/bench_steps.cpp` | 200 |
| `src/app/play/bench_steps.h` | 119 |
| `src/app/play/gpu_profile_out.cpp` | 60 |
| `src/app/play/gpu_profile_out.h` | 39 |
| `src/app/play/gpu_tier.cpp` | 122 |
| `src/app/play/gpu_tier.h` | 58 |
| `src/app/play/machine_calibration.cpp` | 370 |
| `src/app/play/machine_calibration.h` | 122 |
| `src/app/play/settings_bench_results.cpp` | 192 |
| `src/app/play/settings_bench_results.h` | 41 |
| `src/app/play/span_derive.cpp` | 498 |
| `src/app/play/span_derive.h` | 165 |
| `src/app/play/span_int8.cpp` | 272 |
| `src/app/play/span_int8.h` | 134 |
| `src/core/auto_memory.cpp` | 90 |
| `src/core/auto_memory.h` | 49 |
| `src/core/cost_model.cpp` | 101 |
| `src/core/cost_model.h` | 51 |
| `src/media/bench_clip.cpp` | 238 |
| `src/media/bench_clip.h` | 28 |

## Tests rattachés

- `tests/unit/app/auto_bench_test.cpp`
- `tests/unit/app/auto_config_test.cpp`
- `tests/unit/app/auto_watch_test.cpp`
- `tests/unit/app/bench_steps_test.cpp`
- `tests/unit/app/gpu_tier_test.cpp`
- `tests/unit/app/machine_calibration_test.cpp`
- `tests/unit/app/span_derive_test.cpp`
- `tests/unit/app/span_int8_test.cpp`
- `tests/unit/backend_tensorrt/span_derive_test.cpp`
- `tests/unit/core/auto_memory_test.cpp`
- `tests/unit/core/cost_model_test.cpp`
- `tests/unit/media/bench_clip_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
