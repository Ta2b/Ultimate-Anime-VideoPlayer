# Audit 42. Analyse des plans, coupes et pré-analyse

**Audit par zone. Priorité : moyenne.** 17 fichiers, 1 768 lignes ; tests rattachés : 5 fichiers, 477 lignes.

## Mission

La détection des changements de plan, des dessins tenus et des flashs, la pré-analyse du fichier en arrière-plan et son cache, qui pilotent l'interpolation.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Pré-analyse : coût CPU, priorité sous Windows (un seul thread) et sous Linux (WIN-11).
- Cache `.shots` : écriture atomique, validation à la lecture, fichier corrompu (QUA-05).
- Seuils réglés à l'œil sur un épisode (`kPlanCutFloor`, `rife_plan.h`) : sont-ils justifiés ou testés ?
- Limiteur de flashs : effet sur les métadonnées HDR10 (GPU-07).
- Cohérence entre la pré-analyse et les statistiques calculées sur le GPU à l'ingestion.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (6 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (3 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (2 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| QUA-04 | Majeur | Les résultats ignorés ne sont vérifiés ni sous Windows, ni hors des PR. Les sites relevés le 06/11 sont intacts. |
| QUA-05 | Majeur | Six écritures restent faites à la main, sans fsync, contre P5.11. Une coupure peut laisser un moteur Vulkan ou un cache vide. |
| WIN-11 | Mineur | Écarts inchangés : pré-analyse sur un thread, sémaphores non partagés, `timeBeginPeriod(1)` global, manifeste sans `<supportedOS>`, veille jamais suspendue, télémétrie NVIDIA seule. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/anime_shots.cpp` | 61 |
| `src/app/play/anime_shots.h` | 50 |
| `src/app/play/chain_shots.cpp` | 216 |
| `src/app/play/chain_shots.h` | 79 |
| `src/app/play/cut_book.cpp` | 46 |
| `src/app/play/cut_book.h` | 35 |
| `src/media/flash_guard.h` | 62 |
| `src/media/live_scenes.cpp` | 64 |
| `src/media/live_scenes.h` | 55 |
| `src/media/shot_cache.cpp` | 178 |
| `src/media/shot_cache.h` | 34 |
| `src/media/shot_cutter.cpp` | 224 |
| `src/media/shot_cutter.h` | 118 |
| `src/media/shot_probe.cpp` | 274 |
| `src/media/shot_probe.h` | 84 |
| `src/media/shot_timeline.cpp` | 136 |
| `src/media/shot_timeline.h` | 52 |

## Tests rattachés

- `tests/unit/app/anime_shots_test.cpp`
- `tests/unit/app/chain_shots_test.cpp`
- `tests/unit/app/cut_book_test.cpp`
- `tests/unit/media/flash_guard_test.cpp`
- `tests/unit/media/shot_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
