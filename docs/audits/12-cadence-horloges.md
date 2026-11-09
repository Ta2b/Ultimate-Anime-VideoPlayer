# Audit 12. Cadence et horloges

**Audit par zone. Priorité : haute.** 27 fichiers, 2 522 lignes ; tests rattachés : 9 fichiers, 1 670 lignes.

## Mission

Les horloges et la cadence : grille de vsync apprise, rythme fixe ou libre, plancher VRR, choix de l'instant de présentation, calcul du facteur d'interpolation.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- 23,976 et 29,97 i/s sur 60, 144 et 165 Hz, en VRR et sans, sur une heure (GPU-19).
- Grille apprise quand le retour de présentation est pauvre (Windows).
- Plancher VRR recalculé au changement d'écran (GPU-05).
- Dérive numérique sur une longue lecture.
- Tests déterministes avec `FakeClock`.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (6 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (2 inclusions)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (11 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (9 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (6 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (6 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (5 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (5 inclusions)
- [16. Audio](16-audio.md) (5 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (2 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)
- [20. Socle commun (core)](20-socle-core.md) (1 inclusion)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-19 | Mineur | La vitesse est calée sur la fréquence de l'écran même en VRR. À 144 Hz et 47,95 i/s, la lecture va 0,11 % trop vite. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/core/auto_rate.cpp` | 398 |
| `src/core/auto_rate.h` | 250 |
| `src/core/cadence.cpp` | 121 |
| `src/core/cadence.h` | 117 |
| `src/core/clock.h` | 41 |
| `src/core/clock_pll.h` | 48 |
| `src/core/free_grid.cpp` | 106 |
| `src/core/free_grid.h` | 66 |
| `src/core/free_rate.cpp` | 112 |
| `src/core/free_rate.h` | 132 |
| `src/core/moving_area.cpp` | 28 |
| `src/core/moving_area.h` | 59 |
| `src/core/pacing.cpp` | 69 |
| `src/core/pacing.h` | 38 |
| `src/core/present_ledger.cpp` | 60 |
| `src/core/present_ledger.h` | 102 |
| `src/core/steady_grid.cpp` | 85 |
| `src/core/steady_grid.h` | 46 |
| `src/core/steady_rate.cpp` | 216 |
| `src/core/steady_rate.h` | 132 |
| `src/core/vblank_clock.cpp` | 83 |
| `src/core/vblank_clock.h` | 58 |
| `src/core/vrr_floor.h` | 25 |
| `src/core/vsync_index.h` | 41 |
| `src/core/vsync_phase.cpp` | 35 |
| `src/core/vsync_phase.h` | 31 |
| `src/core/wait_plan.h` | 23 |

## Tests rattachés

- `tests/unit/core/auto_rate_test.cpp`
- `tests/unit/core/cadence_test.cpp`
- `tests/unit/core/moving_area_test.cpp`
- `tests/unit/core/pacing_test.cpp`
- `tests/unit/core/present_ledger_test.cpp`
- `tests/unit/core/present_test.cpp`
- `tests/unit/core/steady_grid_test.cpp`
- `tests/unit/core/steady_rate_test.cpp`
- `tests/unit/core/vrr_floor_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
