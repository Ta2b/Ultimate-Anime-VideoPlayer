# Audit 11. Écran et matériel : EDID, VRR, HDR du système, télémétrie

**Audit par zone. Priorité : haute.** 13 fichiers, 2 079 lignes ; tests rattachés : 4 fichiers, 411 lignes.

## Mission

Ce que le lecteur sait de l'écran et de la carte : EDID, VRR armé (NVAPI, `VRR_ENABLED`), état HDR et pic du système, contrat d'écran de KDE, télémétrie GPU.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- VRR : cas sans plage EDID, plein écran, cache global (GPU-01, GPU-02, GPU-05, GPU-18).
- Multi-écran : tout suit l'écran de la fenêtre (GPU-04).
- Requêtes sur le thread de rendu sous Windows (GPU-16, WIN-04).
- Bascule HDR du système suivie avec retard (GPU-13).
- Télémétrie : courses de données (GPU-09).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (7 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (3 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (3 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (10 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-04 | Majeur | Sous KDE en multi-écran, le contrat d'écran garde l'EDID et le connecteur de l'écran d'ouverture. Blanc, MaxFALL, primaires et noir visent le mauvais écran. |
| WIN-01 | Majeur | Le VRR n'est lu que par NVAPI. Le mode Rapide reste impossible sur Radeon et Intel sous Windows. |
| WIN-04 | Majeur | Les requêtes d'affichage restent sur le thread principal, en double toutes les 500 ms, et la fabrique DXGI est recréée toutes les 2 s. Majeur à 144-165 Hz (voir GPU-16). |
| GPU-02 | Mineur | Cache VRR global de 250 ms, et la branche plein écran pose `vrr_armed` sans mettre à jour `vrr_watch`. Un VRR désengagé peut rester vu armé. |
| GPU-09 | Mineur | `GpuTelemetry::sample` est écrit par deux threads sans verrou en fin de chaîne. Comportement indéfini. |
| GPU-16 | Mineur | SetupDi, registre, DXGI, `QueryDisplayConfig` et NVAPI sont appelés sur le thread de rendu toutes les 500 ms ou 2 s. Majeur à 144-165 Hz si confirmé (voir WIN-04). |
| GPU-18 | Mineur | Sous X11 avec le pilote NVIDIA, l'absence de `VRR_ENABLED` donne « non armé » au lieu de « inconnu ». |
| QUA-16 | Mineur | Restes : statiques mutables, `strtoul` sans contrôle, 172 traductions orphelines, CHANGELOG sans 0.10.1 ni 0.10.2, `moving_area` sans effet. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/display_contract.cpp` | 90 |
| `src/app/play/display_contract.h` | 61 |
| `src/app/play/display_edid.cpp` | 440 |
| `src/app/play/display_edid.h` | 80 |
| `src/app/play/gpu_telemetry.cpp` | 388 |
| `src/app/play/gpu_telemetry.h` | 91 |
| `src/app/play/host_facts.cpp` | 382 |
| `src/app/play/host_facts.h` | 72 |
| `src/app/play/system_peak.cpp` | 232 |
| `src/app/play/system_peak.h` | 36 |
| `src/app/play/vrr_watch.h` | 51 |
| `src/app/play/window_size.h` | 36 |
| `src/app/play/window_vrr.cpp` | 120 |

## Tests rattachés

- `tests/unit/app/display_edid_test.cpp`
- `tests/unit/app/gpu_telemetry_test.cpp`
- `tests/unit/app/vrr_watch_test.cpp`
- `tests/unit/app/window_size_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
