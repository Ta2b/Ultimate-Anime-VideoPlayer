# Audit 22. Réglages : modèle, application et persistance

**Audit par zone. Priorité : haute.** 15 fichiers, 2 767 lignes ; tests rattachés : 5 fichiers, 1 114 lignes.

## Mission

Les structures de réglages, ce qui s'applique en direct ou à la réouverture, la lecture et l'écriture de `play.toml` et de `settings.toml`, la lecture tolérante.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Chaque champ est appliqué ou enregistré (UX-01) ; `live_part` et `change_between` complets.
- Écritures : jamais à chaque frappe ni sur le thread de l'interface (UX-19).
- Fichier avec une valeur inconnue ou d'une ancienne version : rien de perdu.
- Accès concurrents aux fichiers de réglages.
- Valeurs par défaut cohérentes avec l'assistant et le README.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (13 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (5 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)
- [16. Audio](16-audio.md) (1 inclusion)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (14 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (6 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (5 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (5 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (5 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (4 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [28. CI, publication et signature](28-ci-publication.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)
- [20. Socle commun (core)](20-socle-core.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| UX-01 | Majeur | « Restore line art » et « Card for playback » manquent dans `live_part` et `change_between`. Ces réglages ne s'appliquent ni ne s'enregistrent jamais. |
| MED-17 | Mineur | Le délai de sous-titres passe à l'épisode suivant par la réouverture, sans message. |
| UX-18 | Mineur | Le curseur du délai des sous-titres va à ±5 s, les touches z et x jusqu'à ±30 s. |
| UX-20 | Mineur | Des textes sont formatés à l'ouverture de la chaîne et restent dans l'ancienne langue jusqu'à la vidéo suivante. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/host_choices.cpp` | 47 |
| `src/app/play/host_choices.h` | 18 |
| `src/app/play/latest_writer.cpp` | 78 |
| `src/app/play/latest_writer.h` | 46 |
| `src/app/play/live_settings.h` | 432 |
| `src/app/play/settings.cpp` | 484 |
| `src/app/play/settings.h` | 357 |
| `src/app/play/settings_merge.cpp` | 37 |
| `src/app/play/settings_toml.cpp` | 629 |
| `src/app/play/settings_toml_lenient.cpp` | 95 |
| `src/app/play/settings_toml_lenient.h` | 35 |
| `src/core/settings.cpp` | 289 |
| `src/core/settings.h` | 83 |
| `src/core/toml_config.h` | 26 |
| `src/core/toml_fields.h` | 111 |

## Tests rattachés

- `tests/unit/app/latest_writer_test.cpp`
- `tests/unit/app/live_settings_test.cpp`
- `tests/unit/app/settings_pages_test.cpp`
- `tests/unit/app/settings_test.cpp`
- `tests/unit/core/settings_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
