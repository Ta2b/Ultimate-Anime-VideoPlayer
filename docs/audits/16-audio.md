# Audit 16. Audio

**Audit par zone. Priorité : haute.** 24 fichiers, 2 111 lignes ; tests rattachés : 15 fichiers, 1 680 lignes.

## Mission

Décodage, conversion, sortie SDL, horloge audio, dérive et délai, changement d'appareil.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Horloge audio et synchronisation : source de vérité, dérive sur une longue lecture, recalage.
- Changement ou retrait d'appareil : délai appliqué, sortie par défaut absente (MED-08).
- Canaux : nombre de canaux figé par la première piste, downmix, changement de piste en cours.
- Changement de fréquence d'échantillonnage en cours de flux.
- Latence Bluetooth et export (MED-07).
- Windows (WASAPI) contre Linux (PipeWire) : mêmes résultats ?

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (21 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (5 inclusions)

Utilisé par :

- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (3 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (2 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-08 | Mineur | Au retrait d'un appareil, son délai reste appliqué aux haut-parleurs. Sans sortie par défaut, chaque bloc échoue au journal et rien n'est retenté. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/audio/CMakeLists.txt` | 7 |
| `src/audio/audio_thread.cpp` | 99 |
| `src/audio/audio_thread.h` | 81 |
| `src/audio/clock.h` | 28 |
| `src/audio/converter.cpp` | 162 |
| `src/audio/converter.h` | 54 |
| `src/audio/decoder.cpp` | 162 |
| `src/audio/decoder.h` | 52 |
| `src/audio/devices.cpp` | 60 |
| `src/audio/devices.h` | 28 |
| `src/audio/drift.cpp` | 38 |
| `src/audio/drift.h` | 41 |
| `src/audio/log.h` | 16 |
| `src/audio/output.cpp` | 141 |
| `src/audio/output.h` | 85 |
| `src/audio/packet.h` | 52 |
| `src/audio/pcm.cpp` | 53 |
| `src/audio/pcm.h` | 55 |
| `src/audio/player.cpp` | 380 |
| `src/audio/player.h` | 134 |
| `src/audio/sdl_sink.cpp` | 247 |
| `src/audio/sink.h` | 67 |
| `src/audio/timeline.cpp` | 29 |
| `src/audio/timeline.h` | 40 |

## Tests rattachés

- `tests/unit/app/decoder_options_test.cpp`
- `tests/unit/audio/clock_test.cpp`
- `tests/unit/audio/converter_test.cpp`
- `tests/unit/audio/decoder_test.cpp`
- `tests/unit/audio/drift_test.cpp`
- `tests/unit/audio/dummy_audio.h`
- `tests/unit/audio/fake_sink.h`
- `tests/unit/audio/output_test.cpp`
- `tests/unit/audio/pcm_test.cpp`
- `tests/unit/audio/rig.h`
- `tests/unit/audio/sine.cpp`
- `tests/unit/audio/sine.h`
- `tests/unit/core/log_test.cpp`
- `tests/unit/media/decoder_test.cpp`
- `tests/unit/media/packet_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
