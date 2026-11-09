# Audit 14. Sources, démultiplexage et erreurs d'E/S

**Audit par zone. Priorité : haute.** 26 fichiers, 2 440 lignes ; tests rattachés : 11 fichiers, 1 539 lignes.

## Mission

L'ouverture du fichier, la lecture des paquets, les nouvelles tentatives sur erreur, le choix des pistes, les chapitres.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Erreurs d'E/S : fin réelle contre panne durable (MED-01), seek pendant une panne (MED-06).
- Partages réseau : lectures bloquantes, fermeture bloquée.
- Choix des pistes et des langues, pistes ajoutées en cours.
- Chapitres ordonnés non gérés.
- Fichiers tronqués ou corrompus.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (21 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (3 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (1 inclusion)

Utilisé par :

- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (12 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (6 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (5 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (5 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (3 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (3 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (3 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [17. Sous-titres](17-sous-titres.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-01 | Majeur | Une erreur d'E/S de plus de 1,55 s devient une fin de flux, et l'épisode suivant démarre. La notice affiche « Playback goes on after an error ». |
| MED-06 | Mineur | Un seek pendant une panne d'E/S peut laisser la lecture figée, sans fin ni délai. Le rognage audio peut viser le seek manqué. |
| MED-10 | Mineur | P3.75 suppose une durée de piste que matroskadec ne remplit pas. En MKV, le gel de 1,5 s revient quand l'audio finit avant la vidéo. |
| MED-14 | Mineur | `duration_to_us` peut rendre `AV_NOPTS_VALUE`, et `chain_dimension(1)` vaut 0. Fichier hostile seulement. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/chapters.cpp` | 83 |
| `src/app/play/chapters.h` | 46 |
| `src/app/play/tracks.cpp` | 103 |
| `src/app/play/tracks.h` | 51 |
| `src/media/CMakeLists.txt` | 20 |
| `src/media/av_util.cpp` | 53 |
| `src/media/av_util.h` | 63 |
| `src/media/demuxer.cpp` | 460 |
| `src/media/demuxer.h` | 83 |
| `src/media/language.cpp` | 108 |
| `src/media/language.h` | 21 |
| `src/media/log.h` | 16 |
| `src/media/packet.cpp` | 114 |
| `src/media/packet.h` | 105 |
| `src/media/packet_reserve.cpp` | 55 |
| `src/media/packet_reserve.h` | 54 |
| `src/media/selection.cpp` | 166 |
| `src/media/selection.h` | 53 |
| `src/media/source.cpp` | 78 |
| `src/media/source.h` | 127 |
| `src/media/source_loop.cpp` | 225 |
| `src/media/source_loop.h` | 56 |
| `src/media/track.cpp` | 67 |
| `src/media/track.h` | 91 |
| `src/media/track_language.cpp` | 121 |
| `src/media/track_language.h` | 21 |

## Tests rattachés

- `tests/unit/app/chapters_test.cpp`
- `tests/unit/app/tracks_test.cpp`
- `tests/unit/media/av_util_test.cpp`
- `tests/unit/media/demuxer_test.cpp`
- `tests/unit/media/language_test.cpp`
- `tests/unit/media/packet_reserve_test.cpp`
- `tests/unit/media/selection_test.cpp`
- `tests/unit/media/source_harness.h`
- `tests/unit/media/source_test.cpp`
- `tests/unit/media/track_language_test.cpp`
- `tests/unit/subs/track_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
