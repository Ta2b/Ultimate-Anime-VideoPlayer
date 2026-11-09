# Audit 17. Sous-titres

**Audit par zone. Priorité : haute.** 30 fichiers, 2 598 lignes ; tests rattachés : 11 fichiers, 1 718 lignes.

## Mission

Lecture des pistes ASS, SRT et des fichiers externes, détection d'encodage, rendu par libass, atlas, délai.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Rendu 4K : réduction de l'atlas et retour à la pleine définition (MED-05).
- Délai négatif sur une piste intégrée (MED-09), délai conservé d'un épisode à l'autre (MED-17).
- Encodages : Big5, GBK, EUC-KR, UTF-16 sans BOM ; différences Windows et Linux (MED-11, WIN-09).
- PGS et VobSub : non gérés, et ils empêchent le repli sur un fichier externe.
- Coût : libass appelé à chaque image de sortie au lieu de chaque événement.
- Polices jointes : fichiers hostiles, taille, mémoire.
- Seek : pré-lecture des sous-titres autour de la nouvelle position.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (17 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (1 inclusion)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (1 inclusion)

Utilisé par :

- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (6 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (5 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (5 inclusions)
- [28. CI, publication et signature](28-ci-publication.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-05 | Mineur | En 4K, la réduction des sous-titres reste active tant que l'écran n'est pas vide. Les dialogues suivants sont flous, parfois pendant des minutes. |
| MED-09 | Mineur | Un délai de sous-titres négatif ne montre rien plus tôt sur une piste intégrée. Chaque événement arrive à son heure d'origine, amputé. |
| MED-11 | Mineur | « zh » impose GBK, et un SRT Big5 sort en charabia sans repli. Pas d'EUC-KR, d'EUC-JP ni d'UTF-16 sans BOM ; les demi-paires passent. |
| WIN-09 | Mineur | Big5 est décodé en CP950 sous Windows et refusé par iconv sous Linux, qui affiche du charabia. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/external_subtitles.cpp` | 128 |
| `src/app/play/external_subtitles.h` | 27 |
| `src/app/play/subtitle_queue.cpp` | 51 |
| `src/app/play/subtitle_queue.h` | 43 |
| `src/subs/CMakeLists.txt` | 13 |
| `src/subs/ass_library.cpp` | 65 |
| `src/subs/ass_library.h` | 45 |
| `src/subs/atlas_packer.cpp` | 94 |
| `src/subs/atlas_packer.h` | 40 |
| `src/subs/atlas_pool.cpp` | 33 |
| `src/subs/atlas_pool.h` | 41 |
| `src/subs/log.h` | 16 |
| `src/subs/srt.cpp` | 183 |
| `src/subs/srt.h` | 44 |
| `src/subs/srt_markup.cpp` | 178 |
| `src/subs/subs_messages.h` | 70 |
| `src/subs/subs_thread.cpp` | 63 |
| `src/subs/subs_thread.h` | 37 |
| `src/subs/subs_worker.cpp` | 163 |
| `src/subs/subs_worker.h` | 72 |
| `src/subs/subtitle_frame.h` | 98 |
| `src/subs/subtitle_renderer.cpp` | 318 |
| `src/subs/subtitle_renderer.h` | 140 |
| `src/subs/subtitle_track.cpp` | 180 |
| `src/subs/subtitle_track.h` | 85 |
| `src/subs/text_encoding.cpp` | 219 |
| `src/subs/text_encoding.h` | 16 |
| `src/subs/timebase.h` | 27 |
| `src/subs/vsfilter_color.cpp` | 67 |
| `src/subs/vsfilter_color.h` | 42 |

## Tests rattachés

- `tests/unit/app/external_subtitles_test.cpp`
- `tests/unit/app/subtitle_queue_test.cpp`
- `tests/unit/subs/atlas_test.cpp`
- `tests/unit/subs/compat_test.cpp`
- `tests/unit/subs/parity_test.cpp`
- `tests/unit/subs/renderer_test.cpp`
- `tests/unit/subs/srt_test.cpp`
- `tests/unit/subs/support.cpp`
- `tests/unit/subs/support.h`
- `tests/unit/subs/text_encoding_test.cpp`
- `tests/unit/subs/worker_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
