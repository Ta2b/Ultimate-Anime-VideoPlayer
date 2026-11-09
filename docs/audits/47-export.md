# Audit 47. Export vidéo

**Audit par zone. Priorité : hors cible.** 24 fichiers, 4 060 lignes ; tests rattachés : 5 fichiers, 1 100 lignes.

**Hors cible pour l'instant : ne pas auditer.** Ce code reste dans le lecteur tel quel, ni supprimé ni caché ; ses bugs et son optimisation viendront plus tard.

## Mission

Le rendu d'une vidéo agrandie et interpolée dans un fichier : encodeurs, conteneurs, sous-titres et polices, audio, annulation.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Sous-titres et polices perdus en MP4 et MOV.
- Délai audio de l'appareil gravé dans le fichier (MED-07).
- Confirmation à la fermeture et annulation (MED-12).
- Écriture sûre du fichier final, place disque.
- **Hors cible pour l'instant : ne pas auditer.** Les points ci-dessus servent le jour où l'export revient.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (27 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (3 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (2 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (1 inclusion)

Utilisé par :

- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (2 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-07 | Mineur | L'export grave la compensation de latence de l'appareil, par exemple +200 ms en Bluetooth. Le fichier est désynchronisé sur tout autre appareil. Mineur à Majeur selon le rapport. |
| MED-12 | Mineur | La confirmation peut s'afficher deux fois et suspend l'export. Sans boîte possible (Wayland sans zenity), la croix ne ferme plus ; un `SIGTERM` attend un clic. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/export_depth.h` | 43 |
| `src/app/play/export_job.cpp` | 134 |
| `src/app/play/export_job.h` | 65 |
| `src/app/play/export_run.cpp` | 453 |
| `src/app/play/export_run.h` | 34 |
| `src/app/play/export_sink.cpp` | 189 |
| `src/app/play/export_sink.h` | 115 |
| `src/app/play/placebo_export.cpp` | 452 |
| `src/app/play/placebo_export.h` | 183 |
| `src/app/play/placebo_export_vulkan.cpp` | 204 |
| `src/export/CMakeLists.txt` | 6 |
| `src/export/av_handles.cpp` | 207 |
| `src/export/av_handles.h` | 73 |
| `src/export/container.cpp` | 61 |
| `src/export/container.h` | 42 |
| `src/export/encoder_choice.cpp` | 260 |
| `src/export/encoder_choice.h` | 75 |
| `src/export/export_writer.cpp` | 304 |
| `src/export/export_writer.h` | 186 |
| `src/export/muxer.cpp` | 420 |
| `src/export/muxer.h` | 83 |
| `src/export/owned_frame.h` | 15 |
| `src/export/video_encoder.cpp` | 360 |
| `src/export/video_encoder.h` | 96 |

## Tests rattachés

- `tests/unit/app/export_job_test.cpp`
- `tests/unit/app/export_sink_test.cpp`
- `tests/unit/export/container_test.cpp`
- `tests/unit/export/encoder_choice_test.cpp`
- `tests/unit/export/export_writer_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
