# Audit 13. Session de lecture : seek, épisodes et reprise

**Audit par zone. Priorité : haute.** 17 fichiers, 2 512 lignes ; tests rattachés : 13 fichiers, 1 938 lignes.

## Mission

La session de lecture : file d'images, choix de l'image à présenter, seek et image par image, statistiques, épisode suivant, reprise de position.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Seek et pas d'image en VFR (MED-16), seek pendant une panne d'E/S (MED-06).
- Épisode suivant déclenché seulement par une vraie fin (MED-01).
- Reprise : `resume.tsv`, écriture atomique, nombre d'entrées.
- File d'images : bornes, images jetées au seek, coût.
- Concurrence entre la session, le décodage et la présentation.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (16 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (6 inclusions)
- [17. Sous-titres](17-sous-titres.md) (6 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (5 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (4 inclusions)
- [16. Audio](16-audio.md) (3 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (1 inclusion)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (1 inclusion)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (10 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (4 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (4 inclusions)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-19 | Mineur | La vitesse est calée sur la fréquence de l'écran même en VRR. À 144 Hz et 47,95 i/s, la lecture va 0,11 % trop vite. |
| MED-10 | Mineur | P3.75 suppose une durée de piste que matroskadec ne remplit pas. En MKV, le gel de 1,5 s revient quand l'audio finit avant la vidéo. |
| MED-15 | Mineur | `Pipeline::open`, la recherche des sous-titres externes et `sibling_episode` tournent sur le thread de la fenêtre. La fenêtre se fige sur un NAS lent. |
| MED-16 | Mineur | En VFR, le seek exact et le pas d'image utilisent la période moyenne. La cible peut sauter l'image visible, ou le pas reculer de deux images. |
| UX-10 | Mineur | Le point décimal devient une virgule sans condition de langue. L'anglais affiche « 23,976 fps ». |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/episodes.cpp` | 172 |
| `src/app/play/episodes.h` | 32 |
| `src/app/play/frame_latency.h` | 59 |
| `src/app/play/frame_queue.cpp` | 59 |
| `src/app/play/frame_queue.h` | 57 |
| `src/app/play/frame_stats.cpp` | 109 |
| `src/app/play/frame_stats.h` | 80 |
| `src/app/play/pipeline.cpp` | 370 |
| `src/app/play/pipeline.h` | 146 |
| `src/app/play/present_stats.cpp` | 66 |
| `src/app/play/present_stats.h` | 18 |
| `src/app/play/resume.cpp` | 131 |
| `src/app/play/resume.h` | 38 |
| `src/app/play/session.cpp` | 779 |
| `src/app/play/session.h` | 318 |
| `src/app/play/video_rate.h` | 38 |
| `src/app/play/video_stage.h` | 40 |

## Tests rattachés

- `tests/unit/app/e2e_play_test.cpp`
- `tests/unit/app/episodes_test.cpp`
- `tests/unit/app/frame_latency_test.cpp`
- `tests/unit/app/pipeline_test.cpp`
- `tests/unit/app/present_stats_test.cpp`
- `tests/unit/app/resume_test.cpp`
- `tests/unit/app/session_period_test.cpp`
- `tests/unit/app/session_refusal_test.cpp`
- `tests/unit/app/session_test.cpp`
- `tests/unit/app/session_watch_test.cpp`
- `tests/unit/app/video_rate_test.cpp`
- `tests/unit/backend_tensorrt/frame_stats_test.cpp`
- `tests/unit/kernels/frame_stats_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
