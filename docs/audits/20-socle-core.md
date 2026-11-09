# Audit 20. Socle commun (core)

**Audit par zone. Priorité : haute.** 33 fichiers, 3 336 lignes ; tests rattachés : 12 fichiers, 1 837 lignes.

## Mission

Les briques partagées par tout le lecteur : journal et son nettoyage, threads et leurs rôles, écritures atomiques, chargement de bibliothèques, texte et chemins UTF-8, types d'erreur (`Refusal`), boîtes aux lettres entre threads.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Écritures atomiques : fsync du fichier et du dossier, comportement sous Windows quand l'antivirus tient le fichier (QUA-05, QUA-07).
- Threads : `start_thread` et `start_task` attrapent-ils tout, et les threads détachés peuvent-ils survivre à la fermeture (QUA-13, WIN-10) ?
- Journal : rotation, débit maximal, nettoyage des chemins et du nom d'utilisateur, casse sous Windows (WIN-08, UX-13).
- `DynamicLibrary` : chemins absolus seulement, ordre de recherche des DLL (SEC-25).
- Conversions UTF-8 et chemins : toutes les sorties de `path::string()` sous Windows (QUA-06).
- Codes de refus : chaque code a-t-il un texte traduit, et aucun détail interne n'atteint-il l'utilisateur ?
- Code mort : modules sans appelant (`debug_flags`, `drawings`, `profile`, `user_overlay`, QUA-03).

## Frontières

Dépend de :

- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (2 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)

Utilisé par :

- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (42 inclusions)
- [47. Export vidéo](47-export.md) (27 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (24 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (22 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (21 inclusions)
- [16. Audio](16-audio.md) (21 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (20 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (19 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (18 inclusions)
- [17. Sous-titres](17-sous-titres.md) (17 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (16 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (16 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (15 inclusions)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (14 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (13 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (13 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (13 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (12 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (11 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (11 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (9 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (7 inclusions)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (7 inclusions)
- [48. Moteur Core ML (macOS)](48-coreml-macos.md) (7 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (6 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (6 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (5 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (4 inclusions)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (3 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (2 inclusions)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| QUA-07 | Majeur | Sous Windows, l'échec d'écriture de `update.toml` (antivirus) est ignoré. Le lancement compte comme un plantage, et la version peut être retirée après 3 lancements. |
| MED-14 | Mineur | `duration_to_us` peut rendre `AV_NOPTS_VALUE`, et `chain_dimension(1)` vaut 0. Fichier hostile seulement. |
| QUA-12 | Mineur | `core` inclut FFmpeg et contient du français en dur, qui atteint l'utilisateur sans `tr()`. |
| WIN-08 | Mineur | Le nettoyage du journal est sensible à la casse et ignore les noms courts. Le nom d'utilisateur fuit, et `/home/al` abîme `/home/alice`. |
| WIN-10 | Mineur | `start_task` passe par `std::async`, qui prend le pool ConcRT sous MSVC, sans nom ni rôle de thread. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/core/CMakeLists.txt` | 24 |
| `src/core/av_error.h` | 37 |
| `src/core/charconv.h` | 41 |
| `src/core/debug_flags.cpp` | 205 |
| `src/core/debug_flags.h` | 58 |
| `src/core/dir_entries.h` | 31 |
| `src/core/display.h` | 68 |
| `src/core/drawings.cpp` | 189 |
| `src/core/drawings.h` | 136 |
| `src/core/dynamic_library.cpp` | 85 |
| `src/core/dynamic_library.h` | 67 |
| `src/core/file_sync.cpp` | 100 |
| `src/core/file_sync.h` | 31 |
| `src/core/frame.h` | 167 |
| `src/core/gpu_device.h` | 117 |
| `src/core/inbox.h` | 122 |
| `src/core/log.cpp` | 179 |
| `src/core/log.h` | 63 |
| `src/core/log_redact.cpp` | 80 |
| `src/core/log_redact.h` | 22 |
| `src/core/messages.h` | 78 |
| `src/core/profile.cpp` | 364 |
| `src/core/profile.h` | 109 |
| `src/core/ring.h` | 114 |
| `src/core/text.h` | 48 |
| `src/core/thread.h` | 129 |
| `src/core/thread_role.cpp` | 79 |
| `src/core/thread_role.h` | 32 |
| `src/core/trace.h` | 20 |
| `src/core/types.h` | 137 |
| `src/core/user_overlay.cpp` | 284 |
| `src/core/user_overlay.h` | 95 |
| `src/core/utf8_path.h` | 25 |

## Tests rattachés

- `tests/unit/audio/thread_test.cpp`
- `tests/unit/core/contracts_test.cpp`
- `tests/unit/core/debug_flags_test.cpp`
- `tests/unit/core/drawings_test.cpp`
- `tests/unit/core/headers_test.cpp`
- `tests/unit/core/inbox_test.cpp`
- `tests/unit/core/log_module_test.cpp`
- `tests/unit/core/profile_format_test.cpp`
- `tests/unit/core/profile_test.cpp`
- `tests/unit/core/thread_test.cpp`
- `tests/unit/core/user_overlay_test.cpp`
- `tests/unit/subs/thread_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
