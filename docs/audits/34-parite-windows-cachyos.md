# Audit 34. Parité Windows et CachyOS

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Que les deux systèmes de la cible aient au moins les mêmes performances et la même stabilité, chacun avec ses voies natives quand elles sont meilleures.

## Méthode

1. Lister tous les blocs propres à un système (`_WIN32`, `__linux__`, `__APPLE__`) et les comportements qu'ils changent.
2. Pour chacun : la voie Windows et la voie Linux sont-elles équivalentes ? Laquelle est native ?
3. Reprendre le tableau des 17 voies natives du contre-audit et son état.

## À couvrir

- Présentation, VRR, HDR, multi-écran.
- Décodage matériel, pré-analyse, priorités et minuteries.
- Téléchargements, proxy, dossiers de données.
- Instance unique, lecteur par défaut, minidumps.
- Encodages de texte (Big5).

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Voies propres à un système, déjà connues

| Point | Où | Zone |
|---|---|---|
| VRR par NVAPI | `src/app/play/window_vrr.cpp:79` | [11](11-ecran-materiel.md) |
| mode d'affichage Windows | `src/app/play/host_facts.cpp:39` | [11](11-ecran-materiel.md) |
| verrou de fichier | `src/prepare/key_lock.cpp:38` | [05](05-construction-moteurs.md) |
| lancement de processus Windows | `src/prepare/process_win.cpp:153` | [05](05-construction-moteurs.md) |
| instance unique : canal nommé | `src/app/play/single_instance.cpp:111` | [19](19-integration-systeme.md) |
| instance unique : socket Unix | `src/app/play/single_instance.cpp:270` | [19](19-integration-systeme.md) |
| minidumps | `src/app/play/crash_dump.cpp:80` | [19](19-integration-systeme.md) |
| migration des données | `src/app/play/data_dirs.cpp:67` | [19](19-integration-systeme.md) |
| synchronisation d'un dossier | `src/core/file_sync.cpp:44` | [20](20-socle-core.md) |

### Tous les fichiers avec du code propre à un système

48 fichiers de la cible. Les blocs `__APPLE__` sont hors cible : vérifier seulement qu'ils ne changent rien sous Windows et CachyOS.

| Fichier | Blocs `#if` | Zone |
|---|---|---|
| `src/gpu/cuda/backend/entry.cpp` | 3 | [01](01-tensorrt-execution.md) |
| `src/gpu/cuda/backend/ring_slot.cpp` | 4 | [01](01-tensorrt-execution.md) |
| `src/gpu/cuda/backend/tensorrt_backend.cpp` | 4 | [01](01-tensorrt-execution.md) |
| `src/prepare/cache_io.cpp` | 2 | [05](05-construction-moteurs.md) |
| `src/prepare/key_lock.cpp` | 2 | [05](05-construction-moteurs.md) |
| `src/prepare/process_posix.cpp` | 3 | [05](05-construction-moteurs.md) |
| `src/prepare/process_text.cpp` | 1 | [05](05-construction-moteurs.md) |
| `src/prepare/process_win.cpp` | 1 | [05](05-construction-moteurs.md) |
| `src/app/play/engine_tools.cpp` | 3 | [06](06-installation-moteurs.md) |
| `src/core/engines.h` | 2 | [06](06-installation-moteurs.md) |
| `src/core/sha256_blocks.cpp` | 3 | [06](06-installation-moteurs.md) |
| `src/app/play/auto_bench_run.cpp` | 1 | [07](07-calibration-niveaux.md) |
| `src/app/play/chain_setup.cpp` | 3 | [08](08-boucle-chaine.md) |
| `src/app/play/placebo_impl.h` | 1 | [10](10-presentation-libplacebo.md) |
| `src/app/play/placebo_open.cpp` | 6 | [10](10-presentation-libplacebo.md) |
| `src/app/play/placebo_ring.cpp` | 2 | [10](10-presentation-libplacebo.md) |
| `src/app/play/vulkan_share.cpp` | 4 | [10](10-presentation-libplacebo.md) |
| `src/app/play/display_edid.cpp` | 7 | [11](11-ecran-materiel.md) |
| `src/app/play/gpu_telemetry.cpp` | 6 | [11](11-ecran-materiel.md) |
| `src/app/play/host_facts.cpp` | 8 | [11](11-ecran-materiel.md) |
| `src/app/play/system_peak.cpp` | 5 | [11](11-ecran-materiel.md) |
| `src/app/play/window_vrr.cpp` | 4 | [11](11-ecran-materiel.md) |
| `src/app/play/resume.cpp` | 1 | [13](13-session-lecture.md) |
| `src/media/cuda_driver.cpp` | 5 | [15](15-decodage-formats.md) |
| `src/media/hw_device.cpp` | 5 | [15](15-decodage-formats.md) |
| `src/subs/text_encoding.cpp` | 2 | [17](17-sous-titres.md) |
| `src/app/main.cpp` | 9 | [18](18-demarrage-arret.md) |
| `src/app/play/loop_wait.cpp` | 2 | [18](18-demarrage-arret.md) |
| `src/app/play/mapped_file.cpp` | 3 | [18](18-demarrage-arret.md) |
| `src/app/play/options.h` | 1 | [18](18-demarrage-arret.md) |
| `src/app/play/crash_dump.cpp` | 4 | [19](19-integration-systeme.md) |
| `src/app/play/data_dirs.cpp` | 2 | [19](19-integration-systeme.md) |
| `src/app/play/default_player.cpp` | 11 | [19](19-integration-systeme.md) |
| `src/app/play/single_instance.cpp` | 2 | [19](19-integration-systeme.md) |
| `src/core/dynamic_library.cpp` | 4 | [20](20-socle-core.md) |
| `src/core/file_sync.cpp` | 3 | [20](20-socle-core.md) |
| `src/core/thread_role.cpp` | 7 | [20](20-socle-core.md) |
| `src/gpu/backend.h` | 1 | [21](21-frontiere-moteurs.md) |
| `src/app/play/settings.h` | 1 | [22](22-reglages-modele.md) |
| `src/app/play/settings_panel.cpp` | 1 | [23](23-reglages-panneau.md) |
| `src/app/play/settings_system.cpp` | 3 | [23](23-reglages-panneau.md) |
| `src/app/play/ui.cpp` | 2 | [24](24-interface-rendu.md) |
| `src/app/play/updater.cpp` | 4 | [26](26-mise-a-jour.md) |
| `src/app/play/placebo_grade.cpp` | 1 | [41](41-hdr-grade.md) |
| `src/media/shot_probe.cpp` | 1 | [42](42-plans-coupes.md) |
| `src/gpu/vulkan/backend/shared_tensor.cpp` | 3 | [45](45-backend-vulkan.md) |
| `src/gpu/vulkan/backend/vulkan_backend_runtime.cpp` | 2 | [45](45-backend-vulkan.md) |
| `src/gpu/vulkan/compute/library.cpp` | 5 | [45](45-backend-vulkan.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| WIN-02 | Majeur | Avec un profil redirigé, la migration copie plus de 1 Go sur le thread principal, avant la fenêtre. Le lecteur semble ne pas démarrer pendant des minutes. |
| WIN-04 | Majeur | Les requêtes d'affichage restent sur le thread principal, en double toutes les 500 ms, et la fabrique DXGI est recréée toutes les 2 s. Majeur à 144-165 Hz (voir GPU-16). |
| WIN-07 | Majeur | Les minidumps sont sans PDB et ne voient ni `__fastfail` ni débordement de pile. Ils gardent les chemins en clair, et Linux n'a rien. |
| GPU-16 | Mineur | SetupDi, registre, DXGI, `QueryDisplayConfig` et NVAPI sont appelés sur le thread de rendu toutes les 500 ms ou 2 s. Majeur à 144-165 Hz si confirmé (voir WIN-04). |
| WIN-05 | Mineur | Le proxy Windows est lu à moitié : ni PAC, ni WPAD, ni exclusions, ni authentification (voir SEC-13). |
| WIN-06 | Mineur | Sous Windows, un client muet bloque la fermeture. Sous Linux, une course entre `bind` et `listen` peut ouvrir deux fenêtres (voir SEC-15). |
| WIN-08 | Mineur | Le nettoyage du journal est sensible à la casse et ignore les noms courts. Le nom d'utilisateur fuit, et `/home/al` abîme `/home/alice`. |
| WIN-09 | Mineur | Big5 est décodé en CP950 sous Windows et refusé par iconv sous Linux, qui affiche du charabia. |
| WIN-10 | Mineur | `start_task` passe par `std::async`, qui prend le pool ConcRT sous MSVC, sans nom ni rôle de thread. |
| WIN-11 | Mineur | Écarts inchangés : pré-analyse sur un thread, sémaphores non partagés, `timeBeginPeriod(1)` global, manifeste sans `<supportedOS>`, veille jamais suspendue, télémétrie NVIDIA seule. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md)
- [17. Sous-titres](17-sous-titres.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md)
- [23. Réglages : le panneau](23-reglages-panneau.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md)
- [41. HDR, grade et looks](41-hdr-grade.md)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md)
