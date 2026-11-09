# Audit 30. Concurrence, threads et durée de vie

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Les courses de données et les blocages se voient rarement dans un seul périmètre. Cet audit fait l'inventaire de tous les threads du lecteur et de ce qu'ils partagent.

## Méthode

1. Lister tous les points de création de threads et de tâches : `start_thread`, `start_task`, `std::jthread`, `std::async`, threads détachés, threads des bibliothèques (FFmpeg, libass, SDL).
2. Pour chacun : propriétaire, données partagées, verrous et atomiques, condition d'arrêt, ordre d'arrêt.
3. Relire chaque donnée partagée sous l'angle de l'accès concurrent ; lancer TSan sur les tests quand c'est possible.

## À couvrir

- Threads détachés qui survivent à la fermeture.
- Appels `pl_gpu` depuis plusieurs threads.
- Attentes sans délai (fermeture bloquée).
- Fichiers écrits par plusieurs threads (`update.toml`, réglages).
- Callbacks de bibliothèques appelés sur leurs propres threads.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Outils de threads du lecteur

| Point | Où | Zone |
|---|---|---|
| création d'un thread nommé | `src/core/thread.h:78` | [20](20-socle-core.md) |
| tâche asynchrone | `src/core/thread.h:107` | [20](20-socle-core.md) |
| compte rendu d'un thread | `src/core/thread.h:48` | [20](20-socle-core.md) |
| boîtes de messages entre threads | `src/core/inbox.h:24` | [20](20-socle-core.md) |
| écrivain en arrière-plan | `src/app/play/latest_writer.cpp:61` | [22](22-reglages-modele.md) |
| étage d'anneau | `src/app/play/ring_stage.cpp:51` | [09](09-etages-traitement.md) |
| voies TensorRT | `src/gpu/cuda/backend/tensorrt_lanes.cpp:23` | [01](01-tensorrt-execution.md) |

### Tous les points de création de threads et de tâches

Inventaire à reprendre en entier : pour chaque ligne, propriétaire, données partagées, condition et ordre d'arrêt. (49 lignes, zones de la cible seulement.)

| Où | Zone | Code |
|---|---|---|
| `src/prepare/preparer.cpp:115` | [05](05-construction-moteurs.md) | `auto thread = start_thread(` |
| `src/prepare/preparer.h:86` | [05](05-construction-moteurs.md) | `std::jthread thread_;  // dernier membre : démarre après la boîte, s'arrête avant elle` |
| `src/app/play/engine_downloads.cpp:48` | [06](06-installation-moteurs.md) | `start_thread("install", ThreadReport([shared](const Refusal& refusal) {` |
| `src/app/play/chain.h:66` | [08](08-boucle-chaine.md) | `std::jthread thread_;` |
| `src/app/play/chain_stats.cpp:187` | [08](08-boucle-chaine.md) | `start_thread("stats", ThreadReport([](const Refusal& refusal) {` |
| `src/app/play/chain_stats.h:82` | [08](08-boucle-chaine.md) | `std::jthread sampler_;` |
| `src/app/play/engine_build.cpp:15` | [08](08-boucle-chaine.md) | `auto thread = start_thread(` |
| `src/app/play/engine_build.h:56` | [08](08-boucle-chaine.md) | `std::jthread thread_;  // dernier membre` |
| `src/app/play/ring_stage.cpp:34` | [09](09-etages-traitement.md) | `start_thread(name, init_.report, [this](const std::stop_token& stop, const ThreadReport& r` |
| `src/app/play/ring_stage.h:131` | [09](09-etages-traitement.md) | `std::jthread thread_;` |
| `src/app/play/ring_watcher.cpp:52` | [10](10-presentation-libplacebo.md) | `auto started = start_thread(` |
| `src/app/play/ring_watcher.h:74` | [10](10-presentation-libplacebo.md) | `std::jthread thread_;` |
| `src/media/source.cpp:66` | [14](14-sources-demux.md) | `start_thread("source", ThreadReport([listener](const Refusal& refusal) { listener->failed(` |
| `src/media/source.h:119` | [14](14-sources-demux.md) | `~SourceThread() = default;  // le std::jthread demande l'arrêt et attend` |
| `src/media/source.h:124` | [14](14-sources-demux.md) | `std::jthread thread_;` |
| `src/audio/audio_thread.cpp:39` | [16](16-audio.md) | `auto thread = start_thread("audio", audio->report_,` |
| `src/audio/audio_thread.h:78` | [16](16-audio.md) | `std::jthread thread_;  // dernier membre : s'arrête avant que le reste ne soit détruit` |
| `src/subs/subs_thread.cpp:53` | [17](17-sous-titres.md) | `auto started = start_thread(` |
| `src/subs/subs_thread.h:29` | [17](17-sous-titres.md) | `~SubsThread() = default;  // le std::jthread demande l'arrêt et attend` |
| `src/subs/subs_thread.h:34` | [17](17-sous-titres.md) | `std::jthread thread_;` |
| `src/app/play/player.cpp:154` | [18](18-demarrage-arret.md) | `std::jthread update_check;` |
| `src/app/play/player.cpp:169` | [18](18-demarrage-arret.md) | `auto started = start_thread(` |
| `src/app/play/pumped_task.h:48` | [18](18-demarrage-arret.md) | `start_thread(name, ThreadReport([shared](const Refusal& refusal) {` |
| `src/app/play/single_instance.cpp:137` | [19](19-integration-systeme.md) | `auto started = start_thread("instance", ThreadReport([](const Refusal& refusal) {` |
| `src/app/play/single_instance.cpp:172` | [19](19-integration-systeme.md) | `std::jthread thread_;` |
| `src/app/play/single_instance.cpp:326` | [19](19-integration-systeme.md) | `auto started = start_thread("instance", ThreadReport([](const Refusal& refusal) {` |
| `src/app/play/single_instance.cpp:357` | [19](19-integration-systeme.md) | `std::jthread thread_;` |
| `src/core/file_sync.cpp:57` | [20](20-socle-core.md) | `const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());` |
| `src/core/thread.h:78` | [20](20-socle-core.md) | `std::expected<std::jthread, Refusal> start_thread(std::string_view name, ThreadReport repo` |
| `src/core/thread.h:85` | [20](20-socle-core.md) | `return std::jthread([where = std::format("thread {}", name), report = std::move(report),` |
| `src/core/thread.h:93` | [20](20-socle-core.md) | `} catch (...) {  // std::system_error de std::thread, ou mémoire insuffisante` |
| `src/core/thread.h:107` | [20](20-socle-core.md) | `std::shared_ptr<std::future<std::expected<T, Refusal>>> start_task(std::string_view name, ` |
| `src/core/thread.h:111` | [20](20-socle-core.md) | `std::async(std::launch::async, [where = std::format("tâche {}", name), fn = std::move(fn)]` |
| `src/app/play/latest_writer.cpp:28` | [22](22-reglages-modele.md) | `start_thread("écriture", ThreadReport([](const Refusal& refusal) { report(std::unexpected(` |
| `src/app/play/latest_writer.h:43` | [22](22-reglages-modele.md) | `std::jthread thread_;` |
| `src/app/play/ui.cpp:289` | [24](24-interface-rendu.md) | `auto thread = start_thread("polices", ThreadReport([](const Refusal& refusal) {` |
| `src/app/play/ui.h:55` | [24](24-interface-rendu.md) | `std::jthread thread_;` |
| `src/app/play/engine_screen.cpp:133` | [25](25-assistant.md) | `auto worker = start_thread("calibration", ThreadReport([shared](const Refusal& refusal) {` |
| `src/app/play/chain_hdr.cpp:162` | [41](41-hdr-grade.md) | `if (busy) thread_.detach();` |
| `src/app/play/chain_hdr.cpp:174` | [41](41-hdr-grade.md) | `auto started = start_thread(` |
| `src/render/grade.cpp:257` | [41](41-hdr-grade.md) | `std::jthread worker;  // arrêté avant la destruction des hooks (~Impl)` |
| `src/render/grade.cpp:316` | [41](41-hdr-grade.md) | `if (!worker.joinable()) worker = std::jthread([this](const std::stop_token& stop) { run(st` |
| `src/media/shot_probe.cpp:94` | [42](42-plans-coupes.md) | `auto thread = start_thread("plans", ThreadReport([self](const Refusal& refusal) { self->fa` |
| `src/media/shot_probe.cpp:170` | [42](42-plans-coupes.md) | `const int auto_threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()) ` |
| `src/media/shot_probe.h:81` | [42](42-plans-coupes.md) | `std::jthread thread_;` |
| `src/gpu/vulkan/backend/deferred_steps.cpp:29` | [45](45-backend-vulkan.md) | `start_thread(thread_, ThreadReport([this](const Refusal& refusal) { note_failure(refusal);` |
| `src/gpu/vulkan/backend/deferred_steps.h:69` | [45](45-backend-vulkan.md) | `std::jthread worker_;` |
| `src/gpu/vulkan/backend/kernel_frames.cpp:292` | [45](45-backend-vulkan.md) | `auto thread = start_thread(` |
| `src/gpu/vulkan/backend/kernel_ops.h:324` | [45](45-backend-vulkan.md) | `std::jthread upload_thread_;` |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-09 | Mineur | `GpuTelemetry::sample` est écrit par deux threads sans verrou en fin de chaîne. Comportement indéfini. |
| GPU-10 | Mineur | Après une erreur de `vkWaitSemaphores`, `forget_all` peut attendre sans fin (latent). La boucle sonde ensuite à 4 ms. |
| GPU-11 | Mineur | Le thread des LUT est un `std::jthread` brut qui crée une texture `pl_gpu` en parallèle du rendu. Un `bad_alloc` mène à `std::terminate`. |
| GPU-12 | Mineur | `LookLoader` détache un thread occupé, qui peut journaliser après `spdlog::shutdown`. |
| GPU-17 | Mineur | Un GPU bloqué sans être perdu n'est pas détecté. La fermeture peut rester bloquée dans `pl_gpu_finish`. |
| QUA-13 | Mineur | La sonde abandonnée reste un thread détaché. Elle peut journaliser après la destruction des statiques. |
| SEC-09 | Mineur | Trois fils lisent et réécrivent `update.toml` sans verrou. Un téléchargement ou une révocation peut être perdu. |
| WIN-06 | Mineur | Sous Windows, un client muet bloque la fermeture. Sous Linux, une course entre `bind` et `listen` peut ouvrir deux fenêtres (voir SEC-15). |
| WIN-10 | Mineur | `start_task` passe par `std::async`, qui prend le pool ConcRT sous MSVC, sans nom ni rôle de thread. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
- [16. Audio](16-audio.md)
- [17. Sous-titres](17-sous-titres.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [25. Assistant de premier lancement](25-assistant.md)
- [41. HDR, grade et looks](41-hdr-grade.md)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md)
