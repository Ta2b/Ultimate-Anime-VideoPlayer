# Rapport de l'audit 30. Concurrence, threads et durée de vie

Branche : `claude/project-thread-on971p`. Commit audité : `401bfa7564240f3adcb133b356a70038e1675932`. Date : `2026-11-09`.

## Verdict

La concurrence du lecteur est tenue par une règle simple et bien suivie (un propriétaire par thread, des boîtes
bornées, l'ordre des membres comme ordre d'arrêt), et aucune course de données nouvelle n'apparaît dans le chemin
d'une image. Les défauts sont tous aux frontières de durée de vie : un majeur à confirmer sur la carte (les moteurs
TensorRT sont détruits pendant que le GPU exécute encore leur dernière passe, à chaque fermeture de chaîne en
lecture), et sept mineurs (threads détachés qui journalisent après la fermeture du journal, deux vérifications de
mise à jour qui écrivent la même archive, attentes sans délai sur le thread de la fenêtre). Rien, dans les threads
ni dans les voies, ne pénalise DistilDRBA plus que RIFE.

## Cadrage retenu

Pas de questions posées : Tatoub a demandé d'auditer au maximum sans cadrage. Retenu :

- la cible est NVIDIA sous TensorRT, Windows 11 et CachyOS. Le VRR est hors sujet. Les threads du moteur Vulkan
  (`deferred_steps.cpp:29`, `kernel_frames.cpp:292`) sont listés mais pas relus en profondeur (audit 45, réduit) ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant : la section « DistilDRBA » dit ce que les threads et
  les voies changent pour lui ;
- toute baisse de qualité d'image faite pour la vitesse est un défaut, sauf mesure : la section « Compromis » liste
  ce que cet audit a croisé ;
- le fil tourne sans carte NVIDIA et sans les dépendances (`~/ajp-deps` absent, spdlog et Catch2 hors d'atteinte du
  réseau) : **rien n'a été compilé ni lancé, TSan compris**. Tous les constats sont lus. Ce qui demande la carte est
  marqué « à confirmer », avec la manipulation.

## Inventaire des threads (cible)

Tous passent par `start_thread` (`src/core/thread.h:78`), sauf deux exceptions notées. « Joint » : par le
destructeur du `std::jthread`, après `request_stop`.

| Thread | Créé à | Propriétaire | Partage | Arrêt |
|---|---|---|---|---|
| source | `src/media/source.cpp:66` | `Pipeline` (détruit le premier) | boîtes (`Inbox`), crédits | `stop_callback` qui interrompt FFmpeg (`source_loop.cpp:44`), joint. Bloqué sur un partage muet : renvoi 14 vers 18 |
| gpu (étage d'anneau) | `src/app/play/ring_stage.cpp:34` | `Pipeline::video_` | boîte, `in_flight_` et `ended_` atomiques, `LiveSettings` (mutex et atomiques), horloge (mutex) | `Quit` ou arrêt, `stop()` joint avant la sous-classe (`ring_stage.cpp:27-30`) |
| audio | `src/audio/audio_thread.cpp:39` | `Pipeline::audio_` | boîte, déclarée avant la sortie dont les rappels SDL y écrivent (`audio_thread.h:74`) | joint ; dernier membre |
| subs | `src/subs/subs_thread.cpp:53` | `Pipeline::subs_thread_` | boîte partagée | joint |
| veilleur d'anneau | `src/app/play/ring_watcher.cpp:52` | `WriteWatch` (`player_run.cpp:186`) | `entries_` sous mutex, sémaphores Vulkan | sémaphore de réveil puis joint. Mort sur erreur : A10-03 |
| stats (relevé à 2 Hz) | `src/app/play/chain_stats.cpp:187` | `ChainStats` | `kernels_`, `telemetry_` | joint, mais **après** la destruction du backend : A08-02 |
| écriture | `src/app/play/latest_writer.cpp:28` | `LatestWriter` (`player_run.cpp:352`) | `pending_` sous mutex | `flush()` sans délai, puis joint |
| looks | `src/app/play/chain_hdr.cpp:174` | `LookLoader` (`Chain`) | `State` en `shared_ptr` | joint, ou **détaché** s'il lit (`chain_hdr.cpp:162`) : A30-02 |
| LUT du grade | `src/render/grade.cpp:316` | `Grade::Impl` | `job`, `done` sous mutex, `pl_gpu` | `std::jthread` **brut**, joint dans `~Impl` : A30-04 |
| plans | `src/media/shot_probe.cpp:94` | `ShotProbe` | `ShotTimeline`, atomiques | interruption FFmpeg (`shot_probe.cpp:154`), joint : A30-08 |
| preparer (construction) | `src/app/play/engine_build.cpp:15` | `EngineBuild` | résultat sous mutex ; sa propre instance de backend | arrêt qui tue `trtexec`, joint (renvoi 08 vers 05) |
| install | `src/app/play/engine_downloads.cpp:48` | local à `run_install` | `Shared` en `shared_ptr` | arrêt, joint au retour de la fonction |
| calibration | `src/app/play/engine_screen.cpp:133` | local à `calibrate_on_screen` | `Shared` en `shared_ptr` | arrêt, joint au retour |
| updates | `src/app/play/player.cpp:185` | local à `play()` | `update.toml`, `playing` atomique | arrêt qui tue `curl`, joint à la fin de `play()` |
| polices | `src/app/play/ui.cpp:289` | `FontSearch` | résultat lu après `join` | joint dans `take()` |
| instance | `src/app/play/single_instance.cpp:137` et `:326` | `InstanceListener` | file d'événements SDL | Linux : tranches de 100 ms. Windows : A30-06 |
| probe (sonde) | `src/app/play/pumped_task.h:48` | personne | `Shared` en `shared_ptr` | **détaché** d'office (`pumped_task.h:58`) : A30-05 |
| tâches `start_task` | `src/core/thread.h:111` | futur gardé par le panneau | par valeur | `std::async` : le destructeur du futur attend (A20-05) |
| preparer (`prepare/`) | `src/prepare/preparer.cpp:115` | aucun appelant | | code mort (A05-08) |

Threads des bibliothèques : la file asynchrone de spdlog (`log.cpp:105`, un thread), les threads de décodage de
FFmpeg, le thread du périphérique audio de SDL (rappel `on_device_request`, `sdl_sink.cpp:115`), les filtres
d'événements SDL (`display_contract.cpp:61`, `sdl_sink.cpp:137`) qui ne posent qu'un atomique ou un message.
Le backend TensorRT n'a **aucun thread** : ses « voies » sont trois files CUDA remplies par le seul thread gpu
(`tensorrt_lanes.cpp:23-53`).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A30-01 | Majeur | `src/gpu/cuda/backend/trt_engine.cpp:115-119` | Les moteurs TensorRT sont détruits sans attendre le travail encore enfilé sur les files CUDA. `~TrtEngine` détruit ses graphes CUDA puis son contexte d'exécution tout de suite ; `~DrbaFusedEngine` (`drba_fused_engine.cpp:89-91`) et `~RifeFusedEngine` (`rife_fused_engine.cpp:97`) détruisent leurs graphes de même. La seule attente des files est dans `destroy_lanes` (`tensorrt_lanes.cpp:55-58`), appelée par le destructeur du backend, donc **après** les moteurs (`chain_setup.h:64`). C'est le renvoi 01 vers 30. | Toute fermeture de chaîne pendant la lecture : épisode suivant, réglage qui rouvre la chaîne, `q`. `run_chain` arrête le thread gpu (`player_run.cpp:745`), qui ne fait qu'enfiler sans attendre, puis détruit les moteurs (`player_run.cpp:748`) moins d'une milliseconde après. Si le thread gpu venait d'enfiler une passe SPAN ou une passe d'interpolation (10 à 20 ms de GPU), elle tourne encore : ses poids et son contexte sont libérés sous elle. Au mieux rien ne se voit (les emplacements sont déjà rendus). Au pire, une erreur CUDA collante (adresse illégale, Xid 31) qui fait refuser la chaîne suivante dans le même processus, donc un repli sur Vulkan ou « Playback stopped » au changement d'épisode. | Attendre les files avant de détruire un moteur. Le plus simple : une opération `Backend::drain()` qui synchronise chaque voie, appelée par `run_chain` entre `pipeline->reset()` et `prepared->reset()` ; et, par sûreté, `~TrtEngine` et les deux moteurs fusionnés synchronisent la file de leur dernière passe. Coût : au plus une image de GPU à la fermeture. | Lu. **À confirmer** : sur la RTX, lire une vidéo en ×3 ou plus (le thread gpu toujours occupé) et enchaîner 200 réouvertures (`--keys` avec la touche de l'épisode suivant, ou un réglage qui rouvre) ; surveiller `dmesg \| grep -i xid`, `ajp.log` (« cuda », refus à l'ouverture) et un repli inattendu sur Vulkan. Test décrit (`tests/unit/backend_tensorrt`) : enfiler 50 passes SPAN sans attendre, détruire le moteur, le recharger, vérifier `health()` et l'absence d'erreur CUDA. |
| A30-02 | Mineur | `src/app/play/chain_hdr.cpp:153-163` | `LookLoader` détache son thread quand une lecture de LUT est en cours (GPU-12, renvoi 18 vers 30). Ce thread journalise (`chain_hdr.cpp:115` et `:119`, `look_catalog.cpp:66` et `:127`) et appelle `SDL_GetBasePath` (`look_catalog.cpp:25`). Rien ne l'attend avant `SDL_Quit` (`main.cpp:518`) ni avant `close_log` (`main.cpp:554`). Or, depuis A18-01, un journal après `close_log` ne se perd pas : `log_channel` déréférence le journal par défaut devenu nul (`log.cpp:77`). | Changer de look, puis quitter avant la fin de la lecture du `.cube` (jusqu'à 16 Mo, `cube_lut.h:21`, plus le remplissage `fill_unreachable`). Si la lecture dure plus que la fermeture du lecteur (disque lent, gros `.cube` de l'utilisateur), le thread écrit sa ligne « look 1 : … » après `close_log` : plantage à la sortie, avec un vidage de plantage pour une fermeture normale. Fenêtre étroite sur un SSD. Par ailleurs le détachement ne protège de rien dans l'autre sens : si la demande est posée mais pas encore prise, `busy` est faux, le thread démarre la lecture malgré l'arrêt (le prédicat est vu avant l'arrêt, `chain_hdr.cpp:179`) et le destructeur l'attend quand même. | Ne plus détacher : joindre toujours. La lecture est bornée (fichier local plafonné à 16 Mo) ; pour une fermeture vive, passer le jeton d'arrêt à `load_looks` et le regarder entre deux LUT. | Lu. Le plantage d'un journal tardif est démontré par A18-01. Test décrit : un `LookLoader` dont la lecture attend un verrou ; le détruire, appeler `close_log()`, libérer le verrou : le processus ne doit pas planter (aujourd'hui il plante). |
| A30-03 | Mineur | `src/app/play/settings_system.cpp:177-216` | Deux vérifications de mise à jour peuvent tourner en même temps : le thread « updates » (`player.cpp:185-207`) et la tâche de « Check now » ou « Install now ». Les deux téléchargent la même archive dans le même fichier (`updater.cpp:466-469`, `curl -o` sans fichier temporaire, `updater.cpp:90-91`) et réécrivent `update.toml` en entier (SEC-09, A26-10). | Mode automatique, une version plus récente existe : au lancement, le thread « updates » commence le téléchargement (jusqu'à 30 minutes, `--max-time 1800`). L'utilisateur ouvre Général, Application et clique « Install now ». Deux `curl` écrivent le même fichier. Le premier fini vérifie l'empreinte pendant que l'autre écrit encore : échec, et `fs::remove` (`updater.cpp:471`) efface le fichier sous le second. « Install now » peut aussi appliquer l'archive (`apply_staged_update`, `settings_system.cpp:209`) pendant que l'autre thread la retélécharge. Résultat : une mise à jour qui échoue sans raison claire, ou `update.toml` qui perd `staged_version`. | Un seul verrou de processus autour de `run_update_check`, `apply_staged_update` et `roll_back_update` (un mutex partagé par le thread « updates » et le panneau), et un téléchargement dans `<nom>.part` renommé après vérification. Le verrou règle aussi A26-10 pour ces écrivains. | Lu. Test décrit : deux `download_update` en parallèle avec un `fetch` simulé qui écrit lentement ; les deux doivent rendre un fichier vérifié. |
| A30-04 | Mineur | `src/render/grade.cpp:316` | Le thread qui compose les LUT est un `std::jthread` brut (GPU-11), contre la règle de `thread.h:1`. Son corps (`grade.cpp:277-301`) n'a pas de `try` : `compose_luts` alloue une LUT 65³, un `bad_alloc` mène à `std::terminate`. Le constructeur du `jthread` peut aussi lever (`std::system_error`) depuis `compose`, appelé par le rendu. La partie « texture créée en parallèle du rendu » de GPU-11 est à confirmer : libplacebo annonce `pl_gpu` sûr entre threads. | Mémoire courte pendant un changement d'intensité de look : le lecteur s'arrête par `std::terminate` au lieu de garder l'ancien look. | `start_thread` avec un rapporteur : sur refus, garder le look en place et écrire une ligne au journal. Si le thread ne démarre pas, composer sur place comme sans `background`. | Lu. À confirmer dans `libplacebo/gpu.h` de la version livrée : « Thread-safety: Safe » sur `pl_gpu`. |
| A30-05 | Mineur | `src/app/play/pumped_task.h:47-58` | La sonde du fichier est détachée d'office (QUA-13, renvoi 20 vers 18, jamais numéroté). Bloquée sur un partage muet, elle peut rendre la main n'importe quand : pendant `SDL_Quit`, après `close_log` (plantage, A18-01) ou pendant la destruction des statiques. | Double-clic sur un fichier d'un NAS endormi : la sonde dépasse 20 s, l'accueil s'ouvre avec le message. L'utilisateur ferme. Le NAS répond pendant la fermeture : FFmpeg journalise, plantage à la sortie. | Garder le `jthread` dans un registre du processus, et à la fin de `run_player` lui laisser un court délai puis terminer par `std::quick_exit` s'il est encore bloqué (les fichiers sont déjà écrits à ce moment). | Lu. |
| A30-06 | Mineur | `src/app/play/single_instance.cpp:155-157` | Windows : `ReadFile` synchrone sans délai sur le canal de l'instance unique (WIN-06, revu par l'audit 19 sans numéro). Un client connecté qui n'envoie rien bloque le thread « instance » ; le destructeur (`single_instance.cpp:124-134`) essaie de se connecter lui-même, reçoit `ERROR_PIPE_BUSY`, et `join` attend sans fin. | Un second lancement tué juste après sa connexion, ou un programme qui ouvre le canal et se tait : à la fermeture, `instance.reset()` (`main.cpp:517`) ne rend jamais la main. Le processus reste en mémoire sans fenêtre et tient le canal : le prochain lancement lui transmet son fichier, et rien ne s'ouvre. | Canal ouvert en `FILE_FLAG_OVERLAPPED`, lecture bornée à 2 s comme sous Linux (`single_instance.cpp:342`), et `CancelIoEx` dans le destructeur avant `join`. | Lu. |
| A30-07 | Mineur | `tests/unit/app/` | Aucun test des durées de vie qui portent les défauts de ce rapport : `LookLoader` détruit en cours de lecture, `ChainStats` détruit après son backend (A08-02), `EngineBuild` détruit en cours de construction, deux vérifications de mise à jour en parallèle, client muet du canal. Le module TensorRT n'a aucun passage sous TSan ni test de destruction avec du travail en vol (A30-01). | Une régression de l'ordre d'arrêt passe la CI, qui a pourtant un job TSan. | Les tests décrits aux lignes A30-01 à A30-03, plus `ChainStats` avec un `KernelOps` simulé dont `device_memory` attend : la destruction doit joindre le relevé avant de rendre les noyaux. | Lu. |
| A30-08 | Mineur | `src/media/shot_probe.cpp:108-110` | Attentes sans délai sur le thread de la fenêtre. `~ShotProbe` joint le thread des plans : l'interruption FFmpeg (`shot_probe.cpp:154`) ne sort pas d'un `read()` bloqué dans le noyau (A14-02). `LatestWriter::flush` (`latest_writer.cpp:57-61`, appelé à `player_run.cpp:674`) attend le disque sans délai. | Vidéo sur un partage qui se tait : changer d'épisode ou quitter fige la fenêtre jusqu'au retour du partage, par la sonde des plans autant que par la source (renvoi 14 vers 18). Dossier de données sur un disque qui se réveille : la fermeture de la chaîne attend `play.toml`. | Le même remède que pour la source : un délai, puis le thread est remis à un registre joint (ou abandonné) à la sortie du processus. Pour `flush`, une attente bornée à quelques secondes avec une ligne au journal. | Lu. |

## DistilDRBA : ce que les threads et les voies changent

Rien, dans la concurrence, ne ralentit DistilDRBA plus que RIFE. Constats :

- **Un seul thread remplit les trois voies.** Le thread gpu enfile sans attendre : ingestion et mesures sur la voie
  Mesure, SPAN sur la voie Agrandissement (priorité haute, `tensorrt_lanes.cpp:36`), RIFE ou DistilDRBA sur la voie
  Interpolation. En régime établi, il ne bloque que sur un emplacement d'anneau, dans sa boîte, sans tourner à
  vide. C'est vrai pour les trois modèles.
- **Les N−1 passes de DistilDRBA d'une paire sont en série par construction** : elles sont toutes sur la file
  Interpolation, et partagent la même zone d'activation (`tensorrt_lanes.cpp:74-93`, zone 1). Les voies ne
  peuvent donc rien rattraper du surcoût d'A03-13 (une passe par image créée) : le remède est dans le modèle et
  dans son entrée (A03-13, A03-14, A03-15), pas dans les threads.
- **Une attente hôte par paire, que RIFE a aussi** : `settle_after_scores` (`upscale_drba.cpp:95-107`) appelle
  `scores_take`, qui attend la voie Mesure. Pour DistilDRBA la mesure se fait à la taille de travail et derrière
  l'ingestion de l'image suivante (renvoi 03 vers 09, A21-06) ; pour RIFE, sur la copie réduite. C'est le seul
  point où le thread gpu peut attendre plus longtemps avec DistilDRBA. À mesurer : `scores_wait_us_`
  (`upscale_stage.h:500`) existe déjà pour le profil.
- **`wants_next()` vaut vrai pour DistilDRBA** (`upscale_stage.h:168`) : l'image de tête attend la suivante. Cela
  coûte une image source de latence, pas de débit.
- **Sous 7,5 Gio de mémoire vidéo, une seule file** (`tensorrt_backend.h:139`, `tensorrt_lanes.cpp:25`) : plus
  aucun recouvrement entre SPAN et l'interpolation. Hors de la carte de Tatoub ; cela touche les RTX à 6 Go
  (2060, 3050, 4050 de portable), pour les trois modèles.
- **Aucun outil livré ne sépare le temps d'attente du temps de calcul** entre DistilDRBA et RIFE (A20-11) : la
  question « qu'est-ce qui l'empêche d'être bien plus rapide » ne se tranche que par `--one-lane` contre trois voies
  sur la RTX, par modèle. Renvoi à 32.

## Compromis qualité contre vitesse croisés

Aucun ne réduit la précision ou le format de l'image. Ce sont des règles de temps réel, sans mesure dans le dépôt.

| Où | Ce qui est fait | Effet sur l'image | Gain | Mesuré ? |
|---|---|---|---|---|
| `src/app/play/upscale_stage.cpp:567-584` | Chaîne en retard : l'image source est sautée sans calcul, et la paire suivante repart « comme au début d'une époque » (`previous_`, cadence et grilles remis à zéro) | Une image source en moins, puis une paire **sans interpolation** | La source, le son et l'horloge ne sont pas retenus | Non. Le compteur « skipped (+ N before compute) » de l'OSD le montre ; aucune mesure ne dit s'il faut vraiment perdre aussi l'interpolation de la paire suivante. À trancher |
| `src/app/play/upscale_stage.cpp:589` | Rapide à pas constant : un instant qui serait passé n'est ni calculé ni écrit (`Fate::Late`) | Une image interpolée en moins | Pas de calcul perdu | Non |
| `src/app/play/upscale_stage.cpp:304` | Fenêtre cachée ou réduite : l'interpolation s'arrête | Aucun (rien n'est vu) | GPU libéré | Sans objet |
| `src/app/play/chain_hdr.cpp:214-223`, `src/render/grade.cpp:296-320` | Look lu et composé sur un thread : posé un ou plusieurs tours plus tard | L'ancien look reste quelques images | Le thread d'affichage n'attend pas le disque | Sans objet |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-09 | Toujours présent | Doublon d'A11-08. `GpuTelemetry::sample` (`gpu_telemetry.cpp:311-372`) lit et écrit `i.last` sans verrou depuis le thread « stats » (`chain_stats.cpp:169`) et depuis main (`player_run.cpp:679`). La course n'existe que si l'OSD a demandé la ligne GPU (le thread naît au premier appel). Correction : un mutex dans `sample`. Renvoi 11 vers 30 : traité. |
| GPU-10 | Toujours présent, latent pour l'attente | Doublon d'A10-03. `ring_watcher.cpp:144` sort sans rien poser ; `forget_all` (`:105-112`) attend alors sans fin. Non atteignable aujourd'hui : `WriteWatch` (`player_run.cpp:186`) détruit le veilleur avant l'anneau. |
| GPU-11 | Toujours présent | A30-04. |
| GPU-12 | Toujours présent, plus grave | A30-02 : avec A18-01, c'est un plantage et non une ligne perdue. |
| GPU-17 | Toujours présent | Doublon d'A10-04 (`pl_gpu_finish` sans délai). Non recreusé. |
| QUA-13 | Toujours présent | A30-05. |
| SEC-09 | Toujours présent | Doublon d'A26-10. Écrivains relevés : le thread « updates » (`player.cpp:197-200`), trois tâches du panneau (`settings_system.cpp:181-229`), le bouton « Offer it again » sur main (`:254-256`), et `main.cpp:522-526` à la sortie. Le thread « updates » relit le fichier puis le réécrit jusqu'à 30 minutes plus tard : tout ce que le panneau a écrit entre-temps est perdu. Aggravé par A30-03. |
| WIN-06 | Toujours présent | A30-06 pour Windows. Linux : la lecture est bornée à 2 s (`single_instance.cpp:342`) ; la course `bind` contre `listen` reste à l'audit 19 (A19-13). |
| WIN-10 | Toujours présent | Doublon d'A20-05. Les futurs de `std::async` bloquent à la destruction : `UpdateNote` demande l'arrêt d'abord (`settings_system.h:102`), `SystemNote` non (`settings_system.h:22-28`, jusqu'à 10 s de `xdg-mime` à la fermeture). |
| A08-02 | Toujours présent, plus probable qu'annoncé | `ChainStats` est détruit à la sortie de `run_chain`, après `prepared->reset()` (`player_run.cpp:748`). La destruction du backend dure des dizaines à des centaines de millisecondes et le relevé passe toutes les 500 ms : dès que l'OSD a montré la mémoire vidéo, un `cudaMemGetInfo` sur un backend en cours de destruction est probable à chaque fermeture de chaîne. Correction : déclarer `ChainStats` de sorte qu'il parte avant, ou l'arrêter explicitement avant `pipeline->reset()`. |
| Renvoi 01 vers 30 | Confirmé à la lecture | A30-01. |
| Renvoi 13 vers 30 | Toujours présent | Doublon d'A13-06. `ThreadStages::play` démarre l'horloge hôte et pose `playing_` avant de poster à l'audio (`wiring.cpp:113-118`) : sur boîte audio pleine, l'état de main et celui de l'audio divergent, et personne ne rejoue. |
| Renvoi 14 vers 30 | Sans gravité | Pendant `wait_before_retry` (`demuxer.cpp:265-272`, 1,55 s au plus sur cinq essais), la boîte de la source n'est pas lue. Un seek refusé pour boîte pleine (4 `Flush`) est rejoué par la session (`session.cpp:201-209`), et l'arrêt coupe l'attente par tranches de 10 ms. Reste un seek qui peut attendre 1,5 s sur un disque en faute. |
| Renvoi 18 vers 30 | Confirmé | A30-02. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 01 | `src/gpu/cuda/backend/tensorrt_backend.cpp:201-204` | `~TensorRtBackend` : c'est ici que `drain()` (A30-01) doit vivre, avec les voies |
| 32 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:140-150` | Aucun `cudaSetDeviceFlags` : par défaut, `cudaEventSynchronize` tourne à vide sur un cœur. Rare en régime établi, mais à mesurer avec un décodage logiciel (H.264 10 bits) qui veut ces cœurs |
| 32 | `src/app/play/upscale_stage.h:500` | Comparer l'attente hôte de `scores_take` pour DistilDRBA (taille de travail) et pour RIFE (copie réduite), `--one-lane` contre trois voies |
| 42 | `src/media/shot_probe.cpp:108-110` | `~ShotProbe` joint sur le thread de la fenêtre un thread qui lit le fichier (A30-08) |
| 26 | `src/app/play/updater.cpp:466-474` | Téléchargement de l'archive sans `.part` ni verrou entre les deux vérifications (A30-03) |
| 41 | `src/render/grade.cpp:277-301` | Corps du thread des LUT sans `try` ; à passer par `start_thread` (A30-04) |
| 09 | `src/app/play/upscale_stage.cpp:574-579` | Après une image sautée pour retard, la paire suivante n'est pas interpolée : choix à mesurer |
| 40 | `.github/workflows/ci.yml:100` | Le job TSan ne couvre ni `LookLoader`, ni `ChainStats`, ni `EngineBuild`, ni le module TensorRT (A30-07) |

## Points positifs

- Un propriétaire par thread et des boîtes bornées (`core/inbox.h`), avec crédits : aucun producteur n'attend dans
  la boîte d'un autre, donc pas d'interblocage entre source, gpu, audio et subs.
- L'ordre des membres est l'ordre d'arrêt, et il est commenté partout où il compte (`pipeline.h:114-131`,
  `audio_thread.h:74-78`, `chain_setup.h:64`).
- `start_thread` ne lève jamais et rend toute exception du corps en refus (`thread.h:78-101`) ; un refus fatal a sa
  place réservée dans la boîte de main (`wiring.cpp:79-83`).
- Le thread gpu n'attend jamais le GPU en régime établi ; les libérations passent derrière toutes les voies
  (`free_after_lanes`, `tensorrt_backend.cpp:376-392`).
- `LaneHazards` (`gpu/lanes.h`) déduit les dépendances entre files des ressources lues et écrites : un mauvais
  choix de voie coûte du recouvrement, jamais un résultat faux.
- `LiveSettings`, `HostClock`, `i18n` et les filtres d'événements SDL sont protégés (mutex ou atomiques).
- La CI a un job TSan, la nuit et sur une partie des tests à chaque PR (`ci.yml:100`), et des tests pour `Inbox`,
  `start_thread`, `LatestWriter`, `run_pumped_for` et le veilleur d'anneau.
- `UpdateNote` et la préparation SPAN demandent l'arrêt avant d'attendre leur tâche (`settings_system.h:102`,
  `settings_panel.cpp:336-339`).

## Questions ouvertes

- TensorRT de la version livrée : la destruction d'un `IExecutionContext` attend-elle l'inférence enfilée par
  `enqueueV3` ? La documentation ne le promet pas ; A30-01 suppose que non.
- `cudaFree` et `cudaGraphExecDestroy` sur un travail en vol : attente implicite ou non, selon la version de CUDA ?
- libplacebo livré : `pl_gpu` est-il bien annoncé sûr entre threads (A30-04) ?
- Faut-il vraiment perdre l'interpolation de la paire qui suit une image sautée pour retard (section Compromis) ?
  Une mesure du nombre de sauts en lecture normale sur la RTX 4070 Ti dirait si la question vaut d'être creusée.
