# Rapport de l'audit 01. TensorRT : exécution, voies, anneau et mémoire

Branche : `claude/project-thread-x36pqh` (la branche de ce fil ; `audit/01-tensorrt-execution` demandée par le
coordinateur n'était pas ouverte en écriture à ce fil). Commit audité : `0be463d`. Date : `2026-11-09`.

## Verdict

Le moteur d'exécution TensorRT est solide sur le chemin nominal (voies ordonnées par ressources, mémoire rendue
derrière les voies, graphes CUDA invalidés au bon moment, anneau importé proprement), mais la détection de la perte
du GPU dépend d'une liste d'erreurs collantes incomplète, et deux défauts connus sur trois sont toujours présents.

## Cadrage retenu

Les questions ont été posées à Tatoub ; à sa demande que le travail avance sans pause, l'audit a suivi les réponses
par défaut proposées, à corriger s'il répond autrement :

- carte de référence : la RTX 4070 Ti (12 Go), avec un œil sur les cartes de 6 à 8 Go (voies, pool CUDA) ;
- perte du GPU (TDR, Xid) : on exige une détection et un message propre, pas une reprise automatique ;
- envoi des images du décodage logiciel : chemin secondaire, un défaut y est au plus Mineur sauf plantage ;
- rien à retirer dans la zone ; l'export (`tensorrt_export.cpp`, `export_kernels.cu`) est hors cible : lu seulement
  pour `health()`, `map_fences` et la mémoire qu'il partage.

Le fil tourne sans carte NVIDIA ni CUDA : rien n'a pu être compilé ni lancé. Les tests proposés sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A01-01 | Majeur | `src/gpu/cuda/backend/cuda_status.h:15-32`, `tensorrt_export.cpp:258-268` | La perte du contexte n'est reconnue que par une liste fermée de codes (`sticky`). `cudaErrorUnknown` (999), `cudaErrorDevicesUnavailable` (46) et `cudaErrorExternalDevice` (911), que rendent un TDR déclenché par Vulkan ou un Xid 79 (carte tombée du bus), n'y sont pas. `health()` prend alors l'erreur pour « une erreur d'appel passée », l'efface et répond que tout va bien. | Sous Windows, un TDR survient pendant la lecture (libplacebo ou le pilote). Les appels CUDA suivants rendent 999. `health()` répond OK : l'étage de l'anneau continue d'attendre un emplacement, puis chaque appel échoue en `VendorCallFailed` (« appel refusé ») au lieu de `DeviceLost` (« relancer le lecteur »). Le message à l'écran est faux et la chaîne peut tenter des reprises vaines. | Ne plus dépendre d'une liste : une erreur que deux `cudaGetLastError()` de suite rendent toutes deux est collante, quel que soit son code. À défaut, ajouter au moins 999, 46 et 911 à `sticky`. Étendre `cuda_status_test.cpp` d'un cas par code ajouté. | À confirmer : sous Windows, pendant une lecture TensorRT, forcer un TDR (`dxcap -forcetdr`, kit Windows SDK) et relever dans le journal le code CUDA et le message affiché ; sous CachyOS, relever le code CUDA après un Xid (journal `dmesg`). |
| A01-02 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:305-307`, `tensorrt_lanes.cpp:103-121` | `begin_slot` appelle `join()` au milieu d'une opération déjà ordonnée. `apply()` pose alors l'événement de l'opération en cours (rang M) avant les noyaux qui restent à enfiler (écriture du slot, queue du SPAN FP8) ; ces noyaux n'appartiennent plus qu'à la jointure, qui ne déclare aucune ressource. Une autre voie qui attend le rang M n'attend pas la fin de ces noyaux. | Plusieurs voies (carte de 8 Go ou plus), anneau recréé (plein écran, changement de taille, bascule HDR) : la première image de la nouvelle génération passe par `enqueue_span_fp8` (`tensorrt_span_fp8.cpp:78-89`, zone 02), dont la queue écrit `outputs` après le `join`. La voie d'interpolation qui lit cette sortie (warp après SPAN) attend le rang M et peut lire la sortie à moitié écrite : une image fausse au moment de la bascule. Même découpage dans `write_ring` : le noyau qui lit `rgb` n'est plus couvert par le rang M. | Faire la jointure de génération avant `order()` (dans `begin_write`, `launch_into` et `enqueue_span_fp8`, par exemple un `ensure_generation(desc)` appelé en premier), pour que l'opération ordonnée contienne tous ses noyaux. | Lu. À confirmer par un test unitaire GPU : deux voies, nouvelle génération sur un `enqueue_span_fp8` suivi d'une lecture de `outputs` sur la voie d'interpolation, comparer la lecture à la référence. |
| A01-03 | Mineur | `src/gpu/cuda/backend/host_upload.cpp:114`, `:118-129`, `:68-69` | L'envoi d'une image du décodage logiciel fait la copie vers la mémoire épinglée sur le thread qui soumet au GPU (le défaut connu de la fiche), et peut y attendre l'envoi d'il y a trois images (`cudaEventSynchronize`). Quand l'image grandit, `reserve` libère l'ancien tampon par `cudaFree` et `cudaFreeHost`, deux appels qui attendent tout l'appareil, sans en vérifier le retour. | Fansub H.264 10 bits (pas de NVDEC) en 1080p : environ 6 Mo copiés par image sur le thread de la chaîne, environ 25 Mo en 4K. Au premier changement de résolution, toutes les voies se vident d'un coup. | Faire la copie sur le thread du décodeur (ou enregistrer le tampon de FFmpeg par `cudaHostRegister`) ; libérer l'ancien tampon d'appareil par `cudaFreeAsync` derrière l'événement `read`. | Lu. À confirmer : chronométrer `upload` dans le profil GPU sur un fichier H.264 10 bits 1080p et 4K. |
| A01-04 | Mineur | `src/gpu/cuda/backend/tensorrt_stats.cpp:32-37` | Si `cudaMalloc` réussit mais `cudaMallocHost` ou la création de l'événement échoue, l'emplacement garde `device` non nul : il n'est jamais réinitialisé, et chaque ticket qui y tombe échoue ensuite (copie vers une adresse hôte nulle). | Mémoire épinglée refusée une fois (pression mémoire de l'hôte) : un ticket d'analyse sur `kPendingReadbacks` échoue pour le reste de la séance, le grade perd ces analyses. | Libérer ce qui a été pris et remettre l'emplacement à zéro sur tout échec partiel, ou tester les trois pointeurs. | Lu. |
| A01-05 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:206-215` | Si `launch_deband` échoue après `launch_ingest`, la fonction rend l'erreur sans `hold(frame)` : l'image NVDEC repart au pool de FFmpeg alors que le noyau d'ingestion déjà enfilé la lit encore. | Échec de lancement du débandage (ressources) : l'ingestion en vol lit une surface que le décodeur réécrit. L'erreur arrête de toute façon l'image, la conséquence est limitée. | Appeler `hold(frame)` dès que l'ingestion est enfilée, avant le débandage. | Lu. |
| A01-06 | Mineur | `src/gpu/cuda/backend/tensorrt_textures.cpp:44`, `:70-72`, `tensorrt_backend.cpp:391`, `tensorrt_export.cpp:250-252` | Appels non vérifiés restants sur le chemin de lecture : `cudaStreamSynchronize` de repli (textures, `unmap_fences`), `cudaFree` de repli dans `free_after_lanes`. Les `cudaGetLastError()` des textures effacent aussi une erreur non collante laissée par un appel précédent sans rapport. Les autres appels non vérifiés de la zone sont des destructions et sont acceptables. | Un échec de création d'événement (rare) vide une voie entière en pleine lecture sans trace au journal ; une erreur d'un lancement précédent disparaît avant d'avoir été lue. | Passer ces appels par `cuda_warn` ; remplacer `cudaGetLastError()` par `clear_error()` seulement après l'appel qui a échoué, pas en aveugle. | Lu. |
| A01-07 | Mineur | `src/gpu/cuda/backend/pixel_kernels.cuh:23-38` | `block_reduce` réutilise un tableau partagé sans `__syncthreads()` final : deux appels dans le même noyau font courir une course (une chaîne écrit `warps` pendant que la chaîne 0 lit encore l'appel précédent). Un seul appelant aujourd'hui (`rife_kernels.cu:105`) : le défaut est latent. | Un futur noyau qui réduit deux valeurs (somme puis maximum) obtient des résultats faux par intermittence. | Ajouter `__syncthreads()` avant le retour, ou le dire dans le commentaire du contrat. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-06 | Toujours présent | `tensorrt_backend.cpp:394-405` règle le seuil du pool par défaut de l'appareil à `UINT64_MAX`, et rien n'appelle `cudaMemPoolTrimTo` dans `src/`. Aggravation : le pool appartient à l'appareil, pas au backend ; détruire le backend (fin de chaîne, repli sur Vulkan dans le même processus) ne rend rien. La zone d'activation ne rétrécit jamais non plus (`trt_engine.cpp:78-113`). À confirmer : `nvidia-smi --query-gpu=memory.used` pendant une 4K, après retour à une 1080p, puis après un repli sur Vulkan. Correction : `cudaMemPoolTrimTo(pool, 0)` à la destruction du backend et à chaque réouverture de chaîne. |
| TRT-12 | Corrigé | `begin_slot` fait `join()` avant que `slot()` pose `retired_done_` (`tensorrt_kernels.cpp:305-307` puis `:256`) : l'événement suit le travail de toutes les voies. La jointure introduit en revanche A01-02. |
| TRT-13 | Toujours présent | `~TensorRtBackend` rend la main si `stream_` est nul (`tensorrt_backend.cpp:203`), et `stream_` n'est posé qu'à la fin de `create_lanes` (`tensorrt_lanes.cpp:50`). Un échec au milieu de `create_lanes` (deuxième file, ou l'un des 64 événements d'une voie) laisse fuir files et événements déjà créés. Les échecs plus tardifs de `create` passent par le destructeur complet. Correction : appeler `destroy_lanes()` sans condition, et ne garder le retour anticipé que pour le reste. |

Autres points de la fiche :

- **Appels CUDA non vérifiés (18 relevés).** Ceux du chemin de lecture sont dans A01-06 ; les autres sont des
  destructions (`~ImportedSlot`, `~GpuTiming`, `~HostUploads`, `~TensorRtBackend`), acceptables.
- **Voies.** Trois voies à partir de 7,5 Gio (`tensorrt_backend.h:139`) : une carte de 8 Go les garde. Priorité
  haute à l'agrandissement. Les attentes passent par un anneau de 64 événements par voie : un rang ancien se lit sur
  un événement plus récent, attente plus forte, jamais plus faible (`tensorrt_lanes.cpp`). Juste, hormis A01-02.
- **Graphes CUDA.** Capturés par jeu d'adresses, jetés quand la zone d'activation change de génération ou quand la
  priorité des files auxiliaires change (`trt_engine.cpp:218-222`, `:247-270`). Les formes des moteurs sont fixes :
  un changement de taille passe par de nouveaux moteurs, donc de nouveaux graphes. Pas de défaut.
- **WDDM.** `push_launches` à chaque étape et après chaque signal de l'anneau (`tensorrt_backend.cpp:501-508`,
  `ring_slot.cpp:169-171`). La parité des temps reste à mesurer (voir questions).

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 02 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:78-89` | La queue du SPAN FP8 est enfilée après `begin_slot`, donc après la jointure de génération : voir A01-02, la correction touche aussi cet appelant. |
| 21 | `src/gpu/graph_book.h` (`captured`, `launch_failed`) | Les compteurs d'échecs de capture et de rejeu sont cumulés sur toute la vie du moteur et jamais remis à zéro : trois échecs isolés en plusieurs heures coupent les graphes pour de bon. Les jeux d'adresses ne sont jamais évincés : au-delà de 8, lancements directs. |
| 21 | `src/gpu/lanes.h` (`states_`) | La table des ressources de `LaneHazards` ne perd jamais une adresse. Bornée en pratique par le pool qui réutilise les adresses, mais rien ne la purge entre deux chaînes du même backend. |
| 30 | `src/app/play/chain_setup.h:64-93`, `trt_engine.cpp:115-119` | Les moteurs sont détruits avant le backend, sans attente de leur travail en vol : `~TrtEngine` détruit le contexte TensorRT et les graphes sans synchroniser sa file. À vérifier que l'arrêt de l'étage (`UpscaleStage::stop`) attend la fin du GPU avant. |
| 31 | `src/app/play/ring_stage.cpp:58-66` | `health()` n'est appelé que pendant l'attente d'un emplacement : une perte du GPU pendant une image n'est vue qu'au prochain appel qui échoue. Avec A01-01, le message peut être le mauvais. |

## Points positifs

- Mémoire rendue derrière toutes les voies par `cudaFreeAsync` (`free_after_lanes`), sans `cudaFree` qui
  synchroniserait l'appareil en pleine lecture.
- `LaneHazards` déduit les dépendances des ressources : l'appelant ne pose aucune attente, une erreur de voie coûte du
  recouvrement, pas un résultat faux. Pur et testable sans GPU.
- Import de l'anneau soigné : descripteurs dupliqués et fermés sur tous les échecs, repli sans le drapeau « dédiée »
  pour les images sous-allouées (bandes du pilote Windows), repli 32 bits pour le RGB10A2.
- Erreurs collantes rendues en `DeviceLost` avec un message qui dit quoi faire ; erreurs non collantes journalisées
  et effacées (`cuda_warn`).
- Graphes CUDA avec lancement chaud, capture, rejeu, et un registre qui le dit au journal quand il se coupe.
- Profil GPU par étape et par rôle, sans attente hôte.

## Questions ouvertes

- WDDM contre Linux : mêmes temps et même stabilité ? À mesurer sur la 4070 Ti, même fichier, `--one-lane` et voies
  multiples, profil GPU de fin de lecture sous Windows 11 (HAGS actif et inactif) et sous CachyOS.
- La file de transfert du décodage logiciel (`host_upload.cpp`) n'est jamais « poussée » sous WDDM : la copie
  part-elle avec le lot de la voie, ou une image plus tard ? À mesurer avec un fichier H.264 10 bits sous Windows.
- Le seuil de 7,5 Gio suffit-il pour les cartes de 8 Go en 4K avec deux zones d'activation et le pool qui ne rend
  rien (TRT-06) ? À confirmer avec `nvidia-smi` sur une carte de 8 Go si Tatoub en a une.
