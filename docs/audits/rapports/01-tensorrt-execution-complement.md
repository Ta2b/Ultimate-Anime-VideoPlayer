# Complément de l'audit 01. Renvois reçus des autres audits

Branche : `claude/project-thread-dkb4u3`. Commit audité : `d5df90d`. Date : `2026-11-09`.

## Verdict

Sur huit renvois, un seul ouvre un défaut nouveau (A01-08, la colle de RIFE fusionné bridée sur RTX 50 sans
mesure) ; trois sont des doublons (A02-03, A05-05, A15-01, plus une précision sur A30-01) et trois sont réfutés à
la lecture, dont le renvoi sur `write_ring`, dont la preuve vaut aussi contre A02-08.

## Cadrage retenu

Celui du rapport 01 (RTX 4070 Ti de référence, perte du GPU détectée sans reprise, aucune question à Tatoub). Le
fil tourne sans carte NVIDIA ni CUDA : rien n'a été compilé ni lancé, chaque conclusion vient de la lecture du code
au commit `d5df90d`.

## Renvois traités

| Renvoi | De | Conclusion | Preuve |
|---|---|---|---|
| `tensorrt_kernels.cpp:325`, `write_ring` ne signale pas l'emplacement commencé | 02 | **Réfuté.** Un emplacement non signalé n'a personne pour l'attendre. Quand `produce` échoue, `RingStage` appelle `fail()`, qui pose `failed_` et arrête l'étage (`ring_stage.cpp:107`), puis sort de la boucle (`break`) avant `init_.ring->try_push(ring)` (`ring_stage.cpp:179-200`) : aucun `RingFrame` n'est publié pour cette valeur. Le côté Vulkan n'attend `ready_value` que pour un `RingFrame` reçu (`placebo_ring.cpp:203-230`, `placebo_screen.cpp:226`). Les images suivantes sont jetées (`ring_stage.cpp:117`). L'attente déjà enfilée par `begin_slot` (valeur rendue par libplacebo) se résout normalement. Les mêmes faits valent pour les deux autres appelants de `begin_slot` : `launch_into` (`tensorrt_backend.h:234-246`) et `enqueue_span_fp8` (`tensorrt_span_fp8.cpp:89-105`). | Lu. **Conséquence pour A02-08 :** le scénario de ce défaut (« la présentation peut attendre cet emplacement ») ne tient pas pour la même raison. Il est à revoir : au mieux un point de robustesse, si un jour un échec d'image cesse d'arrêter l'étage. |
| `tensorrt_lanes.cpp:75`, la zone d'activation partagée suppose que tout moteur qui y écrit se nomme | 02 | **Doublon de A02-03** (et de A03-05, même cause). Tous les chemins d'exécution se nomment et s'ordonnent : la zone figure dans les écritures déclarées à `order()` (`tensorrt_backend.cpp:431-437`, `tensorrt_span_fp8.cpp:72-78`, `tensorrt_rife_fused.cpp:91-97`, `tensorrt_drba_fused.cpp:88-94`), et chaque moteur appelle `set_last_user` (`trt_engine.cpp:205`, `span_fp8_engine.cpp:193`, `rife_fused_engine.cpp:290`, `drba_fused_engine.cpp:316`). Seuls les essais au chargement y écrivent sur un flux privé, hors de l'ordre des voies : SPAN FP8 (A02-03), RIFE fusionné (A03-05), et aussi DistilDRBA fusionné (`drba_fused_engine.cpp:174-179`), qui se nomme mais ne s'ordonne pas. La correction de A02-03 doit couvrir les trois essais. La zone 1 servie à deux voies (Mesure et Interpolation) est sûre : son adresse est une ressource de `LaneHazards`. | Lu. Détail à reprendre avec la correction : l'essai de DistilDRBA laisse `kept_` sur les adresses des tampons d'essai (`drba_fused_engine.cpp:180-185` ne le remet pas à zéro). Sans effet aujourd'hui (les tampons réels existent déjà et ne peuvent pas avoir ces adresses), mais un `kept_.reset()` à la fin de `load` est plus sûr. |
| `tensorrt_lanes.cpp:80-91`, `SharedActivations::reserve` synchronise la file de la voie à chaque croissance | 03 | **Réfuté comme défaut.** La synchronisation (`cudaStreamSynchronize` après `cudaMallocAsync`) n'a lieu qu'à la croissance de la zone : `reserve` rend tout de suite si la taille suffit (`trt_engine.cpp:79`). La zone grandit au chargement d'un moteur (`use_zone`, `TrtEngine::load`) ou au premier passage d'un moteur sur une autre voie (`place`, `trt_engine.cpp:211-215`), jamais à chaque image. L'attente porte au plus sur le travail en vol d'une voie (une image de GPU), à côté de la désérialisation d'un moteur qui coûte des centaines de millisecondes. Elle est aussi nécessaire : une allocation ordonnée par la file n'est utilisable par une autre file (la zone 1 sert à deux voies) qu'après un événement ou une synchronisation. Le `cudaFree` de repli sur une allocation asynchrone est permis. | Lu. Le chemin de repli `cudaDeviceSynchronize` (`trt_engine.cpp:92`) ne sert que sous manque de mémoire. |
| `tensorrt_textures.cpp:15-78`, objets texture du warp gardés par adresse, réutilisés après une réallocation à la même adresse | 03 | **Réfuté.** Un objet texture sur mémoire linéaire (`cudaResourceTypePitch2D`) ne décrit qu'une adresse virtuelle, une géométrie (largeur, lignes, pas) et un format (un `__half`). La clé (`TextureKey{address, width, rows}`, `tensorrt_textures.cpp:17-21`) porte l'adresse avec son décalage, la largeur et le nombre de lignes, le pas en découle (`2 × width`), et seul le FP16 est admis (`:16`). À clé égale, l'objet décrit exactement la même mémoire : une adresse libérée puis rendue à un autre tenseur de même forme se lit juste, comme le dit le commentaire d'en-tête. Une forme différente donne une autre clé, donc un nouvel objet. Un objet ne sert que si un tenseur vivant porte cette adresse : jamais de lecture d'une mémoire rendue. Les objets retirés attendent un événement par voie (`trim_textures`, `:53-78`). | Lu. |
| `entry.cpp:56-59`, `pick_trtexec` appelé une fois, empreinte jamais revérifiée | 05 | **Doublon de A05-05.** Le rapport 05 cite déjà cette ligne (`entry.cpp:58`) comme cause : `trtexec_path()` n'est appelé qu'à `ajp_backend_create` (`entry.cpp:87`), puis le chemin reste celui du backend. La correction de A05-05 (revérifier dans `TrtexecBuilder::run` juste avant `run_process`) suffit. | Lu. |
| `tensorrt_kernels.cpp:24-41`, l'ingestion TensorRT ne lit ni 4:4:4 ni 4:2:2 en mémoire CUDA | 15 | **Doublon de A15-01** (cause côté backend). `ingest_format` n'admet en mémoire CUDA que NV12, P010 et P016, et le 4:2:0 planaire en mémoire hôte seulement (`tensorrt_kernels.cpp:25-41`) ; `import_formats` annonce la même liste (`tensorrt_backend.cpp:285`). Le rapport 15 propose déjà l'ingestion 4:4:4 (le format `kernels::Format::Yuv444p16` existe) ; elle garderait la chroma pleine au lieu de la réduire (A15-07), ce qui est aussi le bon choix pour la qualité. | Lu. |
| `tensorrt_backend.cpp:201-204`, `drain()` à placer dans `~TensorRtBackend` | 30 | **Doublon de A30-01, avec une précision sur la correction.** Placer l'attente dans `~TensorRtBackend` serait trop tard : `ChainEngines` détruit les moteurs avant le backend (`chain_setup.h:64`, ordre de déclaration), et le destructeur du backend synchronise déjà toutes les voies (`destroy_textures` puis `destroy_lanes`, `tensorrt_lanes.cpp:55-58`). L'attente doit précéder la destruction des moteurs : `Backend::drain()` appelé par la fermeture de la chaîne avant `rife_motion_calm`, `rife_motion`, `rife`, `drba` et `engine`, ou une synchronisation de la dernière file dans `~TrtEngine` et les deux moteurs fusionnés, comme le propose déjà A30-01. | Lu. |
| `rife_fused_kernels.cu:141-147`, sur sm_120, colle de RIFE fusionné à deux blocs par SM | 37 | **Confirmé : A01-08.** | Voir ci-dessous. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A01-08 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:141-147` | Sur sm_120 (RTX 50), `fused_glue` passe de `__launch_bounds__(128, 4)` à `(128, 2)` pour éviter les débordements que ptxas refuse (`--warn-on-spills` en erreur, `src/gpu/cuda/backend/CMakeLists.txt:14`). Le plafond de registres monte de 128 à 255 par fil, et l'occupation garantie tombe de 16 à 8 warps par SM. Ce noyau lit en dispersé (warp bilinéaire de 14 canaux, jusqu'à 4 prises par pixel en `reduce` > 1) : il vit du recouvrement de la latence mémoire. Aucune mesure ne dit ce que coûte ce choix sur RTX 50, ni si `(128, 3)` (170 registres) suffirait. La qualité de l'image n'est pas touchée, seulement le temps. | RTX 5070 ou 5080, RIFE fusionné en ×2 ou plus : la colle (trois appels par passe) peut prendre une part plus grande de la passe que sur la 4070 Ti, sans que rien ne le dise. Le budget de l'Auto absorbe la différence en baissant le facteur d'interpolation plus tôt. | Mesurer sur une RTX 50 : `ncu --kernel-name fused_glue --section Occupancy --section SpeedOfLight`, registres par fil et temps par appel, avec `(128, 2)` puis `(128, 3)`. Garder le plus rapide qui compile sans débordement. Sans RTX 50 sous la main, au moins relever `ptxas --verbose` (déjà actif) pour sm_120 et sm_89 et noter le nombre de registres dans le commentaire. Piste plus large : réduire la pression de registres (`float acc[28]` vivant sur toute la boucle des prises) en écrivant les 14 canaux du warp avant de lire le flot du bloc suivant. | Lu. À confirmer : la mesure `ncu` ci-dessus. |

## Défauts connus revérifiés

Aucun nouveau : TRT-06, TRT-12 et TRT-13 sont traités au rapport 01.

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 02 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:104` | A02-08 : le scénario « la présentation attend l'emplacement » est réfuté par la même preuve que le premier renvoi ci-dessus (`ring_stage.cpp:107`, `:179-200`). À reclasser par le coordinateur. |
| 02 | `src/gpu/cuda/backend/drba_fused_engine.cpp:179-185` | L'essai au chargement de DistilDRBA fusionné écrit aussi la zone partagée sur un flux privé : à inclure dans la correction de A02-03. |

## Points positifs

- Tous les chemins d'exécution des quatre familles de moteurs déclarent la zone d'activation comme ressource
  écrite, et se nomment : la règle de la zone tient partout hors des essais au chargement.
- Le cache d'objets texture est juste par construction, et il ne vide jamais les files en pleine lecture.
- Un échec d'image arrête l'étage sans laisser au côté Vulkan une valeur à attendre.

## Questions ouvertes

- Aucune carte RTX 50 n'est à disposition du projet à ce jour : A01-08, comme A37-01, attend une mesure sur cette
  famille.
