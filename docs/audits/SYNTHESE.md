# Synthèse des audits

Écrit par le coordinateur seul, après chaque fusion d'un rapport. Les fils le lisent et ne le modifient pas.

## Bilan

À écrire à la fin de toutes les vagues.

## Avancement

| Audit | Statut | Rapport |
|---|---|---|
| 01. TensorRT : exécution, voies, anneau et mémoire | Fusionné (7 défauts, 1 majeur) | [rapport](rapports/01-tensorrt-execution.md) |
| 02. SPAN FP8 : noyaux CUTLASS et dérivation | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/02-span-fp8.md) |
| 03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp) | Fusionné (12 défauts, 3 majeurs) | [rapport](rapports/03-interpolation-tensorrt.md) |
| 04. Analyse et découpe des modèles ONNX | Fusionné (11 défauts, 1 majeur) | [rapport](rapports/04-onnx-modeles.md) |
| 05. Construction des moteurs, cache et trtexec | Fusionné (8 défauts, 1 majeur) | [rapport](rapports/05-construction-moteurs.md) |
| 06. Installation des moteurs et téléchargements | Fusionné (12 défauts, 2 majeurs) | [rapport](rapports/06-installation-moteurs.md) |
| 07. Calibration, niveaux et mode Auto | En cours | |
| 08. Boucle du lecteur et ouverture de la chaîne | En cours | |
| 09. Étages d'agrandissement et d'interpolation | À faire | |
| 10. Rendu et présentation par libplacebo | À faire | |
| 11. Écran et matériel : EDID, VRR, HDR du système, télémétrie | À faire | |
| 12. Cadence et horloges | À faire | |
| 13. Session de lecture : seek, épisodes et reprise | À faire | |
| 14. Sources, démultiplexage et erreurs d'E/S | À faire | |
| 15. Décodage, matériel, formats et couleurs | À faire | |
| 16. Audio | À faire | |
| 17. Sous-titres | À faire | |
| 18. Démarrage et arrêt | À faire | |
| 19. Intégration au système | À faire | |
| 20. Socle commun (core) | À faire | |
| 21. Frontière des moteurs (interface et ABI) | À faire | |
| 22. Réglages : modèle, application et persistance | À faire | |
| 23. Réglages : le panneau | À faire | |
| 24. Interface : rendu, contrôles et messages | À faire | |
| 25. Assistant de premier lancement | À faire | |
| 26. Mise à jour automatique et signature | À faire | |
| 27. Build, dépendances et ports | À faire | |
| 28. CI, publication et signature | À faire | |
| 29. Parcours utilisateur de bout en bout | À faire | |
| 30. Concurrence, threads et durée de vie | À faire | |
| 31. États dégradés et reprise sur erreur | À faire | |
| 32. Budget par image sur la cible | À faire | |
| 33. Longue durée, mémoire et ressources | À faire | |
| 34. Parité Windows et CachyOS | À faire | |
| 35. Entrées non sûres | À faire | |
| 36. État persistant et compatibilité entre versions | À faire | |
| 37. Configurations NVIDIA variées | À faire | |
| 38. Promesses tenues | À faire | |
| 39. Diagnostic et support | À faire | |
| 40. Tests et couverture | À faire | |
| 41. HDR, grade et looks | À faire | |
| 42. Analyse des plans, coupes et pré-analyse | À faire | |
| 43. Textes et traductions | À faire | |
| 44. Représentation intermédiaire et import ONNX | À faire | |
| 45. Moteur Vulkan : calcul, noyaux et shaders | À faire | |
| 46. Outils de recherche, modèles et données livrées | À faire | |
| 47. Export vidéo | Ne pas lancer (hors cible) | |
| 48. Moteur Core ML (macOS) | Ne pas lancer (hors cible) | |

## Défauts trouvés

| ID | Gravité | Où | Défaut | Audit | Doublon de |
|---|---|---|---|---|---|
| A01-01 | Majeur | `src/gpu/cuda/backend/cuda_status.h:15-32` | Perte du GPU mal reconnue : `sticky` ignore 999, 46 et 911, `health()` répond OK après un TDR ou un Xid (à confirmer) | 01 | |
| A01-02 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:305-307` | `join()` au milieu d'une opération ordonnée : au changement de génération de l'anneau, une autre voie peut lire une sortie à moitié écrite | 01 | |
| A01-03 | Mineur | `src/gpu/cuda/backend/host_upload.cpp:114` | Copie du décodage logiciel sur le thread de soumission, `cudaFree` synchrones au changement de taille | 01 | |
| A01-04 | Mineur | `src/gpu/cuda/backend/tensorrt_stats.cpp:32-37` | Échec partiel d'allocation d'un emplacement de relecture jamais réinitialisé | 01 | |
| A01-05 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:206-215` | Image NVDEC rendue au pool alors que l'ingestion la lit encore, si le débandage échoue | 01 | |
| A01-06 | Mineur | `src/gpu/cuda/backend/tensorrt_textures.cpp:44` | Appels CUDA non vérifiés sur le chemin de lecture, `cudaGetLastError()` en aveugle | 01 | |
| A01-07 | Mineur | `src/gpu/cuda/backend/pixel_kernels.cuh:23-38` | `block_reduce` sans `__syncthreads()` final (latent) | 01 | |
| A02-01 | Majeur | `src/gpu/span_fp8/metrics.cpp:39` | La porte de qualité compte une image avec des NaN à 99 dB et la fait passer (démontré) | 02 | |
| A02-02 | Mineur | `src/gpu/cuda/backend/span_fp8_kernels.cuh:358` | Mémoire partagée de W1 réservée dans les trois modes : noyaux FP16 au-delà de 48 Ko, un bloc par SM | 02 | |
| A02-03 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:43-54` | Essai au chargement dans la zone d'activation partagée, sans ordre ni `set_last_user` : peut écraser les tampons de DistilDRBA | 02 | |
| A02-04 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:57-60` | Sonde de 8 192 octets lue sans vérifier la taille de la sortie | 02 | |
| A02-05 | Mineur | `src/gpu/cuda/backend/span_fp8_kernels.cuh:103` | Noyaux dans un espace de noms nommé : souches pliées entre deux `.cu`, attribut d'opt-in fragile | 02 | |
| A02-06 | Mineur | `src/gpu/span_fp8/onnx_weights.cpp:119-126` | Type d'un poids ONNX non vérifié : lecture hors bornes sur un initialiseur entier | 02 | |
| A02-07 | Mineur | `src/gpu/span_fp8/onnx_weights.cpp:236-242` | Lecture du fichier ONNX sans plafond, `tellg` à −1 non traité | 02 | |
| A02-08 | Mineur | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:104` | Emplacement de l'anneau jamais signalé si to_pixel échoue après `begin_slot` | 02 | |
| A02-09 | Mineur | `src/gpu/span_fp8/derive.h:17` | `kE4m3Max` défini deux fois ; la quantification testée n'est pas celle qui tourne | 02 | |
| A02-10 | Mineur | `src/gpu/cuda/backend/span_fp8_net.cu:248` | Contrats divergents : 1 bloc accepté à la lecture, refusé par les noyaux ; commentaire sm 8.9 seulement | 02 | |
| A02-11 | Mineur | `src/gpu/cuda/backend/span_fp8_run.cpp:88-100` | `ajp_span_fp8_run` recopie `span_fp8_view` et oublie `acc16` | 02 | |
| A02-12 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:96` | Quatre `path::string()` contre la règle de la CI | 02 | |
| A02-13 | Mineur | `tests/unit/gpu_span_fp8/metrics_test.cpp` | Aucun test par défaut de la concordance numérique FP8 ni des entrées non finies | 02 | |
| A03-01 | Majeur | `src/gpu/cuda/backend/slice_kernels.cu:24-45` | L'ingestion TensorRT rééchantillonne la chroma en bilinéaire au lieu du Spline36 du contrat (et de Vulkan) | 03 | |
| A03-02 | Majeur | `tests/unit/backend_tensorrt/rife_fused_test.cpp:397` | Concordance des chaînes fusionnées vérifiée seulement par des tests cachés, à 30 dB pour RIFE | 03 | |
| A03-03 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:284-290` | Variante mouvement de DistilDRBA fusionné sans aucun test ni essai au chargement | 03 | |
| A03-04 | Mineur | `src/gpu/cuda/backend/tensorrt_rife.cpp:118-124` | Plan d'instant compact : un moteur RIFE entier lancé ensuite lit un instant périmé (latent) | 03 | |
| A03-05 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:175-194` | Essai au chargement sur un flux privé dans la zone d'activation partagée | 03 | A02-03 (même cause) |
| A03-06 | Mineur | `src/gpu/cuda/backend/tensorrt_rife_fused.cpp:388` | `path::string()` dans les messages de refus : exception sous Windows avec un profil non ASCII | 03 | |
| A03-07 | Mineur | `src/gpu/cuda/backend/rife_kernels.cu:49-50` | `rife_pair` écrit le plan d'instant entier alors que RIFE fusionné n'en lit qu'un élément | 03 | |
| A03-08 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:221-224` | Chaque paire recopie et réencode les deux images, alors que A(n+1) = B(n) | 03 | |
| A03-09 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:323-324` | Cache de graphes plafonné sans éviction ni notice (latent) | 03 | |
| A03-10 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:361` | Environ 150 lignes dupliquées entre les deux moteurs fusionnés, déjà divergentes | 03 | |
| A03-11 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:241` | `__expf` approché pour RIFE, `expf` pour DistilDRBA | 03 | |
| A03-12 | Mineur | `src/gpu/cuda/backend/slice_kernels.cu:268-273` | `ingest_tiled` compte le poids d'une prise hors tuile avant de la sauter (latent) | 03 | |
| A05-01 | Majeur | `src/prepare/process_posix.cpp:127` | Sous Linux, `trtexec` survit à la mort du lecteur et fausse la construction suivante (démontré) | 05 | |
| A05-02 | Mineur | `src/prepare/key_lock.h:26` | Délais qui ne s'emboîtent pas : l'attente du verrou de clé abandonne avant la fin possible du détenteur | 05 | |
| A05-03 | Mineur | `src/prepare/builder.cpp:86-93` | Aucun repli si le timing cache persistant fait échouer `trtexec` | 05 | |
| A05-04 | Mineur | `src/prepare/cache_io.cpp:95` | Publication en un seul renommage : un antivirus sous Windows fait jeter la construction | 05 | |
| A05-05 | Mineur | `src/prepare/trtexec_command.cpp:162-169` | Empreinte de `trtexec` vérifiée une fois à la création, repli de développement sans vérification | 05 | |
| A05-06 | Mineur | `src/prepare/cache.cpp:52-53` | `intact()` écrit une empreinte manquante sans tenir le verrou | 05 | |
| A05-07 | Mineur | `src/prepare/trtexec_progress.cpp:21-22` | Barre figée à 10 % pendant l'essentiel de la construction | 05 | |
| A05-08 | Mineur | `src/prepare/preparer.cpp` | Environ 700 lignes de code mort (`preparer`, `receipt`, `rebuild`, `rejected_sha256`…) | 05 | |
| A06-01 | Majeur | `src/app/play/engine_downloads.cpp:123` | Une version TensorRT changée dans l'index fait perdre le moteur choisi sans un mot (Vulkan à la place), l'ancienne version n'est jamais purgée | 06 | |
| A06-02 | Majeur | `src/app/play/engine_install.cpp:293` | Windows : chemins passés à curl et tar en page ANSI, l'installation échoue avec un compte accentué | 06 | |
| A06-03 | Mineur | `src/app/play/engine_downloads.cpp:130-166` | Module refusé pour `AbiMismatch` réinstallé depuis le même paquet à chaque lancement (TRT-03) | 06 | |
| A06-04 | Mineur | `src/app/play/engine_install.cpp:271` | `.part` nommé par le fichier et non par son SHA, `.part` orphelins jamais purgés (TRT-10) | 06 | |
| A06-05 | Mineur | `src/app/play/engine_install.cpp:338-346` | Décompression sans plafond (SEC-14, démontré) | 06 | |
| A06-06 | Mineur | `src/app/play/engine_install.cpp:213-224` | Estimation de place qui ignore les liens durs : mise à jour refusée à tort (TRT-11) | 06 | |
| A06-07 | Mineur | `src/app/play/engine_install.cpp:290-306` | Réponse 200 à une requête `--range` collée au `.part` | 06 | |
| A06-08 | Mineur | `src/app/play/engine_install.cpp:585-592` | Arrêt entre deux renommages : les deux seules copies complètes effacées au démarrage suivant | 06 | |
| A06-09 | Mineur | `src/app/play/engine_install.cpp:555` | Windows : dossier non renommable tant qu'une DLL TensorRT reste chargée | 06 | |
| A06-10 | Mineur | `src/app/play/engine_install.cpp:243-249` | `reuse_vendor_file` rehache 1 Go à chaque installation, exception possible sur un dossier illisible | 06 | |
| A06-11 | Mineur | `src/core/model_index.h:61-69` | Code mort de l'index des modèles ; MIGraphX, OpenVINO et DirectML encore « téléchargeables » | 06 | |
| A06-12 | Mineur | `tests/unit/app/engine_install_test.cpp` | Aucun test des mises à jour, reprises, plafonds et du SHA-256 portable | 06 | |
| A04-01 | Majeur | `src/prepare/onnx_scan.cpp:165-168` | Emplacement de poids externe en UTF-8 invalide : exception, donc plantage sous Windows (à confirmer) | 04 | |
| A04-02 | Mineur | `src/prepare/onnx_split.cpp:125` | Produit des dimensions en `int64_t` qui déborde avant le plafond (comportement indéfini) | 04 | |
| A04-03 | Mineur | `src/prepare/rife_split.cpp:210-231` | `analyze_rife` ne reconnaît pas toute la structure promise par l'en-tête | 04 | |
| A04-04 | Mineur | `src/prepare/rife_split.cpp:381` | Noms des morceaux par `stem().string()` : exception sous Windows hors page de code | 04 | |
| A04-05 | Mineur | `src/prepare/model_source.cpp:36-43` | `escapes_folder` aveugle pour un `--model` sans dossier | 04 | |
| A04-06 | Mineur | `src/prepare/model_source.cpp:71-111` | `read_model` ne vérifie pas les sorties (facteur, plans) avant une longue construction | 04 | |
| A04-07 | Mineur | `src/prepare/rife_split.h:78` | Morceaux dérivés jamais réécrits tant que `fusion1` ne change pas | 04 | |
| A04-08 | Mineur | `tests/unit/prepare/rife_split_test.cpp:461-499` | Aucun test de CI ne montre que la découpe garde le calcul du réseau entier (SEC-21) | 04 | |
| A04-09 | Mineur | `src/prepare/onnx_metadata.cpp` | Code mort depuis le retrait de GMFSS | 04 | |
| A04-10 | Mineur | `src/prepare/onnx_scan.cpp:33-91` | Second décodeur protobuf, divergent de celui de `gpu/ir` | 04 | |
| A04-11 | Mineur | `src/prepare/rife_split.cpp:30-90` | Doublons entre `rife_split`, `drba_split` et les utilitaires ONNX | 04 | |

## Renvois

| De | Vers | Où | Constat | Statut |
|---|---|---|---|---|
| 01 | 02 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:78-89` | La queue du SPAN FP8 est enfilée après la jointure de génération (A01-02) | En attente (02 fini) |
| 01 | 21 | `src/gpu/graph_book.h` | Compteurs d'échecs des graphes jamais remis à zéro, jeux d'adresses jamais évincés | Ouvert |
| 01 | 21 | `src/gpu/lanes.h` | La table des ressources de `LaneHazards` ne perd jamais une adresse | Ouvert |
| 01 | 30 | `src/app/play/chain_setup.h:64-93` | Moteurs détruits avant le backend sans attendre leur travail en vol | Ouvert |
| 01 | 31 | `src/app/play/ring_stage.cpp:58-66` | `health()` n'est appelé que pendant l'attente d'un emplacement | Ouvert |
| 02 | 34 | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | FP8 éteint hors Linux : sous Windows 11, le cran Performance reste en FP16 | Ouvert |
| 02 | 01 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:325` | `write_ring` rend l'erreur sans signaler l'emplacement commencé (comme A02-08) | En attente (01 fini) |
| 02 | 07 | `src/app/play/span_int8.cpp:166` | `size_key` plafonné à 1920×1088 : la taille validée n'est pas celle du moteur au-delà | Ouvert |
| 02 | 07 | `src/app/play/span_int8.cpp:129-150` | Seul un refus d'une porte plus ancienne se refait | Ouvert |
| 02 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:75` | La zone d'activation partagée suppose que tout moteur qui y écrit se nomme (voir A02-03) | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:80-91` | `SharedActivations::reserve` synchronise la file de la voie à chaque croissance | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_textures.cpp:15-78` | Objets texture du warp gardés par adresse : réutilisés après une réallocation à la même adresse | En attente (01 fini) |
| 03 | 09 | `src/app/play/upscale_motion.cpp:39` | Reprise à chaud absente sous RIFE fusionné ; invalidation sur coupe et saut à vérifier | Ouvert |
| 03 | 09 | `src/app/play/player_engines.cpp:247-266` | Avec DistilDRBA fusionné, le warp repose sur la branche mouvement non testée (A03-03) | Ouvert |
| 03 | 15 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:142-143` | Position de chroma : seuls `Center` et `TopLeft` sont distingués | Ouvert |
| 03 | 40 | `tests/unit/backend_tensorrt/` | La suite CUDA ne tourne qu'à la main sur un poste NVIDIA, aucun garde-fou de CI | Ouvert |
| 05 | 27 | `src/gpu/cuda/backend/CMakeLists.txt:95` | Repli `trtexec` gardé en `RelWithDebInfo` (SEC-20) | Ouvert |
| 05 | 01 | `src/gpu/cuda/backend/entry.cpp:56-59` | `pick_trtexec` appelé une fois : empreinte jamais revérifiée ensuite (A05-05) | En attente (01 fini) |
| 05 | 06 | `src/app/play/engine_install.cpp:130-158` | TRT-17 : échec de `nvidia-smi` gardé pour le processus, carte inconnue rangée en « ptx » | Traité par 06 (TRT-17 en partie présent) |
| 05 | 18 | `src/app/play/crash_dump.cpp:90` | Le gestionnaire de plantage ne tue pas le `trtexec` en cours (A05-01) | Ouvert |
| 05 | 07 | appelants de `Backend::build` | Une construction par taille de source : coût à mesurer, cache sans purge (TRT-17) | Ouvert |
| 05 | 36 | `src/prepare/identity.cpp:161-164` | Jeton `driver` dans la clé : reconstruction complète à chaque saut de CUDA du pilote | Ouvert |
| 06 | 26 | `src/app/play/engine_online.cpp` | `merge_release_index` : SHA ou version changés du paquet TensorRT (voir A06-01) | Ouvert |
| 06 | 28 | `tools/release/package-linux.sh:190` | Version du paquet TensorRT tirée de `ls libnvinfer.so.*` : « 11 » ou « 11.3.0 » | Ouvert |
| 06 | 28 | `tools/release/package-windows.ps1:101` | Origine et stabilité de `$trtVersion` | Ouvert |
| 06 | 25 | `src/app/play/first_run.cpp:125-131` | « Go » et « Mo » en dur dans l'interface anglaise (UX-09) | Ouvert |
| 06 | 34 | `src/app/play/engine_tools.cpp:19-36` | `curl.exe` de System32 : version minimale, révocation Schannel derrière une inspection TLS | Ouvert |
| 06 | 21 | `src/app/play/backend_library.cpp:83-85` | Module refusé refermé : nvinfer se décharge-t-il sous Windows (A06-09) ? | Ouvert |
| 04 | 09 | `src/app/play/derived_model.cpp:19` | `stem().string()` dans le nom des modèles dérivés (comme A04-04) | Ouvert |
| 04 | 09 | `src/app/play/motion_model.cpp:90` | `filename().string()` et `stem().string()` sur le chemin du RIFE | Ouvert |
| 04 | 09 | `src/app/play/upscale_stage.cpp:198` | L'étage suppose `kSpanFactor` pour un `--model` quelconque (voir A04-06) | Ouvert |
| 04 | 28 | `tests/fuzz/fuzz_parsers.cpp:33-37` | Aucune cible de fuzz pour les lecteurs ONNX de `prepare/` (SEC-21) | Ouvert |
| 04 | 05 | `src/prepare/builder.cpp:174` | `read_model` relit et rehache tout l'ONNX avant et après chaque construction | En attente (05 fini) |
