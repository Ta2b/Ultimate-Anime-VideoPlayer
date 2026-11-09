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
| 07. Calibration, niveaux et mode Auto | Fusionné (12 défauts, 3 majeurs) | [rapport](rapports/07-calibration-niveaux.md) |
| 08. Boucle du lecteur et ouverture de la chaîne | Fusionné (14 défauts, 4 majeurs) | [rapport](rapports/08-boucle-chaine.md) |
| 09. Étages d'agrandissement et d'interpolation | Fusionné (16 défauts, 1 majeur) | [rapport](rapports/09-etages-traitement.md) |
| 10. Rendu et présentation par libplacebo | Fusionné (13 défauts, 2 majeurs) | [rapport](rapports/10-presentation-libplacebo.md) |
| 11. Écran et matériel : EDID, VRR, HDR du système, télémétrie | Fusionné (13 défauts, 2 majeurs) | [rapport](rapports/11-ecran-materiel.md) |
| 12. Cadence et horloges | Fusionné (10 défauts, 3 majeurs) | [rapport](rapports/12-cadence-horloges.md) |
| 13. Session de lecture : seek, épisodes et reprise | Fusionné (13 défauts, 1 bloquant, 1 majeur) | [rapport](rapports/13-session-lecture.md) |
| 14. Sources, démultiplexage et erreurs d'E/S | Fusionné (8 défauts, 2 majeurs) | [rapport](rapports/14-sources-demux.md) |
| 15. Décodage, matériel, formats et couleurs | Fusionné (14 défauts, 2 majeurs) | [rapport](rapports/15-decodage-formats.md) |
| 16. Audio | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/16-audio.md) |
| 17. Sous-titres | En cours | |
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
| A08-01 | Majeur | `src/app/play/chain_setup.cpp:166-171` | « Annuler et regarder » pris pour un refus par les replis, qui relancent une autre construction | 08 | |
| A08-02 | Majeur | `src/app/play/player_run.cpp:191` | Le fil de relevé de `ChainStats` lit `kernels_` après la destruction du backend | 08 | |
| A08-03 | Majeur | `src/app/play/chain.cpp:204-211` | Facteur, Auto ou Rapide changés en direct sans refaire anneau, plus haut facteur ni warp | 08 | |
| A08-04 | Majeur | `src/app/play/chain.cpp:351-358` | Raccourcis de séance écrits dans `play.toml` par toute réouverture | 08 | |
| A08-05 | Mineur | `src/app/play/chain_setup.cpp:160-172` | Repli au cran Équilibre sur n'importe quel refus, chaîne repliée absente de `chain.running` | 08 | |
| A08-06 | Mineur | `src/app/play/chain_setup.cpp:376-389` | DistilDRBA : moteur image 720p construit puis jeté quand la dérivation mouvement échoue | 08 | |
| A08-07 | Mineur | `src/app/play/player_run.cpp:549-562` | `follow_display` seulement dans la branche HDR, jamais avec `--sdr` | 08 | |
| A08-08 | Mineur | `src/app/play/player_engines.cpp:54-67` | Événements jetés pendant une construction ; SPAN FP8 sans « Annuler et regarder » | 08 | |
| A08-09 | Mineur | `src/app/play/chain_setup.cpp:277` | `path::string()` sur des chemins du profil, contre la règle de la CI | 08 | |
| A08-10 | Mineur | `src/app/play/engine_build.cpp:38` | Une instance de backend par construction ou lecture du cache, jusqu'à six de plus par ouverture | 08 | |
| A08-11 | Mineur | `src/app/play/player_run.cpp:98` | `run_chain` d'environ 650 lignes, fichier de 753 lignes (QUA-15) | 08 | |
| A08-12 | Mineur | `tests/unit/app/chain_setup_test.cpp` | Aucun test de `apply`, `switch_chain`, `with_kept_choices`, des replis ni de la fin de `run_chain` | 08 | |
| A08-13 | Mineur | `src/app/play/player_run.cpp:334-340` | E/S synchrones sur le thread de la boucle | 08 | |
| A08-14 | Mineur | `src/app/play/engine_switch.cpp:13-19` | L'onglet Moteurs connaît encore MIGraphX et OpenVINO | 08 | A06-11 (même cause) |
| A07-01 | Majeur | `src/app/play/machine_calibration.cpp:139-151` | La calibration mesure une chaîne différente de celle que l'Auto calibré applique | 07 | |
| A07-02 | Majeur | `src/app/play/machine_calibration.cpp:338` | « Tient » ignore les images non interpolées quand le Rapide coupe RIFE | 07 | |
| A07-03 | Majeur | `src/media/bench_clip.cpp:143` | Chemin du clip de calibration en page ANSI sous Windows : la calibration casse | 07 | |
| A07-04 | Mineur | `src/app/play/span_int8.cpp:139-150` | Un reçu FP8 refusé l'est pour toutes les tailles et pour de bon | 07 | |
| A07-05 | Mineur | `src/app/play/auto_config.cpp:194-201` | Bornes calibrées (1080p, 24 i/s, TensorRT) appliquées à toute source et tout moteur | 07 | |
| A07-06 | Mineur | `src/app/play/auto_config.cpp:214-220` | `preset_chain_differs` ignore `auto_free` et `restore_lines` | 07 | |
| A07-07 | Mineur | `src/core/auto_memory.cpp:38` | Mémoire de l'Auto plafonnée à ×6 alors que l'Auto monte à ×8 | 07 | |
| A07-08 | Mineur | `data/gpu_tiers.toml:19-24` | Cartes portables courantes rangées au rang de la carte de bureau | 07 | |
| A07-09 | Mineur | `src/app/play/machine_calibration.cpp:283` | Aucune garantie de mesure à chaud et stable | 07 | |
| A07-10 | Mineur | `src/app/play/auto_bench_run.cpp:133` | Erreur de copie de `play.toml` ignorée : l'essai joue les réglages par défaut | 07 | |
| A07-11 | Mineur | `tests/unit/app/auto_config_test.cpp` | Aucun test de la branche calibrée ni de la boucle de calibration | 07 | |
| A07-12 | Mineur | `src/app/play/auto_bench_run.cpp:53` | Code mort et deux mécanismes de banc côte à côte | 07 | |
| A09-01 | Majeur | `src/app/play/upscale_drba.cpp:104` | Côté « après » de DistilDRBA, son pont et le Smooth jugent les coupes sur le score brut, sans pré-analyse ni bandes noires | 09 | |
| A09-02 | Mineur | `src/app/play/rife_fused_choice.cpp:11` | Le cran Performance de l'interpolation coupe le moteur fusionné | 09 | |
| A09-03 | Mineur | `src/app/play/span_auto.cpp:22-25` | Sous Windows, message faux « n'est pas une RTX 40 » quand le FP8 est éteint à la construction | 09 | |
| A09-04 | Mineur | `src/app/play/upscale_drba.cpp:135-137` | `drba_next` ingère l'image suivante sans vérifier sa taille | 09 | |
| A09-05 | Mineur | `src/app/play/upscale_motion.cpp:111` | Sans poses, deux passes de RIFE pour une seule image intermédiaire hors de ½ | 09 | |
| A09-06 | Mineur | `src/app/play/upscale_stage.cpp:363` | Différence de `pts` sans vérifier `kNoPts` (débordement signé) | 09 | |
| A09-07 | Mineur | `src/app/play/upscale_drba.cpp` | DistilDRBA privé de trois économies de l'étage (avance, saut, débandage) | 09 | |
| A09-08 | Mineur | `src/app/play/settings.h:114-117` | Auto compare la hauteur de sortie à 1080 : vaut toujours Après par défaut | 09 | |
| A09-09 | Mineur | `src/app/play/upscale_stage.cpp:154-161` | L'étage suppose ×2 pour tout `--model` | 09 | |
| A09-10 | Mineur | `src/app/play/derived_model.cpp:19` | Sept `path::string()` sur des chemins du profil | 09 | |
| A09-11 | Mineur | `src/app/play/derived_model.cpp:18-22` | Modèles dérivés jamais purgés | 09 | |
| A09-12 | Mineur | `src/app/play/upscale_stage.cpp:600-605` | Avec DistilDRBA et `--skip-held`, `skip()` met mal `previous_` à jour | 09 | |
| A09-13 | Mineur | `src/app/play/upscale_motion.cpp:39` | `--rife-warm` sans effet sous TensorRT | 09 | |
| A09-14 | Mineur | `src/app/play/upscale_stage.cpp` | Fichier à 776 lignes, `outputs` de CCN 172 | 09 | |
| A09-15 | Mineur | `src/app/play/rife_plan.h:7` | Commentaires faux | 09 | |
| A09-16 | Mineur | `tests/unit/app/upscale_stage_test.cpp` | Plusieurs branches de l'étage sans test | 09 | |
| A10-01 | Majeur | `src/app/play/placebo_screen.cpp:549` | Perte du GPU ou de la surface prise pour une fenêtre cachée : image figée, journal inondé, aucun message (GPU-06) | 10 | |
| A10-02 | Majeur | `src/app/play/placebo_open.cpp:497` | Avec le défaut `--present=wait`, `wp_presentation` n'est jamais ouvert : aucune grille des vblanks sous CachyOS | 10 | |
| A10-03 | Mineur | `src/app/play/ring_watcher.cpp:474` | Le veilleur de l'anneau sort sans un mot sur une erreur de `vkWaitSemaphores` (GPU-10) | 10 | |
| A10-04 | Mineur | `src/app/play/placebo_open.cpp:323` | Attentes sans délai sur un GPU bloqué (GPU-17) | 10 | |
| A10-05 | Mineur | `src/app/play/placebo_screen.cpp:522` | Rien ne dessine pendant la boucle modale de Windows | 10 | |
| A10-06 | Mineur | `src/app/play/placebo_feedback.cpp:241` | Le niveau de retour ne retombe jamais quand rien ne revient | 10 | |
| A10-07 | Mineur | `src/app/play/placebo_feedback.cpp:378` | Sous Windows, domaine d'horloge `CLOCK_MONOTONIC` qui n'y existe pas | 10 | |
| A10-08 | Mineur | `src/app/play/placebo_screen.cpp:569` | Image acquise jamais soumise sur les sorties en erreur | 10 | |
| A10-09 | Mineur | `src/app/play/placebo_ring.cpp:229` | `retire` ignore l'échec de `hand_to_cuda` | 10 | |
| A10-10 | Mineur | `src/app/play/placebo_screen.cpp:632` | `path::string()` dans les instantanés | 10 | |
| A10-11 | Mineur | `src/app/play/placebo_open.cpp:275` | Cache des shaders lu sans plafond, `.part` jamais purgés | 10 | |
| A10-12 | Mineur | `src/app/play/present_level.h:426` | Commentaires faux sur le niveau de présentation par défaut | 10 | |
| A10-13 | Mineur | `tests/unit/app/present_level_test.cpp` | Aucun test des chemins dégradés de la présentation | 10 | |
| A11-01 | Majeur | `src/app/play/display_contract.h:54` | Contrat d'écran figé sur l'écran d'ouverture (GPU-04) | 11 | |
| A11-02 | Majeur | `src/app/play/host_facts.cpp:39-81` | Windows : requêtes d'écran sur le thread de la boucle (WIN-04, GPU-16, à mesurer) | 11 | |
| A11-03 | Mineur | `src/app/play/display_edid.cpp:421-429` | Écran choisi par sous-chaîne du nom : deux écrans du même modèle confondus | 11 | |
| A11-04 | Mineur | `src/app/play/system_peak.cpp:85-108` | Entrée KWin choisie par le seul connecteur | 11 | |
| A11-05 | Mineur | `src/app/play/display_edid.cpp:56-63` | Demi-flottant infini retenu comme pic | 11 | |
| A11-06 | Mineur | `src/app/play/display_contract.cpp:24-42` | Hors KDE, blanc de référence fixé à 203 et marge mal rapportée sous Windows | 11 | |
| A11-07 | Mineur | `src/app/play/host_facts.cpp:247-250` | Windows : pic DXGI demandé seulement si un EDID a été lu | 11 | |
| A11-08 | Mineur | `src/app/play/gpu_telemetry.cpp:311-372` | `GpuTelemetry::sample` écrit sans verrou depuis deux threads (GPU-09) | 11 | |
| A11-09 | Mineur | `src/app/play/host_facts.cpp:378-380` | `on_battery()` sans cache sur le thread de la boucle | 11 | |
| A11-10 | Mineur | `src/app/play/gpu_telemetry.cpp:58` | « chaleur » en dur dans l'OSD anglais | 11 | |
| A11-11 | Mineur | `src/app/play/display_edid.cpp:139-147` | Pic et noir dépendent de l'ordre des blocs EDID | 11 | |
| A11-12 | Mineur | `tests/unit/app/` | Aucun test du contrat d'écran, du multi-écran ni du pic de contenu | 11 | |
| A11-13 | Mineur | `src/app/play/host_facts.cpp:39-81` | Quatre copies de l'énumération `QueryDisplayConfig` | 11 | |
| A12-01 | Majeur | `src/core/vblank_clock.cpp:32-36` | La grille des vblanks apprend la cadence du contenu au lieu de celle de l'écran (démontré) | 12 | |
| A12-02 | Majeur | `src/core/auto_rate.cpp:281-291` | Stable sans palier calé au-dessus de ×1 à 165 Hz pour 23,976, ni à 144 Hz pour 29,97 (démontré) | 12 | |
| A12-03 | Majeur | `src/core/vsync_phase.h:2-5` | Phase des vsyncs apprise sur le retour de `swap_buffers` au niveau Open (à confirmer sous Windows) | 12 | |
| A12-04 | Mineur | `src/core/auto_rate.cpp:293-301` | Paliers calés sur le plafond passé comme fréquence d'écran | 12 | |
| A12-05 | Mineur | `src/core/auto_rate.cpp:599-608` | Réarmement toutes les 8 s même sans palier calé | 12 | |
| A12-06 | Mineur | `src/core/pacing.cpp:61-67` | Calage de vitesse sur une fréquence fausse, jamais sur la phase | 12 | |
| A12-07 | Mineur | `src/core/cadence.cpp:56-70` | `retarget` sort des bornes de `kRatioTerm` | 12 | |
| A12-08 | Mineur | `src/core/steady_rate.cpp:162-166` | Le repli du Rapide compte les pertes du démarrage et des reprises | 12 | |
| A12-09 | Mineur | `src/core/steady_rate.h:37` | Code mort dans les horloges | 12 | |
| A12-10 | Mineur | `tests/unit/core/` | Aucun test de `VsyncPhase`, des paliers à 144 et 165 Hz, ni de `VblankClock` à la demande | 12 | |
| A13-01 | Bloquant | `src/app/play/pipeline.cpp:172` | Index de sous-titres externes gardé à la réouverture ou à l'épisode suivant : déréférence nulle, plantage | 13 | |
| A13-02 | Majeur | `src/app/play/session.cpp:66-68` | Fin de flux après abandon de lecture prise pour une vraie fin : épisode suivant lancé (MED-01) | 13 | |
| A13-03 | Mineur | `src/app/play/episodes.cpp:43-47` | Un titre d'épisode contenant `MENU`, `PV` ou `CM` est classé en bonus | 13 | |
| A13-04 | Mineur | `src/app/play/episodes.cpp:74-93` | Chiffres pleine chasse comparés octet par octet | 13 | |
| A13-05 | Mineur | `src/app/play/session.cpp:102-120` | Pas d'image vers l'avant bloqué en VFR (MED-16) | 13 | |
| A13-06 | Mineur | `src/app/play/session.cpp:78` | Retours de `play()` et `pause()` ignorés : pause perdue sur boîte audio pleine | 13 | |
| A13-07 | Mineur | `src/app/play/session.cpp:229-247` | Après des pas vers l'avant, l'audio reprend à la position de la pause | 13 | |
| A13-08 | Mineur | `src/app/play/session.cpp:189-227` | Seek refusé sans repositionnement : lecture figée (MED-06) | 13 | |
| A13-09 | Mineur | `src/app/play/player_run.cpp:738-739` | Pistes audio et sous-titres gardées pour toute vidéo suivante | 13 | |
| A13-10 | Mineur | `src/app/play/resume.cpp:107-129` | `resume.tsv` réécrit sans verrou entre deux instances | 13 | |
| A13-11 | Mineur | `src/app/play/resume.cpp:33` | `path::string()` dans la reprise et les sous-titres externes | 13 | |
| A13-12 | Mineur | `src/app/play/resume.cpp:39-45` | Windows : comparaison sans casse en ASCII seulement | 13 | |
| A13-13 | Mineur | `tests/unit/app/resume_test.cpp` | Aucun test de la borne, des chemins non ASCII, du VFR ni de la fin sur erreur | 13 | |
| A14-01 | Majeur | `src/media/demuxer.cpp:367` | `AVERROR_EOF` pris pour une vraie fin sans regarder `pb->error` : panne d'E/S en MP4 qui finit l'épisode sans notice | 14 | |
| A14-02 | Majeur | `src/media/demuxer.cpp:252-254` | `rw_timeout` sans effet sur un partage monté : un NAS muet fige la lecture puis la fermeture | 14 | |
| A14-03 | Mineur | `src/media/demuxer.cpp:273-283` | Chapitres ordonnés et éditions Matroska ni gérés ni détectés | 14 | |
| A14-04 | Mineur | `src/media/source_loop.cpp:337-364` | Flush puis Seek refusé : les paquets jetés ne sont pas relus | 14 | |
| A14-05 | Mineur | `src/media/track_language.cpp:44-46` | « VOSTFR » sur une piste audio lue comme du français | 14 | |
| A14-06 | Mineur | `src/media/source.h:34-35` | File audio pleine qui fait attendre toute la lecture sur un fichier mal entrelacé | 14 | |
| A14-07 | Mineur | `src/media/demuxer.cpp:369-377` | Panne d'E/S, interruption et données abîmées toutes en `DecodeFailed` | 14 | |
| A14-08 | Mineur | `tests/unit/media/source_test.cpp:400-414` | Le test fige MED-01 comme comportement voulu | 14 | |
| A15-01 | Majeur | `src/media/video_decoder.cpp:320-323` | 4:4:4, et 4:2:2 sur RTX 50, via NVDEC arrêtent la lecture sous TensorRT (MED-02) | 15 | |
| A15-02 | Majeur | `src/media/deinterlace.cpp:66` | Pas d'IVTC, et images NVDEC entrelacées non désentrelacées (MED-13) | 15 | |
| A15-03 | Mineur | `src/media/video_decoder.cpp:400` | Pas de repli logiciel sur une erreur NVDEC en cours de flux | 15 | |
| A15-04 | Mineur | `src/media/deinterlace.cpp:47-49` | bwdif perd une image au changement de taille (démontré) | 15 | |
| A15-05 | Mineur | `src/media/deinterlace.cpp:79` | bwdif perd le `best_effort_timestamp` (démontré) | 15 | |
| A15-06 | Mineur | `src/media/frame_data.cpp:247-275` | Conversion hôte 4:4:4 sans pool : 7,5 ms et 25 Mo par image 4K (MED-03, mesuré) | 15 | |
| A15-07 | Mineur | `src/media/frame_data.cpp:253` | Chroma 4:4:4 réduite avant SPAN, formats illisibles mal refusés (MED-04) | 15 | |
| A15-08 | Mineur | `src/media/color.cpp:72` | Matrice non étiquetée devinée par la taille même en BT.2020 ou PQ | 15 | |
| A15-09 | Mineur | `src/media/video_decoder.cpp:166-167` | Durée d'image par la période moyenne sans durée de paquet (MED-16) | 15 | A13-05 (même cause) |
| A15-10 | Mineur | `src/media/video_decoder.cpp:167` | `pts + duration` sans garde (MED-14) | 15 | |
| A15-11 | Mineur | `src/media/stills.cpp:295` | `extract_stills` décode jusqu'à 8 threads à priorité normale | 15 | |
| A15-12 | Mineur | `src/media/stills.cpp:230-236` | Corpus de calibration FP8 à chroma au plus proche, différente de la lecture | 15 | |
| A15-13 | Mineur | `tests/unit/media/` | Sans test : conversion hôte, bascule NVDEC, chemin entrelacé | 15 | |
| A15-14 | Mineur | `src/media/hw_device.cpp:204` | Code mort pour le produit | 15 | |
| A16-01 | Majeur | `src/audio/sdl_sink.cpp:151-160` | Sans sortie par défaut après retrait : refus à chaque bloc, gel à chaque seek, appareil rebranché jamais repris (MED-08) | 16 | |
| A16-02 | Mineur | `src/audio/player.cpp:138-151` | Délai de l'appareil retiré gardé sur la sortie par défaut (MED-08) | 16 | |
| A16-03 | Mineur | `src/audio/sdl_sink.cpp:100-113` | Changement de sortie par défaut invisible au module | 16 | |
| A16-04 | Mineur | `src/audio/sdl_sink.cpp:189-199` | Latence du périphérique comptée sur un seul tampon | 16 | |
| A16-05 | Mineur | `src/audio/pcm.h:25` | Correction de dérive bornée à 0,125 %, au ras du calage de vitesse | 16 | |
| A16-06 | Mineur | `src/audio/packet.h:41-44` | `AudioTrack` sans époque : paquets de l'ancienne piste après un changement | 16 | |
| A16-07 | Mineur | `src/app/play/pipeline.cpp:97-110` | Sortie ouverte au format de la première piste | 16 | |
| A16-08 | Mineur | `src/audio/converter.cpp:132-135` | Mixage surround vers stéréo normalisé : son plus faible | 16 | |
| A16-09 | Mineur | `src/audio/player.cpp:349-364` | Avec la PLL, `set_delay` décale l'horloge avant le son | 16 | |
| A16-10 | Mineur | `src/audio/output.cpp:29-32` | Aucun contrôle de continuité des `pts` audio | 16 | |
| A16-11 | Mineur | `src/audio/devices.cpp:15-18` | Code mort, appareils désignés par leur seul nom | 16 | |
| A16-12 | Mineur | `src/audio/player.h:5-13` | Commentaires faux : l'audio est maître sur tout écran, pas seulement sous VRR | 16 | |
| A16-13 | Mineur | `src/audio/player.h:55` | PLL éteinte par défaut dans les tests, active dans le produit | 16 | |

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
| 02 | 07 | `src/app/play/span_int8.cpp:166` | `size_key` plafonné à 1920×1088 : la taille validée n'est pas celle du moteur au-delà | Traité par 07 |
| 02 | 07 | `src/app/play/span_int8.cpp:129-150` | Seul un refus d'une porte plus ancienne se refait | Traité par 07 |
| 02 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:75` | La zone d'activation partagée suppose que tout moteur qui y écrit se nomme (voir A02-03) | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:80-91` | `SharedActivations::reserve` synchronise la file de la voie à chaque croissance | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_textures.cpp:15-78` | Objets texture du warp gardés par adresse : réutilisés après une réallocation à la même adresse | En attente (01 fini) |
| 03 | 09 | `src/app/play/upscale_motion.cpp:39` | Reprise à chaud absente sous RIFE fusionné ; invalidation sur coupe et saut à vérifier | Traité par 09 |
| 03 | 09 | `src/app/play/player_engines.cpp:247-266` | Avec DistilDRBA fusionné, le warp repose sur la branche mouvement non testée (A03-03) | Traité par 09 |
| 03 | 15 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:142-143` | Position de chroma : seuls `Center` et `TopLeft` sont distingués | Réfuté par 15 |
| 03 | 40 | `tests/unit/backend_tensorrt/` | La suite CUDA ne tourne qu'à la main sur un poste NVIDIA, aucun garde-fou de CI | Ouvert |
| 05 | 27 | `src/gpu/cuda/backend/CMakeLists.txt:95` | Repli `trtexec` gardé en `RelWithDebInfo` (SEC-20) | Ouvert |
| 05 | 01 | `src/gpu/cuda/backend/entry.cpp:56-59` | `pick_trtexec` appelé une fois : empreinte jamais revérifiée ensuite (A05-05) | En attente (01 fini) |
| 05 | 06 | `src/app/play/engine_install.cpp:130-158` | TRT-17 : échec de `nvidia-smi` gardé pour le processus, carte inconnue rangée en « ptx » | Traité par 06 (TRT-17 en partie présent) |
| 05 | 18 | `src/app/play/crash_dump.cpp:90` | Le gestionnaire de plantage ne tue pas le `trtexec` en cours (A05-01) | Ouvert |
| 05 | 07 | appelants de `Backend::build` | Une construction par taille de source : coût à mesurer, cache sans purge (TRT-17) | Traité par 07 |
| 05 | 36 | `src/prepare/identity.cpp:161-164` | Jeton `driver` dans la clé : reconstruction complète à chaque saut de CUDA du pilote | Ouvert |
| 06 | 26 | `src/app/play/engine_online.cpp` | `merge_release_index` : SHA ou version changés du paquet TensorRT (voir A06-01) | Ouvert |
| 06 | 28 | `tools/release/package-linux.sh:190` | Version du paquet TensorRT tirée de `ls libnvinfer.so.*` : « 11 » ou « 11.3.0 » | Ouvert |
| 06 | 28 | `tools/release/package-windows.ps1:101` | Origine et stabilité de `$trtVersion` | Ouvert |
| 06 | 25 | `src/app/play/first_run.cpp:125-131` | « Go » et « Mo » en dur dans l'interface anglaise (UX-09) | Ouvert |
| 06 | 34 | `src/app/play/engine_tools.cpp:19-36` | `curl.exe` de System32 : version minimale, révocation Schannel derrière une inspection TLS | Ouvert |
| 06 | 21 | `src/app/play/backend_library.cpp:83-85` | Module refusé refermé : nvinfer se décharge-t-il sous Windows (A06-09) ? | Ouvert |
| 04 | 09 | `src/app/play/derived_model.cpp:19` | `stem().string()` dans le nom des modèles dérivés (comme A04-04) | Traité par 09 |
| 04 | 09 | `src/app/play/motion_model.cpp:90` | `filename().string()` et `stem().string()` sur le chemin du RIFE | Traité par 09 |
| 04 | 09 | `src/app/play/upscale_stage.cpp:198` | L'étage suppose `kSpanFactor` pour un `--model` quelconque (voir A04-06) | Traité par 09 |
| 04 | 28 | `tests/fuzz/fuzz_parsers.cpp:33-37` | Aucune cible de fuzz pour les lecteurs ONNX de `prepare/` (SEC-21) | Ouvert |
| 04 | 05 | `src/prepare/builder.cpp:174` | `read_model` relit et rehache tout l'ONNX avant et après chaque construction | En attente (05 fini) |
| 08 | 22 | `src/app/play/settings.cpp:453-499` | `change_between` ne classe pas en réouverture plusieurs réglages dont `prepare_chain` dépend | Ouvert |
| 08 | 05 | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` joint le fil de construction sur le thread de la fenêtre | En attente (05 fini) |
| 08 | 11 | `src/app/play/host_facts.cpp:378` | `on_battery()` appelé deux fois par seconde | Traité par 11 (A11-09) |
| 08 | 14 | `src/app/play/player_run.cpp:589-591` | MED-01 : la session doit distinguer vraie fin et erreur d'E/S | Traité par 14 |
| 08 | 18 | `src/app/play/player.cpp:525-534` | La reprise après « Annuler et regarder » modifie `settings` pour toute la suite | Ouvert |
| 08 | 23 | `src/app/play/settings_panel.cpp` | UX-08 : Échap devrait fermer la liste ouverte avant le panneau | Ouvert |
| 08 | 39 | `src/app/play/settings_system.cpp` | UX-13 : chemins hors du dossier personnel dans le journal envoyé | Ouvert |
| 07 | 25 | `src/app/play/engine_screen.cpp:111` | `calibrate_on_screen` rend toujours Auto, même sur « Annuler » : remplace une chaîne Personnalisée | Ouvert |
| 07 | 08 | `src/app/play/chain_setup.cpp:99` | Tailles de travail différentes possibles entre l'anneau et `c.work` | En attente (08 fini) |
| 07 | 09 | `src/app/play/chain_setup.cpp:321` | Préparation FP8 automatique faite une seule fois, à la taille de la lecture en cours (avec A07-04) | En attente (09 fini) |
| 07 | 18 | `src/app/play/player.cpp:411-414` | `calibration.toml` relu à chaque ouverture et remplissage du panneau | Ouvert |
| 07 | 34 | `src/app/play/auto_bench_run.cpp:35-44` | Essais en `--hidden` sur tous les bureaux, GNOME Wayland compris | Ouvert |
| 09 | 23 | `src/app/play/home_screen.cpp:140` | `fp8_engine` vrai pour une RTX 20 ou 30 : Performance proposé sans FP8 | Ouvert |
| 09 | 22 | `src/app/play/settings.h:114-117` | Critère d'Auto sur la hauteur de sortie (A09-08) | Ouvert |
| 09 | 08 | `src/app/play/chain_setup.cpp:420-423` | Performance remplace le RIFE fusionné par un moteur entier (A09-02) | En attente (08 fini) |
| 09 | 41 | `src/app/play/upscale_stage.cpp:244-271` | Statistiques du grade prises sur l'image débandée, ou pas du tout sous DistilDRBA | Ouvert |
| 09 | 31 | `src/app/play/ring_stage.cpp:108-112` | Seul `SourceResized` rouvre la chaîne sur une image de taille nouvelle | Ouvert |
| 10 | 12 | `src/app/play/session.cpp:598-616` | Au niveau `Wait`, `vblank_grid()` toujours vide : effet sur `--vsync-index` et la PLL | Traité par 12 (A12-01, A12-03) |
| 10 | 08 | `src/app/play/player_run.cpp:654` | Une erreur de `show` termine `run_chain` : message et réouverture en boucle ? | En attente (08 fini) |
| 10 | 18 | `src/app/play/placebo_open.cpp:327` | Décodeur Vulkan Video à détruire avant l'écran | Ouvert |
| 10 | 31 | `src/app/play/placebo_ring.cpp:218` | `ready()` vrai quand `vkGetSemaphoreCounterValue` échoue | Ouvert |
| 10 | 34 | `src/app/play/placebo_open.cpp:384-390` | Windows : fenêtre cachée pendant l'ouverture, message d'erreur sans fenêtre ? | Ouvert |
| 11 | 08 | `src/app/play/player_run.cpp:398-400` | Relevé de l'écran fait deux fois sous Windows | En attente (08 fini) |
| 11 | 08 | `src/app/play/player_run.cpp:549-560` | Relecture du HDR à partir de l'événement SDL plutôt que du minuteur | En attente (08 fini) |
| 11 | 08 | `src/app/play/player_run.cpp:619-621` | `p.display` remplacé sans prévenir `contract_watch` | En attente (08 fini) |
| 11 | 24 | `src/app/play/chain_stats.cpp:184-208` | NVML relevé toutes les 500 ms même OSD caché | Ouvert |
| 11 | 30 | `src/app/play/player_run.cpp:679` | `GpuTelemetry::sample` appelé depuis deux threads (A11-08) | Ouvert |
| 11 | 34 | `src/app/play/host_facts.cpp:236-275` | Pic calculé dans deux unités différentes sous Windows et Linux | Ouvert |
| 12 | 08 | `src/app/play/player_run.cpp:622-629` | GPU-05 : plancher VRR non recalculé (VRR hors sujet pour Tatoub) | En attente (08 fini) |
| 12 | 08 | `src/app/play/player_run.cpp:420-438` | Plafond passé comme fréquence d'écran (A12-04) | En attente (08 fini) |
| 12 | 08 | `src/app/play/session.cpp:665-672` | GPU-19 : vitesse calée même en VRR armé (VRR hors sujet pour Tatoub) | En attente (08 fini) |
| 12 | 10 | `src/app/play/placebo_screen.cpp:461-463` | `present_us` est le retour de `swap_buffers`, pas un vsync (A12-03) | En attente (10 fini) |
| 12 | 34 | `src/app/play/loop_wait.cpp:56` | Windows : attente « précise » à 1 ms de résolution | Ouvert |
| 12 | 20 | `src/core/drawings.cpp:113` | `DrawingsTimeline` n'est appelé que par ses tests | Ouvert |
| 13 | 14 | `src/media/source_loop.cpp:110-123` | `recover_position` qui échoue laisse la source inactive, lecture figée | Traité par 14 |
| 13 | 14 | `src/media/source_loop.cpp:182-188` | Lecture abandonnée traitée comme la fin du fichier (origine de MED-01) | Traité par 14 |
| 13 | 17 | `src/app/play/external_subtitles.cpp:92` | `path::string()` sur des noms de sous-titres | Ouvert |
| 13 | 08 | `src/app/play/player_run.cpp:738-739` | Pistes copiées dans `Player` même pour une autre vidéo (A13-01, A13-09) | En attente (08 fini) |
| 13 | 30 | `src/app/play/wiring.cpp:113-124` | `ThreadStages::play/pause` rendent faux sans que personne ne rejoue | Ouvert |
| 14 | 13 | `src/app/play/pipeline.cpp:45-49` | `path::string()` sur les sous-titres externes | En attente (13 fini) |
| 14 | 13 | `src/app/play/pipeline.cpp:65` | `config.file.string()` dans le refus « no video track » | En attente (13 fini) |
| 14 | 13 | `src/app/play/session.cpp:286-287` | La session doit refuser d'enchaîner l'épisode sur une fin en erreur (MED-01) | En attente (13 fini) |
| 14 | 18 | `src/app/play/player.cpp:99` | `file.string()` dans le refus de `probe_once` | Ouvert |
| 14 | 18 | `src/app/play/pipeline.h:76` | `~Pipeline` joint un thread source bloqué sur un partage muet | Ouvert |
| 14 | 16 | `src/audio/` | Rognage audio sur la cible d'un seek refusé (MED-06) | En attente (16 fini) |
| 14 | 30 | `src/media/source_loop.cpp:303-305` | Boîte non lue pendant `wait_before_retry` | Ouvert |
| 15 | 14 | `src/media/source_loop.cpp:153-159` | Image décodée et convertie avant la décision de saut du seek exact | En attente (14 fini) |
| 15 | 14 | `src/media/source_loop.cpp:21-25` | `ends_before` jette les images `kNoPts` : seek exact sans fin (avec A15-05) | En attente (14 fini) |
| 15 | 08 | `src/app/play/player.cpp:95-117` | La sonde devrait refuser les formats illisibles avec un message | En attente (08 fini) |
| 15 | 01 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:24-41` | L'ingestion TensorRT ne lit ni 4:4:4 ni 4:2:2 en mémoire CUDA | En attente (01 fini) |
| 15 | 42 | `src/media/shot_probe.cpp:173-175` | La pré-analyse paie la conversion 4:4:4 en pleine taille | Ouvert |
| 16 | 23 | `src/app/play/settings_sound.cpp:27-33` | En mode « suivre le système », un seul délai pour tous les appareils | Ouvert |
| 16 | 08 | `src/app/play/pipeline.cpp:97-110` | Format de sortie figé, ordre `seek()` puis `AudioTrack` | En attente (08 fini) |
| 16 | 13 | `src/app/play/session.cpp:420-445` | Sans sortie audio, un gel de 0,4 s à chaque seek | En attente (13 fini) |
| 16 | 12 | `src/core/clock_pll.h:28-40` | PLL comptée en lectures, pas en temps | En attente (12 fini) |
| 16 | 20 | `src/core/clock.h:1-3` | Commentaire faux sur l'audio maître | Ouvert |
