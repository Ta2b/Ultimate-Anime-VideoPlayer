# Synthèse des audits

Écrit par le coordinateur seul, après chaque fusion d'un rapport. Les fils le lisent et ne le modifient pas.

## Bilan

À écrire à la fin de toutes les vagues.

## Avancement

| Audit | Statut | Rapport |
|---|---|---|
| 01. TensorRT : exécution, voies, anneau et mémoire | Fusionné (7 défauts, 1 majeur) | [rapport](rapports/01-tensorrt-execution.md) |
| 02. SPAN FP8 : noyaux CUTLASS et dérivation | Fusionné (14 défauts, 1 majeur, complément compris) | [rapport](rapports/02-span-fp8.md), [complément](rapports/02-span-fp8-complement.md) |
| 03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp) | Fusionné (19 défauts, 6 majeurs, complément DistilDRBA compris) | [rapport](rapports/03-interpolation-tensorrt.md), [complément](rapports/03-interpolation-tensorrt-complement.md) |
| 04. Analyse et découpe des modèles ONNX | Fusionné (11 défauts, 1 majeur) | [rapport](rapports/04-onnx-modeles.md) |
| 05. Construction des moteurs, cache et trtexec | Fusionné (8 défauts, 1 majeur) | [rapport](rapports/05-construction-moteurs.md) |
| 06. Installation des moteurs et téléchargements | Fusionné (12 défauts, 2 majeurs) | [rapport](rapports/06-installation-moteurs.md) |
| 07. Calibration, niveaux et mode Auto | Fusionné (13 défauts, 3 majeurs, complément compris) | [rapport](rapports/07-calibration-niveaux.md), [complément](rapports/07-calibration-niveaux-complement.md) |
| 08. Boucle du lecteur et ouverture de la chaîne | Fusionné (17 défauts, 5 majeurs, complément compris) | [rapport](rapports/08-boucle-chaine.md), [complément](rapports/08-boucle-chaine-complement.md) |
| 09. Étages d'agrandissement et d'interpolation | Fusionné (24 défauts, 1 majeur, complément compris) | [rapport](rapports/09-etages-traitement.md), [complément](rapports/09-etages-traitement-complement.md) |
| 10. Rendu et présentation par libplacebo | Fusionné (13 défauts, 2 majeurs) | [rapport](rapports/10-presentation-libplacebo.md) |
| 11. Écran et matériel : EDID, VRR, HDR du système, télémétrie | Fusionné (13 défauts, 2 majeurs) | [rapport](rapports/11-ecran-materiel.md) |
| 12. Cadence et horloges | Fusionné (10 défauts, 3 majeurs) | [rapport](rapports/12-cadence-horloges.md) |
| 13. Session de lecture : seek, épisodes et reprise | Fusionné (13 défauts, 1 bloquant, 1 majeur) | [rapport](rapports/13-session-lecture.md) |
| 14. Sources, démultiplexage et erreurs d'E/S | Fusionné (8 défauts, 2 majeurs) | [rapport](rapports/14-sources-demux.md) |
| 15. Décodage, matériel, formats et couleurs | Fusionné (14 défauts, 2 majeurs) | [rapport](rapports/15-decodage-formats.md) |
| 16. Audio | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/16-audio.md) |
| 17. Sous-titres | Fusionné (13 défauts, 2 majeurs) | [rapport](rapports/17-sous-titres.md) |
| 18. Démarrage et arrêt | Fusionné (15 défauts, 2 majeurs) | [rapport](rapports/18-demarrage-arret.md) |
| 19. Intégration au système | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/19-integration-systeme.md) |
| 20. Socle commun (core) | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/20-socle-core.md) |
| 21. Frontière des moteurs (interface et ABI) | Fusionné (10 défauts, 3 majeurs) | [rapport](rapports/21-frontiere-moteurs.md) |
| 22. Réglages : modèle, application et persistance | Fusionné (16 défauts, 3 majeurs) | [rapport](rapports/22-reglages-modele.md) |
| 23. Réglages : le panneau | Fusionné (15 défauts, 4 majeurs) | [rapport](rapports/23-reglages-panneau.md) |
| 24. Interface : rendu, contrôles et messages | Fusionné (23 défauts, 3 majeurs) | [rapport](rapports/24-interface-rendu.md) |
| 25. Assistant de premier lancement | Fusionné (18 défauts, 4 majeurs) | [rapport](rapports/25-assistant.md) |
| 26. Mise à jour automatique et signature | Fusionné (12 défauts, 4 majeurs) | [rapport](rapports/26-mise-a-jour.md) |
| 27. Build, dépendances et ports | Fusionné (13 défauts, 1 majeur) | [rapport](rapports/27-build-dependances.md) |
| 28. CI, publication et signature | Fusionné (22 défauts, 1 bloquant, 8 majeurs, complément compris) | [rapport](rapports/28-ci-publication.md), [complément](rapports/28-ci-publication-complement.md) |
| 29. Parcours utilisateur de bout en bout | Fusionné (7 défauts, 2 majeurs) | [rapport](rapports/29-parcours-utilisateur.md) |
| 30. Concurrence, threads et durée de vie | Fusionné (8 défauts, 1 majeur) | [rapport](rapports/30-concurrence-duree-de-vie.md) |
| 31. États dégradés et reprise sur erreur | Fusionné (15 défauts, 8 majeurs) | [rapport](rapports/31-etats-degrades.md) |
| 32. Budget par image sur la cible | Fusionné (9 défauts, 4 majeurs) | [rapport](rapports/32-budget-par-image.md) |
| 33. Longue durée, mémoire et ressources | Fusionné (8 défauts, 4 majeurs) | [rapport](rapports/33-longue-duree-ressources.md) |
| 34. Parité Windows et CachyOS | Fusionné (12 défauts, 2 majeurs) | [rapport](rapports/34-parite-windows-cachyos.md) |
| 35. Entrées non sûres | Fusionné (9 défauts, 1 majeur) | [rapport](rapports/35-entrees-hostiles.md) |
| 36. État persistant et compatibilité entre versions | Fusionné (9 défauts, 3 majeurs) | [rapport](rapports/36-etat-persistant.md) |
| 37. Configurations NVIDIA variées | Fusionné (14 défauts, 1 majeur) | [rapport](rapports/37-configurations-nvidia.md) |
| 38. Promesses tenues | Fusionné (15 défauts, 2 majeurs) | [rapport](rapports/38-promesses-tenues.md) |
| 39. Diagnostic et support | Fusionné (9 défauts, 4 majeurs) | [rapport](rapports/39-diagnostic-support.md) |
| 40. Tests et couverture | Fusionné (16 défauts, 5 majeurs) | [rapport](rapports/40-tests-couverture.md) |
| 41. HDR, grade et looks | Fusionné (11 défauts, 2 majeurs) | [rapport](rapports/41-hdr-grade.md) |
| 42. Analyse des plans, coupes et pré-analyse | Fusionné (16 défauts, 1 majeur) | [rapport](rapports/42-plans-coupes.md) |
| 43. Textes et traductions | Fusionné (7 défauts, 1 majeur) | [rapport](rapports/43-textes-traductions.md) |
| 44. Représentation intermédiaire et import ONNX | Fusionné (12 défauts, 1 majeur) | [rapport](rapports/44-ir-onnx.md) |
| 45. Moteur Vulkan : calcul, noyaux et shaders | Fusionné (14 défauts, 1 majeur) | [rapport](rapports/45-backend-vulkan.md) |
| 46. Outils de recherche, modèles et données livrées | Fusionné (15 défauts, 6 majeurs) | [rapport](rapports/46-outils-donnees.md) |
| 47. Export vidéo | Ne pas lancer (hors cible) | |
| 48. Moteur Core ML (macOS) | Ne pas lancer (hors cible) | |
| 49. Qualité du rendu (transversal, ajouté à la demande de Tatoub) | Fusionné (14 défauts, 6 majeurs, 16 compromis à trancher) | [rapport](rapports/49-qualite-rendu.md) |

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
| A02-12 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:96` | Quatre `path::string()` contre la règle de la CI | 02 | Dette (34) |
| A02-13 | Mineur | `tests/unit/gpu_span_fp8/metrics_test.cpp` | Aucun test par défaut de la concordance numérique FP8 ni des entrées non finies | 02 | |
| A03-01 | Majeur | `src/gpu/cuda/backend/slice_kernels.cu:24-45` | L'ingestion TensorRT rééchantillonne la chroma en bilinéaire au lieu du Spline36 du contrat (et de Vulkan) | 03 | |
| A03-02 | Majeur | `tests/unit/backend_tensorrt/rife_fused_test.cpp:397` | Concordance des chaînes fusionnées vérifiée seulement par des tests cachés, à 30 dB pour RIFE | 03 | |
| A03-03 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:284-290` | Variante mouvement de DistilDRBA fusionné sans aucun test ni essai au chargement | 03 | |
| A03-04 | Mineur | `src/gpu/cuda/backend/tensorrt_rife.cpp:118-124` | Plan d'instant compact : un moteur RIFE entier lancé ensuite lit un instant périmé (latent) | 03 | |
| A03-05 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:175-194` | Essai au chargement sur un flux privé dans la zone d'activation partagée | 03 | A02-03 (même cause) |
| A03-06 | Mineur | `src/gpu/cuda/backend/tensorrt_rife_fused.cpp:388` | `path::string()` dans les messages de refus : exception sous Windows avec un profil non ASCII | 03 | Dette (34) |
| A03-07 | Mineur | `src/gpu/cuda/backend/rife_kernels.cu:49-50` | `rife_pair` écrit le plan d'instant entier alors que RIFE fusionné n'en lit qu'un élément | 03 | |
| A03-08 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:221-224` | Chaque paire recopie et réencode les deux images, alors que A(n+1) = B(n) | 03 | |
| A03-09 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:323-324` | Cache de graphes plafonné sans éviction ni notice (latent) | 03 | |
| A03-10 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:361` | Environ 150 lignes dupliquées entre les deux moteurs fusionnés, déjà divergentes | 03 | |
| A03-11 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:241` | `__expf` approché pour RIFE, `expf` pour DistilDRBA | 03 | |
| A03-12 | Mineur | `src/gpu/cuda/backend/slice_kernels.cu:268-273` | `ingest_tiled` compte le poids d'une prise hors tuile avant de la sauter (latent) | 03 | |
| A03-13 | Majeur | `src/app/play/upscale_drba.cpp:247-257` | DistilDRBA fait une passe par image créée, RIFE une par paire : au-delà de ×5, plus de réseau que RIFE 4.25 | 03 (complément) | |
| A03-14 | Majeur | `src/app/play/upscale_drba.cpp:193-208` | Pas d'entrée par pointeurs : quatre `rife_input` de 7 plans par paire, presque rien de lu, plus cher que la passe | 03 (complément) | |
| A03-15 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:225-242` | Encodeur refait à chaque côté : 6 encodages par paire pour une image nouvelle | 03 (complément) | |
| A03-16 | Mineur | `src/gpu/cuda/backend/drba_fused_kernels.cu:277-290` | Flot agrandi puis rééchantillonné aussitôt par le warp | 03 (complément) | |
| A03-17 | Mineur | `tests/unit/backend_tensorrt/drba_fused_test.cpp` | Aucun banc par paire ni par image ; RIFE 4.22 lite fusionné sans chiffre | 03 (complément) | |
| A03-18 | Majeur | `tools/rife/convert_rife_fp16.py:35` | Grille des dix GridSample de DistilDRBA en FP16 dans le modèle livré (pas de 0,31 px à 1280 de large), reproduite par la colle fusionnée | 03 (complément) | |
| A03-19 | Mineur | `src/app/play/chain_setup.cpp:359-363` | La vitesse de DistilDRBA n'est pas réinvestie : estimation à 720p alors qu'une passe pleine résolution reste moins chère que RIFE | 03 (complément) | |
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
| A04-04 | Mineur | `src/prepare/rife_split.cpp:381` | Noms des morceaux par `stem().string()` : exception sous Windows hors page de code | 04 | Dette (34) |
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
| A08-09 | Mineur | `src/app/play/chain_setup.cpp:277` | `path::string()` sur des chemins du profil, contre la règle de la CI | 08 | Dette (34) |
| A08-10 | Mineur | `src/app/play/engine_build.cpp:38` | Une instance de backend par construction ou lecture du cache, jusqu'à six de plus par ouverture | 08 | |
| A08-11 | Mineur | `src/app/play/player_run.cpp:98` | `run_chain` d'environ 650 lignes, fichier de 753 lignes (QUA-15) | 08 | |
| A08-12 | Mineur | `tests/unit/app/chain_setup_test.cpp` | Aucun test de `apply`, `switch_chain`, `with_kept_choices`, des replis ni de la fin de `run_chain` | 08 | |
| A08-13 | Mineur | `src/app/play/player_run.cpp:334-340` | E/S synchrones sur le thread de la boucle | 08 | |
| A08-14 | Mineur | `src/app/play/engine_switch.cpp:13-19` | L'onglet Moteurs connaît encore MIGraphX et OpenVINO | 08 | A06-11 (même cause) |
| A07-01 | Majeur | `src/app/play/machine_calibration.cpp:139-151` | La calibration mesure une chaîne différente de celle que l'Auto calibré applique | 07 | |
| A07-02 | Majeur | `src/app/play/machine_calibration.cpp:338` | « Tient » ignore les images non interpolées quand le Rapide coupe RIFE | 07 | |
| A07-03 | Mineur | `src/media/bench_clip.cpp:143` | Chemin du clip de calibration en page ANSI sous Windows : la calibration casse | 07 | Requalifié en mineur (dette) par 34 |
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
| A09-10 | Mineur | `src/app/play/derived_model.cpp:19` | Sept `path::string()` sur des chemins du profil | 09 | Dette (34) |
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
| A10-10 | Mineur | `src/app/play/placebo_screen.cpp:632` | `path::string()` dans les instantanés | 10 | Dette (34) |
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
| A13-09 | Mineur | `src/app/play/player_run.cpp:738-739` | Pistes audio et sous-titres gardées pour toute vidéo suivante | 13 | Réfuté par 29 (`Player` recréé à chaque fichier) |
| A13-10 | Mineur | `src/app/play/resume.cpp:107-129` | `resume.tsv` réécrit sans verrou entre deux instances | 13 | |
| A13-11 | Mineur | `src/app/play/resume.cpp:33` | `path::string()` dans la reprise et les sous-titres externes | 13 | Dette (34) |
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
| A17-01 | Majeur | `src/subs/subtitle_renderer.cpp:199` | En 4K, un dessin plein écran dépasse l'atlas et laisse les sous-titres suivants à demi-définition (MED-05, démontré) | 17 | |
| A17-02 | Majeur | `src/subs/subs_messages.h:42` | PGS et VobSub ignorés sans un mot, et ils empêchent le repli sur un .ass externe | 17 | |
| A17-03 | Mineur | `src/subs/text_encoding.cpp:212-216` | Indice de langue qui impose son encodage, repli direct sur Windows-1252 (MED-11) | 17 | |
| A17-04 | Mineur | `src/subs/text_encoding.cpp:95-102` | UTF-16 sans BOM non reconnu, demi-paires recopiées | 17 | |
| A17-05 | Mineur | `src/subs/subtitle_track.cpp:137-161` | Pas de pré-lecture des sous-titres au seek | 17 | |
| A17-06 | Mineur | `src/subs/subtitle_renderer.cpp:221` | Délai négatif sans paquets lus en avance (MED-09) | 17 | |
| A17-07 | Mineur | `src/subs/subtitle_track.cpp:118-122` | Une seule cue SRT refusée fait refuser tout le fichier | 17 | |
| A17-08 | Mineur | `src/subs/subtitle_renderer.cpp:229-234` | Un rendu libass par image de sortie, après l'interpolation | 17 | |
| A17-09 | Mineur | `src/subs/ass_library.cpp:53-62` | Polices jointes sans plafond total | 17 | |
| A17-10 | Mineur | `src/subs/subs_thread.cpp:50` | Initialisation de libass sur le thread de la fenêtre | 17 | |
| A17-11 | Mineur | `src/app/play/external_subtitles.cpp:123-124` | Fichiers externes coupés à 16 avant le choix par langue | 17 | |
| A17-12 | Mineur | `src/subs/subtitle_track.cpp:27` | `path::string()` dans les sous-titres | 17 | Dette (34) |
| A17-13 | Mineur | `tests/unit/subs/` | Sans test : retour à pleine taille, encodages coréen et EUC-JP | 17 | |
| A19-01 | Majeur | `src/app/play/data_dirs.cpp:77-93` | Migration coupée : `engines/` tronqué pris pour complet (démontré) | 19 | |
| A19-02 | Mineur | `src/app/play/single_instance.cpp:52` | Accusé « ok » envoyé avant que le fichier soit remis à SDL | 19 | |
| A19-03 | Mineur | `src/app/play/settings.h:246` | La langue du système ne règle pas la langue de l'interface | 19 | |
| A19-04 | Mineur | `src/app/play/default_player.cpp:381-391` | `xdg-mime query` (jusqu'à 10 s) sur le thread principal | 19 | |
| A19-05 | Mineur | `src/app/play/default_player.cpp:379` | Lecteur par défaut : état inconnu sous Windows, seul le MKV interrogé sous Linux | 19 | |
| A19-06 | Mineur | `src/app/play/default_player.cpp:221-244` | Inscription jamais refaite si l'exécutable bouge | 19 | |
| A19-07 | Mineur | `src/app/play/single_instance.cpp:148-150` | Windows : erreur de `ConnectNamedPipe` en boucle sans `DisconnectNamedPipe` | 19 | |
| A19-08 | Mineur | `src/app/play/home_screen.cpp:204-212` | Sous Wayland, la fenêtre ne passe pas au premier plan sur un fichier transmis | 19 | |
| A19-09 | Mineur | `src/app/play/data_dirs.cpp:124` | `directory_iterator` qui peut lever malgré le code d'erreur | 19 | |
| A19-10 | Mineur | `src/app/play/single_instance.cpp:385-386` | Socket trop long : instance unique éteinte sans journal | 19 | |
| A19-11 | Mineur | `src/app/play/default_player.cpp:421` | `SetConsoleOutputCP` change la console du terminal parent | 19 | |
| A19-12 | Mineur | `src/app/play/system_language.cpp:81-82` | Fichiers de KDE lus hors d'une session KDE, avant SDL | 19 | |
| A19-13 | Mineur | `tests/unit/app/single_instance_test.cpp:20` | Sans test : course de deux lancements, migration coupée | 19 | |
| A18-01 | Majeur | `src/core/log.cpp:163-165` | `close_log` ne vide pas `ajp.log`, et tout journal après lui plante (démontré) | 18 | |
| A18-02 | Majeur | `src/app/play/player.cpp:130-139` | `engines-nvidia.lock` téléchargé par curl sur le thread de la fenêtre à chaque lancement, jusqu'à 8 s figée | 18 | |
| A18-03 | Mineur | `src/app/play/player.cpp:183-213` | Le thread des mises à jour part avant l'assistant avec l'ancien mode | 18 | |
| A18-04 | Mineur | `src/app/play/player.cpp:525-534` | « Annuler et regarder » finit écrit dans `play.toml` | 18 | |
| A18-05 | Mineur | `src/app/main.cpp:390-393` | Purge récursive des `*.old` dans tout le dossier d'installation | 18 | |
| A18-06 | Mineur | `src/app/main.cpp:452-454` | `--bake-looks` cherche les LUT dans l'ancien dossier | 18 | |
| A18-07 | Mineur | `src/app/main.cpp:337-343` | `SDL_GetPrefPath` en échec non arrêté | 18 | |
| A18-08 | Mineur | `src/app/play/options.cpp:466-508` | Aide fausse ou incomplète sur les options et touches | 18 | |
| A18-09 | Mineur | `src/app/play/options.cpp:236-243` | Aucune option pour choisir DistilDRBA en ligne de commande | 18 | |
| A18-10 | Mineur | `src/app/main.cpp:459-461` | Deux sorties sans trace au journal | 18 | |
| A18-11 | Mineur | `src/app/main.cpp:247-259` | `ajp calibrate --display-height` remplacé en silence | 18 | |
| A18-12 | Mineur | `src/app/main.cpp:147-150` | `execv` garde les descripteurs ouverts | 18 | |
| A18-13 | Mineur | `src/app/play/player.cpp:445-452` | Échec d'`open_backend` depuis l'accueil : le lecteur quitte | 18 | |
| A18-14 | Mineur | `src/app/play/mapped_file.cpp:229-247` | Police tronquée entre la taille lue et la projection | 18 | |
| A18-15 | Mineur | `src/app/play/bridges.cpp:374-385` | `mov_text` et `webvtt` refusés | 18 | A17-02 (même cause) |
| A20-01 | Majeur | `src/core/file_sync.cpp:81-91` | `write_atomic` sans nouvel essai sous Windows, refus jeté à `main.cpp:526` : la garde peut revenir à l'ancienne version | 20 | |
| A20-02 | Mineur | `src/core/utf8_path.h:3-7` | Prémisse fausse : le manifeste UTF-8 empêche l'exception de `path::string()` dans le lecteur ; le risque est dans `trtexec`, `curl`, `tar` | 20 | |
| A20-03 | Mineur | `src/core/log_redact.cpp:25-29` | Nettoyage du journal par sous-chaîne : casse, noms 8.3, `/run/media/<user>` (démontré) | 20 | |
| A20-04 | Mineur | `src/core/log.cpp:104-112` | Journal asynchrone sans `flush_on` : la dernière seconde est perdue au plantage | 20 | |
| A20-05 | Mineur | `src/core/thread.h:103-127` | `start_task` par `std::async` : pool Windows sans nom, futur bloquant non documenté | 20 | |
| A20-06 | Mineur | `src/core/thread_role.cpp:42-56` | Rôles de thread sans héritage ni priorité d'E/S sous Windows | 20 | |
| A20-07 | Mineur | `src/core/dynamic_library.cpp:28-36` | Aucun `SetDefaultDllDirectories` : DLL secondaires prises dans le `PATH` | 20 | |
| A20-08 | Mineur | `src/core/file_sync.cpp:63-79` | Lien symbolique remplacé, temporaires jamais purgés (démontré) | 20 | |
| A20-09 | Mineur | `src/core/types.h:19-20` | Refus fatal codé par `ThreadFailed` ; 58 codes sur 71 sans phrase | 20 | |
| A20-10 | Mineur | `src/core/debug_flags.cpp` | Environ 1 250 lignes de `core` sans appelant | 20 | |
| A20-11 | Mineur | `src/core/debug_flags.h:27-33` | Aucun outil livré pour mesurer DistilDRBA contre RIFE (NVTX, noms de threads, bascules mortes) | 20 | |
| A20-12 | Mineur | `src/core/av_error.h:12-29` | FFmpeg inclus dans `core`, textes français dans le détail des refus | 20 | |
| A20-13 | Mineur | `tests/unit/core/` | Aucun test des chemins d'échec du socle | 20 | |
| A22-01 | Majeur | `src/app/play/settings.cpp:383-434` | UX-01 plus large : `restore_lines`, `gpu`, `rife_fused`, `drba_fused` ni appliqués ni écrits | 22 | |
| A22-02 | Majeur | `src/app/play/settings_toml.cpp:624-626` | Écriture refusée de `play.toml` : le changement est perdu pour de bon (base de fusion retenue avant l'écriture) | 22 | |
| A22-03 | Majeur | `src/app/play/settings.h:106-117` | Qualité sacrifiée par défaut : l'Auto de l'ordre warp et SPAN vaut toujours « le plus rapide » | 22 | A09-08 (même cause, gravité relevée) |
| A22-04 | Mineur | `src/app/play/settings.cpp:453-499` | `change_between` classe en direct des réglages lus à l'ouverture | 22 | A08-03 (même cause) |
| A22-05 | Mineur | `src/app/play/settings.cpp:336` | `--rife` coupe `rife_auto`, qui n'est pas rendu à l'écriture | 22 | |
| A22-06 | Mineur | `src/app/play/settings_toml.cpp:501-534` | Valeur inconnue d'une clé connue écrasée à l'écriture suivante | 22 | |
| A22-07 | Mineur | `src/core/settings.cpp:231-233` | `settings.toml` refusé entier pour une version différente | 22 | |
| A22-08 | Mineur | `src/core/settings.h:55-58` | Langues et délai audio de `settings.toml` sans lecteur | 22 | |
| A22-09 | Mineur | `src/app/play/settings.cpp:266-324` | `engines_dir` non validé, écrit à chaque frappe | 22 | |
| A22-10 | Mineur | `src/app/play/settings.h:186-191` | Commentaire faux : en Rapide, `rife_factor` ne plafonne rien | 22 | |
| A22-11 | Mineur | `src/app/play/settings.h:299-310` | Trois bornes différentes par délai (UX-18) | 22 | |
| A22-12 | Mineur | `src/app/play/settings.cpp:481-490` | Un cran de précision hors Équilibre coupe le moteur fusionné en silence | 22 | |
| A22-13 | Mineur | `src/app/play/settings_toml.cpp:570` | Cinq `path::string()` (dette, voir A20-02) | 22 | |
| A22-14 | Mineur | `src/app/play/settings.h:180` | Qualité : réduction Hermite par défaut, plus douce que Spline36 annoncé le plus fin | 22 | |
| A22-15 | Mineur | `tests/unit/app/settings_test.cpp` | Aucun test que chaque champ est appliqué ou annoncé ; ni écriture refusée | 22 | |
| A22-16 | Mineur | `src/app/play/settings_toml.cpp:593-627` | Une écriture de `play.toml` fait quatre analyses TOML et ~90 relectures | 22 | |
| A21-01 | Majeur | `cmake/BoundaryHash.cmake:7-9` | L'empreinte d'ABI ne couvre ni `kernels/spec` ni `FrameStats` (démontré) | 21 | |
| A21-02 | Mineur | `src/app/play/backend_library.cpp:55` | `path.string()` dans le chargeur : sans repli Vulkan sous Windows avec un profil non ASCII | 21 | Requalifié en mineur (dette) par 34 |
| A21-03 | Majeur | `src/gpu/backend.h:132-133` | Qualité : anneau en RGB10A2 par défaut avant le grade HDR, sans la porte de qualité de l'ADR 0008 | 21 | |
| A21-04 | Mineur | `src/gpu/guarded_backend.h:56-168` | Rien ne vérifie que `GuardedKernelOps` relaie toutes les virtuelles | 21 | |
| A21-05 | Mineur | `src/app/play/backend_library.cpp:79-86` | `open` n'essaie pas les candidats suivants (latent) | 21 | |
| A21-06 | Mineur | `src/gpu/kernel_ops.h:188-196` | `scores_take` attend toute la file sur une seule voie | 21 | |
| A21-07 | Mineur | `src/gpu/backend.h:217-229` | `Engine::enqueue` sans scalaire : l'instant voyage en plans pleine taille | 21 | A03-14 (même cause) |
| A21-08 | Mineur | `src/gpu/graph_book.h:195-221` | Échecs des graphes CUDA cumulés sur toute la vie du moteur | 21 | |
| A21-09 | Mineur | `src/gpu/lanes.h:113-115` | `LaneHazards` ne perd jamais une adresse | 21 | |
| A21-10 | Mineur | `src/gpu/kernel_ops.h:135` | Virtuelles pures mortes et commentaires périmés dans le contrat | 21 | |
| A23-01 | Majeur | `src/app/play/settings_chain_tabs.cpp:84` | « Restaurer les traits » et la carte graphique jamais appliqués ni enregistrés (démontré) | 23 | A22-01 (même cause) |
| A23-02 | Majeur | `src/app/play/settings_chain_tabs.cpp:375-383` | « Warp and SPAN » grisé pour DistilDRBA alors qu'il décide sa variante | 23 | |
| A23-03 | Majeur | `src/app/play/settings_panel.cpp:250-255` | « Auto (calibré) » choisi au panneau applique une chaîne non calibrée | 23 | |
| A23-04 | Majeur | `src/app/play/settings_system.cpp:107-108` | Page Application : lecture d'`update.toml` et fichier d'essai créé et effacé à chaque image | 23 | |
| A23-05 | Mineur | `src/app/play/settings_panel.cpp:290-301` | « Tout réinitialiser » sans confirmation, langue remise en anglais | 23 | |
| A23-06 | Mineur | `src/app/play/settings_panel.h:53-54` | Aucun contrôle pour `motion_precision`, qui peut couper le fusionné en silence | 23 | |
| A23-07 | Mineur | `src/app/play/settings_chain_tabs.cpp:363-374` | Le panneau ne montre pas l'état réel de DistilDRBA (fusionné, variante, refus) | 23 | |
| A23-08 | Mineur | `src/app/play/settings_chain_tabs.cpp:63-64` | « DistilDRBA : le plus léger » faux au-delà de ×5 | 23 | |
| A23-09 | Mineur | `src/app/play/settings_chain_tabs.cpp:213` | Plafond d'Auto caché, qui dépend du chemin suivi | 23 | |
| A23-10 | Mineur | `src/app/play/settings_panel.cpp:242` | Deux « Niveau » et trois « Rapide » sur une même page | 23 | |
| A23-11 | Mineur | `src/app/play/settings_sound.cpp:72-77` | Délai audio : infobulle ±10 000 ms, borne ±5 000 | 23 | |
| A23-12 | Mineur | `src/app/play/settings_panel.cpp:199` | Délai des sous-titres appliqué à chaque image du glissement | 23 | |
| A23-13 | Mineur | `src/app/play/settings_export.cpp:68` | Page Export : quatre appels au système de fichiers par image | 23 | |
| A23-14 | Mineur | `src/app/play/settings_system.cpp:495-506` | « Signaler un problème » coupe le journal au milieu d'une ligne, sans le dire | 23 | |
| A23-15 | Mineur | `src/app/play/settings_panel.cpp:310-320` | Drapeau `advanced` mort, en-têtes périmés | 23 | |
| A25-01 | Majeur | `src/app/play/engine_screen.cpp:202-311` | UX-02 : assistant relancé en lecture, ses choix et sa chaîne calibrée réécrits par l'ancienne chaîne | 25 | |
| A25-02 | Majeur | `src/app/play/first_run.cpp:263-265` | Sur une RTX, calibration mesurée sous Vulkan sans le dire, jamais refaite quand TensorRT arrive : image plus petite | 25 | |
| A25-03 | Majeur | `src/app/play/engine_screen.cpp:223-227` | UX-05 : durée d'optimisation annoncée fausse (4 min au lieu de 17 à 1080p) | 25 | |
| A25-04 | Majeur | `src/app/play/engine_screen.cpp:286-291` | UX-06 : lecteur par défaut appliqué même quand l'écran Intégration a été passé | 25 | |
| A25-05 | Mineur | `src/app/play/wizard.h:34` | UX-07 : `updates_managed` jamais rempli | 25 | |
| A25-06 | Mineur | `src/app/play/engine_screen.cpp:110-113` | `calibrate_on_screen` rend toujours Auto, même sur annulation | 25 | |
| A25-07 | Mineur | `src/app/play/engine_screen.cpp:274-291` | Inscription « Ouvrir avec » synchrone sur le thread de la fenêtre | 25 | |
| A25-08 | Mineur | `src/app/play/first_run.cpp:461-465` | La calibration ne connaît que RIFE 4.22 lite ×2 : ni DistilDRBA ni RIFE 4.25 essayés | 25 | |
| A25-09 | Mineur | `src/app/play/engine_screen.cpp:218` | Assistant relancé : part des défauts, pas des réglages en place | 25 | |
| A25-10 | Mineur | `src/app/play/engine_screen.cpp:262` | Fenêtre fermée pendant le téléchargement : `settings_after` perdu | 25 | |
| A25-11 | Mineur | `src/app/play/engine_screen.cpp:125` | TRT-18 : dossier de calibration à nom fixe effacé par `remove_all` | 25 | |
| A25-12 | Mineur | `src/app/play/first_run.cpp:195-218` | UX-09 : unités et états en dur dans l'interface | 25 | |
| A25-13 | Mineur | `src/app/play/wizard.cpp:30` | UX-14 : écran Moteurs sauté sans explication | 25 | |
| A25-14 | Mineur | `src/app/play/engine_screen.cpp:72-76` | UX-23 : aucune navigation au clavier | 25 | |
| A25-15 | Mineur | `src/app/play/engine_screen.cpp:43` | Redessin toutes les 16 ms sans événement | 25 | |
| A25-16 | Mineur | `src/app/play/engine_screen.cpp:343` | Quatre `path::string()` (dette, voir A20-02) | 25 | |
| A25-17 | Mineur | `src/app/play/first_run.cpp:138-147` | Code mort : MIGraphX, OpenVINO, DirectML encore nommés | 25 | |
| A25-18 | Mineur | `tests/unit/app/wizard_test.cpp` | `run_engine_screen` et `calibrate_on_screen` sans test | 25 | |
| A26-01 | Majeur | `src/app/play/engine_downloads.cpp:130` | Module TensorRT remplacé seulement sur `AbiMismatch` : les correctifs de noyaux, qualité comprise, n'arrivent jamais | 26 | |
| A26-02 | Majeur | `src/app/play/engine_online.cpp:84-107` | `engines-nvidia.lock` non signé : contourne la signature Ed25519 pour le module et `trtexec` | 26 | |
| A26-03 | Majeur | `src/app/main.cpp:376-386` | Aucune garde contre une version qui ne démarre pas du tout | 26 | |
| A26-04 | Majeur | `src/app/CMakeLists.txt:70` | `PROJECT_VERSION` sans suffixe de préversion : garde désactivée, canal Préversions figé | 26 | |
| A26-05 | Mineur | `src/app/play/settings_system.cpp:200-209` | « Install now » applique `staged_version` sans la comparer | 26 | |
| A26-06 | Mineur | `src/app/play/updater.cpp:466-474` | Archives préparées jamais purgées | 26 | |
| A26-07 | Mineur | `src/app/play/updater.cpp:504-506` | `last_check_unix` posé avant la vérification et le téléchargement | 26 | |
| A26-08 | Mineur | `src/app/play/updater.cpp:490-497` | `update_managed_reason` crée et efface un fichier à chaque appel | 26 | A23-04 (même cause) |
| A26-09 | Mineur | `src/app/main.cpp:526` | Refus d'écriture d'`update.toml` ignoré : compté comme un plantage | 26 | A20-01 (même cause) |
| A26-10 | Mineur | `src/app/play/player.cpp:197-200` | SEC-09 : `update.toml` réécrit en entier par quatre écrivains | 26 | |
| A26-11 | Mineur | `src/app/play/updater.cpp:267-271` | Le canal Préversions ne voit pas une stable plus récente | 26 | |
| A26-12 | Mineur | `src/app/main.cpp:367-389` | Application de la mise à jour sans fenêtre ni journal | 26 | |
| A24-01 | Majeur | `src/app/play/controls.cpp:128-131` | Un seul message de 1,6 s : les avis de qualité perdue (repli DistilDRBA vers RIFE) disparaissent avant d'être lus | 24 | |
| A24-02 | Majeur | `src/app/play/controls.cpp:167-174` | L'interface déclenche des présentations hors cadence pendant la lecture | 24 | |
| A24-03 | Majeur | `src/app/play/ui.cpp:325-343` | Aucun presse-papiers système hors Windows : « Signaler un problème » copie du vide sous CachyOS | 24 | |
| A24-04 | Mineur | `src/app/play/input.cpp:18-58` | `key_of` ne traduit pas des touches que `[keys]` accepte | 24 | |
| A24-05 | Mineur | `src/app/play/input.cpp:76` | AZERTY : répétition et Ctrl + « - » mal traduits (UX-11) | 24 | |
| A24-06 | Mineur | `src/app/play/input.cpp:94-116` | Quinze touches en dur avant `[keys]`, non refusées | 24 | |
| A24-07 | Mineur | `src/app/play/user_keys.cpp:16-24` | Une entrée fautive de `[keys]` remet tous les raccourcis par défaut | 24 | |
| A24-08 | Mineur | `src/app/play/user_keys.cpp:32` | `path::string()` dans le journal (dette, voir A20-02) | 24 | |
| A24-09 | Mineur | `src/app/play/visible_error.h:44-45` | 56 codes sur 71 affichés par leur nom anglais | 24 | A20-09 (même cause) |
| A24-10 | Mineur | `src/app/play/visible_error.h:28` | UX-17 : phrase d'erreur répétée | 24 | |
| A24-11 | Mineur | `src/app/play/controls_draw.cpp:354-371` | UX-15 : textes de préparation sans largeur maximale | 24 | |
| A24-12 | Mineur | `src/app/play/controls_draw.cpp:114-125` | Message bref sans retour à la ligne, avis collés | 24 | |
| A24-13 | Mineur | `src/app/play/panel_style.h:115-130` | UX-22 : badges qui ne suivent pas l'échelle | 24 | |
| A24-14 | Mineur | `src/app/play/ui.cpp:452` | UX-08 : Échap ferme tout le panneau et jette les changements | 24 | |
| A24-15 | Mineur | `src/app/play/home_screen.cpp:205-213` | `dropped()` passe la fenêtre au premier plan et interroge le disque | 24 | |
| A24-16 | Mineur | `src/app/play/home_screen.cpp:244-250` | `save_settings` synchrone sur le thread de l'interface à l'accueil | 24 | |
| A24-17 | Mineur | `src/app/play/home_screen.cpp:336-341` | Accueil redessiné toutes les 16 ms sans événement | 24 | |
| A24-18 | Mineur | `src/app/play/placebo_imgui.cpp:107-124` | Atlas des polices en RGBA32, renvoyé depuis l'origine | 24 | |
| A24-19 | Mineur | `src/app/play/placebo_overlays.cpp:63-76` | Barre visible : panneaux typographiés montés avec les dialogues | 24 | |
| A24-20 | Mineur | `src/app/play/placebo_overlays.cpp:45-47` | Sous-titres étirés à la nouvelle taille jusqu'au rendu suivant | 24 | |
| A24-21 | Mineur | `src/app/play/placebo_imgui.cpp:161-207` | Une passe de dessin par commande ImGui | 24 | |
| A24-22 | Mineur | `src/app/play/osd_text.cpp:68` | Seconde bibliothèque libass pour l'OSD | 24 | |
| A24-23 | Mineur | `src/app/play/ui.cpp:117-129` | Polices de secours CJK différentes sous Windows et Linux | 24 | |
| A49-01 | Majeur | `src/app/play/auto_config.cpp:69-75` | L'Auto ne joue jamais le SPAN b8f64, même en Ultra | 49 | |
| A49-02 | Majeur | `src/app/play/auto_config.cpp:48` | Source 1080p réduite à 720p ou 540p avant SPAN | 49 | Aggravé par 07 (complément) : sur une carte Qualité ou Ultra, calibrer donne une image moins bonne que ne pas calibrer |
| A49-03 | Majeur | `src/app/play/auto_config.cpp:167-169` | Auto calibré en FP8 malgré l'ADR 0007 suspendue, sans essayer le b8f64 FP16, pour 6,5 % | 49 | |
| A49-04 | Majeur | `src/app/play/options.h:113-114` | Anneau RGB10A2 par défaut sans la porte de banding de l'ADR 0008 | 49 | A21-03 (même cause) |
| A49-05 | Majeur | `src/render/cube_lut.cpp:289-314` | Pile de looks composée en LUT 65³ : jusqu'à 15,5 codes PQ d'écart (mesuré) | 49 | |
| A49-06 | Majeur | `src/gpu/cuda/backend/warp_kernels.cu:144-171` | Warp après SPAN bilinéaire, gain jamais mesuré | 49 | |
| A49-07 | Mineur | `src/app/play/span_int8.cpp:194-202` | Cran Équilibre annoncé FP16 mais INT8 dès qu'un reçu existe | 49 | |
| A49-08 | Mineur | `src/gpu/span_fp8/derive.h:17-37` | Porte du FP8 réglée pour laisser passer le modèle | 49 | |
| A49-09 | Mineur | `src/app/play/upscale_motion.cpp:97-193` | Une seule passe RIFE par paire sous 15 px : mouvements non linéaires | 49 | |
| A49-10 | Mineur | `src/app/play/chain_setup.cpp:27` | RIFE estime le mouvement à 720p au-dessus de 720p | 49 | |
| A49-11 | Mineur | `src/app/play/upscale_stage.cpp:498-504` | Dessins « tenus » : sortie SPAN réutilisée, petits changements gelés | 49 | |
| A49-12 | Mineur | `src/app/play/placebo_screen.cpp:130` | Présentation par défaut de libplacebo, sans anti-ringing | 49 | |
| A49-13 | Mineur | `src/media/frame_data.cpp:247-275` | Conversion hôte bilinéaire, 12 bits ramené à 10 | 49 | |
| A49-14 | Mineur | `tools/rife/convert_rife_fp16.py` | Aucune mesure des RIFE et DistilDRBA FP16 contre leurs originaux FP32 | 49 | |
| A29-01 | Majeur | `src/app/play/chain_setup.cpp:497` | Un refus de RIFE ou de SPAN arrête l'épisode et renvoie à l'accueil, sans repli | 29 | |
| A29-02 | Majeur | aucun fichier (`WM_POWERBROADCAST`, `PrepareForSleep` absents) | La mise en veille n'est traitée nulle part | 29 | |
| A29-03 | Mineur | `src/app/play/player_run.cpp:617` | Réouverture du même fichier traitée comme un fichier suivant | 29 | |
| A29-04 | Mineur | `src/app/main.cpp:528` | Après mise à jour d'un moteur, la relance rejoue `argv` et perd la vidéo en cours | 29 | |
| A29-05 | Mineur | `src/app/play/settings_panel.cpp:572-574` | DistilDRBA ne se choisit qu'à la main, hors Auto et hors calibration | 29 | |
| A29-06 | Mineur | `src/app/play/player_engines.cpp:146` | Désérialisation des moteurs TensorRT sur le thread de la fenêtre sans pompe | 29 | |
| A29-07 | Mineur | `src/app/play/engine_screen.cpp:164-167` | « Refaire le banc » annulé en lecture : la chaîne repasse en Auto | 29 | |
| A31-01 | Majeur | `src/app/play/ring_stage.cpp:106-113` | Tout refus GPU devient `ThreadFailed` : `DeviceLost` et sa phrase perdus, figé par un test | 31 | A20-09 (même cause) |
| A31-02 | Majeur | `src/app/play/player.cpp:589-602` | Aucune relance après perte du GPU alors que le mécanisme existe | 31 | |
| A31-03 | Majeur | `src/app/play/engine_screen.cpp:341-344` | TensorRT refusé remplacé par Vulkan sans avis à l'écran | 31 | |
| A31-04 | Majeur | `src/app/play/chain_setup.cpp:355` | Un moteur refusé à l'ouverture empêche toute lecture, rien n'est retenu | 31 | A29-01 (même cause) |
| A31-05 | Majeur | `src/app/play/player.cpp:523-533` | « Annuler et regarder » coupe aussi SPAN déjà prêt | 31 | |
| A31-06 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | Cran Qualité ou Performance retombé en FP16 : garde le refus du fusionné, plus lent qu'Équilibre (toujours sous Windows en Performance) | 31 | Touche aussi CachyOS (34) |
| A31-07 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | DistilDRBA Performance avec graphe FP8 : moins de qualité et plus lent que le fusionné FP16 | 31 | |
| A31-08 | Mineur | `src/gpu/cuda/backend/cuda_status.h:38-39` | `cudaErrorMemoryAllocation` devient `VendorCallFailed` | 31 | |
| A31-09 | Mineur | `src/app/play/upscale_drba.cpp:136` | Image ingérée avant le contrôle de taille | 31 | |
| A31-10 | Mineur | `src/app/play/host_facts.h:48-51` | Sur batterie, Auto plafonné à 60 Hz sans avis, et retenu | 31 | |
| A31-11 | Mineur | `src/core/file_sync.cpp:75-91` | Disque plein : place vérifiée seulement à l'installation des moteurs | 31 | |
| A31-12 | Mineur | `src/app/play/player_run.cpp:510` | Veille et hibernation non gérées | 31 | A29-02 (même cause) |
| A31-13 | Mineur | `src/app/play/placebo_ring.cpp:218` | `ready()` vrai quand le sémaphore échoue | 31 | |
| A31-14 | Mineur | `src/gpu/cuda/backend/tensorrt_export.cpp:262-263` | GPU bloqué sans être perdu : attente sans délai | 31 | |
| A31-15 | Majeur | `src/app/play/player_engines.cpp:259-260` | « Annuler » sur DistilDRBA fusionné lance la construction de l'entier, puis de RIFE ; même chose pour RIFE fusionné | 31 | |
| A30-01 | Majeur | `src/gpu/cuda/backend/trt_engine.cpp:115-119` | Moteurs TensorRT détruits sans attendre les files CUDA, à chaque fermeture de chaîne (à confirmer) | 30 | |
| A30-02 | Mineur | `src/app/play/chain_hdr.cpp:153-163` | `LookLoader` détaché qui journalise après `close_log` : plantage à la sortie avec A18-01 (GPU-12) | 30 | |
| A30-03 | Mineur | `src/app/play/settings_system.cpp:177-216` | Deux vérifications de mise à jour en parallèle sur la même archive | 30 | |
| A30-04 | Mineur | `src/render/grade.cpp:316` | Thread des LUT en `std::jthread` brut, sans `try` (GPU-11) | 30 | |
| A30-05 | Mineur | `src/app/play/pumped_task.h:47-58` | Sonde du fichier détachée d'office (QUA-13) | 30 | |
| A30-06 | Mineur | `src/app/play/single_instance.cpp:155-157` | `ReadFile` sans délai sur le canal de l'instance unique (WIN-06) | 30 | |
| A30-07 | Mineur | `tests/unit/app/` | Aucun test des durées de vie | 30 | |
| A30-08 | Mineur | `src/media/shot_probe.cpp:108-110` | Attentes sans délai sur le thread de la fenêtre | 30 | |
| A27-01 | Majeur | `ports/libplacebo/portfile.cmake:8` | libplacebo épinglé à 7.360.1, l'amont à 7.387.0 avec des correctifs de rendu ; notre correctif ne s'applique plus | 27 | |
| A27-02 | Mineur | `tools/release/package-linux.sh:124` | `patchelf --set-rpath` pose un RUNPATH que `LD_LIBRARY_PATH` devance (démontré) | 27 | |
| A27-03 | Mineur | `cmake/Dependencies.cmake:64` | Configure accepte un libplacebo sans notre correctif | 27 | |
| A27-04 | Mineur | `cmake/Dependencies.cmake:4-5` | « Mêmes versions » Linux et Windows vrai en partie seulement | 27 | |
| A27-05 | Mineur | `tools/deps/linux-from-source.sh:135-138` | Options de libplacebo fixées sous Windows seulement | 27 | |
| A27-06 | Mineur | `tools/deps/linux-from-source.sh:58-66` | Marqueur de dépendance qui ignore les bibliothèques du système liées | 27 | |
| A27-07 | Mineur | `CMakeLists.txt:28` | Le CI CUDA des PR ne compile pas toutes les architectures livrées | 27 | |
| A27-08 | Mineur | `ports/libplacebo/0001-ajp-present-timing.patch:153-170` | Chemin `QUEUE_FULL` du correctif sur une hypothèse non vérifiée | 27 | |
| A27-09 | Mineur | `cmake/Dependencies.cmake:115-117` | ImGui sans FreeType sous Windows | 27 | |
| A27-10 | Mineur | `CMakePresets.json:18` | `AJP_WARNINGS_AS_ERRORS` OFF par défaut, contrairement à la règle | 27 | |
| A27-11 | Mineur | `cmake/AjpSpirv.cmake:35` | `spirv-val` absent sous Windows sans un mot | 27 | |
| A27-12 | Mineur | `third_party/cutlass/` | CUTLASS recopié sans commit ni somme | 27 | |
| A27-13 | Mineur | `cmake/AjpModule.cmake:61-77` | En-têtes précompilés sans effet sur les tests | 27 | |
| A28-01 | Bloquant | `.github/workflows/release-linux.yml:96-102` | Release Linux sans paquet TensorRT publiée sur un simple avertissement : 0.10.3 et 0.10.1 Linux parties sans TensorRT (démontré) | 28 | |
| A28-02 | Majeur | `tools/release/sign-manifest.sh` | Aucun manifeste signé publié depuis la 0.5.1 : tout le canal Stable se croit à jour (démontré) | 28 | |
| A28-03 | Majeur | `.github/workflows/release-linux.yml:146-150` | `--latest` posé par chaque plateforme avant le manifeste | 28 | |
| A28-04 | Majeur | `.github/workflows/release-guard.yml:24-39` | Étiquette jamais comparée à `PROJECT_VERSION` | 28 | |
| A28-05 | Majeur | `src/app/CMakeLists.txt:69` | Trois dépôts de release par défaut différents | 28 | |
| A28-06 | Majeur | `tools/release/fetch-models.sh:172-190` | `models-interp-1` non épinglé remplace le DistilDRBA épinglé (démontré, même graphe aujourd'hui) | 28 | |
| A28-07 | Majeur | `tests/unit/prepare/CMakeLists.txt:13-20` | Tests du chemin livré non lancés avant la publication | 28 | |
| A28-08 | Majeur | `.github/workflows/` (16 `actions/checkout`) | Jetons gardés par `checkout` dans les jobs de release | 28 | |
| A28-09 | Mineur | `tools/release/package-linux.sh:190` | Version TensorRT de `engines.lock` réduite au numéro majeur (démontré) | 28 | |
| A28-10 | Mineur | `tools/release/preflight/nvidia-package.sh:24-32` | Deux chemins publient le même paquet `trt-<empreinte>` | 28 | |
| A28-11 | Mineur | `.github/workflows/ci.yml:219-251` | CUDA des PR : `sm_89` seul, en Debug | 28 | A27-07 (même cause) |
| A28-12 | Mineur | `CMakePresets.json:458` | Filtres des tests de livraison trop larges | 28 | |
| A28-13 | Mineur | `.github/workflows/fuzz.yml:63-64` | UBSan sans `-fno-sanitize-recover` dans le fuzz de nuit | 28 | |
| A28-14 | Mineur | `.github/workflows/export-interpolation-models.yml:58-61` | `models-interp-1` créée sans `--latest=false` | 28 | |
| A28-15 | Mineur | `.github/workflows/release-windows.yml` | Release Windows sans groupe de concurrence | 28 | |
| A28-16 | Mineur | `tools/release/appimage.sh:145` | `appimagetool` téléchargé sans SHA-256 puis exécuté | 28 | |
| A28-17 | Mineur | `.github/workflows/release-linux.yml:10-11` | En-têtes qui promettent des chemins interdits par la garde | 28 | |
| A32-01 | Majeur | `docs/auto-libre.md:142-143` | Le service de 13 à 16 ms date de la chaîne avec RTX HDR : à remesurer ; 1,7 à 2,7 ms de GPU estimés par image à 1440p | 32 | |
| A32-02 | Majeur | `src/gpu/cuda/backend/tensorrt_backend.cpp:407-498` | `gpu_profile` ne voit ni l'écran, ni les sémaphores, ni les attentes hôte | 32 | |
| A32-03 | Majeur | `tests/unit/` | Aucun test de la sortie du lecteur contre un original | 32 | |
| A32-04 | Majeur | `src/app/play/upscale_drba.cpp:179-256` | DistilDRBA refait deux `rife_input` par image créée, plus cher que son réseau | 32 | A03-14 (même cause) |
| A32-05 | Mineur | `src/render/grade.cpp:355` | Passes d'écran du grade à fusionner | 32 | |
| A32-06 | Mineur | `src/gpu/cuda/backend/warp_kernels.cu:100-170` | Warp en plans séparés : une vingtaine de lectures par pixel | 32 | |
| A32-07 | Mineur | `src/app/play/placebo_ring.cpp:214-218` | `ready()` interroge le pilote à chaque question | 32 | |
| A32-08 | Mineur | `src/core/free_rate.cpp:27-36` | Pas du mode Rapide calé sur un temps de service en temps de mur | 32 | |
| A32-09 | Mineur | `docs/handoff-pc.md:77-78` | Temps de référence périmés | 32 | |
| A34-01 | Majeur | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | Auto calibré : b8f64 FP8 CUTLASS sous Linux, b5f48 FP16 ou INT8 sous Windows | 34 | Précisé par 07 (complément) : sous Windows en 1080p, b8f64 INT8 ou FP16 |
| A34-02 | Majeur | `src/media/shot_probe.cpp:165-175` | Pré-analyse sur un thread sous Windows : grade et coupes privés sur les sources lourdes (à confirmer) | 34 | Réfuté en grande partie par 42 (NVDEC par défaut, seul le repli logiciel est touché) |
| A34-03 | Mineur | `src/app/play/single_instance.cpp:114-120` | Canal nommé de l'instance unique à nom prévisible | 34 | |
| A34-04 | Mineur | `src/app/play/default_player.cpp:203` | Plusieurs épisodes ouverts : un processus par fichier | 34 | |
| A34-05 | Mineur | `src/app/play/system_peak.cpp:136-158` | Fabrique DXGI recréée toutes les 2 s | 34 | |
| A34-06 | Mineur | `src/app/play/auto_bench_run.cpp:48-61` | `hide_is_reliable*` morts | 34 | |
| A34-07 | Mineur | `src/app/play/loop_wait.cpp:45-57` | Attente précise à 1 ms sous Windows (`--precise-wait`) | 34 | |
| A34-08 | Mineur | `src/media/cuda_driver.cpp:92-110` | `libcuda.so.1` chargé par son seul nom sous Linux | 34 | |
| A34-09 | Mineur | `cmake/utf8.manifest:1-9` | Manifeste sans `supportedOS` ni `dpiAwareness` | 34 | |
| A34-10 | Mineur | `src/app/play/placebo_open.cpp:510-515` | Présentation FIFO identique partout malgré la swapchain NVIDIA de Windows | 34 | |
| A34-11 | Mineur | `src/subs/text_encoding.cpp:117-145` | WIN-09 vrai en partie seulement | 34 | |
| A34-12 | Mineur | `src/app/play/data_dirs.cpp:67-101` | Migration en cours qui bloque l'instance unique | 34 | |
| A33-01 | Majeur | `src/app/play/placebo_ring.cpp:84` | Mémoire vidéo d'une chaîne fermée gardée (anneau, réserve CUDA) ; « Refaire le banc » mesure avec moins de mémoire (TRT-06 élargi) | 33 | |
| A33-02 | Majeur | `src/app/play/placebo_open.cpp` | SDL coupe l'économiseur et la veille pendant toute la vie du processus, même en pause et à l'accueil | 33 | |
| A33-03 | Majeur | `src/core/clock.h:15` | Aucun trou de temps détecté : au réveil sous Windows, l'horloge court jusqu'à la fin de l'épisode | 33 | A29-02 (même cause, précisé) |
| A33-04 | Majeur | `src/prepare/cache.cpp:81` | Cache des moteurs TensorRT sans aucune purge | 33 | |
| A33-05 | Mineur | `src/gpu/cuda/backend/gpu_timing.h:53` | Profil GPU de fin de lecture tronqué | 33 | |
| A33-06 | Mineur | `src/app/main.cpp:427-440` | Deux rotations du journal qui s'ignorent | 33 | |
| A33-07 | Mineur | `src/app/play/pumped_task.h:58` | Sonde détachée qui garde son descripteur | 33 | A30-05 (même cause) |
| A33-08 | Mineur | `src/app/play/gpu_telemetry.cpp` | Aucun essai de durée ni relevé de ressources | 33 | |
| A35-01 | Majeur | `tools/deps/linux-from-source.sh:35` | FFmpeg, libass (fork), dav1d, FreeType et HarfBuzz figés sans veille des failles ; A28-02 bloque tout correctif | 35 | |
| A35-02 | Mineur | `src/media/demuxer.cpp:190-199` | Chapitres dédoublonnés en O(n²) : 7 s par ouverture à 100 000 chapitres (démontré) | 35 | |
| A35-03 | Mineur | `src/core/update_manifest.cpp:21-28` | Préversion multiligne affichée telle quelle (SEC-10) | 35 | |
| A35-04 | Mineur | `src/core/model_index.cpp:33-40` | `safe_name` accepte les noms réservés de Windows (démontré) | 35 | |
| A35-05 | Mineur | `src/app/play/engine_tools.cpp:79-88` | Téléchargements sans `--proto =https` ni `--proto-redir` | 35 | |
| A35-06 | Mineur | `src/media/demuxer.cpp:116-117` | Aucune borne haute sur la taille annoncée par le fichier | 35 | |
| A35-07 | Mineur | `src/app/play/updater.cpp:557-564` | `staged_file` non confiné à `updates/` | 35 | |
| A35-08 | Mineur | `tests/fuzz/CMakeLists.txt:18-20` | Lecteurs d'octets extérieurs non fuzzés | 35 | |
| A35-09 | Mineur | `src/app/play/look_catalog.cpp:42-44` | Nom d'un look gardé contre `/` et `\` seulement | 35 | |
| A36-01 | Majeur | `src/app/play/settings_toml_lenient.cpp:73-74` | Au retour arrière, `play.toml` perd toutes les clés de la version récente dès qu'une valeur lui est inconnue (démontré) | 36 | A22-06 (même cause, gravité relevée) |
| A36-02 | Majeur | `src/app/play/machine_calibration.cpp:240-252` | `calibration.toml` jamais invalidé par la version du lecteur, du module ni par le pilote : Auto calibré bridé | 36 | |
| A36-03 | Majeur | `src/app/play/engine_install.cpp:539-571` | Paquet TensorRT installé sans fsync et jugé sur sa présence : Vulkan pour toujours après une coupure (à confirmer) | 36 | |
| A36-04 | Mineur | `src/app/play/derived_model.cpp:18-22` | Nom d'un modèle dérivé sans la version du dériveur | 36 | |
| A36-05 | Mineur | `src/app/play/updater.cpp:273-293` | `update.toml` réécrit avec les seuls champs connus | 36 | |
| A36-06 | Mineur | `src/core/model_index.cpp:440-442` | Anciens paquets TensorRT jamais retirés | 36 | |
| A36-07 | Mineur | `tools/release/windows/ajp.iss:16-50` | Désinstaller laisse plusieurs Go sans le dire | 36 | |
| A36-08 | Mineur | `src/app/play/paths.h:13-36` | Caches reconstructibles hors de `~/.cache` sous CachyOS | 36 | |
| A36-09 | Mineur | `tests/unit/app/settings_test.cpp:620-645` | Aucun test de passage entre versions hors des clés inconnues | 36 | |
| A37-01 | Majeur | `src/gpu/cuda/backend/span_fp8_net.cu:225-228` | SPAN FP8 CUTLASS déclaré pris en charge sur RTX 50 sans y avoir tourné ; seule garde la porte aveugle aux NaN (à confirmer) | 37 | |
| A37-02 | Mineur | `src/app/play/home_screen.cpp:141` | FP8 présumé par le nom de la carte à l'accueil (RTX 20 et 30) | 37 | |
| A37-03 | Mineur | `src/app/play/engine_install.cpp:97-115` | `cuda_arch_from_name` faux pour la RTX 2050 et une autre carte de la cible | 37 | |
| A37-04 | Mineur | `data/gpu_tiers.toml:15-24` | Portables RTX 20 sans « Laptop » dans leur nom | 37 | |
| A37-05 | Mineur | `src/app/play/auto_config.cpp:176-180` | Qualité : baisse choisie par le nom de la carte en Auto estimé, sans mesure | 37 | |
| A37-06 | Mineur | `src/app/play/auto_config.cpp:131-138` | Hauteur de travail calée sur l'écran, pas sur la source | 37 | |
| A37-07 | Mineur | `src/app/play/auto_config.cpp:223-225` | Chaîne d'un préréglage refaite à chaque vidéo | 37 | |
| A37-08 | Mineur | `src/app/play/first_run.cpp:245-253` | Pilote minimum de TensorRT vérifié seulement à l'écran des moteurs | 37 | |
| A37-09 | Mineur | `src/app/play/auto_bench.cpp:207` | Deux mesures de la machine qui traitent la batterie différemment | 37 | |
| A37-10 | Mineur | `src/app/play/machine_calibration.cpp:160-166` | Une seule calibration par machine (une carte, un écran) | 37 | |
| A37-11 | Mineur | `src/app/play/first_run.cpp:220-230` | Pilote libre NVK pris pour le pilote NVIDIA | 37 | |
| A37-12 | Mineur | `src/app/play/home_screen.cpp` | Aucun test des décisions par configuration | 37 | |
| A37-13 | Mineur | `src/gpu/cuda/backend/tensorrt_lanes.cpp:25` | Nombre de voies décidé sur la mémoire totale, pas la libre | 37 | |
| A37-14 | Mineur | `src/app/play/auto_config.cpp:55` | En Stable, le plafond ×4 coupe les paliers des écrans rapides | 37 | |
| A39-01 | Majeur | `src/app/main.cpp:424-433` | Après un plantage, « Report a problem » copie la nouvelle séance, pas le journal du plantage | 39 | |
| A39-02 | Majeur | `src/app/play/settings_system.cpp:498` | Le rapport ne dit ni le système, ni le pilote, ni le préréglage, ni la raison du refus de TensorRT | 39 | |
| A39-03 | Majeur | `src/app/play/chain.cpp:352-353` | UX-13 : chemins complets des vidéos dans le journal envoyé | 39 | |
| A39-04 | Majeur | `src/app/play/crash_dump.cpp:47-50` | WIN-07 : minidumps sans PDB, jamais signalés, rien sous Linux | 39 | |
| A39-05 | Mineur | `src/app/main.cpp:458` | Messages de FFmpeg absents d'`ajp.log` | 39 | |
| A39-06 | Mineur | `src/app/play/chain_stats.cpp:167-174` | Télémétrie GPU absente du journal | 39 | |
| A39-07 | Mineur | `src/app/play/chain_setup.cpp:352-353` | Qualité : cran Qualité sans FP32 retombe en silence | 39 | |
| A39-08 | Mineur | `.github/ISSUE_TEMPLATE/bug.yml` | Formulaire de bogue avec des niveaux périmés | 39 | |
| A39-09 | Mineur | `src/app/play/settings_system.cpp:499` | Le bouton lit `ajp.log` au mauvais endroit | 39 | |
| A38-01 | Majeur | `src/app/play/settings_system.cpp:493-511` | « Signaler un problème » copie les chemins complets hors du dossier personnel, nom d'utilisateur compris | 38 | A39-03 (même cause) |
| A38-02 | Majeur | `README.md:32` | Le README promet RIFE 4.25 en Qualité et Ultra, l'Auto calibré impose 4.22 lite : baisse de qualité sans mesure | 38 | A25-08 (même cause) |
| A38-03 | Mineur | `README.md:74-83` | Tableau des niveaux faux | 38 | |
| A38-04 | Mineur | `README.md:29-31` | « Une source 1080p devient 4K » faux en Auto | 38 | |
| A38-05 | Mineur | `README.md:57-66` | « Premier lancement » décrit un assistant qui n'existe plus | 38 | |
| A38-06 | Mineur | `README.md:140` | Promesse fausse sur la construction des moteurs au premier lancement | 38 | |
| A38-07 | Mineur | `README.md:129` | Raccourcis « dans `user.toml` » : quinze touches en dur | 38 | |
| A38-08 | Mineur | `README.md:4` | Badge 0.10.0, code 0.10.3, CHANGELOG arrêté | 38 | |
| A38-09 | Mineur | `THIRD_PARTY_NOTICES.md:40-49` | Notices tierces incomplètes | 38 | |
| A38-10 | Mineur | `tools/release/notes/v0.10.3.md:3` | Notes 0.10.3 inexactes | 38 | |
| A38-11 | Mineur | `src/app/play/options.cpp:470` | `--help` toujours faux | 38 | |
| A38-12 | Mineur | `src/app/play/settings_chain_tabs.cpp:260-262` | Deux infobulles du panneau fausses | 38 | |
| A38-13 | Mineur | `src/app/play/auto_config.h:19-22` | Commentaire qui renvoie à une porte de qualité DistilDRBA inexistante | 38 | |
| A38-14 | Mineur | `.github/ISSUE_TEMPLATE/bug.yml:56-60` | Formulaire sans Économe ni Personnalisé | 38 | A39-08 (même cause) |
| A38-15 | Mineur | `src/render/shaders/sharpen.hook:1-5` | Netteté promise « sans halo » | 38 | |
| A42-01 | Majeur | `src/media/shot_cutter.cpp:163` | Seuil de coupe sur la médiane, dessins tenus compris : un panoramique en trois devient une coupe par dessin (démontré, 24 plans sur 96 images) | 42 | |
| A42-02 | Mineur | `src/media/shot_cutter.cpp:167-168` | Éclair de deux images pris pour une coupe | 42 | |
| A42-03 | Mineur | `src/media/shot_timeline.cpp:49-52` | Un saut pendant l'analyse laisse un trou | 42 | |
| A42-04 | Mineur | `src/app/play/chain_shots.cpp:177-179` | `next_shot` retombe sur la scène en direct | 42 | |
| A42-05 | Mineur | `src/app/play/options.h:138` | Pré-analyse active pendant les essais de calibration | 42 | |
| A42-06 | Mineur | `src/media/shot_probe.h:37-41` | Qualité : analyse en 256×144 sous NVDEC, mesurée en partie | 42 | |
| A42-07 | Mineur | `src/media/shot_probe.cpp:66-68` | G calculé depuis R et B déjà bornés | 42 | |
| A42-08 | Mineur | `src/media/flash_guard.h:1-5` | Limiteur de flash sur la luminance moyenne seule | 42 | |
| A42-09 | Mineur | `src/app/play/chain_shots.cpp:93-96` | Coupes de l'interpolation dépendantes du grade | 42 | |
| A42-10 | Mineur | `src/media/shot_cache.cpp:20-21` | Cache versionné par son format seulement | 42 | |
| A42-11 | Mineur | `src/app/play/chain_shots.cpp:166-174` | Remplacement de plan à nombre égal non suivi | 42 | |
| A42-12 | Mineur | `src/media/shot_probe.cpp:108-110` | Sonde détruite sur le thread de la fenêtre | 42 | A30-08 (même cause) |
| A42-13 | Mineur | `src/media/shot_cutter.cpp:173` | Images du plan en cours toutes gardées et recopiées | 42 | |
| A42-14 | Mineur | `src/app/play/chain_shots.cpp:23-35` | Empreinte sur le premier mégaoctet seul | 42 | |
| A42-15 | Mineur | `src/app/play/chain_shots.cpp:44` | `path::string()` (dette) | 42 | |
| A42-16 | Mineur | `tests/unit/media/shot_test.cpp` | Aucun test de l'animation tenue ni de l'éclair de deux images | 42 | |
| A41-01 | Majeur | `src/render/cube_lut.cpp:287-314` | Pile de LUT composée en 65³ : 18,0 codes PQ d'écart au pire à l'écran, 129³ ne suffit pas, seule la chaîne dans le shader est exacte | 41 | A49-05 (même cause) |
| A41-02 | Majeur | `src/render/shaders/anime-hdr-anime/3-anime.hook:119` | La pile et la passe kwin défont le grade : blanc à 695 nits sur 1 000, montée des émetteurs divisée par deux, limiteur OLED presque sans effet, MaxFALL faux (question à Tatoub) | 41 | |
| A41-03 | Mineur | `data/luts/HDR grading base ver.9.0 33 point.cube` | Ombres écrasées : 0,30 nit (RVB 16) rendu à 0,02 (question à Tatoub) | 41 | |
| A41-04 | Mineur | `src/app/play/placebo_grade.cpp:206-225` | Sans pile, la passe kwin reste active : blanc à 119 nits ; échec du hook non journalisé | 41 | |
| A41-05 | Mineur | `src/render/grade.cpp:361-365` | `set_looks` lit un hook complet pour rien sur le thread de la boucle (images perdues) | 41 | |
| A41-06 | Mineur | `src/render/shaders/sharpen.hook:14` | Netteté sur `SCALED`, domaine gamma ou linéaire selon la mise à l'échelle | 41 | |
| A41-07 | Mineur | `src/render/grade_params.cpp:158-166` | Vrai noir annoncé mais jamais appliqué en sortie SDR | 41 | |
| A41-08 | Mineur | `src/render/grade_params.cpp:115-116` | Source HDR non gradée : pic de l'écran non annoncé, hautes lumières écrêtées | 41 | |
| A41-09 | Mineur | `src/render/grade.cpp:275-298` | Thread des LUT sans `try` | 41 | A30-04 (même cause) |
| A41-10 | Mineur | `src/render/grade_params.cpp:56-70` | Chemins morts du grade SDR, quinze emplacements de LUT, pile composée deux fois (dette) | 41 | |
| A41-11 | Mineur | `tests/unit/render/cube_lut_test.cpp:146` | Aucun test sur les LUT livrées ni sur la chaîne grade, pile, kwin | 41 | |
| A43-01 | Majeur | `tests/unit/app/i18n_test.cpp:83` | Le test i18n laisse passer `ui_tr`, les clés passées par variable et les textes français en dur | 43 | |
| A43-02 | Mineur | `src/app/play/i18n_fr_sound.cpp` | 192 entrées de traduction mortes sur 946 (UX-21) | 43 | |
| A43-03 | Mineur | `src/app/play/i18n.h:30-33` | `tr_format` ne localise pas les nombres : point en français, virgule en anglais (UX-10) | 43 | |
| A43-04 | Mineur | `src/app/play/i18n_fr_wizard.cpp:11` | Cinq noms pour le test de la machine qui règle Auto | 43 | |
| A43-05 | Mineur | `src/app/play/settings_panel.cpp:283` | Infobulle « Finish » alors que l'assistant n'a pas ce bouton | 43 | |
| A43-06 | Mineur | `src/app/play/i18n_fr_image.cpp:88` | Qualité : le coût en précision du FP8 n'est jamais dit ; « Balanced » nomme deux choses | 43 | |
| A43-07 | Mineur | `src/app/play/french_text.h:12-30` | Typographie française appliquée à moitié | 43 | |
| A45-01 | Majeur | `src/gpu/vulkan/backend/net_presets.cpp:1-29` | Qualité : la table pré-réglée NvidiaAda met 28 couches sur 29 de SPAN et RIFE 4.25 en sommes FP16 dans le repli Vulkan, sans mesure (erreur par couche ×4 à ×9 en simulation) | 45 | |
| A45-02 | Mineur | `src/gpu/vulkan/backend/net_autotune.cpp:101-115` | Qualité : porte de l'autotuner trop large et jugée contre FP16 | 45 | |
| A45-03 | Mineur | `src/gpu/vulkan/backend/kernel_ops.h:116-143` | Qualité : pas de débandage avant SPAN sous Vulkan | 45 | |
| A45-04 | Mineur | `src/gpu/vulkan/backend/drba_fused_engine.cpp:335-341` | DistilDRBA fusionné Vulkan : encodeur refait trois images par appel | 45 | A03-15 (même cause) |
| A45-05 | Mineur | `src/gpu/vulkan/backend/queue.cpp:117-125` | Un `vkQueueSubmit` refusé fige la file pour de bon | 45 | |
| A45-06 | Mineur | `src/gpu/vulkan/backend/kernel_frames.cpp:238-248` | Échec des vues : refus rendu au lieu de la copie promise | 45 | |
| A45-07 | Mineur | `src/gpu/vulkan/backend/net_build.cpp:194-218` | Écriture d'artefacts sans fsync, temporaires jamais purgés (QUA-05) | 45 | |
| A45-08 | Mineur | `src/gpu/vulkan/backend/entry.cpp:7-9` | `--profile-dispatches` sans effet dans le lecteur | 45 | |
| A45-09 | Mineur | `src/gpu/vulkan/compute/library.cpp:43-99` | `VulkanLibrary` recopie `DynamicLibrary` (dette) | 45 | |
| A45-10 | Mineur | `src/gpu/vulkan/backend/entry_points.h:1-6` | Commentaires et code d'interop MIGraphX, OpenVINO, HIP morts (dette) | 45 | |
| A45-11 | Mineur | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70` | Tests de concordance des moteurs fusionnés Vulkan cachés, hors CI | 45 | |
| A45-12 | Mineur | `src/gpu/vulkan/backend/net_presets.cpp:4` | Table pré-réglée limitée à SPAN b5f48 720p et RIFE 4.25 | 45 | |
| A45-13 | Mineur | `src/gpu/vulkan/backend/vulkan_backend.cpp:384-395` | Artefact réabaissé et réécrit sans repasser la porte | 45 | |
| A45-14 | Mineur | `src/gpu/vulkan/backend/kernel_ops.h:116-143` | Ni mémoire de la carte ni temps par étage dans les statistiques du repli | 45 | |
| A44-01 | Majeur | `src/gpu/ir/pass_cast.cpp` | Qualité, repli Vulkan : chaîne de grille de DistilDRBA en F16 (jusqu'à 2,07 dB et écart max 16,3/255 sur un panoramique réel, coût du F32 négligeable) | 44 | A03-18 (même cause, mesuré côté IR) |
| A44-02 | Mineur | `src/gpu/ir/pass_fuse.cpp` | Pas de fusion élément par élément : DistilDRBA 230 dispatchs contre 134 pour RIFE 4.22 lite, lié aux dispatchs (à confirmer sur GPU) | 44 | |
| A44-03 | Mineur | `src/gpu/ir/import.cpp:141` | Tri topologique quadratique : un ONNX hostile de 0,9 Mo prend 8,5 s | 44 | |
| A44-04 | Mineur | `src/gpu/ir/onnx_reader.cpp` | Amplification mémoire ×8 à ×16 bornée seulement par la taille du fichier | 44 | |
| A44-05 | Mineur | `src/gpu/ir/import.cpp:238` | `path.string()` dans les messages d'erreur (dette) | 44 | |
| A44-06 | Mineur | `src/gpu/ir/pass_fuse.cpp:103` | Facteur par canal fondu dans des poids F16 : milliers de poids à zéro, sans perte mesurée | 44 | |
| A44-07 | Mineur | `src/gpu/ir/onnx_reader.cpp:208` | Valeurs INT32 non vérifiées en plage | 44 | |
| A44-08 | Mineur | `src/gpu/ir/onnx_io_type.cpp:648` | `with_float32` mort depuis le retrait de VitisAI (dette) | 44 | |
| A44-09 | Mineur | `src/gpu/ir/describe.cpp:117` | `describe_stats` n'affiche pas les quantifications FP8 (dette) | 44 | |
| A44-10 | Mineur | `src/gpu/ir/onnx_info_main.cpp` | `ajp_onnx_info` ne rejoue pas les passes propres à Vulkan | 44 | |
| A44-11 | Mineur | `tests/unit/gpu_ir/` | Aucun test sur les modèles livrés ; passes Vulkan hors fuzz | 44 | |
| A44-12 | Mineur | `src/gpu/ir/pass_transpose.cpp` | Deux DepthToSpace là où un suffit en mode phases | 44 | |
| A46-01 | Majeur | `tools/release/fetch-models.sh:80-98` | Le DistilDRBA épinglé est écrasé dans les paquets par une autre version, celle de `models-interp-1` | 46 | |
| A46-02 | Majeur | `.github/workflows/export-interpolation-models.yml:34-35` | Export de DistilDRBA non reproductible : torch, onnx et onnxruntime non épinglés | 46 | |
| A46-03 | Majeur | `tools/drba/export_distil_drba.py:194-197` | Qualité : le graphe FP16 livré n'est jamais comparé au réseau d'origine (16,5 codes 10 bits d'écart mesuré) | 46 | |
| A46-04 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:64-69` | Qualité : la voie fusionnée de DistilDRBA construit aussi la grille en FP16, alors que RIFE fusionné la calcule en FP32 | 46 | A03-18 (même cause) |
| A46-05 | Majeur | `tools/release/fetch-models.sh:39` | 41,6 Mo de RIFE jamais proposés dans chaque paquet | 46 | |
| A46-06 | Mineur | `src/app/play/model_choice.h:47-49` | Commentaire faux sur l'origine des RIFE FP16 livrés | 46 | |
| A46-07 | Majeur | `tools/upscale/models/2x_ArtCNN_C4F32_rgb_fp16.onnx` | ArtCNN « fp16 » calcule entièrement en FP32 (TRT-07, partie modèle) | 46 | |
| A46-08 | Mineur | `data/luts/ATTRIBUTION.md:11-13` | Auteur et licence des quatre LUT livrées absents | 46 | |
| A46-09 | Mineur | `THIRD_PARTY_NOTICES.md:8-12` | Avis des données livrées faux ou incomplets (ArtCNN, RIFE) | 46 | |
| A46-10 | Mineur | `data/calibration/calibration.mkv` | 19,8 Mo d'extraits de Blu-ray commerciaux dans un dépôt public, plus livrés | 46 | |
| A46-11 | Mineur | `tools/vk/presets/NvidiaAda-rtx4070ti-1.tsv` | Qualité : sommes FP16 pour 40 couches sur 58 dans les tables Ada, sans mesure d'image | 46 | A45-01 (même cause) |
| A46-12 | Mineur | `src/render/cube_lut.cpp:325-334` | Deux des quatre LUT livrées non reconnues comme LUT d'HDR | 46 | |
| A46-13 | Mineur | `data/luts/HDR grading base ver.9.0 33 point.cube` | Qualité : LUT de base à 33 points sur une courbe raide, 2,7 codes 10 bits d'erreur d'interpolation | 46 | |
| A46-14 | Mineur | `data/gpu_tiers.toml:45` | Erreurs de tranche sur NVIDIA (Quadro RTX 6000, RTX PRO) | 46 | |
| A46-15 | Mineur | `tools/release/fetch-models.sh:86-98` | GMFSS encore listé dans `SHA256SUMS` de `models-interp-1` ; fichier facultatif annoncé absent | 46 | |
| A40-01 | Majeur | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70` | Aucun réseau livré ne tourne en CI, alors que les essais cachés passent en 4 min sous lavapipe | 40 | |
| A40-02 | Majeur | `.github/workflows/ci.yml:219-258` | Le chemin de la cible (TensorRT, CUDA, SPAN FP8, RIFE et DistilDRBA fusionnés) n'est jamais exécuté avant une release (B3) | 40 | |
| A40-03 | Majeur | `tools/bench/rife_triplets.py:96-100` | Qualité : rien ne mesure DistilDRBA contre RIFE ; le banc de qualité le refuse | 40 | |
| A40-04 | Majeur | `tests/unit/app/e2e_play_test.cpp:96-104` | Le seul bout en bout joue sans SPAN, RIFE ni DistilDRBA | 40 | |
| A40-05 | Mineur | `tests/unit/app/e2e_play_test.cpp:119-124` | Les cas e2e jouent tous les clips du dossier | 40 | |
| A40-06 | Majeur | `tests/unit/app/e2e_play_test.cpp:236-243` | Qualité : un test fige une image SDR 38 % plus sombre que la source, grade coupé (à confirmer) | 40 | |
| A40-07 | Mineur | `tests/unit/backend_tensorrt/backend_test.cpp:92` | Le test d'ingestion TensorRT fige la chroma bilinéaire | 40 | A03-01 (même cause) |
| A40-08 | Mineur | `src/gpu/guarded_backend.h:56-170` | `GuardedKernelOps` jamais testé (26,5 % couvert) | 40 | |
| A40-09 | Mineur | `CMakePresets.json:348` | TSan ne lance que les 6 tests `Inbox` à chaque poussée | 40 | |
| A40-10 | Mineur | `src/app/play/span_int8.cpp:194-202` | `span_variant_order` non testé avec les capacités d'une RTX sous Windows | 40 | |
| A40-11 | Mineur | `tests/unit/vk_backend/rife_fused_chain_test.cpp:100-106` | Variante mouvement des chaînes fusionnées jugée sur « flot borné » seulement | 40 | |
| A40-12 | Mineur | `cmake/AjpModule.cmake:77` | Les 21 cas cachés sont invisibles à `ctest -N` et à `check-skips.sh` | 40 | |
| A40-13 | Mineur | `tests/unit/app/ring_stage_test.cpp:150-153` | Tests à l'horloge réelle (26 fichiers avec `sleep_for`) | 40 | |
| A40-14 | Mineur | `.github/workflows/cloud.yml:57-58` | Tâches qui ne passent pas `check-skips.sh` ; préréglages de release non testés | 40 | |
| A40-15 | Mineur | `.github/workflows/ci.yml` | Aucune mesure de couverture en CI (76 % mesuré ici) | 40 | |
| A40-16 | Mineur | `tools/ci/check-cuda.sh:2-6` | Commentaire faux sur la construction CUDA en CI | 40 | |
| A07-13 | Mineur | `src/app/play/auto_config.h:19-22` | La « porte de qualité » qui tient DistilDRBA hors de l'Auto n'est écrite nulle part ; le commentaire renvoie à un ADR sans rapport | 07 | |
| A28-18 | Majeur | `tools/release/trt-fingerprint.sh:9-10` | Empreinte TensorRT sans le CMakeLists racine, les CMakePresets, CUTLASS ni le Dockerfile du préflight | 28 | |
| A28-19 | Mineur | `tools/ci/deps_lag.py:20-21` | Veille des dépendances limitée à la baseline vcpkg, tâche verte avec retards restants | 28 | |
| A28-20 | Mineur | `.github/workflows/release-linux.yml:64-70` | Clé de cache de la release sans version d'image | 28 | A27-06 (même cause) |
| A28-21 | Mineur | `docs/auto-libre.md:140-174` | § 3 non daté, ligne GMFSS, rangé parmi les documents à jour | 28 | |
| A28-22 | Mineur | `src/prepare/cache.cpp:117-133` | Aucun test de disque plein à l'écriture d'un moteur | 28 | |
| A08-15 | Majeur | `src/app/play/chain_setup.cpp:99` | Qualité : « Restaurer les traits » coupé, la source 1080p est réduite à 720p par l'étage alors que la chaîne la croit gardée ; refus « forme différente » en RIFE image | 08 | |
| A08-16 | Mineur | `src/app/play/chain_setup.cpp:162-171` | Le repli remet les deux crans à Équilibre même quand un seul moteur est refusé | 08 | |
| A08-17 | Mineur | `src/app/play/player.cpp:162` | Treize `path::string()` (dette) | 08 | A20-02 (même cause) |
| A09-17 | Mineur | `src/app/play/span_auto.cpp:67-71` | Clé de tentative FP8 « modèle|vidéo » sans la taille de travail | 09 | |
| A09-18 | Mineur | `src/app/play/upscale_drba.cpp:144` | DistilDRBA mesure à la taille de travail, RIFE sur la copie 720p | 09 | |
| A09-19 | Mineur | `src/app/play/upscale_drba.cpp:99` | Attente du côté b non comptée au journal | 09 | |
| A09-20 | Mineur | `src/app/play/upscale_stage.cpp:574-579` | Qualité : baisse pour la vitesse sans mesure (gain estimé à 30 ms par saut à ×3) | 09 | |
| A09-21 | Mineur | `src/app/play/drba_model.cpp:14-28` | Repli silencieux entre flot entier et demi-flot, dans les deux sens | 09 | |
| A09-22 | Mineur | `src/app/play/rife_plan.cpp:10-14` | `kPlanCutFloor` comparé au score brut ; contradiction sous une zone utile de 0,75 | 09 | |
| A09-23 | Mineur | `src/app/play/rife_plan.h:39-42` | `kPlanCutFloor` tenu sur une seule série | 09 | |
| A09-24 | Mineur | `src/app/play/drba_model.h:17-21` | Qualité : demi-flot de DistilDRBA dès 1 440 lignes sans mesure ; seuil sur la hauteur d'estimation | 09 | |
| A02-14 | Mineur | `src/gpu/cuda/backend/span_derive.cpp:27-35` | Paquet `.span8` et reçu écrits sans fsync : après une coupure, refus ordinaire et Performance bloqué en FP16 (QUA-05) | 02 | |

## Renvois

| De | Vers | Où | Constat | Statut |
|---|---|---|---|---|
| 01 | 02 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:78-89` | La queue du SPAN FP8 est enfilée après la jointure de génération (A01-02) | Traité par 02 (complément) |
| 01 | 21 | `src/gpu/graph_book.h` | Compteurs d'échecs des graphes jamais remis à zéro, jeux d'adresses jamais évincés | Traité par 21 |
| 01 | 21 | `src/gpu/lanes.h` | La table des ressources de `LaneHazards` ne perd jamais une adresse | Traité par 21 |
| 01 | 30 | `src/app/play/chain_setup.h:64-93` | Moteurs détruits avant le backend sans attendre leur travail en vol | Traité par 30 |
| 01 | 31 | `src/app/play/ring_stage.cpp:58-66` | `health()` n'est appelé que pendant l'attente d'un emplacement | Traité par 31 |
| 02 | 34 | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | FP8 éteint hors Linux : sous Windows 11, le cran Performance reste en FP16 | Traité par 34 (A34-01) |
| 02 | 01 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:325` | `write_ring` rend l'erreur sans signaler l'emplacement commencé (comme A02-08) | En attente (01 fini) |
| 02 | 07 | `src/app/play/span_int8.cpp:166` | `size_key` plafonné à 1920×1088 : la taille validée n'est pas celle du moteur au-delà | Traité par 07 |
| 02 | 07 | `src/app/play/span_int8.cpp:129-150` | Seul un refus d'une porte plus ancienne se refait | Traité par 07 |
| 02 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:75` | La zone d'activation partagée suppose que tout moteur qui y écrit se nomme (voir A02-03) | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:80-91` | `SharedActivations::reserve` synchronise la file de la voie à chaque croissance | En attente (01 fini) |
| 03 | 01 | `src/gpu/cuda/backend/tensorrt_textures.cpp:15-78` | Objets texture du warp gardés par adresse : réutilisés après une réallocation à la même adresse | En attente (01 fini) |
| 03 | 09 | `src/app/play/upscale_motion.cpp:39` | Reprise à chaud absente sous RIFE fusionné ; invalidation sur coupe et saut à vérifier | Traité par 09 |
| 03 | 09 | `src/app/play/player_engines.cpp:247-266` | Avec DistilDRBA fusionné, le warp repose sur la branche mouvement non testée (A03-03) | Traité par 09 |
| 03 | 15 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:142-143` | Position de chroma : seuls `Center` et `TopLeft` sont distingués | Réfuté par 15 |
| 03 | 40 | `tests/unit/backend_tensorrt/` | La suite CUDA ne tourne qu'à la main sur un poste NVIDIA, aucun garde-fou de CI | Traité par 40 (A40-02) |
| 05 | 27 | `src/gpu/cuda/backend/CMakeLists.txt:95` | Repli `trtexec` gardé en `RelWithDebInfo` (SEC-20) | Traité par 27 |
| 05 | 01 | `src/gpu/cuda/backend/entry.cpp:56-59` | `pick_trtexec` appelé une fois : empreinte jamais revérifiée ensuite (A05-05) | En attente (01 fini) |
| 05 | 06 | `src/app/play/engine_install.cpp:130-158` | TRT-17 : échec de `nvidia-smi` gardé pour le processus, carte inconnue rangée en « ptx » | Traité par 06 (TRT-17 en partie présent) |
| 05 | 18 | `src/app/play/crash_dump.cpp:90` | Le gestionnaire de plantage ne tue pas le `trtexec` en cours (A05-01) | Traité par 18 (réfuté sous Windows) |
| 05 | 07 | appelants de `Backend::build` | Une construction par taille de source : coût à mesurer, cache sans purge (TRT-17) | Traité par 07 |
| 05 | 36 | `src/prepare/identity.cpp:161-164` | Jeton `driver` dans la clé : reconstruction complète à chaque saut de CUDA du pilote | Traité par 36 |
| 06 | 26 | `src/app/play/engine_online.cpp` | `merge_release_index` : SHA ou version changés du paquet TensorRT (voir A06-01) | Traité par 26 |
| 06 | 28 | `tools/release/package-linux.sh:190` | Version du paquet TensorRT tirée de `ls libnvinfer.so.*` : « 11 » ou « 11.3.0 » | Traité par 28 |
| 06 | 28 | `tools/release/package-windows.ps1:101` | Origine et stabilité de `$trtVersion` | Traité par 28 |
| 06 | 25 | `src/app/play/first_run.cpp:125-131` | « Go » et « Mo » en dur dans l'interface anglaise (UX-09) | Traité par 25 |
| 06 | 34 | `src/app/play/engine_tools.cpp:19-36` | `curl.exe` de System32 : version minimale, révocation Schannel derrière une inspection TLS | Traité par 34 |
| 06 | 21 | `src/app/play/backend_library.cpp:83-85` | Module refusé refermé : nvinfer se décharge-t-il sous Windows (A06-09) ? | Traité par 21 (à confirmer) |
| 04 | 09 | `src/app/play/derived_model.cpp:19` | `stem().string()` dans le nom des modèles dérivés (comme A04-04) | Traité par 09 |
| 04 | 09 | `src/app/play/motion_model.cpp:90` | `filename().string()` et `stem().string()` sur le chemin du RIFE | Traité par 09 |
| 04 | 09 | `src/app/play/upscale_stage.cpp:198` | L'étage suppose `kSpanFactor` pour un `--model` quelconque (voir A04-06) | Traité par 09 |
| 04 | 28 | `tests/fuzz/fuzz_parsers.cpp:33-37` | Aucune cible de fuzz pour les lecteurs ONNX de `prepare/` (SEC-21) | Traité par 28 |
| 04 | 05 | `src/prepare/builder.cpp:174` | `read_model` relit et rehache tout l'ONNX avant et après chaque construction | En attente (05 fini) |
| 08 | 22 | `src/app/play/settings.cpp:453-499` | `change_between` ne classe pas en réouverture plusieurs réglages dont `prepare_chain` dépend | Traité par 22 |
| 08 | 05 | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` joint le fil de construction sur le thread de la fenêtre | En attente (05 fini) |
| 08 | 11 | `src/app/play/host_facts.cpp:378` | `on_battery()` appelé deux fois par seconde | Traité par 11 (A11-09) |
| 08 | 14 | `src/app/play/player_run.cpp:589-591` | MED-01 : la session doit distinguer vraie fin et erreur d'E/S | Traité par 14 |
| 08 | 18 | `src/app/play/player.cpp:525-534` | La reprise après « Annuler et regarder » modifie `settings` pour toute la suite | Traité par 18 |
| 08 | 23 | `src/app/play/settings_panel.cpp` | UX-08 : Échap devrait fermer la liste ouverte avant le panneau | Traité par 23 |
| 08 | 39 | `src/app/play/settings_system.cpp` | UX-13 : chemins hors du dossier personnel dans le journal envoyé | Traité par 39 (A39-03) |
| 07 | 25 | `src/app/play/engine_screen.cpp:111` | `calibrate_on_screen` rend toujours Auto, même sur « Annuler » : remplace une chaîne Personnalisée | Traité par 25 |
| 07 | 08 | `src/app/play/chain_setup.cpp:99` | Tailles de travail différentes possibles entre l'anneau et `c.work` | Traité par 08 (complément) |
| 07 | 09 | `src/app/play/chain_setup.cpp:321` | Préparation FP8 automatique faite une seule fois, à la taille de la lecture en cours (avec A07-04) | Traité par 09 (complément) |
| 07 | 18 | `src/app/play/player.cpp:411-414` | `calibration.toml` relu à chaque ouverture et remplissage du panneau | Traité par 18 |
| 07 | 34 | `src/app/play/auto_bench_run.cpp:35-44` | Essais en `--hidden` sur tous les bureaux, GNOME Wayland compris | Réfuté par 34 |
| 09 | 23 | `src/app/play/home_screen.cpp:140` | `fp8_engine` vrai pour une RTX 20 ou 30 : Performance proposé sans FP8 | Traité par 23 |
| 09 | 22 | `src/app/play/settings.h:114-117` | Critère d'Auto sur la hauteur de sortie (A09-08) | Traité par 22 |
| 09 | 08 | `src/app/play/chain_setup.cpp:420-423` | Performance remplace le RIFE fusionné par un moteur entier (A09-02) | Traité par 08 (complément) |
| 09 | 41 | `src/app/play/upscale_stage.cpp:244-271` | Statistiques du grade prises sur l'image débandée, ou pas du tout sous DistilDRBA | Réfuté par 41 (3/1000 du code, moyenne nulle) |
| 09 | 31 | `src/app/play/ring_stage.cpp:108-112` | Seul `SourceResized` rouvre la chaîne sur une image de taille nouvelle | Traité par 31 |
| 10 | 12 | `src/app/play/session.cpp:598-616` | Au niveau `Wait`, `vblank_grid()` toujours vide : effet sur `--vsync-index` et la PLL | Traité par 12 (A12-01, A12-03) |
| 10 | 08 | `src/app/play/player_run.cpp:654` | Une erreur de `show` termine `run_chain` : message et réouverture en boucle ? | Traité par 08 (complément) |
| 10 | 18 | `src/app/play/placebo_open.cpp:327` | Décodeur Vulkan Video à détruire avant l'écran | En attente (18 fini) |
| 10 | 31 | `src/app/play/placebo_ring.cpp:218` | `ready()` vrai quand `vkGetSemaphoreCounterValue` échoue | Traité par 31 |
| 10 | 34 | `src/app/play/placebo_open.cpp:384-390` | Windows : fenêtre cachée pendant l'ouverture, message d'erreur sans fenêtre ? | Réfuté par 34 |
| 11 | 08 | `src/app/play/player_run.cpp:398-400` | Relevé de l'écran fait deux fois sous Windows | Traité par 08 (complément) |
| 11 | 08 | `src/app/play/player_run.cpp:549-560` | Relecture du HDR à partir de l'événement SDL plutôt que du minuteur | Traité par 08 (complément) |
| 11 | 08 | `src/app/play/player_run.cpp:619-621` | `p.display` remplacé sans prévenir `contract_watch` | Traité par 08 (complément) |
| 11 | 24 | `src/app/play/chain_stats.cpp:184-208` | NVML relevé toutes les 500 ms même OSD caché | Traité par 24 |
| 11 | 30 | `src/app/play/player_run.cpp:679` | `GpuTelemetry::sample` appelé depuis deux threads (A11-08) | Traité par 30 |
| 11 | 34 | `src/app/play/host_facts.cpp:236-275` | Pic calculé dans deux unités différentes sous Windows et Linux | Réfuté par 34 |
| 12 | 08 | `src/app/play/player_run.cpp:622-629` | GPU-05 : plancher VRR non recalculé (VRR hors sujet pour Tatoub) | Traité par 08 (complément) |
| 12 | 08 | `src/app/play/player_run.cpp:420-438` | Plafond passé comme fréquence d'écran (A12-04) | Traité par 08 (complément) |
| 12 | 08 | `src/app/play/session.cpp:665-672` | GPU-19 : vitesse calée même en VRR armé (VRR hors sujet pour Tatoub) | Traité par 08 (complément) |
| 12 | 10 | `src/app/play/placebo_screen.cpp:461-463` | `present_us` est le retour de `swap_buffers`, pas un vsync (A12-03) | En attente (10 fini) |
| 12 | 34 | `src/app/play/loop_wait.cpp:56` | Windows : attente « précise » à 1 ms de résolution | Traité par 34 (A34-07) |
| 12 | 20 | `src/core/drawings.cpp:113` | `DrawingsTimeline` n'est appelé que par ses tests | Traité par 20 |
| 13 | 14 | `src/media/source_loop.cpp:110-123` | `recover_position` qui échoue laisse la source inactive, lecture figée | Traité par 14 |
| 13 | 14 | `src/media/source_loop.cpp:182-188` | Lecture abandonnée traitée comme la fin du fichier (origine de MED-01) | Traité par 14 |
| 13 | 17 | `src/app/play/external_subtitles.cpp:92` | `path::string()` sur des noms de sous-titres | Traité par 17 (A17-12) |
| 13 | 08 | `src/app/play/player_run.cpp:738-739` | Pistes copiées dans `Player` même pour une autre vidéo (A13-01, A13-09) | Traité par 08 (complément) |
| 13 | 30 | `src/app/play/wiring.cpp:113-124` | `ThreadStages::play/pause` rendent faux sans que personne ne rejoue | Traité par 30 |
| 14 | 13 | `src/app/play/pipeline.cpp:45-49` | `path::string()` sur les sous-titres externes | En attente (13 fini) |
| 14 | 13 | `src/app/play/pipeline.cpp:65` | `config.file.string()` dans le refus « no video track » | En attente (13 fini) |
| 14 | 13 | `src/app/play/session.cpp:286-287` | La session doit refuser d'enchaîner l'épisode sur une fin en erreur (MED-01) | En attente (13 fini) |
| 14 | 18 | `src/app/play/player.cpp:99` | `file.string()` dans le refus de `probe_once` | Traité par 18 |
| 14 | 18 | `src/app/play/pipeline.h:76` | `~Pipeline` joint un thread source bloqué sur un partage muet | En attente (18 fini) |
| 14 | 16 | `src/audio/` | Rognage audio sur la cible d'un seek refusé (MED-06) | En attente (16 fini) |
| 14 | 30 | `src/media/source_loop.cpp:303-305` | Boîte non lue pendant `wait_before_retry` | Traité par 30 |
| 15 | 14 | `src/media/source_loop.cpp:153-159` | Image décodée et convertie avant la décision de saut du seek exact | En attente (14 fini) |
| 15 | 14 | `src/media/source_loop.cpp:21-25` | `ends_before` jette les images `kNoPts` : seek exact sans fin (avec A15-05) | En attente (14 fini) |
| 15 | 08 | `src/app/play/player.cpp:95-117` | La sonde devrait refuser les formats illisibles avec un message | Traité par 08 (complément) |
| 15 | 01 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:24-41` | L'ingestion TensorRT ne lit ni 4:4:4 ni 4:2:2 en mémoire CUDA | En attente (01 fini) |
| 15 | 42 | `src/media/shot_probe.cpp:173-175` | La pré-analyse paie la conversion 4:4:4 en pleine taille | Traité par 42 |
| 16 | 23 | `src/app/play/settings_sound.cpp:27-33` | En mode « suivre le système », un seul délai pour tous les appareils | Traité par 23 |
| 16 | 08 | `src/app/play/pipeline.cpp:97-110` | Format de sortie figé, ordre `seek()` puis `AudioTrack` | Traité par 08 (complément) |
| 16 | 13 | `src/app/play/session.cpp:420-445` | Sans sortie audio, un gel de 0,4 s à chaque seek | En attente (13 fini) |
| 16 | 12 | `src/core/clock_pll.h:28-40` | PLL comptée en lectures, pas en temps | En attente (12 fini) |
| 16 | 20 | `src/core/clock.h:1-3` | Commentaire faux sur l'audio maître | Traité par 20 (renvoyé à 12) |
| 17 | 13 | `src/app/play/pipeline.cpp:164-170` | Une piste PGS empêche le repli sur un fichier externe | En attente (13 fini) |
| 17 | 13 | `src/app/play/pipeline.cpp:183-191` | Polices recopiées deux fois, attente sur le thread de la fenêtre | En attente (13 fini) |
| 17 | 13 | `src/app/play/pipeline.cpp:47` | Extension `.Srt` étiquetée « ass » | En attente (13 fini) |
| 17 | 14 | `src/media/source_loop.cpp:90-106` | Aucune pré-lecture des sous-titres au seek ni pour un délai négatif | En attente (14 fini) |
| 17 | 22 | `src/app/play/settings.cpp:420` | MED-17 : délai de sous-titres gardé d'un épisode à l'autre | Réfuté par 22 |
| 17 | 24 | `src/app/play/osd_text.cpp:70` | Seconde bibliothèque libass pour l'OSD | Traité par 24 |
| 17 | 35 | `tests/fuzz/fuzz_parsers.cpp:40-44` | `to_utf8` fuzzé sans indice : branches iconv non couvertes | Traité par 35 (en partie) |
| 19 | 28 | `tools/release/windows/ajp.iss` | Pas de `--unregister` à la désinstallation | Traité par 28 |
| 19 | 28 | `CMakePresets.json:108-121` | Aucun PDB produit ni gardé | Traité par 28 |
| 19 | 25 | `src/app/play/engine_screen.cpp:277-290` | Inscription synchrone à la fin de l'assistant | Traité par 25 |
| 19 | 24 | `src/app/play/home_screen.cpp:204-212` | Passage au premier plan en effet de bord de `dropped()` | Traité par 24 |
| 19 | 18 | `src/app/main.cpp:376-400` | Mise à jour appliquée avant `install_crash_dumps` | En attente (18 fini) |
| 18 | 07 | `src/app/play/auto_config.h:22` | DistilDRBA n'entre jamais dans Auto et la calibration ne l'essaie pas | Traité par 07 (complément) |
| 18 | 30 | `src/app/play/chain_hdr.cpp:162` | `LookLoader` détaché qui peut journaliser après `close_log` | Traité par 30 |
| 18 | 26 | `src/app/play/updater.cpp:504-506` | `last_check_unix` posé avant le téléchargement | Traité par 26 |
| 18 | 26 | `src/app/play/updater.cpp:637-644` | Purge des `.old` dans `guard_new_version` (A18-05) | Traité par 26 |
| 18 | 17 | `src/app/play/bridges.cpp:383-385` | PGS et VobSub refusés | En attente (17 fini) |
| 18 | 39 | `src/app/main.cpp:378-401` | Mises à jour absentes d'`ajp.log` | Traité par 39 (doublon d'A26-12) |
| 18 | 20 | `src/core/log.cpp:82-95` | `log_channel` déréférence un journal nul (A18-01) | En attente (20 fini) |
| 03 | 32 | `docs/auto-libre.md:157-172` | Le service d'une image (13 à 16 ms) borne la cadence, pas la passe d'interpolation : relevé à refaire par étape | Traité par 32 |
| 03 | 07 | `src/app/play/auto_config.h:22` | Calibration sans DistilDRBA, Auto calibré imposé à RIFE 4.22 lite | Traité par 07 (complément) |
| 03 | 23 | `src/app/play/settings_chain_tabs.cpp:375-383` | « Warp and SPAN » grisé hors RIFE alors qu'il décide la variante DistilDRBA | Traité par 23 |
| 03 | 09 | `src/app/play/upscale_drba.cpp:144` | DistilDRBA mesure la paire à la taille de travail, RIFE sur la copie réduite | Traité par 09 (complément) |
| 03 | 05 | `src/prepare/trtexec_command.cpp:105-106` | Formats des frontières des morceaux DistilDRBA laissés à TensorRT : reformatages possibles | En attente (05 fini) |
| 20 | 05 | `src/prepare/process_win.cpp:63-65` | `trtexec.exe` sans manifeste UTF-8 : chemin hors page ANSI abîmé | En attente (05 fini) |
| 20 | 06 | `src/app/play/engine_install.cpp:293` | A06-02 ne tient que pour `curl.exe` et `tar.exe` | En attente (06 fini) |
| 20 | 34 | A02-12, A03-06, A04-04, A07-03, A08-09, A09-10, A10-10, A13-11, A17-12 | Supposent une exception de `path::string()` que le manifeste UTF-8 empêche sous Windows 11 : à requalifier | Traité par 34 (tous en dette) |
| 20 | 18 | `src/app/play/crash_dump.cpp` | Le gestionnaire de plantage ne vide pas le journal (A20-04) | En attente (18 fini) |
| 20 | 18 | `src/app/play/pumped_task.h:58` | Sonde détachée qui peut journaliser après `close_log()` | En attente (18 fini) |
| 20 | 26 | `src/app/main.cpp:526` | Refus d'écriture d'`update.toml` compté comme un plantage (A20-01) | Traité par 26 |
| 20 | 31 | `src/app/play/ring_stage.cpp:112` | Tout refus GPU devient `ThreadFailed`, `DeviceLost` perdu (A20-09) | Traité par 31 |
| 20 | 24 | `src/app/play/visible_error.h:44-45` | 58 codes sur 71 affichés par leur nom anglais | Traité par 24 |
| 20 | 21 | `tools/release/package-linux.sh:124` | `RUNPATH` devancé par `LD_LIBRARY_PATH` : TensorRT système avant celui du paquet | Traité par 27 (A27-02) |
| 20 | 02 | `src/app/play/span_derive.cpp:27-35` | Écriture « atomique » sans fsync | Traité par 02 (complément) |
| 20 | 45 | `src/gpu/vulkan/compute/library.cpp:48` | Copie de `DynamicLibrary` ; `profile_dispatches` jamais posé | Traité par 45 (A45-08, A45-09) |
| 20 | 12 | `src/core/clock.h:1-3` | Commentaire faux sur l'audio maître | En attente (12 fini) |
| 20 | 14 | `src/media/av_util.cpp:48-51` | `duration_to_us` peut rendre `kNoPts` sur une base de temps nulle | En attente (14 fini) |
| 22 | 24 | `src/app/play/home_screen.cpp:244-250` | `save_settings` synchrone sur le thread de l'interface pendant un réglage à l'accueil | Traité par 24 |
| 22 | 07 | `src/app/play/auto_config.cpp:202` | Qualité : l'Auto calibré passe SPAN en FP8 malgré `auto_config.cpp:166-169` | Traité par 07 (complément) |
| 22 | 07 | `src/app/play/auto_config.cpp:203-205` | L'Auto calibré écrase `auto_free` et le modèle RIFE à chaque vidéo | Traité par 07 (complément) |
| 22 | 23 | `src/app/play/settings_sound.cpp:72` | Bornes des curseurs de délai différentes des réglages et des touches | Traité par 23 (A23-11) |
| 22 | 23 | `src/app/play/settings_engines.cpp:81-84` | Dossier des moteurs recopié à chaque frappe | En attente (23 fini) |
| 22 | 09 | `src/app/play/chain_setup.cpp:359-361` | Qualité : estimation RIFE sur copie 720p au-dessus de 720 lignes | Traité par 09 (complément) |
| 22 | 18 | `src/app/play/options.cpp:298` | `--rife-model` n'entre pas dans `settings.rife_model` | En attente (18 fini) |
| 21 | 26 | `src/app/play/engine_downloads.cpp:130` | Module mis à jour seulement sur `AbiMismatch` : un correctif hors frontière n'arrive jamais | Traité par 26 |
| 21 | 09 | `src/app/play/upscale_drba.cpp:66-67` | Garder `filled01` et `filled12` d'une paire à la suivante | Traité par 09 (complément) |
| 21 | 09 | `src/app/play/upscale_drba.cpp:99` | `scores_take` bloque le thread GPU à chaque paire (A21-06) | Traité par 09 (complément) |
| 21 | 32 | `docs/auto-libre.md` § 3 | Décomposer les ≈ 11 ms de GPU par image hors réseau, qui ôtent à DistilDRBA son avance | Traité par 32 |
| 21 | 10 | `audit/mesures-auto-4070ti.md` | DistilDRBA ×6 : 962 images manquées sans paire en retard, la présentation plafonne | En attente (10 fini) |
| 21 | 08 | `src/app/play/chain_setup.cpp:361-363` | Qualité : mouvement estimé à 720p, à remesurer surtout pour DistilDRBA | Traité par 08 (complément) |
| 21 | 08 | `src/app/play/engine_screen.cpp:343` | `path().string()` dans le journal | Traité par 08 (complément) |
| 21 | 45 | `src/gpu/vulkan/backend/entry_points.h:1-6` | Commentaire qui nomme MIGraphX et OpenVINO | Traité par 45 (A45-10) |
| 21 | 40 | `tests/unit/gpu/backend_test.cpp` | `GuardedKernelOps` jamais testé ; `IngestSpec::chroma` non vérifié | Traité par 40 (A40-08, A40-07) |
| 23 | 22 | `src/app/play/settings.cpp:383-433` | Cause d'A23-01 (`restore_lines`, `gpu`) | Traité par 22 (A22-01) |
| 23 | 22 | `src/app/play/settings.h:304-307` | Commentaire du seuil HDR posé au mauvais endroit | En attente (22 fini) |
| 23 | 26 | `src/app/play/updater.cpp:490-497` | `update_managed_reason` écrit un fichier d'essai à chaque appel | Traité par 26 |
| 23 | 07 | `src/app/play/auto_config.cpp:134-147` | Qualité : écran 1080p, source 1080p réduite à 540p avant SPAN | Traité par 07 (complément) |
| 23 | 08 | `src/app/play/chain.cpp:183-185` | Une notification par image pendant le glissement du délai des sous-titres | Traité par 08 (complément) |
| 25 | 24 | `src/app/play/home_screen.cpp:307-316` | « Run the wizard again » depuis l'accueil n'ouvre que l'écran des moteurs | En attente (24 fini) |
| 25 | 07 | `src/app/play/machine_calibration.cpp:115-135` | `best_candidate` ignore `priority` : la fluidité passe toujours avant la taille d'image | Traité par 07 (complément) |
| 25 | 07 | `src/app/play/machine_calibration.cpp:240-252` | `calibration.toml` ne note pas le moteur mesuré | Traité par 07 (complément) |
| 25 | 07 | `src/app/play/machine_calibration.cpp:288-290` | Écriture refusée de `chaine-N.toml` : toute la calibration s'arrête | Traité par 07 (complément) |
| 25 | 08 | `src/app/play/player.cpp:559` | Après `handle_engine_change`, chaîne rouverte avec les réglages d'avant | Traité par 08 (complément) |
| 25 | 19 | `src/app/play/default_player.cpp:334-336` | `update-desktop-database` lancé deux fois de suite | En attente (19 fini) |
| 25 | 43 | `src/app/play/settings_panel.cpp:283` | Infobulle « Finish » sans bouton Terminer | Traité par 43 (A43-05) |
| 26 | 28 | `src/app/CMakeLists.txt:69` | Trois dépôts de release par défaut différents | Traité par 28 |
| 26 | 28 | `.github/workflows/release-guard.yml` | Étiquette jamais comparée à `PROJECT_VERSION` (cause d'A26-04) | Traité par 28 |
| 26 | 28 | `.github/workflows/release-linux.yml:150` | `--latest` posé avant le manifeste signé (SEC-05) | Traité par 28 |
| 26 | 36 | `src/app/main.cpp:398-401` | Retour arrière sur des fichiers écrits par la version plus récente | Traité par 36 |
| 26 | 35 | `src/core/update_manifest.cpp:93-96` | Version du manifeste non assainie (SEC-10 démontré) | Traité par 35 |
| 26 | 06 | `src/app/play/engine_downloads.cpp:130-166` | A06-03 et A26-01 : même correction, à faire ensemble | En attente (06 fini) |
| 24 | 23 | `src/app/play/settings_system.cpp:497-509` | « Signaler un problème » promet un journal sans nom d'utilisateur mais copie la fin brute | En attente (23 fini) |
| 24 | 38 | `src/app/play/settings_system.cpp:507-509` | Même promesse non tenue | Traité par 38 (A38-01) |
| 24 | 08 | `src/app/play/player.cpp:163` | `filename().string()` dans un message | Traité par 08 (complément) |
| 24 | 05 | `src/app/play/player_engines.cpp:135` | `stem().string()` et `path.string()` | En attente (05 fini) |
| 24 | 03 | `src/app/play/chain_setup.cpp:396-397` | Repli de DistilDRBA fusionné vers l'entier écrit au journal seulement | En attente (03 fini) |
| 24 | 12 | `src/app/play/session.cpp:506` | Présentation pour l'interface seule hors échéance | En attente (12 fini) |
| 24 | 29 | `src/app/play/player_engines.cpp:54-57` | Boucles d'attente qui jettent les fichiers d'un second lancement | Traité par 29 |
| 24 | 17 | `src/subs/subtitle_frame.h` | Le rectangle d'un sous-titre ne dit pas s'il est positionné | En attente (17 fini) |
| 24 | 27 | `cmake/Dependencies.cmake:123` | `imgui_impl_sdl3.cpp` compilé sans être utilisé | En attente (27 fini) |
| 49 | 07 | `src/app/play/machine_calibration.h:37-47` | Calibration : ajouter le b8f64 FP16, les hauteurs au-dessus de la moitié de l'écran et le RIFE du rang | Traité par 07 (complément) |
| 49 | 10 | `src/app/play/placebo_screen.cpp:130` | Paramètres par défaut de libplacebo, pas d'anti-ringing | En attente (10 fini) |
| 49 | 23 | `src/app/play/auto_config.cpp:167-169` | Trois commentaires contredits par le code (FP8, RGB10A2, LUT) | En attente (23 fini) |
| 49 | 32 | `tests/unit/` | Aucun test de qualité de bout en bout contre l'original | Traité par 32 |
| 49 | 34 | `src/app/play/span_auto.cpp:22-25` | Windows et Linux ne jouent pas le même SPAN en Auto calibré | Traité par 34 (A34-01) |
| 49 | 41 | `src/render/cube_lut.cpp:289-314` | Écart de la LUT composée ; statistiques du grade sur l'image débandée | Traité par 41 (A41-01 ; débandage réfuté) |
| 29 | 13 | `src/app/play/player.cpp:454` | A13-09 réfuté : `Player` recréé à chaque fichier | Traité par 29 |
| 29 | 31 | `src/app/play/chain_setup.cpp:497` | Refus de RIFE sans repli (A29-01) à compter dans la grille | Traité par 31 |
| 29 | 31 | `src/app/play/player_engines.cpp:270-272` | `Cancelled` de DistilDRBA fusionné lance la construction de l'entier | Traité par 31 |
| 29 | 33 | aucun fichier | La veille (A29-02) à inclure dans le banc de soirée | Traité par 33 |
| 29 | 34 | `src/app/play/default_player.cpp:203` | Plusieurs épisodes ouverts : un processus par fichier | Traité par 34 (A34-04) |
| 29 | 36 | `src/app/play/player.cpp:574-581` | `save_settings` de l'épisode suivant écrit les raccourcis de séance avant l'ouverture | Traité par 36 |
| 31 | 40 | `tests/unit/app/ring_stage_test.cpp:243` | Test qui fige `ThreadFailed` ; aucun test d'injection de `DeviceLost` | Traité par 40 (test écrit, échoue : A31-01) |
| 31 | 39 | `src/app/play/engine_screen.cpp:342-344` | Raison du refus de TensorRT au journal seulement | Traité par 39 (A39-02) |
| 31 | 08 | `src/app/play/chain_setup.cpp:160-171` | Repli au cran Équilibre : baisse de qualité sans avis | Traité par 08 (complément) |
| 31 | 34 | `src/app/play/chain_setup.cpp:415-427` | Sous Windows, Performance retombe toujours en FP16 entier | Réfuté par 34 (A31-06 touche les deux systèmes) |
| 31 | 32 | `src/gpu/cuda/backend/tensorrt_export.cpp` | « CUDA Sysmem Fallback » sous Windows : lenteur sans refus | En attente (32 fini) |
| 30 | 01 | `src/gpu/cuda/backend/tensorrt_backend.cpp:201-204` | `drain()` à placer dans `~TensorRtBackend` (A30-01) | En attente (01 fini) |
| 30 | 32 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:140-150` | Aucun `cudaSetDeviceFlags` : attente active sur un cœur | En attente (32 fini) |
| 30 | 32 | `src/app/play/upscale_stage.h:500` | Comparer l'attente de `scores_take` pour DistilDRBA et RIFE | En attente (32 fini) |
| 30 | 42 | `src/media/shot_probe.cpp:108-110` | `~ShotProbe` joint sur le thread de la fenêtre | Traité par 42 (A42-12) |
| 30 | 26 | `src/app/play/updater.cpp:466-474` | Téléchargement sans `.part` ni verrou | En attente (26 fini) |
| 30 | 41 | `src/render/grade.cpp:277-301` | Thread des LUT sans `try` | Traité par 41 (A41-09) |
| 30 | 09 | `src/app/play/upscale_stage.cpp:574-579` | Qualité : après une image sautée, la paire suivante n'est pas interpolée | Traité par 09 (complément) |
| 30 | 40 | `.github/workflows/ci.yml:100` | TSan ne couvre ni `LookLoader`, ni `ChainStats`, ni le module TensorRT | Traité par 40 (A40-09) |
| 27 | 28 | `tools/release/trt-fingerprint.sh:17-19` | Empreinte TensorRT qui ignore le `CMakeLists.txt` racine, les presets et CUTLASS | Traité par 28 (complément) |
| 27 | 28 | `.github/workflows/deps-watch.yml` | Baseline vcpkg sans les versions Linux ; libplacebo en retard non bloqué | Traité par 28 (complément) |
| 27 | 28 | `.github/workflows/release-linux.yml:67-70` | `restore-keys` qui restaure un préfixe d'une autre image | Traité par 28 (complément) |
| 27 | 05 | `src/gpu/cuda/backend/CMakeLists.txt:83` | Même RUNPATH pour `trtexec` | En attente (05 fini) |
| 27 | 49 | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | Qualité : noyaux FP8 de SPAN activés par défaut sous Linux (9,1 contre 14,25 ms pour 50-52 dB) | En attente (49 fini) |
| 28 | 26 | `src/app/play/updater.cpp:410-415` | Un 404 du manifeste se lit « à jour » sans journal | En attente (26 fini) |
| 28 | 09 | `src/app/play/drba_model.cpp:14-28` | Demi-flot DistilDRBA manquant : repli silencieux sur le flot entier | Traité par 09 (complément) |
| 28 | 40 | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70-72` | Essais du fusionné contre le réseau entier cachés | Traité par 40 (A40-01) |
| 32 | 03 | `src/app/play/upscale_drba.cpp:179-256` | L'entrée de DistilDRBA, pas seulement son nombre de passes | Traité par 03 (complément, A03-14) |
| 32 | 07 | `src/app/play/auto_config.cpp:167-169` | Commentaire et code contradictoires sur le FP8 dans l'Auto (TRT-08) | Traité par 07 (complément) |
| 32 | 10 | `src/render/shaders/anime-hdr-look.hook:100` | Commentaire périmé ; `pl_render_default_params` | En attente (10 fini) |
| 32 | 12 | `src/core/free_rate.cpp:27` | En ×2, le mode Rapide n'a aucune mesure de service | En attente (12 fini) |
| 32 | 21 | `src/gpu/gpu_profile.h` | `gpu_profile` sans équivalent Vulkan pour l'écran | En attente (21 fini) |
| 32 | 28 | `docs/auto-libre.md` § 3 | Tableaux de temps à dater et marquer historiques | Traité par 28 (complément) |
| 32 | 49 | `docs/adr/0008-anneau-rgb10a2.md` | Porte de qualité de l'anneau 10 bits jamais passée | Traité par 49 (A49-04) |
| 34 | 07 | `src/app/play/auto_config.cpp:192-200` | L'Auto calibré force RIFE 4.22 lite, n'essaie ni DistilDRBA ni RIFE 4.25 | Traité par 07 (complément) |
| 34 | 07 | `src/app/play/machine_calibration.cpp:279-330` | Premier essai FP8 perdu sans CUTLASS | Traité par 07 (complément) |
| 34 | 31 | `src/app/play/drba_fused_choice.cpp:9` | A31-06 touche les deux systèmes | En attente (31 fini) |
| 34 | 06 | `src/app/play/engine_tools.cpp:76-87` | `curl.exe` sans `--ssl-revoke-best-effort` | En attente (06 fini) |
| 34 | 15 | `src/media/cuda_driver.cpp:105-109` | `libcuda.so.1` chargé par son nom | En attente (15 fini) |
| 34 | 19 | `src/app/play/single_instance.cpp:114-120` | Canal nommé à nom prévisible | En attente (19 fini) |
| 34 | 11 | `src/app/play/system_peak.cpp:150-151` | Maximum de toutes les sorties quand le moniteur manque | En attente (11 fini) |
| 34 | 28 | `.github/workflows/release-windows.yml:16-19` | PDB et CUTLASS absents de la release Windows : à trancher | Traité par 28 (complément) |
| 34 | 40 | `tests/unit/` | `span_variant_order` jamais testé avec les capacités d'une RTX sous Windows | Traité par 40 (A40-10) |
| 33 | 08 | `src/app/play/engine_build.cpp` | Un `TensorRtBackend` créé puis détruit par moteur préparé | Traité par 08 (complément) |
| 33 | 28 | `src/prepare/cache.cpp` | Disque plein par le cache : l'écriture d'un moteur doit refuser proprement | Traité par 28 (complément) |
| 33 | 36 | `src/prepare/cache.cpp` | Purge des moteurs orphelins avec le changement de pilote | Traité par 36 |
| 33 | 31 | `src/app/play/player_run.cpp:510` | Au réveil, perte du contexte CUDA et saut d'horloge ensemble | En attente (31 fini) |
| 35 | 33 | `src/prepare/builder.cpp:139-145` | Un moteur par forme, sans purge | Traité par 33 (A33-04) |
| 35 | 17 | `src/subs/subtitle_track.cpp:146-149` | Aucun plafond d'événements d'une piste ASS intégrée | En attente (17 fini) |
| 35 | 28 | `.github/dependabot.yml` | Seules les actions GitHub suivies | Traité par 28 (complément) |
| 36 | 07 | `src/app/play/machine_calibration.cpp:240-252` | Un seul jeton d'identité de la mesure (avec A36-02 et le renvoi 25→07) | Traité par 07 (complément) |
| 36 | 02 | `src/app/play/span_int8.cpp:114-125` | Reçu FP8/INT8 sans le SHA du paquet dérivé | Traité par 02 (complément) |
| 36 | 26 | `src/app/play/updater.cpp:670-679` | `updates/previous` jamais retiré après confirmation | En attente (26 fini) |
| 37 | 15 | `src/app/play/chain_setup.cpp:43` | RTX 20 : AV1 en décodage logiciel puis copie hôte sur le thread de soumission | En attente (15 fini) |
| 37 | 02 | `src/gpu/cuda/backend/span_fp8_kernels.cuh:241` | Tag `Sm89` de CUTLASS compilé aussi pour sm_120 | Traité par 02 (complément) |
| 37 | 01 | `src/gpu/cuda/backend/rife_fused_kernels.cu:141-147` | Sur sm_120, colle de RIFE fusionné à deux blocs par SM | En attente (01 fini) |
| 39 | 43 | `src/app/play/gpu_telemetry.cpp:58` | « chaleur » en dur, sans `tr()` | Traité par 43 (doublon d'A11-10) |
| 39 | 18 | `src/app/main.cpp:108-111` | `fail()` n'écrit que sur stderr | En attente (18 fini) |
| 39 | 03 | `src/gpu/cuda/backend/tensorrt_backend.cpp:205-224` | Lignes « profil GPU » écrites seulement à la fermeture | En attente (03 fini) |
| 38 | 43 | `src/app/play/i18n_fr_*.cpp` | Environ 180 traductions sans texte anglais correspondant | Traité par 43 (A43-02, 192) |
| 38 | 39 | `src/core/log_redact.cpp:52-73` | Nettoyage du journal à étendre aux racines de montage | En attente (39 fini) |
| 38 | 23 | `src/app/play/auto_config.cpp:167-169` | Commentaire FP8 des préréglages contredit | En attente (23 fini) |
| 38 | 28 | `tools/release/notes/v0.10.3.md` | Notes publiées muettes sur l'absence de TensorRT Linux | Traité par 28 (complément) |
| 38 | 07 | `src/app/play/auto_config.cpp:203-204` | L'Auto calibré écrase le modèle RIFE de la tranche | Traité par 07 (complément) |
| 42 | 09 | `src/app/play/rife_plan.cpp:10-14` | `kPlanCutFloor` comparé au score brut de RIFE | Traité par 09 (complément) |
| 42 | 09 | `src/app/play/rife_plan.h:39-42` | `kPlanCutFloor` tenu sur une seule série | Traité par 09 (complément) |
| 42 | 41 | `src/app/play/chain_hdr.cpp:252` | GPU-07 : métadonnées HDR10 fixes, grade seul plafonné | Réfuté par 41 (pic plafonné lu, GPU-07 reste vrai) |
| 42 | 41 | `src/kernels/spec/frame_stats.h:5-6` | Statistiques de lumière du grade sur 9 216 points | Réfuté par 41 pour le grade (reste à 42 pour les flashs) |
| 42 | 07 | `src/app/play/auto_bench_run.cpp:37` | Pré-analyse active pendant les essais de calibration | Traité par 07 (complément) |
| 41 | 10 | `src/app/play/placebo_screen.cpp:155` | Commentaire « netteté sur MAIN » faux | En attente (10 fini) |
| 41 | 32 | `src/render/shaders/anime-hdr-anime/3-anime.hook:60-130` | Trois ou quatre passes HDR plein écran, valeurs constantes recalculées par pixel | En attente (32 fini) |
| 41 | 38 | `docs/hdr-anime.md:63-67` | Trois promesses non tenues : nuits lisibles, blanc à 203 nits, vrai noir en SDR | En attente (38 fini) |
| 41 | 42 | `src/app/play/chain_hdr.cpp:321` | Seuils de `next_flashing` non audités | En attente (42 fini) |
| 41 | 43 | `src/app/play/placebo_screen.cpp:64-66` | `output_description` en français hors `tr()` | Traité par 43 (confirmé) |
| 41 | 46 | `data/luts/ATTRIBUTION.md` | Auteur et licence des quatre LUT livrées | Traité par 46 (A46-08) |
| 43 | 11 | `src/app/play/gpu_telemetry.cpp:73-76` | « Go » et virgule décimale en dur dans la ligne VRAM de l'OSD anglais (UX-09) | En attente (11 fini) |
| 43 | 24 | `src/app/play/home_screen.cpp:178` | « EDID illisible : OLED inconnu » en dur dans Vidéo > Affichage | En attente (24 fini) |
| 43 | 41 | `src/app/play/placebo_screen.cpp:64-66` | Ligne de sortie en français en dur | En attente (41 fini) |
| 43 | 41 | `src/app/play/chain_hdr.cpp:367` | GPU-14 toujours présent (« Vrai noir : mesure impossible » en dur) | En attente (41 fini) |
| 43 | 25 | `src/app/play/first_run.cpp:29-31` | La `note` de l'assistant n'applique pas `french_spaces` | En attente (25 fini) |
| 43 | 25 | `src/app/play/first_run.cpp:445` | Deux textes pour l'installation de la mise à jour | En attente (25 fini) |
| 43 | 08 | `src/app/play/chain.cpp:113-131` | UX-20 : rien ne reformate l'affichage quand la langue change en direct | Traité par 08 (complément) |
| 43 | 22 | `src/app/play/settings.cpp:130-138` | `label(MotionEngine)` nomme encore GMFSS, retiré | En attente (22 fini) |
| 43 | 23 | `src/app/play/settings.cpp:493` | Séparateur « ; » en dur dans le bandeau anglais « To apply » | En attente (23 fini) |
| 43 | 07 | `src/app/play/player.cpp:486-487` | « New graphics card » renvoie à un bouton du mode avancé, sous un autre nom | Traité par 07 (complément) |
| 45 | 21 | `src/gpu/kernel_ops.h:49-53` | `IngestSpec::rife_conversion` posé par personne : option morte du contrat | En attente (21 fini) |
| 45 | 21 | `src/gpu/kernel_ops.h:42-56` | Rééchantillonnage de l'image décodée différent sous TensorRT et Vulkan, rien dans le contrat ne l'impose | En attente (21 fini) |
| 45 | 08 | `src/app/play/chain_setup.h:89-93` | Ordre de destruction moteurs Vulkan puis backend au passage vers TensorRT | Traité par 08 (complément) |
| 45 | 03 | `src/kernels/glsl/rife_fused_glue.comp:121-128` | Grille de base FP16 dans RIFE v1 (comme A03-18), sur les deux moteurs | En attente (03 fini) |
| 44 | 46 | voie fusionnée DistilDRBA | Vérifier la grille en F32 sur la voie fusionnée, comme A44-01 | Traité par 46 (A46-04 : grille en FP16) |
| 44 | 49 | qualité DistilDRBA F16 | DistilDRBA F16 contre FP32 : 51 à 52 dB, écart max 13 à 19/255 (pour A49-14) | En attente (49 fini) |
| 44 | 45 | shaders Vulkan | Arrondi f32 vers f16 (RTE ou RTZ), sous-normaux F16, accumulation des coopmat | En attente (45 fini) |
| 46 | 20 | `src/core/model_index.cpp:356-410` | `models.lock` lu par aucun code du lecteur : code mort | En attente (20 fini) |
| 46 | 45 | `src/gpu/vulkan/backend/net_gate.h:24-25` | Porte qui admet les sommes FP16 à 2 % sur du bruit, pour toutes les cartes | En attente (45 fini) |
| 46 | 09 | `src/app/play/drba_model.h:17-21` | Flot de DistilDRBA à l'échelle 0,5 dès 1 440 lignes, sans mesure | Traité par 09 (complément) |
| 46 | 03 | `tools/drba/export_distil_drba.py:46` | DistilDRBA v1 jamais comparé à v2 lite et à RIFE | En attente (03 fini) |
| 46 | 28 | `THIRD_PARTY_NOTICES.md:49` | Moteurs retirés encore listés dans l'avis livré | Traité par 28 (complément) |
| 46 | 18 | `tools/grade/compare_shots.sh:18` | Chemin de la vidéo de calibration absente du paquet | En attente (18 fini) |
| 40 | 10 | `src/app/play/placebo_snapshot.cpp:16-40` | Sortie SDR ou instantané ×0,62, grade coupé ; contrat d'écran HDR au journal d'une sortie SDR | En attente (10 fini) |
| 40 | 41 | `tests/unit/app/e2e_play_test.cpp:241-243` | Image SDR assombrie, à croiser avec le grade SDR | En attente (41 fini) |
| 40 | 22 | `src/app/play/options.cpp` | `--data-dir` n'isole pas les moteurs (dossier `ajp` doublé) | En attente (22 fini) |
| 40 | 13 | `tests/unit/app/e2e_play_test.cpp:221` | Avec `--seconds`, l'épisode suivant s'enchaîne quand même | En attente (13 fini) |
| 40 | 46 | `tools/bench/rife_triplets.py:100` | Le banc de qualité refuse DistilDRBA | En attente (46 fini) |
| 28 | 39 | `src/prepare/builder.cpp:233` | Disque plein soupçonné au journal seulement, pas nommé à l'utilisateur | En attente (39 fini) |
| 09 | 07 | `src/app/play/span_auto.cpp:67-71` | A09-17 se corrige au même endroit qu'A07-04 | Note pour la correction |
| 09 | 21 | `src/gpu/kernel_ops.h:188-196` | La correction proposée pour A21-06 est une baisse de qualité à mesurer avant de la prendre | Note pour la correction |
| 09 | 32 | `src/app/play/upscale_stage.cpp:391-402` | Attente de `scores_take` non mesurable au journal tant qu'A09-19 reste | Note pour la correction |
| 09 | 42 | `src/app/play/rife_plan.cpp:10-14` | Après A42-01, rejuger le plancher sur le score rapporté et plusieurs séries | Note pour la correction |
| 09 | 46 | `src/app/play/drba_model.cpp:14-24` | Paquet sans demi-flot non signalé (A09-21) | Note pour la correction |
