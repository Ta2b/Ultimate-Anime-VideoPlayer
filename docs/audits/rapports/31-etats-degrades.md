# Rapport de l'audit 31. États dégradés et reprise sur erreur

Branche : `claude/project-thread-k6e9qu`. Commit audité : `401bfa7`. Date : `2026-11-09`.

## Verdict

Le lecteur sait dire une panne d'E/S ou un appareil audio retiré, mais une panne du GPU finit presque toujours en
« error ThreadFailed » sans reprise, un moteur refusé à l'ouverture empêche toute lecture, et plusieurs replis
baissent la qualité ou la vitesse (TensorRT vers Vulkan, cran Qualité vers FP16, « Annuler et regarder »,
DistilDRBA hors fusionné) sans un mot à l'écran.

## Cadrage retenu

Pas de questions à Tatoub (consigne du coordinateur) : tout est audité au maximum. Retenu de la consigne :
cible NVIDIA sous TensorRT, Windows 11 et CachyOS ; RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; ce
qui empêche DistilDRBA d'être bien plus rapide que RIFE est cherché en priorité ; le VRR est hors sujet ; toute
baisse de qualité d'image faite pour la vitesse est un défaut, sauf mesure qui la justifie. Point imposé :
`ring_stage.cpp:112` (tout refus GPU devient `ThreadFailed`).

Le fil tourne sans carte NVIDIA et sans les dépendances construites : aucun test n'a pu être lancé. Chaque défaut
majeur décrit le test qui échoue.

### Matrice des pannes

Lecture : ce que fait le code, ce que voit l'utilisateur, ce qui reste sur le disque. Les cases renvoient aux
défauts ci-dessous ou à ceux d'autres audits.

| Panne | Lecture en cours (TensorRT) | Ouverture d'une vidéo | Accueil et installation | Disque |
|---|---|---|---|---|
| Perte du GPU (TDR, Xid, eGPU) | refus `DeviceLost` changé en `ThreadFailed` (A31-01), retour à l'accueil sans reprise (A31-02) ; côté écran, image figée (A10-01, A31-13) | `DeviceLost` à la création des voies : « the graphics card stopped responding » au second essai seulement (A31-02) | même contexte mort gardé (A31-02) | rien |
| Erreur collante non reconnue (999, 46) | `VendorCallFailed`, « error … (details in the log) » (A01-01) | idem | idem | rien |
| GPU bloqué sans être perdu | image figée, son qui continue, aucun délai (A31-14, A10-04) | — | — | l'Auto peut retenir un palier bas (A31-10) |
| Mémoire vidéo pleine | `cudaErrorMemoryAllocation` dit « VendorCallFailed » (A31-08), puis `ThreadFailed` (A31-01) | moteur refusé : aucune lecture (A31-04) ; cran Qualité replié en FP16 sans mot (A08-05, renvoi vers 08) | — | rien |
| Mémoire vive pleine | `bad_alloc` d'un thread : `ThreadFailed` ; sur le thread de la fenêtre : plantage | idem | idem | — |
| Disque plein ou lecture seule | `play.toml` perdu (A22-02) | construction refusée, message générique, refaite à chaque ouverture (A31-11, A31-04) | installation vérifiée (seul endroit) | moteurs et dérivés écrits par renommage atomique : pas de fichier à moitié écrit |
| Antivirus qui tient un fichier | — | — | `update.toml` refusé compté comme un plantage (QUA-07, A20-01) | renommage sans nouvel essai (A05-04) |
| Réseau coupé (SMB) | panne de plus de 1,55 s prise pour la fin : épisode suivant (MED-01, A14-01) ; partage muet qui fige (A14-02) | — | `engines-nvidia.lock` sur le thread de la fenêtre (A18-02) | `.part` repris (A06) |
| Fichier abîmé ou tronqué | `DecodeFailed` recouvrable, puis fin (A14-07) | « the file cannot be read » | — | — |
| Taille de la source qui change | réouverture (`SourceResized`), sauf image suivante plus petite avec DistilDRBA : arrêt (A31-09) | — | — | — |
| Pilote trop ancien, DLL absente, TensorRT abîmé | — | Vulkan sans message à l'écran (A31-03) ; `AbiMismatch` réinstallé à chaque lancement (TRT-03) | — | ~1 Go rehaché (A06-10) |
| Écran débranché | contrat figé sur l'écran d'ouverture (A11-01) | — | — | — |
| Sortie audio débranchée | reprise sur la sortie par défaut, sans défaut : refus à chaque bloc (A16-01) | — | — | — |
| Veille et hibernation | rien n'est géré (A31-12) | — | — | — |
| Batterie | Auto plafonné à 60 Hz sans un mot (A31-10) | — | — | palier bas retenu |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A31-01 | Majeur | `src/app/play/ring_stage.cpp:106-113` | Tout refus de l'étage GPU (`DeviceLost`, `GpuMemoryExhausted`, `ShapeInvalid`, `VendorCallFailed`…) est réécrit en `ThreadFailed` pour être fatal (`wiring.cpp:73-76` ne rend fatal que ce code). `Session::on_refusal` garde ce code (`session.cpp:283-284`) et `visible_error` ne connaît pas `ThreadFailed` (`visible_error.h:44-45`) : la phrase « the graphics card stopped responding: restart the player » (`visible_error.h:40-41`) ne sort jamais en lecture. Le test `ring_stage_test.cpp:243` fige ce comportement. | TDR sous Windows pendant un épisode en RIFE 4.25 : `cudaErrorLaunchTimeout` (collant) → `DeviceLost` dans `produce` → `ThreadFailed` → l'accueil affiche « "ep01.mkv" stopped: error ThreadFailed (details in the log) ». Même chose pour une mémoire vidéo pleine. | Garder le code d'origine et porter la fatalité à part : `report(refusal)` avec un drapeau `fatal` (le champ `StageRefusal::fatal` existe déjà), `ThreadReport` à deux arguments ou `Refusal::fatal`. `visible_error` reçoit alors `DeviceLost`. | Lu ; test qui échoue : dans `ring_stage_test.cpp:225-246`, attendre `refusals.front().code == RefusalCode::DeviceLost` et un refus fatal pour la session (`session_refusal_test.cpp` : un `StageRefusal{fatal}` de code `DeviceLost` donne `failure()->code == DeviceLost`). |
| A31-02 | Majeur | `src/app/play/player.cpp:589-602` | Après un arrêt sur perte du GPU, aucune reprise : retour à l'accueil (ou sortie pour un lancement scripté), la bibliothèque du backend et son contexte CUDA mort sont gardés (`player.cpp:601`, `BackendLibrary::open` rend la bibliothèque tenue, `backend_library.cpp:58-61`). Le mécanisme de relance existe pourtant (`paths.relaunch`, `player.cpp:441-444`) et la position est connue (`remember()`). | TDR ou Xid 79 : la vidéo s'arrête ; l'utilisateur la relance depuis l'accueil, la création des voies échoue de nouveau (contexte collant), et seulement là le message « restart the player » paraît. Sous Windows le TDR réinitialise aussi l'appareil Vulkan de l'écran : l'accueil peut rester figé (A10-01). | Sur `DeviceLost` (CUDA ou Vulkan, écran compris) : écrire la position de reprise, poser `*paths.relaunch = true` avec la vidéo et la position, relancer une fois ; au second échec dans la minute, message clair et accueil. | Lu ; à confirmer : sous Windows, `dxcap -forcetdr` (Graphics Tools) pendant une lecture TensorRT ; sous CachyOS, un noyau CUDA en boucle infinie lancé par un outil de test sur le même GPU. |
| A31-03 | Majeur | `src/app/play/engine_screen.cpp:341-344` | TensorRT refusé au lancement (pilote trop ancien, `nvinfer` absent ou abîmé, DLL manquante, CUDA qui refuse) : Vulkan prend sa place « sans bruit ». La raison va au journal ; la seule trace est une note dans Général > Moteurs (`fallback_note`, `engine_screen.cpp:408-414`, `settings_engines.cpp:26`), qui dit « does not open on this GPU » même quand c'est le pilote. `TensorRtBackend::create` range toute erreur de `cudaGetDeviceCount`, `cudaErrorInsufficientDriver` compris, en `LibraryUnavailable` (`tensorrt_backend.cpp:144-146`). | RTX 4070 sous Windows 11, pilote resté en 53x après une mise à jour du paquet qui demande un CUDA plus récent : chaque épisode tourne sous Vulkan, plus lent ; l'Auto calibré pour TensorRT (A07-05) perd des images et baisse le facteur. L'utilisateur ne sait pas pourquoi. | Avis à l'écran au premier lancement de la lecture (`controls().defer`) : « TensorRT unavailable: NVIDIA driver {} too old (needs {}). Vulkan is used, slower. » ; `RefusalCode::DriverTooOld` (ou un `key`) pour `cudaErrorInsufficientDriver` et `cudaErrorCompatNotSupportedOnDevice`. | Lu ; à confirmer : renommer `nvinfer_10.dll` dans le dossier du moteur et lancer une vidéo : aucun message à l'écran. |
| A31-04 | Majeur | `src/app/play/chain_setup.cpp:355` | Un moteur refusé à l'ouverture (construction en échec ou hors délai, mémoire vidéo pleine au chargement, désérialisation) empêche toute lecture : SPAN (`:355`), RIFE mouvement (`:497`), RIFE image (`:516`) rendent le refus, la vidéo revient à l'accueil. Seul DistilDRBA se replie (sur RIFE, `:401-406`). Aucun souvenir de l'échec : la même construction recommence à chaque ouverture (avec TRT-05, `builder.h:35`). | Portable RTX 4060 : construction de RIFE 4.25 à une nouvelle taille au-delà de 20 min → « stopped: error BuilderTimeout (details in the log) », et la vidéo suivante relance les 20 min. Jeu en arrière-plan qui tient la mémoire vidéo : SPAN refusé, rien ne se lit. | Refus d'un moteur d'interpolation : lecture avec SPAN seul et avis ; refus de SPAN : lecture sans agrandissement et avis, ou Vulkan pour cet étage. Retenir (clé du moteur, code) pour la séance, avec « Réessayer » dans le panneau. | Lu ; test qui échoue : `chain_setup_test.cpp` avec un backend factice dont `build` rend `BuilderTimeout` pour RIFE : `prepare_chain` doit rendre une chaîne sans interpolation, pas un refus. |
| A31-05 | Majeur | `src/app/play/player.cpp:523-533` | « Annuler et regarder » coupe à la fois l'agrandissement et l'interpolation, quel que soit le moteur annulé, même quand SPAN est déjà prêt (en cache ou chargé l'instant d'avant). Baisse de qualité sans gain : SPAN ne coûte rien à garder s'il est prêt. | Nouvelle taille de source : SPAN sort du cache en 1 s, puis RIFE 4.25 se construit ; l'utilisateur annule : l'épisode se lit en 1080p brut, sans SPAN, jusqu'à sa fin. | Le refus `kWatchNowKey` porte le moteur annulé ; ne couper que l'étage dont la construction a été annulée (l'interpolation dans le cas courant), garder SPAN. | Lu. |
| A31-06 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | Un cran Qualité ou Performance qui retombe en FP16 (graphe `_fp32` ou `_fp8` absent, moteur sans FP8) garde le refus du fusionné : `drba_fused_declined` et `rife_fused_declined` (`rife_fused_choice.cpp:11`) lisent `settings.motion_precision`, que le repli « Équilibre à la place » (`player_engines.cpp:219-230`, `chain_setup.cpp:415-427`) ne réécrit pas. Résultat : le même modèle FP16 qu'en Équilibre, mais en moteur entier, plus lent. Le panneau affiche « Équilibre » (`settings_chain_tabs.cpp:33-37`). Aucun graphe FP8 de RIFE n'est publié (`publier-modeles-fp8.sh:12`) : RIFE en Performance tourne toujours ainsi. | Windows 11 (FP8 éteint, renvoi 02→34), DistilDRBA en Performance : « Équilibre à la place », DistilDRBA entier FP16 au lieu du fusionné ; l'avance de DistilDRBA sur RIFE disparaît, l'Auto baisse le facteur. Même chose en Qualité sans `distil_drba_v2_lite_fp32.onnx`. | Calculer le cran effectif (celui du graphe trouvé) avant de juger le fusionné, et passer ce cran à `*_fused_declined`. | Lu ; test qui échoue : `drba_fused_choice` appelé avec Performance et un moteur sans FP8 doit accepter le fusionné une fois le cran effectif ramené à Équilibre (test unitaire sur une fonction `effective_motion_tier`). |
| A31-07 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | DistilDRBA en Performance avec le graphe facultatif `distil_drba_v2_lite_fp8.onnx` présent : moteur entier FP8, que le script de publication mesure « plus lent que le DistilDRBA fusionné » (`publier-modeles-fp8.sh:10-12`) ; le fusionné FP16 est refusé hors Équilibre. Le cran Performance donne donc moins de qualité (FP8) et moins de vitesse. | RTX 40 sous CachyOS, graphe FP8 posé à côté du modèle comme le dit la doc : Performance plus lent qu'Équilibre. | Hors fusionné, ne prendre le FP8 de DistilDRBA que s'il bat le fusionné FP16 à la mesure ; sinon retirer le cran Performance de DistilDRBA du panneau. | Lu ; à confirmer : `--bench-out` Performance contre Équilibre avec DistilDRBA ×4 sur une RTX 40. |
| A31-08 | Mineur | `src/gpu/cuda/backend/cuda_status.h:38-39` | `cudaErrorMemoryAllocation` devient `VendorCallFailed`, pas `GpuMemoryExhausted` : la phrase « not enough GPU memory » ne sort pas sous TensorRT hors des quelques chemins qui la codent eux-mêmes (`trt_engine.cpp:81`, SPAN FP8). | Mémoire vidéo prise par un jeu : `SharedActivations::reserve` refusé → « error VendorCallFailed (details in the log) ». | `cuda_refusal` : `cudaErrorMemoryAllocation` → `GpuMemoryExhausted`. | Lu ; test : `cuda_status_test.cpp`, `cuda_refusal(cudaErrorMemoryAllocation, …).code == GpuMemoryExhausted`. |
| A31-09 | Mineur | `src/app/play/upscale_drba.cpp:136` | L'image suivante est ingérée avant le contrôle de taille de `outputs` (`upscale_stage.cpp:276-279`) : plus petite que le tenseur, `ingest` rend `ShapeInvalid` (`tensorrt_kernels.cpp:107-112`), fatal (`ThreadFailed`) au lieu de rouvrir la chaîne ; plus grande, elle est réduite en Area dans le tenseur sans refus, et une image intermédiaire mêle deux cadrages avant la réouverture. Même cause qu'A09-04. | Enregistrement TS qui passe de 1920×1080 à 1440×1080 (publicité, générique) en DistilDRBA : la lecture s'arrête au lieu de se rouvrir. | Contrôler la taille de `next()` avant de l'ingérer et rendre `SourceResized` (traité comme une fin de paire). | Lu ; renvoi 09→31 confirmé. |
| A31-10 | Mineur | `src/app/play/host_facts.h:48-51` | Sur batterie, l'Auto est plafonné à 60 Hz (`player_run.cpp:414`), sans avis à l'écran ni réglage, et ce palier tenu 6 s est retenu dans `auto_memory.toml` (`auto_watch.cpp:33`), clé sans l'alimentation : la lecture suivante, branchée, repart de là. Même mémoire pour un palier bas pris pendant un état passager (GPU bloqué, autre application). Baisse de fluidité non mesurée. | Portable débranché pendant un épisode en 144 Hz : passe de ×6 à ×2,5 sans un mot ; rebranché le lendemain, l'épisode repart bas. | Avis une fois (« On battery: smoothing limited to 60 fps ») et un réglage ; ne pas retenir un palier atteint sous plafond de batterie ou pendant des images manquées massives. | Lu ; à confirmer : vitesse de remontée depuis un palier retenu. |
| A31-11 | Mineur | `src/core/file_sync.cpp:75-91` | Disque plein : seule l'installation des moteurs vérifie la place (`engine_install.cpp:229`). Construction (moteurs de centaines de Mo, un par taille), modèles dérivés, découpes fusionnées, cache de shaders : `ENOSPC` devient `FileUnwritable`, `CacheIo` ou `BuilderFailed`, montrés « a file could not be read or written » ou « error BuilderFailed ». | SSD plein : chaque ouverture relance une construction de plusieurs minutes qui échoue à l'écriture, message générique. | Mapper `no_space_on_device` en `InsufficientSpace` dans `write_atomic` et dans le jugement de `trtexec` ; vérifier la place avant une construction. | Lu. |
| A31-12 | Mineur | `src/app/play/player_run.cpp:510` | Veille et hibernation ne sont pas gérées : pas de pause à l'entrée, pas de contrôle de l'appareil au réveil. Sous CachyOS sans `NVreg_PreserveVideoMemoryAllocations`, le contexte CUDA ne survit pas à la veille : la reprise tombe dans A31-01 et A31-02. | Portable mis en veille en pleine lecture, réveillé : erreur `ThreadFailed` ou image figée. | Détecter un saut de l'horloge hôte (plus de 2 s entre deux tours) : pause, `health()`, relance propre (A31-02) si l'appareil est perdu. | Lu ; à confirmer : `systemctl suspend` pendant une lecture TensorRT. |
| A31-13 | Mineur | `src/app/play/placebo_ring.cpp:218` | `ready()` rend vrai quand `vkGetSemaphoreCounterValue` échoue (perte de l'appareil, mémoire) : main acquiert l'image et libplacebo attend une valeur qui ne viendra pas ; la perte n'est pas remontée, elle finit en image figée (A10-01). | Perte de l'appareil Vulkan pendant la lecture. | Rendre faux et poser un refus `DeviceLost` pour la session. | Lu ; renvoi 10→31 confirmé. |
| A31-14 | Mineur | `src/gpu/cuda/backend/tensorrt_export.cpp:262-263` | GPU bloqué sans être perdu : `health()` prend `cudaErrorNotReady` pour une bonne santé, l'étage attend un slot sans délai (`ring_stage.cpp:60-64`), main attend `ready()` sans délai (`session.cpp:604`). Image figée, son qui continue, aucun message ; l'Auto compte des images manquées et baisse. | Linux sans délai de garde sur un GPU sans écran branché, ou pilote qui se bloque : lecture figée indéfiniment. | Délai de garde : un slot attendu plus de 2 s alors que la même file reste `NotReady` sans progrès → refus `DeviceHung` montré, puis A31-02. | Lu ; famille de GPU-17 (A10-04 côté présentation). |
| A31-15 | Majeur | `src/app/play/player_engines.cpp:259-260` | Un refus `Cancelled` (« Annuler et regarder », clé `kWatchNowKey`) pendant la construction de DistilDRBA fusionné est pris pour un refus du fusionné : `fused->declined` est posé et la construction du DistilDRBA entier démarre aussitôt. Annulé de nouveau, il devient « DistilDRBA unavailable: using RIFE » (`chain_setup.cpp:401-406`) et RIFE se construit à son tour. Même schéma pour RIFE fusionné : `try_fused` rend `Declined` sur tout refus (`chain_setup.cpp:466-470`), puis RIFE entier se construit (`:488-497`). Proche d'A08-01 (replis du cran Équilibre), autre endroit. | DistilDRBA ×4 à une nouvelle taille : l'utilisateur clique « Annuler et regarder » et voit une seconde barre « Preparing engine DistilDRBA », puis une troisième pour RIFE : jusqu'à trois annulations pour regarder. | Dans `prepare_distil_drba` et `try_fused`, rendre tel quel un refus de clé `kWatchNowKey` (et `Cancelled`), sans repli ; `chain_setup.cpp:401` ne replie pas sur RIFE pour ce refus. | Lu ; test : `chain_setup_test.cpp` avec un backend factice dont `build` rend `Cancelled` de clé `kWatchNowKey` pour le fusionné : `prepare_chain` doit rendre ce refus sans appeler une seconde construction. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-06 | Toujours présent | `src/app/play/placebo_screen.cpp:355-356` : `pl_swapchain_start_frame` en échec rendu comme une fenêtre cachée (A10-01). A31-13 y mène aussi. |
| MED-01 | Toujours présent | `src/media/source_loop.cpp:183-186` : une lecture recouvrable en erreur devient la fin du flux ; `player_run.cpp:718-725` enchaîne l'épisode suivant ; la notice dit « Playback goes on after an error » (`player_run.cpp:589-591`). A14-01, A14-08. |
| QUA-07 | Toujours présent | `src/app/main.cpp:526` : `std::ignore` sur `save_update_state` ; `file_sync.cpp:87-91` renomme sans nouvel essai. A20-01. |
| GPU-17 | Toujours présent | Côté présentation A10-04 ; côté étage A31-14. |
| MED-06 | Toujours présent | `src/media/source_loop.cpp:110-116` : `recover_position` qui échoue laisse `idle_`, lecture figée ; `session.cpp:189-227` ne relance le seek que sur boîte pleine. A13-08, A14-04. |
| MED-08 | Toujours présent | `src/audio/sdl_sink.cpp:150-165` : `reopen_default` seul chemin de reprise, sans défaut pas d'autre essai. A16-01, A16-02 (revu en surface). |
| TRT-03 | Toujours présent | `src/app/play/engine_downloads.cpp:130-161` : réinstallation à chaque lancement sur `AbiMismatch`. A06-03. |
| TRT-05 | Toujours présent | `src/prepare/builder.h:35` : 20 min ; aucun souvenir de l'échec, la construction recommence à chaque ouverture (A31-04). |
| Renvoi 20→31 (`ring_stage.cpp:112`) | Confirmé | A31-01. |
| Renvoi 01→31 (`ring_stage.cpp:58-66`) | Confirmé, portée réduite | Une perte se voit aussi au `produce` suivant ; le manque réel est le GPU bloqué (A31-14). |
| Renvoi 09→31 (`ring_stage.cpp:108-112`) | Confirmé | A31-09. |
| Renvoi 10→31 (`placebo_ring.cpp:218`) | Confirmé | A31-13. |
| Coordinateur : refus de RIFE sans repli (`chain_setup.cpp:497`) | Confirmé | A31-04 (avec `:516` pour RIFE image et `:355` pour SPAN). |
| Coordinateur : `Cancelled` de DistilDRBA fusionné (`player_engines.cpp:270-272`) | Confirmé | A31-15 ; la ligne est aujourd'hui `:259-260`, et RIFE fusionné fait de même. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 40 | `tests/unit/app/ring_stage_test.cpp:243` | Le test fige `ThreadFailed` pour un appareil perdu ; aucun test d'injection de `DeviceLost` de bout en bout (étage, session, message, relance). |
| 39 | `src/app/play/engine_screen.cpp:342-344` | La raison du refus de TensorRT (`passed_over`) n'existe qu'au journal ; « Signaler un problème » devrait la porter. |
| 08 | `src/app/play/chain_setup.cpp:160-171` | Le repli au cran Équilibre (A08-05) baisse aussi la qualité (Qualité FP32 → FP16) sans avis à l'écran, et sur un refus passager (mémoire vidéo). |
| 34 | `src/app/play/chain_setup.cpp:415-427` | Sous Windows 11 (FP8 éteint), Performance retombe toujours en FP16 entier : voir A31-06. |
| 32 | `src/gpu/cuda/backend/tensorrt_export.cpp` | Sous Windows, la politique « CUDA Sysmem Fallback » du pilote peut placer les moteurs en mémoire système quand la mémoire vidéo est pleine : lenteur sans refus. À mesurer. |

## Points positifs

- Les écritures du projet (`write_atomic`, découpes RIFE et DistilDRBA, modèles dérivés) passent par un temporaire,
  un fsync et un renommage : un disque plein ou une coupure ne laisse pas de fichier à moitié écrit.
- Le cache des moteurs garde une empreinte SHA-256 par entrée (`cache.cpp:48-57`) : un moteur abîmé se reconstruit.
- Un refus fatal a son emplacement réservé dans la boîte de main (`wiring.cpp:79-82`) : il arrête la lecture même
  derrière 64 refus.
- DistilDRBA indisponible se replie sur RIFE avec un avis à l'écran (`chain_setup.cpp:401-406`), et le Rapide
  revient au Stable avec un avis (`player_run.cpp:579-586`).
- `health()` voit une perte de contexte en 0,2 s quand l'étage attend un slot (`ring_stage_test.cpp:225-246`).
- Un changement de taille de la source rouvre la chaîne au même endroit, borné à 4 par minute
  (`player_run.cpp:701-712`).

## Questions ouvertes

- Relance automatique sur perte du GPU (A31-02) : une seule tentative, à la même position, en pause ou en lecture ?
  Option retenue pour le rapport : une tentative, en lecture, avis à l'écran.
- Lecture sans l'étage refusé (A31-04) : Tatoub préfère-t-il un repli Vulkan pour cet étage seul, ou la lecture
  sans lui avec un avis ? Option retenue : sans lui, avec avis, et « Réessayer ».
- Plafond de 60 Hz sur batterie (A31-10) : décision P4.1 à confirmer face à la règle de qualité.
