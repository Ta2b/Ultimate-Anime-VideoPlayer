# Priorités de correction

Écrit par le coordinateur à la fin des audits (9 novembre 2026), à partir de `SYNTHESE.md`. Les 661 défauts y sont tous rangés une fois, par chantier. **L'ordre des chantiers est l'ordre de prise en charge.** Dans un chantier, l'ordre est : bloquants, majeurs, mineurs de qualité d'image, autres mineurs ; un doublon se corrige avec le défaut qu'il cite.

Règle pour tout chantier : une baisse de qualité d'image faite pour la vitesse est un défaut, sauf justification mesurée ; la signaler avec son gain. Chaque correction de qualité donne sa mesure avant et après (banc du chantier C03).

## Modèle Claude conseillé

| Modèle | Coût relatif (API, entrée / sortie par million de jetons) | Pour quoi |
|---|---|---|
| Fable 5.2 | 8 $ / 40 $ | Le plus capable. Numérique et qualité d'image (FP16 contre FP32, grilles, sommes, LUT, courbes HDR, mesures PSNR et ΔE), horloges et cadence, durées de vie entre threads et files CUDA. Effort high ; xhigh seulement où un gain se voit. Prévoir des tours longs. |
| Opus 5.5 | 4 $ / 20 $ | Par défaut pour le C++ du lecteur : CI et releases, résilience, moteurs TensorRT, média, performance. Effort high (son défaut est medium). |
| Sonnet 5.5 | 2 $ / 10 $ | Interface, menu, assistant, textes, rapports de diagnostic : du code bien délimité, vérifiable à l'écran. |
| Haiku 5.5 | 0,10 $ / 0,50 $ | Commentaires faux, pages de documentation, traductions : texte seul, sans logique. |

En pratique : un fil par chantier (ou par groupe de défauts liés dans un gros chantier), sur le modèle indiqué. Une relecture par Fable 5.2 de chaque correction des chantiers C04 à C06 avant fusion, parce qu'une erreur de numérique ne se voit ni à la compilation ni dans les tests actuels.

## Ordre des chantiers

| Ordre | Chantier | Défauts | Bloquants et majeurs | Modèle |
|---|---|---|---|---|
| 1 | [C01. Releases et mises à jour sûres](#c01) | 93 | 22 | Opus 5.5 (effort high) |
| 2 | [C02. Plantages et pertes de données](#c02) | 12 | 12 | Opus 5.5 (effort high) ; Fable 5.2 pour A08-02 et A30-01 (durées de vie entre threads et files CUDA) |
| 3 | [C03. Banc de qualité et tests du chemin de la cible](#c03) | 16 | 8 | Fable 5.2 pour la conception des mesures (références FP32, PSNR, ΔE_ITP, seuils) ; Opus 5.5 pour la CI et les scripts |
| 4 | [C04. Qualité : SPAN et mode Auto](#c04) | 65 | 21 | Fable 5.2 (effort high) |
| 5 | [C05. Qualité : interpolation (RIFE 4.22 lite, RIFE 4.25, DistilDRBA)](#c05) | 76 | 12 | Fable 5.2 (effort high ; xhigh pour la numérique FP16 des grilles et des sommes) |
| 6 | [C06. Qualité : grade HDR et plans](#c06) | 47 | 6 | Fable 5.2 (effort high) |
| 7 | [C07. Résilience : GPU, étages refusés, veille, périphériques](#c07) | 41 | 17 | Opus 5.5 (effort high) |
| 8 | [C08. Cadence, horloges et présentation](#c08) | 63 | 8 | Fable 5.2 pour les horloges (A12-*) ; Opus 5.5 pour le reste |
| 9 | [C09. Moteurs NVIDIA : TensorRT, CUDA, FP8, ONNX](#c09) | 65 | 0 | Opus 5.5 (effort high) |
| 10 | [C10. Média : session, sources, décodage, audio, sous-titres](#c10) | 57 | 2 | Opus 5.5 ; Sonnet 5.5 pour les mineurs isolés |
| 11 | [C11. Performance et ressources](#c11) | 34 | 8 | Opus 5.5 |
| 12 | [C12. Interface, réglages et assistant](#c12) | 34 | 7 | Sonnet 5.5 (effort high) |
| 13 | [C13. Diagnostic et signalement](#c13) | 17 | 5 | Sonnet 5.5 |
| 14 | [C14. Dette et documentation](#c14) | 41 | 0 | Haiku 5.5 pour les textes et commentaires ; Sonnet 5.5 pour retirer le code mort |

<a id="c01"></a>

## C01. Releases et mises à jour sûres

Modèle : Opus 5.5 (effort high).

Une correction n'arrive chez personne tant que le canal Stable se croit à jour (A28-02) : ce chantier passe avant tout le reste. Décisions 1 à 5 et 13 de Tatoub : mise à jour en simple notification, canal Préversions retiré, clé Ed25519 en secret de CI avec clé de secours hors ligne, un paquet TensorRT Windows et Linux à chaque release, libass amont plus le correctif AnimeJaNai.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A28-01 | Bloquant | `.github/workflows/release-linux.yml:96-102` | Release Linux sans paquet TensorRT publiée sur un simple avertissement : 0.10.3 et 0.10.1 Linux parties sans TensorRT (démontré) |  |
| 2 | A06-01 | Majeur | `src/app/play/engine_downloads.cpp:123` | Une version TensorRT changée dans l'index fait perdre le moteur choisi sans un mot (Vulkan à la place), l'ancienne version n'est jamais purgée |  |
| 3 | A19-01 | Majeur | `src/app/play/data_dirs.cpp:77-93` | Migration coupée : `engines/` tronqué pris pour complet (démontré) |  |
| 4 | A20-01 | Majeur | `src/core/file_sync.cpp:81-91` | `write_atomic` sans nouvel essai sous Windows, refus jeté à `main.cpp:526` : la garde peut revenir à l'ancienne version |  |
| 5 | A21-01 | Majeur | `cmake/BoundaryHash.cmake:7-9` | L'empreinte d'ABI ne couvre ni `kernels/spec` ni `FrameStats` (démontré) |  |
| 6 | A26-01 | Majeur | `src/app/play/engine_downloads.cpp:130` | Module TensorRT remplacé seulement sur `AbiMismatch` : les correctifs de noyaux, qualité comprise, n'arrivent jamais |  |
| 7 | A26-02 | Majeur | `src/app/play/engine_online.cpp:84-107` | `engines-nvidia.lock` non signé : contourne la signature Ed25519 pour le module et `trtexec` |  |
| 8 | A26-03 | Majeur | `src/app/main.cpp:376-386` | Aucune garde contre une version qui ne démarre pas du tout |  |
| 9 | A26-04 | Majeur | `src/app/CMakeLists.txt:70` | `PROJECT_VERSION` sans suffixe de préversion : garde désactivée, canal Préversions figé |  |
| 10 | A28-02 | Majeur | `tools/release/sign-manifest.sh` | Aucun manifeste signé publié depuis la 0.5.1 : tout le canal Stable se croit à jour (démontré) |  |
| 11 | A28-03 | Majeur | `.github/workflows/release-linux.yml:146-150` | `--latest` posé par chaque plateforme avant le manifeste |  |
| 12 | A28-04 | Majeur | `.github/workflows/release-guard.yml:24-39` | Étiquette jamais comparée à `PROJECT_VERSION` |  |
| 13 | A28-05 | Majeur | `src/app/CMakeLists.txt:69` | Trois dépôts de release par défaut différents |  |
| 14 | A28-06 | Majeur | `tools/release/fetch-models.sh:172-190` | `models-interp-1` non épinglé remplace le DistilDRBA épinglé (démontré, même graphe aujourd'hui) |  |
| 15 | A28-07 | Majeur | `tests/unit/prepare/CMakeLists.txt:13-20` | Tests du chemin livré non lancés avant la publication |  |
| 16 | A28-08 | Majeur | `.github/workflows/` (16 `actions/checkout`) | Jetons gardés par `checkout` dans les jobs de release |  |
| 17 | A28-18 | Majeur | `tools/release/trt-fingerprint.sh:9-10` | Empreinte TensorRT sans le CMakeLists racine, les CMakePresets, CUTLASS ni le Dockerfile du préflight |  |
| 18 | A35-01 | Majeur | `tools/deps/linux-from-source.sh:35` | FFmpeg, libass (fork), dav1d, FreeType et HarfBuzz figés sans veille des failles ; A28-02 bloque tout correctif |  |
| 19 | A36-03 | Majeur | `src/app/play/engine_install.cpp:539-571` | Paquet TensorRT installé sans fsync et jugé sur sa présence : Vulkan pour toujours après une coupure (à confirmer) |  |
| 20 | A46-01 | Majeur | `tools/release/fetch-models.sh:80-98` | Le DistilDRBA épinglé est écrasé dans les paquets par une autre version, celle de `models-interp-1` |  |
| 21 | A46-02 | Majeur | `.github/workflows/export-interpolation-models.yml:34-35` | Export de DistilDRBA non reproductible : torch, onnx et onnxruntime non épinglés |  |
| 22 | A46-05 | Majeur | `tools/release/fetch-models.sh:39` | 41,6 Mo de RIFE jamais proposés dans chaque paquet |  |
| 23 | A06-02 | Mineur | `src/app/play/engine_install.cpp:293` | Windows : chemins passés à curl et tar en page ANSI, l'installation échoue avec un compte accentué | Requalifié en mineur par 06 (complément) : manifeste UTF-8, A20-02 (même cause) |
| 24 | A06-03 | Mineur | `src/app/play/engine_downloads.cpp:130-166` | Module refusé pour `AbiMismatch` réinstallé depuis le même paquet à chaque lancement (TRT-03) |  |
| 25 | A06-04 | Mineur | `src/app/play/engine_install.cpp:271` | `.part` nommé par le fichier et non par son SHA, `.part` orphelins jamais purgés (TRT-10) |  |
| 26 | A06-05 | Mineur | `src/app/play/engine_install.cpp:338-346` | Décompression sans plafond (SEC-14, démontré) |  |
| 27 | A06-06 | Mineur | `src/app/play/engine_install.cpp:213-224` | Estimation de place qui ignore les liens durs : mise à jour refusée à tort (TRT-11) |  |
| 28 | A06-07 | Mineur | `src/app/play/engine_install.cpp:290-306` | Réponse 200 à une requête `--range` collée au `.part` |  |
| 29 | A06-08 | Mineur | `src/app/play/engine_install.cpp:585-592` | Arrêt entre deux renommages : les deux seules copies complètes effacées au démarrage suivant |  |
| 30 | A06-09 | Mineur | `src/app/play/engine_install.cpp:555` | Windows : dossier non renommable tant qu'une DLL TensorRT reste chargée |  |
| 31 | A06-10 | Mineur | `src/app/play/engine_install.cpp:243-249` | `reuse_vendor_file` rehache 1 Go à chaque installation, exception possible sur un dossier illisible |  |
| 32 | A06-12 | Mineur | `tests/unit/app/engine_install_test.cpp` | Aucun test des mises à jour, reprises, plafonds et du SHA-256 portable |  |
| 33 | A06-13 | Mineur | `src/app/play/engine_tools.cpp:76-87` | `curl.exe` sans `--ssl-revoke-best-effort` sous Windows : échec si la révocation est injoignable (aussi `updater.cpp:88-91`) |  |
| 34 | A19-02 | Mineur | `src/app/play/single_instance.cpp:52` | Accusé « ok » envoyé avant que le fichier soit remis à SDL |  |
| 35 | A19-03 | Mineur | `src/app/play/settings.h:246` | La langue du système ne règle pas la langue de l'interface |  |
| 36 | A19-04 | Mineur | `src/app/play/default_player.cpp:381-391` | `xdg-mime query` (jusqu'à 10 s) sur le thread principal |  |
| 37 | A19-05 | Mineur | `src/app/play/default_player.cpp:379` | Lecteur par défaut : état inconnu sous Windows, seul le MKV interrogé sous Linux |  |
| 38 | A19-06 | Mineur | `src/app/play/default_player.cpp:221-244` | Inscription jamais refaite si l'exécutable bouge |  |
| 39 | A19-07 | Mineur | `src/app/play/single_instance.cpp:148-150` | Windows : erreur de `ConnectNamedPipe` en boucle sans `DisconnectNamedPipe` |  |
| 40 | A19-08 | Mineur | `src/app/play/home_screen.cpp:204-212` | Sous Wayland, la fenêtre ne passe pas au premier plan sur un fichier transmis |  |
| 41 | A19-09 | Mineur | `src/app/play/data_dirs.cpp:124` | `directory_iterator` qui peut lever malgré le code d'erreur |  |
| 42 | A19-10 | Mineur | `src/app/play/single_instance.cpp:385-386` | Socket trop long : instance unique éteinte sans journal |  |
| 43 | A19-11 | Mineur | `src/app/play/default_player.cpp:421` | `SetConsoleOutputCP` change la console du terminal parent |  |
| 44 | A19-12 | Mineur | `src/app/play/system_language.cpp:81-82` | Fichiers de KDE lus hors d'une session KDE, avant SDL |  |
| 45 | A19-13 | Mineur | `tests/unit/app/single_instance_test.cpp:20` | Sans test : course de deux lancements, migration coupée |  |
| 46 | A20-02 | Mineur | `src/core/utf8_path.h:3-7` | Prémisse fausse : le manifeste UTF-8 empêche l'exception de `path::string()` dans le lecteur ; le risque est dans `trtexec`, `curl`, `tar` |  |
| 47 | A20-03 | Mineur | `src/core/log_redact.cpp:25-29` | Nettoyage du journal par sous-chaîne : casse, noms 8.3, `/run/media/<user>` (démontré) |  |
| 48 | A20-04 | Mineur | `src/core/log.cpp:104-112` | Journal asynchrone sans `flush_on` : la dernière seconde est perdue au plantage |  |
| 49 | A20-05 | Mineur | `src/core/thread.h:103-127` | `start_task` par `std::async` : pool Windows sans nom, futur bloquant non documenté |  |
| 50 | A20-06 | Mineur | `src/core/thread_role.cpp:42-56` | Rôles de thread sans héritage ni priorité d'E/S sous Windows |  |
| 51 | A20-07 | Mineur | `src/core/dynamic_library.cpp:28-36` | Aucun `SetDefaultDllDirectories` : DLL secondaires prises dans le `PATH` |  |
| 52 | A20-08 | Mineur | `src/core/file_sync.cpp:63-79` | Lien symbolique remplacé, temporaires jamais purgés (démontré) |  |
| 53 | A20-09 | Mineur | `src/core/types.h:19-20` | Refus fatal codé par `ThreadFailed` ; 58 codes sur 71 sans phrase |  |
| 54 | A20-10 | Mineur | `src/core/debug_flags.cpp` | Environ 1 250 lignes de `core` sans appelant |  |
| 55 | A20-11 | Mineur | `src/core/debug_flags.h:27-33` | Aucun outil livré pour mesurer DistilDRBA contre RIFE (NVTX, noms de threads, bascules mortes) |  |
| 56 | A20-12 | Mineur | `src/core/av_error.h:12-29` | FFmpeg inclus dans `core`, textes français dans le détail des refus |  |
| 57 | A20-13 | Mineur | `tests/unit/core/` | Aucun test des chemins d'échec du socle |  |
| 58 | A20-14 | Mineur | `src/core/model_index.h:1-6` | En-tête faux : modèles téléchargés et vérifiés, alors qu'ils sont livrés avec le lecteur |  |
| 59 | A26-05 | Mineur | `src/app/play/settings_system.cpp:200-209` | « Install now » applique `staged_version` sans la comparer |  |
| 60 | A26-06 | Mineur | `src/app/play/updater.cpp:466-474` | Archives préparées jamais purgées |  |
| 61 | A26-07 | Mineur | `src/app/play/updater.cpp:504-506` | `last_check_unix` posé avant la vérification et le téléchargement |  |
| 62 | A26-10 | Mineur | `src/app/play/player.cpp:197-200` | SEC-09 : `update.toml` réécrit en entier par quatre écrivains |  |
| 63 | A26-11 | Mineur | `src/app/play/updater.cpp:267-271` | Le canal Préversions ne voit pas une stable plus récente |  |
| 64 | A26-12 | Mineur | `src/app/main.cpp:367-389` | Application de la mise à jour sans fenêtre ni journal |  |
| 65 | A26-13 | Mineur | `src/app/play/updater.cpp:646-651` | Retour arrière de deux versions sur une installation mise à jour à la main |  |
| 66 | A28-09 | Mineur | `tools/release/package-linux.sh:190` | Version TensorRT de `engines.lock` réduite au numéro majeur (démontré) |  |
| 67 | A28-10 | Mineur | `tools/release/preflight/nvidia-package.sh:24-32` | Deux chemins publient le même paquet `trt-<empreinte>` |  |
| 68 | A28-12 | Mineur | `CMakePresets.json:458` | Filtres des tests de livraison trop larges |  |
| 69 | A28-13 | Mineur | `.github/workflows/fuzz.yml:63-64` | UBSan sans `-fno-sanitize-recover` dans le fuzz de nuit |  |
| 70 | A28-14 | Mineur | `.github/workflows/export-interpolation-models.yml:58-61` | `models-interp-1` créée sans `--latest=false` |  |
| 71 | A28-15 | Mineur | `.github/workflows/release-windows.yml` | Release Windows sans groupe de concurrence |  |
| 72 | A28-16 | Mineur | `tools/release/appimage.sh:145` | `appimagetool` téléchargé sans SHA-256 puis exécuté |  |
| 73 | A28-17 | Mineur | `.github/workflows/release-linux.yml:10-11` | En-têtes qui promettent des chemins interdits par la garde |  |
| 74 | A28-19 | Mineur | `tools/ci/deps_lag.py:20-21` | Veille des dépendances limitée à la baseline vcpkg, tâche verte avec retards restants |  |
| 75 | A28-22 | Mineur | `src/prepare/cache.cpp:117-133` | Aucun test de disque plein à l'écriture d'un moteur |  |
| 76 | A35-02 | Mineur | `src/media/demuxer.cpp:190-199` | Chapitres dédoublonnés en O(n²) : 7 s par ouverture à 100 000 chapitres (démontré) |  |
| 77 | A35-03 | Mineur | `src/core/update_manifest.cpp:21-28` | Préversion multiligne affichée telle quelle (SEC-10) |  |
| 78 | A35-04 | Mineur | `src/core/model_index.cpp:33-40` | `safe_name` accepte les noms réservés de Windows (démontré) |  |
| 79 | A35-05 | Mineur | `src/app/play/engine_tools.cpp:79-88` | Téléchargements sans `--proto =https` ni `--proto-redir` |  |
| 80 | A35-06 | Mineur | `src/media/demuxer.cpp:116-117` | Aucune borne haute sur la taille annoncée par le fichier |  |
| 81 | A35-07 | Mineur | `src/app/play/updater.cpp:557-564` | `staged_file` non confiné à `updates/` |  |
| 82 | A35-08 | Mineur | `tests/fuzz/CMakeLists.txt:18-20` | Lecteurs d'octets extérieurs non fuzzés |  |
| 83 | A35-09 | Mineur | `src/app/play/look_catalog.cpp:42-44` | Nom d'un look gardé contre `/` et `\` seulement |  |
| 84 | A36-04 | Mineur | `src/app/play/derived_model.cpp:18-22` | Nom d'un modèle dérivé sans la version du dériveur |  |
| 85 | A36-05 | Mineur | `src/app/play/updater.cpp:273-293` | `update.toml` réécrit avec les seuls champs connus |  |
| 86 | A36-06 | Mineur | `src/core/model_index.cpp:440-442` | Anciens paquets TensorRT jamais retirés |  |
| 87 | A36-07 | Mineur | `tools/release/windows/ajp.iss:16-50` | Désinstaller laisse plusieurs Go sans le dire |  |
| 88 | A36-08 | Mineur | `src/app/play/paths.h:13-36` | Caches reconstructibles hors de `~/.cache` sous CachyOS |  |
| 89 | A36-09 | Mineur | `tests/unit/app/settings_test.cpp:620-645` | Aucun test de passage entre versions hors des clés inconnues |  |
| 90 | A05-11 | Mineur | `src/prepare/process_win.cpp:128` | `trtexec.exe` relit sa ligne de commande en page ANSI : compte non ANSI, TensorRT inutilisable | A20-02 (même cause) |
| 91 | A08-17 | Mineur | `src/app/play/player.cpp:162` | Treize `path::string()` (dette) | A20-02 (même cause) |
| 92 | A24-09 | Mineur | `src/app/play/visible_error.h:44-45` | 56 codes sur 71 affichés par leur nom anglais | Corrigé : chaque code de refus a une phrase traduite, suivie de son nom (test sur les 71 codes) |
| 93 | A26-09 | Mineur | `src/app/main.cpp:526` | Refus d'écriture d'`update.toml` ignoré : compté comme un plantage | A20-01 (même cause) |

<a id="c02"></a>

## C02. Plantages et pertes de données

Modèle : Opus 5.5 (effort high) ; Fable 5.2 pour A08-02 et A30-01 (durées de vie entre threads et files CUDA).

Les deux bloquants restants et tout ce qui plante ou perd un réglage.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A13-01 | Bloquant | `src/app/play/pipeline.cpp:172` | Index de sous-titres externes gardé à la réouverture ou à l'épisode suivant : déréférence nulle, plantage |  |
| 2 | A04-01 | Majeur | `src/prepare/onnx_scan.cpp:165-168` | Emplacement de poids externe en UTF-8 invalide : exception, donc plantage sous Windows (à confirmer) |  |
| 3 | A05-01 | Majeur | `src/prepare/process_posix.cpp:127` | Sous Linux, `trtexec` survit à la mort du lecteur et fausse la construction suivante (démontré) |  |
| 4 | A08-02 | Majeur | `src/app/play/player_run.cpp:191` | Le fil de relevé de `ChainStats` lit `kernels_` après la destruction du backend |  |
| 5 | A08-04 | Majeur | `src/app/play/chain.cpp:351-358` | Raccourcis de séance écrits dans `play.toml` par toute réouverture |  |
| 6 | A13-02 | Majeur | `src/app/play/session.cpp:66-68` | Fin de flux après abandon de lecture prise pour une vraie fin : épisode suivant lancé (MED-01) |  |
| 7 | A14-01 | Majeur | `src/media/demuxer.cpp:367` | `AVERROR_EOF` pris pour une vraie fin sans regarder `pb->error` : panne d'E/S en MP4 qui finit l'épisode sans notice |  |
| 8 | A18-01 | Majeur | `src/core/log.cpp:163-165` | `close_log` ne vide pas `ajp.log`, et tout journal après lui plante (démontré) |  |
| 9 | A22-02 | Majeur | `src/app/play/settings_toml.cpp:624-626` | Écriture refusée de `play.toml` : le changement est perdu pour de bon (base de fusion retenue avant l'écriture) |  |
| 10 | A25-01 | Majeur | `src/app/play/engine_screen.cpp:202-311` | UX-02 : assistant relancé en lecture, ses choix et sa chaîne calibrée réécrits par l'ancienne chaîne |  |
| 11 | A30-01 | Majeur | `src/gpu/cuda/backend/trt_engine.cpp:115-119` | Moteurs TensorRT détruits sans attendre les files CUDA, à chaque fermeture de chaîne (à confirmer) |  |
| 12 | A36-01 | Majeur | `src/app/play/settings_toml_lenient.cpp:73-74` | Au retour arrière, `play.toml` perd toutes les clés de la version récente dès qu'une valeur lui est inconnue (démontré) | A22-06 (même cause, gravité relevée) |

<a id="c03"></a>

## C03. Banc de qualité et tests du chemin de la cible

Modèle : Fable 5.2 pour la conception des mesures (références FP32, PSNR, ΔE_ITP, seuils) ; Opus 5.5 pour la CI et les scripts.

Préalable aux chantiers de qualité : sans mesure contre un original, aucune correction de qualité ne se prouve, et la règle « une baisse de qualité pour la vitesse doit être mesurée » ne peut pas s'appliquer.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A40-03 | Majeur | `tools/bench/rife_triplets.py:96-100` | Qualité : rien ne mesure DistilDRBA contre RIFE ; le banc de qualité le refuse |  |
| 2 | A03-02 | Majeur | `tests/unit/backend_tensorrt/rife_fused_test.cpp:397` | Concordance des chaînes fusionnées vérifiée seulement par des tests cachés, à 30 dB pour RIFE |  |
| 3 | A03-03 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:284-290` | Variante mouvement de DistilDRBA fusionné sans aucun test ni essai au chargement |  |
| 4 | A32-03 | Majeur | `tests/unit/` | Aucun test de la sortie du lecteur contre un original |  |
| 5 | A40-01 | Majeur | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70` | Aucun réseau livré ne tourne en CI, alors que les essais cachés passent en 4 min sous lavapipe |  |
| 6 | A40-02 | Majeur | `.github/workflows/ci.yml:219-258` | Le chemin de la cible (TensorRT, CUDA, SPAN FP8, RIFE et DistilDRBA fusionnés) n'est jamais exécuté avant une release (B3) |  |
| 7 | A40-04 | Majeur | `tests/unit/app/e2e_play_test.cpp:96-104` | Le seul bout en bout joue sans SPAN, RIFE ni DistilDRBA |  |
| 8 | A43-01 | Majeur | `tests/unit/app/i18n_test.cpp:83` | Le test i18n laisse passer `ui_tr`, les clés passées par variable et les textes français en dur |  |
| 9 | A40-08 | Mineur | `src/gpu/guarded_backend.h:56-170` | `GuardedKernelOps` jamais testé (26,5 % couvert) |  |
| 10 | A40-09 | Mineur | `CMakePresets.json:348` | TSan ne lance que les 6 tests `Inbox` à chaque poussée |  |
| 11 | A40-10 | Mineur | `src/app/play/span_int8.cpp:194-202` | `span_variant_order` non testé avec les capacités d'une RTX sous Windows |  |
| 12 | A40-11 | Mineur | `tests/unit/vk_backend/rife_fused_chain_test.cpp:100-106` | Variante mouvement des chaînes fusionnées jugée sur « flot borné » seulement |  |
| 13 | A40-12 | Mineur | `cmake/AjpModule.cmake:77` | Les 21 cas cachés sont invisibles à `ctest -N` et à `check-skips.sh` |  |
| 14 | A40-13 | Mineur | `tests/unit/app/ring_stage_test.cpp:150-153` | Tests à l'horloge réelle (26 fichiers avec `sleep_for`) |  |
| 15 | A40-14 | Mineur | `.github/workflows/cloud.yml:57-58` | Tâches qui ne passent pas `check-skips.sh` ; préréglages de release non testés |  |
| 16 | A40-15 | Mineur | `.github/workflows/ci.yml` | Aucune mesure de couverture en CI (76 % mesuré ici) |  |

<a id="c04"></a>

## C04. Qualité : SPAN et mode Auto

Modèle : Fable 5.2 (effort high).

Source 1080p gardée, b8f64 essayé, FP8 seulement avec reçu, RIFE 4.25 enfin essayé par le banc. Décisions 8 et 14 : DistilDRBA hors de l'Auto jusqu'à la mesure, RTX 50 retirée du FP8 jusqu'à une mesure.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A08-15 | Majeur | `src/app/play/chain_setup.cpp:99` | Qualité : « Restaurer les traits » coupé, la source 1080p est réduite à 720p par l'étage alors que la chaîne la croit gardée ; refus « forme différente » en RIFE image |  |
| 2 | A18-18 | Majeur | `src/app/play/auto_bench_run.cpp:363-367` | Qualité : le banc essaie RIFE 4.25 sans dossier, l'essai échoue toujours en silence et l'Auto reste sur 4.22 lite | Cause directe d'A25-08 et d'A38-02 |
| 3 | A21-03 | Majeur | `src/gpu/backend.h:132-133` | Qualité : anneau en RGB10A2 par défaut avant le grade HDR, sans la porte de qualité de l'ADR 0008 |  |
| 4 | A49-01 | Majeur | `src/app/play/auto_config.cpp:69-75` | L'Auto ne joue jamais le SPAN b8f64, même en Ultra |  |
| 5 | A49-02 | Majeur | `src/app/play/auto_config.cpp:48` | Source 1080p réduite à 720p ou 540p avant SPAN | Aggravé par 07 (complément) : sur une carte Qualité ou Ultra, calibrer donne une image moins bonne que ne pas calibrer |
| 6 | A49-03 | Majeur | `src/app/play/auto_config.cpp:167-169` | Auto calibré en FP8 malgré l'ADR 0007 suspendue, sans essayer le b8f64 FP16, pour 6,5 % |  |
| 7 | A49-06 | Majeur | `src/gpu/cuda/backend/warp_kernels.cu:144-171` | Warp après SPAN bilinéaire, gain jamais mesuré |  |
| 8 | A22-03 | Majeur | `src/app/play/settings.h:106-117` | Qualité sacrifiée par défaut : l'Auto de l'ordre warp et SPAN vaut toujours « le plus rapide » | A09-08 (même cause, gravité relevée) |
| 9 | A49-04 | Majeur | `src/app/play/options.h:113-114` | Anneau RGB10A2 par défaut sans la porte de banding de l'ADR 0008 | A21-03 (même cause) |
| 10 | A02-01 | Majeur | `src/gpu/span_fp8/metrics.cpp:39` | La porte de qualité compte une image avec des NaN à 99 dB et la fait passer (démontré) |  |
| 11 | A03-01 | Majeur | `src/gpu/cuda/backend/slice_kernels.cu:24-45` | L'ingestion TensorRT rééchantillonne la chroma en bilinéaire au lieu du Spline36 du contrat (et de Vulkan) |  |
| 12 | A07-01 | Majeur | `src/app/play/machine_calibration.cpp:139-151` | La calibration mesure une chaîne différente de celle que l'Auto calibré applique |  |
| 13 | A07-02 | Majeur | `src/app/play/machine_calibration.cpp:338` | « Tient » ignore les images non interpolées quand le Rapide coupe RIFE |  |
| 14 | A23-03 | Majeur | `src/app/play/settings_panel.cpp:250-255` | « Auto (calibré) » choisi au panneau applique une chaîne non calibrée |  |
| 15 | A25-02 | Majeur | `src/app/play/first_run.cpp:263-265` | Sur une RTX, calibration mesurée sous Vulkan sans le dire, jamais refaite quand TensorRT arrive : image plus petite |  |
| 16 | A31-06 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | Cran Qualité ou Performance retombé en FP16 : garde le refus du fusionné, plus lent qu'Équilibre (toujours sous Windows en Performance) | Touche aussi CachyOS (34) ; Passé à C05, qui possède ces fichiers |
| 17 | A31-07 | Majeur | `src/app/play/drba_fused_choice.cpp:9` | DistilDRBA Performance avec graphe FP8 : moins de qualité et plus lent que le fusionné FP16 | Passé à C05, qui possède ces fichiers |
| 18 | A34-01 | Majeur | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | Auto calibré : b8f64 FP8 CUTLASS sous Linux, b5f48 FP16 ou INT8 sous Windows | Précisé par 07 (complément) : sous Windows en 1080p, b8f64 INT8 ou FP16 ; Traité par A49-03 : l'Auto calibré ne pose plus le FP8 |
| 19 | A36-02 | Majeur | `src/app/play/machine_calibration.cpp:240-252` | `calibration.toml` jamais invalidé par la version du lecteur, du module ni par le pilote : Auto calibré bridé |  |
| 20 | A37-01 | Majeur | `src/gpu/cuda/backend/span_fp8_net.cu:225-228` | SPAN FP8 CUTLASS déclaré pris en charge sur RTX 50 sans y avoir tourné ; seule garde la porte aveugle aux NaN (à confirmer) |  |
| 21 | A38-02 | Majeur | `README.md:32` | Le README promet RIFE 4.25 en Qualité et Ultra, l'Auto calibré impose 4.22 lite : baisse de qualité sans mesure | A25-08 (même cause) |
| 22 | A49-07 | Mineur | `src/app/play/span_int8.cpp:194-202` | Cran Équilibre annoncé FP16 mais INT8 dès qu'un reçu existe |  |
| 23 | A49-08 | Mineur | `src/gpu/span_fp8/derive.h:17-37` | Porte du FP8 réglée pour laisser passer le modèle |  |
| 24 | A49-09 | Mineur | `src/app/play/upscale_motion.cpp:97-193` | Une seule passe RIFE par paire sous 15 px : mouvements non linéaires |  |
| 25 | A49-10 | Mineur | `src/app/play/chain_setup.cpp:27` | RIFE estime le mouvement à 720p au-dessus de 720p |  |
| 26 | A49-11 | Mineur | `src/app/play/upscale_stage.cpp:498-504` | Dessins « tenus » : sortie SPAN réutilisée, petits changements gelés |  |
| 27 | A49-12 | Mineur | `src/app/play/placebo_screen.cpp:130` | Présentation par défaut de libplacebo, sans anti-ringing |  |
| 28 | A49-13 | Mineur | `src/media/frame_data.cpp:247-275` | Conversion hôte bilinéaire, 12 bits ramené à 10 |  |
| 29 | A49-14 | Mineur | `tools/rife/convert_rife_fp16.py` | Aucune mesure des RIFE et DistilDRBA FP16 contre leurs originaux FP32 |  |
| 30 | A49-16 | Mineur | `tools/drba/export_distil_drba.py:155-198` | Qualité : DistilDRBA n'existe qu'en FP16, sans porte ni graphe FP32 livré, gain du FP16 jamais mesuré |  |
| 31 | A03-23 | Mineur | `tools/drba/export_distil_drba.py:46` | Qualité : DistilDRBA v1 jamais exporté ni comparé ; gain de v2 lite inconnu | A07-13 (même cause pour la porte de qualité) |
| 32 | A07-03 | Mineur | `src/media/bench_clip.cpp:143` | Chemin du clip de calibration en page ANSI sous Windows : la calibration casse | Requalifié en mineur (dette) par 34 |
| 33 | A07-04 | Mineur | `src/app/play/span_int8.cpp:139-150` | Un reçu FP8 refusé l'est pour toutes les tailles et pour de bon |  |
| 34 | A07-05 | Mineur | `src/app/play/auto_config.cpp:194-201` | Bornes calibrées (1080p, 24 i/s, TensorRT) appliquées à toute source et tout moteur |  |
| 35 | A07-06 | Mineur | `src/app/play/auto_config.cpp:214-220` | `preset_chain_differs` ignore `auto_free` et `restore_lines` |  |
| 36 | A07-07 | Mineur | `src/core/auto_memory.cpp:38` | Mémoire de l'Auto plafonnée à ×6 alors que l'Auto monte à ×8 |  |
| 37 | A07-08 | Mineur | `data/gpu_tiers.toml:19-24` | Cartes portables courantes rangées au rang de la carte de bureau |  |
| 38 | A07-09 | Mineur | `src/app/play/machine_calibration.cpp:283` | Aucune garantie de mesure à chaud et stable |  |
| 39 | A07-10 | Mineur | `src/app/play/auto_bench_run.cpp:133` | Erreur de copie de `play.toml` ignorée : l'essai joue les réglages par défaut |  |
| 40 | A07-11 | Mineur | `tests/unit/app/auto_config_test.cpp` | Aucun test de la branche calibrée ni de la boucle de calibration |  |
| 41 | A07-13 | Mineur | `src/app/play/auto_config.h:19-22` | La « porte de qualité » qui tient DistilDRBA hors de l'Auto n'est écrite nulle part ; le commentaire renvoie à un ADR sans rapport |  |
| 42 | A25-05 | Mineur | `src/app/play/wizard.h:34` | UX-07 : `updates_managed` jamais rempli |  |
| 43 | A25-06 | Mineur | `src/app/play/engine_screen.cpp:110-113` | `calibrate_on_screen` rend toujours Auto, même sur annulation |  |
| 44 | A25-07 | Mineur | `src/app/play/engine_screen.cpp:274-291` | Inscription « Ouvrir avec » synchrone sur le thread de la fenêtre |  |
| 45 | A25-08 | Mineur | `src/app/play/first_run.cpp:461-465` | La calibration ne connaît que RIFE 4.22 lite ×2 : ni DistilDRBA ni RIFE 4.25 essayés |  |
| 46 | A25-09 | Mineur | `src/app/play/engine_screen.cpp:218` | Assistant relancé : part des défauts, pas des réglages en place |  |
| 47 | A25-10 | Mineur | `src/app/play/engine_screen.cpp:262` | Fenêtre fermée pendant le téléchargement : `settings_after` perdu |  |
| 48 | A25-11 | Mineur | `src/app/play/engine_screen.cpp:125` | TRT-18 : dossier de calibration à nom fixe effacé par `remove_all` |  |
| 49 | A25-12 | Mineur | `src/app/play/first_run.cpp:195-218` | UX-09 : unités et états en dur dans l'interface |  |
| 50 | A25-13 | Mineur | `src/app/play/wizard.cpp:30` | UX-14 : écran Moteurs sauté sans explication |  |
| 51 | A25-14 | Mineur | `src/app/play/engine_screen.cpp:72-76` | UX-23 : aucune navigation au clavier |  |
| 52 | A25-15 | Mineur | `src/app/play/engine_screen.cpp:43` | Redessin toutes les 16 ms sans événement |  |
| 53 | A25-16 | Mineur | `src/app/play/engine_screen.cpp:343` | Quatre `path::string()` (dette, voir A20-02) |  |
| 54 | A25-18 | Mineur | `tests/unit/app/wizard_test.cpp` | `run_engine_screen` et `calibrate_on_screen` sans test |  |
| 55 | A25-19 | Mineur | `src/app/play/first_run.cpp:445` | Texte de la mise à jour automatique différent entre l'assistant et le panneau |  |
| 56 | A34-03 | Mineur | `src/app/play/single_instance.cpp:114-120` | Canal nommé de l'instance unique à nom prévisible |  |
| 57 | A34-04 | Mineur | `src/app/play/default_player.cpp:203` | Plusieurs épisodes ouverts : un processus par fichier |  |
| 58 | A34-05 | Mineur | `src/app/play/system_peak.cpp:136-158` | Fabrique DXGI recréée toutes les 2 s |  |
| 59 | A34-07 | Mineur | `src/app/play/loop_wait.cpp:45-57` | Attente précise à 1 ms sous Windows (`--precise-wait`) |  |
| 60 | A34-08 | Mineur | `src/media/cuda_driver.cpp:92-110` | `libcuda.so.1` chargé par son seul nom sous Linux |  |
| 61 | A34-09 | Mineur | `cmake/utf8.manifest:1-9` | Manifeste sans `supportedOS` ni `dpiAwareness` |  |
| 62 | A34-10 | Mineur | `src/app/play/placebo_open.cpp:510-515` | Présentation FIFO identique partout malgré la swapchain NVIDIA de Windows |  |
| 63 | A34-11 | Mineur | `src/subs/text_encoding.cpp:117-145` | WIN-09 vrai en partie seulement |  |
| 64 | A34-12 | Mineur | `src/app/play/data_dirs.cpp:67-101` | Migration en cours qui bloque l'instance unique |  |
| 65 | A40-07 | Mineur | `tests/unit/backend_tensorrt/backend_test.cpp:92` | Le test d'ingestion TensorRT fige la chroma bilinéaire | A03-01 (même cause) |

<a id="c05"></a>

## C05. Qualité : interpolation (RIFE 4.22 lite, RIFE 4.25, DistilDRBA)

Modèle : Fable 5.2 (effort high ; xhigh pour la numérique FP16 des grilles et des sommes).

Grilles en FP32 dans les trois réseaux, sommes FP16 du repli Vulkan jugées contre FP32 sur une vraie image, DistilDRBA avec porte et mesure.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A03-20 | Majeur | `src/kernels/glsl/rife_fused_glue.comp:121-128` | Qualité : grille de base de RIFE v1 arrondie en FP16 dans les deux RIFE du menu, TensorRT et Vulkan (jusqu'à 0,234 px à 1920 de large), sans gain de vitesse |  |
| 2 | A12-12 | Majeur | `src/core/free_rate.cpp:28` | Qualité : le Rapide ignore les paires d'une image ; après un seul à-coup GPU, l'estimation se fige et l'interpolation s'arrête pour la séance (démontré) |  |
| 3 | A45-01 | Majeur | `src/gpu/vulkan/backend/net_presets.cpp:1-29` | Qualité : la table pré-réglée NvidiaAda met 28 couches sur 29 de SPAN et RIFE 4.25 en sommes FP16 dans le repli Vulkan, sans mesure (erreur par couche ×4 à ×9 en simulation, peut-être sous-estimée : troncature des Tensor Cores) |  |
| 4 | A45-02 | Majeur | `src/gpu/vulkan/backend/net_autotune.cpp:101-115` | Qualité : porte de l'autotuner trop large et jugée contre FP16 ; règle le rendu normal sur AMD et Intel | Requalifié en majeur par le complément 45 |
| 5 | A46-03 | Majeur | `tools/drba/export_distil_drba.py:194-197` | Qualité : le graphe FP16 livré n'est jamais comparé au réseau d'origine (16,5 codes 10 bits d'écart mesuré) |  |
| 6 | A44-01 | Majeur | `src/gpu/ir/pass_cast.cpp` | Qualité, repli Vulkan : chaîne de grille de DistilDRBA en F16 (jusqu'à 2,07 dB et écart max 16,3/255 sur un panoramique réel, coût du F32 négligeable) | A03-18 (même cause, mesuré côté IR) |
| 7 | A46-04 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:64-69` | Qualité : la voie fusionnée de DistilDRBA construit aussi la grille en FP16, alors que RIFE fusionné calcule la somme en FP32 (sa base reste FP16 : A03-20) | A03-18 (même cause) |
| 8 | A03-13 | Majeur | `src/app/play/upscale_drba.cpp:247-257` | DistilDRBA fait une passe par image créée, RIFE une par paire : au-delà de ×5, plus de réseau que RIFE 4.25 |  |
| 9 | A03-14 | Majeur | `src/app/play/upscale_drba.cpp:193-208` | Pas d'entrée par pointeurs : quatre `rife_input` de 7 plans par paire, presque rien de lu, plus cher que la passe |  |
| 10 | A03-18 | Majeur | `tools/rife/convert_rife_fp16.py:35` | Grille des dix GridSample de DistilDRBA en FP16 dans le modèle livré (pas de 0,31 px à 1280 de large), reproduite par la colle fusionnée |  |
| 11 | A09-01 | Majeur | `src/app/play/upscale_drba.cpp:104` | Côté « après » de DistilDRBA, son pont et le Smooth jugent les coupes sur le score brut, sans pré-analyse ni bandes noires |  |
| 12 | A32-04 | Majeur | `src/app/play/upscale_drba.cpp:179-256` | DistilDRBA refait deux `rife_input` par image créée, plus cher que son réseau | A03-14 (même cause) |
| 13 | A09-20 | Mineur | `src/app/play/upscale_stage.cpp:574-579` | Qualité : baisse pour la vitesse sans mesure (gain estimé à 30 ms par saut à ×3) |  |
| 14 | A09-24 | Mineur | `src/app/play/drba_model.h:17-21` | Qualité : demi-flot de DistilDRBA dès 1 440 lignes sans mesure ; seuil sur la hauteur d'estimation |  |
| 15 | A45-03 | Mineur | `src/gpu/vulkan/backend/kernel_ops.h:116-143` | Qualité : pas de débandage avant SPAN sous Vulkan |  |
| 16 | A45-15 | Mineur | `src/gpu/vulkan/shaders/net_tensor.glsl:75-77` | Qualité : arrondi f32 → f16 et sous-normaux F16 laissés au pilote, sans `float_controls` ni test exact |  |
| 17 | A46-13 | Mineur | `data/luts/HDR grading base ver.9.0 33 point.cube` | Qualité : LUT de base à 33 points sur une courbe raide, 2,7 codes 10 bits d'erreur d'interpolation |  |
| 18 | A46-11 | Mineur | `tools/vk/presets/NvidiaAda-rtx4070ti-1.tsv` | Qualité : sommes FP16 pour 40 couches sur 58 dans les tables Ada, sans mesure d'image | A45-01 (même cause) |
| 19 | A03-04 | Mineur | `src/gpu/cuda/backend/tensorrt_rife.cpp:118-124` | Plan d'instant compact : un moteur RIFE entier lancé ensuite lit un instant périmé (latent) |  |
| 20 | A03-06 | Mineur | `src/gpu/cuda/backend/tensorrt_rife_fused.cpp:388` | `path::string()` dans les messages de refus : exception sous Windows avec un profil non ASCII | Dette (34) |
| 21 | A03-07 | Mineur | `src/gpu/cuda/backend/rife_kernels.cu:49-50` | `rife_pair` écrit le plan d'instant entier alors que RIFE fusionné n'en lit qu'un élément |  |
| 22 | A03-08 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:221-224` | Chaque paire recopie et réencode les deux images, alors que A(n+1) = B(n) |  |
| 23 | A03-09 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:323-324` | Cache de graphes plafonné sans éviction ni notice (latent) |  |
| 24 | A03-10 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:361` | Environ 150 lignes dupliquées entre les deux moteurs fusionnés, déjà divergentes |  |
| 25 | A03-11 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:241` | `__expf` approché pour RIFE, `expf` pour DistilDRBA |  |
| 26 | A03-12 | Mineur | `src/gpu/cuda/backend/slice_kernels.cu:268-273` | `ingest_tiled` compte le poids d'une prise hors tuile avant de la sauter (latent) |  |
| 27 | A03-15 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:225-242` | Encodeur refait à chaque côté : 6 encodages par paire pour une image nouvelle |  |
| 28 | A03-16 | Mineur | `src/gpu/cuda/backend/drba_fused_kernels.cu:277-290` | Flot agrandi puis rééchantillonné aussitôt par le warp |  |
| 29 | A03-17 | Mineur | `tests/unit/backend_tensorrt/drba_fused_test.cpp` | Aucun banc par paire ni par image ; RIFE 4.22 lite fusionné sans chiffre |  |
| 30 | A03-19 | Mineur | `src/app/play/chain_setup.cpp:359-363` | La vitesse de DistilDRBA n'est pas réinvestie : estimation à 720p alors qu'une passe pleine résolution reste moins chère que RIFE |  |
| 31 | A03-21 | Mineur | `src/app/play/chain_setup.cpp:396-397` | Repli du fusionné vers l'entier écrit au journal seulement (2,10 ms contre 0,92) |  |
| 32 | A03-22 | Mineur | `src/gpu/cuda/backend/tensorrt_backend.cpp:205-223` | Étapes de `stage()` relevées à la fermeture seulement ; profil de fin tronqué à 262 144 mesures sans le dire |  |
| 33 | A09-02 | Mineur | `src/app/play/rife_fused_choice.cpp:11` | Le cran Performance de l'interpolation coupe le moteur fusionné |  |
| 34 | A09-03 | Mineur | `src/app/play/span_auto.cpp:22-25` | Sous Windows, message faux « n'est pas une RTX 40 » quand le FP8 est éteint à la construction |  |
| 35 | A09-04 | Mineur | `src/app/play/upscale_drba.cpp:135-137` | `drba_next` ingère l'image suivante sans vérifier sa taille |  |
| 36 | A09-05 | Mineur | `src/app/play/upscale_motion.cpp:111` | Sans poses, deux passes de RIFE pour une seule image intermédiaire hors de ½ |  |
| 37 | A09-06 | Mineur | `src/app/play/upscale_stage.cpp:363` | Différence de `pts` sans vérifier `kNoPts` (débordement signé) |  |
| 38 | A09-07 | Mineur | `src/app/play/upscale_drba.cpp` | DistilDRBA privé de trois économies de l'étage (avance, saut, débandage) |  |
| 39 | A09-08 | Mineur | `src/app/play/settings.h:114-117` | Auto compare la hauteur de sortie à 1080 : vaut toujours Après par défaut |  |
| 40 | A09-09 | Mineur | `src/app/play/upscale_stage.cpp:154-161` | L'étage suppose ×2 pour tout `--model` |  |
| 41 | A09-10 | Mineur | `src/app/play/derived_model.cpp:19` | Sept `path::string()` sur des chemins du profil | Dette (34) |
| 42 | A09-11 | Mineur | `src/app/play/derived_model.cpp:18-22` | Modèles dérivés jamais purgés |  |
| 43 | A09-12 | Mineur | `src/app/play/upscale_stage.cpp:600-605` | Avec DistilDRBA et `--skip-held`, `skip()` met mal `previous_` à jour |  |
| 44 | A09-13 | Mineur | `src/app/play/upscale_motion.cpp:39` | `--rife-warm` sans effet sous TensorRT |  |
| 45 | A09-14 | Mineur | `src/app/play/upscale_stage.cpp` | Fichier à 776 lignes, `outputs` de CCN 172 |  |
| 46 | A09-16 | Mineur | `tests/unit/app/upscale_stage_test.cpp` | Plusieurs branches de l'étage sans test |  |
| 47 | A09-17 | Mineur | `src/app/play/span_auto.cpp:67-71` | Clé de tentative FP8 « modèle\|vidéo » sans la taille de travail |  |
| 48 | A09-18 | Mineur | `src/app/play/upscale_drba.cpp:144` | DistilDRBA mesure à la taille de travail, RIFE sur la copie 720p |  |
| 49 | A09-19 | Mineur | `src/app/play/upscale_drba.cpp:99` | Attente du côté b non comptée au journal |  |
| 50 | A09-21 | Mineur | `src/app/play/drba_model.cpp:14-28` | Repli silencieux entre flot entier et demi-flot, dans les deux sens |  |
| 51 | A09-22 | Mineur | `src/app/play/rife_plan.cpp:10-14` | `kPlanCutFloor` comparé au score brut ; contradiction sous une zone utile de 0,75 |  |
| 52 | A09-23 | Mineur | `src/app/play/rife_plan.h:39-42` | `kPlanCutFloor` tenu sur une seule série |  |
| 53 | A44-02 | Mineur | `src/gpu/ir/pass_fuse.cpp` | Pas de fusion élément par élément : DistilDRBA 230 dispatchs contre 134 pour RIFE 4.22 lite, lié aux dispatchs (à confirmer sur GPU) |  |
| 54 | A44-03 | Mineur | `src/gpu/ir/import.cpp:141` | Tri topologique quadratique : un ONNX hostile de 0,9 Mo prend 8,5 s |  |
| 55 | A44-04 | Mineur | `src/gpu/ir/onnx_reader.cpp` | Amplification mémoire ×8 à ×16 bornée seulement par la taille du fichier |  |
| 56 | A44-06 | Mineur | `src/gpu/ir/pass_fuse.cpp:103` | Facteur par canal fondu dans des poids F16 : milliers de poids à zéro, sans perte mesurée |  |
| 57 | A44-07 | Mineur | `src/gpu/ir/onnx_reader.cpp:208` | Valeurs INT32 non vérifiées en plage |  |
| 58 | A44-10 | Mineur | `src/gpu/ir/onnx_info_main.cpp` | `ajp_onnx_info` ne rejoue pas les passes propres à Vulkan |  |
| 59 | A44-11 | Mineur | `tests/unit/gpu_ir/` | Aucun test sur les modèles livrés ; passes Vulkan hors fuzz |  |
| 60 | A44-12 | Mineur | `src/gpu/ir/pass_transpose.cpp` | Deux DepthToSpace là où un suffit en mode phases |  |
| 61 | A45-05 | Mineur | `src/gpu/vulkan/backend/queue.cpp:117-125` | Un `vkQueueSubmit` refusé fige la file pour de bon |  |
| 62 | A45-06 | Mineur | `src/gpu/vulkan/backend/kernel_frames.cpp:238-248` | Échec des vues : refus rendu au lieu de la copie promise |  |
| 63 | A45-07 | Mineur | `src/gpu/vulkan/backend/net_build.cpp:194-218` | Écriture d'artefacts sans fsync, temporaires jamais purgés (QUA-05) |  |
| 64 | A45-08 | Mineur | `src/gpu/vulkan/backend/entry.cpp:7-9` | `--profile-dispatches` sans effet dans le lecteur |  |
| 65 | A45-11 | Mineur | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70` | Tests de concordance des moteurs fusionnés Vulkan cachés, hors CI |  |
| 66 | A45-12 | Mineur | `src/gpu/vulkan/backend/net_presets.cpp:4` | Table pré-réglée limitée à SPAN b5f48 720p et RIFE 4.25 |  |
| 67 | A45-13 | Mineur | `src/gpu/vulkan/backend/vulkan_backend.cpp:384-395` | Artefact réabaissé et réécrit sans repasser la porte |  |
| 68 | A45-14 | Mineur | `src/gpu/vulkan/backend/kernel_ops.h:116-143` | Ni mémoire de la carte ni temps par étage dans les statistiques du repli |  |
| 69 | A46-08 | Mineur | `data/luts/ATTRIBUTION.md:11-13` | Auteur et licence des quatre LUT livrées absents |  |
| 70 | A46-09 | Mineur | `THIRD_PARTY_NOTICES.md:8-12` | Avis des données livrées faux ou incomplets (ArtCNN, RIFE) |  |
| 71 | A46-10 | Mineur | `data/calibration/calibration.mkv` | 19,8 Mo d'extraits de Blu-ray commerciaux dans un dépôt public, plus livrés |  |
| 72 | A46-12 | Mineur | `src/render/cube_lut.cpp:325-334` | Deux des quatre LUT livrées non reconnues comme LUT d'HDR |  |
| 73 | A46-14 | Mineur | `data/gpu_tiers.toml:45` | Erreurs de tranche sur NVIDIA (Quadro RTX 6000, RTX PRO) |  |
| 74 | A46-15 | Mineur | `tools/release/fetch-models.sh:86-98` | GMFSS encore listé dans `SHA256SUMS` de `models-interp-1` ; fichier facultatif annoncé absent |  |
| 75 | A21-07 | Mineur | `src/gpu/backend.h:217-229` | `Engine::enqueue` sans scalaire : l'instant voyage en plans pleine taille | A03-14 (même cause) |
| 76 | A45-04 | Mineur | `src/gpu/vulkan/backend/drba_fused_engine.cpp:335-341` | DistilDRBA fusionné Vulkan : encodeur refait trois images par appel | A03-15 (même cause) |

<a id="c06"></a>

## C06. Qualité : grade HDR et plans

Modèle : Fable 5.2 (effort high).

Décisions 16 et 17 : recaler la pile pour que les surfaces blanches ne soient pas toutes éclatantes, limiteur et Éclats après la pile, vrai MaxFALL ; pied de courbe adouci, vrai noir gardé sous RVB 4 environ. Pile des looks calculée dans le shader au lieu d'une LUT 65³.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A40-06 | Majeur | `tests/unit/app/e2e_play_test.cpp:236-243` | Qualité : un test fige une image SDR 38 % plus sombre que la source, grade coupé (à confirmer) |  |
| 2 | A49-05 | Majeur | `src/render/cube_lut.cpp:289-314` | Pile de looks composée en LUT 65³ : jusqu'à 15,5 codes PQ d'écart (mesuré) |  |
| 3 | A27-01 | Majeur | `ports/libplacebo/portfile.cmake:8` | libplacebo épinglé à 7.360.1, l'amont à 7.387.0 avec des correctifs de rendu ; notre correctif ne s'applique plus |  |
| 4 | A41-02 | Majeur | `src/render/shaders/anime-hdr-anime/3-anime.hook:119` | La pile et la passe kwin défont le grade : blanc à 695 nits sur 1 000, montée des émetteurs divisée par deux, limiteur OLED presque sans effet, MaxFALL faux (question à Tatoub) |  |
| 5 | A42-01 | Majeur | `src/media/shot_cutter.cpp:163` | Seuil de coupe sur la médiane, dessins tenus compris : un panoramique en trois devient une coupe par dessin (démontré, 24 plans sur 96 images) |  |
| 6 | A41-01 | Majeur | `src/render/cube_lut.cpp:287-314` | Pile de LUT composée en 65³ : 18,0 codes PQ d'écart au pire à l'écran, 129³ ne suffit pas, seule la chaîne dans le shader est exacte | A49-05 (même cause) |
| 7 | A42-06 | Mineur | `src/media/shot_probe.h:37-41` | Qualité : analyse en 256×144 sous NVDEC, mesurée en partie |  |
| 8 | A27-02 | Mineur | `tools/release/package-linux.sh:124` | `patchelf --set-rpath` pose un RUNPATH que `LD_LIBRARY_PATH` devance (démontré) |  |
| 9 | A27-03 | Mineur | `cmake/Dependencies.cmake:64` | Configure accepte un libplacebo sans notre correctif |  |
| 10 | A27-04 | Mineur | `cmake/Dependencies.cmake:4-5` | « Mêmes versions » Linux et Windows vrai en partie seulement |  |
| 11 | A27-05 | Mineur | `tools/deps/linux-from-source.sh:135-138` | Options de libplacebo fixées sous Windows seulement |  |
| 12 | A27-06 | Mineur | `tools/deps/linux-from-source.sh:58-66` | Marqueur de dépendance qui ignore les bibliothèques du système liées |  |
| 13 | A27-07 | Mineur | `CMakeLists.txt:28` | Le CI CUDA des PR ne compile pas toutes les architectures livrées |  |
| 14 | A27-08 | Mineur | `ports/libplacebo/0001-ajp-present-timing.patch:153-170` | Chemin `QUEUE_FULL` du correctif sur une hypothèse non vérifiée |  |
| 15 | A27-09 | Mineur | `cmake/Dependencies.cmake:115-117` | ImGui sans FreeType sous Windows |  |
| 16 | A27-10 | Mineur | `CMakePresets.json:18` | `AJP_WARNINGS_AS_ERRORS` OFF par défaut, contrairement à la règle |  |
| 17 | A27-11 | Mineur | `cmake/AjpSpirv.cmake:35` | `spirv-val` absent sous Windows sans un mot |  |
| 18 | A27-12 | Mineur | `third_party/cutlass/` | CUTLASS recopié sans commit ni somme |  |
| 19 | A27-13 | Mineur | `cmake/AjpModule.cmake:61-77` | En-têtes précompilés sans effet sur les tests |  |
| 20 | A27-14 | Mineur | `cmake/Dependencies.cmake:123` | `imgui_impl_sdl3.cpp` compilé et SDL3 lié en PUBLIC sans appelant ; liaisons du port imgui inutilisées sous Windows |  |
| 21 | A38-16 | Mineur | `docs/hdr-anime.md:56-60` | « Noirs profonds selon la scène » : réglé sur la médiane de l'épisode, et à 64 % sans mesure |  |
| 22 | A41-03 | Mineur | `data/luts/HDR grading base ver.9.0 33 point.cube` | Ombres écrasées : 0,30 nit (RVB 16) rendu à 0,02 (question à Tatoub) |  |
| 23 | A41-04 | Mineur | `src/app/play/placebo_grade.cpp:206-225` | Sans pile, la passe kwin reste active : blanc à 119 nits ; échec du hook non journalisé |  |
| 24 | A41-05 | Mineur | `src/render/grade.cpp:361-365` | `set_looks` lit un hook complet pour rien sur le thread de la boucle (images perdues) |  |
| 25 | A41-06 | Mineur | `src/render/shaders/sharpen.hook:14` | Netteté sur `SCALED`, domaine gamma ou linéaire selon la mise à l'échelle |  |
| 26 | A41-07 | Mineur | `src/render/grade_params.cpp:158-166` | Vrai noir annoncé mais jamais appliqué en sortie SDR |  |
| 27 | A41-08 | Mineur | `src/render/grade_params.cpp:115-116` | Source HDR non gradée : pic de l'écran non annoncé, hautes lumières écrêtées |  |
| 28 | A41-11 | Mineur | `tests/unit/render/cube_lut_test.cpp:146` | Aucun test sur les LUT livrées ni sur la chaîne grade, pile, kwin |  |
| 29 | A41-12 | Mineur | `src/app/play/placebo_screen.cpp:64-66` | Ligne de sortie en français en dur ; « · grade » affiché même sans hook posé |  |
| 30 | A42-02 | Mineur | `src/media/shot_cutter.cpp:167-168` | Éclair de deux images pris pour une coupe |  |
| 31 | A42-03 | Mineur | `src/media/shot_timeline.cpp:49-52` | Un saut pendant l'analyse laisse un trou |  |
| 32 | A42-04 | Mineur | `src/app/play/chain_shots.cpp:177-179` | `next_shot` retombe sur la scène en direct |  |
| 33 | A42-05 | Mineur | `src/app/play/options.h:138` | Pré-analyse active pendant les essais de calibration |  |
| 34 | A42-07 | Mineur | `src/media/shot_probe.cpp:66-68` | G calculé depuis R et B déjà bornés |  |
| 35 | A42-08 | Mineur | `src/media/flash_guard.h:1-5` | Limiteur de flash sur la luminance moyenne seule |  |
| 36 | A42-09 | Mineur | `src/app/play/chain_shots.cpp:93-96` | Coupes de l'interpolation dépendantes du grade |  |
| 37 | A42-10 | Mineur | `src/media/shot_cache.cpp:20-21` | Cache versionné par son format seulement |  |
| 38 | A42-11 | Mineur | `src/app/play/chain_shots.cpp:166-174` | Remplacement de plan à nombre égal non suivi |  |
| 39 | A42-13 | Mineur | `src/media/shot_cutter.cpp:173` | Images du plan en cours toutes gardées et recopiées |  |
| 40 | A42-14 | Mineur | `src/app/play/chain_shots.cpp:23-35` | Empreinte sur le premier mégaoctet seul |  |
| 41 | A42-16 | Mineur | `tests/unit/media/shot_test.cpp` | Aucun test de l'animation tenue ni de l'éclair de deux images |  |
| 42 | A42-17 | Mineur | `src/media/flash_guard.h:26-29` | Sécurité : limiteur de flashs tardif (1 000 ms à 4 Hz), première montée non comptée, pré-analyse non utilisée |  |
| 43 | A42-18 | Mineur | `src/app/play/upscale_stage.cpp:259` | Analyse jetée quand deux relectures attendent : elle manque au limiteur |  |
| 44 | A42-19 | Mineur | `src/app/play/chain_hdr.cpp:252` | Plafond sans effet en sortie SDR mais journalisé comme posé |  |
| 45 | A10-17 | Mineur | `src/app/play/placebo_screen.cpp:155` | Commentaire « netteté sur MAIN » faux | A41-06 (même cause) |
| 46 | A28-11 | Mineur | `.github/workflows/ci.yml:219-251` | CUDA des PR : `sm_89` seul, en Debug | A27-07 (même cause) |
| 47 | A28-20 | Mineur | `.github/workflows/release-linux.yml:64-70` | Clé de cache de la release sans version d'image | A27-06 (même cause) |

<a id="c07"></a>

## C07. Résilience : GPU, étages refusés, veille, périphériques

Modèle : Opus 5.5 (effort high).

Décisions 6, 7, 9 et 10 : un étage refusé laisse la vidéo continuer sans lui avec un avis et « Réessayer » ; perte du GPU : une relance avec avis ; veille : pause avant, recalage au réveil.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A01-01 | Majeur | `src/gpu/cuda/backend/cuda_status.h:15-32` | Perte du GPU mal reconnue : `sticky` ignore 999, 46 et 911, `health()` répond OK après un TDR ou un Xid (à confirmer) |  |
| 2 | A08-01 | Majeur | `src/app/play/chain_setup.cpp:166-171` | « Annuler et regarder » pris pour un refus par les replis, qui relancent une autre construction |  |
| 3 | A10-01 | Majeur | `src/app/play/placebo_screen.cpp:549` | Perte du GPU ou de la surface prise pour une fenêtre cachée : image figée, journal inondé, aucun message (GPU-06) |  |
| 4 | A14-02 | Majeur | `src/media/demuxer.cpp:252-254` | `rw_timeout` sans effet sur un partage monté : un NAS muet fige la lecture puis la fermeture |  |
| 5 | A15-01 | Majeur | `src/media/video_decoder.cpp:320-323` | 4:4:4, et 4:2:2 sur RTX 50, via NVDEC arrêtent la lecture sous TensorRT (MED-02) |  |
| 6 | A16-01 | Majeur | `src/audio/sdl_sink.cpp:151-160` | Sans sortie par défaut après retrait : refus à chaque bloc, gel à chaque seek, appareil rebranché jamais repris (MED-08) |  |
| 7 | A17-02 | Majeur | `src/subs/subs_messages.h:42` | PGS et VobSub ignorés sans un mot, et ils empêchent le repli sur un .ass externe |  |
| 8 | A29-01 | Majeur | `src/app/play/chain_setup.cpp:497` | Un refus de RIFE ou de SPAN arrête l'épisode et renvoie à l'accueil, sans repli |  |
| 9 | A29-02 | Majeur | aucun fichier (`WM_POWERBROADCAST`, `PrepareForSleep` absents) | La mise en veille n'est traitée nulle part |  |
| 10 | A31-02 | Majeur | `src/app/play/player.cpp:589-602` | Aucune relance après perte du GPU alors que le mécanisme existe |  |
| 11 | A31-03 | Majeur | `src/app/play/engine_screen.cpp:341-344` | TensorRT refusé remplacé par Vulkan sans avis à l'écran |  |
| 12 | A31-05 | Majeur | `src/app/play/player.cpp:523-533` | « Annuler et regarder » coupe aussi SPAN déjà prêt |  |
| 13 | A31-15 | Majeur | `src/app/play/player_engines.cpp:259-260` | « Annuler » sur DistilDRBA fusionné lance la construction de l'entier, puis de RIFE ; même chose pour RIFE fusionné |  |
| 14 | A33-02 | Majeur | `src/app/play/placebo_open.cpp` | SDL coupe l'économiseur et la veille pendant toute la vie du processus, même en pause et à l'accueil |  |
| 15 | A31-01 | Majeur | `src/app/play/ring_stage.cpp:106-113` | Tout refus GPU devient `ThreadFailed` : `DeviceLost` et sa phrase perdus, figé par un test | A20-09 (même cause) |
| 16 | A31-04 | Majeur | `src/app/play/chain_setup.cpp:355` | Un moteur refusé à l'ouverture empêche toute lecture, rien n'est retenu | A29-01 (même cause) |
| 17 | A33-03 | Majeur | `src/core/clock.h:15` | Aucun trou de temps détecté : au réveil sous Windows, l'horloge court jusqu'à la fin de l'épisode | A29-02 (même cause, précisé) |
| 18 | A08-05 | Mineur | `src/app/play/chain_setup.cpp:160-172` | Repli au cran Équilibre sur n'importe quel refus, chaîne repliée absente de `chain.running` |  |
| 19 | A08-06 | Mineur | `src/app/play/chain_setup.cpp:376-389` | DistilDRBA : moteur image 720p construit puis jeté quand la dérivation mouvement échoue |  |
| 20 | A08-07 | Mineur | `src/app/play/player_run.cpp:549-562` | `follow_display` seulement dans la branche HDR, jamais avec `--sdr` |  |
| 21 | A08-08 | Mineur | `src/app/play/player_engines.cpp:54-67` | Événements jetés pendant une construction ; SPAN FP8 sans « Annuler et regarder » |  |
| 22 | A08-09 | Mineur | `src/app/play/chain_setup.cpp:277` | `path::string()` sur des chemins du profil, contre la règle de la CI | Dette (34) |
| 23 | A08-10 | Mineur | `src/app/play/engine_build.cpp:38` | Une instance de backend par construction ou lecture du cache, jusqu'à six de plus par ouverture |  |
| 24 | A08-11 | Mineur | `src/app/play/player_run.cpp:98` | `run_chain` d'environ 650 lignes, fichier de 753 lignes (QUA-15) |  |
| 25 | A08-12 | Mineur | `tests/unit/app/chain_setup_test.cpp` | Aucun test de `apply`, `switch_chain`, `with_kept_choices`, des replis ni de la fin de `run_chain` |  |
| 26 | A08-13 | Mineur | `src/app/play/player_run.cpp:334-340` | E/S synchrones sur le thread de la boucle |  |
| 27 | A08-16 | Mineur | `src/app/play/chain_setup.cpp:162-171` | Le repli remet les deux crans à Équilibre même quand un seul moteur est refusé |  |
| 28 | A29-03 | Mineur | `src/app/play/player_run.cpp:617` | Réouverture du même fichier traitée comme un fichier suivant |  |
| 29 | A29-04 | Mineur | `src/app/main.cpp:528` | Après mise à jour d'un moteur, la relance rejoue `argv` et perd la vidéo en cours |  |
| 30 | A29-05 | Mineur | `src/app/play/settings_panel.cpp:572-574` | DistilDRBA ne se choisit qu'à la main, hors Auto et hors calibration |  |
| 31 | A29-06 | Mineur | `src/app/play/player_engines.cpp:146` | Désérialisation des moteurs TensorRT sur le thread de la fenêtre sans pompe |  |
| 32 | A29-07 | Mineur | `src/app/play/engine_screen.cpp:164-167` | « Refaire le banc » annulé en lecture : la chaîne repasse en Auto |  |
| 33 | A31-08 | Mineur | `src/gpu/cuda/backend/cuda_status.h:38-39` | `cudaErrorMemoryAllocation` devient `VendorCallFailed` |  |
| 34 | A31-09 | Mineur | `src/app/play/upscale_drba.cpp:136` | Image ingérée avant le contrôle de taille |  |
| 35 | A31-11 | Mineur | `src/core/file_sync.cpp:75-91` | Disque plein : place vérifiée seulement à l'installation des moteurs |  |
| 36 | A31-13 | Mineur | `src/app/play/placebo_ring.cpp:218` | `ready()` vrai quand le sémaphore échoue |  |
| 37 | A31-14 | Mineur | `src/gpu/cuda/backend/tensorrt_export.cpp:262-263` | GPU bloqué sans être perdu : attente sans délai |  |
| 38 | A31-16 | Mineur | `src/media/video_decoder.cpp:400` | Au réveil, NVDEC touche le contexte mort en premier et rend `DecodeFailed` fatal : la relance d'A31-02 ne se ferait pas |  |
| 39 | A05-10 | Mineur | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` joint le fil de construction sur le thread de la fenêtre sans pompe | A08-10 (même cause) |
| 40 | A18-15 | Mineur | `src/app/play/bridges.cpp:374-385` | `mov_text` et `webvtt` refusés | A17-02 (même cause) |
| 41 | A31-12 | Mineur | `src/app/play/player_run.cpp:510` | Veille et hibernation non gérées | A29-02 (même cause) |

<a id="c08"></a>

## C08. Cadence, horloges et présentation

Modèle : Fable 5.2 pour les horloges (A12-*) ; Opus 5.5 pour le reste.

Décision 11 : plus de plafond à 60 Hz sur batterie, le plafond est toujours le maximum de l'écran. VRR hors sujet.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A08-03 | Majeur | `src/app/play/chain.cpp:204-211` | Facteur, Auto ou Rapide changés en direct sans refaire anneau, plus haut facteur ni warp | Corrigé |
| 2 | A10-02 | Majeur | `src/app/play/placebo_open.cpp:497` | Avec le défaut `--present=wait`, `wp_presentation` n'est jamais ouvert : aucune grille des vblanks sous CachyOS | Corrigé |
| 3 | A11-01 | Majeur | `src/app/play/display_contract.h:54` | Contrat d'écran figé sur l'écran d'ouverture (GPU-04) | Corrigé |
| 4 | A11-02 | Majeur | `src/app/play/host_facts.cpp:39-81` | Windows : requêtes d'écran sur le thread de la boucle (WIN-04, GPU-16, à mesurer) | Corrigé en partie : pic relu sur événement (filet à 10 s) et un seul relevé du mode par tour ; le fil « écran » et la fabrique DXGI gardée restent à faire, après mesure sous Windows |
| 5 | A12-01 | Majeur | `src/core/vblank_clock.cpp:32-36` | La grille des vblanks apprend la cadence du contenu au lieu de celle de l'écran (démontré) | Corrigé |
| 6 | A12-02 | Majeur | `src/core/auto_rate.cpp:281-291` | Stable sans palier calé au-dessus de ×1 à 165 Hz pour 23,976, ni à 144 Hz pour 29,97 (démontré) | Corrigé |
| 7 | A12-03 | Majeur | `src/core/vsync_phase.h:2-5` | Phase des vsyncs apprise sur le retour de `swap_buffers` au niveau Open (à confirmer sous Windows) | Corrigé |
| 8 | A24-02 | Majeur | `src/app/play/controls.cpp:167-174` | L'interface déclenche des présentations hors cadence pendant la lecture | Corrigé |
| 9 | A10-03 | Mineur | `src/app/play/ring_watcher.cpp:474` | Le veilleur de l'anneau sort sans un mot sur une erreur de `vkWaitSemaphores` (GPU-10) | Corrigé |
| 10 | A10-04 | Mineur | `src/app/play/placebo_open.cpp:323` | Attentes sans délai sur un GPU bloqué (GPU-17) | Corrigé : un garde de 60 s sur le démontage du GPU à la fermeture ; passé ce délai le lecteur le dit au journal et sort (code 75) au lieu de rester sans fenêtre (test du garde avec une horloge courte) |
| 11 | A10-05 | Mineur | `src/app/play/placebo_screen.cpp:522` | Rien ne dessine pendant la boucle modale de Windows | Non corrigé : la cause se lit (la boucle modale de Windows retient SDL_PollEvent, donc la boucle du lecteur), mais le remède fait tourner un tour de lecture depuis un observateur d'événements SDL, dans run_chain, et rien ne peut l'exercer hors de Windows |
| 12 | A10-06 | Mineur | `src/app/play/placebo_feedback.cpp:241` | Le niveau de retour ne retombe jamais quand rien ne revient | Corrigé |
| 13 | A10-07 | Mineur | `src/app/play/placebo_feedback.cpp:378` | Sous Windows, domaine d'horloge `CLOCK_MONOTONIC` qui n'y existe pas | Corrigé d'après le code : sous Windows le calage du domaine local se fait contre le compteur de performance (l'horloge de host_now_us), converti de tics en ns ; avant il demandait CLOCK_MONOTONIC, absent, et le niveau timing ne datait rien (test de la conversion) |
| 14 | A10-08 | Mineur | `src/app/play/placebo_screen.cpp:569` | Image acquise jamais soumise sur les sorties en erreur | Corrigé d'après le code : une image de la swapchain acquise et jamais soumise (sortie de show sur une erreur) est soumise au tour suivant avant d'en acquérir une autre, avec une ligne au journal |
| 15 | A10-09 | Mineur | `src/app/play/placebo_ring.cpp:229` | `retire` ignore l'échec de `hand_to_cuda` | Corrigé |
| 16 | A10-10 | Mineur | `src/app/play/placebo_screen.cpp:632` | `path::string()` dans les instantanés | Corrigé ; Dette (34) |
| 17 | A10-11 | Mineur | `src/app/play/placebo_open.cpp:275` | Cache des shaders lu sans plafond, `.part` jamais purgés | Corrigé |
| 18 | A10-13 | Mineur | `tests/unit/app/present_level_test.cpp` | Aucun test des chemins dégradés de la présentation | Corrigé en partie : les chemins dégradés qui se simulent ont leur test (repli de present timing muet et table des niveaux dans present_level_test, registre des présentations dans present_ledger_test, veilleur mort dans ring_watcher_test, ring hors CUDA par les tests e2e sous lavapipe) ; la perte du GPU dans show suit A10-01 (C07) ; le choix du domaine de temps de PresentFeedback reste sans test, il demande un pilote Vulkan réel |
| 19 | A10-14 | Mineur | `src/app/play/placebo_screen.cpp:464-470` | `present_us` et `queue_wait_us` sont le retour de `swap_buffers`, pas un vsync (rend inopérante la vérification d'A12-03) | Corrigé en partie : commentaires et phase des vsyncs ; shown_at_us_ et les intervalles restent sur le retour de swap_buffers |
| 20 | A10-15 | Mineur | `src/app/play/placebo_screen.cpp:328` | Présentation plafonnée vers 127 i/s à 165 Hz, RIFE comme DistilDRBA, cause à mesurer | Non corrigé : la cause n'est pas lisible dans le code, il faut la mesure |
| 21 | A10-18 | Mineur | `src/app/play/placebo_snapshot.cpp:45` | Instantané SDR en sRGB, pas dans l'espace de la fenêtre | Corrigé : l'instantané d'une fenêtre SDR est rendu dans l'espace de la fenêtre, plus toujours en sRGB ; une fenêtre HDR sans « .pq.ppm » reste en sRGB (même image qu'avant sous lavapipe, surface sRGB) |
| 22 | A10-19 | Mineur | `src/app/play/display_contract.cpp:85` | Contrat 203/1000 par défaut écrit au journal comme s'il avait été lu | Corrigé |
| 23 | A11-03 | Mineur | `src/app/play/display_edid.cpp:421-429` | Écran choisi par sous-chaîne du nom : deux écrans du même modèle confondus | Corrigé : l'écran se choisit par le connecteur entier (« DP-2 » ne prend plus « DP-10 »), puis par le nom EDID ; deux écrans du même nom ne donnent plus le connecteur du premier (connecteur vidé, VRR non armé), test `pick_display` |
| 24 | A11-04 | Mineur | `src/app/play/system_peak.cpp:85-108` | Entrée KWin choisie par le seul connecteur | Corrigé |
| 25 | A11-05 | Mineur | `src/app/play/display_edid.cpp:56-63` | Demi-flottant infini retenu comme pic | Corrigé |
| 26 | A11-06 | Mineur | `src/app/play/display_contract.cpp:24-42` | Hors KDE, blanc de référence fixé à 203 et marge mal rapportée sous Windows | Corrigé d'après le code : sous Windows le contrat d'écran prend la marge × le blanc SDR que SDL rapporte (niveau × 80 nits), plus 203 fixe ; hors Windows et hors KDE, SDL ne donne pas ce blanc et 203 reste, test |
| 27 | A11-07 | Mineur | `src/app/play/host_facts.cpp:247-250` | Windows : pic DXGI demandé seulement si un EDID a été lu | Corrigé |
| 28 | A11-08 | Mineur | `src/app/play/gpu_telemetry.cpp:311-372` | `GpuTelemetry::sample` écrit sans verrou depuis deux threads (GPU-09) | Corrigé |
| 29 | A11-09 | Mineur | `src/app/play/host_facts.cpp:378-380` | `on_battery()` sans cache sur le thread de la boucle | Corrigé par A31-10 : on_battery n'est plus appelé par la boucle ; Corrigé (plus d'appel dans la boucle : le plafond de batterie est retiré) |
| 30 | A11-10 | Mineur | `src/app/play/gpu_telemetry.cpp:58` | « chaleur » en dur dans l'OSD anglais | Corrigé |
| 31 | A11-11 | Mineur | `src/app/play/display_edid.cpp:139-147` | Pic et noir dépendent de l'ordre des blocs EDID | Corrigé en partie : taille des descripteurs Adaptive-Sync ; le journal montre désormais le pic et le noir de chaque bloc quand les deux parlent (`luminance_sources`, test). Le choix d'une seule source par grandeur change le pic ou le noir de l'écran cible : décision de Tatoub, à prendre sur le relevé de son écran |
| 32 | A11-12 | Mineur | `tests/unit/app/` | Aucun test du contrat d'écran, du multi-écran ni du pic de contenu | Corrigé : tests du choix d'écran (`pick_display`), du contrat avec le blanc SDR du système, et de la règle du pic de contenu sortie de `content_peak_nits` (`uncalibrated_peak_nits`, Windows et ailleurs) ; la lecture des propriétés SDL elles-mêmes reste sans test |
| 33 | A11-13 | Mineur | `src/app/play/host_facts.cpp:39-81` | Quatre copies de l'énumération `QueryDisplayConfig` | Corrigé d'après le code : une seule énumération `active_paths()` pour les quatre usages, qui recommence si la configuration change pendant la lecture (ERROR_INSUFFICIENT_BUFFER) ; code Windows, construit par la CI MSVC seulement |
| 34 | A11-14 | Mineur | `src/app/play/gpu_telemetry.cpp:73-76` | Unité « Go » en dur dans la ligne VRAM de l'OSD anglais (UX-09) | Corrigé |
| 35 | A12-04 | Mineur | `src/core/auto_rate.cpp:293-301` | Paliers calés sur le plafond passé comme fréquence d'écran | Corrigé |
| 36 | A12-05 | Mineur | `src/core/auto_rate.cpp:599-608` | Réarmement toutes les 8 s même sans palier calé | Corrigé |
| 37 | A12-06 | Mineur | `src/core/pacing.cpp:61-67` | Calage de vitesse sur une fréquence fausse, jamais sur la phase | Corrigé pour la phase (`core/phase_trim.h`) ; le calage sur la fréquence annoncée reste permis faute de mesure sous Windows |
| 38 | A12-07 | Mineur | `src/core/cadence.cpp:56-70` | `retarget` sort des bornes de `kRatioTerm` | Corrigé |
| 39 | A12-08 | Mineur | `src/core/steady_rate.cpp:162-166` | Le repli du Rapide compte les pertes du démarrage et des reprises | Corrigé |
| 40 | A12-10 | Mineur | `tests/unit/core/` | Aucun test de `VsyncPhase`, des paliers à 144 et 165 Hz, ni de `VblankClock` à la demande | Corrigé |
| 41 | A12-11 | Mineur | `src/core/clock_pll.h:28-40` | PLL audio réglée en lectures et non en temps (effet de 1 à 2 µs) | Corrigé |
| 42 | A24-04 | Mineur | `src/app/play/input.cpp:18-58` | `key_of` ne traduit pas des touches que `[keys]` accepte | Corrigé |
| 43 | A24-05 | Mineur | `src/app/play/input.cpp:76` | AZERTY : répétition et Ctrl + « - » mal traduits (UX-11) | Corrigé |
| 44 | A24-06 | Mineur | `src/app/play/input.cpp:94-116` | Quinze touches en dur avant `[keys]`, non refusées | Corrigé |
| 45 | A24-07 | Mineur | `src/app/play/user_keys.cpp:16-24` | Une entrée fautive de `[keys]` remet tous les raccourcis par défaut | Corrigé |
| 46 | A24-08 | Mineur | `src/app/play/user_keys.cpp:32` | `path::string()` dans le journal (dette, voir A20-02) | Corrigé |
| 47 | A24-10 | Mineur | `src/app/play/visible_error.h:28` | UX-17 : phrase d'erreur répétée | Corrigé |
| 48 | A24-11 | Mineur | `src/app/play/controls_draw.cpp:354-371` | UX-15 : textes de préparation sans largeur maximale | Corrigé d'après le code : le texte de préparation s'arrête avant « Annuler et regarder » et au bord de la fenêtre |
| 49 | A24-12 | Mineur | `src/app/play/controls_draw.cpp:114-125` | Message bref sans retour à la ligne, avis collés | Corrigé |
| 50 | A24-13 | Mineur | `src/app/play/panel_style.h:115-130` | UX-22 : badges qui ne suivent pas l'échelle | Corrigé d'après le code : pastilles aux mesures de la police, à la ligne quand elles ne tiennent pas |
| 51 | A24-14 | Mineur | `src/app/play/ui.cpp:452` | UX-08 : Échap ferme tout le panneau et jette les changements | Corrigé : Échap sur une liste ouverte ne ferme que la liste (vérifié sur capture, avant et après) |
| 52 | A24-15 | Mineur | `src/app/play/home_screen.cpp:205-213` | `dropped()` passe la fenêtre au premier plan et interroge le disque | Corrigé en partie : tri par extension séparé et testé, fenêtre passée devant dans sa propre fonction ; le disque reste lu une fois par fichier glissé (c'est ce qui écarte un dossier) |
| 53 | A24-16 | Mineur | `src/app/play/home_screen.cpp:244-250` | `save_settings` synchrone sur le thread de l'interface à l'accueil | Corrigé : à l'accueil, play.toml s'écrit par le thread d'écriture (LatestWriter), comme en lecture |
| 54 | A24-17 | Mineur | `src/app/play/home_screen.cpp:336-341` | Accueil redessiné toutes les 16 ms sans événement | Corrigé |
| 55 | A24-18 | Mineur | `src/app/play/placebo_imgui.cpp:107-124` | Atlas des polices en RGBA32, renvoyé depuis l'origine | Corrigé : atlas des polices en un octet par pixel (`ImTextureFormat_Alpha8`), quatre fois moins d'octets à chaque envoi ; capture du menu identique au pixel, sauf le titre (98 pixels plus clairs de 19 niveaux en moyenne : la frange sombre du filtrage linéaire d'un atlas RGBA disparaît). L'envoi part toujours de l'origine (limite de libplacebo 7.360) |
| 56 | A24-19 | Mineur | `src/app/play/placebo_overlays.cpp:63-76` | Barre visible : panneaux typographiés montés avec les dialogues | Corrigé : la barre visible fait monter les dialogues par libass (`ass_set_line_position`), événement entier et seulement sans position explicite ; un panneau `\pos` ou `\move` reste à sa place. Capture avant et après : panneau en (200,460) déplacé de 87 px avant, à sa place après ; réplique de deux lignes montée d'un bloc. Le décalage par morceau ne sert plus que le temps que l'image replacée arrive |
| 57 | A24-20 | Mineur | `src/app/play/placebo_overlays.cpp:45-47` | Sous-titres étirés à la nouvelle taille jusqu'au rendu suivant | Corrigé : quand la taille de sortie change, l'image à l'écran redemande ses sous-titres à la nouvelle taille, en pause aussi (avant : étirés jusqu'à l'image suivante, donc toute la pause), test de session |
| 58 | A24-21 | Mineur | `src/app/play/placebo_imgui.cpp:161-207` | Une passe de dessin par commande ImGui | Non corrigé : la cause se lit (une passe par commande ImGui, rejouée à chaque présentation) mais pas son coût ; le correctif proposé (interface dans une texture, un mélange par présentation) change le mélange des couches translucides en HDR et ajoute un mélange plein écran par présentation, ce qui peut coûter plus que les passes actuelles sur un vrai GPU. À décider sur mesure |
| 59 | A24-22 | Mineur | `src/app/play/osd_text.cpp:68` | Seconde bibliothèque libass pour l'OSD | Corrigé en partie : le relevé de la mémoire vidéo et du GPU dort dès que l'OSD ne le lit plus depuis 1 s et repart à la première lecture (`sample_demand.h`, test). Non corrigé : la seconde bibliothèque libass de l'OSD ; la remplacer par ImGui déplace l'OSD dans la couche de l'interface (autre aspect, autre place dans la sortie HDR), à voir à l'écran, et son coût réel n'est pas mesuré |
| 60 | A24-23 | Mineur | `src/app/play/ui.cpp:117-129` | Polices de secours CJK différentes sous Windows et Linux | Corrigé d'après le code : sous Windows, Malgun Gothic (coréen) et Microsoft YaHei (chinois) s'ajoutent en repli après le japonais (`ui_fonts.h`, test de la table) ; sous Linux, une police du bureau qui couvre déjà le japonais n'est plus annoncée « aucune police CJK » |
| 61 | A24-24 | Mineur | `src/app/play/home_screen.cpp:307-316` | « Run the wizard again » à l'accueil n'ouvre que l'écran des moteurs | Corrigé d'après le code : l'accueil lance l'assistant entier et relit play.toml ; à voir à l'écran |
| 62 | A31-10 | Mineur | `src/app/play/host_facts.h:48-51` | Sur batterie, Auto plafonné à 60 Hz sans avis, et retenu | Corrigé : plafond retiré (décision 11) |
| 63 | A22-04 | Mineur | `src/app/play/settings.cpp:453-499` | `change_between` classe en direct des réglages lus à l'ouverture | Corrigé ; A08-03 (même cause) |

<a id="c09"></a>

## C09. Moteurs NVIDIA : TensorRT, CUDA, FP8, ONNX

Modèle : Opus 5.5 (effort high).

Mineurs de l'exécution TensorRT, des noyaux CUTLASS et de la construction des moteurs. Toute modification de src/core, src/gpu, src/kernels ou src/prepare demande un nouveau paquet Linux.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A37-05 | Mineur | `src/app/play/auto_config.cpp:176-180` | Qualité : baisse choisie par le nom de la carte en Auto estimé, sans mesure |  |
| 2 | A01-02 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:305-307` | `join()` au milieu d'une opération ordonnée : au changement de génération de l'anneau, une autre voie peut lire une sortie à moitié écrite |  |
| 3 | A01-03 | Mineur | `src/gpu/cuda/backend/host_upload.cpp:114` | Copie du décodage logiciel sur le thread de soumission, `cudaFree` synchrones au changement de taille |  |
| 4 | A01-04 | Mineur | `src/gpu/cuda/backend/tensorrt_stats.cpp:32-37` | Échec partiel d'allocation d'un emplacement de relecture jamais réinitialisé |  |
| 5 | A01-05 | Mineur | `src/gpu/cuda/backend/tensorrt_kernels.cpp:206-215` | Image NVDEC rendue au pool alors que l'ingestion la lit encore, si le débandage échoue |  |
| 6 | A01-06 | Mineur | `src/gpu/cuda/backend/tensorrt_textures.cpp:44` | Appels CUDA non vérifiés sur le chemin de lecture, `cudaGetLastError()` en aveugle |  |
| 7 | A01-07 | Mineur | `src/gpu/cuda/backend/pixel_kernels.cuh:23-38` | `block_reduce` sans `__syncthreads()` final (latent) |  |
| 8 | A01-08 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:141-147` | Sur sm_120, colle de RIFE fusionné bridée à deux blocs par SM, sans mesure |  |
| 9 | A02-02 | Mineur | `src/gpu/cuda/backend/span_fp8_kernels.cuh:358` | Mémoire partagée de W1 réservée dans les trois modes : noyaux FP16 au-delà de 48 Ko, un bloc par SM |  |
| 10 | A02-03 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:43-54` | Essai au chargement dans la zone d'activation partagée, sans ordre ni `set_last_user` : peut écraser les tampons de DistilDRBA |  |
| 11 | A02-04 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:57-60` | Sonde de 8 192 octets lue sans vérifier la taille de la sortie |  |
| 12 | A02-05 | Mineur | `src/gpu/cuda/backend/span_fp8_kernels.cuh:103` | Noyaux dans un espace de noms nommé : souches pliées entre deux `.cu`, attribut d'opt-in fragile |  |
| 13 | A02-06 | Mineur | `src/gpu/span_fp8/onnx_weights.cpp:119-126` | Type d'un poids ONNX non vérifié : lecture hors bornes sur un initialiseur entier |  |
| 14 | A02-07 | Mineur | `src/gpu/span_fp8/onnx_weights.cpp:236-242` | Lecture du fichier ONNX sans plafond, `tellg` à −1 non traité |  |
| 15 | A02-08 | Mineur | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:104` | Emplacement de l'anneau jamais signalé si to_pixel échoue après `begin_slot` | Réfuté par 01 (complément) : l'échec arrête l'étage avant de publier l'image |
| 16 | A02-09 | Mineur | `src/gpu/span_fp8/derive.h:17` | `kE4m3Max` défini deux fois ; la quantification testée n'est pas celle qui tourne |  |
| 17 | A02-11 | Mineur | `src/gpu/cuda/backend/span_fp8_run.cpp:88-100` | `ajp_span_fp8_run` recopie `span_fp8_view` et oublie `acc16` |  |
| 18 | A02-12 | Mineur | `src/gpu/cuda/backend/span_fp8_engine.cpp:96` | Quatre `path::string()` contre la règle de la CI | Dette (34) |
| 19 | A02-13 | Mineur | `tests/unit/gpu_span_fp8/metrics_test.cpp` | Aucun test par défaut de la concordance numérique FP8 ni des entrées non finies |  |
| 20 | A02-14 | Mineur | `src/app/play/span_derive.cpp:27-35` | Paquet `.span8` et reçu écrits sans fsync : après une coupure, refus ordinaire et Performance bloqué en FP16 (QUA-05) |  |
| 21 | A04-02 | Mineur | `src/prepare/onnx_split.cpp:125` | Produit des dimensions en `int64_t` qui déborde avant le plafond (comportement indéfini) |  |
| 22 | A04-03 | Mineur | `src/prepare/rife_split.cpp:210-231` | `analyze_rife` ne reconnaît pas toute la structure promise par l'en-tête |  |
| 23 | A04-04 | Mineur | `src/prepare/rife_split.cpp:381` | Noms des morceaux par `stem().string()` : exception sous Windows hors page de code | Dette (34) |
| 24 | A04-05 | Mineur | `src/prepare/model_source.cpp:36-43` | `escapes_folder` aveugle pour un `--model` sans dossier |  |
| 25 | A04-06 | Mineur | `src/prepare/model_source.cpp:71-111` | `read_model` ne vérifie pas les sorties (facteur, plans) avant une longue construction |  |
| 26 | A04-07 | Mineur | `src/prepare/rife_split.h:78` | Morceaux dérivés jamais réécrits tant que `fusion1` ne change pas |  |
| 27 | A04-08 | Mineur | `tests/unit/prepare/rife_split_test.cpp:461-499` | Aucun test de CI ne montre que la découpe garde le calcul du réseau entier (SEC-21) |  |
| 28 | A04-10 | Mineur | `src/prepare/onnx_scan.cpp:33-91` | Second décodeur protobuf, divergent de celui de `gpu/ir` |  |
| 29 | A04-11 | Mineur | `src/prepare/rife_split.cpp:30-90` | Doublons entre `rife_split`, `drba_split` et les utilitaires ONNX |  |
| 30 | A05-02 | Mineur | `src/prepare/key_lock.h:26` | Délais qui ne s'emboîtent pas : l'attente du verrou de clé abandonne avant la fin possible du détenteur |  |
| 31 | A05-03 | Mineur | `src/prepare/builder.cpp:86-93` | Aucun repli si le timing cache persistant fait échouer `trtexec` |  |
| 32 | A05-04 | Mineur | `src/prepare/cache_io.cpp:95` | Publication en un seul renommage : un antivirus sous Windows fait jeter la construction |  |
| 33 | A05-05 | Mineur | `src/prepare/trtexec_command.cpp:162-169` | Empreinte de `trtexec` vérifiée une fois à la création, repli de développement sans vérification |  |
| 34 | A05-06 | Mineur | `src/prepare/cache.cpp:52-53` | `intact()` écrit une empreinte manquante sans tenir le verrou |  |
| 35 | A05-07 | Mineur | `src/prepare/trtexec_progress.cpp:21-22` | Barre figée à 10 % pendant l'essentiel de la construction |  |
| 36 | A05-09 | Mineur | `src/prepare/builder.cpp:174` | L'ONNX relu et haché deux fois à chaque `build()`, cache compris |  |
| 37 | A21-02 | Mineur | `src/app/play/backend_library.cpp:55` | `path.string()` dans le chargeur : sans repli Vulkan sous Windows avec un profil non ASCII | Requalifié en mineur (dette) par 34 |
| 38 | A21-04 | Mineur | `src/gpu/guarded_backend.h:56-168` | Rien ne vérifie que `GuardedKernelOps` relaie toutes les virtuelles |  |
| 39 | A21-05 | Mineur | `src/app/play/backend_library.cpp:79-86` | `open` n'essaie pas les candidats suivants (latent) |  |
| 40 | A21-06 | Mineur | `src/gpu/kernel_ops.h:188-196` | `scores_take` attend toute la file sur une seule voie |  |
| 41 | A21-08 | Mineur | `src/gpu/graph_book.h:195-221` | Échecs des graphes CUDA cumulés sur toute la vie du moteur |  |
| 42 | A21-09 | Mineur | `src/gpu/lanes.h:113-115` | `LaneHazards` ne perd jamais une adresse |  |
| 43 | A30-02 | Mineur | `src/app/play/chain_hdr.cpp:153-163` | `LookLoader` détaché qui journalise après `close_log` : plantage à la sortie avec A18-01 (GPU-12) |  |
| 44 | A30-03 | Mineur | `src/app/play/settings_system.cpp:177-216` | Deux vérifications de mise à jour en parallèle sur la même archive |  |
| 45 | A30-04 | Mineur | `src/render/grade.cpp:316` | Thread des LUT en `std::jthread` brut, sans `try` (GPU-11) |  |
| 46 | A30-05 | Mineur | `src/app/play/pumped_task.h:47-58` | Sonde du fichier détachée d'office (QUA-13) |  |
| 47 | A30-06 | Mineur | `src/app/play/single_instance.cpp:155-157` | `ReadFile` sans délai sur le canal de l'instance unique (WIN-06) |  |
| 48 | A30-07 | Mineur | `tests/unit/app/` | Aucun test des durées de vie |  |
| 49 | A30-08 | Mineur | `src/media/shot_probe.cpp:108-110` | Attentes sans délai sur le thread de la fenêtre |  |
| 50 | A37-02 | Mineur | `src/app/play/home_screen.cpp:141` | FP8 présumé par le nom de la carte à l'accueil (RTX 20 et 30) |  |
| 51 | A37-03 | Mineur | `src/app/play/engine_install.cpp:97-115` | `cuda_arch_from_name` faux pour la RTX 2050 et une autre carte de la cible |  |
| 52 | A37-04 | Mineur | `data/gpu_tiers.toml:15-24` | Portables RTX 20 sans « Laptop » dans leur nom |  |
| 53 | A37-06 | Mineur | `src/app/play/auto_config.cpp:131-138` | Hauteur de travail calée sur l'écran, pas sur la source |  |
| 54 | A37-07 | Mineur | `src/app/play/auto_config.cpp:223-225` | Chaîne d'un préréglage refaite à chaque vidéo |  |
| 55 | A37-08 | Mineur | `src/app/play/first_run.cpp:245-253` | Pilote minimum de TensorRT vérifié seulement à l'écran des moteurs |  |
| 56 | A37-09 | Mineur | `src/app/play/auto_bench.cpp:207` | Deux mesures de la machine qui traitent la batterie différemment |  |
| 57 | A37-10 | Mineur | `src/app/play/machine_calibration.cpp:160-166` | Une seule calibration par machine (une carte, un écran) |  |
| 58 | A37-11 | Mineur | `src/app/play/first_run.cpp:220-230` | Pilote libre NVK pris pour le pilote NVIDIA |  |
| 59 | A37-12 | Mineur | `src/app/play/home_screen.cpp` | Aucun test des décisions par configuration |  |
| 60 | A37-13 | Mineur | `src/gpu/cuda/backend/tensorrt_lanes.cpp:25` | Nombre de voies décidé sur la mémoire totale, pas la libre |  |
| 61 | A37-14 | Mineur | `src/app/play/auto_config.cpp:55` | En Stable, le plafond ×4 coupe les paliers des écrans rapides |  |
| 62 | A03-05 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:175-194` | Essai au chargement sur un flux privé dans la zone d'activation partagée | A02-03 (même cause) |
| 63 | A33-07 | Mineur | `src/app/play/pumped_task.h:58` | Sonde détachée qui garde son descripteur | A30-05 (même cause) |
| 64 | A41-09 | Mineur | `src/render/grade.cpp:275-298` | Thread des LUT sans `try` | A30-04 (même cause) |
| 65 | A42-12 | Mineur | `src/media/shot_probe.cpp:108-110` | Sonde détruite sur le thread de la fenêtre | A30-08 (même cause) |

<a id="c10"></a>

## C10. Média : session, sources, décodage, audio, sous-titres

Modèle : Opus 5.5 ; Sonnet 5.5 pour les mineurs isolés.

IVTC, 4:4:4 sous NVDEC, PGS, atlas de sous-titres en 4K.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A15-02 | Majeur | `src/media/deinterlace.cpp:66` | Pas d'IVTC, et images NVDEC entrelacées non désentrelacées (MED-13) |  |
| 2 | A17-01 | Majeur | `src/subs/subtitle_renderer.cpp:199` | En 4K, un dessin plein écran dépasse l'atlas et laisse les sous-titres suivants à demi-définition (MED-05, démontré) |  |
| 3 | A13-03 | Mineur | `src/app/play/episodes.cpp:43-47` | Un titre d'épisode contenant `MENU`, `PV` ou `CM` est classé en bonus |  |
| 4 | A13-04 | Mineur | `src/app/play/episodes.cpp:74-93` | Chiffres pleine chasse comparés octet par octet |  |
| 5 | A13-05 | Mineur | `src/app/play/session.cpp:102-120` | Pas d'image vers l'avant bloqué en VFR (MED-16) |  |
| 6 | A13-06 | Mineur | `src/app/play/session.cpp:78` | Retours de `play()` et `pause()` ignorés : pause perdue sur boîte audio pleine |  |
| 7 | A13-07 | Mineur | `src/app/play/session.cpp:229-247` | Après des pas vers l'avant, l'audio reprend à la position de la pause |  |
| 8 | A13-08 | Mineur | `src/app/play/session.cpp:189-227` | Seek refusé sans repositionnement : lecture figée (MED-06) |  |
| 9 | A13-09 | Mineur | `src/app/play/player_run.cpp:738-739` | Pistes audio et sous-titres gardées pour toute vidéo suivante | Réfuté par 29 (`Player` recréé à chaque fichier) |
| 10 | A13-10 | Mineur | `src/app/play/resume.cpp:107-129` | `resume.tsv` réécrit sans verrou entre deux instances |  |
| 11 | A13-11 | Mineur | `src/app/play/resume.cpp:33` | `path::string()` dans la reprise et les sous-titres externes | Dette (34) |
| 12 | A13-12 | Mineur | `src/app/play/resume.cpp:39-45` | Windows : comparaison sans casse en ASCII seulement |  |
| 13 | A13-13 | Mineur | `tests/unit/app/resume_test.cpp` | Aucun test de la borne, des chemins non ASCII, du VFR ni de la fin sur erreur |  |
| 14 | A13-14 | Mineur | `src/app/play/session.cpp:691` | `--seconds` termine sans quitter : l'épisode suivant s'enchaîne |  |
| 15 | A14-03 | Mineur | `src/media/demuxer.cpp:273-283` | Chapitres ordonnés et éditions Matroska ni gérés ni détectés |  |
| 16 | A14-04 | Mineur | `src/media/source_loop.cpp:337-364` | Flush puis Seek refusé : les paquets jetés ne sont pas relus |  |
| 17 | A14-05 | Mineur | `src/media/track_language.cpp:44-46` | « VOSTFR » sur une piste audio lue comme du français |  |
| 18 | A14-06 | Mineur | `src/media/source.h:34-35` | File audio pleine qui fait attendre toute la lecture sur un fichier mal entrelacé |  |
| 19 | A14-07 | Mineur | `src/media/demuxer.cpp:369-377` | Panne d'E/S, interruption et données abîmées toutes en `DecodeFailed` |  |
| 20 | A14-08 | Mineur | `tests/unit/media/source_test.cpp:400-414` | Le test fige MED-01 comme comportement voulu |  |
| 21 | A14-09 | Mineur | `src/media/source_loop.cpp:153-159` | Seek exact : chaque image jetée est convertie avant la décision (jusqu'à 1,8 s en 4K 4:4:4 logiciel) |  |
| 22 | A15-03 | Mineur | `src/media/video_decoder.cpp:400` | Pas de repli logiciel sur une erreur NVDEC en cours de flux |  |
| 23 | A15-04 | Mineur | `src/media/deinterlace.cpp:47-49` | bwdif perd une image au changement de taille (démontré) |  |
| 24 | A15-05 | Mineur | `src/media/deinterlace.cpp:79` | bwdif perd le `best_effort_timestamp` (démontré) |  |
| 25 | A15-06 | Mineur | `src/media/frame_data.cpp:247-275` | Conversion hôte 4:4:4 sans pool : 7,5 ms et 25 Mo par image 4K (MED-03, mesuré) |  |
| 26 | A15-07 | Mineur | `src/media/frame_data.cpp:253` | Chroma 4:4:4 réduite avant SPAN, formats illisibles mal refusés (MED-04) |  |
| 27 | A15-08 | Mineur | `src/media/color.cpp:72` | Matrice non étiquetée devinée par la taille même en BT.2020 ou PQ |  |
| 28 | A15-10 | Mineur | `src/media/video_decoder.cpp:167` | `pts + duration` sans garde (MED-14) |  |
| 29 | A15-11 | Mineur | `src/media/stills.cpp:295` | `extract_stills` décode jusqu'à 8 threads à priorité normale |  |
| 30 | A15-12 | Mineur | `src/media/stills.cpp:230-236` | Corpus de calibration FP8 à chroma au plus proche, différente de la lecture |  |
| 31 | A15-13 | Mineur | `tests/unit/media/` | Sans test : conversion hôte, bascule NVDEC, chemin entrelacé |  |
| 32 | A16-02 | Mineur | `src/audio/player.cpp:138-151` | Délai de l'appareil retiré gardé sur la sortie par défaut (MED-08) |  |
| 33 | A16-03 | Mineur | `src/audio/sdl_sink.cpp:100-113` | Changement de sortie par défaut invisible au module |  |
| 34 | A16-04 | Mineur | `src/audio/sdl_sink.cpp:189-199` | Latence du périphérique comptée sur un seul tampon |  |
| 35 | A16-05 | Mineur | `src/audio/pcm.h:25` | Correction de dérive bornée à 0,125 %, au ras du calage de vitesse |  |
| 36 | A16-06 | Mineur | `src/audio/packet.h:41-44` | `AudioTrack` sans époque : paquets de l'ancienne piste après un changement |  |
| 37 | A16-07 | Mineur | `src/app/play/pipeline.cpp:97-110` | Sortie ouverte au format de la première piste |  |
| 38 | A16-08 | Mineur | `src/audio/converter.cpp:132-135` | Mixage surround vers stéréo normalisé : son plus faible |  |
| 39 | A16-09 | Mineur | `src/audio/player.cpp:349-364` | Avec la PLL, `set_delay` décale l'horloge avant le son |  |
| 40 | A16-10 | Mineur | `src/audio/output.cpp:29-32` | Aucun contrôle de continuité des `pts` audio |  |
| 41 | A16-13 | Mineur | `src/audio/player.h:55` | PLL éteinte par défaut dans les tests, active dans le produit |  |
| 42 | A16-14 | Mineur | `src/app/play/wiring.cpp:103-108` | Seek exact raté puis position rattrapée : audio rogné jusqu'à la cible manquée (gel de 1,5 s, silence) |  |
| 43 | A17-03 | Mineur | `src/subs/text_encoding.cpp:212-216` | Indice de langue qui impose son encodage, repli direct sur Windows-1252 (MED-11) |  |
| 44 | A17-04 | Mineur | `src/subs/text_encoding.cpp:95-102` | UTF-16 sans BOM non reconnu, demi-paires recopiées |  |
| 45 | A17-05 | Mineur | `src/subs/subtitle_track.cpp:137-161` | Pas de pré-lecture des sous-titres au seek |  |
| 46 | A17-06 | Mineur | `src/subs/subtitle_renderer.cpp:221` | Délai négatif sans paquets lus en avance (MED-09) |  |
| 47 | A17-07 | Mineur | `src/subs/subtitle_track.cpp:118-122` | Une seule cue SRT refusée fait refuser tout le fichier |  |
| 48 | A17-08 | Mineur | `src/subs/subtitle_renderer.cpp:229-234` | Un rendu libass par image de sortie, après l'interpolation |  |
| 49 | A17-09 | Mineur | `src/subs/ass_library.cpp:53-62` | Polices jointes sans plafond total |  |
| 50 | A17-10 | Mineur | `src/subs/subs_thread.cpp:50` | Initialisation de libass sur le thread de la fenêtre |  |
| 51 | A17-11 | Mineur | `src/app/play/external_subtitles.cpp:123-124` | Fichiers externes coupés à 16 avant le choix par langue |  |
| 52 | A17-12 | Mineur | `src/subs/subtitle_track.cpp:27` | `path::string()` dans les sous-titres | Dette (34) |
| 53 | A17-13 | Mineur | `tests/unit/subs/` | Sans test : retour à pleine taille, encodages coréen et EUC-JP |  |
| 54 | A17-14 | Mineur | `src/subs/subtitle_track.cpp:146-149` | Aucun plafond cumulé des paquets d'une piste intégrée : MKV forgé, plusieurs Gio dans libass |  |
| 55 | A14-10 | Mineur | `src/media/source_loop.cpp:21-25` | `ends_before` jette les images sans PTS sans borne : seek sans fin | A15-05 (même cause) |
| 56 | A15-09 | Mineur | `src/media/video_decoder.cpp:166-167` | Durée d'image par la période moyenne sans durée de paquet (MED-16) | A13-05 (même cause) |
| 57 | A40-05 | Mineur | `tests/unit/app/e2e_play_test.cpp:119-124` | Les cas e2e jouent tous les clips du dossier | A13-14 (même cause) |

<a id="c11"></a>

## C11. Performance et ressources

Modèle : Opus 5.5.

Mesurer avant de changer : toute accélération qui touche l'image passe par le banc du C03.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A18-02 | Majeur | `src/app/play/player.cpp:130-139` | `engines-nvidia.lock` téléchargé par curl sur le thread de la fenêtre à chaque lancement, jusqu'à 8 s figée |  |
| 2 | A23-04 | Majeur | `src/app/play/settings_system.cpp:107-108` | Page Application : lecture d'`update.toml` et fichier d'essai créé et effacé à chaque image |  |
| 3 | A32-01 | Majeur | `docs/auto-libre.md:142-143` | Le service de 13 à 16 ms date de la chaîne avec RTX HDR : à remesurer ; 1,7 à 2,7 ms de GPU estimés par image à 1440p |  |
| 4 | A32-02 | Majeur | `src/gpu/cuda/backend/tensorrt_backend.cpp:407-498` | `gpu_profile` ne voit ni l'écran, ni les sémaphores, ni les attentes hôte |  |
| 5 | A33-01 | Majeur | `src/app/play/placebo_ring.cpp:84` | Mémoire vidéo d'une chaîne fermée gardée (anneau, réserve CUDA) ; « Refaire le banc » mesure avec moins de mémoire (TRT-06 élargi) |  |
| 6 | A33-04 | Majeur | `src/prepare/cache.cpp:81` | Cache des moteurs TensorRT sans aucune purge |  |
| 7 | A34-02 | Majeur | `src/media/shot_probe.cpp:165-175` | Pré-analyse sur un thread sous Windows : grade et coupes privés sur les sources lourdes (à confirmer) | Réfuté en grande partie par 42 (NVDEC par défaut, seul le repli logiciel est touché) |
| 8 | A46-07 | Majeur | `tools/upscale/models/2x_ArtCNN_C4F32_rgb_fp16.onnx` | ArtCNN « fp16 » calcule entièrement en FP32 (TRT-07, partie modèle) |  |
| 9 | A18-03 | Mineur | `src/app/play/player.cpp:183-213` | Le thread des mises à jour part avant l'assistant avec l'ancien mode |  |
| 10 | A18-04 | Mineur | `src/app/play/player.cpp:525-534` | « Annuler et regarder » finit écrit dans `play.toml` |  |
| 11 | A18-05 | Mineur | `src/app/main.cpp:390-393` | Purge récursive des `*.old` dans tout le dossier d'installation |  |
| 12 | A18-06 | Mineur | `src/app/main.cpp:452-454` | `--bake-looks` cherche les LUT dans l'ancien dossier |  |
| 13 | A18-07 | Mineur | `src/app/main.cpp:337-343` | `SDL_GetPrefPath` en échec non arrêté |  |
| 14 | A18-08 | Mineur | `src/app/play/options.cpp:466-508` | Aide fausse ou incomplète sur les options et touches |  |
| 15 | A18-09 | Mineur | `src/app/play/options.cpp:236-243` | Aucune option pour choisir DistilDRBA en ligne de commande |  |
| 16 | A18-10 | Mineur | `src/app/main.cpp:459-461` | Deux sorties sans trace au journal |  |
| 17 | A18-11 | Mineur | `src/app/main.cpp:247-259` | `ajp calibrate --display-height` remplacé en silence |  |
| 18 | A18-12 | Mineur | `src/app/main.cpp:147-150` | `execv` garde les descripteurs ouverts |  |
| 19 | A18-13 | Mineur | `src/app/play/player.cpp:445-452` | Échec d'`open_backend` depuis l'accueil : le lecteur quitte |  |
| 20 | A18-14 | Mineur | `src/app/play/mapped_file.cpp:229-247` | Police tronquée entre la taille lue et la projection |  |
| 21 | A18-16 | Mineur | `src/app/main.cpp:371-401` | Ni minidump ni journal pendant l'application d'une mise à jour |  |
| 22 | A18-17 | Mineur | `src/app/play/settings.cpp:326-343` | `--rife-model` absent de `settings.rife_model` : menu et `--bench-out` donnent le mauvais modèle |  |
| 23 | A32-05 | Mineur | `src/render/grade.cpp:355` | Passes d'écran du grade à fusionner |  |
| 24 | A32-06 | Mineur | `src/gpu/cuda/backend/warp_kernels.cu:100-170` | Warp en plans séparés : une vingtaine de lectures par pixel |  |
| 25 | A32-07 | Mineur | `src/app/play/placebo_ring.cpp:214-218` | `ready()` interroge le pilote à chaque question |  |
| 26 | A32-08 | Mineur | `src/core/free_rate.cpp:27-36` | Pas du mode Rapide calé sur un temps de service en temps de mur |  |
| 27 | A32-10 | Mineur | `src/gpu/cuda/backend/trt_engine.cpp:140-165` | Mémoire vidéo pleine sous Windows : débordement en mémoire système ni refusé ni signalé |  |
| 28 | A32-11 | Mineur | `src/gpu/cuda/backend/tensorrt_lanes.cpp:17-19` | Attente active au-delà de 200 µs : événements sans `cudaEventBlockingSync` |  |
| 29 | A32-12 | Mineur | `src/app/play/upscale_drba.cpp:124-157` | DistilDRBA relit sa mesure dans la paire même : la file se vide à chaque paire |  |
| 30 | A32-13 | Mineur | `src/render/shaders/anime-hdr-anime/3-anime.hook:27` | Quatre `pq_of` de constantes recalculés par pixel (environ 0,09 ms par présentation) |  |
| 31 | A33-05 | Mineur | `src/gpu/cuda/backend/gpu_timing.h:53` | Profil GPU de fin de lecture tronqué |  |
| 32 | A33-06 | Mineur | `src/app/main.cpp:427-440` | Deux rotations du journal qui s'ignorent |  |
| 33 | A33-08 | Mineur | `src/app/play/gpu_telemetry.cpp` | Aucun essai de durée ni relevé de ressources |  |
| 34 | A26-08 | Mineur | `src/app/play/updater.cpp:490-497` | `update_managed_reason` crée et efface un fichier à chaque appel | A23-04 (même cause) |

<a id="c12"></a>

## C12. Interface, réglages et assistant

Modèle : Sonnet 5.5 (effort high).

Réglages du menu enfin appliqués et enregistrés, assistant qui n'écrase plus la chaîne, avis lisibles.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A22-01 | Majeur | `src/app/play/settings.cpp:383-434` | UX-01 plus large : `restore_lines`, `gpu`, `rife_fused`, `drba_fused` ni appliqués ni écrits |  |
| 2 | A23-02 | Majeur | `src/app/play/settings_chain_tabs.cpp:375-383` | « Warp and SPAN » grisé pour DistilDRBA alors qu'il décide sa variante |  |
| 3 | A24-01 | Majeur | `src/app/play/controls.cpp:128-131` | Un seul message de 1,6 s : les avis de qualité perdue (repli DistilDRBA vers RIFE) disparaissent avant d'être lus |  |
| 4 | A24-03 | Majeur | `src/app/play/ui.cpp:325-343` | Aucun presse-papiers système hors Windows : « Signaler un problème » copie du vide sous CachyOS |  |
| 5 | A25-03 | Majeur | `src/app/play/engine_screen.cpp:223-227` | UX-05 : durée d'optimisation annoncée fausse (4 min au lieu de 17 à 1080p) |  |
| 6 | A25-04 | Majeur | `src/app/play/engine_screen.cpp:286-291` | UX-06 : lecteur par défaut appliqué même quand l'écran Intégration a été passé |  |
| 7 | A23-01 | Majeur | `src/app/play/settings_chain_tabs.cpp:84` | « Restaurer les traits » et la carte graphique jamais appliqués ni enregistrés (démontré) | A22-01 (même cause) |
| 8 | A22-14 | Mineur | `src/app/play/settings.h:180` | Qualité : réduction Hermite par défaut, plus douce que Spline36 annoncé le plus fin |  |
| 9 | A43-06 | Mineur | `src/app/play/i18n_fr_image.cpp:88` | Qualité : le coût en précision du FP8 n'est jamais dit ; « Balanced » nomme deux choses |  |
| 10 | A22-05 | Mineur | `src/app/play/settings.cpp:336` | `--rife` coupe `rife_auto`, qui n'est pas rendu à l'écriture |  |
| 11 | A22-06 | Mineur | `src/app/play/settings_toml.cpp:501-534` | Valeur inconnue d'une clé connue écrasée à l'écriture suivante |  |
| 12 | A22-07 | Mineur | `src/core/settings.cpp:231-233` | `settings.toml` refusé entier pour une version différente |  |
| 13 | A22-08 | Mineur | `src/core/settings.h:55-58` | Langues et délai audio de `settings.toml` sans lecteur |  |
| 14 | A22-09 | Mineur | `src/app/play/settings.cpp:266-324` | `engines_dir` non validé, écrit à chaque frappe |  |
| 15 | A22-11 | Mineur | `src/app/play/settings.h:299-310` | Trois bornes différentes par délai (UX-18) |  |
| 16 | A22-12 | Mineur | `src/app/play/settings.cpp:481-490` | Un cran de précision hors Équilibre coupe le moteur fusionné en silence |  |
| 17 | A22-13 | Mineur | `src/app/play/settings_toml.cpp:570` | Cinq `path::string()` (dette, voir A20-02) |  |
| 18 | A22-15 | Mineur | `tests/unit/app/settings_test.cpp` | Aucun test que chaque champ est appliqué ou annoncé ; ni écriture refusée |  |
| 19 | A22-16 | Mineur | `src/app/play/settings_toml.cpp:593-627` | Une écriture de `play.toml` fait quatre analyses TOML et ~90 relectures |  |
| 20 | A23-05 | Mineur | `src/app/play/settings_panel.cpp:290-301` | « Tout réinitialiser » sans confirmation, langue remise en anglais |  |
| 21 | A23-06 | Mineur | `src/app/play/settings_panel.h:53-54` | Aucun contrôle pour `motion_precision`, qui peut couper le fusionné en silence |  |
| 22 | A23-07 | Mineur | `src/app/play/settings_chain_tabs.cpp:363-374` | Le panneau ne montre pas l'état réel de DistilDRBA (fusionné, variante, refus) |  |
| 23 | A23-08 | Mineur | `src/app/play/settings_chain_tabs.cpp:63-64` | « DistilDRBA : le plus léger » faux au-delà de ×5 |  |
| 24 | A23-09 | Mineur | `src/app/play/settings_chain_tabs.cpp:213` | Plafond d'Auto caché, qui dépend du chemin suivi |  |
| 25 | A23-10 | Mineur | `src/app/play/settings_panel.cpp:242` | Deux « Niveau » et trois « Rapide » sur une même page |  |
| 26 | A23-11 | Mineur | `src/app/play/settings_sound.cpp:72-77` | Délai audio : infobulle ±10 000 ms, borne ±5 000 |  |
| 27 | A23-12 | Mineur | `src/app/play/settings_panel.cpp:199` | Délai des sous-titres appliqué à chaque image du glissement |  |
| 28 | A23-13 | Mineur | `src/app/play/settings_export.cpp:68` | Page Export : quatre appels au système de fichiers par image |  |
| 29 | A23-14 | Mineur | `src/app/play/settings_system.cpp:495-506` | « Signaler un problème » coupe le journal au milieu d'une ligne, sans le dire |  |
| 30 | A23-16 | Mineur | `src/app/play/settings_engines.cpp:75-85` | Annuler et Rétablir changent `engines_dir` sans mettre à jour le champ ; chemin long coupé à 512 octets sans le dire |  |
| 31 | A43-03 | Mineur | `src/app/play/i18n.h:30-33` | `tr_format` ne localise pas les nombres : point en français, virgule en anglais (UX-10) |  |
| 32 | A43-04 | Mineur | `src/app/play/i18n_fr_wizard.cpp:11` | Cinq noms pour le test de la machine qui règle Auto |  |
| 33 | A43-05 | Mineur | `src/app/play/settings_panel.cpp:283` | Infobulle « Finish » alors que l'assistant n'a pas ce bouton |  |
| 34 | A43-07 | Mineur | `src/app/play/french_text.h:12-30` | Typographie française appliquée à moitié |  |

<a id="c13"></a>

## C13. Diagnostic et signalement

Modèle : Sonnet 5.5.

Rapport utile (système, pilote, préréglage, refus), chemins anonymisés, journal du plantage.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A39-01 | Majeur | `src/app/main.cpp:424-433` | Après un plantage, « Report a problem » copie la nouvelle séance, pas le journal du plantage |  |
| 2 | A39-02 | Majeur | `src/app/play/settings_system.cpp:498` | Le rapport ne dit ni le système, ni le pilote, ni le préréglage, ni la raison du refus de TensorRT |  |
| 3 | A39-03 | Majeur | `src/app/play/chain.cpp:352-353` | UX-13 : chemins complets des vidéos dans le journal envoyé |  |
| 4 | A39-04 | Majeur | `src/app/play/crash_dump.cpp:47-50` | WIN-07 : minidumps sans PDB, jamais signalés, rien sous Linux |  |
| 5 | A38-01 | Majeur | `src/app/play/settings_system.cpp:493-511` | « Signaler un problème » copie les chemins complets hors du dossier personnel, nom d'utilisateur compris | A39-03 (même cause) |
| 6 | A39-07 | Mineur | `src/app/play/chain_setup.cpp:352-353` | Qualité : cran Qualité sans FP32 retombe en silence |  |
| 7 | A38-09 | Mineur | `THIRD_PARTY_NOTICES.md:40-49` | Notices tierces incomplètes |  |
| 8 | A38-10 | Mineur | `tools/release/notes/v0.10.3.md:3` | Notes 0.10.3 inexactes |  |
| 9 | A38-11 | Mineur | `src/app/play/options.cpp:470` | `--help` toujours faux |  |
| 10 | A38-12 | Mineur | `src/app/play/settings_chain_tabs.cpp:260-262` | Deux infobulles du panneau fausses |  |
| 11 | A38-15 | Mineur | `src/render/shaders/sharpen.hook:1-5` | Netteté promise « sans halo » |  |
| 12 | A39-05 | Mineur | `src/app/main.cpp:458` | Messages de FFmpeg absents d'`ajp.log` |  |
| 13 | A39-06 | Mineur | `src/app/play/chain_stats.cpp:167-174` | Télémétrie GPU absente du journal |  |
| 14 | A39-08 | Mineur | `.github/ISSUE_TEMPLATE/bug.yml` | Formulaire de bogue avec des niveaux périmés |  |
| 15 | A39-09 | Mineur | `src/app/play/settings_system.cpp:499` | Le bouton lit `ajp.log` au mauvais endroit |  |
| 16 | A39-10 | Mineur | `src/prepare/builder.cpp:233` | Disque plein montré comme « BuilderFailed » alors que le code « not enough disk space » existe |  |
| 17 | A38-14 | Mineur | `.github/ISSUE_TEMPLATE/bug.yml:56-60` | Formulaire sans Économe ni Personnalisé | A39-08 (même cause) |

<a id="c14"></a>

## C14. Dette et documentation

Modèle : Haiku 5.5 pour les textes et commentaires ; Sonnet 5.5 pour retirer le code mort.

Code mort, commentaires et pages fausses. Aucun risque, à faire en fin de file ou entre deux chantiers.

| Rang | ID | Gravité | Où | Défaut | Note |
|---|---|---|---|---|---|
| 1 | A49-15 | Mineur | `docs/nvidia-linux.md:45` | La page annonce le FP8 de SPAN dans les préréglages ; le code pose le FP16 hors Auto calibré |  |
| 2 | A02-10 | Mineur | `src/gpu/cuda/backend/span_fp8_net.cu:248` | Contrats divergents : 1 bloc accepté à la lecture, refusé par les noyaux ; commentaire sm 8.9 seulement |  |
| 3 | A04-09 | Mineur | `src/prepare/onnx_metadata.cpp` | Code mort depuis le retrait de GMFSS |  |
| 4 | A05-08 | Mineur | `src/prepare/preparer.cpp` | Environ 700 lignes de code mort (`preparer`, `receipt`, `rebuild`, `rejected_sha256`…) |  |
| 5 | A06-11 | Mineur | `src/core/model_index.h:61-69` | Code mort de l'index des modèles ; MIGraphX, OpenVINO et DirectML encore « téléchargeables » |  |
| 6 | A07-12 | Mineur | `src/app/play/auto_bench_run.cpp:53` | Code mort et deux mécanismes de banc côte à côte |  |
| 7 | A09-15 | Mineur | `src/app/play/rife_plan.h:7` | Commentaires faux |  |
| 8 | A10-12 | Mineur | `src/app/play/present_level.h:426` | Commentaires faux sur le niveau de présentation par défaut |  |
| 9 | A10-16 | Mineur | `src/render/shaders/anime-hdr-look.hook:100` | Commentaire périmé |  |
| 10 | A12-09 | Mineur | `src/core/steady_rate.h:37` | Code mort dans les horloges |  |
| 11 | A15-14 | Mineur | `src/media/hw_device.cpp:204` | Code mort pour le produit |  |
| 12 | A16-11 | Mineur | `src/audio/devices.cpp:15-18` | Code mort, appareils désignés par leur seul nom |  |
| 13 | A16-12 | Mineur | `src/audio/player.h:5-13` | Commentaires faux : l'audio est maître sur tout écran, pas seulement sous VRR |  |
| 14 | A21-10 | Mineur | `src/gpu/kernel_ops.h:135` | Virtuelles pures mortes et commentaires périmés dans le contrat |  |
| 15 | A21-11 | Mineur | `src/gpu/kernel_ops.h:49-53` | `IngestSpec::rife_conversion` morte, encore traitée par les deux moteurs (TensorRT la refuse avec un message trompeur) |  |
| 16 | A22-10 | Mineur | `src/app/play/settings.h:186-191` | Commentaire faux : en Rapide, `rife_factor` ne plafonne rien |  |
| 17 | A22-17 | Mineur | `src/app/play/settings.h:305-307` | Commentaire du seuil HDR posé au-dessus de la mauvaise constante |  |
| 18 | A22-18 | Mineur | `src/app/play/settings.cpp:130-138` | Valeurs et gardes GMFSS mortes (aucune n'atteint `PlaySettings`) |  |
| 19 | A23-15 | Mineur | `src/app/play/settings_panel.cpp:310-320` | Drapeau `advanced` mort, en-têtes périmés |  |
| 20 | A25-17 | Mineur | `src/app/play/first_run.cpp:138-147` | Code mort : MIGraphX, OpenVINO, DirectML encore nommés |  |
| 21 | A28-21 | Mineur | `docs/auto-libre.md:140-174` | § 3 non daté, ligne GMFSS, rangé parmi les documents à jour |  |
| 22 | A32-09 | Mineur | `docs/handoff-pc.md:77-78` | Temps de référence périmés |  |
| 23 | A34-06 | Mineur | `src/app/play/auto_bench_run.cpp:48-61` | `hide_is_reliable*` morts |  |
| 24 | A38-03 | Mineur | `README.md:74-83` | Tableau des niveaux faux |  |
| 25 | A38-04 | Mineur | `README.md:29-31` | « Une source 1080p devient 4K » faux en Auto |  |
| 26 | A38-05 | Mineur | `README.md:57-66` | « Premier lancement » décrit un assistant qui n'existe plus |  |
| 27 | A38-06 | Mineur | `README.md:140` | Promesse fausse sur la construction des moteurs au premier lancement |  |
| 28 | A38-07 | Mineur | `README.md:129` | Raccourcis « dans `user.toml` » : quinze touches en dur |  |
| 29 | A38-08 | Mineur | `README.md:4` | Badge 0.10.0, code 0.10.3, CHANGELOG arrêté |  |
| 30 | A38-13 | Mineur | `src/app/play/auto_config.h:19-22` | Commentaire qui renvoie à une porte de qualité DistilDRBA inexistante |  |
| 31 | A40-16 | Mineur | `tools/ci/check-cuda.sh:2-6` | Commentaire faux sur la construction CUDA en CI |  |
| 32 | A41-10 | Mineur | `src/render/grade_params.cpp:56-70` | Chemins morts du grade SDR, quinze emplacements de LUT, pile composée deux fois (dette) |  |
| 33 | A42-15 | Mineur | `src/app/play/chain_shots.cpp:44` | `path::string()` (dette) |  |
| 34 | A43-02 | Mineur | `src/app/play/i18n_fr_sound.cpp` | 192 entrées de traduction mortes sur 946 (UX-21) |  |
| 35 | A44-05 | Mineur | `src/gpu/ir/import.cpp:238` | `path.string()` dans les messages d'erreur (dette) |  |
| 36 | A44-08 | Mineur | `src/gpu/ir/onnx_io_type.cpp:648` | `with_float32` mort depuis le retrait de VitisAI (dette) |  |
| 37 | A44-09 | Mineur | `src/gpu/ir/describe.cpp:117` | `describe_stats` n'affiche pas les quantifications FP8 (dette) |  |
| 38 | A45-09 | Mineur | `src/gpu/vulkan/compute/library.cpp:43-99` | `VulkanLibrary` recopie `DynamicLibrary` (dette) |  |
| 39 | A45-10 | Mineur | `src/gpu/vulkan/backend/entry_points.h:1-6` | Commentaires et code d'interop MIGraphX, OpenVINO, HIP morts (dette) |  |
| 40 | A46-06 | Mineur | `src/app/play/model_choice.h:47-49` | Commentaire faux sur l'origine des RIFE FP16 livrés |  |
| 41 | A08-14 | Mineur | `src/app/play/engine_switch.cpp:13-19` | L'onglet Moteurs connaît encore MIGraphX et OpenVINO | A06-11 (même cause) |
