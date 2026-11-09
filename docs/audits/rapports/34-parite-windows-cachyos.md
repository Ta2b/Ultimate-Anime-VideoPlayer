# Rapport de l'audit 34. Parité Windows et CachyOS

Branche : `claude/project-thread-r9406v`. Commit audité : `c655a7a`. Date : `2026-11-09`.

## Verdict

Sur une RTX, les deux systèmes ne jouent pas la même image : l'Auto calibré joue un autre SPAN sous Windows (Performance
b5f48 en FP16) que sous CachyOS (Balanced b8f64 en FP8 CUTLASS), et la pré-analyse des plans, qui guide le grade et les
coupes de l'interpolation, tourne sur un seul thread sous Windows. En revanche, plusieurs défauts classés « Windows »
ne le sont pas : le repli de Performance pour RIFE et DistilDRBA (A31-06) touche les deux systèmes de la même façon, et
les défauts `path::string()` ne lèvent pas d'exception sous Windows 11 grâce au manifeste UTF-8.

## Cadrage retenu

Pas de questions posées à Tatoub, sur consigne du coordinateur : l'audit couvre tout au maximum. Il retient ce qui
suit :

- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ;
- toute baisse de qualité d'image faite pour la vitesse est un défaut, sauf si une mesure la justifie ;
- le VRR est hors sujet : `window_vrr.cpp` et la lecture du VRR dans `display_edid.cpp` ne sont pas audités ;
- les blocs `__APPLE__` sont vérifiés seulement pour ce qu'ils changent sous Windows et Linux (rien trouvé) ;
- le Vulkan ne compte que comme repli : les sémaphores de `shared_tensor.cpp` et `vulkan_share.cpp` sont vus de haut.

Le fil tourne sans GPU NVIDIA ni Windows. Tout ce qui demande une mesure est marqué « à confirmer », avec la
manipulation à faire.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A34-01 | Majeur | `src/gpu/cuda/backend/CMakeLists.txt:27-31`, `.github/workflows/release-windows.yml:16-19`, `src/app/play/machine_calibration.cpp:279-330`, `:252`, `src/app/play/machine_calibration.h:38-41`, `src/app/play/auto_config.cpp:192-200` | **L'Auto calibré ne joue pas le même SPAN sous Windows et sous Linux.** Les noyaux CUTLASS ne sont construits que sous Linux. La release Windows sur étiquette garde `span_fp8 = false`, donc `span_fp8_cutlass` reste faux. La calibration essaie d'abord le FP8 : sous Windows, l'essai rend une autre variante que `"FP8 CUTLASS"`, il est refait et le reçu écrit `span_fp8 = false`. `out.model` reste alors vide : le lecteur joue le modèle par défaut, **Performance Sharp b5f48 en FP16**, au cran Équilibre. Sous Linux, il joue le **Balanced Sharp b8f64 en FP8**, au cran Performance. Ce n'est ni le même réseau, ni la même précision, ni le même budget, donc la calibration choisit des hauteurs de travail différentes. Sous Windows, le cran Équilibre prend en plus une variante INT8 dès qu'un reçu en existe (A49-07) : un troisième résultat possible. Le premier essai de calibration est aussi perdu sous Windows, puisqu'il demande une précision que la machine n'a pas. | Même PC en double démarrage, RTX 4070 Ti, épisode 1080p, Auto calibré : CachyOS joue b8f64 FP8 à une certaine hauteur, Windows 11 joue b5f48 FP16 (ou INT8) à une autre. Les comparaisons de qualité et de fluidité entre les deux systèmes mesurent deux chaînes différentes. | Une seule politique pour les deux systèmes. Pour la qualité, la plus sûre : la calibration mesure les candidats en FP16 (Balanced et Performance) des deux côtés, le FP8 restant un choix manuel tant que sa porte n'est pas jugée (A49-03, ADR 0007). Sinon, construire CUTLASS dans la release Windows une fois le critère de `docs/span-fp8-cutlass.md` mesuré (PSNR et gain à ±3 % de Linux). Dans les deux cas, `candidate_settings` ne doit plus essayer le FP8 sur une machine qui ne l'annonce pas (`caps.span_fp8_cutlass`), pour ne pas perdre le premier essai. | Lu. À confirmer : `ajp calibrate` sur les deux systèmes du même PC, puis comparer la ligne `SPAN …` imprimée (`main.cpp:280`) et la chaîne retenue. |
| A34-02 | Majeur | `src/media/shot_probe.cpp:165-175` | **La pré-analyse des plans tourne sur un thread sous Windows, contre 2 à 4 sous Linux**, toujours en décodage logiciel de chaque image (`attempts = {HwApi::Software}`). Elle donne les plans du grade HDR, l'APL de la série (`chain_shots.cpp:170-173`) et, par défaut, les coupes de l'interpolation (`options.h:132`, `shot_cuts = true`). Quand la sonde ne suit pas la lecture, `next_shot` retombe sur les scènes vues en direct (`chain_shots.cpp:177-180`). La raison du thread unique est vraie : sous Windows, les threads de FFmpeg n'héritent pas de la priorité basse. Mais on paie cette raison en qualité (grade sans plan, coupes sans plan) sur les sources lourdes. Ce choix n'est appuyé par aucune mesure. | Windows 11, épisode HEVC 10 bits en 4K (ou 1080p sur un CPU modeste) : sur un cœur, le décodage logiciel peut rester sous le temps réel. La frontière de la sonde ne dépasse jamais la tête de lecture, et tout l'épisode est gradé et coupé sur les scènes en direct, alors que CachyOS a les plans en avance. | Ne plus payer la priorité en qualité. (1) Décoder la sonde par NVDEC, une seconde session presque gratuite sur la RTX, réduite par `cuvid_resize` qui existe déjà dans les options. (2) Sinon, sous Windows, sonder dans un processus enfant en `BELOW_NORMAL_PRIORITY_CLASS` (ou EcoQoS) avec 2 à 4 threads. (3) Au minimum, `skip_loop_filter` et `skip_frame = NONREF` pour la sonde. Mesurer la frontière contre la tête de lecture sur les deux systèmes. | Lu. À confirmer : sous Windows et sous CachyOS, la même vidéo HEVC 10 bits en 4K, puis comparer dans `ajp.log` la ligne `pré-analyse des plans : N threads` et la progression de la frontière contre la position, sur la première minute. |
| A34-03 | Mineur | `src/app/play/single_instance.cpp:114-120`, `:211-235` | Le canal nommé de l'instance unique a un nom prévisible : `\\.\pipe\ajp-` suivi du FNV du dossier de réglages. Il est créé avec la sécurité par défaut, dans l'espace global des canaux, et le client ne vérifie pas qui le tient. Sous Linux, le socket est dans le dossier de l'utilisateur, en `0600` (`:404`), et personne d'autre ne peut le prendre. | Poste Windows partagé : un autre compte, ou un processus d'intégrité basse, crée d'abord le canal `ajp-<hash>` (le chemin `C:\Users\<nom>\AppData\Roaming\ajp\ajp\` se devine). Le lecteur de la victime reçoit `ERROR_ACCESS_DENIED`, envoie le chemin de sa vidéo au faux serveur, reçoit « ok » et se ferme sans rien montrer. | Mettre le SID de l'utilisateur et l'identifiant de session dans le nom, poser un descripteur de sécurité qui ne donne l'accès qu'à ce SID, et vérifier côté client le propriétaire du serveur (`GetNamedPipeServerProcessId`, puis le jeton du processus). | Lu. |
| A34-04 | Mineur | `src/app/play/default_player.cpp:203`, `:224`, `src/app/play/options.cpp:153-158`, `src/app/play/single_instance.cpp:152-169`, `:345-352` | Ouvrir plusieurs épisodes lance un processus par fichier sous les deux systèmes : `"%1"` dans le registre, `%f` dans le `.desktop`, et `parse_play_options` n'accepte qu'un fichier. Les deux écouteurs accusent réception puis restent dans `push_files` tant que SDL n'est pas prêt. Sous Windows, le canal n'a qu'une instance (`nMaxInstances = 1`) et ne réécoute qu'après ce retour : un troisième lancement essaie trois fois avec `WaitNamedPipeW(1000)`, puis démarre seul. Sous Linux, la file de `listen(…, 4)` garde les clients, mais l'accusé est attendu 3 s au plus. Le fichier gardé est le dernier arrivé, dans un ordre que l'Explorateur ne garantit pas (renvoi 29). | Explorateur, cinq épisodes sélectionnés, puis « Ouvrir » à froid : selon le moment où SDL est prêt, deux fenêtres peuvent s'ouvrir, et l'épisode affiché est le dernier transmis, pas le premier de la sélection. | Linux : `%F` et plusieurs fichiers dans `parse_play_options` (une file d'épisodes). Windows : `MultiSelectModel = Player`, avec un `DropTarget` ou un `DelegateExecute` qui reçoit tous les fichiers à la fois (à confirmer). Dans les deux écouteurs, rendre la main tout de suite et pousser les fichiers depuis un autre thread, pour réécouter sans attendre SDL. | Lu. À confirmer : sous Windows, cinq fichiers ouverts depuis l'Explorateur ; compter les fenêtres et noter le fichier joué. |
| A34-05 | Mineur | `src/app/play/system_peak.cpp:136-158`, `src/app/play/host_facts.cpp:236-250` | Sous Windows, chaque relevé du pic (toutes les 2 s) recrée une fabrique DXGI et parcourt toutes les sorties. Quand `MonitorFromWindow` échoue (`monitor == nullptr`), il garde la sortie **la plus lumineuse de toutes**, pas celle de la fenêtre. Linux lit une seule entrée de KWin, celle du connecteur. Sur le fond, les « deux unités » du renvoi 11→34 sont réfutées : sous Windows, le PQ d'une application est affiché tel quel, donc nits du contenu = nits de l'écran. Sous KDE, `content_peak_nits` ramène bien le pic en nits du contenu (`× 203 / blanc`). Les deux voies rendent la même grandeur. | Windows, deux écrans (un HDR à 1000 nits, un SDR) : si la fenêtre n'a pas encore de HWND ou passe entre deux écrans, le pic de 1000 nits est appliqué à l'autre écran. | Sans moniteur, ne rien rendre (la voie EDID et le repli prennent la suite), au lieu du maximum. Garder la fabrique et ne la recréer que si `IsCurrent()` est faux (WIN-04). | Lu. |
| A34-06 | Mineur | `src/app/play/auto_bench_run.cpp:48-61`, `src/app/play/player.cpp:223-224` | `hide_is_reliable_here` (faux sous Linux) et `hide_is_reliable` ne sont appelés nulle part hors des tests : les essais passent en `--hidden` sur tous les bureaux. Or `--hidden` vaut `offscreen` (`player.cpp:224`) : rien n'est présenté. La crainte du renvoi 07→34 (fenêtre cachée sous GNOME Wayland, présentations à 20 i/s) ne se produit donc pas, et les deux systèmes calibrent dans les mêmes conditions. Ces conditions omettent pourtant la présentation et le compositeur, des deux côtés : la calibration surestime ce qui tient à l'écran. | CachyOS ou Windows : la calibration choisit une hauteur de travail qui tient hors écran, puis perd des images en plein écran à 144 Hz, où la présentation, le grade et l'OSD s'ajoutent. | Retirer le code mort ou le brancher. Mesurer, sur chacun des deux systèmes, l'écart entre l'essai hors écran et la lecture réelle, puis garder une marge mesurée (renvoi 07). | Lu. À confirmer : la même chaîne en `ajp bench` puis en lecture plein écran, sur les deux systèmes. |
| A34-07 | Mineur | `src/app/play/loop_wait.cpp:45-57`, `src/app/main.cpp:299-301` | L'attente précise (`--precise-wait`, une option) finit par `std::this_thread::sleep_until` : sous Linux, `clock_nanosleep` avec un relâchement de 1 µs (`PR_SET_TIMERSLACK`) ; sous Windows, `Sleep` à la milliseconde, même avec `timeBeginPeriod(1)`. La marge de 1,5 ms (`wait_plan.h:12`) ne suffit pas toujours pour un réveil Windows. Le mode par défaut (`SDL_WaitEventTimeout` en ms) est du même ordre des deux côtés. | Windows, 165 Hz, `--precise-wait` : un réveil en retard de 1 à 2 ms sur une période de 6,06 ms, là où Linux tient à la dizaine de µs. | Sous Windows, un minuteur `CreateWaitableTimerExW(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)` pour l'échéance absolue (P4.38), puis retirer `timeBeginPeriod(1)` global si rien d'autre n'en dépend. | Lu. À confirmer : mesurer l'écart échéance-réveil avec `--precise-wait` sur les deux systèmes. |
| A34-08 | Mineur | `src/media/cuda_driver.cpp:92-110` | Le pilote CUDA se charge par son chemin système sous Windows (`LOAD_LIBRARY_SEARCH_SYSTEM32`), mais par son seul nom sous Linux (`dlopen("libcuda.so.1")`). Le `RUNPATH` du lecteur et `LD_LIBRARY_PATH` passent alors avant. La télémétrie, elle, cherche NVML dans une liste de dossiers absolus (`gpu_telemetry.cpp:269-270`). | CachyOS, archive extraite dans un dossier où quelqu'un a déposé un `libcuda.so.1` à côté de `lib/` : il est chargé à la place du pilote. | Même règle que NVML : une liste de dossiers absolus (`/usr/lib`, `/usr/lib64`, `/usr/lib/x86_64-linux-gnu`…), puis le nom seul en dernier recours, journalisé. | Lu. |
| A34-09 | Mineur | `cmake/utf8.manifest:1-9` | Le manifeste n'a ni `<compatibility>` / `<supportedOS>` (Windows 10/11), ni `dpiAwareness`. Sans `supportedOS`, Windows sert le contexte de compatibilité de Windows 8 aux API qui le lisent. La sensibilité au DPI dépend alors du seul réglage de SDL au démarrage, et `display_pixels` corrige après coup (`host_facts.cpp:122-136`). | Windows 11 à 150 % : si SDL ne passe pas en DPI par écran assez tôt (une fenêtre de message créée avant, une DLL), les tailles lues sont virtualisées et la mise à l'échelle de la fenêtre se trompe. | Ajouter `supportedOS` (Win10/11) et `<dpiAwareness>PerMonitorV2</dpiAwareness>` au manifeste, à côté de `activeCodePage`. | Lu. À confirmer : journal du mode lu sous Windows à 150 %, avec et sans le manifeste complété. |
| A34-10 | Mineur | `src/app/play/placebo_open.cpp:510-515` | La présentation est la même partout : Vulkan en FIFO. Sous Windows, la swapchain Vulkan de NVIDIA en fenêtre passe, selon le réglage « Vulkan/OpenGL present method » du panneau NVIDIA, par une copie vers DWM ou par une swapchain DXGI en flip. Le lecteur ne le journalise pas et ne le dit pas, alors que la régularité de l'image en dépend. La voie native DXGI (P3.49) n'est pas faite. | Windows 11, fenêtre (pas plein écran), méthode « Native » au panneau NVIDIA : une copie de plus par image et une cadence moins régulière qu'en flip. Rien dans `ajp.log` ne permet de comparer avec CachyOS. | Journaliser le mode de composition vu par le retour de présentation. Recommander dans `docs/` la méthode « Prefer layered on DXGI swapchain », puis trancher P3.49 par une mesure. | À confirmer : 1 % low et images perdues à 144 Hz en fenêtre, avec les deux méthodes du panneau NVIDIA. |
| A34-11 | Mineur | `src/subs/text_encoding.cpp:117-145` | WIN-09 n'est vrai qu'en partie : la glibc de CachyOS prend `BIG5` avec les extensions ETEN de CP950 (essai : `F9D6` → 碁, `A3E1` → €, `F9F9` décodés). Les tables restent différentes (Windows 950 contre glibc BIG5) sur quelques points de code, et rien ne le teste. Linux a aussi `CP950` dans iconv, qui donnerait le même résultat que Windows. | Sous-titre CP950 avec un caractère hors de la table glibc : décodé sous Windows, repli en Windows-1252 sous Linux. | Sous Linux, essayer `CP950` avant `BIG5` ; ajouter un test qui décode le même fichier sous les deux systèmes. | Démontré en partie : `iconv -f BIG5` et `-f CP950` sur ces octets (glibc 2.39). |
| A34-12 | Mineur | `src/app/play/data_dirs.cpp:67-101`, `src/app/main.cpp:478-483`, `src/app/play/single_instance.cpp:207-226` | Une migration en cours bloque aussi l'instance unique (aggrave WIN-02). L'écouteur reste dans `push_files` jusqu'à `SDL_Init`. Si ce n'est pas le cas (migration après `SDL_Init`), un lancement suivant que le canal refuse démarre seul et refait `move_heavy_data` **en même temps** que le premier, sur les mêmes dossiers. | Windows, profil redirigé, double-clic sur deux vidéos pendant la première migration : deux processus copient puis effacent `engines` en parallèle. La copie de l'un peut être retirée par l'autre (`remove_all(target)` en cas d'échec). | Un verrou de fichier (`key_lock.h` existe) autour de la migration, et la migration dans un thread derrière l'écran de préparation (A19-01, WIN-02). | Lu. À confirmer : deux lancements à 1 s d'intervalle pendant une migration simulée (dossier de 1 Go sur un autre volume). |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| WIN-02 | Toujours présent | `main.cpp:478-483`, déjà confirmé par 18 et 19 (A19-01). A34-12 ajoute la course entre deux lancements. |
| WIN-04 | Toujours présent | `player_run.cpp:534-560` : `read_display_facts` toutes les 500 ms (deux `QueryDisplayConfig`, `display_refresh_hz` puis `display_period_us`), `content_peak_nits` toutes les 2 s (fabrique DXGI recréée, SetupAPI et registre deux fois par `read_display`), sur le thread de la boucle. Doublon d'A11-02. |
| GPU-16 | Toujours présent | Même relevé que WIN-04 (A11-02). Sous Linux, le même tour lit le JSON de KWin et l'EDID de `/sys` : moins cher, mais lui aussi sur le thread de la boucle. |
| WIN-07 | Toujours présent | `crash_dump.cpp` : vidage dans le processus qui plante, depuis le thread en faute (sans pile de secours : un débordement de pile ne s'écrit pas), `__fastfail` hors de portée de `SetUnhandledExceptionFilter`, `MiniDumpWithIndirectlyReferencedMemory` qui garde les chemins en clair. Aucun PDB publié (`package-windows.ps1`, `release-windows.yml` : aucune occurrence). Linux : rien (`install_crash_dumps` est vide hors `_WIN32`), pas même une indication `coredumpctl`. |
| GPU-16 / A11-07 | Toujours présent | Voir A34-05 pour le cas multi-écran sans moniteur. |
| WIN-05 | Toujours présent | `engine_tools.cpp:56-87` : `WinHttpGetIEProxyConfigForCurrentUser` lit `lpszProxy` seulement. `lpszProxyBypass`, `fAutoDetect` (WPAD) et `lpszAutoConfigUrl` (PAC) sont lus puis libérés sans servir, et rien ne gère l'authentification. Sous Linux, curl lit `https_proxy`, hérité tel quel (`process_posix.cpp:42-47`) : le réglage de proxy de KDE, qui ne pose pas ces variables, est ignoré. |
| WIN-06 | Toujours présent | Windows : `ReadFile` bloquant sans délai (`single_instance.cpp:154-158`), donc un client muet bloque `thread_.join()` à la fermeture. Linux : `ECONNREFUSED` entre le `bind` et le `listen` d'un autre lancement fait `unlink` son socket (`:399`), et deux fenêtres s'ouvrent. |
| WIN-08 | Toujours présent | Doublon d'A20-03. |
| WIN-09 | Réfuté en grande partie | Voir A34-11 : la glibc décode le Big5 étendu ; seul l'écart de tables subsiste. |
| WIN-10 | Toujours présent | Doublon d'A20-05. |
| WIN-11 | Toujours présent | Pré-analyse sur un thread : A34-02 (conséquence sur la qualité). `timeBeginPeriod(1)` global : `main.cpp:299-301`, en partie retenu par `PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION`. Manifeste sans `<supportedOS>` : A34-09. Veille : SDL la bloque pendant toute la vie de la fenêtre, pause et accueil compris, des deux côtés (aucun `SDL_EnableScreenSaver` dans `src/`, P3.100). Télémétrie : NVML seul sous Windows, NVML et sysfs AMD sous Linux (`gpu_telemetry.cpp:242-256`), sans effet sur la cible NVIDIA. Sémaphores non partagés : backend Vulkan seulement, hors cible (repli). |
| Renvoi 02→34 | Confirmé, requalifié en A34-01 | `CMakeLists.txt:27-31` : FP8 CUTLASS éteint hors Linux, entrée `span_fp8` à `false` par défaut dans la release Windows. Pour la qualité, c'est plutôt Linux qui joue moins bien (FP8 sans décision, A49-03). Le défaut est l'écart, pas l'absence de FP8 sous Windows. |
| Renvoi 06→34 | Toujours présent, à préciser | `engine_tools.cpp:19-35` : `curl.exe` de System32 (8.x, Schannel). Derrière une inspection TLS d'entreprise, Schannel vérifie la révocation et peut échouer (`CRYPT_E_NO_REVOCATION_CHECK`). Il faudrait `--ssl-revoke-best-effort` sous Windows. Les chemins passés à `curl.exe` et `tar.exe` (enfants sans manifeste UTF-8) restent en page ANSI : A06-02 tient pour eux. |
| Renvoi 07→34 | Réfuté sur le fond | A34-06 : `--hidden` ne présente rien, sous GNOME comme ailleurs ; reste le code mort. |
| Renvoi 10→34 | Réfuté | `placebo_open.cpp:384-390` : la fenêtre cachée est détruite avec `impl` sur un refus, puis `main.cpp:533-543` montre une boîte sans parent. Le message s'affiche. Sous Windows, pendant l'ouverture (1 à 2 s), rien ne se voit, là où Linux montre une fenêtre vide : écart sans conséquence. |
| Renvoi 11→34 | Réfuté | A34-05 : les deux voies rendent des nits du contenu. Reste le maximum de toutes les sorties quand le moniteur manque. |
| Renvoi 12→34 | Confirmé, requalifié en A34-07 | Seulement avec `--precise-wait`. |
| Renvoi 20→34 | Requalifiés | Voir la section suivante. |
| Renvoi 49→34 | Confirmé | A34-01. `span_auto.cpp:22-25` ne fait qu'en donner le message faux (A09-03). |
| Renvoi 29→34 | Confirmé | A34-04. |
| Renvoi 31→34 | **Réfuté pour Windows** | `chain_setup.cpp:415-427` et `player_engines.cpp:219-230` lisent `capabilities().fp8`, qui vaut vrai sur toute carte sm ≥ 8.9 (`tensorrt_backend.cpp:278`), Windows compris. L'interrupteur CMake ne touche que `span_fp8_cutlass` (`:279`). Sous les deux systèmes, Performance cherche `rife_v4.25_fp8.onnx` ou `distil_drba_v2_lite_fp8.onnx`. RIFE n'en a aucun (`publier-modeles-fp8.sh:13`), et ceux de DistilDRBA sont facultatifs, jamais téléchargés (`:10`, `:23`). Les deux systèmes retombent donc pareil en Équilibre **et perdent le fusionné** (A31-06). Le défaut est réel et touche CachyOS autant que Windows. Il compte pour la question de DistilDRBA : au cran Performance, DistilDRBA tourne en moteur entier, plus lent que DistilDRBA Équilibre fusionné, sur les deux systèmes. |

### Défauts `path::string()` requalifiés (A20-02)

Chaque exécutable embarque `activeCodePage = UTF-8` (`cmake/utf8.manifest:5`, posé sur toutes les cibles exécutables par
`cmake/WindowsUtf8.cmake:13-30` sous MSVC, appelé à `CMakeLists.txt:113-114`). Sous Windows 11, `GetACP()` vaut
65001. La STL de MSVC convertit `path::string()` dans cette page, donc en UTF-8, sans exception. Le backend TensorRT,
une DLL chargée dans `ajp.exe`, a la page du processus. FFmpeg lit les noms de fichier en UTF-8 sous Windows. Aucun de
ces défauts ne fait donc échouer le lecteur sur la cible. Ils restent de la dette (règle de la CI, portabilité,
Windows 10 avant 1903).

| ID | Gravité d'avant | Gravité proposée | Note |
|---|---|---|---|
| A02-12 | Mineur | Mineur (dette) | `span_fp8_engine.cpp:96` : message d'erreur, de plus dans un module construit seulement sous Linux par défaut. |
| A03-06 | Mineur | Mineur (dette) | `tensorrt_rife_fused.cpp:388` : nom de fichier dérivé, en UTF-8, cohérent avec les autres écritures. |
| A04-04 | Mineur | Mineur (dette) | `rife_split.cpp:381` : idem. |
| A07-03 | **Majeur** | **Mineur (dette)** | `bench_clip.cpp:143` : la chaîne passe à `avformat_alloc_output_context2`, qui attend de l'UTF-8 sous Windows et l'obtient. La calibration ne casse pas avec un profil accentué. |
| A08-09 | Mineur | Mineur (dette) | `chain_setup.cpp:277` : journal. |
| A09-10 | Mineur | Mineur (dette) | `derived_model.cpp:19` : nom de fichier en UTF-8. |
| A10-10 | Mineur | Mineur (dette) | `placebo_screen.cpp:632` : chemin de l'instantané. |
| A13-11 | Mineur | Mineur (dette) | `resume.cpp:33` : la clé de reprise est en UTF-8 des deux côtés. Elle changerait si le manifeste disparaissait. |
| A17-12 | Mineur | Mineur (dette) | `subtitle_track.cpp:27` : extension. |
| A21-02 | **Majeur** | **Mineur (dette)** | `backend_library.cpp:55` : texte du journal seulement. Le repli Vulkan n'est pas perdu. |

Le risque réel est hors du lecteur : les **enfants sans manifeste** (`trtexec.exe`, `curl.exe`, `tar.exe`) relisent
leur ligne de commande UTF-16 dans la page ANSI (`process_win.cpp:128`, `CreateProcessW`, puis le `main(char**)` de
l'enfant). Un profil cyrillique sur un Windows français perd ses caractères. A06-02 et le renvoi 05 d'A20-02 tiennent
pour eux. Aucun test ni aucun journal ne vérifie que le manifeste reste actif : ajouter au démarrage une ligne
`page de code : GetACP()` sous Windows, et un test Windows qui la vérifie (A20-13).

## Voies natives (tableau du contre-audit)

Le tableau des 17 voies du contre-audit n'est ni dans le dépôt ni dans le document du contre-audit, qui ne donne que
son bilan (« 4 voies natives sur 17 sont faites »). Son état est reconstitué ci-dessous à partir des lignes « Écarts
Windows Linux » de `plan-audit-0.9.1.md` (dossier du projet) et du code au commit audité.

| Voie | Fiche | État |
|---|---|---|
| VRR lu par NVAPI | P0.15 | Fait (#550), hors sujet ici. |
| Minidumps Windows | P2.2 | Fait en partie : WIN-07 (Linux rien, pas de PDB). |
| Instance unique : canal nommé et socket Unix | P2.13 | Fait. Restent WIN-06, A34-03 et A34-04. |
| Données lourdes en `%LOCALAPPDATA%` | P3.15 | Fait. Migration sur le thread principal : WIN-02, A19-01, A34-12. |
| Proxy système | P3.7 | Fait en partie : WIN-05. |
| Pic par moniteur (DXGI `Monitor`) | P3.57 | Fait. Reste le maximum sans moniteur (A34-05). |
| Requêtes d'écran hors du thread principal, `WM_DISPLAYCHANGE`, `IsCurrent()` | P3.46 | Pas fait : WIN-04, A11-02. |
| Redessin pendant la boucle modale | P3.48 | Pas fait : A10-05. |
| Présentation DXGI flip native | P3.49 | Pas fait, rien de mesuré : A34-10. |
| État du lecteur par défaut (`AssocQueryStringW`) | P3.99 | Pas fait : A19-05. |
| Veille seulement pendant la lecture | P3.100 | Pas fait, des deux côtés. |
| Pré-analyse en priorité basse sans perte | P4.3 | Pas fait : A34-02. |
| Copies D3D11VA ou VA-API | P4.4 | Hors cible NVIDIA (NVDEC passe par CUDA des deux côtés, `hw_device.cpp:205-211`). |
| Sémaphores Win32 partagés avec un runtime externe | P4.36 | Le ring CUDA importe bien des HANDLE NT (`ring_slot.cpp:15-17`, `:123-128`) ; le reste est le repli Vulkan. |
| EcoQoS pour les travaux de fond | P4.37 | Pas fait (aucun `THREAD_POWER_THROTTLING` dans `src/`). |
| Minuteur haute résolution | P4.38 | Pas fait : A34-07. |
| Horloge audio par `IAudioClock` | P4.39 | Pas fait, rien de mesuré. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 07 | `src/app/play/auto_config.cpp:192-200` | L'Auto calibré force RIFE 4.22 lite (`out.rife_model`, `motion_engine = Rife`) et n'essaie jamais DistilDRBA ni RIFE 4.25 : la calibration ne peut pas montrer que DistilDRBA tient mieux à qualité proche. |
| 07 | `src/app/play/auto_bench_run.cpp:48-61` | `hide_is_reliable*` sont du code mort (A34-06). |
| 07 | `src/app/play/machine_calibration.cpp:279-330` | Premier essai FP8 perdu sur toute machine sans CUTLASS (A34-01). |
| 31 | `src/app/play/drba_fused_choice.cpp:9` | A31-06 n'est pas propre à Windows : à corriger pour les deux systèmes (le scénario du rapport 31 cite Windows à tort). |
| 06 | `src/app/play/engine_tools.cpp:76-87` | `curl.exe` de Windows sans `--ssl-revoke-best-effort` : échec derrière une inspection TLS. |
| 15 | `src/media/cuda_driver.cpp:105-109` | `libcuda.so.1` chargé par son nom (A34-08). |
| 19 | `src/app/play/single_instance.cpp:114-120` | Canal nommé à nom prévisible, sans contrôle du serveur (A34-03). |
| 11 | `src/app/play/system_peak.cpp:150-151` | Maximum de toutes les sorties quand le moniteur manque (A34-05). |
| 28 | `.github/workflows/release-windows.yml:16-19` | La release Windows ne publie ni les PDB ni CUTLASS : deux choix de publication à trancher (WIN-07, A34-01). |
| 40 | `tests/unit/` | Aucun test ne fait tourner `span_variant_order` avec les capacités d'une RTX sous Windows (`cutlass = false`, `fp8 = true`), ni `calibrate_machine` dans ce cas. A34-01 passe la CI. |

## Points positifs

- Le lancement des processus est sain des deux côtés. Windows passe par `CreateProcessW` en UTF-16, un job
  `KILL_ON_JOB_CLOSE` et une liste d'héritage limitée. Linux passe par `posix_spawn`, avec `FD_CLOEXEC` posé sur le
  socket.
- Le ring CUDA importe des HANDLE NT sous Windows et des descripteurs dupliqués sous Linux, avec le même repli
  RGB10A2. Le cas Windows de la mémoire non dédiée, à l'origine des bandes verticales, est traité
  (`ring_slot.cpp:93-99`).
- WDDM est pris au sérieux. `push_launches` envoie le lot à chaque étape et après chaque signal. Les moteurs passent
  par des graphes CUDA (`trt_engine.cpp`, `drba_fused_engine.cpp`), ce qui réduit le coût des lancements sous
  Windows. L'état de HAGS est journalisé.
- Le HDR suit la sémantique de chaque système. Sous Windows, PQ affiché tel quel, avec le blanc SDR seulement pour la
  marge de SDL. Sous KDE, le blanc de référence est lu dans `kwinoutputconfig.json`. Les deux voies rendent des nits
  du contenu.
- `nvcuda.dll` et `nvml.dll` sont chargés depuis System32 seulement.

## Questions ouvertes

- Pour l'Auto calibré, faut-il une seule politique SPAN (FP16 des deux côtés, en attendant la décision sur le FP8),
  ou la parité par CUTLASS dans la release Windows ? Pour la qualité, le fil recommande le FP16 des deux côtés
  (A34-01, A49-03).
- La pré-analyse peut-elle passer par NVDEC (A34-02) ? Le fil le recommande : la même qualité de plans sous les deux
  systèmes, sans prendre de CPU à la lecture.
