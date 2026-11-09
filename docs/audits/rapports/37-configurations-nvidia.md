# Rapport de l'audit 37. Configurations NVIDIA variées

Branche : `claude/project-thread-8tlelv`. Commit audité : `58ddedf`. Date : `2026-11-09`.

## Verdict

Les décisions prises selon la carte sont justes sur la RTX 4070 Ti de la maison, mais trois familles de la cible
sont jouées sur des hypothèses jamais vérifiées : la RTX 50 reçoit le SPAN FP8 CUTLASS par l'Auto calibré sans
qu'aucun noyau n'ait tourné sur sm_120 (A37-01), les RTX 20 et 30 se voient proposer à l'accueil un cran FP8
qu'elles ne calculent pas (A37-02), et les cartes des rangs Rapide et Normal (RTX 2060 à 4060 Ti, presque tous
les portables) reçoivent RIFE 4.22 lite sur la foi du nom de la carte, sans mesure (A37-05). Aucun défaut
bloquant ; un majeur, à confirmer sur une RTX 50.

## Cadrage retenu

Pas de questions posées à Tatoub (consigne du coordinateur) : l'audit couvre tout, à égalité. Consignes reçues :
RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; chercher ce qui empêche DistilDRBA d'être bien plus
rapide que RIFE à qualité proche ; le VRR est hors sujet (ses trois défauts connus sont seulement revérifiés) ;
toute baisse de qualité d'image faite pour la vitesse est un défaut sauf si une mesure la justifie, à signaler
avec son gain.

Configurations déroulées : RTX 20 (Turing, sm 7.5), 30 (Ampere, sm 8.6), 40 (Ada, sm 8.9), 50 (Blackwell,
sm 12.0) ; cartes de 6 et 8 Go ; portables avec Optimus ; deux GPU ; pilote 580 et plus, `nvidia-open`, NVK ;
écrans de 1080p à 4K, 16:9, 16:10 et ultra-larges, de 60 à 240 Hz, un ou plusieurs ; Windows 11 et CachyOS
(KDE, GNOME, X11). Le fil tourne sans carte NVIDIA : les calculs purs (architecture d'après le nom, hauteur de
travail par rang et par écran) ont été recopiés et exécutés à part ; le reste est lu ou « à confirmer ».

## Les configurations déroulées

Ce que le code décide, sans calibration (Auto estimé), puis ce qui change avec elle. « Rang » est la tranche de
`data/gpu_tiers.toml` lue d'après le nom Vulkan de la carte. Hauteur de travail calculée par `apply_preset`
(`auto_config.cpp:131-149`), exécutée hors du dépôt pour chaque écran.

| Carte (exemples) | Rang estimé | Travail 1080p / 1440p / 4K | RIFE, plafond ×k | Voies TensorRT | SPAN de l'Auto calibré | Interpolation basse précision |
|---|---|---|---|---|---|---|
| RTX 2060 6 Go, 3050 6 Go, 4050 portable 6 Go | Normal (1 pour 3050 portable) | 540 / 720 / 720 | 4.22 lite, ×3 | 1 (moins de 7,5 Gio) | b5f48 FP16 (2060, 3050) ; b8f64 FP8 (4050) | FP8 absent sur 20 et 30 |
| RTX 2070, 3060, 3060 Ti, 4060, 4060 Ti, 5060 | Normal | 540 / 720 / 720 | 4.22 lite, ×3 | 3 | b5f48 FP16 (20, 30) ; b8f64 FP8 (40, 50) | idem |
| RTX 2080, 3070, 3080, 3090, 4070, 5070 | Qualité | 720 / 720 / 720 | 4.25, ×4 | 3 | idem | idem |
| RTX 4080, 4090, 5080, 5090 | Ultra | 900 / 1080 / 1080 | 4.25, ×4 | 3 | b8f64 FP8 | FP8 possible |
| Portables RTX 30 à 50 « Laptop GPU » haut de gamme | une tranche sous le bureau | selon le rang | selon le rang | selon la VRAM | idem bureau | idem |
| Portables RTX 20 (« with Max-Q Design », ou sans suffixe) | rang du bureau | idem bureau | idem bureau | idem | b5f48 FP16 | — |

Lectures transverses de la matrice :

- **La qualité du SPAN dépend de l'architecture, pas de la puissance mesurée.** Une RTX 3090 calibrée joue le
  b5f48 FP16, une RTX 4060 le b8f64 FP8 (`machine_calibration.cpp:251`) ; sans calibration, personne ne joue le
  b8f64 (A49-01). Même cause qu'A49-01 et A49-03, rien de neuf à classer.
- **Écran 1440p ou 4K : Normal et Qualité travaillent à la même hauteur (720).** Seuls le modèle RIFE et le
  plafond ×k les séparent ; sur 4K, Normal à 720 donne une sortie SPAN de 1440 que libplacebo agrandit (A49-02).
- **DistilDRBA.** Jamais dans l'Auto ni dans la calibration, sur aucune carte (`auto_config.h:22`, A25-08, A29-05).
  Sur RTX 20 et 30, le cran Performance retombe en FP16 sans le fusionné (A37-02, A31-06) ; sur les cartes de
  moins de 7,5 Gio, une seule file CUDA sert SPAN et l'interpolation (A37-13), ce qui pèse plus sur DistilDRBA
  qui fait une passe par image créée (A03-13). Le reste de ce qui le ralentit ne dépend pas de la carte (A03-13,
  A03-14, A32-04).
- **Deux GPU NVIDIA.** libplacebo prend la première carte discrète ; TensorRT, NVDEC et NVML suivent son UUID, ce
  qui est juste, mais le choix de la carte (`gpu`) n'est jamais appliqué (A22-01) : la plus puissante peut rester
  inutilisée.
- **Optimus.** `NvOptimusEnablement` est exporté (`main.cpp:69`) et libplacebo préfère la carte discrète ; si la
  carte NVIDIA est éteinte (mode « intégré seul »), TensorRT est refusé par UUID (`tensorrt_backend.cpp:152-154`)
  et la lecture passe par Vulkan sur l'iGPU, ce qui est attendu. À confirmer sur un portable : présentation par la
  carte NVIDIA vers un écran branché sur l'iGPU, sous Windows et sous Wayland.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A37-01 | Majeur | `src/gpu/cuda/backend/span_fp8_net.cu:225-228`, `src/app/play/machine_calibration.cpp:251`, `src/gpu/cuda/backend/span_fp8_kernels.cuh:241` | Le SPAN FP8 CUTLASS est déclaré pris en charge sur sm 12.0 (RTX 50) alors qu'il n'a jamais tourné sur une RTX 50 (`docs/span-fp8-cutlass.md:211-217` : débordements, mémoire partagée, SASS FP8 natif et tuile à vérifier). Les noyaux portent le tag `cutlass::arch::Sm89`. La seule garde est la porte de qualité, aveugle aux NaN (A02-01) et dont la référence partage `unshuffle_kernel` avec le réseau jugé (A02-13). Baisse de qualité (FP8 au lieu de FP16) sans aucune mesure de gain sur cette architecture. | RTX 5070, premier lancement : l'assistant calibre, le premier essai SPAN prend le FP8 (`calibrate_machine`), `calibrated_chain` rend le b8f64 en précision Performance, la tâche de fond prépare le paquet en silence (`span_auto.cpp:78-82`) et la vidéo suivante le joue. Une erreur propre à sm_120 qui donne des NaN sur une partie des images de validation passe la porte : pixels faux sur toute la famille RTX 50. Si les noyaux tiennent mais n'accélèrent pas (tuile mal choisie), l'image perd 49 à 52 dB contre le FP16 pour rien. | Retirer `(12, 0)` de `supported_device` tant que la liste de `span-fp8-cutlass.md:214-217` n'est pas faite sur une RTX 50 et le gain mesuré (visé ≤ 0,70 × FP16) ; ou, au minimum, corriger A02-01 et comparer, à la première préparation sur une nouvelle architecture, la sortie FP8 au moteur TensorRT FP16 plutôt qu'à la référence maison. | Lu. À confirmer sur une RTX 50 : `ajp play --stats` avec le b8f64 FP8, puis `cuobjdump -sass -arch sm_120` sur la bibliothèque des noyaux, et `[.span_derive]` contre TensorRT FP16. Test qui échoue à décrire : `CHECK_FALSE(SpanFp8Net::supported_device(12, 0))` tant que la mesure n'existe pas (test de garde dans `tests/unit/backend_tensorrt/span_fp8_test.cpp`). |
| A37-02 | Mineur | `src/app/play/home_screen.cpp:141`, `:148-149`, `src/app/play/model_choice.cpp:90-92` | À l'accueil (aucun moteur chargé), le FP8 se présume par le nom de la carte. Le FP8 CUTLASS n'est présumé que si `cuda_arch_from_name` rend `sm89` : une RTX 50 se voit refuser le cran Performance du SPAN avec la phrase « n'est pas une RTX 40 (Ada) ou RTX 50 (Blackwell) » (`span_auto.cpp:22-25`). À l'inverse, le FP8 de RIFE et DistilDRBA est présumé sur toute carte à cœurs Tensor (`fp8_engine = tensor_cores`) : une RTX 20 ou 30 voit le cran Performance de l'interpolation. | RTX 3080, accueil, Vidéo > Interpolation : Performance est proposé pour DistilDRBA. À l'ouverture, le moteur ne calcule pas le FP8 et rend l'Équilibre (`player_engines.cpp:220-222`), mais `drba_fused_declined` refuse le fusionné hors Équilibre : DistilDRBA entier, plus lent que l'Équilibre fusionné (A31-06). RTX 5080, accueil : le SPAN Performance est grisé avec une raison fausse. | Présumer à l'accueil avec la même règle que le moteur : FP8 seulement pour sm ≥ 8.9 (`cuda_arch_from_name` ou mieux `detect_cuda_arch`), CUTLASS pour sm89 et sm120 (`SpanFp8Net::supported_device`). Et corriger A31-06 : un cran qui retombe en Équilibre doit reprendre le fusionné. | Lu. |
| A37-03 | Mineur | `src/app/play/engine_install.cpp:97-115` | `cuda_arch_from_name` se trompe sur deux cartes de la cible : la RTX 2050 (portable, puce GA107, Ampere) donne `sm75`, la RTX 3050 A Laptop (puce AD106, Ada) donne `sm86`. Ce nom sert de repli à `nvidia-smi` pour choisir les fichiers TensorRT par architecture (`effective_cuda_arch`) et, à l'accueil, de présomption du FP8 CUTLASS (A37-02). Aucun test ne couvre ces deux noms. | Portable RTX 2050 dont `nvidia-smi` échoue (délai de 10 s dépassé au réveil du GPU, échec gardé pour tout le processus, TRT-17) : le paquet tire `libnvinfer_builder_resource_sm75` au lieu de `sm86` ; la construction retombe sur la ressource PTX, ou échoue. RTX 3050 A : l'accueil refuse le cran Performance du SPAN. | Ajouter les exceptions avant la règle par série (`has("rtx 2050")` → sm86, `has("3050 a ")` → sm89), et les deux cas aux tests de `engine_install_test.cpp:525-530`. | Exécuté : copie de la fonction hors du dépôt, « NVIDIA GeForce RTX 2050 » → `sm75`, « NVIDIA GeForce RTX 3050 A Laptop GPU » → `sm86`. À confirmer : le comportement de TensorRT 11.3 sans la ressource de l'architecture. |
| A37-04 | Mineur | `data/gpu_tiers.toml:15-24`, `src/app/play/engine_screen.cpp:227` | Les portables RTX 20 ne portent pas « Laptop » dans leur nom (« GeForce RTX 2080 with Max-Q Design », ou le nom du bureau). La règle `["rtx 2080", "laptop"]` ne peut donc rien attraper, et la détection du portable de l'assistant (`gpu.name.find("Laptop")`) non plus. La correction proposée par A07-08 (une règle `laptop` générale) ne les couvrirait pas. | Portable RTX 2080 Max-Q (80 W) : rang Qualité comme la RTX 4070 de bureau, 720p + RIFE 4.25 ×4 en Auto estimé ; l'assistant annonce la durée d'un poste de bureau (150 s par construction au lieu de 300). | Ajouter `max-q` aux marques de portable (règles de la table et `engine_screen.cpp:227`) ; pour les RTX 20 portables sans suffixe, rien ne les distingue par le nom : s'appuyer sur la calibration, ou sur la batterie (`SDL_GetPowerInfo` rend autre chose que `NO_BATTERY`). | Lu. À confirmer : `nvidia-smi -L` sur un portable RTX 20 avec un pilote 580. |
| A37-05 | Mineur | `src/app/play/auto_config.cpp:176-180`, `src/app/play/auto_config.cpp:55` | Baisse de qualité choisie par le nom de la carte, sans mesure : en Auto estimé, les rangs Rapide et Normal jouent RIFE 4.22 lite au lieu de RIFE 4.25, avec un plafond ×2 ou ×3. Ces rangs couvrent les RTX 2060 à 2070, 3050 à 3060 Ti, 4060 et 4060 Ti, 5060 et 5060 Ti, et presque tous les portables. Aucun chiffre n'existe du coût de RIFE 4.25 contre lite sur ces cartes (A03-17 : lite sans banc), donc aucun gain ne justifie la perte. Le compromis C4 de l'audit 49 ne vise que l'Auto calibré. | RTX 4060 Ti 16 Go, écran 1080p, 24 i/s : SPAN à 540p (le plus léger des SPAN joués) et RIFE lite ×2, alors que la même carte tient probablement RIFE 4.25 (le mouvement est estimé à 720p au plus). | Mesurer RIFE 4.25 et lite par paire sur une carte du rang Normal (ou les estimer d'après la 4070 Ti au prorata des TFLOPS FP16) ; si 4.25 tient, le donner à tous les rangs ; sinon le faire choisir par la calibration (A25-08). | Lu. À confirmer : `ajp play --stats --rife-model=rife_v4.25.onnx` contre `rife_v4.22_lite.onnx` sur une RTX 4060. |
| A37-06 | Mineur | `src/app/play/auto_config.cpp:131-138`, `src/app/play/machine_calibration.cpp:81-86`, `src/app/play/settings_bench_results.cpp:42` | La hauteur de travail suit la hauteur de l'écran, pas celle de l'image affichée, et l'Auto estimé arrondit vers le haut quand la calibration arrondit vers le bas. Écran 16:10 (1920 × 1200, courant sur les portables récents) : l'Auto estimé vise 600, `step_of` le monte à 720 ; la calibration plafonne à 600, donc 540. Une vidéo 16:9 n'y occupe que 1920 × 1080 : 540 suffit. Le banc ouvre aussi sa fenêtre en 16:9 (`display_height * 16 / 9`), fausse sur 16:10 et ultra-large (TRT-17). | Portable RTX 4060, écran 1920 × 1200, non calibré : SPAN à 720p (1,78 fois le travail de 540p) puis réduction à 1080, sur une carte que le rang Normal pense à la limite ; images manquées que l'Auto rattrape en baissant l'interpolation. Une fois calibré, la même machine passe à 540. | Calculer la hauteur cible d'après le rectangle de l'image dans la fenêtre (`min(écran_h, écran_w × source_h / source_w)`), et arrondir pareil dans les deux chemins. Passer la vraie largeur de l'écran au banc. | Exécuté : copie de `apply_preset` hors du dépôt, écran 1200 → Normal 720, Qualité 720, Ultra 1080 ; écran 1080 → 540, 720, 900. |
| A37-07 | Mineur | `src/app/play/auto_config.cpp:223-225`, `src/app/play/player_run.cpp:614-617`, `src/app/play/auto_config.cpp:160-163` | La chaîne d'un préréglage se refait quand la fenêtre passe sur un écran d'une autre hauteur, jamais sur un écran d'une autre fréquence. Or `rife_factor` et `rife_auto` viennent de la fréquence de l'écran d'ouverture (`wanted = floor(hz / fps)`), et `rife_factor` borne le mode Stable (`auto_limits`, `player_run.cpp:429-430`). | Deux écrans 1440p, 60 Hz et 165 Hz. Vidéo ouverte sur le 60 Hz : plafond ×2, puis fenêtre sur le 165 Hz : Stable reste à ×2. Source à 30 i/s ouverte sur une télé à 50 Hz : `rife_auto` faux, puis sur l'écran 144 Hz : aucune interpolation, même en Rapide (`player_run.cpp:412`). | Faire dépendre `display_moves_chain` de la fréquence aussi, ou recalculer seulement `rife_factor` et `rife_auto` du préréglage quand `known_hz` change. | Lu. |
| A37-08 | Mineur | `src/app/play/first_run.cpp:245-253`, `src/app/play/engine_screen.cpp:408-414` | Le pilote minimum de TensorRT (580) n'est vérifié qu'à l'écran des moteurs, avant le téléchargement, et sur la seule majeure : CUDA 13 demande au moins 580.65 sous Linux. Pilote plus ancien après l'installation : `cudaGetDeviceCount` échoue (pilote insuffisant), le lecteur passe à Vulkan et le panneau dit seulement « TensorRT est choisi mais ne s'ouvre pas sur ce GPU (voir le journal) ». | Portable sous Windows : Windows Update remplace le pilote 58x par un pilote OEM 56x plus ancien (cas courant). Toutes les vidéos passent en Vulkan, plus lent, sans que l'écran dise de mettre le pilote à jour. | Au refus du backend TensorRT, relire `nvidia_driver(gpu)` et, sous le minimum, afficher « Pilote NVIDIA {x} trop ancien : TensorRT demande {min} » ; comparer majeure et mineure (580.65 sous Linux, à confirmer pour CUDA 13.4 sous Windows). | Lu. À confirmer : minimum exact de CUDA 13.4 et de TensorRT 11.3 dans leurs notes de version. |
| A37-09 | Mineur | `src/app/play/auto_bench.cpp:207`, `src/app/play/player.cpp:402-404`, `src/app/play/machine_calibration.cpp:244-246` | Les deux mesures de la machine traitent le pilote à l'opposé. Le rang mesuré par « Évaluer ma machine » (`bench.toml`) est jeté dès que la version du pilote change, sans avis ; la calibration (`calibration.toml`) ne regarde jamais le pilote. | CachyOS (pilote mis à jour toutes les quelques semaines) : le rang mesuré disparaît à chaque mise à jour, l'Auto retombe sur l'estimation par le nom sans que rien ne le dise. À l'inverse, un pilote qui change les temps de TensorRT laisse les bornes calibrées telles quelles (même cause que A07-05 pour le moteur). | Une seule règle : garder la mesure à travers une mise à jour mineure du pilote, la marquer « à refaire » (avis dans le panneau) à un changement de branche ou de version CUDA du pilote, pour les deux fichiers. | Lu. |
| A37-10 | Mineur | `src/app/play/machine_calibration.cpp:160-166`, `:244-246` | Une seule calibration par machine : `calibration.toml` garde une carte et une hauteur d'écran. Sur un poste à deux écrans de hauteurs différentes, calibrer l'un efface l'autre, et l'écran non calibré retombe toujours sur l'estimation par le nom. | Écran 1440p de bureau et télé 4K : calibré sur le 1440p, la lecture sur la télé ignore la calibration (hauteur différente) ; recalibrer sur la télé perd celle du 1440p. | Garder une entrée par (carte, hauteur d'écran) dans le fichier, en relisant l'ancien format. | Lu. |
| A37-11 | Mineur | `src/app/play/first_run.cpp:220-230`, `src/app/play/host_facts.cpp:374`, `src/app/play/placebo_open.cpp:546` | Le pilote libre NVK (Mesa) est pris pour le pilote de NVIDIA : `nvidia_driver` décode la version de Mesa avec le codage de NVIDIA. Le Vulkan du lecteur sait pourtant les distinguer (`family.cpp:162`). | CachyOS avec NVK au lieu de `nvidia-open` : l'écran des moteurs dit « pilote NVIDIA 580 ou plus récent requis, vous avez 25.00 » (si NVK expose les matrices coopératives), ou ne propose pas TensorRT du tout, sans dire que le pilote de NVIDIA est nécessaire. | Lire `VkPhysicalDeviceDriverProperties::driverID` (`VK_DRIVER_ID_NVIDIA_PROPRIETARY`) dans `device_facts`, et afficher « TensorRT demande le pilote NVIDIA (nvidia ou nvidia-open) » hors de ce pilote. | Lu. À confirmer : `vulkaninfo --summary` sous NVK (Mesa 25) pour la version et `VK_KHR_cooperative_matrix`. |
| A37-12 | Mineur | `src/app/play/home_screen.cpp`, `src/app/play/engine_install.cpp:97-115`, `data/gpu_tiers.toml` | Aucun test des décisions par configuration au-delà de quelques noms : rien sur les portables RTX 20, la RTX 2050, la RTX 3050 A, les noms des RTX PRO Blackwell (« rtx pro 6000 » ne prend aucune règle et tombe sur l'estimation), la présomption FP8 de l'accueil, la hauteur de travail sur 16:10, le plafond du mode Stable par rang. | Les défauts A37-02 à A37-06 et A37-14 passent la CI. | Un test table par famille (nom, rang, architecture, FP8 présumé, hauteur de travail par écran) dans `gpu_tier_test.cpp` et `engine_install_test.cpp`. | Lu. |
| A37-13 | Mineur | `src/gpu/cuda/backend/tensorrt_lanes.cpp:25`, `src/gpu/cuda/backend/tensorrt_backend.h:139` | Le nombre de voies se décide sur la mémoire totale (7,5 Gio), sans mesure ni de la mémoire réellement demandée ni du gain des voies. Les cartes de 6 Go font tout sur une file (aucun recouvrement de SPAN et de l'interpolation) ; les cartes de 8 Go, les plus serrées en mémoire, prennent trois voies et deux zones d'activation. | RTX 4050 portable 6 Go, DistilDRBA ×3 : chaque passe de DistilDRBA attend SPAN sur la même file, la perte est la plus forte justement là où DistilDRBA devait aider. RTX 4060 8 Go, 1440p, Ultra choisi à la main : deux zones d'activation plus l'anneau de 8 images RGBA16F (environ 0,5 Go en 4K) ; risque d'échec d'allocation (`VendorCallFailed`, A31-08). | Décider le nombre de voies sur la mémoire libre après chargement des moteurs (`cudaMemGetInfo`), et mesurer le gain des voies sur une carte de 6 et de 8 Go. | Lu. À confirmer : `--stats` sur une carte de 6 Go avec `kLaneMemory` abaissé à 5 Gio, et pic de VRAM (OSD, touche Tab) sur une carte de 8 Go en Ultra. |
| A37-14 | Mineur | `src/app/play/auto_config.cpp:55`, `src/core/auto_rate.cpp:43-50` | En mode Stable, le plafond ×k du rang (×4 au plus) coupe les paliers calés des écrans rapides : à 120 et 240 Hz, pour 23,976 i/s, le premier palier calé au-dessus de ×2,5 est ×5, que le plafond (×4,5 en demi-paliers) interdit. | RTX 4090 (Ultra), télé 120 Hz ou écran 240 Hz sans VRR, mode Stable : 60 i/s, alors que ×5 (120 i/s) tient. Le Rapide (défaut) n'est pas touché. | Prendre comme plafond Stable le plus haut palier calé que le coût mesuré tient, pas le plafond du rang ; ou ne borner par le rang que l'Auto estimé non mesuré. | Lu, et calculé : `locked_steps(120, 23.976, 2)` rend ×1, ×2,5, ×5 ; `auto_limits(…, 4)` plafonne à 9 demi-paliers. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-01 | Toujours présent (VRR, hors sujet) | `window_vrr.cpp:95-96` : l'état armé de NVAPI n'est lu que si un EDID a été lu. Déjà revérifié par l'audit 11. |
| GPU-05 | Toujours présent (VRR, hors sujet) | Zone 08, revérifié par les audits 08 et 11. |
| GPU-18 | Toujours présent (VRR, hors sujet) | `display_edid.cpp:384-389`, revérifié par l'audit 11. |
| TRT-09 | Toujours présent | `auto_config.cpp:147` : source de 720p ou moins, Qualité et Ultra donnent la même chaîne. S'y ajoute, pour toute source, que Normal et Qualité travaillent à 720 sur un écran 1440p ou 4K (matrice ci-dessus). |
| TRT-17 | En partie présent | Échec de `nvidia-smi` gardé pour le processus : `engine_install.cpp:160-162`. Carte inconnue en « ptx » : acceptable, la ressource PTX de TensorRT est sans architecture (`tensorrt-linux-x64.lock:20`), comme l'a dit l'audit 06. Moteurs reconstruits : la clé porte la version CUDA du pilote (`identity.cpp:162`, `cudaDriverGetVersion`), donc à chaque branche du pilote qui la change, sans purge (A33-04). Documentation MPEG-4 : A07-12. Largeur en 16/9 : toujours aux quatre endroits (`settings_bench_results.cpp:42`, `main.cpp:271`, `bench_steps.cpp:144`, `settings_system.cpp:405`), voir A37-06. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 15 | `src/app/play/chain_setup.cpp:43` | RTX 20 (Turing) : NVDEC ne décode pas l'AV1, une source AV1 passe en décodage logiciel puis par la copie hôte sur le thread de soumission (A01-03). À mesurer en 4K AV1 10 bits sur une RTX 20. |
| 02 | `src/gpu/cuda/backend/span_fp8_kernels.cuh:241` | Le tag `cutlass::arch::Sm89` est compilé aussi pour sm_120 : vérifier que CUTLASS n'y choisit pas un chemin propre à Ada (voir A37-01). |
| 01 | `src/gpu/cuda/backend/rife_fused_kernels.cu:141-147` | Sur sm_120, la colle de RIFE fusionné passe à deux blocs par SM pour éviter les débordements : occupation moitié moindre sur RTX 50, jamais mesurée. |

## Points positifs

- TensorRT, NVDEC et NVML suivent tous l'UUID de la carte de rendu (`tensorrt_backend.cpp:144-151`,
  `chain_setup.cpp:67`, `gpu_telemetry.cpp:290`) : sur deux GPU, rien ne se mélange.
- `detect_cuda_arch` cherche la carte de rendu par son nom parmi celles de `nvidia-smi` (`engine_install.cpp:164-166`),
  et seuls les fichiers de son architecture sont tirés.
- Le module est compilé pour sm 7.5, 8.6, 8.9 et 12.0 avec le PTX de 12.0 (`CMakeLists.txt:28`) ; les ressources
  de TensorRT 11.3 couvrent sm75 à sm120 : les quatre familles de la cible sont servies.
- La définition de l'écran est lue en pixels réels, mise à l'échelle comprise, sous Windows et ailleurs
  (`host_facts.cpp:107-130`).
- La version du pilote NVIDIA est décodée correctement depuis Vulkan, sous Windows comme sous Linux, et vérifiée
  avant 650 Mo de téléchargement.
- Une calibration faite sur batterie n'est pas reprise (`machine_calibration.cpp:243-246`).
- Aucun chemin ne distingue `nvidia` et `nvidia-open` : rien à faire pour CachyOS ; NVML est cherchée dans
  `/usr/lib`, où CachyOS la range.

## Questions ouvertes

- Faut-il retirer la RTX 50 du SPAN FP8 jusqu'à une mesure sur une vraie carte (A37-01), ou la garder derrière
  une porte corrigée ? Option recommandée : la retirer, l'Auto calibré y joue alors le b8f64 FP16 dès que A49-03
  est corrigé.
- Quel est le nom exact des portables RTX 20 sous le pilote 580 (`nvidia-smi -L`) ? A37-04 en dépend.
- Quel est le pilote minimum de CUDA 13.4 et de TensorRT 11.3 sous Windows et sous Linux (A37-08) ?
