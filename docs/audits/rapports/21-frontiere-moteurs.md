# Rapport de l'audit 21. Frontière des moteurs (interface et ABI)

Branche : `claude/project-thread-hcw9xe`. Commit audité : `3b62477`. Date : `2026-11-09`.

## Verdict

Le contrat est sain dans son principe (aucune exception ne traverse, chaque virtuelle est gardée, l'empreinte
refuse un module d'une autre forme), mais l'empreinte ne voit pas tout ce qui traverse la frontière, le chargeur
lève sous Windows avec un profil non ASCII au lieu de se replier, et l'anneau qui sort des moteurs perd six bits
par défaut avant le grade HDR, sans que la porte de qualité ait été passée.

## Cadrage retenu

Aucune question posée, à la demande de Tatoub : audit au maximum. RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent
autant ; la section « DistilDRBA » ci-dessous cherche ce qui l'empêche d'être bien plus rapide que RIFE. Le VRR est
hors sujet. Consigne ajoutée en cours d'audit : relever tout endroit où la qualité du rendu est sacrifiée pour la
vitesse (A21-03, A21-07 et la section « Qualité »). Core ML et l'export sont hors cible ; Vulkan seulement comme
repli sur une RTX. Pas de construction complète (dépendances absentes du conteneur) : les preuves sont lues, plus
une expérience compilée pour A21-01.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A21-01 | Majeur | `cmake/BoundaryHash.cmake:7-9`, `src/gpu/backend.h:346-364`, `src/gpu/kernel_ops.h:27,32,261` | L'empreinte de la frontière ne couvre pas tout ce qui la traverse. Le texte haché oublie `kernels/spec/*.h` (inclus par `kernel_ops.h:32` : `FrameStats`, `Filter`, `FrameTransfer`, `Window`, `Format`, `Matrix`, `Range`, `Siting`, `Dither`, et les constantes que les noyaux CUDA recopient) et `core/moving_area.h` (`PairScores::moving`). Les tailles hachées (`shape_fingerprint`) oublient `kernels::FrameStats`, `std::optional<FrameStats>`, `std::expected<std::optional<FrameStats>, Refusal>` (retour de `frame_stats_take`), `KernelOps::DeviceMemory` et `GpuStageSummary`. Un champ ajouté, une énumération renumérotée ou une formule changée passent le contrôle. Or ce contrôle est le seul déclencheur de mise à jour du module installé (`engine_downloads.cpp:130` : réinstallation sur `AbiMismatch` seulement) | La v+1 ajoute un champ à `FrameStats` pour le grade. Le module TensorRT de la v précédente reste installé (même `kBackendLayout`) : `frame_stats_take` remplit un objet de 1 416 octets que le lecteur lit sur 1 420. L'indicateur `has_value` de l'`expected` est lu au mauvais décalage : grade sur des statistiques fausses, ou plantage. De même, un correctif d'un noyau de `kernels/spec` (chroma, DistilDRBA, débandage) n'atteint jamais l'utilisateur tant que les en-têtes de la frontière ne bougent pas | Calculer la liste par l'inclusion réelle de `gpu/backend.h` (fichier de dépendances du compilateur, ou au moins tout `src/kernels/spec/` et `core/moving_area.h`). Ajouter les types oubliés à `shape_fingerprint`, et `kernels::kSpecVersion` à `kBackendLayout`. Décider la mise à jour du module sur le SHA du paquet (renvoi 26, A06-03), pas sur l'empreinte | **Expérience faite** : copie du dépôt, `float sky` ajouté à `FrameStats`. `sizeof(FrameStats)` passe de 1 416 à 1 420 ; `AJP_BOUNDARY_HASH` reste `9d9c399d6c29ac8f` et `kBackendLayout` reste `c2d1284ba3b6a243` (g++ 14, `-std=c++23`). Test à écrire : `boundary_hash_test.cmake` relève les `#include "…"` de `backend.h` par récurrence et échoue si l'un manque à `AJP_BOUNDARY_HEADERS` |
| A21-02 | Majeur | `src/app/play/backend_library.cpp:55,81,84,113` | Les messages de refus du chargeur sont construits avec `path.string()`. Sous Windows, cette conversion vers la page de code ANSI lève `std::system_error` pour un caractère hors de la page. L'exception sort de `choose` ; `start_task` la change en `ThreadFailed` (`core/thread.h:107-117`). `open_backend` échoue alors (`engine_screen.cpp:339`), et `player.cpp:449` rend l'erreur : la vidéo ne s'ouvre pas, même pas sous Vulkan. Dans `update_stale_engines`, le refus devient `ThreadFailed` au lieu d'`AbiMismatch` : le module périmé n'est jamais réinstallé | Windows 11 en français, profil `C:\Users\さくら\` (donc `%LOCALAPPDATA%` aussi). TensorRT choisi, et l'un des cas suivants : pilote trop ancien (la sonde refuse, l. 113), module d'une version précédente (`AbiMismatch`, l. 84), ou module absent (l. 55). Au lieu de passer à Vulkan, le lecteur n'ouvre aucune vidéo | `core::to_utf8(path)` aux quatre endroits (règle de `CLAUDE.md`, `check-path-string.sh`) ; baisser `kMax` d'autant | Lu. Test à écrire, lancé par la CI Windows : `BackendLibrary::open` sur `…/さくら/absent.dll` doit rendre `LibraryUnavailable` sans lever. À confirmer sur Windows : compte local au nom japonais, TensorRT choisi, module renommé |
| A21-03 | Majeur | `src/gpu/backend.h:132-133` (`BackendCaps::ring_rgb10a2`), `src/app/play/options.h:112-114`, `src/gpu/cuda/backend/ring_slot.h:34` | **Qualité sacrifiée pour la mémoire.** Sous TensorRT, l'anneau qui porte chaque image du moteur à l'écran est en RGB10A2 **par défaut**, au lieu de RGBA16. L'image perd ses 6 bits de poids faible **avant** le grade HDR anime et la pile de LUT, qui étirent les dégradés. L'ADR 0008 (§ 5) dit que ce défaut a changé **sans** la porte de l'étape 5.5c (`banding.py` ≤ 1,10 sur les plans sombres, export ≥ 50 dB). Le tramage TPDF ajouté pour compenser pose un bruit fixe de ±1 code sur toute image | Ciel de crépuscule d'un anime en 1080p agrandi en 4K, sortie HDR sur une RTX 4070 Ti : le grade étire les ombres, et les paliers de 10 bits (ou le grain fixe du tramage) deviennent visibles là où RGBA16 restait lisse. Le gain, 33 Mo par emplacement en 4K, ne sert qu'une carte qui manque de VRAM | Remettre RGBA16 par défaut, et ne garder RGB10A2 qu'en repli quand la VRAM manque (le seuil de `kLaneMemory` existe déjà). Ou passer d'abord la porte 5.5c sur la RTX, avec le grade HDR actif | Lu (`options.h:114`, ADR 0008 § 5). À confirmer sur la RTX : `--snapshot` sur les plans sombres de `calibration.mkv`, avec et sans `--no-ring-rgb10a2`, grade HDR actif, puis `tools/ring/banding.py --compare` |
| A21-04 | Mineur | `src/gpu/guarded_backend.h:56-168`, `tests/unit/gpu/backend_test.cpp:239-273` | Rien ne vérifie que `GuardedKernelOps` relaie **toutes** les virtuelles de `KernelOps`. Les extensions ont une version par défaut qui refuse (« absent de ce backend »). Une virtuelle ajoutée à `KernelOps` et oubliée dans le mandataire compile, mais rend toujours ce refus : la fonction disparaît en silence sous TensorRT comme sous Vulkan. Le seul test du mandataire passe un backend dont `kernels()` est nul : `GuardedKernelOps` n'est jamais testé. Aujourd'hui les 30 virtuelles sont bien relayées | Une prochaine économie de DistilDRBA ajoute `KernelOps::rife_shift`. Sans ligne dans `GuardedKernelOps`, la chaîne prend toujours le chemin lent, sans erreur ni journal | Un test qui appelle chaque virtuelle par le mandataire sur des noyaux qui comptent les appels. Ou une liste X-macro unique qui génère les deux | Lu |
| A21-05 | Mineur | `src/app/play/backend_library.cpp:79-86`, `backend_library.h:74-75` | `open` promet « le premier fichier présent de candidates », mais rend le refus dès le premier candidat sans symbole ou d'une autre ABI, sans essayer les suivants | Latent : chaque appelant passe un seul chemin (`choose`, `engine_downloads.cpp:128`). Un appelant futur qui passe une liste verrait un module périmé masquer un module valide | `continue` en ajoutant la raison à `tried`, comme pour un fichier absent | Lu |
| A21-06 | Mineur | `src/gpu/kernel_ops.h:188-196` ; usage `src/app/play/upscale_drba.cpp:99,228,251` | Le contrat de `scores_take` (« attend la fin de cette mesure seulement, pas le travail enfilé derrière elle ») est faux avec une seule voie. C'est le cas de TensorRT sous 7,5 Gio (`tensorrt_backend.h:139`, `lanes_ = 1`) et de Vulkan sans files séparées : l'attente porte sur toute la file. DistilDRBA en fait une à chaque paire (`settle_after_scores` avant le premier instant du côté b) | RTX 2060, 3050 6 Go, 4050 portable : à chaque paire, le thread GPU attend côté hôte que SPAN et la passe précédente finissent. La file se vide, d'où une bulle par paire, précisément sur les cartes qui ont le moins de marge | Dire la vraie garantie dans le contrat. Avec une voie, relire le ticket au plus tard possible : avant la passe du côté b, sans attente, et repli sur `after_ok = false` si la mesure n'est pas finie (renvoi 09) | Lu. À confirmer : profil Nsight Systems d'une 6 Go, DistilDRBA ×3, trous de la file entre deux paires |
| A21-07 | Mineur | `src/gpu/backend.h:217-229`, `src/app/play/upscale_drba.cpp:193-210` | `Engine::enqueue` n'a aucun paramètre scalaire : l'instant de RIFE et de DistilDRBA voyage en **plans pleine taille** (plan 6 de `pair01` et de `pair12`, réécrits par `rife_time` à chaque instant), que le moteur relit ensuite pour en tirer un seul nombre | DistilDRBA en 1088 × 1920 : environ 8 Mo écrits puis relus par instant pour transmettre un flottant, soit 1 à 2 % de la passe fusionnée de 0,92 ms | `Engine::set_time(float)` (défaut : rien), lu par les moteurs fusionnés. Les moteurs entiers gardent le plan | Lu ; ordre de grandeur calculé |
| A21-08 | Mineur | `src/gpu/graph_book.h:195-221,229-232` | Renvoi 01 confirmé. `capture_failures` et `launch_failures` s'additionnent sur toute la vie du moteur ; `clear()` ne remet ni eux ni `off_` à zéro. Trois rejeux refusés, même espacés de plusieurs heures, coupent les graphes CUDA jusqu'au rechargement. Un registre plein ne remplace jamais une entrée (`Overflow` pour toujours) | Soirée de lecture : deux échecs isolés au retour de veille, un troisième plus tard. SPAN et le moteur entier de RIFE ou de DistilDRBA repassent en lancements directs (plus de CPU par image), avec une seule notice au journal | Remettre les compteurs à zéro après N rejeux réussis et à `clear()`. Remplacer l'entrée la plus ancienne quand le registre est plein | Lu (`graph_book_test.cpp` ne couvre pas la remise à zéro) |
| A21-09 | Mineur | `src/gpu/lanes.h:113-115` | Renvoi 01 confirmé. `LaneHazards::states_` ne perd jamais une adresse. Chaque génération de l'anneau (seek, changement de taille) importe ses emplacements à de nouvelles adresses, qui s'ajoutent. Le résultat reste juste : une adresse réutilisée ne coûte que des attentes en trop | Longue session avec beaucoup de seeks : la table grossit de quelques entrées par génération ; une adresse reprise par un autre tampon hérite de lectures périmées, d'où une attente entre voies sans raison | `forget(addresses)` à la libération d'un tampon et à la fin d'une génération de l'anneau | Lu |
| A21-10 | Mineur | `src/gpu/kernel_ops.h:135,156,160,162,278-288`, `src/gpu/backend.h:256`, `backend_library.h:4`, `guarded_backend.h:5-6`, `src/gpu/vulkan/backend/entry_points.h:2` | Dette dans le contrat. Quatre virtuelles **pures** ne servent plus au lecteur : `rife_pad`, `measure`, `previous_scores` (`KernelOps`) et `import` (`Backend`, avec `ExternalImage` et `import_formats`). Un `grep` dans `src/` hors `src/gpu/` ne trouve aucun appel, mais chaque moteur doit encore les écrire et les garder. `SceneStats`, un type du lecteur seul, vit dans `KernelOps`, dans un en-tête haché : y toucher change l'empreinte et force une réinstallation du module. Commentaires périmés : « Provisoire… pc/gpu », « tranche verticale », « les moteurs… ne font que rendre des noms » (alors que `next_times_only` et `next_warm` portent un état), MIGraphX et OpenVINO dans `entry_points.h` | Chaque version de l'interface demande d'écrire et de tester ces quatre fonctions dans chaque moteur, sans aucun usage | Retirer les quatre virtuelles à la prochaine version de `kBackendAbi`. Sortir `SceneStats` vers `chain_shots.h`. Corriger les commentaires | Lu (`grep "\(->\|\.\)\(import\|rife_pad\|measure\|previous_scores\)("` hors `src/gpu` : 0) |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-03 | Toujours présent, aggravé sous Windows | Le chargeur refuse bien en `AbiMismatch` et cite les valeurs (`backend.h:368-382`). La boucle de réinstallation reste chez 06 (A06-03). Avec un profil non ASCII sous Windows, le refus devient `ThreadFailed` (A21-02) : ni réinstallation, ni repli. |
| Renvoi 01 (`graph_book.h`) | Toujours présent | A21-08. |
| Renvoi 01 (`lanes.h`) | Toujours présent | A21-09. |
| Renvoi 06 (`backend_library.cpp:83-85`, A06-09) | À confirmer, cause aggravante trouvée | Le module refusé est bien refermé (le `shared_ptr` local meurt, `FreeLibrary` appelé). Mais le contrôle d'ABI ne vient **qu'après** `LoadLibraryExW` : la table d'import a déjà chargé nvinfer et CUDA, et les initialisations statiques du module ont tourné. Que nvinfer se décharge ou non dépend de NVIDIA (une DLL qui s'épingle ou garde un thread reste). Manipulation : Process Explorer, onglet DLL, après « module d'une version précédente » au journal. Correction plus sûre, déjà proposée par A06-09 : décider de la mise à jour sur `paquet.txt` **avant** de charger le module. |
| Points de la fiche | — | Chemin absolu : oui (`backend_library.cpp:66`). Message quand le module manque : oui, sauf A21-02. Exceptions : `GuardedBackend` couvre les 13 virtuelles de `Backend` et les 30 de `KernelOps` ; les points d'entrée sont `noexcept` avec un `try` (`cuda/backend/entry.cpp:84-96`, `vulkan/backend/entry_points.h:37-50`). Seule limite : `refusal_from_exception` alloue dans le `catch` d'une fonction `noexcept`, donc une exception de mémoire épuisée y deviendrait `std::terminate` ; c'est accepté. Capacités : vraies pour TensorRT et Vulkan (`fp8` absent sous Vulkan, `span_fp8_cutlass` faux hors Linux par `span_fp8_net_none.cpp`, `rife_fused` et `drba_fused` vrais sous Vulkan avec FP16). Durée de vie : la bibliothèque survit à ses instances (`Destroy::library`) ; l'ordre « moteurs et tampons avant le backend » n'est tenu que par contrat (renvoi 01 → 30 déjà ouvert). |

## DistilDRBA : ce qui l'empêche d'être bien plus rapide que RIFE

À la frontière, rien n'empêche DistilDRBA d'être rapide : son moteur fusionné est déjà environ six fois moins cher
que RIFE. Ce que la frontière transmet coûte peu (A21-07). La perte vient d'ailleurs. Voici ce que montrent les
mesures déjà faites, puis les causes, de la plus lourde à la plus légère.

**Les mesures.** RTX 4070 Ti (`audit/mesures-auto-4070ti.md`) : une passe DistilDRBA coûte 0,73 à 0,77 ms, RIFE 4.25
4,2 à 4,5 ms. Pourtant, en lecture, DistilDRBA tient ×5 quand RIFE tient ×4. En mode libre (`docs/auto-libre.md`
§ 3), DistilDRBA ×3 fusionné rend 67,5 i/s contre 60 pour RIFE, soit +12 % seulement, avec « limite GPU 79 % ». À ×6,
DistilDRBA n'a **aucune** paire en retard ni aucun instant sauté, mais 962 images manquées ; RIFE, lui, a 205 paires en
retard. Le calcul de DistilDRBA suit donc ; c'est ce qui vient **après** le réseau qui plafonne.

1. **Le coût fixe par image produite domine** (renvois 32 et 10). Avec 79 % de GPU à 67,5 i/s, environ 11 ms de GPU
   par image ne sont pas du réseau : warp à la taille de sortie après SPAN, écriture de l'anneau, rendu et grade par
   libplacebo, analyse `frame_stats` (et le RTX HDR, retiré depuis). Un réseau 3,5 ms plus rapide ne déplace que la
   part du réseau. Mesure à faire : `<stats>.gpu.tsv` (`KernelOps::gpu_profile`) de DistilDRBA et de RIFE, même
   scène, même multiplicateur.
2. **Images manquées sans retard de calcul** (renvoi 10) : à ×6 et 165 Hz, DistilDRBA tient le calcul mais la
   présentation manque des images. C'est la présentation qui limite, pas le moteur.
3. **L'encodeur est recalculé deux fois par image source** (renvoi 09). Le côté b de la paire n (`(a, b)`,
   `(b, après)`) et le côté a de la paire n + 1 (`(avant', a')` = `(a, b)`, `(a', b')` = `(b, après)`) ont
   **exactement** les mêmes entrées ; seul l'instant change. Pourtant `drba_arrive` remet `filled01` et `filled12` à
   zéro (`upscale_drba.cpp:66-67`) : `rife_input` réécrit les deux entrées en pleine taille, et le moteur refait
   réductions, encodeur et `drba_post`. À ×3, il n'y a aucune reprise D6 (un instant par côté) ; toute la partie
   « par trio d'images » est faite deux fois. Le contrat `next_times_only` le permet déjà : mêmes tampons, même
   zone (`drba_fused_engine.cpp:310-313`). Il suffit que l'étage garde l'état « rempli » à travers `drba_arrive`
   quand la suite est continue (pas de coupe ni de saut). À confirmer : profil des étapes du moteur fusionné, avant
   et après.
4. **Les trois économies de l'étage refusées à DistilDRBA** (A09-07, déjà au tableau) : pas de mesure en avance,
   pas de saut des instants en retard, pas de débandage à l'ingestion.
5. **Une attente hôte par paire** (A21-06), sur les cartes à une voie.
6. **L'instant voyage en plans pleine taille** (A21-07) : 1 à 2 %.
7. **Le cran Performance coupe le moteur fusionné** (A09-02, déjà au tableau) : ≈ 1,9 ms au lieu de 0,92.

Côté RIFE, le défaut symétrique est déjà au tableau : A03-08 (chaque paire réencode les deux images alors que
A(n+1) = B(n)). Les deux moteurs perdent donc de la même façon le travail « par image » qu'ils pourraient garder.

## Qualité sacrifiée pour la vitesse, dans la zone et à ses bords

- **A21-03** (anneau RGB10A2 par défaut, porte non passée) : le seul cas vraiment dans la zone. À trancher en
  premier.
- **Contrat de la chroma non tenu sans refus** : `IngestSpec::chroma` vaut Spline36 (`kernel_ops.h:52`) et le contrat
  dit « refus nommé, jamais une conversion silencieuse » (`kernel_ops.h:6-8`). TensorRT l'ignore et prend du
  bilinéaire (A03-01, déjà au tableau). Aucun test de la frontière ne vérifie qu'un moteur honore `chroma` : à
  ajouter à la suite commune (`vk_backend/kernel_ops_test.cpp` et son pendant TensorRT).
- Hors zone, pour information (renvois 08 et 09) : le mouvement de RIFE et de DistilDRBA s'estime à 720p pour un
  travail plus grand (D4, `chain_setup.cpp:361-363,380-382`), et DistilDRBA passe au demi-flot à partir de 1440p
  (`drba_half_flow`). Ce sont des choix de vitesse à remesurer en qualité, puisque DistilDRBA a de la marge (point 1
  ci-dessus). `kRifeChromaFilter` = bilinéaire pour l'entrée de RIFE est justifié : c'est la conversion de
  l'entraînement (`rife.h:71`).

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 26 | `src/app/play/engine_downloads.cpp:130` | La mise à jour du module ne se déclenche que sur `AbiMismatch`. Un correctif du module (noyaux, DistilDRBA, chroma) qui ne touche pas les en-têtes de la frontière n'arrive jamais chez l'utilisateur (voir A21-01). Comparer le SHA de `paquet.txt` à celui de l'index. |
| 09 | `src/app/play/upscale_drba.cpp:66-67,193-210` | Garder `filled01` et `filled12` à travers `drba_arrive` quand la suite est continue : le côté a de la paire n + 1 reprend les entrées du côté b de la paire n, d'où `next_times_only` (point 3 de la section DistilDRBA). |
| 09 | `src/app/play/upscale_drba.cpp:99,228,251` | `scores_take` bloque le thread GPU à chaque paire ; sur une seule voie, cette attente vide la file (A21-06). |
| 32 | `docs/auto-libre.md` § 3 ; `KernelOps::gpu_profile` | Décomposer les ≈ 11 ms de GPU par image hors réseau (warp, anneau, rendu, grade, statistiques). C'est ce qui ôte à DistilDRBA son avance sur RIFE. |
| 10 | `audit/mesures-auto-4070ti.md` (×6 à 165 Hz) | DistilDRBA ×6 : 0 paire en retard, 962 images manquées. La présentation plafonne, pas le calcul. |
| 08 | `src/app/play/chain_setup.cpp:361-363,380-382` | Mouvement estimé à 720p pour un travail plus grand (D4) : choix de vitesse à remesurer en qualité, surtout pour DistilDRBA qui a de la marge. |
| 08 | `src/app/play/engine_screen.cpp:343` | `(*library)->path().string()` dans le journal : même exception sous Windows qu'A21-02, mais après le choix, donc sans repli perdu. |
| 45 | `src/gpu/vulkan/backend/entry_points.h:1-6` | Commentaire qui nomme encore MIGraphX et OpenVINO, retirés des sources. |
| 40 | `tests/unit/gpu/backend_test.cpp` | `GuardedKernelOps` n'est jamais testé (A21-04) ; aucun test commun ne vérifie `IngestSpec::chroma`. |

## Points positifs

- Le chargeur lit `ajp_backend_abi` et `ajp_backend_layout` **avant** de passer un `BackendInit`. Le refus cite les
  deux valeurs, et chaque point d'entrée est `noexcept`, vérifié par des `static_assert`
  (`backend_test.cpp:26-29`).
- L'empreinte combine la version, le texte des en-têtes sans commentaires (testé par `boundary_hash_test.cmake`), la
  bibliothèque standard (CRT `/MD` contre `/MT`, niveau des itérateurs de débogage) et la taille de chaque type
  passé. Seul son périmètre est trop court (A21-01).
- `GuardedBackend` relaie aujourd'hui toutes les virtuelles. Le destructeur détruit `kernels_` avant `in_`, et la
  bibliothèque survit à chaque instance par le `shared_ptr` du suppresseur.
- `keep_open` évite de recharger TensorRT entre la vérification de version et le choix, et `replaced_on_disk`
  empêche de rendre une bibliothèque remplacée sur le disque.
- `LaneHazards` reste juste quand une adresse est réutilisée (des attentes en trop, jamais en moins), et ne répète
  pas une attente déjà posée (`waited_`).
- Le registre des graphes dit au journal pourquoi il se coupe, au lieu de lancer en direct sans un mot.

## Questions ouvertes

- A21-03 : RGB10A2 par défaut se justifie-t-il encore sur une carte de 8 Go ou plus, alors qu'il passe avant le grade
  HDR ? Option prise dans ce rapport : RGBA16 par défaut, RGB10A2 seulement quand la VRAM manque.
- Le point 3 de la section DistilDRBA (encodeur calculé deux fois) demande une mesure sur la RTX pour chiffrer le
  gain. Ordre de grandeur attendu : la part « par trio » de la passe fusionnée, que les médianes des sous-moteurs
  (0,064 + 0,075 + 0,129 ms pour les blocs, sur 0,92 ms) laissent estimer à quelques dixièmes de milliseconde.
