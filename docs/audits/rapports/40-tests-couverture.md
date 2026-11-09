# Rapport de l'audit 40. Tests et couverture

Branche : `claude/project-thread-2n7i72`. Commit audité : `4a83038`. Date : `2026-11-09`.

## Verdict

La suite est large (1 557 tests ctest, 76 % des lignes de `src/` couvertes sous lavapipe) et presque toute verte,
mais elle ne protège pas ce qui fait le produit : aucun des réseaux livrés (SPAN, RIFE 4.25, RIFE 4.22 lite,
DistilDRBA) ne tourne jamais en CI, le chemin TensorRT n'est jamais exécuté, le seul bout en bout joue sans
agrandissement ni interpolation, et plusieurs tests figent un défaut de qualité au lieu de le refuser. Tout cela
se corrige à faible coût : les tests cachés du fusionné passent en 4 minutes sous lavapipe avec les vrais
modèles, téléchargés en une minute.

## Cadrage retenu

Pas de questions à Tatoub (consigne du coordinateur) : tout est audité au maximum. Retenu de la consigne : cible
NVIDIA sous TensorRT, Windows 11 et CachyOS ; RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; ce qui
empêche DistilDRBA d'être bien plus rapide que RIFE à qualité proche est cherché en priorité ; le VRR est hors
sujet ; toute baisse de qualité faite pour la vitesse est un défaut sauf mesure qui la justifie. Points imposés :
`ring_stage_test.cpp:243`, `GuardedKernelOps`, la portée de TSan, les essais cachés du fusionné,
`span_variant_order` avec les capacités d'une RTX sous Windows.

**Ce qui a tourné ici** (sans carte NVIDIA) : dépendances construites par `tools/deps/linux-from-source.sh`, le
préréglage `linux-prefix` (Debug, gcc 13) construit avec `--coverage`, toute la suite sous Xvfb et lavapipe avec
la couche de validation exigée (`AJP_TEST_VULKAN_VALIDATION=require`, comme la tâche lavapipe), puis gcovr. Les
vrais modèles ont été pris par `tools/release/fetch-models.sh` (SPAN HD V3.1, ArtCNN, RIFE 4.22 lite, 4.25, 4.26,
DistilDRBA v2 lite) pour lancer les tests cachés du backend Vulkan et le lecteur complet. Les tests écrits pour
cet audit ont été lancés puis retirés (le fil n'écrit que ce rapport) ; leur texte est décrit dans chaque défaut.

Résultats bruts :

| Mesure | Valeur |
|---|---|
| Tests ctest (lavapipe) | 1 557 : 1 556 passent, 10 sautés (plafond 12), 1 échec intermittent (A40-05) |
| Durée de la suite | 160 s à 4 fils (build Debug avec couverture) |
| Couverture des lignes de `src/` compilé sous Linux sans CUDA | 76 % (40 230 / 52 943) |
| `src/app/play` | 61,7 % |
| `src/gpu/vulkan/backend/drba_fused_engine.cpp` (DistilDRBA fusionné, repli Vulkan) | 1,1 % (3 / 284) |
| `src/gpu/vulkan/backend/vulkan_drba_fused.cpp` | 0 % |
| `src/gpu/vulkan/backend/rife_fused_engine.cpp` | 33 % |
| `src/gpu/guarded_backend.h` | 26,5 % |
| `src/app/play/chain.cpp` (changements en direct) | 10,5 % |
| `src/app/play/chain_stats.cpp` | 10,7 % |
| `src/app/play/chain_setup.cpp` | 41 % |
| `src/gpu/cuda/**` (backend TensorRT, noyaux) | non compilé en CI hors tâche `cuda`, jamais exécuté : 0 % |
| `[.drba_fused]` Vulkan, vrai DistilDRBA, lavapipe | passe en 1 min 14 : 60,9 à 63,0 dB contre le réseau entier |
| `[.rife_fused]` Vulkan, vrais RIFE 4.22 lite et 4.25, lavapipe | passe en 2 min 51 : 74,2 et 73,8 dB |
| Lecteur complet, SPAN b5f48 + RIFE 4.22 lite fusionné, 320×180, lavapipe | construit en 4 min, joue (525 ms par image) |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A40-01 | Majeur | `tests/unit/vk_backend/drba_fused_chain_test.cpp:70`, `tests/unit/vk_backend/rife_fused_chain_test.cpp:130`, `tests/unit/vk_backend/CMakeLists.txt:54`, `.github/workflows/ci.yml:99-100` | **Aucun réseau livré ne tourne en CI.** Les seuls tests qui font calculer le vrai SPAN, le vrai RIFE 4.25, 4.22 lite ou le vrai DistilDRBA sont cachés (`[.rife_fused]`, `[.drba_fused]`, `[.rife_profil]`) ou se sautent faute de modèle (`AJP_RIFE_MODEL_DIR` vaut `$HOME/ajp/models/rife`, vide sur un runner ; « Les DistilDRBA du catalogue se coupent en quatre morceaux » sauté). Les modèles synthétiques de `test_models.cpp` ne voient ni la structure de DistilDRBA, ni la découpe fusionnée réelle, ni les tailles réelles. Couverture : `drba_fused_engine.cpp` 1,1 %, `vulkan_drba_fused.cpp` 0 %, `rife_fused_engine.cpp` 33 %. | Une modification de `prepare/drba_split.cpp` ou de la colle fusionnée qui change l'image de DistilDRBA (ou de RIFE) passe toute la CI. Le repli Vulkan d'une RTX joue alors une interpolation fausse, et rien ne le dit avant Tatoub à l'œil. | Une tâche « modèles » dans `ci.yml` (nuit, et chaque PR qui touche `src/gpu`, `src/prepare`, `src/kernels`) : `tools/release/fetch-models.sh` avec cache `actions/cache` sur le SHA du script, puis `ajp_vk_backend_tests "[.rife_fused],[.drba_fused]"` et les tests de découpe de `prepare` avec `-DAJP_RIFE_MODEL_DIR` et `-DAJP_DRBA_MODEL_DIR`. Rendre ces cas visibles à ctest (`add_test` explicite, comme `e2e:`) pour que `check-skips.sh` les compte. | Lancé ici : les deux tests cachés passent sous lavapipe avec la validation, 4 min 05 en tout (60,9 à 74,2 dB) ; le téléchargement des modèles prend une minute. |
| A40-02 | Majeur | `.github/workflows/ci.yml:219-258`, `tests/unit/backend_tensorrt/CMakeLists.txt:40`, `tools/ci/check-cuda.sh` | **Le chemin de la cible (TensorRT, noyaux CUDA, SPAN FP8, RIFE et DistilDRBA fusionnés) n'est jamais exécuté avant une release.** La tâche `cuda` ne fait que compiler. Les tests CUDA se sautent sans carte, leurs essais de concordance sont cachés (`[.rife_fused]`, `[.drba_fused]`, `[.span_fp8]`, `[.span_derive]`), et aucun script ne les enchaîne : `docs/tests-pc.md` les liste à la main, chacun avec ses variables CMake (`AJP_SPAN_FP8_PACKAGE`, `AJP_DRBA_IMAGE`…). `release-guard.yml` n'exige aucune trace d'un passage sur la carte. C'est B3, toujours présent pour TensorRT. | Une régression d'un noyau `.cu` (A03-01, A03-11, A03-18 sont de ce type) ou de l'anneau CUDA part dans un paquet Windows ou Linux sans qu'aucun test ne l'ait exécutée. | Un script `tools/ci/nvidia-suite.sh` qui configure `linux-prefix-cuda` (ou `windows-rtx`) avec les modèles de `fetch-models.sh`, lance `ajp_backend_tensorrt_tests "[cuda],[.rife_fused],[.drba_fused],[.span_fp8]"` et écrit un reçu signé du commit ; `release-guard.yml` refuse une étiquette sans reçu du même commit. À terme, un runner auto-hébergé sur le PC de Tatoub, déclenché à la main. | Lu. À confirmer : sur la RTX 4070 Ti, le script proposé passe en entier, et chaque test caché donne son chiffre. |
| A40-03 | Majeur | `tools/bench/rife_triplets.py:96-100`, `tests/unit/backend_tensorrt/drba_fused_test.cpp:402`, `tests/unit/vk_backend/drba_fused_chain_test.cpp:70` | **Rien ne mesure ni ne protège « DistilDRBA bien plus rapide que RIFE à qualité proche ».** Le seul banc de qualité d'interpolation, `rife_triplets.py`, refuse DistilDRBA (« ce banc prend l'entrée v1 à 11 plans » ; DistilDRBA prend deux entrées de 7 plans). Les tests de DistilDRBA ne le comparent qu'à lui-même (fusionné contre entier), jamais à la vraie image du milieu ni à RIFE 4.25 ou 4.22 lite. Aucun test ni banc ne compte les passes par paire : la régression qui fait faire à DistilDRBA plus de réseau que RIFE au-delà de ×5 (A03-13) et les quatre `rife_input` par paire (A03-14) passent la CI. | On ne peut ni affirmer que DistilDRBA est « à qualité proche » de RIFE 4.25, ni voir qu'un changement de l'étage lui fait perdre son avance de vitesse. Le choix du moteur par défaut repose sur des chiffres qu'aucun outil du dépôt ne reproduit. | Étendre `rife_triplets.py` à DistilDRBA (entrées `pair01`/`pair12`, triplets k, k+1, k+2 tirés de quatre images, image k+1 contre la vraie) et sortir PSNR, SSIM et ms par paire des trois modèles sur `calibration.mkv`. Dans la suite : un test de l'étage (`upscale_stage_drba_test.cpp`, banc sans GPU) qui fixe le nombre de passes et de `rife_input` par paire à ×2, ×3, ×6 et le compare à RIFE (`rig.backend.count("engine drba")`). | Lu ; `rife_triplets.py:100` refuse le modèle. |
| A40-04 | Majeur | `tests/unit/app/e2e_play_test.cpp:96-104`, `:201-246` | **Le seul bout en bout ne joue pas la chaîne du produit.** Les quatre cas `[.e2e]` passent `--no-upscale --rife=0` : ni SPAN, ni RIFE, ni DistilDRBA, ni anneau d'interpolation. La pile de looks par défaut est absente (« look : HDR grading base ver.9.0 33 point.cube introuvable » dans le journal du test) : l'image vérifiée est un grade sans sa LUT de base. SEC-22, partie bout en bout, toujours présent. | Une rupture entre la chaîne et la présentation (anneau, warp, ordre des images interpolées, look) passe la CI tant que la lecture sans réseau marche. | Un cas e2e avec SPAN et RIFE fusionné (et un avec DistilDRBA) sur un clip de 320×180, modèles de A40-01, `--snapshot-at` placé après la première image interpolée, et la LUT de base fournie au test. Le lecteur sous lavapipe ne tient pas la cadence (525 ms par image ici) : juger l'image et le compte des images interpolées, pas la cadence ; ou passer par l'export, qui rejoue la même chaîne image par image. | Lancé ici : le lecteur complet (SPAN b5f48 + RIFE 4.22 lite fusionné, `--rife=2`) construit ses moteurs en 4 min et joue sous lavapipe ; à 525 ms par image il n'interpole aucune paire en 2 s (« 0 interpolées »), d'où l'export ou un mode sans horloge pour juger l'image. |
| A40-05 | Mineur | `tests/unit/app/e2e_play_test.cpp:119-124`, `:176`, `:221`, `tests/unit/app/CMakeLists.txt:33` | **Les cas e2e jouent tous les clips du dossier.** Les clips sont écrits dans un même dossier (`AJP_MEDIA_CLIP_DIR`) ; à la fin de son fichier, le lecteur enchaîne « l'épisode suivant ». Le cas « joue un clip de 2 s » joue en fait `e2e-2s`, `e2e-853x480`, `e2e-pause` puis `e2e-tronque-source` (quatre « épisode suivant » au journal) : ses `--stats` et sa durée mélangent quatre fichiers. Le cas 853×480 (`--seconds=1`, `fresh >= 3`) dépend de la vitesse de la machine. | Échec observé ici sous charge (`e2e_play_test.cpp:176`, `fresh >= 3` faux), vert seul à la relance. Un nouveau clip dans le dossier change ce que mesurent les autres cas. | Un dossier par cas (comme `fresh_dir`), ou une option qui coupe l'enchaînement ; juger le nombre d'images neuves sur le nombre d'images décodées, pas sur une seconde d'horloge. | Lancé : échec puis succès ; journal `ajp.log` du cas, quatre lignes « épisode suivant ». |
| A40-06 | Majeur | `tests/unit/app/e2e_play_test.cpp:236-243` | **Un test fige une image SDR 38 % plus sombre que la source.** Le clip a une luma de 200 en plage limitée ; converti en BT.709, le centre vaut RGB 155, 238, 151 (luma 214 sur 255). Le test exige une luma entre 100 et 160 « mesurée le 07/11/2026 » : la valeur fidèle échouerait. Mesuré ici : 132 avec le grade, **134 avec `[grade] enabled = false`** (RGB 96, 150, 91, chaque canal × 0,62). L'écart ne vient donc pas du grade : la sortie SDR, ou l'instantané `--snapshot`, assombrit tout. | Sur un écran SDR, toute la lecture serait nettement plus sombre que l'original ; si seul l'instantané est faux, l'outil de vérification ment, et le test le grave. | Juger contre la valeur attendue de la source (214 ± quelques codes, grade coupé) ; chercher d'où vient le facteur 0,62 (contrat d'écran à 203 nits de référence et pic de 1 000 nits appliqué à une sortie SDR ?). | Lancé ici sous lavapipe. À confirmer sur le PC, HDR du système coupé : `ajp play clip.mkv --snapshot=c.ppm` avec `[grade] enabled = false`, comparé à une capture d'écran de la fenêtre et à l'image décodée par ffmpeg. |
| A40-07 | Mineur | `tests/unit/backend_tensorrt/backend_test.cpp:92`, `:104`, `:255` | **Le test d'ingestion TensorRT fige la chroma bilinéaire (A03-01).** Il compare l'ingestion à `yuv_to_rgb(..., Filter::Bilinear)` alors que le contrat (`IngestSpec::chroma = kChainChromaFilter`) et Vulkan prennent Spline36, et exclut les bords (« ce noyau provisoire »). `IngestSpec::chroma` n'est lu que par Vulkan (`kernel_frames.cpp:405`). | La correction d'A03-01 fait échouer ce test ; un retour au bilinéaire le fait passer. | Comparer à `IngestSpec{}.chroma`, comme `tests/unit/vk_backend/kernel_ops_test.cpp:119`, bords compris. | Lu. |
| A40-08 | Mineur | `src/gpu/guarded_backend.h:56-170`, `tests/unit/gpu/backend_test.cpp:239` | **`GuardedKernelOps` n'est jamais testé** (renvoi 21). Seul `GuardedBackend` l'est, avec `kernels()` nul. Couverture du fichier : 26,5 %. Rien n'empêche une virtuelle ajoutée à `KernelOps` avec un corps par défaut d'être oubliée par le mandataire : elle répondrait « non pris en charge » sous TensorRT et Vulkan sans erreur de compilation. Le module `gpu` n'est pas dans les préréglages de test `windows-ci` et `windows-release` : le mandataire, posé surtout pour MSVC, n'est jamais testé sous Windows. | Une nouvelle opération (l'entrée par pointeurs de DistilDRBA proposée par A03-14) ajoutée à `KernelOps` et oubliée dans `GuardedKernelOps` : la chaîne retombe sur le chemin lent sans un mot. | Ajouter le test ci-contre ; ajouter `gpu` au filtre `windows-ci` ; faire échouer la compilation si `KernelOps` gagne une virtuelle sans que `GuardedKernelOps` change (empreinte de la frontière, comme `gpu: l'empreinte…`). | Écrit et lancé ici : un `KernelOps` qui lève dans ses 30 virtuelles, enveloppé ; chaque appel rend `VendorCallFailed` ou la valeur prudente ; 54 assertions passent. |
| A40-09 | Mineur | `CMakePresets.json:348`, `:392`, `:415`, `.github/workflows/ci.yml:48-54`, `:100` | **TSan voit peu de la concurrence réelle** (renvoi 30). La tâche « Cœur (tsan) », la seule à chaque poussée sur main, ne lance que les 6 tests nommés `Inbox`. La tâche `tsan-pr` filtre sur `audio`, `Session`, `Pipeline`, `Stage` : les tests de l'anneau (`Ring : …`), des plans et de la boucle n'y sont pas. `LookLoader` et `ChainStats` n'ont aucun test (`chain_stats.cpp` couvert à 10,7 %) : TSan ne peut pas voir leurs courses (A08-02). Le module TensorRT n'est jamais construit sous TSan. | La course d'A08-02 (`ChainStats` lit `kernels_` après la destruction du backend) ne peut être vue par aucune tâche. | Un test de `ChainStats` et de `LookLoader` avec leurs threads ; élargir `tsan-pr` à `[Rr]ing\|[Ss]hot\|[Ll]oop` ; dans la tâche Cœur, toute la suite du cœur sous TSan (elle tourne en quelques secondes). | Lu ; couverture mesurée. |
| A40-10 | Mineur | `src/app/play/span_int8.cpp:194-202`, `tests/unit/app/span_fp8_cutlass_test.cpp:93-106` | **`span_variant_order` n'est pas testé avec les capacités d'une RTX sous Windows** (renvoi 34 ; `cutlass = false`, `fp8 = true`). Le test écrit ici montre : un préréglage prend l'INT8 seul ; le cran Performance choisi à la main prend d'abord le **FP8 par Q/DQ, aux seuils « lâches »** de l'aveu du code, gain non mesuré dans ce cas ; le cran Équilibre prend l'INT8 (A49-07). | Sous Windows, Performance à la main joue une précision qu'aucun test ni mesure ne juge, et différente de Linux (A34-01). | Ajouter le cas Windows au test ; décider la politique (A34-01, A49-07) et la figer. | Écrit et lancé ici : les deux premiers `CHECK` passent, `span_variant_order(Balanced, windows_rtx, true).empty()` échoue (rend `{Int8}`). |
| A40-11 | Mineur | `tests/unit/vk_backend/rife_fused_chain_test.cpp:100-106`, `tests/unit/vk_backend/drba_fused_chain_test.cpp:156-161` | **La variante mouvement des chaînes fusionnées n'est jugée que par « flot borné »** (`largest < 64`). Un flot nul, de mauvais signe ou à la mauvaise échelle passe. RIFE 4.25 en mouvement n'est pas essayé, et le flot n'est jamais comparé à celui du réseau entier. C'est la sortie que le warp utilise quand SPAN tourne par source (le cas par défaut avec agrandissement). Pendant Vulkan d'A03-03. | Un flot divisé par deux à la découpe donne des images interpolées à mi-chemin, et le test passe. | Comparer flot et masque à ceux du réseau entier (la sortie flot de `derive_*_fused` a son équivalent dans le graphe d'origine), ou juger l'image warpée contre l'image du réseau entier. | Lu ; test lancé (il passe avec la seule borne). |
| A40-12 | Mineur | `cmake/AjpModule.cmake:77`, `tests/unit/media/shot_test.cpp:247`, `tools/ci/check-skips.sh` | **Les 21 cas cachés sont invisibles.** `catch_discover_tests` ne les déclare pas : `ctest -N` ne les liste pas, `check-skips.sh` ne les compte pas, et 17 d'entre eux ne tournent nulle part (seul `e2e:` a un `add_test`). La consigne de `shot_test.cpp:247` (« ctest -R shot_nvdec ») ne lance rien. Aucune liste ne dit lesquels lancer avant une release. | Un essai de concordance qui casse reste invisible jusqu'au jour où quelqu'un pense à le lancer. | Un `add_test` par cas caché, avec un libellé (`LABELS nvidia;modeles`) et une propriété `DISABLED` ou un filtre de préréglage ; ainsi `ctest -L nvidia` les lance et `check-skips` les voit. | Vérifié : `ctest -N` ne contient pas `shot_nvdec`. |
| A40-13 | Mineur | `tests/unit/app/ring_stage_test.cpp:150-153`, `:241`, `tests/unit/app/loop_wait_test.cpp:26`, `tests/unit/audio/thread_test.cpp:161` | **Tests à l'horloge réelle.** 26 fichiers de tests appellent `sleep_for` ; des bornes de temps (0,3 s, 1,5 s sous `CI`, 500 ms, 20 ms) jugent des threads réels, aussi sous TSan et ASan (×5 à ×15 plus lents). | Échec intermittent sur un runner chargé ; un échec de ce type finit en « relancer la CI » et masque un vrai défaut. | Une horloge injectée (`core/clock.h` en a déjà une pour la session) pour l'anneau et la sonde ; garder une seule borne de temps par module, marquée `[timing]` et exclue des tâches sanitizers. | Lu ; A40-05 en est un cas observé. |
| A40-14 | Mineur | `.github/workflows/cloud.yml:57-58`, `CMakePresets.json` (`windows-ci`, `linux-release`, `windows-release`) | **Ce que chaque tâche ne lance pas.** `cloud.yml` ne passe pas `check-skips.sh` : un test sauté y compte pour réussi. Les préréglages de release ne lancent ni `gpu`, ni `gpu_span_fp8`, ni `gpu_ir`, alors que ces tests n'ont pas besoin d'appareil (les tests de la porte FP8, A02-13, et le mandataire, A40-08, ne passent jamais sur le binaire livré). | Un test de `gpu_span_fp8` qui casse avec les options de la release (Release, `-O2`) n'arrête pas la publication. | Ajouter `gpu`, `gpu_ir` et `gpu_span_fp8` aux filtres de release et de `windows-ci` ; `check-skips.sh` dans `cloud.yml`. | Lu. |
| A40-15 | Mineur | `.github/workflows/ci.yml` | **Aucune mesure de couverture en CI.** Les chiffres de ce rapport n'existent que parce qu'ils ont été mesurés ici ; rien ne montre qu'un module devient moins couvert. Les fichiers les moins couverts sont ceux des bugs déjà trouvés : `chain.cpp` 10,5 % (A08-03), `player_engines.cpp` 7 % (A08-08), `engine_screen.cpp` 10 % (A08-01), `settings_panel.cpp` 2 %, `system_peak.cpp` 14,5 %. | Les zones sans test grandissent sans signal. | Une tâche de nuit `--coverage` + gcovr (40 s sur le build ici), rapport en artefact et plancher par dossier pour `src/gpu`, `src/app/play/upscale_*`, `src/prepare`. | Mesuré ici. |
| A40-16 | Mineur | `tools/ci/check-cuda.sh:2-6` | Commentaire faux : il dit que la CI ne construit pas la partie CUDA, alors que la tâche `cuda` de `ci.yml` la construit sur chaque PR. | Un lecteur croit le backend TensorRT non compilé en CI. | Corriger le commentaire. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| B3 | Toujours présent (pour TensorRT) | La CI fait tourner les tests GPU sous lavapipe (Vulkan) et compile CUDA, mais n'exécute jamais TensorRT ni CUDA (A40-02). |
| QUA-04 | Corrigé en grande partie | `.clang-tidy` : `bugprone-unused-return-value` sur `std::expected` ; tout `src/` analysé la nuit, les fichiers changés sur chaque PR ; MSVC marque `std::expected` `[[nodiscard]]` sous `/WX`. Reste : une poussée sur main n'est pas analysée ; les `.cu` et les `std::optional`/`bool` ignorés ne sont pas vérifiés. |
| SEC-22 | Toujours présent | Révocation : `updater_test.cpp:284-293` ne teste que le filtre `update_public_keys` et la relecture d'`update.toml` ; jamais le refus d'un manifeste signé par une clé qu'il révoque (`updater.cpp:429`) ni l'accumulation des révocations (`updater.cpp:507-510`). Bout en bout : A40-04. |
| Renvoi 03→40 | Confirmé | A40-02. |
| Renvoi 21→40 | Confirmé | A40-08 ; `IngestSpec::chroma` ignoré par TensorRT, vérifié seulement sous Vulkan (A40-07). |
| Renvoi 30→40 | Confirmé | A40-09. |
| Renvoi 28→40 | Confirmé | A40-01 : les essais cachés passent sous lavapipe et pourraient tourner en CI. |
| Renvoi 31→40 | Confirmé | `ring_stage_test.cpp:243` attend `ThreadFailed` après un `DeviceLost` de la sonde. Test écrit : la même scène avec `CHECK(code == RefusalCode::DeviceLost)` **échoue** (A31-01). |
| Renvoi 34→40 | Confirmé | A40-10. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 10 | `src/app/play/placebo_snapshot.cpp:16-40` | Sortie SDR (ou instantané) × 0,62 sur chaque canal, grade coupé ; « contrat d'écran : référence 203 nits, pic 1000 nits » au journal d'une sortie SDR (A40-06). |
| 41 | `tests/unit/app/e2e_play_test.cpp:241-243` | Même constat, à croiser avec le grade SDR (A40-06). |
| 22 | `src/app/play/options.cpp` (`--data-dir`) | `--data-dir` n'isole pas les moteurs : le lecteur lancé avec `--data-dir=…` écrit ses moteurs dans `~/.local/share/ajp/ajp/engines` et `…/ajp/modeles` (dossier `ajp` doublé). |
| 13 | `tests/unit/app/e2e_play_test.cpp:221` | Avec `--seconds`, le lecteur enchaîne quand même l'épisode suivant du dossier (A40-05) : voulu ou non ? |
| 46 | `tools/bench/rife_triplets.py:100` | Le banc de qualité refuse DistilDRBA (A40-03). |

## Points positifs

- La suite tourne vite et vert sous lavapipe avec la couche de validation exigée : 1 557 tests en 160 s.
- `check-skips.sh` et `skip-ceiling.txt` empêchent un test sauté de passer pour réussi (sauf A40-12, A40-14).
- `src/kernels` (94,6 %), `src/gpu/ir` (92 %), `src/prepare` (93,9 %), `src/gpu/span_fp8` (96,4 %),
  `src/subs` (96,7 %) sont bien couverts, avec des références CPU en double pour chaque noyau.
- Les étages du lecteur se testent sans GPU (`stage_rig`, `upscale_rig`) : `upscale_stage.cpp` 94 %,
  `ring_stage.cpp` 95 %, `upscale_drba.cpp` 80 %.
- Les essais cachés du fusionné sont bons quand on les lance : 60 à 74 dB contre le réseau entier sous lavapipe.

## Plan de tests, du plus rentable au moins rentable

1. **Tâche « modèles » sous lavapipe** (A40-01) : `fetch-models.sh` en cache, `[.rife_fused]`, `[.drba_fused]`,
   tests de découpe de `prepare`. Environ 5 min, couvre RIFE 4.25, 4.22 lite et DistilDRBA sur le repli Vulkan.
2. **Script NVIDIA et reçu exigé par `release-guard`** (A40-02) : la seule façon de voir TensorRT avant une release.
3. **Banc de qualité à trois modèles** (A40-03) : DistilDRBA dans `rife_triplets.py`, et le compte des passes par
   paire dans `upscale_stage_drba_test.cpp`, pour tenir « plus rapide à qualité proche ».
4. **Tests qui figent un défaut, à retourner** : `ring_stage_test.cpp:243` (DeviceLost), `backend_test.cpp:92`
   (chroma Spline36), `e2e_play_test.cpp:242-243` (luma fidèle), `source_test.cpp:400-414` (A14-08).
5. **Bout en bout avec la chaîne** (A40-04, A40-05) : un dossier par cas, SPAN + RIFE et SPAN + DistilDRBA,
   jugés sur l'image et le compte d'images interpolées.
6. **`GuardedKernelOps`, `ChainStats`, `LookLoader`** (A40-08, A40-09), avec TSan élargi.
7. **Couverture de nuit** (A40-15) et modules sans appareil dans les préréglages de release (A40-14).
8. **Horloge injectée** dans les tests de threads (A40-13).

## Questions ouvertes

- A40-06 : le facteur 0,62 vient-il de la sortie SDR à l'écran ou seulement de `--snapshot` ? Une capture de la
  fenêtre sur le PC, HDR du système coupé, tranche.
- A40-02 : Tatoub accepte-t-il un runner auto-hébergé sur son PC, ou préfère-t-il un reçu produit à la main avant
  chaque release ?
