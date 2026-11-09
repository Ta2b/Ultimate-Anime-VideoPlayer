# Rapport de l'audit 03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)

Branche : `claude/project-thread-t2egtr` (le fil coordinateur demandait `audit/03-interpolation-tensorrt` ; la
session ne peut pousser que sur sa branche désignée). Commit audité : `0be463d`. Date : `2026-11-09`.

## Verdict

Les noyaux de colle de RIFE et de DistilDRBA sont justes et bornés (formes vérifiées au chargement, indices tenus
dans les plans, flot hors image ramené au bord) ; les défauts sérieux sont ailleurs : l'ingestion TensorRT
rééchantillonne la chroma en bilinéaire au lieu du Spline36 du contrat, et la concordance des chaînes fusionnées
avec le réseau entier n'est vérifiée que par des tests cachés, au seuil lâche pour RIFE et absents pour la
variante mouvement de DistilDRBA.

## Cadrage retenu

La question de cadrage (RIFE d'abord, RIFE et DistilDRBA à égalité, ou lister aussi le code à retirer) a été
posée à Tatoub ; sa consigne générale étant de ne pas laisser le travail en pause, l'audit a suivi l'option
recommandée : **RIFE 4.22 lite en mode Auto d'abord** (le chemin de la cible), DistilDRBA ensuite, la dette
couverte par l'axe 9 sans en faire le centre. Rien n'est sauté. Le fil tourne sans GPU NVIDIA ni CUDA : aucun
fichier de la zone (`.cu`, TensorRT) ne se compile ici, aucun test n'a pu être lancé ; ce qui demande une mesure
est marqué « à confirmer » avec la manipulation.

Remarque de périmètre : `slice_kernels.cu` et `slice_kernels.h` (ingestion NV12/P010 vers RGB, réductions,
écriture dans l'anneau) sont rattachés à cette zone sans rapport avec l'interpolation ; ils ont été audités ici
puisque la fiche les liste.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A03-01 | Majeur | `src/gpu/cuda/backend/slice_kernels.cu:24-45`, `src/gpu/cuda/backend/tensorrt_kernels.cpp:95-145` | L'ingestion du backend TensorRT rééchantillonne la chroma 4:2:0 en **bilinéaire**, et ignore `IngestSpec::chroma` : le contrat (`src/gpu/kernel_ops.h:49-52`, `kChainChromaFilter = Spline36` dans `src/kernels/spec/resample.h:110`) et le chemin Vulkan prennent Spline36. L'en-tête (`slice_kernels.h:7-9`) dit ces noyaux « provisoires » en attendant le socle `pc/cuda-compute`, qui n'existe pas dans `src/gpu/cuda/` | Épisode en NV12 lu sur une RTX sous TensorRT : les contours colorés (traits rouges, cheveux sur fond uni) arrivent à SPAN plus mous et légèrement baveux qu'avec le même épisode sous Vulkan ; l'écart « sous un niveau 8 bits sur les aplats » de l'en-tête ne dit rien des bords, là où l'anime vit | Porter la chroma Spline36 (poids de `kernels::ref::compute_weights`, déjà utilisés par `tensorrt_export.cpp:100`) dans `rgb_at`, en séparable ou en tuile comme `ingest_tiled` ; ou refuser `spec.chroma != Bilinear`. Mettre à jour l'en-tête | Lu. À confirmer à l'œil : même plan coloré, `--snapshot` sous TensorRT puis sous `--backend vulkan`, comparer les bords de chroma (ou PSNR de la chroma contre `kernels::ref::yuv_to_rgb`) |
| A03-02 | Majeur | `tests/unit/backend_tensorrt/rife_fused_test.cpp:397,414,421`, `tests/unit/backend_tensorrt/drba_fused_test.cpp:403` | La concordance « chaîne fusionnée contre moteur entier » n'est vérifiée que par des tests **cachés** (`[.rife_fused]`, `[.drba_fused]`), jamais lancés par `ctest` ni par la CI (sans GPU), et le seuil de RIFE est de **30 dB** de PSNR alors que l'en-tête des noyaux promet un écart « qui ne vient que de l'ordre des sommes » (`rife_fused_kernels.h:7-8`) ; DistilDRBA tient 50 dB. Le seuil vient de la référence choisie : le moteur entier change de tactiques d'une construction à l'autre (34 à 56 dB mesurés, commentaire `:394-396`) | Une régression de la colle (maillage décalé d'un demi-pixel, arrondi FP16 du temps oublié, échelle du flot d'un bloc) baisse le PSNR de 50 à 35 dB : le test, s'il est lancé, passe ; s'il ne l'est pas, rien ne la voit, et RIFE fusionné est le chemin par défaut de la cible | Comparer la chaîne fusionnée à une référence stable : le moteur entier construit une fois avec un cache de tactiques figé, ou la référence CPU `kernels/ref/rife_fused.h` sur une petite taille, avec un seuil proche de celui de DistilDRBA (45 à 50 dB) ; inscrire le lancement des tests cachés dans la liste d'avant release (`docs/a-faire-1-0.md`) | Lu. Test à écrire : `compare_engines` contre la référence CPU à 64 × 128, seuil 45 dB, qui échoue si on retire l'arrondi de `mesh_at` (`rife_fused_kernels.cu:22-27`) |
| A03-03 | Majeur | `src/gpu/cuda/backend/drba_fused_kernels.cu:284-290`, `src/gpu/cuda/backend/drba_fused_engine.cpp:174-179`, `src/app/play/player_engines.cpp:247-266` | La **variante mouvement** de DistilDRBA fusionné (deux sorties : flot et masque, branche `p.flow != nullptr` de `finish_kernel`) sert en lecture dès que `drba_motion` est vrai (la chaîne fusionnée est rendue sans toucher à `*motion`), mais **aucun test** ne la compare au graphe mouvement entier (`with_drba_motion_outputs`) et l'essai au chargement ne lance que la sortie image | DistilDRBA choisi, warp après SPAN : une erreur dans l'ordre des plans de flot ou dans le facteur ×t de la branche mouvement donne des images intermédiaires déformées, que ni le chargement (« un noyau qui ne tourne pas se voit ici ») ni un test ne voit | Lancer l'essai de `load` avec les deux formes de sortie ; ajouter au test caché `[.drba_fused]` une section mouvement contre le DistilDRBA entier dérivé par `drba_motion_model`, comparée après warp | Lu (`grep motion tests/unit/backend_tensorrt/drba_fused_test.cpp` : 0 ligne). À confirmer sur la RTX par ce test |
| A03-04 | Mineur | `src/gpu/cuda/backend/tensorrt_rife.cpp:118-124`, `src/gpu/cuda/backend/tensorrt_backend.cpp:419` | `rife_time` n'écrit que l'élément 0 du plan de l'instant quand le **dernier** moteur qui a lu ce tampon était RIFE fusionné ; rien ne garantit que le **prochain** lecteur l'est aussi. Un moteur entier lancé ensuite sur le même tampon lit un plan d'instant périmé (celui du dernier `rife_input`/`rife_pair`) | Aujourd'hui latent : le RIFE des paires calmes (`--rife-calm-model`, essai) est toujours un moteur entier et partage `rife_in()` avec le RIFE mouvement fusionné ; il suffit qu'un changement de moteur tombe entre deux instants d'une même paire (ou un futur repli du fusionné vers l'entier en cours de lecture) pour qu'une image intermédiaire soit calculée au mauvais instant | Garder l'instant posé par tampon et, à l'`enqueue` d'un moteur non fusionné sur un tampon marqué compact, réécrire le plan entier avant de lancer ; ou décider « compact » à l'`enqueue` plutôt qu'au `rife_time` | Lu. À confirmer : test unitaire CUDA `rife_time` compact puis `enqueue` d'un moteur entier, sortie comparée à `rife_input` au même instant |
| A03-05 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:175-194`, `src/gpu/cuda/backend/drba_fused_engine.cpp:168-179` | L'essai au chargement tourne sur un flux privé non bloquant et écrit dans la zone d'activation **partagée** de la voie, sans aucun ordre avec les flux du backend | Aujourd'hui chaque chaîne crée son backend et charge ses moteurs avant la lecture, donc rien n'est en vol ; mais avec une seule voie la zone est celle de SPAN : un travail encore en file sur `stream_` (essai ou allocation de SPAN, rechargement à chaud futur) et l'essai de RIFE s'écrivent dessus en même temps | Lancer l'essai sur le flux de la voie (`stream_`), ou faire attendre le flux d'essai sur un événement posé sur toutes les voies | Lu. À confirmer : `compute-sanitizer --tool racecheck` n'aide pas entre flux ; relire `TrtEngine::load` et `SpanFp8Engine::load` (audit 01) pour savoir si SPAN laisse du travail en vol |
| A03-06 | Mineur | `src/gpu/cuda/backend/tensorrt_rife_fused.cpp:388`, `src/gpu/cuda/backend/tensorrt_drba_fused.cpp:23` | Message de refus construit avec `model.onnx_path.string()` : sous Windows, la conversion vers la page de code ANSI lève `std::system_error` pour un caractère hors de cette page, au lieu de rendre le refus ; contraire à la règle `core::to_utf8` de `CLAUDE.md`. Même usage, moins exposé, pour `onnx.stem().string()` (`:54`, `:52`) et `filename().string()` (`:70`, `:68`) | Windows 11 en français, profil `C:\Users\さくら\` : le manifeste devient illisible (antivirus, disque), l'exception remonte au lieu du repli sur le RIFE entier | `core::to_utf8(...)` partout dans ces deux fichiers | Lu |
| A03-07 | Mineur | `src/gpu/cuda/backend/rife_kernels.cu:49-50`, `src/gpu/cuda/backend/tensorrt_rife.cpp:48-89` | Performance de l'entrée de RIFE (fiche : 0,4 à 0,8 ms). `rife_pair` écrit le plan de l'instant en entier (1 plan sur 7 écrits) alors que RIFE fusionné n'en lit que l'élément 0 ; la décision « compact » n'est appliquée qu'à `rife_time` | Chaque paire en 4.22 lite à 1280 × 768 : environ 2 Mo écrits pour rien sur 14 | Appliquer à `fill_rife` la même décision compacte que `rife_time` (écrire l'élément 0 seul, sous la réserve d'A03-04) | Lu. À confirmer : `nsys` sur 300 paires, temps de l'étape « entrée de RIFE » avant et après |
| A03-08 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:221-224`, `src/gpu/cuda/backend/tensorrt_rife.cpp:69-87` | Piste plus large pour fondre l'entrée : à chaque paire, `rife_pair` recopie les **deux** images dans le tenseur rempli et l'encodeur TensorRT réencode les **deux** (lot de 2), alors que l'image A de la paire n + 1 est l'image B de la paire n, déjà copiée et encodée | Lecture continue : la moitié de la copie d'entrée et la moitié de l'encodeur refaites à chaque paire | Encodeur construit en lot de 1 ; deux emplacements d'image et d'encodage hors de la zone partagée (SPAN l'écrase entre deux paires), en alternance ; les noyaux `fused_entry`/`fused_glue`/`fused_finish` reçoivent les adresses d'A et de B séparément. Le padding noir, constant, s'écrit une fois | À confirmer : `nsys`, part de l'encodeur et de `rife_input` dans une passe RIFE 4.22 lite à 720p, sur la 4070 Ti |
| A03-09 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:323-324`, `src/gpu/cuda/backend/drba_fused_engine.cpp:345-346` | Le cache de graphes CUDA s'arrête à 8 (16) jeux d'adresses sans jamais en retirer : au-delà, chaque nouveau jeu tourne en lancement direct, pour toujours et sans notice | Aujourd'hui au plus 6 jeux (2 entrées × 3 sorties) : latent ; un changement futur qui réalloue les tampons en cours de lecture ferait perdre les graphes en silence | Remplacer l'entrée la plus ancienne, ou poser une notice au premier débordement | Lu |
| A03-10 | Mineur | `src/gpu/cuda/backend/rife_fused_engine.cpp:1-383`, `src/gpu/cuda/backend/drba_fused_engine.cpp:1-407` | Dette : les deux moteurs fusionnés dupliquent environ 150 lignes (`Scratch`, `parse_int`, `split`, `tensor_of`, cache, capture et rejeu des graphes), avec déjà des écarts : DistilDRBA ignore le retour de `cudaGraphExecDestroy` (`drba_fused_engine.cpp:361`, RIFE passe par `cuda_warn`, `rife_fused_engine.cpp:337`), `load` ne remet à zéro ni `launch_failures_` ni `kept_` (`drba_fused_engine.cpp:182-184` : `kept_` garde les adresses des tampons d'essai libérés, sans effet aujourd'hui grâce au protocole de l'appelant) | Une correction de la capture faite dans un moteur et oubliée dans l'autre | Un `FusedGraphCache` commun (clé d'adresses, capture, rejeu, compteurs, notice) ; remise à zéro complète après l'essai | Lu |
| A03-11 | Mineur | `src/gpu/cuda/backend/rife_fused_kernels.cu:241` | Le sigmoïde de la fin de RIFE prend `__expf` (approché) quand DistilDRBA prend `expf` (`drba_fused_kernels.cu:283`) ; l'en-tête promet un écart au graphe venu « de l'ordre des sommes » seulement | Écart d'un ulp FP16 sur le masque, rarement ; fragilise surtout la promesse du commentaire | `expf`, ou corriger le commentaire | Lu |
| A03-12 | Mineur | `src/gpu/cuda/backend/slice_kernels.cu:268-273` | Dans `ingest_tiled`, une prise hors de la tuile est sautée **après** que son poids a été compté dans `total` : la normalisation assombrirait le pixel sans le dire. La marge de `tile_rows_for` (`:343-348`) l'évite aujourd'hui (commentaire « jamais ») | Un changement de la marge ou du rayon d'un filtre ferait des lignes sombres en bas de tuile | Compter le poids après le test de bornes, ou `assert` côté hôte que la tuile couvre ses prises | Lu |

## Défauts connus revérifiés

La fiche n'en liste aucun et `SYNTHESE.md` ne contient encore ni défaut ni renvoi adressé à 03.

| ID | Statut | Note |
|---|---|---|
| — | — | Aucun défaut connu, aucun renvoi reçu |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 01 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:80-91` | `SharedActivations::reserve` passe par `cudaMallocAsync` puis `cudaStreamSynchronize(stream_)` : chaque croissance de la zone en cours de lecture vide la file de la voie (à-coup) ; et l'essai des moteurs fusionnés (A03-05) suppose que rien n'est en vol |
| 01 | `src/gpu/cuda/backend/tensorrt_textures.cpp:15-78` | Les objets texture du warp sont gardés par adresse et dimensions ; un tampon libéré puis réalloué à la même adresse réutilise l'objet (sain tant que la forme est la même) : à confirmer avec la durée de vie des tampons |
| 09 | `src/app/play/upscale_motion.cpp:39,55-60`, `src/app/play/upscale_stage.cpp:385,422` | Reprise à chaud (`next_warm`, P6.15) : sous TensorRT, `RifeFusedEngine` ne la fait pas (défaut `Engine::next_warm` → faux, documenté dans `docs/phase6-a-tester.md:14,24`) ; l'invalidation sur coupe et sur saut (`forget_motion`) reste à vérifier côté étage et côté Vulkan |
| 09 | `src/app/play/player_engines.cpp:247-266` | Quand DistilDRBA fusionné est pris, `*motion` reste à sa valeur demandée : le choix du warp après SPAN repose alors sur la branche mouvement non testée (A03-03) ; vérifier que le journal dit bien quelle variante tourne |
| 15 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:142-143` | Position de chroma : seuls `Center` et `TopLeft` sont distingués ; `Top`, `Bottom`, `BottomLeft` retombent sur gauche-milieu (en lien avec MED-04) |
| 40 | `tests/unit/backend_tensorrt/` | Toute la suite CUDA (`ajp_backend_tensorrt_tests`) ne tourne que sur un poste NVIDIA, à la main ; aucun garde-fou de CI pour la colle des chaînes fusionnées (A03-02, A03-03) |

## Points positifs

- Les formes de chaque morceau TensorRT sont vérifiées contre le découpage au chargement
  (`rife_fused_engine.cpp:133-148`, `drba_fused_engine.cpp:123-139`), et `rife_fused_size_valid` /
  `drba_fused_size_valid` garantissent que toutes les sorties de blocs sont entières : aucun dépassement de borne
  trouvé dans `fused_entry`, `fused_glue`, `fused_finish`, `pre_kernel`, `post_kernel`, `glue_kernel`,
  `finish_kernel` (indices recalculés pour 4.25, 4.22 lite, t = 4 et t = 8).
- Flot hors image et flot non fini : toutes les lectures bilinéaires bornent les coordonnées par
  `fminf(fmaxf(…))`, qui rend la borne pour un NaN (`rife_fused_kernels.cu:75-88`, `drba_fused_kernels.cu:77-94`,
  `bilinear.cuh:30-61`) ; `flow_max` range un infini dans la dernière case et ignore un NaN.
- Instants différents de ½ : le warp passe en loi « estimé au milieu » dès que t ≠ ½
  (`tensorrt_warp.cpp:129`), et le côté choisi de DistilDRBA lu sur le périphérique concorde avec celui de l'hôte
  (`drba_call` : τ ≥ 1 avant ½, τ < 1 après).
- La clé d'artefact refuse tout nom de moteur qui sortirait du dossier du cache
  (`rife_fused_engine.cpp:86-89`, `drba_fused_engine.cpp:78-80`).
- Reprise D6 de DistilDRBA bien gardée : mêmes adresses, même zone, même génération, et personne d'autre n'a
  écrit dans la zone (`drba_fused_engine.cpp:311-313`).
- Graphes CUDA : erreur collante distinguée d'un refus, coupure après trois échecs avec une notice au journal.

## Questions ouvertes

1. Le remplissage de RIFE pose un noir à 0 (`tensorrt_rife.cpp:77`) ; la spécification (`kernels/spec/rife.h:58-59`,
   `std.AddBorders`) pose Y = 16 quelle que soit la plage, soit 16/255 après la matrice pour une source en plage
   pleine. L'écart est-il voulu (sources pleine plage très rares en anime) ?
2. Le socle `pc/cuda-compute` annoncé par `slice_kernels.h` et `tensorrt_backend.h:8` est-il abandonné ? Si oui, A03-01
   se corrige dans `slice_kernels.cu` et l'en-tête « provisoire » doit partir.
3. A03-08 demande de construire l'encodeur de RIFE en lot de 1 : acceptable pour le cache de moteurs déjà construits
   chez les utilisateurs (nouvelle clé, nouvelle construction au premier lancement) ?
