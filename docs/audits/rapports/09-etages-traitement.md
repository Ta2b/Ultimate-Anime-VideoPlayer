# Rapport de l'audit 09. Étages d'agrandissement et d'interpolation

Branche : `claude/project-thread-0dwb1b`. Commit audité : `7963225`. Date : `2026-11-09`.

## Verdict

L'étage tient son contrat sur le chemin par défaut (RIFE mouvement, poses gardées, Rapide, warp après SPAN) et
aucun défaut bloquant n'y a été trouvé. Les défauts se concentrent sur DistilDRBA (coupes jugées sans la
pré-analyse ni les bandes noires, changement de taille, économies de l'étage dont il est privé), sur le cran
Performance de l'interpolation, qui fait perdre les moteurs fusionnés, et sur un étage arrivé à la limite de taille
de la CI.

## Cadrage retenu

Réponses de Tatoub (09/11) :

1. **Ordre warp et SPAN** : Auto en priorité ; regarder si Avant et Après peuvent sortir du menu.
2. **Interpolation** : DistilDRBA ne se met pas de côté s'il peut être nettement plus rapide, comme le modèle le
   promet. RIFE 4.22 lite / 4.25 et DistilDRBA, versions fusionnées TensorRT en priorité.
3. **Précisions de SPAN** : FP8, INT8 et FP16 sur RTX ; vérifier surtout l'accord entre le choix et les capacités du
   moteur, et Windows, où le FP8 est éteint.
4. **Seuils réglés à l'œil** : signaler les incohérences **et** proposer des valeurs, marquées « à confirmer à
   l'œil » (section « Questions ouvertes »).

Les options d'essai (`--skip-held`, `--smooth-drawings`, `--rife-warm`, `rife_motion_calm`) sont lues mais classées
plus bas. L'export (`YuvSlots`) et `apple_model` sont hors cible : lus, pas creusés.

Le fil n'a ni GPU ni dépendances construites (`AJP_DEPS_PREFIX` absent, une construction des dépendances prendrait
des heures sur 4 cœurs). Aucun test n'a pu être lancé. Les tests qui démontreraient chaque défaut sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A09-01 | Majeur | `src/app/play/upscale_drba.cpp:104`, `src/app/play/upscale_drba.cpp:155`, `src/app/play/upscale_drba.cpp:77`, `src/app/play/upscale_smooth.cpp:34`, `src/app/play/upscale_stage.cpp:408` | Il existe trois façons de juger une coupe. `pair_cut` (`rife_plan.cpp:9-18`) suit la pré-analyse (8.7) et rapporte le score à la zone hors bandes noires (P3.32). Le côté « après » de DistilDRBA, son pont après une image sautée et le Smooth, eux, ne regardent que `is_scene_cut(score brut)`. Le journal des désaccords compare lui aussi au score brut. | Film en 2,39:1 (zone utile ≈ 0,74) avec DistilDRBA : une coupe dont le score brut vaut 0,10 compte comme coupe pour `pair_cut` (0,135 une fois rapporté), mais pas pour `drba_.after_ok`. Les instants t ≥ ½ de la paire qui précède la coupe lisent alors l'image du plan suivant (`run_drba`, img2 = after) : un fantôme du plan suivant paraît une image avant la coupe. Une coupe connue seulement par `--shot-cuts` a le même effet. Le journal compte de faux désaccords sur toute vidéo à bandes. | Une seule fonction `cut_between(scores, known_cut, useful_area)` (celle de `pair_cut`), appelée pour (b, après), pour le pont et pour le Smooth. Pour (b, après), interroger `live->cuts().cut_between(pts b, pts après)`. Comparer le journal au score rapporté. | Lu. Test à ajouter dans `upscale_stage_drba_test.cpp` : `rig.backend.score(2, 3, 0.10, 255)` avec une zone utile de 0,74 (ou une coupe connue entre 2 et 3), 1 à 4 en ×2. Attendu : la passe de l'instant ½ de (1, 2) se fait sans l'image 3 (`drba_.edges` augmente). Aujourd'hui, l'image 3 est lue. À confirmer à l'œil sur un film à bandes. |
| A09-02 | Mineur | `src/app/play/rife_fused_choice.cpp:11`, `src/app/play/drba_fused_choice.cpp:9` | Le cran Performance de l'interpolation coupe le moteur fusionné. Aucun graphe FP8 de RIFE n'est publié (`tools/release/publier-modeles-fp8.sh:12-13` : « plus lents ou hors des gardes de qualité »). Ceux de DistilDRBA sont « plus lents que le DistilDRBA fusionné ». « Performance » rend donc la chaîne plus lente qu'« Équilibre ». | Utilisateur sous TensorRT avec `distil_drba_v2_lite_fp8.onnx` posé à côté du modèle (groupe facultatif) qui choisit Performance : `drba_fused_declined` répond « cran de précision hors Équilibre ». Le moteur entier FP8 tourne, soit environ 1,9 ms au lieu de 0,92 ms par passe d'après les chiffres de `settings.h:207-212`. Pour RIFE, ce cran est du code mort hors fichier posé à la main. | Ne proposer Performance pour l'interpolation que s'il bat le fusionné (ou le retirer du menu de l'interpolation), ou laisser le fusionné FP16 l'emporter et le dire dans le journal. | Lu. À confirmer : `ajp --stats` avec DistilDRBA, Équilibre puis Performance (fichier FP8 posé) ; comparer le temps GPU par image dans le journal de fin. |
| A09-03 | Mineur | `src/app/play/span_auto.cpp:22-25` | Sous Windows, les noyaux FP8 sont éteints à la construction (renvoi 02→34). Une RTX 40 sous TensorRT reçoit pourtant « NVIDIA GeForce RTX 4070 Ti is not an NVIDIA RTX 40 (Ada) or RTX 50 (Blackwell) card, or the engine was built without these kernels » : la première moitié est fausse. | Windows 11, RTX 4070 Ti, TensorRT en service : la raison affichée au menu (`home_screen.cpp:182`, `player_run.cpp:236`) dit que la carte n'est pas une RTX 40. | Ajouter à `SpanFp8Facts` « noyaux absents de cette construction » (connu à la compilation) et le dire : « Performance mode is not available on Windows yet ». | Lu. À confirmer sous Windows : Réglages > Vidéo, cran Performance de SPAN. |
| A09-04 | Mineur | `src/app/play/upscale_drba.cpp:135-137` | `drba_next` ingère l'image suivante sans vérifier sa taille, alors que `look_ahead` le fait (`upscale_ahead.cpp:76`). L'ingestion TensorRT refuse une image plus petite que le tenseur (`tensorrt_kernels.cpp:107-112`, `ShapeInvalid`). | Flux qui passe de 1080p à 720p en cours de lecture (enregistrement TV, montage) avec DistilDRBA : `outputs()` de la dernière image 1080p échoue en `ShapeInvalid`. `RingStage::fail` le transforme en `ThreadFailed` et la lecture s'arrête, alors que `SourceResized` aurait rouvert la chaîne. Dans l'autre sens (720p → 1080p), l'image d'après est réduite sans respecter son format et sert à une interpolation. | Même garde que `look_ahead` : `after->width != source_width \|\| after->height != source_height` → pas d'image d'après (`return {}`). | Lu. Test : `upscale_stage_drba_test.cpp`, flux 1 à 3 dont la 3 est plus petite. Attendu : refus `SourceResized` sur la 3. Aujourd'hui : refus `ShapeInvalid` sur la 2. |
| A09-05 | Mineur | `src/app/play/upscale_motion.cpp:111`, `src/app/play/upscale_motion.cpp:189-192` | Sans poses, une paire à une seule image intermédiaire placée ailleurs qu'en ½ fait deux passes de RIFE : celle du milieu (`run_motion(0.5F)`), puis celle de l'instant, puisque `single_run_` reste faux quand `intermediates_ == 1`. Le mouvement du milieu ne sert ni au warp ni aux poses. | Mode Rapide (le défaut) avec « Garder les poses » éteint : presque chaque paire n'a qu'un instant, à t quelconque, et coûte donc deux passes. C'est aussi le cas de la seconde paire du Smooth, où les poses sont forcément éteintes (`smooth_stage_test.cpp:62-63` ne vérifie que la présence de 0,75 dans les instants de RIFE). | Quand `!poses_ && intermediates_ == 1 && t != 0.5F`, passer RIFE directement à t (comme `direct_run_`), sans passe au milieu. Lancer la mesure `flow_max_start` sur ce mouvement, mis à l'échelle comme en `direct_run_`. | Lu. Test : rig RIFE mouvement, poses éteintes, `steady` ou ×1,5. Attendu : `rife_ts()` d'une paire à t = 1/3 vaut `{1/3}`. Aujourd'hui : `{0.5, 1/3}`. |
| A09-06 | Mineur | `src/app/play/upscale_stage.cpp:363` | `frame.pts_us - previous_->pts_us` est calculé sans vérifier `kNoPts` (= `INT64_MIN`), contrairement à `gap_us` (lignes 381-383). Le débordement signé est un comportement indéfini. | Image sans horodatage (source cassée, flux brut) en lecture Stable, à l'heure, sur la grille : la soustraction déborde avant le `std::max`, et la période obtenue fausse les instants sautés en retard. | Calculer `period` à partir de `facts.gap_us`, ou sauter le bloc quand l'un des pts vaut `kNoPts`. | Lu (latent). Test UBSan : rig avec horloge, images 1 et 2, pts de 1 à `kNoPts`. |
| A09-07 | Mineur | `src/app/play/upscale_drba.cpp` (via `upscale_stage.h:221`, `upscale_stage.h:319`, `upscale_stage.cpp:511`) | DistilDRBA est privé de trois économies de l'étage. Pas de mesure en avance (`looks_ahead` exclut DistilDRBA). Pas de saut des instants passés en Rapide (`late_fate_now_` l'exclut). Pas de débandage avant SPAN : `wants_deband` l'exclut et `drba_arrive`/`drba_next` ingèrent sans débandage, si bien que l'écran débande l'image agrandie, autrement que sous RIFE. Le moteur que Tatoub veut garder pour sa vitesse paie donc plus que son réseau. | DistilDRBA fusionné en Rapide sur une scène lourde : un instant déjà passé est quand même calculé puis jeté par la présentation (le critère « 0 calculé en retard » du chantier 7.6 ne vaut pas pour lui). Avec le grade, le rendu de DistilDRBA n'a pas le même débandage que celui de RIFE. | Débander à l'ingestion de `drba_arrive` et de `drba_next` (`IngestSpec::deband`). Étendre `late_fate` à DistilDRBA, dont l'image d'après est déjà tenue par `drba_.after`. | Lu. À confirmer : `ajp --stats`, DistilDRBA puis RIFE sur la même scène en Rapide : compter dans le journal de fin les instants « calculés après leur instant ». |
| A09-08 | Mineur | `src/app/play/settings.h:114-117` (critère), `src/app/play/upscale_stage.h:12-13` | Auto compare la hauteur de **sortie** à 1080. Avec la taille de travail par défaut (720), la sortie fait 1440 : Auto vaut toujours Après, et Avant n'est retenu que sous 541 lignes de travail en ×2. Le libellé « après SPAN au-dessus de 1080p » se lit comme une hauteur de source. L'en-tête de l'étage dit encore « RIFE … avant SPAN ». | Tatoub laisse Auto en pensant que SPAN nettoie le warp en 720p → 1440p. C'est l'inverse : le warp se fait à 1440p sur les images agrandies, sans le nettoyage de SPAN. | Fonder le critère sur la hauteur de travail, ou retirer Avant et Après du menu en gardant Auto et `span_order` dans `play.toml` (voir « Questions ouvertes »). Corriger l'en-tête. | Lu (`settings_test.cpp:147-148` confirme Auto à 1440). |
| A09-09 | Mineur | `src/app/play/upscale_stage.cpp:154-161`, `src/app/play/upscale_stage.cpp:198` | L'étage suppose `kSpanFactor` (×2) pour tout `--model` et ne lit jamais la forme de sortie du moteur (renvoi 04→09, A04-06). | `--model` d'un ×4 ou d'un ×1 : la chaîne alloue des slots ×2, puis le moteur écrit une autre taille dans `output_`. L'erreur sort de `enqueue` en plein vol (ou pas du tout, selon le backend), au lieu d'un refus clair au démarrage. | Dans `start`, comparer `engine->output_shape()` (ou la forme lue par `read_model`) à `work × kSpanFactor`, et refuser par `ShapeInvalid`. | Lu. |
| A09-10 | Mineur | `src/app/play/derived_model.cpp:19`, `src/app/play/motion_model.cpp:90`, `src/app/play/motion_model.cpp:121`, `src/app/play/model_choice.cpp:18`, `src/app/play/model_choice.cpp:21`, `src/app/play/model_choice.cpp:84-85` | Sept `path::string()` (renvoi 04→09, comme A04-04). Sous MSVC, la conversion vers la page ANSI lève une exception sur un caractère hors page de code. | Windows 11 en français, un fichier `アニメ.onnx` posé dans le dossier des modèles : `choose_in` énumère le dossier (`model_choice.cpp:21`) et lève une exception à l'ouverture de toute vidéo. Un `--rife-model` au nom japonais casse `derived_model`. | `core::to_utf8(path.filename())` partout (la règle de la CI veut que leur nombre baisse). | Lu. À confirmer sous Windows avec un fichier au nom japonais dans le dossier des modèles. |
| A09-11 | Mineur | `src/app/play/derived_model.cpp:18-22` | Les modèles dérivés ne sont jamais purgés. Il y en a un par empreinte de source (variante mouvement) et un par taille remplie de DistilDRBA (`static_distil_drba`, environ 10 Mo chacun). | Chaque définition de vidéo différente laisse un `distil_drba_v2_lite-<l>x<h>-<sha>.onnx`, et chaque mise à jour de modèle laisse les anciens. Le dossier de données grossit sans fin. | Purger au démarrage les dérivés dont l'empreinte n'est plus celle d'un modèle présent, et garder les N dernières tailles. | Lu. |
| A09-12 | Mineur | `src/app/play/upscale_stage.cpp:600-605` | Avec DistilDRBA et `--skip-held`, `skip()` ne met `previous_` à jour que pour la source (t ≥ 1). `late()` le fait pour la dernière sortie de la paire, quelle qu'elle soit (`upscale_late.cpp:37-38`). | Auto à facteur fin (×1,25, pas 30/24) : une paire n'a que t = 0,4, côté a tenu, donc `Same`. La paire suivante prend alors pour image précédente celle d'avant : écart doublé dans `auto_pair` et `left` faux dans le `RingFrame`. | Dans la branche DistilDRBA de `skip`, poser `previous_` quand `output + 1 >= outputs_`. | Lu (option d'essai). |
| A09-13 | Mineur | `src/app/play/upscale_motion.cpp:39`, `src/app/play/upscale_motion.cpp:55-60` | P6.15 (`--rife-warm`) est sans effet sur la cible : seul le RIFE fusionné de Vulkan implémente `next_warm` (`gpu/vulkan/backend/rife_fused_engine.h:82`). De plus, `warm_from_` n'est oublié qu'aux coupes et aux sauts, pas après des paires tenues ou non interpolées entre deux passes. | TensorRT : `warm_runs_` reste à 0. Vulkan : après trois dessins tenus, la passe du milieu reprend les blocs grossiers d'une paire vieille de quatre sources. | Remettre `warm_from_` à `nullptr` dans `outputs()` quand la paire ne tourne pas. Dire au journal que l'essai est inerte sous TensorRT. | Lu (renvoi 03→09). |
| A09-14 | Mineur | `src/app/play/upscale_stage.cpp` (776 lignes), `src/app/play/upscale_stage.h` (504 lignes) | `upscale_stage.cpp` est à 24 lignes de l'échec de la CI (800) ; l'en-tête dépasse la cible de 500. `outputs` a une CCN de 172 sur 265 lignes (`tools/ci/complexity-baseline.txt:42`) et mêle grille, retard, mesures, coupes, Auto, Smooth, poses et DistilDRBA. | Le prochain correctif de l'étage (A09-01 ou A09-05) fera échouer la CI, ou sera écrit à l'économie de lignes. | Sortir de `outputs` la grille (`step_for`), le retard Stable (`late_count`), les mesures et coupes (`measure_pair`) et la tenue (`settle_hold`), chacun dans son fichier `upscale_*.cpp`, comme `upscale_late`. | Lu. |
| A09-15 | Mineur | `src/app/play/rife_plan.h:7`, `src/app/play/upscale_stage.cpp:247`, `src/app/play/span_auto.h:4` | Commentaires faux. Le bloc des dessins tenus fait 16 × 16 depuis P3.23 (`detect.h:41`), pas 32 × 32. Une analyse refusée ne « coupe » pas l'analyse : elle est jetée et la suivante continue. La préparation FP8 ne « rouvre » plus la chaîne (P3.24, `span_auto.cpp:88`). | Un lecteur du code règle un seuil d'après le commentaire. | Corriger les trois commentaires. | Lu. |
| A09-16 | Mineur | `tests/unit/app/upscale_stage_test.cpp`, `tests/unit/app/upscale_stage_drba_test.cpp` | Plusieurs branches de l'étage n'ont aucun test. `copy_pair_` (b écrite à ½ après un dessin tenu, poses et `--skip-held`), l'image sautée de DistilDRBA et son pont (`keep_skipped`, `bridge_ticket`), un changement de taille avec DistilDRBA, `rife_motion_calm`, l'écriture YUV, et le nombre de passes de RIFE pour une seule image intermédiaire sans poses. | Les défauts A09-04, A09-05 et A09-12 seraient sortis d'un test. | Ajouter les tests décrits dans A09-01, A09-04, A09-05 et A09-12. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| SEC-21 | Toujours présent, hors des fichiers de la fiche | `tests/fuzz/corpus/onnx/seed.bin` fait 16 octets. `tests/fuzz/fuzz_parsers.cpp:33-37` ne fuzze que `gpu::ir::load_onnx`, pas `prepare::scan_onnx`, que `motion_model.cpp:66` et `:83` appellent sur des fichiers fournis par l'utilisateur. `codeql.yml` tourne en `build-mode none`. Déjà couvert par A04-08 et le renvoi 04→28. |
| Renvoi 03→09 (`upscale_motion.cpp:39`) | Confirmé en partie | Voir A09-13. L'invalidation sur coupe et saut est correcte (`forget_motion`, appelé à `upscale_stage.cpp:385` et `:422`). Pas d'invalidation après des paires tenues. Reprise à chaud absente sous TensorRT. |
| Renvoi 03→09 (`player_engines.cpp:247-266`) | Confirmé, et aggravé côté étage | Avec la taille de travail par défaut, Auto vaut Après (A09-08). DistilDRBA y prend la variante mouvement (`chain_setup.cpp:376-377`), servie par le fusionné à deux sorties : la branche non testée d'A03-03 est donc **le** chemin par défaut de DistilDRBA sous TensorRT. Côté étage, `interpolate_drba_motion` est couvert par des doubles (`upscale_stage_drba_test.cpp:70`, `:89`), mais pas le calcul du moteur. |
| Renvoi 04→09 (`derived_model.cpp:19`) | Toujours présent | A09-10. |
| Renvoi 04→09 (`motion_model.cpp:90`) | Toujours présent | A09-10. Le `stem().string()` de la ligne 121 aussi. |
| Renvoi 04→09 (`upscale_stage.cpp:198`) | Toujours présent | A09-09. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 23 | `src/app/play/home_screen.cpp:140` | `fp8_engine` vaut « NVIDIA et cœurs tensoriels », donc vrai pour une RTX 20 ou 30, qui n'ont pas le FP8. `player_run.cpp:230` lit `capabilities().fp8`. Le menu de l'écran d'accueil peut proposer Performance pour RIFE ou DistilDRBA, que la chaîne refuse ensuite (`chain_setup.cpp:418`). |
| 22 | `src/app/play/settings.h:114-117` | Le critère d'Auto est sur la hauteur de sortie (A09-08). La fonction est dans la zone 22. |
| 08 | `src/app/play/chain_setup.cpp:420-423` | `tier_variant` est choisi sans lien avec le fusionné : Performance remplace le RIFE fusionné par un moteur entier (A09-02). |
| 41 | `src/app/play/upscale_stage.cpp:244-271` | Les statistiques du grade sont prises sur l'image débandée, et pas du tout débandée sous DistilDRBA (A09-07) : la même scène peut se grader autrement selon le moteur d'interpolation. |
| 31 | `src/app/play/ring_stage.cpp:108-112` | Seul `SourceResized` rouvre la chaîne : toute autre erreur d'ingestion d'une image de taille nouvelle arrête la lecture (voir A09-04). |

## Points positifs

- La rotation des tampons (`rgb_`, `ahead_`, `drba_.before/after`, `spanned_`) se fait sans copie et reste juste
  dans les cas suivis à la main : Smooth avec warp après SPAN, image sautée de DistilDRBA puis reprise.
- Les tickets de mesure périmés sont rattrapés sans erreur (`pair_scores` retombe sur `scores_now`). `analyze_source`
  borne sa file à `kPendingReadbacks`.
- Les refus se propagent proprement jusqu'à `RingStage::fail`, qui relâche les crédits et vide les images en attente.
- Les tests à doubles (`stage_rig`) couvrent bien la grille, le retard, les voies, les poses et le Smooth.
- Les compteurs du journal de fin (passes, paires, coupes, désaccords, attentes hôte) rendent l'étage vérifiable sur
  la machine de Tatoub sans outil de plus.

## Questions ouvertes

Propositions de seuils (cadrage 4.b), toutes **à confirmer à l'œil** sur la machine de Tatoub, avec le journal de
fin (`ajp --stats`) :

1. **Ordre warp et SPAN.** Proposition : retirer Avant et Après du menu, garder Auto, et fonder Auto sur la hauteur
   de travail : Avant si le travail fait ≤ 720 lignes **et** si le facteur le plus haut est < 3, Après sinon.
   `span_order` resterait lu dans `play.toml` pour la compatibilité. Comparaison : une scène de panoramique 720p à ×2,
   Avant puis Après, en regardant les contours (le warp à 1440p double-t-il les traits que SPAN effaçait ?) et le temps
   GPU par image. En Rapide sur un écran à 144 Hz ou plus, le facteur dépasse 3 et Après reste de toute façon le choix.
2. **Plancher des coupes de la pré-analyse** (`kPlanCutFloor` = 0,09, `rife_plan.h:42`) : le comparer au score
   **rapporté** à la zone utile, comme la mesure, et non au score brut. Sur un film à bandes, 0,09 brut vaut
   aujourd'hui environ 0,12 rapporté, soit le seuil de RIFE lui-même : le plancher n'y sert plus. Aucune nouvelle
   valeur.
3. **Seuil de coupe** (`kSceneCutThreshold` = 0,120) : garder. Les 30 désaccords relevés (`rife_plan.h:38-41`) le
   justifient. À revoir seulement si A09-01 corrigé montre de nouveaux désaccords.
4. **Dessin tenu** (`kHeldThreshold` = 0,70 niveau, blocs 16 × 16) : garder pour le 8 bits. Proposition d'essai pour
   les sources 10 bits peu bruitées : 0,55, qui reste au-dessus du plancher de bruit de 0,40. Comparaison : compteur
   « dessins tenus » du journal et clignements d'yeux lointains sur un épisode en 10 bits, à 0,70 puis 0,55.
5. **Une passe RIFE par paire** (`kSingleRunMotion720` = 15 px) : garder avec les poses (le défaut). Sans les poses,
   la question disparaît si A09-05 est corrigé.
6. **Reprise de SPAN sur un dessin tenu** (`kMaxSpanReuse` = 2) : garder. Un fondu lent ne fige pas l'image plus de
   deux sources.
7. **DistilDRBA demi-flot** (`drba_half_flow` : travail ≥ 1440) : cohérent avec la taille de travail par défaut, à
   720. Aucune proposition.

Autres questions :

- DistilDRBA (cadrage 2) : avec A09-07 corrigé, mesurer DistilDRBA fusionné contre RIFE fusionné en Rapide, à
  qualité jugée égale, pour décider lequel est le défaut sur RTX.
