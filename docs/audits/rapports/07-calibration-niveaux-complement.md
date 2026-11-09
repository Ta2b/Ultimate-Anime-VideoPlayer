# Complément de l'audit 07. Calibration, niveaux et mode Auto : les renvois reçus

Branche : `claude/project-thread-xrrfgt`. Commit audité : `d5df90d`. Date : `2026-11-09`.

## Verdict

Les seize renvois reçus par l'audit 07 tombent tous sur des défauts déjà inscrits à `SYNTHESE.md` (surtout A25-08,
A38-02, A49-02, A49-03, A34-01, A36-02) : aucun n'est réfuté, quinze sont des doublons, et un seul ouvre un défaut
neuf, mineur (A07-13 : la « porte de qualité » qui tient DistilDRBA hors de l'Auto n'est écrite nulle part).
Deux précisions corrigent la lecture d'A34-01 et aggravent A49-02 et A49-03.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les renvois de la liste
(`audits-docs/complements/07.md`) dans le code, conclure chacun par confirmé, réfuté ou doublon, et signaler toute
baisse de qualité d'image faite pour la vitesse sans mesure. Aucune question posée à Tatoub. Le fil tourne sans GPU :
tout est lu sur `d5df90d`, dont les numéros de ligne de `auto_config.cpp`, `machine_calibration.cpp` et
`player.cpp` sont ceux des renvois.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A07-13 | Mineur | `src/app/play/auto_config.h:19-22`, `src/app/play/auto_config.cpp:184-185` | La « porte de qualité » qui tient DistilDRBA hors de l'Auto n'existe nulle part. Le commentaire renvoie à `docs/adr/0003`, qui traite du choix entre Rapide et Stable et ne nomme pas DistilDRBA ; ni `docs/interpolation.md` (qui le classe « Très rapide »), ni un autre ADR, ni `docs/a-faire-1-0.md` ne donnent de critère, de mesure ou de responsable. Le commentaire parle aussi des « tranches Rapide et Ultra » alors que le code vise `Tier::Fast` et `Tier::Ultra`, affichés « Qualité » et « Ultra » (nommage croisé d'A07-12). Et la calibration, seul endroit où la machine est mesurée, ne l'essaie pas (A25-08) : `kDistilDrbaInAuto` ne peut passer à vrai par aucune procédure écrite. | Quelqu'un veut ouvrir l'Auto à DistilDRBA : rien ne dit quoi mesurer (PSNR contre RIFE 4.25 ? jugement à l'œil sur `calibration.mkv` ? tenue sur quelle carte ?), ni où écrire le résultat. La constante reste fausse par défaut, et DistilDRBA ne tourne que choisi à la main. | Écrire la porte dans un ADR (ou dans `docs/interpolation.md`) : la séquence, la référence (RIFE 4.25 fusionné), le seuil et la carte ; corriger le renvoi du commentaire et les noms des tranches. La mesure elle-même relève d'A25-08 (calibration). | Lu : `grep` de « porte de qualité » et de `kDistilDrbaInAuto` hors des audits, deux seuls résultats dans le code ; `docs/adr/0003-un-seul-mode-auto.md` entier. |

## Renvois traités

| # | De | Où | Conclusion | Preuve et note |
|---|---|---|---|---|
| 1 | 18 | `src/app/play/auto_config.h:22` | Doublon d'**A25-08**, plus A07-13 | La calibration n'essaie que `kCalibrationRife` (`machine_calibration.h:35`, `candidate_settings` `:145-146`). `kDistilDrbaInAuto = false` est une décision, pas une baisse de qualité (DistilDRBA est la voie rapide) ; ce qui manque est la procédure pour la lever : A07-13. |
| 2 | 03 | `src/app/play/auto_config.h:22` | Doublon d'**A25-08** et d'**A38-02** | Même constat que le n° 1, côté Auto calibré : `auto_config.cpp:203-204` pose `MotionEngine::Rife` et `rife_v4.22_lite.onnx`. |
| 3 | 22 | `src/app/play/auto_config.cpp:202` | Doublon d'**A49-03** (et TRT-08, toujours présent) | `:202` pose `PrecisionTier::Performance` dès que `cal.span_model` n'est pas vide, contre le commentaire `:167-168`. Baisse de qualité pour la vitesse : le gain mesuré du FP8 en lecture est de 6,5 % (`docs/span-fp8-cutlass.md`, cité par A49-03), sans jugement de l'image. |
| 4 | 22 | `src/app/play/auto_config.cpp:203-205` | Doublon d'**A07-06** (`auto_free`) et d'**A38-02** (modèle RIFE) | `apply_preset` réécrit `auto_free = cal.rife` à chaque vidéo, et `preset_chain_differs` (`:214-220`) ne compare pas `auto_free` : un « Stable » choisi au menu reste sous le préréglage Auto et se perd à la vidéo suivante. |
| 5 | 23 | `src/app/play/auto_config.cpp:134-147` | Doublon d'**A49-02** (cas 3), aggravé | Écran 1080p : `ideal = 540` (`:137`), donc Léger 480, Normal 540, Qualité 720 et Ultra 900 sans calibration (`kWorkOffset`, `kWorkCap`, `:48-53`). Aggravation : la calibration plafonne à la moitié de l'écran (`calibration_heights`, `machine_calibration.cpp:81-87`), donc une RTX 4090 (Ultra) **calibrée** sur un écran 1080p travaille au plus en 540p, contre 900p si elle ne l'est pas. Calibrer réduit l'image, sans mesure de qualité ni de gain. |
| 6 | 25 | `src/app/play/machine_calibration.cpp:115-135` | Doublon d'**A25-08** | `best_candidate` ne reçoit pas les réglages ; `priority` n'a plus de contrôle au panneau (lu seulement dans `play.toml`, `settings_toml.cpp:168-181`). Dans la branche calibrée, `priority` agit encore sur le plafond du facteur (`auto_config.cpp:158-159`) mais plus sur la hauteur (`:197-200`) : incohérence de portée faible, à traiter avec A25-08. |
| 7 | 25 | `src/app/play/machine_calibration.cpp:240-252` | Doublon d'**A25-02**, d'**A36-02** et d'**A07-05** | `format_calibration` (`:160-190`) n'écrit ni moteur ni jeton ; `calibrated_chain` (`:244-246`) ne compare que la carte, la hauteur d'écran et la batterie. La correction d'A07-05 (« mettre le moteur dans `calibration.toml` ») couvre ce renvoi. |
| 8 | 25 | `src/app/play/machine_calibration.cpp:288-290` | Doublon d'**A25-11** (TRT-18) | Confirmé dans le code : un refus de `save_settings` rend `std::unexpected` et arrête la boucle. La cause pratique est le dossier à nom fixe partagé entre comptes d'A25-11, dont la correction (un dossier unique par processus sous le dossier de données) suffit. |
| 9 | 49 | `src/app/play/machine_calibration.h:37-47` | Doublon d'**A49-01**, **A49-02**, **A49-03** et **A38-02** | Candidats confirmés : `kCalibrationSpanFp8` (b8f64 joué en FP8) ou le modèle par défaut b5f48 FP16, jamais le b8f64 FP16 ; hauteurs ≤ moitié de l'écran (`kCalibrationHeights`, `calibration_heights`) ; RIFE 4.22 lite seul. Voir aussi la précision 2 ci-dessous. |
| 10 | 32 | `src/app/play/auto_config.cpp:167-169` | Doublon d'**A49-03** (TRT-08) | Commentaire `:167-168` contre code `:202`, comme au n° 3. |
| 11 | 34 | `src/app/play/auto_config.cpp:192-200` | Doublon d'**A38-02** et d'**A25-08** | `:183` sort de la tranche Qualité ou Ultra avant `use_rife(…, "rife_v4.25.onnx")`, puis `:204` impose 4.22 lite. Baisse de qualité pour la vitesse, sans mesure : la calibration n'a jamais joué 4.25. |
| 12 | 34 | `src/app/play/machine_calibration.cpp:279-330` | Doublon d'**A34-01**, à corriger dans le détail | Ce n'est pas le premier essai qui est perdu, mais le premier essai SPAN à `kFp8JudgeHeight` (720) ou plus (`:311`). Voir la précision 1. |
| 13 | 36 | `src/app/play/machine_calibration.cpp:240-252` | Doublon d'**A36-02** | Un seul jeton (version du lecteur, moteur, `TrtRuntime::token()`) répond à A36-02, A25-02 et A07-05 ensemble ; rien de plus à ouvrir. |
| 14 | 38 | `src/app/play/auto_config.cpp:203-204` | Doublon d'**A38-02** | Même lecture qu'aux n° 2 et 11. |
| 15 | 42 | `src/app/play/auto_bench_run.cpp:37` | Doublon d'**A42-05** | Confirmé : les arguments d'un essai (`window_args` `:35-44`, `:92`) ne passent ni `--no-probe-nvdec` ni `--no-shot-cuts`, et `player.cpp:332` arme NVDEC pour la pré-analyse. Le clip du banc est court (`kBenchSeconds`) : la rafale l'est aussi, mais elle tombe sur le premier essai, celui qui construit les moteurs. |
| 16 | 43 | `src/app/play/player.cpp:486-487` | Doublon d'**A43-04** | Confirmé : la notification dit « optimise for this PC again (Video > Post-processing) », et sur cette page le bouton de la calibration n'est rendu que sous `if (advanced_)` (`settings_panel.cpp:485-491`), sous le nom « Run the bench ». |

## Précisions sur des défauts existants

1. **A34-01, le premier essai FP8 perdu.** Les essais SPAN seul montent du palier le plus léger (`next_candidate`,
   `machine_calibration.cpp:93`), et le FP8 n'est jugé qu'à 720 ou plus (`:311`). Sur une machine sans FP8 CUTLASS
   (Windows, ou toute carte sous Vulkan) :
   - **écran 1440p ou plus** : les essais à 360, 480 et 540 jouent le b8f64 en précision Performance, non stricte,
     donc FP8 par Q/DQ si une variante existe, sinon INT8, sinon FP16 (`span_variant_order`,
     `span_int8.cpp:194-202`) ; ils sont gardés. L'essai à 720 rend une variante autre que « FP8 CUTLASS » : il est
     jeté **même s'il a tenu** (`:324-331`), avec la construction TensorRT du b8f64 à 720 qu'il a pu coûter
     (2,5 min sur la 4070 Ti, `calibration_minutes`), et la suite passe au b5f48 FP16. Le b8f64 FP16, qui tenait
     peut-être, est remplacé par le b5f48 sans mesure : c'est la baisse de qualité d'A49-03, ici démontrée par
     le chemin du code ;
   - **écran 1080p** : aucun palier n'atteint 720, rien n'est jugé, `span_fp8` reste vrai et passe dans
     `calibration.toml` (TRT-08). L'Auto calibré, strict, joue alors le b8f64 en INT8 si un reçu existe, sinon en
     FP16 : sous Windows sur un écran 1080p, ce n'est donc **pas** le b5f48 qu'annonce A34-01, et l'écran
     « Résultats du bench » annonce un FP8 qui ne tourne pas.

   La correction d'A34-01 (mesurer les candidats en FP16, Balanced puis Performance, des deux côtés) règle les deux
   cas ; à défaut, juger le FP8 sur le premier essai SPAN quelle que soit sa hauteur, et ne jeter un essai que
   s'il ne tient pas.

2. **A49-02 et A38-02, calibrer donne une image moins bonne que ne pas calibrer.** Sur une carte Qualité ou Ultra,
   ne pas calibrer donne RIFE 4.25 (A38-02) et, sur un écran 1080p, une hauteur de travail de 720 ou 900 ; calibrer
   donne RIFE 4.22 lite et au plus 540. Aucune des deux baisses n'a de mesure de gain au dépôt : elles viennent du
   périmètre de la calibration (candidats et paliers), pas d'un essai qui aurait échoué.

## Points positifs

- Le code de la calibration se lit sans surprise : chaque renvoi se vérifie en quelques lignes, et les numéros de
  ligne cités par les autres audits sont encore exacts à `d5df90d`.
- Le choix strict ou non des variantes de SPAN (`span_variant_order`) est explicite, ce qui permet de dire
  exactement quel réseau tourne dans chaque cas ci-dessus.
