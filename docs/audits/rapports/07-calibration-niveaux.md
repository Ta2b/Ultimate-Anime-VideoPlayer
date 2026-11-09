# Rapport de l'audit 07. Calibration, niveaux et mode Auto

Branche : `claude/project-thread-bnlnqo`. Commit audité : `37a8431`. Date : `2026-11-09`.

## Verdict

L'Auto calibré n'applique pas la chaîne qu'il a mesurée : quatre des cinq défauts connus sont toujours là (TRT-01,
TRT-08 et TRT-09 démontrés par un test qui échoue), et la recherche de la calibration croit tenir SPAN + RIFE
alors que le mode Rapide a pu couper RIFE pour tenir. TRT-19 est corrigé.

## Cadrage retenu

Question posée à Tatoub : sur quoi porter en priorité. Réponse (carte de décision) : **l'Auto calibré sur une
RTX d'abord** (la 4070 Ti : chaîne mesurée contre chaîne appliquée, FP8, niveaux), puis la table des cartes et
les portables. Rien n'est mis de côté.

Le fil tourne sans GPU ni dépendances du dépôt. `auto_config.cpp`, `machine_calibration.cpp`, `auto_bench.cpp`,
`gpu_tier.cpp`, `auto_memory.cpp`, `cost_model.cpp` et `auto_watch.cpp` ont été compilés avec g++ 13
(en-têtes de toml++ 3.4, SDL 3.2 et Catch2 3.7 récupérés, journal remplacé par une doublure muette). Les
tests existants du module passent dans ce montage (hors deux qui écrivent sur disque par une doublure qui n'écrit
pas). Les tests de l'annexe ont été lancés : **tous échouent**, comme attendu. `calibrate_machine` y tourne en
entier, ses essais (`play_variant`) remplacés par une doublure qui rend des mesures choisies. Le reste est lu, ou
« à confirmer » avec la manipulation.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A07-01 | Majeur | `src/app/play/machine_calibration.cpp:139-151` | Les essais de la calibration ne jouent pas la chaîne que l'Auto calibré appliquera. `candidate_settings` garde `span_order`, `keep_poses`, `motion_precision` et `restore_lines` de l'utilisateur, que `apply_preset` force à Auto, vrai et Équilibre (`auto_config.cpp:153-170`). Les essais sont en Personnalisé, donc non stricts (`chain_setup.cpp:295`) : ils peuvent jouer un SPAN FP8 par Q/DQ ou un FP8 CUTLASS non jugé à cette taille (`chain_setup.cpp:337`, « joué pour le bench seulement »), que l'Auto strict refuse ensuite. | Un utilisateur a `span_order = Before` et `keep_poses = false` dans `play.toml`. La calibration mesure SPAN + RIFE avec le warp avant SPAN ; l'Auto calibré joue l'ordre Auto (warp après SPAN dès ×3) et garde les poses. Autre cas : un paquet FP8 refusé à 540p est joué par l'essai (léger), la lecture repasse en FP16 (plus lourd) avec les bornes du FP8. | Faire jouer aux essais la chaîne d'`apply_preset` elle-même (préréglage Auto, `Machine::calibrated` = le candidat), en strict, et ne jamais prendre `unjudged` pendant une calibration. | Démontré pour `span_order` et `keep_poses` (test A07-02) ; le FP8 non jugé est lu. |
| A07-02 | Majeur | `src/app/play/machine_calibration.cpp:338` | « Tient » ne regarde que les images perdues et le retard (`holds`, `auto_bench.cpp:24-29`). En mode Rapide, le contrôleur baisse le facteur au lieu de perdre des images, et ces images non interpolées comptent à part (`frame.uninterpolated`, `session.cpp:684` et `frame_stats.cpp:81`), jamais comme manquées. La cadence obtenue (`trial.fps`) ne sert qu'à décider un second passage (`needs_second_run`), pas le verdict. | 4070 Ti, écran 1080p : SPAN à 540p tient seul, puis SPAN + RIFE à 540p : RIFE ne suit pas, le Rapide redescend vers ×1, 26 i/s obtenues pour 48 visées, aucune perte. L'essai « tient », il est rejoué une fois, tient encore : la recommandation devient « 540p · SPAN · RIFE », et la famille SPAN + RIFE ne descend jamais aux paliers plus bas où RIFE aurait vraiment tourné. | Un essai RIFE tient seulement si `fps ≥ 0,95 × target_fps` (cible prise à la cadence que le Rapide vise sur cet écran, pas 48 en dur), ou si `uninterpolated` reste sous un seuil. Écrire `uninterpolated` dans la mesure du banc. | Démontré sur la logique (test A07-03, mesures simulées). À confirmer : `ajp calibrate --visible` sur la 4070 Ti, puis comparer `fps` et `target_fps` de la ligne recommandée dans `calibration.toml`. |
| A07-03 | Majeur | `src/media/bench_clip.cpp:143` | Le chemin du clip de calibration passe à FFmpeg par `path::string()`. Sous Windows, c'est la page de code ANSI (ou une exception pour un caractère hors de cette page), alors que FFmpeg attend de l'UTF-8. Même conversion, sur le chemin d'erreur, dans `load_calibration` (`machine_calibration.cpp:234`), `load_bench` et `load_reading` (`auto_bench.cpp:109`, `auto_bench.cpp:196`). | Windows 11, profil `C:\Users\Élodie` : le dossier de données (`SDL_GetPrefPath`) contient « É », `avio_open` ne trouve pas le chemin, la calibration de l'assistant échoue et l'Auto reste estimé. Profil `C:\Users\さくら` : `calibration.toml` absent, donc à chaque ouverture de vidéo `load_calibration` construit son refus avec `file.string()`, qui lève sous MSVC. | `core::to_utf8` partout (règle de la CI), comme `prepare::utf8_path` le fait déjà pour les arguments de l'enfant. | Lu. À confirmer : un compte Windows au nom accentué, lancer l'assistant jusqu'à la calibration, lire `ajp.log`. |
| A07-04 | Mineur | `src/app/play/span_int8.cpp:139-150` | Un reçu FP8 CUTLASS refusé l'est pour toutes les tailles et pour de bon. `derive_span_fp8` juge à une seule taille, celle de la lecture, et un échec écrit `passed = false` sans `model_sha256` et retire le paquet (`span_derive.cpp:267`, `span_derive.cpp:284`). `find_span_variant` rend alors un refus sans `stale` ni `unjudged`, sauf porte plus ancienne. L'architecture et la version des noyaux ne sont vérifiées qu'après ce test (`span_int8.cpp:156-165`) : un refus venu d'une autre carte ou d'autres noyaux n'est jamais refait. `make_span_prepare` le marque `refused`, la préparation automatique ne le relance plus (`span_derive.cpp:418-420`). | Auto calibré sur un écran 1080p (TRT-08 : `span_fp8 = true`) : la préparation discrète se fait sur l'épisode à 540p, où la porte « refuse souvent » (commentaire de `machine_calibration.cpp:308-310`). Le reçu est refusé ; passé sur un écran 4K (720p de travail), le FP8 ne se reprépare jamais tout seul. Même chose après une mise à jour qui corrige les noyaux sans changer `kGateVersion`. | Un refus de porte va dans `refused_sizes` avec un paquet gardé (comme `validate_span_size`), pas dans un reçu `passed = false`. Tester l'architecture et `kernels_version` avant `passed`, et rendre `stale` quand ils diffèrent. | Lu. À confirmer : sur la 4070 Ti, préparer le FP8 à 540p (`--work-height=540`), puis rouvrir à 720p et lire la note de la page Performance. |
| A07-05 | Mineur | `src/app/play/auto_config.cpp:194-201` | Les bornes calibrées valent pour un clip 1080p à 24 i/s, joué sous TensorRT, sur l'écran de la mesure. Elles s'appliquent pourtant à toute source et tout moteur. `calibrated_chain` ne compare ni le moteur chargé ni la cadence de l'écran (`machine_calibration.cpp:244-246`), et la hauteur de travail ne tient pas compte de la cadence de la source. | Source 1080p à 30 i/s : SPAN tourne 25 % plus souvent que mesuré. Source 4K : décodage 4K et réduction jamais mesurés. TensorRT refusé après une mise à jour du pilote, repli sur Vulkan : les bornes de TensorRT restent. Mesure faite à 60 Hz, lecture à 144 Hz : le Rapide vise ×6. | Ramener la hauteur calibrée au prorata de `source.fps / 24` (et de la taille de source au-delà de 1080p). Mettre le moteur dans `calibration.toml` et l'exiger dans `calibrated_chain`. | Lu. À confirmer : une source à 29,97 i/s en Auto calibré, compter les images perdues dans `ajp.log` (« fin : … manquées »). |
| A07-06 | Mineur | `src/app/play/auto_config.cpp:214-220` | `preset_chain_differs` ignore `auto_free` et `restore_lines`, alors que l'Auto calibré écrit `auto_free` (`auto_config.cpp:205`) et que la calibration garde le `restore_lines` de l'utilisateur. | Auto calibré, l'utilisateur choisit « Auto : Stable » dans le menu : le préréglage reste Auto, la vidéo suivante repasse en Rapide sans le dire. Avec « Restaurer les traits » décoché, `span_runs` est faux pour le clip 1080p à toute hauteur sous 1080 : chaque essai « SPAN » joue la source sans SPAN, tient, et la calibration recommande le plus haut palier. | Ajouter `auto_free` à `preset_chain_differs` (ou ne plus l'écrire). Forcer `restore_lines = true` dans les essais. | Lu (`options.cpp:342-345`, `chain_setup.cpp:227`). |
| A07-07 | Mineur | `src/core/auto_memory.cpp:38` | La mémoire de l'Auto jette tout palier au-dessus de 600 centièmes, alors que l'Auto monte jusqu'à ×8 (`kAutoMaxStep`) et, VRR armé, au 1/24 près. | Écran de 165 Hz en VRR, 23,976 i/s : palier tenu ×6,88 (688) écrit dans `auto_memory.toml`, ignoré à la relecture ; chaque reprise repart de ×2. | Borne à `kAutoMaxStep * 100`. | Démontré (test A07-04). |
| A07-08 | Mineur | `data/gpu_tiers.toml:19-24` | Table des cartes, portables : seuls dix modèles « Laptop » ont leur règle. Les RTX 4060, 4050, 3060, 5060 et 2060/2070 portables tombent sur la règle générale de la famille (`rtx 40`, `rtx 30`… ligne 50-53), au même rang que la carte de bureau, contre la règle d'en-tête (« une tranche sous la carte de bureau »). La RTX 4050 portable a ainsi le rang de la RTX 4070 portable. Autres écarts : `rtx 6000` (ligne 45) prend la Quadro RTX 6000 (Turing) en Ultra avant la règle `quadro rtx` ; la RTX A400 (4 Go, très faible) n'a pas de règle et l'estimation par cœurs Tensor la met en Normal. | Un portable RTX 4050 non calibré (ou calibré sur batterie) part en Normal : 540p + RIFE ×3 sur un écran 1080p. | Ajouter `{ contains = ["laptop"], tier = 1 }` pour NVIDIA juste après les portables nommés (ou une règle par famille 40/30/50). Ajouter `quadro rtx 6000` en 3 avant `rtx 6000`, et `rtx a400` en 0. | Lu ; test descriptible sur `estimate_tier`. |
| A07-09 | Mineur | `src/app/play/machine_calibration.cpp:283` | Rien ne garantit une mesure faite à chaud et stable : batterie lue une fois au départ, ni température, ni horloges, ni chauffe avant le premier essai. La télémétrie NVML existe pourtant dans le lecteur. Depuis le panneau, la largeur de l'écran est supposée 16:9 (`settings_bench_results.cpp:42`). | Portable débranché pendant la calibration : `on_battery` reste faux, les bornes d'une carte bridée deviennent celles de l'Auto. Écran 3440×1440 : les essais présentent en 2560×1440. | Relire `on_battery()` à chaque essai. Écrire la température et l'horloge de fin dans la mesure de l'essai (`--bench-out`), et rejouer un essai fait sous bridage thermique. Passer la vraie largeur de l'écran. | Lu. À confirmer : débrancher pendant `ajp calibrate`, lire `on_battery` dans `calibration.toml`. |
| A07-10 | Mineur | `src/app/play/auto_bench_run.cpp:133` | `copy_file` du `play.toml` de la variante ignore son erreur : l'essai joue alors les réglages par défaut sous le nom de la chaîne demandée. Même chose pour `settings.toml` (ligne 126) : sans moteurs choisis, l'essai rouvrirait l'assistant. | Dossier temporaire plein ou verrouillé par un antivirus : l'essai « 720p · SPAN · RIFE » mesure la chaîne par défaut, et sa ligne entre au tableau. | Rendre un refus quand la copie échoue. | Lu. |
| A07-11 | Mineur | `tests/unit/app/auto_config_test.cpp` | Aucun test de la branche calibrée d'`apply_preset` (le mot `calibrated` n'y apparaît pas). La boucle `calibrate_machine` (choix FP8, rejeux, délai de construction) n'a aucun test : `play_variant` n'est pas injectable. | TRT-01, TRT-08 et A07-02 sont passés sans qu'un test ne bouge. | Les tests de l'annexe ; rendre l'essai injectable (un `std::function` dans `BenchPlan`), comme `SpanDeriver` l'est déjà pour `derive_span_fp8`. | Lu. |
| A07-12 | Mineur | `src/app/play/auto_bench_run.cpp:53` | Dette : `hide_is_reliable_here` n'est appelé nulle part (seulement testé), et le commentaire de `BenchPlan::silent` le cite. Deux mécanismes vivent côte à côte : niveaux plus exploration (`bench.toml`, `run_bench`) et calibration (`calibration.toml`). Noms croisés : `Tier::Light` s'affiche « Rapide », `Tier::Fast` « Qualité » (`gpu_tier.cpp:51-56`), contre l'en-tête de `gpu_tiers.toml`. L'en-tête de `bench_clip.h:22` dit MPEG-4, le code encode en MPEG-2. Des mots français en dur dans des textes traduits : « tient », « inconnu » (`auto_bench_run.cpp:254`, `auto_bench_run.cpp:411`, `auto_bench_run.cpp:420`). `restore_bench_file` écrit sans renommage atomique (`bench_steps.cpp:193`). | Le rapport du banc en anglais affiche « tient ». Une coupure pendant « Annuler » laisse `bench.toml` tronqué. | Retirer `hide_is_reliable*`, ou l'appeler. Décider lequel des deux mécanismes reste. Renommer les tranches comme le menu. Corriger le commentaire. Passer par `tr`. `write_text_atomically`. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-03 | Toujours présent | `src/app/play/auto_config.cpp:223-225` : toute hauteur différente rouvre la chaîne d'un préréglage, sans hystérésis et sans comparer les chaînes d'`apply_preset` avant et après (1440p vers 2160p au niveau Normal donne la même chaîne, mais rouvre quand même). En plus : `calibrated_chain` exige la même hauteur d'écran (`machine_calibration.cpp:245`), donc passer sur l'autre écran perd la calibration et revient à l'estimation. |
| TRT-01 | Toujours présent | `src/app/play/auto_config.cpp:129` puis `:196` : une calibration « RIFE seul » ou « rien » (`span = false`) garde la hauteur de travail du niveau, 480 à 900 selon la tranche sur un écran 1080p. `chain_setup.cpp:238` réduit alors la source sans SPAN, et RIFE tourne à cette taille au lieu de la source mesurée. Démontré (test A07-01). |
| TRT-08 | Toujours présent | `src/app/play/machine_calibration.cpp:279` et `:311` : écran 1080p, paliers 540, 480 et 360, jamais 720 ; `span_fp8` reste vrai et passe dans `calibration.toml`. L'Auto prend alors le Balanced Sharp en précision Performance au lieu du Performance Sharp FP16 (`machine_calibration.cpp:252`), et l'écran « Résultats du bench » annonce « Balanced Sharp in FP8 ». Le bench joue aussi un FP8 non jugé (`chain_setup.cpp:337`). Démontré (test TRT-08). |
| TRT-09 | Toujours présent | `src/app/play/auto_config.cpp:147` : source 720p sur écran 1080p, Qualité (720) et Ultra (900, ramené à 720) donnent la même chaîne, RIFE 4.25, plafond ×4. Démontré (test TRT-09). |
| TRT-19 | Corrigé | `src/app/play/settings_bench_results.cpp:181-185` : pendant une lecture, « Refaire le banc » émet `StartCalibration`. La chaîne se ferme (`chain.cpp:413-418`), la calibration tourne puis la lecture reprend (`player.cpp:560-569`). Nouveau défaut sur ce chemin : renvoi vers 25 ci-dessous. |
| Renvoi 02 (`span_int8.cpp:166`) | Toujours présent, portée réduite | `size_key` passe par `kernel_crop` (1920×1088 au plus) : une taille de travail plus large (source 2560×1080 à 1080 de travail) est validée sous le nom « 1920x1080 ». Dans l'Auto (16:9, au plus 1080 de travail), cela ne touche que les sources plus larges que 16:9. |
| Renvoi 02 (`span_int8.cpp:129-150`) | Toujours présent, aggravé | Voir A07-04 : en plus, l'architecture et les noyaux sont vérifiés après le refus. |
| Renvoi 05 (une construction par taille) | Confirmé, coût à confirmer | `auto_config.cpp:198-201` : la hauteur calibrée est `min(cal.height, source)`, donc chaque hauteur de source sous la borne (480p, 576p, 720p) et chaque rapport d'aspect donne une forme de moteur. La calibration elle-même construit une forme par palier (jusqu'à six sur un écran 4K, `calibration_minutes`). À confirmer : le nombre de `.engine` dans le cache après une saison mêlant 720p et 1080p. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 25 | `src/app/play/engine_screen.cpp:111`, `:167`, `:191`, `:197` | `calibrate_on_screen` rend toujours le préréglage Auto, même sur « Annuler » ou un échec. Lancée par « Refaire le banc » pendant une lecture (`player.cpp:562-569`), elle remplace une chaîne Personnalisée de l'utilisateur, alors que le même bouton hors lecture ne change aucun réglage. |
| 08 | `src/app/play/chain_setup.cpp:99` et `src/app/play/player_run.cpp:170` | `upscale()` et l'anneau prennent `work_size(settings.work_height)`, alors que `c.work` (`chain_setup.cpp:238`) met 0 quand `upscale && !span` (restaurer les traits coupé) : tailles différentes possibles entre l'anneau, l'étage et la chaîne. |
| 09 | `src/app/play/chain_setup.cpp:321`, `src/app/play/span_auto.cpp` | La préparation FP8 automatique se fait à la taille de travail de la lecture en cours, une seule fois par modèle et vidéo : avec A07-04, un premier essai à 540p ferme la voie FP8. |
| 18 | `src/app/play/player.cpp:411-414` | `calibrated_chain` relit `calibration.toml` à chaque ouverture et à chaque remplissage du panneau (`player_run.cpp:293`), et passe par `load_calibration` (A07-03 sous Windows). |
| 34 | `src/app/play/auto_bench_run.cpp:35-44` | Les essais passent `--hidden` sur tous les bureaux, GNOME Wayland compris, alors que le commentaire de `hide_is_reliable` dit que la mesure y serait faussée : à revérifier avec le rendu hors écran actuel. |

## Points positifs

- La recherche, la décision et le fichier sont purs et séparés du lancement : `next_candidate`, `best_candidate`,
  `apply_preset`, `holds` se testent sans GPU (c'est ce qui a permis de démontrer quatre défauts ici).
- Les essais tournent dans des processus enfants avec un dossier jetable : un essai qui plante ne fait pas tomber
  le lecteur, et le refus de l'enfant remonte en une phrase (`bench_events`, `ChildTail`).
- Rejeux P3.12 bien pensés : délai de construction séparé, second passage dans les deux sens.
- Écritures atomiques partout sauf une (`write_text_atomically`, `commit_derivation` avec retour arrière).
- TRT-19 corrigé proprement : la chaîne se ferme avant la mesure.
- Le classement de `best_candidate` par rang entier, pour contourner MSVC, est commenté.

## Questions ouvertes

- La calibration doit-elle mesurer la chaîne de l'Auto telle qu'`apply_preset` la construit (A07-01) ? C'est la
  correction la plus simple de TRT-01, TRT-08 et A07-01 ensemble.
- Le FP8 doit-il entrer dans l'Auto calibré ? `auto_config.cpp:167-168` dit que les préréglages ne le posent plus
  tant qu'il n'est pas jugé à l'œil, mais `apply_preset` le pose dès que `calibration.toml` dit `span_fp8`.
- Faut-il garder les niveaux plus l'exploration (`bench.toml`) maintenant que la calibration existe ?

## Annexe : les tests qui échouent

Montage : les sources de la zone compilées avec `-DAJP_BOUNDARY_HASH=0 -DTOML_EXCEPTIONS=0`, une doublure de
`core/log.h`, et des doublures pour `i18n::tr`, `save_settings` (garde les réglages de l'essai), `play_variant`
(rend la mesure choisie par le test), `write_text_atomically` et `write_bench_clip`. Résultat : 7 cas, 7
échecs.

```cpp
// A07-01 (TRT-01) : 480, 480, 540, 720, 900 au lieu de 0.
TEST_CASE("Auto calibré sans SPAN ne réduit pas la source") {
  for (const Tier tier : {Tier::Economy, Tier::Light, Tier::Normal, Tier::Fast, Tier::Ultra}) {
    Machine m{.tier = tier, .display_height = 1080, .display_hz = 60.0};
    m.calibrated = CalibratedChain{.span = false, .rife = true, .height = 0};
    const PlaySettings out = apply_preset(auto_settings(), m, k1080p24);
    CHECK(out.work_height == 0);
  }
}
// A07-01 (A07-02 dans le montage) : span_order 0 contre 1, keep_poses vrai contre faux.
TEST_CASE("Auto calibré applique la chaîne mesurée") {
  PlaySettings base = auto_settings();
  base.span_order = SpanOrder::Before;
  base.keep_poses = false;
  const PlaySettings measured = candidate_settings({.span = true, .rife = true, .height = 540}, base, false);
  Machine m{.tier = Tier::Normal, .display_height = 1080, .display_hz = 60.0};
  m.calibrated = CalibratedChain{.span = true, .rife = true, .height = 540};
  const PlaySettings applied = apply_preset(base, m, k1080p24);
  CHECK(applied.span_order == measured.span_order);
  CHECK(applied.keep_poses == measured.keep_poses);
}
// TRT-09 : faux.
TEST_CASE("Qualité et Ultra diffèrent pour une source 720p") {
  const core::SourceInfo s720{.width = 1280, .height = 720, .fps = 23.976};
  const Machine m{.tier = Tier::Normal, .display_height = 1080, .display_hz = 60.0};
  CHECK(preset_chain_differs(apply_preset(with(Preset::Quality), m, s720),
                             apply_preset(with(Preset::Ultra), m, s720)));
}
// TRT-08 : span_fp8 vrai alors que chaque essai a joué « FP16 ».
TEST_CASE("calibration 1080p sans FP8 joué") {
  g_trial = [](const PlaySettings& s) { return clean(s, s.rife_auto ? 48.0 : 24.0); };
  const auto done = calibrate_machine(BenchPlan{.video = "/dev/null"}, PlaySettings{}, 1080, {}, {});
  CHECK_FALSE(done->span_fp8);
}
// A07-02 (A07-03 dans le montage) : recommandé 540p · SPAN · RIFE avec 26 i/s pour 48.
TEST_CASE("SPAN + RIFE tient sans que RIFE ait tourné") {
  g_trial = [](const PlaySettings& s) { return clean(s, s.rife_auto ? 26.0 : 24.0); };
  const auto done = calibrate_machine(BenchPlan{.video = "/dev/null"}, PlaySettings{}, 1080, {}, {});
  CHECK_FALSE(done->chosen.rife);
}
// A07-07 (A07-04 dans le montage) : rien relu.
TEST_CASE("la mémoire de l'Auto relit ×6,88") {
  core::AutoMemory m;
  m.put("k", 688);
  CHECK(core::AutoMemory::parse(m.dump()).find("k") == 688);
}
```

`clean(s, fps)` rend une mesure sans perte (600 présentations, 60 Hz), `interval_mean_us = 1e6 / fps` et
`span_variant = "FP16"` quand SPAN tourne. Un test de plus, décrit : TRT-08 côté écran 1080p, `calibration_heights(1080)`
ne contient aucun palier à `kFp8JudgeHeight` ou plus (échoue aussi).
