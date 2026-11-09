# Rapport de l'audit 12. Cadence et horloges

Branche : `claude/project-thread-oun35p`. Commit audité : `7a0d0dc`. Date : `2026-11-09`.

## Verdict

Les grilles d'instants (`CadenceAccumulator`, `FreeGrid`, `SteadyGrid`) sont exactes et ne dérivent pas, mais sur
un écran à fréquence fixe, le cas de Tatoub, deux briques de cette zone se trompent : la grille des vblanks
apprend la cadence du contenu au lieu de celle de l'écran quand les présentations se font à la demande (A12-01),
et le mode Stable n'a aucun palier d'interpolation à 165 Hz pour du 23,976 (A12-02).

## Cadrage retenu

- **Écrans** : la fréquence fixe (sans VRR) d'abord. Le VRR bugue sur l'écran de Tatoub et « on s'en fout un peu » :
  ses défauts sont relevés, mais classés plus bas. La fréquence exacte de l'écran de Tatoub (144 ou 165 Hz) est
  demandée : elle fixe la gravité réelle de A12-02.
- **Sources** : 23,976 i/s d'abord, puis 29,97.
- **Ce qui compte** : l'absence de saccade et la vitesse exacte (synchro audio) à égalité.
- **Mis de côté** : rien. Le code mort et les doublons sont relevés.
- Pas de carte NVIDIA ni de Catch2 dans le conteneur : les défauts de calcul sont démontrés par un petit programme
  compilé contre les sources de `src/core` (résultats cités en preuve). Ce qui demande l'écran est « à confirmer ».

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A12-01 | Majeur | `src/core/vblank_clock.cpp:32-36` | La grille des vblanks apprend la période du **contenu**, pas celle de l'écran, dès que ses premières observations sont des présentations à la demande. L'estimation part du plus petit écart vu (« un vblank »), puis est aussitôt remplacée par la période ajustée (ligne 36) : un écart d'un seul vblank vu plus tard ne la corrige jamais. | Écran fixe 144 Hz, `--present=wait` et `--vsync-index` (les défauts), 23,976 sans OSD : les images partent tous les 6 ou 7 vblanks. Chaque écart compte pour un pas, la régression rend 41,7 ms : 23,98 Hz au lieu de 144. `Session::set_display` rejette cette valeur (écart > 2 %), mais `Session::pick` (`session.cpp:401-410`) choisit l'image sur cette grille : une image devient éligible dès que son « vblank » grossier est le suivant, donc jusqu'à une période d'image trop tôt. La grille du Rapide sur écran fixe (`live_settings.h:306-312`) la prend aussi, et ne place plus d'images qu'aux multiples de cette période. Même chose après chaque nouvelle swapchain (`placebo_feedback.cpp:156` remet l'horloge à zéro). | Ne pas déduire le vblank des écarts : prendre la période de l'écran (SDL, ou EDID) comme pas initial et ne garder que les observations dont l'écart est proche d'un multiple entier de cette période. Garder l'estimation la plus fine au lieu de l'écraser par l'ajustement. Test : observations tous les 6 ou 7 vblanks à 144 Hz depuis zéro, `hz()` doit valoir 144 à 0,01 Hz près. | Démontré : `hz()` = 23,977 pour des présentations à 23,976 sur 144 Hz ; 144,000 si les huit premières tombent à chaque vblank. Effet à l'écran à confirmer : `--stats` sur un écran fixe, sans OSD ni interface, comparer `vsync_k` et l'intervalle des images neuves, puis refaire avec `--no-vsync-index`. |
| A12-02 | Majeur | `src/core/auto_rate.cpp:281-291` | Sur écran fixe, le mode Stable n'accepte que les paliers entiers ou demi-entiers qui tombent sur un nombre entier de vsyncs à 1 % près. Pour 23,976 i/s à 165 Hz, et pour 29,97 à 144 Hz, il n'en reste aucun : l'interpolation reste à ×1. Le Rapide à pas constant sait pourtant caler 165 Hz par une fraction exacte (`pacing.h:7`, 11011/3200 soit 82,5 i/s). | Tatoub sans VRR sur un écran à 165 Hz, une source à 23,976 : le Stable joue à ×1, en motif 7-7-7-6. On y arrive quand le Rapide se replie (pertes d'images, `free_rate.h:13`, au moins 3 min), quand Stable est choisi, ou quand le moteur ne sait pas calculer un instant libre. Le menu peut proposer un niveau ×4 qui ne s'applique jamais. | Ajouter aux paliers calés les facteurs `lock_factor(hz / (k · fps))` pour k vsyncs par image (comme `SteadyRate::configure`), exprimés en fraction et non en palier / granularité ; ou, à défaut, signaler à l'écran que le Stable ne peut pas interpoler sur cet écran. Test : `auto_limits(165, 23.976, false, 4).locked` doit contenir au moins un palier au-dessus de ×1. | Démontré : `auto_limits(165, 23.976, false, 4)` → paliers calés {×1} ; `auto_limits(144, 29.97, false, 4)` → {×1}. À confirmer : la fréquence réelle de l'écran de Tatoub. |
| A12-03 | Majeur | `src/core/vsync_phase.h:2-5` | Sans retour de présentation (niveau Open), la phase des vsyncs du Rapide sur écran fixe est apprise sur l'heure où `pl_swapchain_swap_buffers` rend la main (`placebo_screen.cpp:461-463`). Cet instant n'est un vsync que si la file de l'écran est pleine ; avec des présentations espacées de plusieurs vsyncs, il suit l'heure de soumission, elle-même choisie d'après cette phase : la boucle se nourrit d'elle-même et ne voit jamais le vrai vsync. | Windows 11 (ou tout pilote sans `present_wait2`) : le niveau retombe à Open. Le Rapide, défaut sur écran fixe, place ses instants « sur les vsyncs » d'une phase arbitraire : des images à mi-chemin entre deux vsyncs, donc tenues tantôt un vsync, tantôt deux. | Ne nourrir `VsyncPhase` que des présentations qui ont attendu une place (`queue_wait_us` proche d'une période), ou de l'heure d'acquisition quand elle a bloqué ; sans aucune mesure fiable, ne pas caler du tout. Test : des présentations à heure arbitraire ne doivent pas rendre `known()`. | Lu. À confirmer sous Windows 11 : `--stats` en Rapide sur écran fixe, lire le niveau de présentation (`level`) puis l'histogramme des `interval_us` des images neuves ; des intervalles qui alternent 1 et 2 périodes pour un pas constant confirment. |
| A12-04 | Mineur | `src/core/auto_rate.cpp:293-301` | `auto_limits` calcule les paliers calés sur la fréquence qu'on lui donne, et l'appelant lui passe le plafond (`auto_limit_hz`, réglage ou batterie), pas l'écran. Même erreur pour le Rapide à pas constant (`FreeDisplay.hz`). | Écran fixe à 165 Hz, plafond de l'Auto à 120 i/s : les paliers ×2,5 et ×5, calés pour 120 Hz, sont permis. Sur l'écran, 59,94 i/s tient 2,75 vsyncs : saccade 3-3-3-2. | Séparer les deux grandeurs : la fréquence de l'écran pour le calage, le plafond pour la borne haute. | Lu (`player_run.cpp:420-436`). Démontrable : `auto_limits(120, 23.976, false, 4)` rend {×1, ×2,5, ×5}. |
| A12-05 | Mineur | `src/core/auto_rate.cpp:599-608` | Le réarmement au plancher se déclenche toutes les 8 s même quand aucun palier calé n'existe au-dessus de ×1 : le palier ne bouge pas, mais le changement est compté, sa raison devient « réarmement au plancher » et les mesures sont ignorées 0,7 s. | 165 Hz fixe, 23,976 (voir A12-02) : une ligne de journal et un `changes` de plus toutes les 8 s, toute la séance. | Ne réarmer que si `snap_up` rend un palier. Test : AutoRate avec `locked = {2}`, 60 s de `tick` : `changes()` reste 0. | Démontré : 8 changements en 60 s, palier toujours ×1. |
| A12-06 | Mineur | `src/core/pacing.cpp:61-67` | Le calage de vitesse n'est juste que si la fréquence reçue l'est, et il ne cale que la fréquence, jamais la phase. Sans grille mesurée (A12-01, A12-03), la session prend la fréquence annoncée par SDL ; un écran annoncé 60 Hz qui tourne à 59,94 fait lire 0,1 % trop vite, l'horloge hôte menant (`wiring.h:28-29`). Même avec la grille mesurée, l'écart résiduel n'est rattrapé par rien. | 23,976 à ×2,5 sur une dalle à 59,94 annoncée 60 : une image doublée ou sautée toutes les 17 s environ. Grille mesurée à 0,002 Hz près à 144 Hz : une image tous les 8 min environ, plusieurs sur une heure. | Recaler la vitesse sur la phase (petite correction proportionnelle à l'écart entre l'horloge et la grille des vblanks), et refuser de caler sur une fréquence non mesurée. | À confirmer : `--stats` une heure sur écran fixe, compter les `interval_us` qui sortent du motif ; et relever ce que SDL annonce sous Windows pour un mode 59,94 Hz. |
| A12-07 | Mineur | `src/core/cadence.cpp:56-70` | `retarget` multiplie les termes par 64 sans les reborner : le facteur et les instants `t` sortent de `kRatioTerm` (1 à 100 000), contrairement au contrat de l'en-tête (« refuse comme create() »). | Rapide à pas constant à 165 Hz : 11011/3200 devient 704 704/204 800. Sans effet aujourd'hui (seul `DrawingsTimeline::decide`, non appelé, vérifie ces bornes), mais tout consommateur qui les vérifiera refusera la paire. | Mettre à l'échelle seulement si les termes restent dans la borne, ou documenter que `t` peut en sortir. | Démontré : `factor()` = 704704/204800 après `retarget({11011, 3200})`. |
| A12-08 | Mineur | `src/core/steady_rate.cpp:162-166`, `:203` | Le repli du Rapide à pas constant compte les pertes du démarrage et des reprises (la case est remplie avant le test de `settle`), alors que le Rapide les ignore 5 s (`free_rate.cpp:51`). Et les deux n'ont pas la même règle des 5 % (perdues / produites contre perdues / (produites + perdues)), ni la même exigence de pertes réparties. | `--fast-steady` sans a priori : N part de 1, le démarrage perd une dizaine d'images, le repli vers Stable se déclenche dès la première seconde. | Une seule règle de repli, partagée, avec la période de calme du démarrage. | Lu. |
| A12-09 | Mineur | `src/core/steady_rate.h:37`, `src/core/vblank_clock.h:51-54` | Code mort : `FreeParams::vrr_floor_us` n'est lu nulle part ; `VblankClock::first_us_` et `last_id_` sont écrits, jamais lus. `moving_area.*` n'a rien de la cadence et appartient aux étages. | — | Retirer, et rattacher `moving_area` à la zone 09. | Lu (`grep`). |
| A12-10 | Mineur | `tests/unit/core/` | Aucun test de `VsyncPhase`, de `locked_steps` ou `auto_limits` à 144 et 165 Hz, de `VblankClock` sur des présentations à la demande, du réarmement avec paliers calés, des bornes de `retarget`, ni du repli de `FreeRate`. La durée d'une heure n'est éprouvée que pour `SteadyGrid`. | Les défauts A12-01 à A12-05 passent la CI. | Ajouter les tests décrits dans chaque ligne, et un test d'une heure de `CadenceAccumulator` et de `FreeGrid` sur une grille de vsyncs à 165 Hz. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-19 | Toujours présent | `session.cpp:665-672` : `follow_display` appelle `vsync_speed` sans regarder le VRR. `vsync_speed(144, 47,952)` = 1,001001 : 0,1 % trop vite en VRR armé. Correction : vitesse 1 quand le VRR est armé. Faible priorité (VRR mis de côté). |
| GPU-05 (point de la fiche) | Toujours présent, hors zone | `player_run.cpp:463-466`, `:622-629` : `apply_floor` n'est rappelé que si l'état armé change ; d'un écran VRR à un autre écran VRR, le plancher reste celui du premier. Envoyé à 08. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 08 | `src/app/play/player_run.cpp:622-629` | GPU-05 : plancher VRR non recalculé quand l'écran change sans que l'état armé change |
| 08 | `src/app/play/player_run.cpp:420-438` | Le plafond (`limit_hz`) passé comme fréquence d'écran à `auto_limits`, `set_steady` et à l'écart minimal (voir A12-04) |
| 08 | `src/app/play/session.cpp:665-672` | GPU-19 : `follow_display` cale la vitesse même en VRR armé |
| 10 | `src/app/play/placebo_screen.cpp:461-463` | `present_us` est l'heure de retour de `swap_buffers`, pas un vsync ; il nourrit `VsyncPhase` (A12-03) |
| 34 | `src/app/play/loop_wait.cpp:56` | L'attente « précise » est un `sleep_until` : sous Windows, résolution de 1 ms (`timeBeginPeriod(1)`), pas de minuterie haute résolution ; à 165 Hz, une demi-période fait 3 ms |
| 20 | `src/core/drawings.cpp:113` | `DrawingsTimeline` n'est appelé que par ses tests : code mort du socle |

## Points positifs

- Les grilles d'instants se comptent en entiers (`between`, `cadence.cpp:20-26`) : aucune dérive sur des heures,
  vérifié pour `SteadyGrid` par un test d'une heure.
- `FreeGrid` place l'image sur le vsync suivant et jamais entre deux sur écran fixe, avec un écart minimal
  tolérant aux arrondis (`free_grid.cpp:47`).
- Toutes les classes de la zone sont pures et prennent l'heure en argument : les tests sont déterministes sans
  `FakeClock` (qui sert aux tests de la session, `tests/unit/app/doubles.h:103`).
- `VsyncPicker` supprime l'alternance 3-2-3-3-2 près du milieu, avec un test de gigue.
- Le choix de la descente de l'Auto ignore les paires faites au palier d'avant : pas de cascade.

## Questions ouvertes

- Quelle est la fréquence de l'écran de Tatoub (144 ou 165 Hz) ? À 165 Hz fixe, A12-02 touche chaque repli du
  Rapide.
- Sous Windows 11 avec la carte NVIDIA, quel niveau de présentation le lecteur obtient-il (`level` dans
  `--stats`) ? S'il retombe à Open, A12-03 s'applique au Rapide par défaut.
- Le lecteur présente-t-il des images à chaque vblank au démarrage (écran de chargement, interface) ? Si oui,
  A12-01 ne frappe qu'après un changement de mode ; sinon, à chaque lecture.
