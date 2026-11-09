# Complément de l'audit 12. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-ape1xu`. Commit audité : `e5a7e72`. Date : `2026-11-09`.

## Verdict

Sur les quatre renvois, deux donnent un défaut nouveau : le mode Rapide peut perdre l'interpolation pour toute
la séance après un seul à-coup du GPU (A12-12, majeur, démontré), et la PLL audio compte en lectures au lieu du
temps (A12-11, mineur, effet de l'ordre de la microseconde). Les deux autres sont des doublons (A16-12, A24-02).
Le constat du renvoi 32 tel qu'il est écrit (« en ×2, aucune mesure ») est faux, mais la règle qu'il vise cause
une baisse de qualité d'image sans aucune mesure qui la justifie.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les quatre renvois de `audits-docs/complements/12.md`, sans question
à Tatoub, sans toucher au code. Les ID suivent le rapport 12, qui s'arrête à A12-10. Pas de GPU ni de Catch2 dans
le conteneur : les deux défauts de calcul sont démontrés par un petit programme compilé contre `src/core`
(`free_rate.cpp`, `free_grid.cpp`, `cadence.cpp`, `clock_pll.h`), résultats cités en preuve.

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 16 | `src/core/clock_pll.h:28-40` | Confirmé : **A12-11** (mineur) | `rate_` est une dérive par lecture, ramenée en vitesse par le seul dernier intervalle (ligne 37) ; Kp et Ki sont par lecture, donc la bande passante dépend de la période du périphérique. Les lectures perdues existent bien (boîte bornée à 1, `audio_thread.cpp:20-22`, `:60`). Mais l'horloge publiée est recalée sur la phase filtrée à chaque lecture (`player.cpp:294`) : l'erreur de vitesse ne joue que sur l'extrapolation d'un intervalle, soit 1 à 2 µs. |
| 20 | `src/core/clock.h:1-3` | Doublon d'A16-12 | Le commentaire dit « avec VRR, l'audio la publie ; sur écran fixe, present la publie ». Le code fait mener l'audio sur tout écran et l'hôte seulement sous calage (`wiring.h:110-114`). A16-12 nomme déjà ce fichier dans sa correction (« et `core/clock.h:1-3`, renvoi 20 »). Rien de plus. |
| 24 | `src/app/play/session.cpp:506` | Doublon d'A24-02 | La ligne est bien la cause côté session : `ui_changed` suffit à présenter tout de suite (`session.cpp:506`, puis `:518-522`), hors échéance de l'image neuve. A24-02 cite cette ligne, ce scénario (vblank pris à l'image neuve sur écran fixe) et la correction (livrer l'interface avec la présentation suivante). Côté cadence, la seule suite est que `shown_at_us_` (`:536`) repart de cette présentation, ce qui décale aussi les échéances `c` et `f` ; la correction d'A24-02 la règle avec. |
| 32 | `src/core/free_rate.cpp:27` | Réfuté tel qu'écrit, mais défaut nouveau : **A12-12** (majeur) | En ×2, une paire du Rapide a deux images calculées (`upscale_stage.cpp:476-489`, `:516` ; `repeated` ne vient que de DistilDRBA tenu, `upscale_stage.cpp:699-728`) : `cost(2, …)` est accepté et le coût est mesuré. Ce qui est vrai : toute paire d'une seule image est ignorée (`images < 2`), alors qu'en Rapide cette image est souvent une image interpolée qui a coûté RIFE et SPAN. Entre ×1 et ×2, la moitié des paires n'est pas mesurée ; et si le pas dépasse l'écart des sources, plus aucune ne l'est, l'estimation reste figée en haut. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A12-11 | Mineur | `src/core/clock_pll.h:28-40` | La PLL est réglée en lectures, pas en temps : `rate_` est une dérive par lecture, convertie en vitesse par l'intervalle de la seule dernière lecture (ligne 37), et Kp 0,1 et Ki 0,005 sont appliqués par lecture. Une lecture perdue (boîte bornée à 1, `audio_thread.cpp:22`) fait varier la vitesse proposée ; un quantum PipeWire de 1 024 trames contre 10 ms sous WASAPI donne une boucle deux fois plus lente sous Linux. | Dérive réelle de 100 ppm, une lecture sur trois perdue : la vitesse proposée oscille entre 1,000067 et 1,000133 au lieu de 1,000100. Comme l'horloge est recalée sur la phase filtrée à chaque lecture (`player.cpp:294`), l'écart à l'image reste de 1 à 2 µs : aucune saccade, un filtrage de la gigue qui change d'une plateforme à l'autre. | Compter `rate_` en µs par µs (diviser l'erreur par l'intervalle de la lecture) et exprimer Kp et Ki par seconde, multipliés par l'intervalle. Test : dérive constante, intervalles de 10 et 21 ms, une lecture sur trois perdue : vitesse proposée à 1 ppm près de la dérive, même constante de temps. | Démontré : 1,000100 sans perte ; 1,000067 à 1,000133 avec une perte sur trois (dérive 100 ppm, intervalle 21,3 ms). |
| A12-12 | Majeur | `src/core/free_rate.cpp:28`, `src/app/play/live_settings.h:417-421` | Le Rapide n'apprend son temps de service que des paires d'au moins deux images calculées. Or son pas vient de ce même temps : si `service × 1,1` dépasse l'écart des sources, la grille ne place plus qu'une image par paire (`free_grid.cpp:62-77`), aucune mesure ne rentre et l'estimation ne redescend jamais. Rien d'autre ne la fait baisser : `settle` ne la touche pas, seule la réactivation du mode la remet à zéro (`free_rate.cpp:59-65`). Pour une paire en mouvement fort, `heavy_us_` se fige de la même façon. C'est une baisse de qualité (plus d'interpolation) faite pour la vitesse, sans mesure qui la justifie : le GPU n'est plus mesuré du tout. `SteadyRate::pair_cost` accepte, lui, une paire d'une image (`steady_rate.cpp:103`). | 23,976 i/s, Rapide, service mesuré 10 ms (pas 11 ms). Un seul à-coup du GPU pendant la lecture (changement de fenêtre, pilote, compilation d'un shader) : une paire de deux images coûte 200 ms. La montée rapide (× 0,6) porte le service à 64 ms, le pas à 70,4 ms, plus long qu'une paire (41,7 ms). Ensuite chaque paire n'a qu'une image, aucune mesure : la lecture reste sans interpolation jusqu'à ce que Tatoub change de mode. Sans perte d'image, le repli vers l'Auto ne se déclenche pas non plus. Même chose sur une scène lourde : le Rapide ne repasse jamais à deux images dans les paires en mouvement fort. | Mesurer aussi les paires d'une image en Rapide quand cette image est calculée (le coût par image vaut `busy / computed`, la règle « une image seule ne dit rien » ne vaut que pour l'Auto, où cette image est la source) ; et, en filet, faire redescendre le service vers la dernière mesure fiable quand aucune n'arrive pendant une seconde, ou tenter une paire de deux images de temps à autre. Test : `FreeRate` à 10 ms, un échantillon à 100 ms par image, puis 60 s de `FreeGrid::push` à 23,976 dont les paires à une image coûtent 10 ms : `step_us` doit revenir sous 41,7 ms. | Démontré : pas de 11 000 µs, puis 70 400 µs après l'à-coup ; une minute plus tard, 1 440 paires à une image, aucune à deux, le pas resté à 70 400 µs. Effet à l'écran à confirmer : `--stats` en Rapide, provoquer un à-coup (changement de fenêtre plein écran), puis lire `step` et `fps` dans l'OSD. |

## Baisses de qualité pour la vitesse

| Où | Ce qui est perdu | Gain | Mesure qui la justifie |
|---|---|---|---|
| `src/core/free_rate.cpp:28` (A12-12) | L'interpolation, pour le reste de la séance, après un à-coup ; la moitié des mesures entre ×1 et ×2 | Aucun : rien n'est mesuré | Aucune |

## Renvois

Aucun.
