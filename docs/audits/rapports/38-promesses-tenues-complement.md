# Complément de l'audit 38. Promesses tenues

Branche : `claude/project-thread-rxap43`. Commit vérifié : `1ecd936`. Date : `2026-11-09`.

Ce complément ne traite que le renvoi de l'audit 41 vers 38 (`docs/hdr-anime.md:63-67`, « trois promesses non
tenues : nuits lisibles, blanc à 203 nits, vrai noir en SDR »). La numérotation suit le rapport
[38-promesses-tenues.md](38-promesses-tenues.md), qui s'arrête à A38-15.

## Verdict

Les trois promesses sont fausses. Le blanc à 203 nits et le vrai noir en SDR sont des doublons d'A41-02 et
d'A41-07, avec des textes de plus à corriger. Les nuits lisibles sont un doublon d'A41-03 pour l'écrasement
par la LUT, mais la doc décrit aussi un réglage « selon la scène » que le code fait selon l'épisode : défaut
nouveau, A38-16.

Note de repérage : au commit vérifié comme au commit d'audit 41 (`f7e00ae`), `docs/hdr-anime.md:63-67` est le
début de la section « Sortie HDR ou SDR ». La promesse sur les nuits est en `:56-60`. Les lignes `:63-67` ont
été relues : le seuil de 350 nits (`host_facts.h:42`) et la relecture toutes les 2 s (`player_run.cpp:90`,
`:549-552`) sont tenus.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A38-16 | Mineur | `docs/hdr-anime.md:56-60` ; `src/render/shaders/anime-hdr-anime/3-anime.hook:132-137` ; `src/app/play/chain_shots.cpp:172` ; `src/app/play/anime_shots.cpp:17-26` | La doc promet des « Noirs profonds selon la scène » : effet maximal en scène très claire, nul en scène très sombre, « pour garder lisibles les détails des nuits », et le curseur « tel quel » sans mesure. Le code règle l'effet sur la clé de la **série** : la médiane des APL des plans déjà analysés de l'épisode (`series_apl_of`), qui ne bouge pas aux coupes (choix assumé par le commentaire du shader, P3.52). Une nuit dans un épisode clair reçoit donc tout l'effet. Sans mesure, la clé vaut 0,25 du blanc SDR et l'effet `smoothstep(0.03, 0.40, 0.25)` = 64 % du curseur, pas 100 %. La médiane change aussi tant que la pré-analyse avance (`feed_plan_shots` la recalcule à chaque plan nouveau), donc les noirs d'un même plan peuvent bouger au début de l'épisode. | Épisode surtout diurne (médiane ≥ 40 % du blanc SDR) avec une scène de nuit : « Noirs profonds » s'applique en entier sur la nuit, par-dessus l'écrasement de la LUT (A41-03). C'est l'inverse de la promesse. | Choisir entre les deux comportements, puis aligner : soit réécrire la section (« selon l'épisode », clé médiane, 64 % sans mesure), soit prendre le minimum de la clé du plan (`fs_apl`) et de celle de la série pour qu'une nuit garde ses détails sans que les cheveux sombres sautent d'un plan clair à l'autre. Figer la clé de série une fois la pré-analyse assez avancée. | Lu. Calcul : t = (0,25 − 0,03)/0,37 = 0,595, t²(3 − 2t) = 0,640. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A41-03 (nuits lisibles) | Doublon, toujours présent | La promesse de `docs/hdr-anime.md:58-60` tombe d'abord sur la LUT de base, qui envoie RVB 16 à 0,02 nit. Rien à ajouter au rapport 41 ; le défaut propre au texte est A38-16. |
| A41-02 (blanc à 203 nits) | Doublon, toujours présent | `docs/hdr-anime.md` ne contient plus « 203 » ; la promesse est dans le commentaire `src/render/grade_params.h:59-61` (« le blanc de la source à 203 nits et les aplats à la moitié du pic ») et, côté utilisateur, dans `README.md:35-36` : « Les surfaces restent à leur place, seuls les éclats montent ». A41-02 mesure le blanc des surfaces à 695 nits sur 1 000 et la montée des émetteurs divisée par deux : les surfaces bougent et les éclats montent peu. À la correction d'A41-02, réécrire ces deux textes avec les niveaux retenus. |
| A41-07 (vrai noir en SDR) | Doublon, toujours présent | `grade_params.cpp:164-166` met `black_lo` et `black_hi` à 0 dans `sdr_tonemap`, appelé pour toute sortie SDR d'une source SDR (`chain_hdr.cpp:100-104`). `refresh_black` mesure quand même et annonce « True black: 0 below … nit » (`chain_hdr.cpp:381-383`). Deux textes de plus, non cités par 41 : le commentaire `chain_hdr.cpp:349` (« Le grade s'applique sur toutes les routes : HDR et SDR ») et la note fixe du panneau `settings_panel.cpp:151` (« Near-blacks become true black (0), up to the threshold. Live. »), affichée quelle que soit la sortie. Les corriger avec A41-07 : sur une sortie SDR, afficher « sans effet sur cette sortie » ou appliquer le pied. |

## Baisse de qualité pour la vitesse

Aucune dans ce renvoi.

## Renvois

Aucun.
