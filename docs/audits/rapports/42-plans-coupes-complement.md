# Complément de l'audit 42. Analyse des plans, coupes et pré-analyse

Branche : `claude/project-thread-jnvwkc`. Commit vérifié : `1559e8b`. Date : `2026-11-09`.

Renvois reçus après le rapport `42-plans-coupes.md`, vérifiés dans le code. Les IDs suivent ceux du rapport
(dernier : A42-16).

## Verdict

Les seuils de `next_flashing` suivent bien BT.1702-2 (20 cd/m², image sombre sous 160 cd/m², plus de trois
flashs par seconde : 3 Hz passe, 3,5 Hz déclenche), mais le plafond arrive tard : il faut sept transitions dans
la seconde, la première montée n'est pas comptée, et rien ne regarde en avant, alors que la pré-analyse le
pourrait. Une suite à 4 Hz est montrée une seconde entière au pic HDR avant que le plafond prenne (démontré).
Trois défauts mineurs nouveaux ; la mesure sur la seule moyenne et en nits de source est déjà A42-08, le saut
des métadonnées HDR10 déjà GPU-07.

## Renvois traités

| De | Où | Conclusion |
|---|---|---|
| 41 | `src/app/play/chain_hdr.cpp:321` | Seuils conformes à BT.1702-2 ; confirmé pour le délai (A42-17), les analyses perdues (A42-18) et la sortie SDR (A42-19). Le reste est doublon de A42-08 et GPU-07. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A42-17 | Mineur (sécurité) | `src/media/flash_guard.h:26-29`, `:41`, `src/app/play/chain_shots.cpp:160-164`, `src/media/live_scenes.cpp:20` | Le plafond ne prend qu'à la septième transition comptée dans la seconde, et la première montée depuis le départ (ou après un saut, `:24`) n'est jamais comptée : `rising_` vaut `true` au départ, seule une descente peut donc faire la première transition. Le limiteur ne voit que les images déjà analysées à l'ingestion (`next_flashing` lit `LiveScenes`), avec deux relectures GPU de retard (`kPendingReadbacks = 2`), puis le grade change après la dernière image préparée (`set_grade_after`). La pré-analyse (`shot_probe.cpp`, `measure_frame`), qui court devant la lecture et mesure la même lumière par image, n'alimente aucun limiteur. | Alternance noir et blanc (0 et 203 cd/m² de moyenne), départ sur le noir : premier plafond à 1 000 ms à 4 Hz, 667 ms à 6 Hz, 333 ms à 12 Hz (60 et 24 i/s, mêmes délais) ; départ sur le blanc à 4 Hz et 24 i/s : 875 ms. S'y ajoutent deux ou trois images de relecture. Les quatre premiers flashs d'une séquence dangereuse passent donc au pic HDR plein (695 nits de blanc sous 1 000, A41-02), et chaque saut dans la séquence recommence le compte. | Faire tourner `FlashGuard` sur les images de la pré-analyse et marquer dans la chronologie (`ShotTimeline`) les intervalles dangereux depuis la première transition de la fenêtre, pas depuis la septième ; `next_flashing` lit d'abord ces intervalles, l'analyse à l'ingestion restant le secours. Dans `FlashGuard`, partir sans sens (`rising_` indéterminé) pour compter la première variation dans les deux sens. Ajouter au test les cas 4 Hz (limite), départ sombre et délai. | Démontré : `FlashGuard` compilé seul (banc du fil, 3 s par cas). |
| A42-18 | Mineur | `src/app/play/upscale_stage.cpp:259` | Quand deux relectures attendent encore, la plus ancienne est jetée sans être passée aux scènes : son image manque au limiteur (et à la moyenne de la scène). | Un flash d'une image (alternance image par image, 12 Hz à 24 i/s) sous une charge GPU qui retarde la relecture : l'image claire jetée efface deux transitions, et le plafond prend plus tard ou pas du tout. | Passer l'analyse jetée comme manquante à `LiveScenes` (ou attendre la relecture la plus ancienne au lieu de la jeter) et le journaliser une fois. | À confirmer : compter les analyses jetées sur `calibration.mkv` avec SPAN et RIFE sur un GPU chargé. |
| A42-19 | Mineur | `src/app/play/chain_hdr.cpp:252`, `:101-104`, `:321-327` | Le plafond ne touche que `choice.pq`. Vers un écran SDR, `sdr_anime` est déjà tiré du grade PQ non plafonné dans `grade_of`, et `sdr` n'est pas touché : le limiteur ne change rien à l'image, mais le journal annonce « pic du grade plafonné à 250 nits » et le grade est renvoyé deux fois. | Écran SDR, séquence de flashs : le journal dit le plafond posé, l'image est la même. Un rapport de support conclut à tort que le limiteur a agi. | Ne lever `flash_limited` (et ne journaliser) que si la sortie est en PQ (`system_hdr`) ; en SDR, rien à faire, la sortie ne dépasse pas le blanc de l'écran. | Lu. |

## Doublons

| Constat | Défaut existant |
|---|---|
| 20 cd/m² sur la moyenne de toute l'image au lieu d'un quart de l'écran ; seuil sombre de 160 et variations jugés en nits de source au blanc 203, pas en nits de sortie du grade (`flash_guard.h:1-5`, `:34-37`) | A42-08 |
| Plafond posé d'un coup et métadonnées HDR10 réannoncées (1 000, puis 250, puis 1 000 nits) à l'entrée et à la sortie du limiteur (`chain_hdr.cpp:252`, `:321-327`) | GPU-07 (rapport 41) |

## Seuils vérifiés

| Seuil | Valeur | Jugement |
|---|---|---|
| `kFlashStepNits` | 20 cd/m² | Celui de BT.1702-2, mais sur la moyenne (A42-08). |
| `kFlashDarkNits` | 160 cd/m² | Celui de BT.1702-2, appliqué à l'image la plus sombre de la paire. |
| `kFlashesPerSecond` | plus de 3, soit 7 transitions | Conforme, un peu prudent : 3,5 Hz déclenche, 3 Hz non (banc). |
| `kFlashHoldUs` | 1 s après le dernier instant dangereux | Raisonnable : évite le battement du plafond entre deux rafales. |
| `kFlashPeakNits` | 250 nits | Choisi comme « plein champ d'un écran OLED », sans mesure ; sous plafond le blanc tombe à 228 nits (rapport 41), soit à peu près le niveau SDR. Le limiteur retire l'amplification du grade, il ne rend pas sûre une source déjà dangereuse : choix défendable, à écrire dans le commentaire. |

Le limiteur ne joue qu'avec le grade allumé (`to.running.grade`) : une source HDR non gradée passe telle que
masterisée. Ce n'est pas un défaut, le lecteur n'y ajoute alors aucune lumière.

## Baisse de qualité pour la vitesse

Aucune dans ce renvoi : le plafond de 250 nits est une baisse voulue pour la sécurité (rapport, « Limiteur de
flashs »), pas pour la vitesse.
