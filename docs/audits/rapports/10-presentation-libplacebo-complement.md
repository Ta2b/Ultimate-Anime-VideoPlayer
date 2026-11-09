# Complément de l'audit 10. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-qsby22`. Commit audité : `f8cb2cd`. Date : `2026-11-09`.

## Verdict

Sur les six renvois, quatre donnent un défaut nouveau (A10-14 à A10-17, tous mineurs) et deux sont des doublons
(A49-12, A40-06) qui apportent chacun un défaut voisin (A10-18, A10-19). Aucun renvoi n'est réfuté. Une baisse de
qualité faite pour la vitesse reste sans mesure : la présentation part de `pl_render_default_params` (doublon
d'A49-12), alors que l'export prend `pl_render_high_quality_params`.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les six renvois de `audits-docs/complements/10.md`, sans question à
Tatoub, sans toucher au code. Les ID suivent le rapport 10, qui s'arrête à A10-13. Le conteneur n'a ni GPU, ni
libplacebo, ni `~/ajp-deps` : tout est lu, rien n'est lancé. Les numéros de ligne sont ceux du commit audité ; ils
ont glissé de quelques lignes depuis le rapport 10 (`63289c5`). Ce que je cite de libplacebo hors de notre correctif
(`ports/libplacebo/0001-ajp-present-timing.patch`) vient de mémoire et est marqué « à confirmer ».

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 12 | `src/app/play/placebo_screen.cpp:461-463` | Doublon d'A12-03 pour la phase des vsyncs, plus un défaut nouveau : **A10-14** | `present_us` et `queue_wait_us` nourrissent aussi la cadence de la session, le remplissage de la file et `--stats`, dont la manipulation proposée pour confirmer A12-03. |
| 21 | `audit/mesures-auto-4070ti.md` | Confirmé comme constat de mesure, cause non localisée : **A10-15** | Le plafond est commun aux deux modèles : il n'est pas dans le calcul. |
| 49 | `src/app/play/placebo_screen.cpp:130` | Doublon d'A49-12 | Voir « Doublons » et « Baisses de qualité pour la vitesse ». |
| 32 | `src/render/shaders/anime-hdr-look.hook:100` | Confirmé : **A10-16** | Aucune « passe du grade fusionné » ne lit `ajl_look`. Pour `pl_render_default_params`, voir A49-12. |
| 41 | `src/app/play/placebo_screen.cpp:155` | Confirmé : **A10-17** (commentaire), doublon d'A41-06 pour le fond | Le hook de netteté est sur `SCALED` (`sharpen.hook:14`). |
| 40 | `src/app/play/placebo_snapshot.cpp:16-40` | Doublon d'A40-06 pour l'image ×0,62, plus deux défauts voisins : **A10-18** et **A10-19** | Rien dans `save_snapshot` ne met l'image à l'échelle : la cause du facteur 0,62 n'est pas dans ces lignes. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A10-14 | Mineur | `src/app/play/placebo_screen.cpp:464-470` | `present_us` est l'heure où `pl_swapchain_swap_buffers` rend la main, et `queue_wait_us` le temps passé dedans. L'en-tête dit « instant hôte où l'image est partie à l'écran » (`links.h:104`) et le commentaire « FIFO : attend une place en file » (`placebo_screen.cpp:465`). Dans libplacebo, `swap_buffers` attend que le nombre d'images en vol passe sous `swapchain_depth` (2 ici, `placebo_screen.h:38`) ; une image sort du compte au rappel `present_cb`, posé sur la commande de rendu, pas au vsync. L'attente de l'écran FIFO tombe, elle, dans `start_frame` (barrière de présentation, `0001-ajp-present-timing.patch:888-908`), donc dans `acquire_us`. Au-delà de `VsyncPhase` (A12-03), cette heure sert à `shown_at_us_` (`session.cpp:542`), qui règle la représentation à cadence fixe et le plancher VRR (`session.cpp:516`, `:519`, `:613-615`) ; `queue_wait_us` règle `queue_fill_` (`session.cpp:548-551`) ; et `interval_us`, `frame_interval` et la fluidité affichée (`session.cpp:563`, `frame_stats.cpp:53-72`) en sont des écarts. | Écran fixe 144 Hz, niveau Open ou Wait : la file n'est presque jamais pleine, `present_us` suit l'heure de soumission à la durée du rendu près. La session croit l'image affichée un ou deux vsyncs plus tôt qu'elle ne l'est, et la manipulation qui doit confirmer A12-03 (histogramme des `interval_us` sous `--stats`) mesure le même instant que la boucle a choisi : elle montre un pas régulier même quand l'écran alterne 1 et 2 périodes. | Renommer le champ (`swapped_us`) et corriger les deux commentaires. Aux niveaux Wait et Timing, prendre `visible_us` du retour (`placebo_screen.cpp:479`) pour `interval_us` et la fluidité ; sans retour, écrire dans `--stats` que l'intervalle est celui de la soumission. Confirmer A12-03 avec `visible_us` (niveau Wait) ou une capture à haute vitesse, pas avec `interval_us` au niveau Open. | Lu. À confirmer : `vk_sw_swap_buffers` et `present_cb` dans `src/vulkan/swapchain.c` de libplacebo v7.360.1. |
| A10-15 | Mineur | `src/app/play/placebo_screen.cpp:328` (`show`), mesure dans `audit/mesures-auto-4070ti.md` | La présentation plafonne vers 127 images par seconde sur la RTX 4070 Ti à 165 Hz VRR. DistilDRBA ×6 (143,9 i/s visées) : 0 paire en retard, 0 instant sauté, 962 images manquées en 60 s (11 %), intervalle p50 7,9 ms et p90 8,8 ms ; RIFE ×6 donne 7,8 / 9,1 ms, et RIFE ×5 et DistilDRBA ×5 le même p50 de 7,8 et 7,9 ms. Le réseau ne prend que 0,73 à 0,77 ms par passe : la limite est après le calcul. Le relevé ne donne ni `acquire_us`, ni `render_us`, ni `queue_wait_us`, ni `display_gpu_us`, qui diraient quel étage de la présentation tient le temps. Candidats lus : la composition de chaque image neuve à la taille de l'écran (grade, sous-titres, interface), l'attente sur deux images en vol (A10-14), le réveil de la boucle au pas de 1 à 4 ms (`session.cpp:598-616`). | Écran 165 Hz, ×6 choisi à la main ou par l'Auto : une image créée sur neuf est jetée, à pleine charge GPU. L'Auto voit bien les images manquées (`auto_overloaded`, `core/auto_rate.h:77`) et redescend, mais le plafond coupe ×6 pour tous les modèles. | Refaire la mesure avec `--stats` à ×6 (DistilDRBA) et lire les quatre temps par présentation ; selon l'étage, alléger la composition des images neuves, ou passer `swapchain_depth` à 3 au niveau Wait. La mesure date de `be07da9`, avant le retrait de GMFSS : la refaire sur le code actuel avant de corriger. | Mesure de Tatoub citée ; cause non établie. |
| A10-16 | Mineur | `src/render/shaders/anime-hdr-look.hook:100` | Commentaire périmé : « aussi lu dans la passe du grade fusionné (grade.cpp) ». Aucune passe de `grade.cpp` ne lit `ajl_look` ; elle n'est appelée que dans ce hook (lignes 130 et 141), et `grade.cpp:135` ne fait qu'y insérer la pile (`//@LOOK_CHAIN@`). | Un relecteur cherche une seconde passe du grade à garder en accord avec le hook, ou en conclut que le grade est fusionné alors qu'A32-05 demande justement de le fusionner. | Retirer la mention, ou la remplacer par celle de `grade.cpp:103-135`. | Lu. |
| A10-17 | Mineur | `src/app/play/placebo_screen.cpp:155` | Commentaire faux : « La netteté sur MAIN d'abord ». Le hook est déclaré `//!HOOK SCALED` (`sharpen.hook:14`), et l'en-tête du shader dit aussi « sur MAIN » (`sharpen.hook:1`). | Un relecteur croit la netteté appliquée avant la mise à l'échelle finale et dans un domaine fixe. Le fond (domaine qui change avec la mise à l'échelle) est A41-06. | Corriger les deux commentaires en même temps qu'A41-06. | Lu. |
| A10-18 | Mineur | `src/app/play/placebo_snapshot.cpp:45` | L'instantané SDR est toujours rendu en `pl_color_space_srgb`, pas dans l'espace de la fenêtre (`target.color`), alors que l'en-tête promet « la composition de la fenêtre rendue une fois de plus ». Sur une sortie SDR en BT.1886 ou gamma 2.4, le grade chargé pour la fenêtre (exposant 2.4, `placebo_grade.cpp:47-49`) reste actif, car `graded()` accepte aussi le sRGB (`placebo_grade.cpp:44-46`, `:250`), et décode avec 2.4 une cible encodée en sRGB. | Sortie SDR à 2.4 : l'instantané ne montre pas ce que montre la fenêtre (ombres et tons moyens décalés), et un test ou une comparaison à l'œil s'y fie. | Sur une sortie SDR, rendre l'instantané dans `target.color` (texture `rgba8` non sRGB, octets écrits tels quels) ; ne garder `pl_color_space_srgb` que pour l'image sans grade d'une sortie HDR10. | Lu. |
| A10-19 | Mineur | `src/app/play/display_contract.cpp:85` | Le journal écrit « contrat d'écran : référence 203 nits, pic 1000 nits » sans dire que ce sont les valeurs par défaut (`render/display_model.h:21-22`, `:52-53`) quand ni KDE, ni SDL, ni l'EDID ne donnent rien, et le fait aussi pour une sortie SDR. | Lavapipe sous `xvfb-run` (le cas d'A40-06), ou un écran sans EDID HDR : on lit au journal un écran HDR à 1 000 nits qui n'existe pas, et on cherche dans ce contrat la cause d'une image SDR assombrie. | Écrire la source de chaque valeur (« KDE », « SDL », « EDID », « par défaut »), et préciser si la sortie est SDR. | Lu. |

## Doublons

| Renvoi | Doublon de | Pourquoi | Ce que le renvoi ajoute |
|---|---|---|---|
| 12 → 10, `placebo_screen.cpp:461-463` | A12-03 | Même ligne, même instant : `vsync_note((*record)->host_us, …)` (`player_run.cpp:658`) nourrit `VsyncPhase` avec `present_us`. | A10-14 : les autres usages du même instant, et une manipulation de confirmation d'A12-03 à changer. |
| 49 → 10, `placebo_screen.cpp:130` | A49-12 | Même ligne : `pl_render_params params = pl_render_default_params;`, sans `antiringing_strength`. L'EWA Lanczos n'arrive que par `final_scaler`, `Default` par défaut (`settings.h:233`). | L'export, lui, prend `pl_render_high_quality_params` (`placebo_export.cpp:302`) : la lecture n'a pas les filtres que le projet juge assez bons pour l'export. Sans effet à l'export même, où la cible a la taille de l'anneau. La composition des OSD (`placebo_screen.cpp:320`) n'est pas concernée : elle ne met aucune image à l'échelle. |
| 40 → 10, `placebo_snapshot.cpp:16-40` | A40-06 | `save_snapshot` ne multiplie rien : il recopie la cible, remplace la couleur par `pl_color_space_srgb` et relit des octets `rgba8`. L'image garde `image.color = s.source` (`placebo_screen.cpp:383`). Le facteur 0,62 vient donc d'ailleurs. Pistes, à vérifier sous lavapipe : l'étiquette de couleur de l'anneau sans SPAN (`set_source_color`, `player_run.cpp:146`) face à ce que contient l'anneau, et l'espace de la swapchain choisi par lavapipe (une cible PQ expliquerait que le grade coupé ne change rien, `placebo_snapshot.cpp:26-30`). | A10-18 (espace de l'instantané SDR) et A10-19 (contrat par défaut écrit comme lu). |
| 41 → 10, `placebo_screen.cpp:155` | A41-06 pour le fond | Le hook sur `SCALED` est le défaut d'A41-06. | A10-17 : les deux commentaires faux. |

## Baisses de qualité pour la vitesse

| Choix | Où | Ce que l'image perd | Gain | Mesure au dépôt | Statut |
|---|---|---|---|---|---|
| Présentation par `pl_render_default_params` (Lanczos sans anti-ringing vers le haut, Hermite vers le bas, à confirmer dans libplacebo v7.360.1) au lieu de `pl_render_high_quality_params` | `placebo_screen.cpp:130` | Halos sur les traits noirs à chaque mise à l'échelle finale (écran 4K, fenêtre) ; aucune perte quand l'anneau a la taille de l'écran | Inconnu : une passe à la taille de l'écran | Aucune | A49-12 |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A12-03 | Toujours présent | `src/app/play/placebo_screen.cpp:464-470`, `src/app/play/player_run.cpp:658`, `src/app/play/live_settings.h:294`. |
| A49-12 | Toujours présent | `src/app/play/placebo_screen.cpp:130-141`. |
| A41-06 | Toujours présent | `src/render/shaders/sharpen.hook:14`. |
| A40-06 | Toujours présent, cause hors de `placebo_snapshot.cpp` | `tests/unit/app/e2e_play_test.cpp:236-243`. |
| A10-12 | Toujours présent | `src/app/play/present_level.h:426` dit encore `open` par défaut. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 12 | `src/app/play/session.cpp:542` | La cadence fixe et le plancher VRR comptent depuis `shown_at_us_`, qui n'est pas l'heure d'affichage (A10-14) : à reprendre avec A12-03. |
| 32 | `audit/mesures-auto-4070ti.md` | Le plafond de 127 i/s (A10-15) est un budget par image de la présentation : à remesurer avec les quatre temps de `--stats`. |
| 40 | `tests/unit/app/e2e_play_test.cpp:236-243` | Chercher le facteur 0,62 dans l'étiquette de couleur de l'anneau sans SPAN et dans l'espace de la swapchain sous lavapipe, pas dans l'instantané. |
| 11 | `src/app/play/display_contract.cpp:22-41` | Le contrat garde 203 et 1 000 nits par défaut sans le dire (A10-19). |

## Points positifs

- L'instantané d'une sortie HDR10 sans `.pq.ppm` le dit au journal (`placebo_snapshot.cpp:24-30`) : l'outil ne
  ment pas en silence sur le grade.
- Le retour de présentation date déjà l'apparition réelle (`visible_us`) aux niveaux Wait et Timing : A10-14 ne
  demande aucune mesure nouvelle, seulement d'utiliser celle qui existe.
- La rampe du vrai noir rend sans tramage (`placebo_grade.cpp:328`) : la mesure n'est pas bruitée.

## Questions ouvertes

- A10-15 : quel étage tient les 7,9 ms à ×6 ? Une séance de 60 s avec `--stats`, DistilDRBA ×6, plein écran
  165 Hz, suffit.
- A40-06 : la swapchain de lavapipe sous `xvfb-run` est-elle PQ ? La ligne « présentation » et `hdr_output` au
  journal le disent.
