# Rapport de l'audit 22. Réglages : modèle, application et persistance

Branche : `claude/project-thread-5unmhd`. Commit audité : `71b43a91c9d7c992fed863c1ee7086e7bf67bd03`. Date : `2026-11-09`.

## Verdict

Le modèle des réglages est solide pour la lecture et l'écriture tolérantes de `play.toml`, mais deux défauts font perdre
des choix de l'utilisateur sans un mot : des réglages du menu ne s'appliquent ni ne s'enregistrent jamais (UX-01,
toujours présent, plus large qu'annoncé), et une seule écriture refusée de `play.toml` (cas courant sous Windows avec
un antivirus, A20-01) efface pour de bon le changement qu'elle portait. Le défaut « Auto » de l'ordre warp/SPAN
sacrifie aussi la qualité au profit de la vitesse sur presque toute lecture.

## Cadrage retenu

Pas de questions posées : Tatoub a demandé d'auditer au maximum sans cadrage. Retenu :

- cible NVIDIA sous TensorRT, Windows 11 et CachyOS. Le VRR est hors sujet ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; chercher en particulier ce qui, dans les réglages, empêche
  DistilDRBA d'être bien plus rapide que RIFE (section dédiée plus bas) ;
- consigne ajoutée en cours d'audit : relever tout endroit de la zone où la qualité du rendu est sacrifiée pour la
  performance (section dédiée plus bas) ;
- le fil tourne sans GPU ni dépendances (`~/ajp-deps` absent) : les tests du dépôt n'ont pas été compilés. Chaque
  défaut est prouvé par lecture, et le test qui échouerait est décrit.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A22-01 | Majeur | `src/app/play/settings.cpp:383-434`, `:453-499` | UX-01, plus large qu'annoncé. Un champ de `PlaySettings` qui n'est ni dans `live_part` ni dans une condition de `change_between` reste dans le brouillon du panneau : pas de « Appliquer » (`settings_panel.cpp:583-593` ne montre le bouton que pour `Reopen`), pas d'application directe, et la fermeture du panneau le jette. Champs touchés : `restore_lines` (« Restore line art »), `gpu` (« Card for playback »), `rife_fused` et `drba_fused` quand l'interpolation est coupée (`rife_factor < 2`, conditions `:463`, `:468`), `audio_device_delays_ms` seul. | Menu > Post-traitement, décocher « Restore line art » sur une source 1080p : rien ne change, rien n'est écrit, la case revient cochée à la réouverture du panneau. Même chose en choisissant la carte d'un portable à deux GPU. | `restore_lines` dans `change_between` (réouverture, SPAN se refait) ; `gpu` dans `live_part` (vaut au prochain lancement, comme `smooth_motion`) ; `rife_fused` et `drba_fused` dans `live_part` quand `wanted.rife_factor < 2`. Puis le test d'A22-15. | Lu. |
| A22-02 | Majeur | `src/app/play/settings_toml.cpp:624-626` | `save_settings` retient la table « vue » (`remember`) **avant** l'écriture atomique, et même si elle échoue. La fusion P2.14 de l'écriture suivante ne reporte que les clés qui diffèrent de cette table : le changement qui n'a pas été écrit n'en diffère plus, il n'est jamais réécrit. | Windows 11, antivirus ou indexeur qui tient `play.toml` ouvert : le renommage échoue une fois (A20-01, un seul essai). L'utilisateur avait passé la netteté à 40 %. Il change ensuite le volume : l'écriture réussit, mais la netteté du disque reste à 0. À l'épisode suivant (`player.cpp:374` relit le fichier) la netteté est revenue à 0, sans message. | Ne faire `remember` qu'après un `write_text_atomically` réussi ; en cas d'échec, garder l'ancienne base pour que l'écriture suivante reporte tout. | Lu. Test : rendre le dossier non inscriptible (`chmod 0555` sous Linux), changer une clé, constater l'échec, rendre les droits, changer une autre clé, relire : la première manque. |
| A22-03 | Majeur | `src/app/play/settings.h:106-117`, `src/app/play/settings.cpp:158-168` | Qualité sacrifiée par défaut. Le menu dit lui-même « Before SPAN (best quality) » et « After SPAN (fastest) ». Le défaut Auto passe après SPAN dès que la sortie dépasse 1080 lignes ; or la sortie est `work_height × 2`, soit 1440 au défaut de 720p : Auto vaut donc toujours « le plus rapide », jamais « la meilleure qualité », même sur une RTX 4070 Ti qui tiendrait l'autre. Même cause qu'A09-08, l'angle qualité en plus. | RTX 4070 Ti, source 720p, ×3 : les images intermédiaires sont déformées après SPAN, sans que SPAN nettoie les défauts du warp (`settings.h:100-101`). L'utilisateur qui ne passe pas en réglages avancés ne voit jamais l'ordre de qualité. | Faire décider Auto par la mesure (le budget par image tenu en Avant) plutôt que par la hauteur de sortie ; ou garder Avant par défaut et ne passer Après que si la calibration montre qu'Avant ne tient pas. À confirmer à l'œil. | Lu. À confirmer : sur la machine de Tatoub, comparer Avant et Après à ×3 sur un plan en mouvement (captures `--snapshot`). |
| A22-04 | Mineur | `src/app/play/settings.cpp:453-499` | `change_between` classe en direct ce dont la chaîne dépend à l'ouverture : `rife_factor` avec un moteur chargé, `rife_auto`, `auto_free`, `auto_cap`. Or `top_factor` (warp avant ou après SPAN, taille de l'anneau) se calcule à l'ouverture (`chain_setup.cpp:127`, `:135-149`). Même cause qu'A08-03 ; confirme le renvoi 08 → 22. | Travail 540p (sortie 1080), RIFE ×2 manuel : warp avant SPAN, anneau de 4. Passage à ×4 dans le menu : appliqué en direct, warp toujours avant SPAN (trois SPAN par paire au lieu d'un), anneau de 4. | Classer `Reopen` quand `top_factor` ou `ring_depth` calculés pour `wanted` diffèrent de ceux de `running`. | Lu. |
| A22-05 | Mineur | `src/app/play/settings.cpp:336`, `:345-368` | `settings_for` coupe `rife_auto` quand `--rife` est donné, mais `settings_to_save` ne rend que `rife_factor` et `rife_factor_den`, pas `rife_auto`. | Préréglage Personnalisé, interpolation en Auto. `ajp --rife 2 episode.mkv` une fois, puis un réglage quelconque change : `interpolation.auto = false` est écrit, l'Auto est perdu pour les lancements suivants. | `keep("--rife", &PlaySettings::rife_auto)`. | Lu. Test : `settings_to_save` avec `--rife` donné et `running.rife_auto == started.rife_auto == false` doit rendre `from_file.rife_auto`. |
| A22-06 | Mineur | `src/app/play/settings_toml.cpp:501-534`, `src/app/play/settings_toml_lenient.cpp:72-74` | « Rien de perdu » n'est tenu que pour une clé inconnue, pas pour une valeur inconnue d'une clé connue. La lecture tolérante l'écarte (le défaut la remplace) et l'écriture suivante l'écrase. Pire : `keep_unknown_keys` lit `existing` en strict ; une seule valeur refusée fait échouer la base, et **aucune** clé inconnue n'est alors gardée. | Retour à la version précédente (le lecteur l'offre, `settings_system.cpp`) après une version qui a ajouté un moteur ou un préréglage : à la première fin d'épisode (`player.cpp:575`, écriture sans condition) la valeur nouvelle et toutes les clés nouvelles sont effacées. | Garder dans `fresh` le texte d'origine d'une clé écartée par la lecture tolérante tant que l'utilisateur ne l'a pas changée ; calculer la base de `keep_unknown_keys` en lecture tolérante. | Lu. Test : fichier avec `[interpolation] engine = "futur"` et `future_option = 7`, charger, sauver : les deux doivent rester. |
| A22-07 | Mineur | `src/core/settings.cpp:231-233`, `src/core/toml_fields.h:211-218` | `settings.toml` refuse le fichier entier si `version` manque ou vaut autre chose que 1, alors que `play.toml` lit une version plus récente et garde ses clés (P2.22). Le refus met le fichier de côté (`player.cpp:263-266`). | Retour d'une version qui écrirait `version = 2` : l'assistant se rouvre, les moteurs choisis reviennent à Vulkan seul, TensorRT n'est plus le moteur préféré. | Lire une version plus récente comme `play.toml` : clés connues prises, inconnues gardées (`format_settings` garde déjà les clés inconnues). | Lu. |
| A22-08 | Mineur | `src/core/settings.h:55-58`, `src/core/settings.cpp:163-179` | Code mort et doublon : `audio_languages`, `subtitle_languages` et `audio_delay_ms` de `settings.toml` n'ont aucun lecteur dans le produit. Les langues et le délai vivent dans `play.toml`, avec d'autres règles (balises ISO 639 contre BCP 47, ±5 s contre ±10 s). | Un utilisateur qui édite `[playback] audio_delay_ms` ou `[languages]` dans `settings.toml` ne voit aucun effet. | Retirer ces champs (les garder lisibles et ignorés, avec un test). | Lu (`grep` : aucun usage hors `src/core/settings.*`). |
| A22-09 | Mineur | `src/app/play/settings.cpp:266-324` | `validate` n'examine pas `engines_dir` : un chemin relatif ou à moitié tapé est accepté. Le champ du menu est du texte libre, recopié dans le brouillon à chaque frappe puis écrit (au plus une fois par seconde). | L'utilisateur tape `D:\Moteurs` lettre à lettre et ferme le lecteur au milieu : `D:\Mo` est gardé. Ou un chemin relatif : au prochain lancement, TensorRT s'installe sous le dossier courant, qui change selon qu'on lance par double-clic ou depuis un terminal. | Refuser un `engines.install_dir` non absolu dans `validate` ; ne recopier le champ qu'à la validation (Entrée, perte du focus). | Lu. |
| A22-10 | Mineur | `src/app/play/settings.h:186-191` | Le commentaire dit que le mode Auto suit le GPU « jusqu'à rife_factor ». En mode Rapide (le défaut, `auto_free = true`), `rife_factor` ne plafonne rien : `top_factor` prend la cadence de l'écran (`chain_setup.cpp:141`) et l'écart minimal vient de l'écran (`player_run.cpp:434`). Le ×3 montré par le menu n'a aucun effet ; seul `auto_cap` plafonne. | Écran 144 Hz, source 23,976 : jusqu'à six images par paire, alors que le menu affiche ×3. Voir la section DistilDRBA : c'est le régime où DistilDRBA perd contre RIFE (A03-13). | Faire de `rife_factor` le plafond aussi en Rapide, ou le cacher et le dire ; corriger le commentaire. | Lu. |
| A22-11 | Mineur | `src/app/play/settings.h:299-310` | UX-18 et sa sœur audio : trois bornes par délai. Sous-titres : ±30 s (`kSubtitleDelayMs`, touches z et x) contre ±5 s au curseur. Audio : ±5 s dans `play.toml` et aux touches, ±1 s au curseur (`settings_sound.cpp:72`), ±10 s dans `settings.toml` (mort). Le commentaire du seuil HDR (`:305-306`) est au-dessus de la mauvaise ligne. | Délai audio porté à 2,5 s aux touches pour un casque Bluetooth, puis le curseur du menu touché : il saute à 1 s. | Une seule borne par délai, partagée par le curseur et les touches ; déplacer le commentaire. | Lu. |
| A22-12 | Mineur | `src/app/play/settings.cpp:481-490`, `src/app/play/settings.h:207-218` | Un cran de précision autre qu'Équilibre coupe en silence le moteur fusionné (DistilDRBA : `drba_fused_choice.cpp:9` ; RIFE : A09-02). `change_between` annonce « DistilDRBA switches to Performance » sans dire que le moteur fusionné est abandonné, et `motion_precision` est commun à RIFE et DistilDRBA : passer de RIFE en Performance à DistilDRBA garde Performance. | DistilDRBA en Performance (FP8) pour aller plus vite : il quitte le moteur fusionné (0,92 ms par passe en FP16 d'après `settings.h:208-209`) pour le moteur entier ; il peut être plus lent qu'en Équilibre. | Dire dans la raison que le moteur fusionné est quitté ; ou proposer Performance à DistilDRBA seulement quand une variante fusionnée FP8 existe. | Lu. À confirmer : sur la 4070 Ti, temps par passe de DistilDRBA en Équilibre fusionné contre Performance entier, à 1080p (statistiques `Tab`). |
| A22-13 | Mineur | `src/app/play/settings_toml.cpp:570`, `:579`, `:583`, `:586`, `src/app/play/settings.cpp:331` | Cinq `path::string()` dans la zone, contre la règle de la CI. D'après A20-02 le manifeste UTF-8 évite l'exception sous Windows 11 : dette plutôt que plantage. | Message ou journal mal encodé sur un profil non ASCII si le manifeste manque (outil lancé à part). | `core::to_utf8`. | Lu. |
| A22-14 | Mineur | `src/app/play/settings.h:180`, `:183`, `src/app/play/settings.cpp:249-259` | Deux défauts de qualité à juger à l'œil. Réduction : le défaut est Hermite (cubique B = C = 0, sans lobe négatif, donc plus doux), alors que le menu marque Spline36 « le plus fin » et le dit « choix net d'AnimeJaNai ». Restaurer les traits, actif par défaut : une source plus haute que `work_height` (1080p pour 720p de travail) est réduite puis réagrandie, ce qui jette du détail de la source « pour quatre fois moins de calcul » (`settings.h:177-179`). | Source 1080p Blu-ray nette sur une RTX 4070 Ti : la chaîne par défaut réduit à 720p en Hermite, puis SPAN refait 1440p. | Mesurer et regarder : Hermite contre Spline36 avant SPAN ; Restaurer les traits activé contre coupé sur des sources 1080p nettes. Ne garder le choix rapide par défaut que s'il ne perd pas à l'œil. | À confirmer : `ajp compare` (essai B `--no-restore-lines`) et captures `--snapshot` avec les deux filtres. |
| A22-15 | Mineur | `tests/unit/app/settings_test.cpp` | Aucun test ne vérifie que chaque champ de `PlaySettings` est soit appliqué (`live_part`), soit annoncé (`change_between`) : d'où A22-01. Aucun test d'écriture refusée (A22-02), d'une valeur inconnue d'une version plus récente (A22-06), ni de `settings_to_save` avec `--rife` (A22-05). | Un champ ajouté demain retombe dans le cas d'A22-01 sans que la CI le voie. | Un test qui, pour chaque champ, change la valeur dans `wanted` et exige `live_part(...) != running` ou `change_between(...).cost == Reopen` ; plus les trois tests décrits. | Lu. |
| A22-16 | Mineur | `src/app/play/settings_toml.cpp:593-627`, `src/app/play/settings_toml_lenient.cpp:72-93` | Une écriture de `play.toml` fait quatre analyses TOML, trois mises en forme, une lecture tolérante, puis `keep_unknown_keys` copie la table et relit tous les réglages une fois par feuille du fichier (environ 90). À l'accueil, cela tourne sur le thread de l'interface une fois par seconde pendant qu'on règle (renvoi 24). | Curseur tiré à l'accueil sur un disque lent : l'interface accroche à chaque seconde. | Ne relire pour une feuille que si elle n'est pas une clé connue (liste des clés écrites par `format_settings`) ; passer l'accueil par `LatestWriter`. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| UX-01 | Toujours présent, plus large | A22-01 : `restore_lines` et `gpu`, mais aussi `rife_fused`, `drba_fused` sans interpolation et les délais par appareil seuls. |
| MED-17 | Réfuté | Le délai n'est jamais écrit dans `play.toml` (`format_settings` ne le connaît pas), et chaque fichier suivant repart du fichier relu : `player.cpp:374` (nouvelle vidéo) et `:616` (fin de lecture). `settings.cpp:420` ne le garde que d'une réouverture à l'autre de la même vidéo, ce qui est voulu. Le renvoi 17 → 22 est donc réfuté. |
| UX-18 | Toujours présent | A22-11 : ±30 s aux touches, ±5 s au curseur ; le délai audio a le même défaut. |
| UX-20 | Toujours présent (hors zone) | Le code est dans la zone 08, déjà confirmé par l'audit 08. Dans la zone 22, `label()` et la raison de `change_between` sont traduits à chaque image : rien à corriger ici. |
| UX-19 (point de la fiche) | Corrigé en lecture, toujours présent ailleurs | En lecture, `play.toml` passe par `LatestWriter`, au plus une fois par seconde (`player_run.cpp:642-646`). À l'accueil l'écriture reste synchrone sur le thread de l'interface (`home_screen.cpp:244-250`, renvoi 24), comme à la réouverture (`player_run.cpp:334-339`, A08-13) et au fichier suivant (`player.cpp:575-578`). Le champ du dossier des moteurs écrit à chaque frappe (A22-09). |
| Renvoi 08 → 22 | Confirmé | A22-04 (même cause qu'A08-03). `restore_lines` y est traité par A22-01 ; `preset` reste direct à raison : le préréglage refait la chaîne à la vidéo suivante, et le panneau recalcule le brouillon. |
| Renvoi 09 → 22 | Confirmé | A22-03 (même cause qu'A09-08), avec la perte de qualité qu'il entraîne. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 24 | `src/app/play/home_screen.cpp:244-250`, `:333` | `save_settings` synchrone sur le thread de l'interface, une fois par seconde pendant un réglage à l'accueil (UX-19, A22-16). |
| 07 | `src/app/play/auto_config.cpp:202` | Qualité : l'Auto calibré passe SPAN en Performance (FP8) dès que la calibration a retenu un modèle, alors que `auto_config.cpp:166-169` dit que les préréglages ne posent plus le FP8 tant que sa qualité n'est pas jugée à l'œil. |
| 07 | `src/app/play/auto_config.cpp:203-205` | L'Auto calibré écrase `auto_free` et le modèle RIFE à chaque vidéo sans passer en Personnalisé : un choix Stable fait au menu ne survit pas à la vidéo suivante (proche d'A07-06). |
| 23 | `src/app/play/settings_sound.cpp:72`, `src/app/play/settings_panel.cpp:199` | Bornes des curseurs de délai différentes de celles des réglages et des touches (A22-11). |
| 23 | `src/app/play/settings_engines.cpp:81-84` | Dossier des moteurs recopié à chaque frappe (A22-09). |
| 09 | `src/app/play/chain_setup.cpp:359-361` | Qualité : au-dessus de 720 lignes de travail, RIFE estime le mouvement sur une copie réduite à 720p. Choix de vitesse à confirmer à l'œil sur une source 1080p. |
| 18 | `src/app/play/options.cpp:298`, `src/app/play/settings.cpp:326-343` | `--rife-model` n'entre pas dans `settings.rife_model` : le menu et `change_between` raisonnent sur un autre modèle que celui chargé (semble ; non suivi). |

## DistilDRBA dans les réglages

Rien dans la zone ne ralentit le calcul de DistilDRBA lui-même. Trois choses faussent en revanche la comparaison avec RIFE
ou privent DistilDRBA de sa voie rapide :

- **Le mode Rapide pousse DistilDRBA dans son pire régime.** Par défaut (`rife_auto = true`, `auto_free = true`,
  `auto_cap = 0`) le nombre d'images par paire suit l'écran, pas `rife_factor` (A22-10) : environ 6 sur un écran 144 Hz.
  DistilDRBA fait une passe par image créée, RIFE une par paire (A03-13) : au-delà de ×5, DistilDRBA calcule plus de
  réseau que RIFE. Pour comparer vraiment les deux, il faut fixer le même facteur : `interpolation.auto = false` et
  `rife = 2` puis `rife = 3`, ou `auto_cap = 72` (×3 sur 24 i/s). C'est la première manipulation à faire sur la 4070 Ti
  avant de conclure que DistilDRBA n'est pas plus rapide.
- **Le cran Performance quitte le moteur fusionné** sans le dire (A22-12). Le moteur fusionné n'existe qu'en Équilibre.
- **L'ordre warp/SPAN** n'est réglable dans le menu que pour RIFE (renvoi 03 → 23 déjà ouvert), alors qu'il décide la
  variante mouvement de DistilDRBA. Avec le défaut Auto, DistilDRBA est toujours en variante mouvement après SPAN
  (A22-03), la plus rapide : ce n'est pas lui qui freine.

`kDistilDrbaInAuto = false` (`auto_config.h:22`) fait qu'aucun préréglage ne choisit DistilDRBA : il faut le prendre à la
main (niveau « Très rapide »), ce qui passe le préréglage en Personnalisé (`preset_chain_differs` compare
`motion_engine`). C'est le bon comportement ; le choix est bien gardé dans `play.toml`.

## Qualité sacrifiée pour la vitesse (zone 22)

| Où | Choix par défaut | Avis |
|---|---|---|
| `settings.h:106-117` | Warp après SPAN (« le plus rapide ») sur presque toute lecture | A22-03, Majeur. |
| `settings.h:183` | Réduction Hermite, alors que Spline36 est marqué « le plus fin » | A22-14, à confirmer à l'œil. |
| `settings.h:180` | Restaurer les traits : une source 1080p est réduite au travail puis réagrandie | A22-14, à confirmer à l'œil. |
| `settings.h:213-218` | Précision Équilibre (FP16) pour SPAN, RIFE et DistilDRBA | Bon défaut : le FP8 reste un choix manuel. Hors zone, l'Auto calibré passe pourtant SPAN en FP8 (renvoi 07). |
| `settings.h:206`, `:212` | Moteurs fusionnés actifs | Sans perte annoncée ; la concordance reste à prouver (A03-02, A03-03). |

## Points positifs

- La lecture tolérante écarte une valeur fautive seule, nomme la clé, et ne remet jamais tout le fichier aux défauts ;
  un `play.toml` illisible est mis de côté, pas écrasé (`player.cpp:176-182`).
- L'écriture est atomique, garde les clés d'une version plus récente (P2.22) et fusionne avec un autre lancement (P2.14).
- `LatestWriter` est simple et correct : une écriture par fichier, exceptions captées, `flush` avant la relecture.
- Les modèles SPAN et RIFE nommés dans `play.toml` sont limités au catalogue (`model_choice.cpp:48`, `:63`) : pas de
  chemin arbitraire.
- `LiveSettings` garde le facteur dans un seul atomique (numérateur et dénominateur jamais séparés).
- `subtitle_delay_ms` et `hdr_native` ne sont jamais écrits, comme voulu.

## Questions ouvertes

- A22-03 et A22-14 sont des choix de qualité contre vitesse : la consigne de Tatoub est de ne presque jamais sacrifier
  la qualité. Le fil a retenu l'option la plus probable (les relever comme défauts, à confirmer à l'œil) sans poser la
  question.
