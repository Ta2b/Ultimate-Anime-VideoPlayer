# Rapport de l'audit 08. Boucle du lecteur et ouverture de la chaîne

Branche : `claude/project-thread-a15k0r`. Commit audité : `73f81b3`. Date : `2026-11-09`.

## Verdict

La boucle tient debout et son ordre d'arrêt est pensé, mais tout ce qui rouvre ou replie la chaîne est fragile :
« Annuler et regarder » est avalé par les replis, un fil de relevé lit le backend après sa destruction, un
facteur changé en direct garde l'anneau et le warp de l'ouverture, et les raccourcis de séance finissent dans
`play.toml` ; les 13 défauts connus sont tous encore là.

## Cadrage retenu

Question posée à Tatoub le 09/11 : ce qui compte le plus dans la boucle. Réponse de Tatoub : **tout à égalité**.
Les réouvertures (et les constructions TensorRT qu'elles provoquent), l'écart entre la chaîne voulue et la chaîne
jouée, la fenêtre qui ne doit jamais figer, et la complexité de `run_chain` ont reçu la même profondeur. Rien n'est
mis de côté ; l'export (`StartExport`) et Core ML (`prepare_apple`) sont hors cible, lus seulement là où ils
passent par la boucle.

Le conteneur n'a ni GPU ni les dépendances du projet (`~/ajp-deps` absent) : rien n'a été compilé ni lancé. Les
défauts sont lus dans le code ; ce qui demande une mesure est marqué « à confirmer » avec la manipulation, et le
test qui échouerait est décrit.

Pour se repérer : `run_chain` (`player_run.cpp:98-751`) ouvre les moteurs par `prepare_chain`
(`chain_setup.cpp:160`), puis la lecture, puis tourne la boucle ; il rend un `Reopen` que `play()`
(`player.cpp:518-590`, zone 18) rejoue. `apply` (`chain.cpp:246`) traduit chaque action en changement en direct
ou en demande de réouverture.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A08-01 | Majeur | `src/app/play/chain_setup.cpp:166-171` | « Annuler et regarder » est pris pour un refus par tous les replis de la préparation, qui lancent alors une autre construction. Le refus `Cancelled` (`kWatchNowKey`, `player_engines.cpp:50-52`) est traité comme un échec ordinaire : variante SPAN suivante (`chain_setup.cpp:345-350`), RIFE à la place de DistilDRBA (`:402-406`), RIFE entier à la place du fusionné (`:470-472`), DistilDRBA entier à la place du fusionné (`player_engines.cpp:259-260`), et pour finir toute la chaîne refaite au cran Équilibre (`chain_setup.cpp:162-171`). | Préréglage avec SPAN en Performance sur la 4070 Ti, premier lancement : la construction de la variante dure, Tatoub clique « Annuler et regarder ». La variante suivante se construit, puis le FP16, puis la chaîne repart en Équilibre : il faut cliquer trois ou quatre fois, et chaque clic coûte une construction de plus. | Faire remonter `Cancelled` sans repli : dans chaque repli et dans `prepare_chain`, tester `error().key == kWatchNowKey` (ou `code == Cancelled`) et rendre l'erreur telle quelle. | Lu. Test décrit : un `Backend` factice dont `build` rend `Cancelled` avec la clé ; `prepare_chain` avec `span_precision = Performance` doit rendre ce refus sans second appel à `build` (aujourd'hui, deux appels au moins). |
| A08-02 | Majeur | `src/app/play/player_run.cpp:191` | Le fil de relevé de `ChainStats` lit `kernels_` (qui appartient au backend) après la destruction du backend. `chain_stats` est déclaré après `prepared` et n'est détruit qu'à la sortie de `run_chain`, mais `prepared->reset()` détruit le backend avant (`:748`). Le fil (`chain_stats.cpp:184-208`) se réveille toutes les 500 ms et appelle `kernels_.device_memory()` (`cudaMemGetInfo` sous TensorRT). | OSD des statistiques ouvert (Tab), Tatoub change la hauteur de travail : la chaîne se ferme, la destruction des moteurs TensorRT dure plusieurs centaines de millisecondes, le relevé tombe dedans et lit un objet détruit. Plantage ou valeur fausse, au hasard. | Arrêter le fil avant `prepared->reset()` : un `chain_stats.stop()` explicite après `session.end()`, ou déclarer `chain_stats` dans un bloc qui se ferme avant la remise à zéro des moteurs. | Lu (ordre des déclarations). À confirmer : construction ASan sur Linux, `--stats` avec Tab, puis 20 réouvertures par changement de hauteur ; ou test unitaire avec un `KernelOps` factice détruit pendant que le relevé dort. |
| A08-03 | Majeur | `src/app/play/chain.cpp:204-211` | Le facteur d'interpolation, Auto et Rapide se changent en direct, alors que l'ouverture en a tiré trois décisions qui restent figées : la profondeur de l'anneau (`player_run.cpp:177`, `ring_depth` : 4 à ×2, 6 à ×3-×4, 8 en Rapide), le plus haut facteur (`chain_setup.cpp:237`) et le warp après SPAN (`chain_setup.cpp:127`, `:376-377`, qui décide aussi la variante mouvement de DistilDRBA). `change_between` (zone 22) ne classe aucun de ces réglages en réouverture. | Vidéo ouverte à ×2 manuel en 1080p vers un écran 1440p : anneau de 4, warp avant SPAN. Tatoub passe en Auto Rapide à 165 Hz : jusqu'à 6 instants par paire dans 4 emplacements, et un SPAN par image créée au lieu d'un par source. L'Auto redescend faute de tenir, sans un mot, jusqu'à la vidéo suivante. | Dans `switch_chain` et `ApplySettings`, comparer `ring_depth`, `top_factor` et `warp_after_span` de l'ancienne et de la nouvelle chaîne ; s'ils changent, rouvrir (le menu le dit déjà pour les autres réouvertures). | Lu. À confirmer : sur la 4070 Ti, ouvrir à ×2 manuel, passer en Auto Rapide, noter le palier tenu et les images perdues (`--stats`), puis rouvrir la même vidéo et comparer. |
| A08-04 | Majeur | `src/app/play/chain.cpp:351-358` | Les raccourcis de séance (Ctrl+G, Ctrl+I, Ctrl+0), que `with_kept_choices` garde hors de `play.toml`, y sont écrits par toute réouverture qui part de `to.running` : fichier glissé, épisode voisin (`:339-349`), changement d'écran et épisode suivant (`player_run.cpp:617`, `:723`), moteurs, export, banc. `play()` enregistre alors `Reopen::settings` tel quel (`player.cpp:576`, ou `:585` puis `player_run.cpp:334-340`). À l'inverse, une réouverture demandée par le menu repart de `with_kept_choices` (`chain.cpp:389-393`) et annule le raccourci. | Tatoub coupe tout avec Ctrl+0 pour comparer, l'épisode se finit, le suivant démarre : `play.toml` garde « grade coupé, sans interpolation », et le lancement suivant joue la source brute. Autre cas : Ctrl+I coupe l'interpolation, puis la hauteur de travail changée au menu la rallume. | Porter `kept_grade` et `kept_rife` dans `Reopen` (ou dans `Player`) et écrire `with_kept_choices(...)` partout où `play()` enregistre ; une seule règle pour les deux sortes de réouverture. | Lu. Test décrit : `apply(Ctrl+0)` puis `apply(StepEpisode)` sur une `Chain` factice ; `settings_to_save(*chain.reopen, ...)` ne doit pas changer `grade` ni `rife_factor` (aujourd'hui, il les change). |
| A08-05 | Mineur | `src/app/play/chain_setup.cpp:160-172` | `prepare_chain` refait toute la chaîne au cran Équilibre sur **n'importe quel** refus (modèle absent, backend refusé, mémoire), et le journal parle toujours de crans de précision. La chaîne repliée n'est pas reportée dans `chain.running` : le menu affiche Performance et `play.toml` le garde, alors que le FP16 tourne. | Un modèle RIFE manquant fait tout reconstruire une seconde fois avant le même refus ; un FP8 refusé laisse le menu dire « Performance » pendant toute la lecture, et chaque réouverture refait l'essai. | Ne replier que sur un refus de chargement d'une variante (code dédié) ; rendre la chaîne réellement chargée pour que `run_chain` corrige `settings` et le dise une fois à l'écran. | Lu. |
| A08-06 | Mineur | `src/app/play/chain_setup.cpp:376-389` | DistilDRBA en variante mouvement sous 4K : quand la dérivation mouvement échoue, `prepare_distil_drba` construit et charge quand même le moteur image à 720p, que `prepare_chain_once` jette aussitôt pour en charger un second à la taille de travail. | À chaque ouverture : une construction (la première fois) puis un chargement inutile, et la mémoire vidéo de deux moteurs en même temps. | Rendre l'échec de la dérivation avant de construire (essayer `drba_motion_model` d'abord), ou passer la taille de repli à `prepare_distil_drba`. | Lu. À confirmer : journal d'une ouverture 4K avec DistilDRBA, deux lignes « moteur DistilDRBA » attendues. |
| A08-07 | Mineur | `src/app/play/player_run.cpp:549-562` | `follow_display` (pic automatique, soins OLED, vrai noir de l'écran courant) n'est appelé que dans la branche du HDR, relue toutes les 2 s, et jamais avec `--sdr`. | Fenêtre glissée d'un OLED vers un écran LCD : jusqu'à 2 s avec le noir et le pic de l'ancien écran ; avec `--sdr`, toute la lecture. | Appeler `follow_display` dès que `display_changed` (`:563-566`), hors de la condition `!p.options.sdr`. | Lu. |
| A08-08 | Mineur | `src/app/play/player_engines.cpp:54-67` | Pendant une construction de moteur, la boucle d'attente jette les événements qu'elle ne connaît pas : un fichier glissé, un changement d'écran. La préparation du SPAN FP8 (`:169-187`) n'offre pas « Annuler et regarder » (`watch_now` absent de sa vue) : seule la sortie l'arrête. | Une vidéo glissée pendant une construction de plusieurs minutes est perdue sans un mot. Pendant la préparation FP8 à la main, on ne peut que quitter. | Garder les fichiers glissés pour la réouverture (comme `take_drop`), passer `watch_now` à la vue de `prepare_span_fp8`. | Lu. |
| A08-09 | Mineur | `src/app/play/chain_setup.cpp:277` | `path::string()` sur des chemins du profil de l'utilisateur, contre la règle de la CI : la bibliothèque du backend (`:277`), l'artefact du moteur (`player_engines.cpp:143`), les dossiers de DistilDRBA (`player_engines.cpp:235`), le modèle RIFE choisi (`chain_setup.cpp:84`, `:91`). | Windows 11, compte au nom japonais sur un système en page de codes latine : la conversion lève une exception à l'ouverture de la chaîne. | `core::to_utf8` partout (même cause que A03-06). | Lu. |
| A08-10 | Mineur | `src/app/play/engine_build.cpp:38` | Chaque construction ou lecture du cache crée sa propre instance de backend (`BackendLibrary::create`) : une ouverture TensorRT avec SPAN, RIFE fusionné et DistilDRBA en crée jusqu'à six de plus, même quand tous les moteurs sont en cache. | Temps d'ouverture et de réouverture plus long que nécessaire (contexte CUDA, chargement de nvinfer par instance). | Une instance de construction pour toute la préparation, ou lire le cache sans backend. | À confirmer : journal horodaté d'une réouverture TensorRT tout en cache, temps entre « Loading » et la première image. |
| A08-11 | Mineur | `src/app/play/player_run.cpp:98` | `run_chain` fait environ 650 lignes en une fonction (complexité désactivée par `NOLINT`), le fichier 753 lignes contre 500 visés et 800 qui font échouer la CI ; `prepare_chain_once` fait environ 300 lignes (QUA-15). Les trois lambdas de la boucle (`apply_auto`, `show_auto`, la branche de l'écran) partagent une quinzaine de variables locales. | Toute nouvelle fonction dans la boucle fait passer le fichier au-dessus de 800 lignes ; aucune partie de la boucle ne se teste seule (voir A08-12). | Sortir l'ouverture (jusqu'à `Session`), le suivi de l'écran (VRR, HDR, cadence) et la fin (`Reopen`, épisode suivant) en objets à part, chacun testable sans fenêtre. | Lu. |
| A08-12 | Mineur | `tests/unit/app/chain_setup_test.cpp` | Les tests de la zone ne couvrent que `top_factor`, `ring_depth` et `free_requested`. Rien sur `apply`, `switch_chain`, `with_kept_choices`, les replis de `prepare_chain`, ni la fin de `run_chain` (épisode suivant, refus de taille). | Les défauts A08-01, A08-03, A08-04 et A08-05 passent sans qu'aucun test ne bouge. | Tests de `apply` sur une `Chain` factice (A08-04), de `prepare_chain` sur un backend factice (A08-01, A08-05), de la décision de réouverture (A08-03). | Lu. |
| A08-13 | Mineur | `src/app/play/player_run.cpp:334-340` | Des E/S restent sur le thread de la boucle : `play.toml` d'une réouverture écrit de façon synchrone (`:334-340`) alors que `LatestWriter` existe trois lignes plus bas, `file_size` sur la vidéo (`:446`), `SDL_GetPowerInfo` deux fois par seconde (`apply_auto`, `:414`), qui lit `/sys/class/power_supply` sous CachyOS. | Vidéo sur un NAS lent ou disque en veille : la première image attend le disque ; sur un portable, une lecture de sysfs par tour de relevé. | Passer l'écriture par `writer`, lire la taille dans `Pipeline::open`, relever l'alimentation toutes les 10 s. | Lu. À confirmer : `strace -f -e trace=openat` sur le thread principal pendant une minute de lecture. |
| A08-14 | Mineur | `src/app/play/engine_switch.cpp:13-19` | L'onglet Moteurs connaît encore MIGraphX et OpenVINO, retirés des sources (CLAUDE.md), et leurs tests les vérifient (`engine_switch_test.cpp:28`, `:38`). L'en-tête d'`engine_build.h:5` annonce que `preparer` prendra sa place, alors que `preparer` est mort (A05-08). | Code et commentaires qui décrivent un produit qui n'existe plus ; un `.so` d'une ancienne version laissé dans le dossier serait encore proposé. | Retirer les deux entrées et leurs tests, corriger l'en-tête (avec A06-11). | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-01 | Toujours présent | `player_run.cpp:383-384` et `:600` : l'état armé n'est pris que si l'EDID dit le VRR possible (`display->vrr`) ; sans EDID lisible (`p.display` vide), jamais. La cause de lecture est en zone 11. |
| GPU-03 | Toujours présent | `player_run.cpp:563-568`, `:615-617`, `auto_config.cpp:223` : toute autre hauteur rouvre aussitôt, sans délai ni retour possible ; `opened_height` est celle de la nouvelle ouverture, donc un aller-retour entre deux écrans rouvre deux fois. `play()` refait le préréglage pour la nouvelle hauteur (`player.cpp:405-414`) : nouvelle hauteur de travail, donc construction TensorRT possible. La relecture toutes les 500 ms le déclenche dès que le centre de la fenêtre passe la frontière, en plein glisser. |
| MED-01 | Toujours présent | `player_run.cpp:718-725` : l'épisode suivant part sur toute fin de session qui n'est pas un refus fatal, sans vérifier que la position est près de la durée. La cause (erreur d'E/S rendue en fin de flux) est en zone 14. |
| TRT-02 | Toujours présent | `player.cpp:525-534` coupe `upscale` et l'interpolation, mais `chain_setup.cpp:238` garde `work_height` quand `upscale` est faux : la source est réduite à la hauteur de travail puis agrandie par libplacebo. Même chose avec `--no-upscale`. Voir aussi A08-01 : le clic n'arrive souvent même pas jusque-là. |
| UX-02 | Toujours présent | `player.cpp:541-559` puis `:585` : après l'assistant relancé (`handle_engine_change`, qui écrit `play.toml`), `settings` reprend `Reopen::settings`, les anciens, que `player_run.cpp:334-340` réécrit à l'ouverture suivante. |
| UX-13 | Hors zone, non revérifié | Le texte est dans `settings_system.cpp` (zones 23 et 39) ; rien dans cette zone ne touche au journal envoyé. |
| GPU-02 | Toujours présent | `player_run.cpp:619-629` : la branche plein écran pose `vrr_armed` sans `vrr_watch`, qui garde un état périmé ; si elle passe à armé et que le VRR se désengage ensuite, `observe` ne voit aucun changement. Le cache de 250 ms est en zone 11. |
| GPU-05 | Toujours présent | `player_run.cpp:619-629` : `apply_floor()` n'est appelé que si l'état armé change ; un écran VRR remplacé par un autre écran VRR garde le plancher de l'ancien. |
| GPU-13 | Toujours présent | `player_run.cpp:90`, `:549` : la bascule du HDR attend la relecture des 2 s, même quand `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` est arrivé (seul `contract_watch` l'écoute). |
| MED-07 | Toujours présent | `chain.cpp:410` : `audio_delay_ms` est le délai propre à l'appareil (`settings_sound.cpp:29-31`) et part dans l'export. |
| MED-15 | Toujours présent | `player_run.cpp:166` (`Pipeline::open`), `:720` et `chain.cpp:339` (`sibling_episode`, liste du dossier) sur le thread de la fenêtre. |
| UX-08 | Toujours présent | `chain.cpp:306-312` : Échap arrive en `LeaveFullscreen` et ferme tout le panneau ; la cause (le panneau ne garde pas la touche quand une liste est ouverte) est en zone 23. |
| UX-20 | Toujours présent | `player_run.cpp:148`, `:199`, `:210` : en-tête et ligne de sortie formatés à l'ouverture ; `apply_live` change la langue (`chain.cpp:113`) mais ne refait l'en-tête que si le facteur, la réduction ou le HDR changent (`:129-132`), et jamais la sortie ni les `choices`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 22 | `src/app/play/settings.cpp:453-499` | `change_between` ne classe en réouverture ni `rife_factor` (moteur déjà chargé), ni `rife_auto`, `auto_free`, `auto_cap`, `restore_lines`, `preset`, dont `prepare_chain` dépend (voir A08-03, UX-01). |
| 05 | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` joint le fil de construction sur le thread de la fenêtre : si `trtexec` n'honore pas l'arrêt vite, la fenêtre fige après « Annuler et regarder » ou la sortie. |
| 11 | `src/app/play/host_facts.cpp:378` | `on_battery()` (`SDL_GetPowerInfo`) appelé à chaque relevé de l'écran, deux fois par seconde (voir A08-13). |
| 14 | `src/app/play/player_run.cpp:589-591` | MED-01 : la session doit dire si sa fin est une vraie fin ou une erreur d'E/S, pour que la boucle ne lance pas l'épisode suivant. |
| 18 | `src/app/play/player.cpp:525-534` | La reprise après « Annuler et regarder » modifie `settings` pour toute la suite de la vidéo, y compris après une réouverture demandée par le menu. |
| 23 | `src/app/play/settings_panel.cpp` | UX-08 : Échap devrait fermer la liste ouverte avant le panneau. |
| 39 | `src/app/play/settings_system.cpp` | UX-13 : chemins hors du dossier personnel restés dans le journal envoyé. |

## Points positifs

- L'ordre d'arrêt est écrit et suivi (`player_run.cpp:741-748`, déclarations de `ChainEngines`) : session, écran,
  étages, moteurs, puis backend.
- `play.toml`, la position et la mémoire de l'Auto passent par `LatestWriter` hors du thread de la boucle, et
  une chaîne qui ne s'ouvre pas n'est jamais enregistrée (`Player::unsaved`).
- Une source qui change de taille sans fin est refusée (4 fois par minute) au lieu de boucler.
- Les relevés de mémoire et de NVML de l'OSD tournent sur un fil à part, démarré seulement quand l'OSD les
  demande.
- La lecture du VRR demande deux lectures concordantes avant d'agir (`VrrWatch`), ce qui évite les clignotements.

## Questions ouvertes

- Les raccourcis de séance doivent-ils survivre à l'épisode suivant (sans aller dans `play.toml`), ou s'arrêter
  à la vidéo ? A08-04 propose la première lecture.
- Pour GPU-03 : une hystérésis (attendre quelques secondes sur le nouvel écran) suffit-elle, ou faut-il ne rouvrir
  que si la chaîne du préréglage change vraiment pour la nouvelle hauteur ?
