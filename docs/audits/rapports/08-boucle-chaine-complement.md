# Complément de l'audit 08. Renvois reçus des autres audits

Branche : `claude/project-thread-lg04t4`. Commit audité : `d5df90d` (aucun fichier de `src/` n'a changé depuis
`73f81b3`, le commit du rapport 08). Date : `2026-11-09`.

## Verdict

Sur les 21 renvois, trois ouvrent un défaut nouveau (A08-15 à A08-17), dont un majeur : avec « Restaurer les
traits » coupé, la source que la chaîne croit garder telle quelle est réduite à 720p par l'étage puis agrandie
par libplacebo. Quinze renvois sont des doublons de défauts déjà relevés, et trois sont réfutés.

## Cadrage retenu

Consigne du coordinateur, relayée sans question à Tatoub : vérifier seulement les renvois de
`complements/08.md` dans le code, et conclure pour chacun : confirmé, réfuté ou doublon. Une baisse de qualité
d'image faite pour la vitesse compte comme défaut tant qu'aucune mesure ne la justifie. Le VRR reste hors sujet
pour Tatoub, et les renvois qui en parlent sont classés sans recherche de plus.

Comme pour le rapport 08, le conteneur n'a ni GPU ni dépendances : rien n'a été compilé ni lancé. Tout est lu
dans le code, et ce qui demande une manipulation le dit.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A08-15 | Majeur | `src/app/play/chain_setup.cpp:99`, `:238` ; `src/app/play/player_run.cpp:170` | La taille de travail est calculée deux fois avec deux règles. `prepare_chain_once` met la source telle quelle quand SPAN ne tourne pas parce que « Restaurer les traits » est coupé (`upscale && !span`, `:238`). Mais l'anneau (`player_run.cpp:170`) et l'étage (`ChainEngines::upscale`, `:99`) reprennent `work_size(settings.work_height)` sans cette exception. Les moteurs, le libellé (`c.reduce`) et le journal (« source gardée telle quelle, sans SPAN », `:229`) suivent la première taille, l'image suit la seconde. | Source 1080p, hauteur de travail 720, `restore_lines = false` (par `--no-restore-lines` ou `play.toml` ; depuis le menu une fois A22-01 corrigé). L'étage réduit la source en 1280 × 720, l'anneau est en 1280 × 720 × 1, et libplacebo l'agrandit à l'écran. Tatoub voit une image 720p plus douce que la source, alors que l'OSD annonce « 1920x1080 » et « sans agrandissement ». Il y a aussi un refus en pleine lecture : en RIFE image (`--rife-image`, ou dérivation mouvement refusée), le moteur est construit sur `c.image_geometry` (1920 × 1088, `:448`) alors que l'étage lui passe des tenseurs en 1280 × 736. `TrtEngine::place` refuse alors à la première image créée (« forme ou type différents du moteur », `trt_engine.cpp:229-233`), et la lecture s'arrête. DistilDRBA image rechargé à `c.work` (`:384-388`) donne le même refus. | Calculer la taille de travail une seule fois, avec une seule fonction (par exemple `chain_work_size(settings, span, w, h)`), et s'en servir pour `c.work`, l'anneau et `upscale()`. Un test de `chain_setup_test.cpp` le garde : avec `restore_lines = false`, `work_height = 720` et une source 1920 × 1080, `c.upscale(...).work_height` doit valoir `c.work.height` (1080). Aujourd'hui il vaut 720. | Lu. À confirmer : `ajp --no-restore-lines` sur une source 1080p, puis comparer la taille de l'anneau dans le journal (« slot … ») et l'OSD. |
| A08-16 | Mineur | `src/app/play/chain_setup.cpp:162-171` | Le repli de `prepare_chain` remet **les deux** crans à Équilibre, même quand un seul moteur a été refusé. Le cran qui a chargé sans souci perd aussi sa précision. Si RIFE `_fp32` est refusé, `prepare_chain_once` rend une erreur (`:497`, `:515`) : le SPAN FP32, pourtant chargé, est jeté et reconstruit en FP16. Dans l'autre sens, un SPAN refusé fait passer RIFE en FP16. C'est une baisse de qualité qui n'a rien à voir avec la vitesse, et qu'aucune mesure ne justifie. | Cran Qualité pour SPAN et pour l'interpolation sur la 4070 Ti, et RIFE FP32 refusé à 1080p (mémoire vidéo, par exemple). Toute la lecture se joue en SPAN FP16 alors que le SPAN FP32 tenait, et le premier essai a pu coûter une construction TensorRT. Le menu affiche toujours Qualité (A08-05). | Replier seulement le cran du moteur refusé : faire porter l'étage fautif par le refus (SPAN ou mouvement), et ne remettre à Équilibre que ce cran-là. L'affichage à l'écran et la chaîne réellement jouée restent ceux de la correction d'A08-05. | Lu. Test décrit : `prepare_chain` sur un backend factice qui refuse seulement le RIFE `_fp32` doit rendre `span_precision = Quality` dans la chaîne chargée. |
| A08-17 | Mineur | `src/app/play/player.cpp:162` | Treize `path::string()` dans `player.cpp`, contre la règle de la CI (`tools/ci/check-path-string.sh`) : lignes 86, 89, 90-91, 101, 109, 162, 217, 238, 366, 378-379, 598, 653-654. Le manifeste UTF-8 (`cmake/utf8.manifest`) empêche l'exception sous Windows 10 1903 et plus (A20-02), donc ces appels n'ont pas d'effet visible : c'est une dette. | Aucun plantage. Le compteur de la CI ne baisse pas, et un poste sans le manifeste (Windows plus ancien, binaire lancé autrement) retomberait sur l'exception exactement quand un fichier au nom japonais ne se lit pas (`:162`, `:366`, `:598`). | `core::to_utf8(...)` partout, comme A08-09 et A25-16. | Lu. |

## Renvois traités

| De | Où | Conclusion | Note |
|---|---|---|---|
| 07 | `src/app/play/chain_setup.cpp:99` | Confirmé : **A08-15** | Voir plus haut. |
| 09 | `src/app/play/chain_setup.cpp:420-423` | Doublon d'**A09-02** et d'**A31-06** | `rife_fused_declined` (`rife_fused_choice.cpp:11`) refuse le fusionné dès que `motion_precision` n'est pas Équilibre, même quand le graphe `_fp8` manque et que le repli « Équilibre à la place » (`:418-427`) reprend le modèle FP16 : on obtient un RIFE entier, plus lent, sans gain de qualité. |
| 10 | `src/app/play/player_run.cpp:654` | Réfuté pour la boucle ; le reste est un doublon d'**A31-02** et d'**A10-01** | Une erreur de `show` (`session.cpp:535-536`) devient `failure`. `run_chain` enregistre la position (`remember()`) et sort par l'ordre d'arrêt normal. Aucun `Reopen` n'est rendu (`:727`, `!failure`), et `play()` sort de la boucle (`player.cpp:536-539`). Lancé par double-clic, l'accueil s'ouvre avec « "…" stopped: … » (`:598`) ; un lancement scripté sort avec l'erreur. Il n'y a donc pas de réouverture en boucle. Ce qui manque, c'est une relance après la perte du GPU (A31-02) ; et la perte la plus courante, l'acquisition refusée, ne remonte même pas jusqu'ici (A10-01). |
| 11 | `src/app/play/player_run.cpp:398-400` | Doublon d'**A11-02** | Toujours présent : `display_period_us` rappelle `display_refresh_hz` (`host_facts.cpp:89`), donc deux `windows_mode` par relevé. A11-02 le cite déjà. |
| 11 | `src/app/play/player_run.cpp:549-560` | Doublon de **GPU-13**, déjà revérifié par 08 ; la correction est dans **A11-02** | Le relevé du HDR attend toujours `kHdrPollUs`. `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` n'est écouté que par `ContractWatch`. Voir aussi A08-07 (`follow_display` hors de la branche HDR). |
| 11 | `src/app/play/player_run.cpp:619-621` | Doublon d'**A11-01** | `ContractWatch` copie `DisplayFacts` à sa construction (`display_contract.h:54`) et `p.display = display` (`:621`) ne le prévient pas. |
| 12 | `src/app/play/player_run.cpp:622-629` | Doublon de **GPU-05**, déjà revérifié par 08 | VRR hors sujet. `apply_floor()` n'est rappelé que si l'état armé change (`:625-628`). |
| 12 | `src/app/play/player_run.cpp:420-438` | Doublon d'**A12-04** | `limit_hz` (plafond ou batterie) sert de fréquence d'écran à `auto_limits` (`:429`), à `set_free` (`:434`) et à `set_steady` (`:436`). Conséquence de plus, même cause, même correction : à fréquence fixe, l'écart minimal du Rapide vaut `1e6 / limit_hz`. Avec un plafond de 120 sur un écran à 165 Hz, on obtient 8,33 ms, soit deux vsyncs (12,1 ms) : environ 82 i/s au lieu de 120 (calculé, non mesuré). |
| 12 | `src/app/play/session.cpp:665-672` | Doublon de **GPU-19**, revérifié par 12 et 13 | VRR hors sujet. `follow_display` appelle `vsync_speed` sans regarder `config_.vrr_floor_us`. |
| 13 | `src/app/play/player_run.cpp:738-739` | Réfuté (comme l'a conclu 29 pour A13-09) | Quand la réouverture vise une autre vidéo (`next_file`, épisode suivant, fichier glissé, `PlayFile`), `play()` sort de la boucle (`player.cpp:574-584`), et un nouveau `Player` est construit sans `audio_track` ni `subtitle_track` (`player.cpp:454-468`). Les pistes copiées aux lignes 738-739 partent avec l'ancien `Player`. Pour la même vidéo, A13-01 (index externe gardé à la réouverture) reste vrai. |
| 15 | `src/app/play/player.cpp:95-117` | Doublon d'**A15-07** | `probe_once` ne refuse que le Dolby Vision profil 5 (`:108-112`). Les formats que `track_color` sait illisibles dès la sonde sont refusés à la première image : c'est le point (4) d'A15-07. |
| 16 | `src/app/play/pipeline.cpp:97-110` | Doublon d'**A16-07** (format figé) et d'**A16-06** (ordre `seek()` puis `AudioTrack`) | Code inchangé. |
| 21 | `src/app/play/chain_setup.cpp:361-363` | Doublon d'**A49-10** (RIFE) et d'**A03-19** (DistilDRBA) | Baisse de qualité faite pour la vitesse. Aucune mesure ne la justifie : `docs/adr/0002-span-une-fois-par-source.md:12-13` pose la règle des 720p sans chiffre de qualité ni de temps propre à l'estimation, et le seul gain chiffré de l'ADR (« environ la moitié du temps GPU ») vient du warp après SPAN, pas de l'estimation réduite. |
| 21 | `src/app/play/engine_screen.cpp:343` | Doublon d'**A25-16** | Dette (A20-02). |
| 23 | `src/app/play/chain.cpp:183-185` | Doublon d'**A23-12** | Côté chaîne, chaque appel fait `set_subtitle_delay` et `notify`. La garde manque dans le panneau (`settings_panel.cpp:199`). |
| 25 | `src/app/play/player.cpp:559` | Doublon d'**A25-01** (UX-02, déjà revérifié par 08) | `settings = (*outcome)->settings` reprend la chaîne d'avant `handle_engine_change` (`:541`), et `player.unsaved` (`:585`) la réécrit à l'ouverture suivante. |
| 24 | `src/app/play/player.cpp:163` | Confirmé : **A08-17** | Dette, sans exception sous Windows 11 (A20-02). |
| 31 | `src/app/play/chain_setup.cpp:160-171` | Doublon d'**A08-05** pour le silence et le menu faux ; confirmé : **A08-16** pour le repli des deux crans | Le repli est retenté à chaque ouverture, donc un refus passager (mémoire vidéo) ne dure que le temps de cette lecture. |
| 33 | `src/app/play/engine_build.cpp` | Doublon d'**A08-10** | `EngineBuild::run` crée toujours une instance par moteur (`engine_build.cpp:38`), en plus de celle de la chaîne (`chain_setup.cpp:272`). |
| 43 | `src/app/play/chain.cpp:113-131` | Doublon d'**UX-20**, déjà revérifié par 08 et 43 | `set_language` en direct (`:113`), alors que l'en-tête n'est refait que si le facteur, la réduction ou le HDR changent (`:129-132`). |
| 45 | `src/app/play/chain_setup.h:89-93` | Réfuté | `backend` est le premier membre de `ChainEngines`, déclaré avant `engine`, `drba`, `rife`, `rife_motion` et `rife_motion_calm` : les moteurs sont détruits avant leur backend, et donc avant le `VkContext` qu'ils référencent. `prepare_chain_once` qui échoue en route détruit son `c` local dans le même ordre. En fin de `run_chain`, l'étage (qui garde des pointeurs bruts) part avec `pipeline->reset()` avant `prepared->reset()` (`player_run.cpp:745-748`). Le passage à TensorRT (`handle_engine_change`) n'a lieu qu'après le retour de `run_chain`, quand toute la chaîne est détruite. Seule exception, déjà relevée : `ChainStats` lit encore le backend après sa destruction (A08-02). |

## Points positifs

- L'ordre de destruction de `ChainEngines` est voulu et commenté (`chain_setup.h:64`), et il tient même sur les
  sorties en erreur de `prepare_chain_once`.
- Une erreur de présentation finit proprement : position enregistrée, ordre d'arrêt suivi, pas de boucle.
- `TrtEngine::place` vérifie la forme de chaque tenseur : l'écart de taille d'A08-15 donne un refus net plutôt
  qu'une lecture hors des tampons.
