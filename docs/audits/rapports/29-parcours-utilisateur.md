# Rapport de l'audit 29. Parcours utilisateur de bout en bout

Branche : `claude/project-thread-2govbf`. Commit audité : `401bfa7`. Date : `2026-11-09`.

## Verdict

Les parcours courants tiennent, mais ils cassent dès qu'un moteur refuse ou qu'un événement arrive au mauvais
moment. Sept défauts sont nouveaux, dont deux majeurs. Un refus de RIFE ou de SPAN arrête l'épisode au lieu de le
jouer en mode dégradé. La mise en veille n'est traitée nulle part. Les sept défauts connus de la fiche sont
toujours présents. A13-09 est réfuté.

## Cadrage retenu

Comme le coordinateur l'a demandé, aucune question n'a été posée à Tatoub. Le cadrage suit sa consigne du
09/11 : toute baisse de qualité faite pour la vitesse est un défaut, sauf si une mesure la justifie. RIFE 4.25,
RIFE 4.22 lite et DistilDRBA comptent autant, et le VRR est hors sujet. La cible est une RTX sous TensorRT, sous
Windows 11 et sous CachyOS. Faute de GPU NVIDIA dans le fil, tout ce qui demande une mesure est marqué « à
confirmer ». Les dépendances ne sont pas construites ici : les tests sont décrits, pas écrits.

### Parcours suivis

| Parcours | Chemin d'appels (thread principal sauf mention) | Verdict |
|---|---|---|
| Double-clic, premier lancement | `main.cpp:357-366` (instance) → `:369-403` (mise à jour) → `player.cpp:150` (sonde, thread `probe`) → `:186` (thread `updates`) → `:218` (fenêtre) → `:289` (assistant) → `:305` (moteurs périmés) → boucle `:346` | La vidéo attend l'assistant ; un second fichier reçu entre-temps est jeté (UX-03) |
| Ouverture, moteurs prêts | `player.cpp:441-452` (`open_backend`, pompé) → `player_run.cpp:111` `prepare_chain` → `player_engines.cpp:129` (thread `preparer`, attente pompée) → `:146` `backend.load` (non pompé) | A29-06 |
| Ouverture, moteur à construire | idem, `wait_engine` `player_engines.cpp:40-102` ; fermeture : `EngineBuild::~EngineBuild` arrête `trtexec` (`process_*.cpp`, `stop`) | Fermeture propre ; refus final : A29-01 |
| « Annuler et regarder » | `player_engines.cpp:48-53` → replis (A08-01) → `player.cpp:525-534` | TRT-02, A08-01, A18-04 |
| Seek, image par image, pause | `chain.cpp:283-293` → `session.cpp:189-227` → `source_loop.cpp` | Rien de nouveau (A13-05 à A13-08, A14-04) |
| Réglage en direct, réglage qui rouvre | `chain.cpp:378-391` → `apply_live` `:110` ou `to.reopen` → `player.cpp:536-586` | UX-01 ; échec de réouverture : A29-01 ; calibration : A29-07 |
| Pistes, délais | `chain.cpp:69-107`, `:300-316` | Pistes perdues sur réouverture par `PlayFile` : A29-03 |
| Fin d'épisode, suivant absent | `player_run.cpp:718-726` → `player.cpp:574-583`, `:612-617` | MED-01 ; sans suivant, fermeture (double-clic) ou accueil |
| Écran, plein écran, réduction, HDR | `player_run.cpp:532-635` ; HDR : `:548-561` en direct ; réduction : `chain_hdr.cpp:309` | GPU-03, A29-03 ; HDR et réduction corrects |
| Veille | aucun code | A29-02 |
| Fermeture pendant construction, calibration, téléchargement, mise à jour | `wait_engine`, `engine_screen.cpp:155-160`, `engine_downloads.cpp:67-71`, `player.cpp:186` (`jthread`, `stop`) | Fermeture propre partout |
| Mise à jour, relancement | `main.cpp:369-403` → `relaunch` `:113` ; moteurs : `player.cpp:306`, `:421`, `engine_screen.cpp:374-388` → `main.cpp:528` | A29-04 |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A29-01 | Majeur | `src/app/play/chain_setup.cpp:497`, `:516`, `:355` ; `src/app/play/player.cpp:589-612` | Un refus de RIFE ou de SPAN (construction, chargement, essai au chargement) arrête tout l'épisode. DistilDRBA a un repli vers RIFE (`chain_setup.cpp:400-406`), pas RIFE. RIFE n'a de repli ni vers « sans interpolation », ni vers RIFE entier quand le refus vient du moteur mouvement (`:497`). SPAN n'a pas de repli vers « sans SPAN ». Si le refus arrive à une réouverture demandée depuis le panneau, la lecture en cours est perdue : `player.cpp:609-612` renvoie à l'accueil, même après un double-clic. | 4070 Ti, source 4K, Personnalisé avec RIFE 4.25 sans réduction. `trtexec` échoue à la taille 4K (mémoire, délai de 30 min, A05-03). La vidéo ne joue pas : l'accueil s'ouvre avec « stopped ». Même chose en lecture : RIFE 4.25 choisi dans le panneau puis « Appliquer », la construction échoue, l'épisode se ferme et l'accueil s'ouvre. | Pour l'interpolation, faire comme DistilDRBA : un refus garde la chaîne sans interpolation, avec un avis visible (`rife_factor = 0` pour la séance, sans l'écrire). Pour SPAN, garder la source à sa taille (sans réduction : voir TRT-02). Pour une réouverture, revenir aux réglages d'avant au même endroit, avec l'avis. | Lu ; à confirmer : `--model` vers un ONNX tronqué, ou `trtexec` remplacé par un script qui échoue, puis ouvrir un épisode |
| A29-02 | Majeur | aucun fichier : `WM_POWERBROADCAST`, `PrepareForSleep` et `SDL_EVENT_*` de mise en arrière-plan sont absents de `src/` | La mise en veille n'est pas traitée. La lecture ne se met pas en pause avant la veille. Au réveil, rien ne vérifie le backend ni ne recale les horloges. Sous CachyOS, `steady_clock` (CLOCK_MONOTONIC) s'arrête pendant la veille. Sous Windows, QPC continue peut-être : l'horloge maître extrapolée (`audio/player.cpp:270-294`) ferait alors un saut. Sous Linux, sans `NVreg_PreserveVideoMemoryAllocations`, la mémoire de CUDA est perdue au réveil. `health()` ne reconnaît pas l'erreur 999 qui suit (A01-01), et l'image reste figée ou abîmée. | Portable sous CachyOS, épisode en lecture, couvercle fermé 10 min puis rouvert : image figée ou corrompue sans message, son peut-être reparti seul. Sous Windows, veille puis réveil : les images sont jetées en masse pendant le recalage, la sortie WASAPI est invalidée (A16-01). | Pause sur `PBT_APMSUSPEND` (Windows) et sur le signal logind `PrepareForSleep(true)` (Linux, D-Bus). Au réveil : contrôle de `health()`, puis seek exact à la position courante (flush, horloges recalées), ou réouverture de la chaîne si l'appareil est perdu. | À confirmer : lecture TensorRT sous CachyOS puis `systemctl suspend`, et Windows 11 en veille S3 ou S0ix ; relire `ajp.log` (erreurs CUDA, images jetées, `missing`) |
| A29-03 | Mineur | `src/app/play/player_run.cpp:617`, `:715` ; `src/app/play/player.cpp:574-583`, `:454` | Deux réouvertures du même fichier passent par `PlayFile` et donc par `next_file` : le changement d'écran (GPU-03) et le fichier qui grandit. Elles recréent tout le `Player`, et perdent ce qui ne vit que dans le `Player` : pistes audio et sous-titres choisies, sous-titres externes glissés (`chain.cpp:232`), tâche SPAN FP8 en cours (`span_job`), compteur de redimensionnements. | Épisode en VO, piste anglaise et sous-titres `.ass` glissés sur la fenêtre. La fenêtre passe de l'écran 1440p à la TV 4K : la chaîne rouvre au même instant, mais avec la piste par défaut et sans le `.ass`. | Pour le même fichier, rouvrir la chaîne (`to.reopen` avec la position) au lieu de `PlayFile`. Ou porter `audio_track`, `subtitle_track`, les sous-titres externes et `span_job` d'un `Player` au suivant quand `next_file == file`. | Lu |
| A29-04 | Mineur | `src/app/main.cpp:528` ; `src/app/play/player.cpp:421-425`, `:440-443` ; `src/app/play/engine_screen.cpp:374-388` | Après une mise à jour d'un moteur en cours de séance, `relaunch` rejoue `argv` : le fichier du lancement, ou l'accueil. Ce n'est pas la vidéo en cours. Le commentaire de `player.cpp:419` promet « le lecteur se relance sur cette vidéo ». | Double-clic sur l'épisode 1. Les épisodes s'enchaînent jusqu'au 4, puis Paramètres > Moteurs > mise à jour de TensorRT. Le lecteur se relance sur l'épisode 1, depuis le début puisque sa fin a été vue (P3.83). Lancé depuis l'accueil, il rouvre l'accueil. | Passer à `relaunch` le fichier en cours et sa position (`--start`), ou les écrire dans un fichier de reprise lu au lancement suivant. | Lu |
| A29-05 | Mineur | `src/app/play/settings_panel.cpp:572-574`, `src/app/play/auto_config.cpp:220`, `src/app/play/auto_config.h:22` | DistilDRBA ne se choisit qu'à la main. Ni l'Auto (`kDistilDrbaInAuto = false`) ni la calibration (A25-08) ne le proposent. Le choisir fait passer le préréglage à Personnalisé, car `preset_chain_differs` compte `motion_engine`. L'utilisateur perd alors aussi ce que l'Auto adapte à chaque vidéo et à chaque écran : hauteur de travail selon la source, plafond du facteur selon l'écran, réouverture au changement d'écran (`display_moves_chain` exige un préréglage). C'est un frein au moteur censé être le plus rapide. | Auto sur une TV 4K à 120 Hz, épisode 1080p. Tatoub choisit DistilDRBA : la chaîne devient Personnalisé. L'épisode suivant, une source 720p, garde la hauteur de travail de l'épisode 1080p. La fenêtre passée sur l'écran 1440p ne refait plus la chaîne. | Faire du moteur d'interpolation un choix qui se combine à l'Auto (comme la priorité), au lieu d'une sortie de l'Auto. Ou lever `kDistilDrbaInAuto` une fois sa porte de qualité passée (A03-13, A03-14, A03-19 d'abord). | Lu |
| A29-06 | Mineur | `src/app/play/player_engines.cpp:146` | `backend.load` désérialise chaque moteur TensorRT sur le thread de la fenêtre, sans pomper les événements (au contraire d'`open_backend`, `engine_screen.cpp:339`). L'ouverture avec des moteurs prêts enchaîne : SPAN, RIFE fusionné (six moteurs) avec son essai au chargement, la variante calme. | Windows 11, RTX, moteurs en cache : pendant « Loading the engines », la fenêtre peut passer « Ne répond pas » si le total dépasse 5 s sans message traité. | Charger sur un thread avec `run_pumped`, comme `open_backend`. | À confirmer : chronométrer l'ouverture en cache sous Windows (journal : « moteur … » puis l'image) et regarder le titre de la fenêtre |
| A29-07 | Mineur | `src/app/play/engine_screen.cpp:164-167`, `:186-191` ; `src/app/play/player.cpp:560-585` | En lecture, « Refaire le banc » puis « Annuler et regarder », ou une calibration qui échoue, rend `as_auto(base)` : le préréglage dit Auto, mais la chaîne reste celle d'avant. La boucle la rouvre telle quelle, sans `apply_preset`, et l'écrit dans `play.toml` (`player.unsaved`). C'est la même cause qu'A23-03 : une réouverture dans la même vidéo ne refait jamais le préréglage. | Personnalisé avec RIFE 4.25 ×4, « Refaire le banc », puis « Annuler et regarder » : le panneau montre Auto, mais RIFE 4.25 ×4 reste en service jusqu'à la vidéo suivante, où l'Auto change tout. | Après la calibration, repasser `settings` par `apply_preset`, comme `player.cpp:405`. Ou rendre `base` tel quel sur une annulation. | Lu |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-03 | Toujours présent | `player_run.cpp:612-617` et `auto_config.cpp:223-225` : toute hauteur d'écran différente rouvre, sans hystérésis ni délai. Le passage par `PlayFile` perd en plus les pistes (A29-03), et l'Auto refait la chaîne pour le nouvel écran (`player.cpp:405-414`), ce qui peut lancer une construction. |
| MED-01 | Toujours présent | `demuxer.cpp:283-293` rend un refus `recoverable` ; `source_loop.cpp:182-188` le change en fin de flux ; `player_run.cpp:718-726` lance l'épisode suivant. La notice « Playback goes on after an error » (`player_run.cpp:589-591`) dit le contraire. Voir A13-02 et A14-01. |
| TRT-01 | Toujours présent | `auto_config.cpp:129` monte Économe à Léger dès qu'une calibration existe. Si elle n'a pas vu tenir SPAN, `:195` coupe `upscale`, mais `work_height` reste celui de `set_span` (`:151`). `chain_setup.cpp:238` réduit donc la source sans SPAN. RIFE 4.22 lite (`:204`) tourne à cette taille, pas à celle mesurée. |
| TRT-02 | Toujours présent | `player.cpp:525-534` coupe SPAN et l'interpolation, et `chain_setup.cpp:238` réduit quand même à `work_height`. De plus, quand seul RIFE se construisait (SPAN déjà en cache), SPAN est jeté sans raison : c'est une baisse de qualité sans aucun gain de temps. Le clic n'arrive d'ailleurs souvent pas jusque-là (A08-01). |
| UX-01 | Toujours présent | `live_part` et `change_between` (`settings.cpp:383-499`) ne nomment ni `restore_lines` ni `gpu`. Voir A22-01 et A23-01. |
| UX-02 | Toujours présent | `player.cpp:546-585` : après `handle_engine_change`, `settings = (*outcome)->settings` reprend l'ancienne chaîne, qui est écrite par `player.unsaved`. Voir A25-01. |
| UX-03 | Toujours présent, plus large que la fiche | Six boucles jettent le `SDL_EVENT_DROP_FILE` qu'un second lancement pousse (`single_instance.cpp:49-66`) ou qu'un glisser-déposer produit : assistant (`engine_screen.cpp:68`), calibration (`:155`), téléchargement (`engine_downloads.cpp:67`), export (`export_run.cpp:99`), **construction d'un moteur** (`player_engines.cpp:55`) et **préparation SPAN FP8** (`:171`). La construction est le cas le plus fréquent : un premier épisode à 1080p, c'est environ 17 min (A25-03). L'accusé « ok » part avant la remise à SDL (`single_instance.cpp:161`, A19-02), donc le second lancement sort avec le code 0 sans rien dire. Correction : une file `pending_drops` dans `Player`, remplie par ces boucles et relue par `run_chain` comme `take_drop`, ou les événements remis dans la file de SDL (`SDL_PushEvent`) à la sortie de la boucle. |
| Renvoi 24 → 29 (`player_engines.cpp:54-57`) | Confirmé | Voir UX-03 juste au-dessus. A08-08 ne couvrait que cette boucle. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 13 | `src/app/play/player.cpp:454`, `:574-583` | A13-09 est réfuté : `Player` est recréé à chaque `next_file` (vidéo glissée, épisode voisin ou suivant). `p.audio_track` et `p.subtitle_track` (`player_run.cpp:738-739`) meurent donc avec lui, et une autre vidéo repart du choix par langue. Pour la même raison, la partie « à l'épisode suivant » d'A13-01 ne peut pas arriver ; la réouverture de la même vidéo reste à vérifier. |
| 31 | `src/app/play/chain_setup.cpp:497`, `:516` | Le refus de RIFE n'a pas de repli (A29-01) : la grille des états dégradés doit le compter. |
| 31 | `src/app/play/player_engines.cpp:270-272` | Un refus `Cancelled` de DistilDRBA fusionné devient `fused->declined` et lance la construction de DistilDRBA entier (cas d'A08-01, à vérifier dans la grille). |
| 33 | aucun fichier | La veille (A29-02) est un cas de longue durée, à inclure dans le banc de soirée de l'audit 33. |
| 34 | `src/app/play/default_player.cpp:203`, `:222` | Plusieurs épisodes sélectionnés puis « Ouvrir » lancent un processus par fichier, des deux côtés. Chacun est transmis à la fenêtre déjà là, et `take_drop` garde le dernier arrivé (`chain.cpp:229-231`), dans un ordre non défini. |
| 36 | `src/app/play/player.cpp:574-581` | Le `save_settings` de l'épisode suivant écrit `chain.running` en entier, raccourcis de séance compris (A08-04), avant que la chaîne suivante ne s'ouvre. Si elle échoue (A29-01), `play.toml` garde quand même cette chaîne. |

## Points positifs

- Fermer pendant une construction, une calibration, un téléchargement ou une vérification de mise à jour est
  propre. Chaque attente rend la main sur `SDL_EVENT_QUIT`, son thread est arrêté par un `stop_token`, et
  `trtexec` ou `curl` sont tués (`process_posix.cpp:196`, `process_win.cpp:172`).
- La sonde du fichier est bornée à 20 s et pompée (`player.cpp:121-125`). Le chargement de TensorRT aussi
  (`engine_screen.cpp:339`).
- Le HDR du système activé ou coupé passe en direct, sans réouverture (`player_run.cpp:548-561` puis
  `apply_live`) ; la recomposition se fait aussi en pause (`chain.cpp:194-196`).
- Une fenêtre réduite ou cachée arrête le travail du GPU (`chain_hdr.cpp:309`, `upscale_stage.cpp:304`).
- La position est retenue toutes les `kResumeEveryUs` et à la fin, par un écrivain à part, vidé avant la vidéo
  suivante (`player_run.cpp:648-651`, `:674`).
- Un réglage qui rouvre la chaîne n'est écrit qu'une fois la chaîne rouverte (`player.unsaved`,
  `player_run.cpp:334-340`), sauf dans le cas de l'épisode suivant (renvoi 36).

## Questions ouvertes

- A29-01 : quand RIFE ou SPAN ne se construisent pas, Tatoub préfère-t-il la vidéo sans ce moteur, avec un avis,
  ou un arrêt avec la raison ? Option retenue ici, la plus probable : la vidéo continue, avec l'avis.
- A29-02 : faut-il mettre la lecture en pause à la mise en veille (comportement retenu ici), ou seulement la
  recaler au réveil ?
- A29-05 : faut-il que DistilDRBA puisse se choisir sans quitter l'Auto ? Cela dépend des gains attendus aux
  audits 03 et 49.
