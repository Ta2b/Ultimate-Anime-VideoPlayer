# Complément de l'audit 13. Session de lecture : seek, épisodes et reprise

Branche : `claude/project-thread-jod31z`. Commit audité : `2b42394`. Date : `2026-11-09`.

## Verdict

Sur les huit renvois reçus, un seul est un défaut nouveau (A13-14 : `--seconds` n'arrête pas la lecture, l'épisode
suivant du dossier s'enchaîne) ; six sont des doublons de défauts déjà dans `SYNTHESE.md`, et un est réfuté
(l'étiquette « ass » d'un `.Srt` est sans effet).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A13-14 | Mineur | `src/app/play/session.cpp:691`, `src/app/play/player_run.cpp:717-725`, `src/app/play/player.cpp:574-584` | `--seconds=S` est documenté « quits after S seconds of playback » (`options.cpp:476`), mais `check_end` ne fait que `finished_ = true`, sans `quit_`. Pour `run_chain`, c'est une fin de fichier comme une autre : `autoplay_next` (vrai par défaut, `settings.h:254`) ouvre `sibling_episode(file, 1)`, et `play()` reboucle sur `next_file` sans regarder `scripted`. Chaque épisode rejoue S secondes, jusqu'au dernier du dossier. | Un banc ou un script lance `ajp play "Ep 01.mkv" --seconds=10 --stats=...` dans un dossier de 12 épisodes : il joue 12 × 10 s, `--stats` mêle douze fichiers, la durée du script est multipliée. C'est la cause produit de A40-05 (le cas e2e « clip de 2 s » joue quatre clips). | Dans `check_end`, la limite atteinte met aussi `quit_ = true` (ou un drapeau `stopped_by_limit_` que `run_chain` traite comme une demande de quitter) ; garder `finished_` pour la vraie fin. Test : deux clips dans un dossier, `--seconds=1`, un seul « épisode suivant » attendu à zéro au journal. | Lu ; le journal de A40-05 montre quatre « épisode suivant » sous `--seconds=2`. |

## Renvois reçus

| De | Où | Conclusion | Note |
|---|---|---|---|
| 14 | `src/app/play/pipeline.cpp:45-49` | Doublon de A13-11 | `pipeline.cpp:44` et `:49` y sont déjà ; classé en dette par 34 (manifeste UTF-8 sous Windows). La ligne 163 (`title` de `tracks_`) aussi. |
| 14 | `src/app/play/pipeline.cpp:65` | Doublon de A13-11 | Ligne déjà citée dans A13-11 (message « no video track »). |
| 14 | `src/app/play/session.cpp:286-287` | Doublon de A13-02 | MED-01 côté session : un refus `recoverable` devient `notice_`, puis `StreamEnded` met `finished_` sans `failure_`, et `player_run.cpp:718` enchaîne l'épisode. La cause côté source est A14-01 et A14-07. |
| 16 | `src/app/play/session.cpp:420-445` | Doublon de A16-01 | Le gel de 0,4 s par seek est décrit dans A16-01 (`kAudioStallRepeatUs`, `session.cpp:25`). Volet session à joindre à sa correction : `ThreadStages::flush` (`wiring.cpp:87-94`) rend la main à l'audio à chaque époque sans savoir si l'audio a produit un seul bloc depuis la perte. Ne la rendre qu'après un `push` réussi (ou un appareil rouvert) supprime le gel même si la sortie reste morte. |
| 17 | `src/app/play/pipeline.cpp:164-170` | Doublon de A17-02 | Lu : le test `none_of(... kind == Subtitle)` compte une piste PGS illisible. Correction de A17-02 : tester `media::subtitle_readable(t.codec)` (ou l'absence de piste du conteneur dans `menu_.subtitles`) plutôt que `kind`. |
| 17 | `src/app/play/pipeline.cpp:183-191` | Doublon de A17-09 | Copies et attente sur le thread de la fenêtre (`player_run.cpp:166`) déjà dans A17-09 et A17-10. Précision pour la correction : la boucle construit `subs::AttachFont{font}` à chaque essai, donc une copie complète de la police (jusqu'à 64 Mo) par milliseconde d'attente, jetée quand `try_push` refuse (`inbox.h:43` ne prend le message qu'en cas de succès). Construire le message une fois et repasser `std::move(message)` à chaque essai suffit, puisqu'un refus ne le déplace pas. |
| 17 | `src/app/play/pipeline.cpp:47` | Réfuté | Le `.codec` des candidats externes ne sert qu'à `subtitle_readable` (`selection.cpp:79-84`), qui accepte « ass » comme « subrip », et le rang ne regarde pas le format. Le chargement choisit le lecteur d'après `lowercase_extension` (`subtitle_track.cpp:26-29`, `:128`), et `find_external_subtitles` filtre déjà par `ascii_lower` (`external_subtitles.cpp:40-41`) : un `.Srt` est trouvé, choisi et lu en SRT. Seul défaut, de forme : comparer `core::ascii_lower(extension)` pour l'étiquette, sans effet visible aujourd'hui. |
| 40 | `tests/unit/app/e2e_play_test.cpp:221` | Confirmé : A13-14 | Non voulu : l'aide promet de quitter. Le test lui-même reste A40-05 (un dossier par cas). |

## Qualité d'image

Aucun des renvois ne touche la chaîne d'image : pas de baisse de qualité faite pour la vitesse à signaler ici.

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 40 | `tests/unit/app/e2e_play_test.cpp:221` | Une fois A13-14 corrigé, le cas « clip de 2 s » ne joue plus que son clip ; A40-05 reste utile pour les cas sans `--seconds`. |
