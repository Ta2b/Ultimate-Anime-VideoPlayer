# Rapport de l'audit 14. Sources, démultiplexage et erreurs d'E/S

Branche : `claude/project-thread-98z28y`. Commit audité : `9f3b9e2`. Date : `2026-11-09`.

## Verdict

La zone est propre et bien testée sur le fichier local sain, mais elle ne sait pas distinguer une vraie fin
d'une panne : une erreur d'E/S, durable ou parfois immédiate, devient une fin d'épisode, et un partage réseau
qui ne répond plus peut figer la lecture puis la fermeture.

## Cadrage retenu

Question posée à Tatoub le 09/11 : ce qui compte le plus dans cette zone. Réponse : **tout à égalité**. Erreurs
d'E/S et partages réseau, choix des pistes et des langues, chapitres, fichiers tronqués ou abîmés ont reçu la même
profondeur. Consigne reçue ensuite : ne plus poser de questions et couvrir tous les cas quand un doute se présente.

Le conteneur n'a ni les dépendances du projet (`~/ajp-deps` absent) ni les en-têtes de FFmpeg : rien n'a été
compilé ni lancé. Les défauts sont lus dans le code, en suivant les appels jusqu'à `wiring.cpp`, `session.cpp`,
`pipeline.cpp` et `player_run.cpp` (zones 08 et 13) là où ils passent la frontière. Le comportement de FFmpeg
cité vient de sa source (`libavformat/avio.c`, `file.c`, `matroskadec.c`). Ce qui dépend d'un vrai partage
réseau ou d'un démultiplexeur précis est marqué « à confirmer », avec la manipulation ; les tests qui
échoueraient sont décrits.

Pour se repérer : `Pipeline::open` (`pipeline.cpp:33`, zone 13) ouvre le `Demuxer` (`demuxer.cpp:243`), choisit
les pistes (`choose_tracks`, `selection.cpp:163`), puis lance `SourceThread` (`source.cpp:183`). La boucle
`SourceLoop::run` (`source_loop.cpp:303`) lit les paquets, décode la vidéo, route l'audio et les sous-titres, et
signale la fin ou un refus par `SourceListener` (`MainMail`, `wiring.cpp:60-71`).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A14-01 | Majeur | `src/media/demuxer.cpp:367` | `AVERROR_EOF` est pris pour une vraie fin sans regarder `pb->error`. Beaucoup de démultiplexeurs (MP4 par `avio_feof`, AVI, MPEG-TS) rendent `AVERROR_EOF` quand l'AVIOContext est en erreur, car `eof_reached` est posé aussi sur une erreur de lecture. La panne saute alors les nouvelles tentatives, n'affiche aucun message, et finit l'épisode comme s'il était fini. | MP4 sur un disque USB ou un partage SMB : une lecture rend EIO au milieu de l'épisode. `mov_read_packet` voit `avio_feof` et rend `AVERROR_EOF`. `read_file` rend `nullopt`, le décodeur se vide, `end_of_stream` part, et l'épisode suivant démarre sans un mot. Pire que MED-01, qui affiche au moins la notice. | Sur `AVERROR_EOF`, regarder `ctx.pb->error` (et `avio_feof` contre `avio_tell`/`avio_size`) : une erreur collée est une erreur d'E/S et passe par les nouvelles tentatives. Signaler aussi une fin « précoce » quand la dernière position lue est loin de `duration_us` (voir MED-01). | Lu dans la source de FFmpeg. À confirmer : MP4 sur une clé USB, débrancher la clé à mi-lecture ; ou test `ReadFault` qui pose `pb->error` puis rend `AVERROR_EOF` |
| A14-02 | Majeur | `src/media/demuxer.cpp:252-254` | `rw_timeout` et le rappel d'interruption ne coupent pas une lecture bloquée sur un partage **monté** (`/mnt/nas`, `Z:\`, `\\NAS\anime`). Ces chemins passent par le protocole `file`, dont `read()` bloque dans le noyau : `rw_timeout` ne joue que sur `EAGAIN` (`retry_transfer_wrapper`), et l'interruption n'est lue qu'entre deux appels. Le commentaire « rend la main au bout de 30 s » est faux pour ce cas, le seul courant. | NAS en veille ou Wi-Fi coupé pendant la lecture. Le thread source reste dans `read()` : délai du client SMB de Windows (environ 60 s par lecture), indéfini sur un montage NFS `hard` ou CIFS sous CachyOS. Puis cinq nouvelles tentatives, chacune aussi bloquée. Pendant ce temps, fermer la fenêtre ou passer à l'épisode suivant détruit `SourceThread`, dont le `jthread` attend (`source.h:119`, `~Pipeline`) : la fenêtre se fige. | Lire par un AVIOContext propre (`avio_alloc_context`) dont la lecture se fait sur un thread d'E/S avec un délai réel, ou au moins détacher l'attente de fermeture (délai borné, puis abandon du thread bloqué, journalisé). Corriger le commentaire. Ne retenter qu'après un délai total borné, pas par lecture. | Lu (`libavformat/avio.c`, `file.c`). À confirmer : lire depuis un partage SMB, couper le réseau du NAS, puis fermer la fenêtre ; mesurer le temps de fermeture sous Windows 11 et sous CachyOS |
| A14-03 | Mineur | `src/media/demuxer.cpp:273-283` | Chapitres ordonnés, éditions et chapitres cachés de Matroska ne sont ni gérés ni détectés. FFmpeg aplatit toutes les éditions, ignore `EditionFlagOrdered`, `ChapterSegmentUID` et `ChapterFlagHidden`, et ne garde un chapitre que s'il commence après le précédent. Le dédoublonnage par début et titre arrive donc après ce filtre et ne sert presque jamais. | Sortie de fansub BD de 2008 à 2014 avec l'OP et l'ED dans des fichiers à part (chapitres ordonnés liés) : l'épisode joue sans générique, sans message. Édition ordonnée interne (récapitulatif à sauter, scènes réordonnées) : le fichier se lit dans l'ordre physique. « Passer le générique » vise alors un chapitre qui n'existe pas dans la lecture. | Lire soi-même l'élément `Chapters` (EBML, quelques centaines de lignes) pour au moins détecter une édition ordonnée ou un segment lié et afficher une notice. Plus tard : une timeline comme mpv (segments liés cherchés dans le dossier). | Lu (`matroskadec.c`, `matroska_read_header`). À confirmer : un MKV avec `mkvmerge --chapters` ordonnés (`EditionFlagOrdered=1`) |
| A14-04 | Mineur | `src/media/source_loop.cpp:337-364` | Un Flush suivi d'un Seek refusé sans déplacer le fichier (cible hors bornes) « reprend où elle était », mais `on_flush` a déjà jeté l'image tenue et la réserve des pistes annexes, et remis `input_ended_` à faux. Le décodeur n'est pas vidé : la lecture repart sous la nouvelle époque avec un trou (image et paquets audio tenus perdus), alors que l'audio a vidé sa file au Flush. | Durée du conteneur estimée trop courte (AVI ou TS sans index, durée tirée du débit) : un seek vers la fin réelle est refusé (`demuxer.cpp:509`). La vidéo saute en avant de ce que tenaient les files et l'audio repart décalé de ce qu'il a perdu. | Au refus sans déplacement, faire comme `recover_position` : vider le décodeur et revenir à `last_pts_us_` par un seek exact. Ou ne pas refuser une cible au-delà d'une durée estimée (`AVFMT_FLAG`/`duration_estimation_method`). | Lu. Test décrit : Flush puis Seek à `duration + 1 s`, vérifier que les images livrées ensuite suivent la dernière livrée sans trou |
| A14-05 | Mineur | `src/media/track_language.cpp:44-46` | La langue tirée du titre ignore le type de piste. « VOSTFR » et « VOSTA » désignent une version originale avec sous-titres : sur une piste **audio** sans balise, elles la classent « fre » ou « eng » alors qu'elle est japonaise. | Pack français à balises vides : audio « VF » et audio « VOSTFR » (le japonais). Tatoub veut l'audio japonais : les deux pistes valent 0 pour `jpn` et le drapeau défaut décide. Un utilisateur qui veut la VF : les deux valent `fre`, égalité, et la première piste (souvent le japonais) peut gagner. | `track_language` prend le type : sur une piste audio, « vostfr », « vosta » et « vo » valent « langue originale » (inconnue, ou `jpn` si un autre indice le dit), jamais `fre` ou `eng`. Ajouter les deux cas à `selection_test`. | Lu |
| A14-06 | Mineur | `src/media/source.h:34-35` | La source lit le fichier dans l'ordre et une file audio pleine (2 s, `packet.h:62`) fait attendre toute la lecture. Un fichier mal entrelacé, avec l'audio plusieurs secondes avant la vidéo, prive la vidéo de paquets alors que l'audio attend son horloge. | MP4 ou AVI remuxés à la main, audio en blocs de 5 s ou plus : après l'amorçage, la file audio se remplit, la source attend un Credit, et la vidéo manque d'images (saccades, mode Rapide qui retombe). | Borner l'attente par la durée lue d'avance sur la piste vidéo : tant que la vidéo n'a pas de paquet en avance, accepter la file annexe au-delà de sa borne (jusqu'au plafond en octets), comme le cache par piste de mpv. | À confirmer : `ffmpeg -i ep.mkv -c copy -interleave_delta 0 -max_interleave_delta 0` puis réentrelacer à 5 s, et lire |
| A14-07 | Mineur | `src/media/demuxer.cpp:369-377` | Une panne d'E/S, une interruption et des données abîmées sortent toutes en `DecodeFailed`. L'application ne peut pas dire « disque ou réseau » ni distinguer une panne d'un fichier abîmé ; `MediaUnreadable` existe et ne sert qu'à l'ouverture. Le crochet de test `ReadFault` et `fail_reads` sont dans l'interface publique de production (`demuxer.h:30-36`, `66-67`). | Le message de la notice dit une erreur de décodage quand le NAS a décroché. Toute la zone 13 qui voudrait traiter MED-01 n'a pas de code sur lequel s'appuyer. | Un code `SourceIo` (ou `MediaUnreadable` avec un drapeau) pour l'E/S, `DecodeFailed` pour les données ; passer `ReadFault` par une option de construction réservée aux tests. | Lu |
| A14-08 | Mineur | `tests/unit/media/source_test.cpp:400-414` | Le test fige MED-01 comme comportement voulu (« finit le flux proprement »). Aucun test d'un seek pendant une panne (MED-06), d'un `AVERROR_EOF` posé sur une erreur (A14-01), de `interrupt()` pendant une lecture ou un seek, des chapitres ordonnés, ni d'une piste audio « VOSTFR ». | Une correction de MED-01 casse ce test, et les régressions sur les autres cas passent sans bruit. | Réécrire ce test pour exiger un signal distinct de la fin ; ajouter les tests décrits dans A14-01, A14-04, A14-05 et MED-06. | Lu |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| MED-01 | Toujours présent | `src/media/demuxer.cpp:373-377` rend un refus `recoverable` après 1,55 s ; `src/media/source_loop.cpp:446-449` le signale puis vide le décodeur et envoie `end_of_stream`. Côté application : `wiring.cpp:67-71` puis `session.cpp:286-287` en font une notice, `session.cpp:697-700` finit la session et l'épisode suivant démarre ; `player_run.cpp:589-591` affiche « Playback goes on after an error ». Même chose après 64 blocs abîmés de suite (`kDataRetries`). A14-01 l'aggrave (fin sans notice). Correction : un signal `ended_on_error` (ou un `StreamEnded` qui porte la cause) que la session traite en pause avec message et proposition de réessayer, sans enchaîner l'épisode. |
| MED-06 | Toujours présent (côté source) | `src/media/source_loop.cpp:362` puis `371-383` : si le seek a déplacé le fichier et que le retour à `last_pts_us_` échoue aussi (panne toujours là), `idle_` reste vrai sans `end_of_stream` ni nouvel essai. La session attend une image qui ne vient jamais : lecture figée jusqu'au seek suivant, même quand le partage revient. Le rognage audio relève de la zone 16 (renvoi). Correction : réessayer `recover_position` sur un délai, puis finir par le signal d'erreur de MED-01. |
| MED-10 | Toujours présent | `src/media/demuxer.cpp:193` ne lit que `AVStream::duration`, que matroskadec ne remplit pas : il pose l'étiquette `DURATION` (« 00:23:40.123000000 ») de mkvmerge dans les métadonnées de la piste. `TrackInfo::duration_us` vaut donc 0 en MKV et `pipeline.cpp:343` éteint P3.75. Correction : à défaut de `duration`, lire l'étiquette `DURATION` (ou `DURATION-eng`). |
| MED-14 | Toujours présent (fichier hostile seulement) | `src/media/av_util.cpp:111-114` : `av_rescale_q` rend `INT64_MIN` sur un débordement, rendu tel quel (négatif, au lieu de 0). `demuxer.cpp:199-200` : `chain_dimension(1)` (`core/frame.h:143`) vaut 0, une piste de largeur 1 sort en 0. Aucun effet vu sur un fichier sain ; borner le résultat comme `to_us` et refuser une piste vidéo de taille nulle à l'ouverture. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 13 | `src/app/play/pipeline.cpp:45-49` | `extension().string()` et `filename().string()` sur les sous-titres externes : sous Windows, un nom japonais hors page de code lève une exception à l'ouverture de l'épisode (règle `path::string()`) |
| 13 | `src/app/play/pipeline.cpp:65` | `config.file.string()` dans le refus « no video track » |
| 13 | `src/app/play/session.cpp:286-287` | Réponse au renvoi 08 → 14 (MED-01) : la source ne distingue pas encore fin et erreur (A14-01, A14-07) ; la session doit aussi refuser d'enchaîner l'épisode après une fin précédée d'un refus d'E/S |
| 18 | `src/app/play/player.cpp:99` | `file.string()` dans le refus de `probe_once` (fichier sans vidéo, nom japonais sous Windows) |
| 18 | `src/app/play/pipeline.h:76` | `~Pipeline` joint le thread source, qui peut être bloqué dans `read()` sur un partage muet (A14-02) : quel thread détruit la `Pipeline`, et la fenêtre se fige-t-elle ? |
| 16 | `src/audio/` (rognage après seek) | MED-06 : le rognage audio peut viser la cible d'un seek que la source a refusé |
| 30 | `src/media/source_loop.cpp:303-305` | Pendant `wait_before_retry` (jusqu'à 1,55 s par série) la boîte n'est pas lue : un seek ou un `cmd::Quit` attendent ; seul l'arrêt du `jthread` coupe l'attente |

## Points positifs

- Chemins passés à FFmpeg en UTF-8 (`demuxer.cpp:148-150`), y compris sous Windows ; test d'un chemin non ASCII.
- Nouvelles tentatives avec attente qui double, découpée en tranches de 10 ms pour ne pas retarder une fermeture,
  et erreur collée de l'AVIOContext effacée avant de réessayer.
- Seek exact soigné : recul sous un point clé affiché après la cible (MP4 à images B, GOP ouvert), lecture
  d'avance rendue dans l'ordre, retour à la dernière image sûre quand un seek échoue après avoir bougé.
- Réserve d'amorçage par piste, plafonnée en octets, avec un refus nommé plutôt qu'un blocage ; contrat des
  consommateurs écrit en tête de `packet.h`.
- Choix des pistes repris de mpv et étendu à l'usage anime : langue tirée du titre, pistes « Signs & Songs »
  écartées, mots japonais et français, préréglages audio et sous-titres ensemble.
- `to_us` sature au lieu de déborder (P2.29). Bon jeu de tests de source avec `ReadFault` (panne passagère,
  durable, bloc abîmé, fichier tronqué).

## Questions ouvertes

- Faut-il viser la vraie prise en charge des chapitres ordonnés (segments liés) pour 1.0, ou seulement la notice ?
  Le rapport recommande la notice d'abord (A14-03).
- Délai acceptable avant d'abandonner un partage muet : le rapport propose 10 à 15 s au total, puis la pause avec
  message de MED-01.
