# Rapport de l'audit 13. Session de lecture : seek, épisodes et reprise

Branche : `claude/project-thread-qenph4`. Commit audité : `90512c0`. Date : `2026-11-09`.

## Verdict

La session est solide sur ses chemins testés (amorce, seek rejoué, choix par l'horloge, fin de flux), mais une piste
de sous-titres externe gardée d'une ouverture à l'autre fait planter le lecteur (A13-01), une erreur d'E/S enchaîne
encore l'épisode suivant (MED-01), et l'ordre des épisodes se trompe sur deux formes de noms courantes en anime.

## Cadrage retenu

Réponses de Tatoub :

- **Reprise de position** : il s'en sert, elle doit être fiable (priorité haute).
- **Épisode suivant automatique** : secondaire.
- **Seek** : l'usage réel est le saut de ±5 s et la barre ; le pas d'image est secondaire, le VFR (MED-16) reste mineur.
- **Statistiques** : auditées pour leur exactitude (diagnostic, mode Auto), sans plus.
- Le VRR ne compte pas (GPU-19 revérifié, laissé mineur).
- Après le cadrage, Tatoub ne veut plus de questions : chaque choix ouvert est couvert dans les deux sens.

Aucune carte NVIDIA ni dépendance (FFmpeg, libplacebo, SDL) dans le conteneur : la suite Catch2 n'a pas pu être
construite. `episodes.cpp` et `resume.cpp`, sans dépendance externe, ont été compilés seuls (g++ 13, C++23) avec un
petit programme de test ; les autres tests sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A13-01 | Bloquant | `src/app/play/pipeline.cpp:60` et `:172` | Un index de sous-titres externes (≥ `kExternalSubtitleBase`) passé par `PipelineConfig::subtitle_track` est accepté sans vérifier qu'il existe ; absent du menu, il mène à `subtitle_track(*demuxer->track(index), 0)` : `track()` rend `nullptr` pour un index externe, déréférencé. | 1) Glisser un `.ass` d'un autre dossier sur la fenêtre (P3.85, index `base + n`), puis tout changement qui rouvre la chaîne (réglage, taille de source) : `find_external_subtitles` ne retrouve que les fichiers voisins, l'index manque, plantage. 2) Épisode 1 avec `Ep01.fr.ass` choisi, épisode 2 sans sous-titres externes : `player_run.cpp:739` garde l'index pour la vidéo suivante, plantage au passage d'épisode. | Ligne 60 : n'accepter un index externe que s'il est `< base + externals.size()`. Ligne 172 : ne jamais déréférencer `track()` sans test (un index externe absent est simplement oublié, journalisé). | Lu (chemin complet : `chain.cpp:235`, `player_run.cpp:162-163`, `:738-739`, `demuxer.cpp:222-225`). Test à écrire dans `pipeline_test.cpp` : `PipelineConfig{.subtitle_track = std::optional<int>(kExternalSubtitleBase + 5)}` sur un fichier sans sous-titres voisins ; aujourd'hui segfault, attendu : ouverture sans sous-titres. |
| A13-02 | Majeur | `src/app/play/session.cpp:66-68`, `:277-287`, `:697-700` | MED-01 toujours présent : la session ne distingue pas une fin de flux après abandon de lecture d'une vraie fin. Le refus `recoverable` de la source devient un simple `notice_`, puis `StreamEnded` met `finished_` sans `failure_`. | NAS débranché à la 10ᵉ minute : `Demuxer::read_file` abandonne après ses essais (refus `recoverable`, `demuxer.cpp:283-293`), `source_loop.cpp:182-188` vide le décodeur et envoie la fin de flux ; `player_run.cpp:718-723` lance l'épisode suivant (lui aussi illisible). La position est retenue, mais l'utilisateur ne sait pas ce qui s'est passé. | Retenir dans la session un `read_lost_` quand un refus `source` `DecodeFailed` recoverable arrive dans l'époque courante, et l'exposer (`ended_on_error()`), ou en faire `failure_` ; `player_run` n'enchaîne pas et dit l'erreur. Épisode suivant secondaire pour Tatoub, mais le message manque aussi. | Lu. Test à écrire dans `session_refusal_test.cpp` : refus `{source, DecodeFailed, recoverable}` puis `StreamEnded` de la même époque ; attendu `failure()` ou `ended_on_error()` vrai. |
| A13-03 | Mineur | `src/app/play/episodes.cpp:43-47` | `extra()` classe en bonus tout nom qui contient le mot `MENU`, `PREVIEW`, `TRAILER`, `PV` ou `CM`, même un titre d'épisode. Un tel épisode sort de la liste : il est sauté, et une fois ouvert, « suivant » passe aux bonus. | `Soma - 04 - Le dessert.mkv`, `Soma - 05 - The Menu.mkv`, `Soma - 06 - Le duel.mkv` : l'épisode suivant du 04 est le 06. | Ne tenir un nom pour bonus que si le mot est en tête ou suit le groupe/la série sans numéro d'épisode devant (NCOP, NCED gardés tels quels), ou exiger qu'aucun nombre d'épisode ne précède le mot. | Test compilé, échoue : `sibling_episode(".../Soma - 04 - Le dessert.mkv", 1)` rend `Soma - 06 - Le duel.mkv`. |
| A13-04 | Mineur | `src/app/play/episodes.cpp:74-93` | `natural_compare` ne lit comme nombres que les chiffres ASCII : les chiffres pleine chasse (`１２`) se comparent octet par octet. | Enregistrements japonais `第１話`, `第２話`, `第１０話` : après le 1, le lecteur ouvre le 10. | Décoder les chiffres U+FF10 à U+FF19 (3 octets UTF-8 `EF BC 90..99`) comme des chiffres dans `natural_compare` et `without_version`. | Test compilé, échoue : `sibling_episode(".../第１話.mkv", 1)` rend `第１０話.mkv`. |
| A13-05 | Mineur | `src/app/play/session.cpp:102-120`, `:229-237` | Pas d'image vers l'avant bloqué : la cible est `pts + 1,5 période moyenne`. Quand l'image montrée dure plus de 1,5 période (VFR, ou cadence annoncée plus haute que la réelle) et que la suivante n'est pas dans la file à `+1,25 période`, le seek exact retombe dans l'image montrée ; la touche suivante repart de la même image. Aggrave MED-16. | MKV VFR dont `avg_frame_rate` dit 59,94 alors que la scène est à 23,976 : chaque pression vise `pts + 25 ms`, dans l'image de 41,7 ms ; rien ne bouge, indéfiniment. | Avancer au moins jusqu'à la fin de l'image montrée : viser `max(pts + 1,5 p, fin de l'image montrée + 1)` (la durée est connue de la source), ou faire chercher par la source « l'image suivante » plutôt qu'un temps. | Lu. Test à écrire dans `session_test.cpp` : `source_period_us = 16'683`, image montrée à 1 000 000, `StepFrame{+1}` deux fois sans image en file ; attendu deux cibles distinctes au-delà de la fin de l'image. À confirmer sur un VFR réel. |
| A13-06 | Mineur | `src/app/play/session.cpp:78`, `:83`, `:195`, `:365` | Les retours de `stages_.play()` et `stages_.pause()` sont ignorés. Si la boîte audio est pleine, le `Pause` n'atteint pas l'audio : en audio maître, l'horloge court, la session (pause) continue de choisir des images, la barre dit « pause ». Un `Play` perdu fige la vidéo puis déclenche à tort « audio manquant » au bout de 1,5 s. | Boîte de commandes audio pleine (rafale de volume à la molette + pause). | Rejouer la commande au tour suivant comme `follow_display` le fait pour `set_speed`, ou réserver une place aux commandes d'horloge. | Lu ; probabilité faible, à confirmer en remplissant la boîte dans `session_test` (`Rig.stages` refusant `pause`). |
| A13-07 | Mineur | `src/app/play/session.cpp:229-247`, `:78` | Après des pas vers l'avant pris dans la file (P3.42), aucun seek n'a lieu : l'audio reste à la position de la pause. À la reprise, `Play` repart sans position pour l'audio ; en audio maître la vidéo attend que l'audio rattrape, figée le temps des pas. | 10 pas avant puis Espace : environ 0,4 s d'image figée avec le son. | À la reprise après un pas pris dans la file, faire un seek exact à `current_->pts_us` (comme un pas arrière). | Lu ; à confirmer sur la machine : `p`, `.` dix fois, Espace, observer. |
| A13-08 | Mineur | `src/app/play/session.cpp:189-227` | Un seek refusé par la source sans repositionnement sûr (`recover_position` échoue, `source_loop.cpp:110-123`) laisse la source inactive : la session reste en amorce sans fin ni délai, la barre montre la cible, et le message dit « la lecture continue après une erreur ». Volet session de MED-06. | Seek pendant une coupure du NAS : image figée, aucun nouvel essai quand le partage revient. | Dans la session : un délai d'amorce (quelques secondes) au-delà duquel le seek est rejoué, ou un message vrai (« lecture interrompue »). | Lu ; à confirmer : débrancher le partage, flèche droite, rebrancher. |
| A13-09 | Mineur | `src/app/play/player_run.cpp:738-739` (appelant), `src/app/play/pipeline.cpp:56-62` | Les pistes audio et sous-titres choisies sont gardées par `Player` pour toute vidéo suivante, pas seulement pour une réouverture de la même : le choix par langue est court-circuité dès que l'index existe. | Épisode 1 : piste audio 2 (anglais) choisie. Fichier d'une autre série glissé : sa piste 2 (commentaire) est prise au lieu du japonais. | Ne garder les pistes que si `next_file` est vide, ou les retrouver par langue et titre plutôt que par index. Utile entre épisodes d'une série : à trancher avec Tatoub. | Lu. |
| A13-10 | Mineur | `src/app/play/resume.cpp:107-129` | `remember_position` relit et réécrit tout `resume.tsv` sans verrou : deux instances d'ajp (deux fenêtres) perdent la position de l'une. | Deux épisodes ouverts dans deux fenêtres, fermés à quelques secondes d'écart : l'un retrouve sa position, l'autre non (selon l'ordre des écritures toutes les 15 s, la perte est aléatoire). | Fusionner à l'écriture (relire juste avant de renommer, sous un verrou de fichier), ou un fichier par vidéo. | Lu. |
| A13-11 | Mineur | `src/app/play/resume.cpp:33`, `src/app/play/pipeline.cpp:44`, `:49`, `:65`, `:163` | `path::string()` dans la zone (identification de la reprise, titres des sous-titres externes, message d'erreur), contre la règle de `CLAUDE.md`. Sous Windows, sans danger tant que le manifeste UTF-8 (`cmake/utf8.manifest`) s'applique ; ajp lancé autrement (tests, outils) lèverait une exception sur un nom japonais. | Test unitaire Windows de la reprise sur `進撃の巨人 01.mkv` hors manifeste. | `core::to_utf8`. | Lu. |
| A13-12 | Mineur | `src/app/play/resume.cpp:39-45` | Sous Windows, `same_path` compare sans casse en ASCII seulement : `Épisode` et `épisode` sont deux fichiers pour la reprise. | Dossier renommé avec une autre casse accentuée : la reprise est perdue (le repli par nom et taille compare aussi en ASCII). | Comparer par `CompareStringOrdinal(..., TRUE)` sous Windows. | Lu. |
| A13-13 | Mineur | `tests/unit/app/resume_test.cpp`, `episodes_test.cpp`, `session_refusal_test.cpp` | Rien ne teste la borne de 500 entrées, un chemin non ASCII, le pas d'image en VFR, la fin de flux après erreur de lecture, ni les noms d'épisodes avec un mot de bonus. | — | Ajouter les tests décrits en A13-01 à A13-05, et les deux tests compilés ici. | La borne de 500 entrées a été vérifiée par le test compilé (elle tient). |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-19 | Toujours présent | `src/app/play/session.cpp:665-678` : `follow_display` cale la vitesse dès que `display_sync`, sans regarder le VRR (`config_.vrr_floor_us > 0` quand il est armé). 144 Hz, 47,95 i/s : 3 vsyncs, vitesse 1,0011. Correction : vitesse 1 quand le VRR est armé. Mineur ; le VRR est hors des priorités de Tatoub. |
| MED-10 | Toujours présent | `src/app/play/pipeline.cpp:343-346` lit `TrackInfo::duration_us`, tiré de `AVStream::duration` (`demuxer.cpp:110`) que matroskadec ne remplit pas : `audio_end_us = 0` en MKV, le gel de 1,5 s revient. Correction : estimer la fin audio par le dernier paquet audio vu, ou lire le tag `DURATION`. |
| MED-15 | Toujours présent | `Pipeline::open` et `find_external_subtitles` (appelés par `player_run`), `sibling_episode` (`player_run.cpp:720`, `chain.cpp:339`, qui ouvre aussi chaque candidat par `readable`) tournent sur le thread de la fenêtre. S'y ajoute `GrowingFile::grew` (`player_run.cpp:713`, `episodes.cpp:157-170`) qui dort jusqu'à 3 s sur ce thread en fin de fichier écrit il y a moins de 10 s. |
| MED-16 | Toujours présent | `src/app/play/session.cpp:108-119` et `:231-236` : la période moyenne de la source sert au pas et au repérage dans la file. Cas aggravé : A13-05. |
| UX-10 | Toujours présent | `src/app/play/frame_stats.cpp:93`, `:99` et `src/app/play/session.cpp:732` : la virgule remplace le point quelle que soit la langue. |
| MED-01 (renvoi 08) | Toujours présent | Voir A13-02. |
| MED-06 | En partie présent | Le seek rejoué sur boîte pleine marche (`session.cpp:201-210`, testé) ; le cas d'une source restée sans position est A13-08, sa cause est en zone 14. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 14 | `src/media/source_loop.cpp:110-123` | `recover_position` qui échoue laisse `idle_ = true` sans fin de flux ni refus nouveau : la lecture reste figée jusqu'au prochain seek de l'utilisateur. |
| 14 | `src/media/source_loop.cpp:182-188` | Une lecture abandonnée (refus `recoverable`) est traitée comme la fin du fichier : origine de MED-01, la source pourrait marquer la fin de flux comme « sur erreur ». |
| 17 | `src/app/play/external_subtitles.cpp:92`, `:99`, `:107`, `:114`, `:123` | `path::string()` sur des noms de sous-titres (même remarque que A13-11). |
| 08 | `src/app/play/player_run.cpp:738-739` | Les pistes de la vidéo courante sont copiées dans `Player` même quand la réouverture est pour une autre vidéo (A13-01, A13-09). |
| 30 | `src/app/play/wiring.cpp:113-124` | `ThreadStages::play/pause` rendent `false` sur boîte audio pleine sans que personne ne rejoue (A13-06). |

## Points positifs

- Le protocole de seek (Pause, Flush{e+1}, Seek, rejeu à époque neuve tant qu'une boîte refuse) est clair et testé ;
  les images d'époque passée sont rendues seulement une fois écrites (`retire`, `retire_written`).
- La file d'images est bornée par construction (`kMaxRingSlots`, une `RingFrame` par slot) ; `late_`, `unwritten_` et
  `unmeasured_` sont réservés une fois.
- La reprise s'écrit sur le fil d'écriture (`LatestWriter`), par écriture atomique, au plus toutes les 15 s ; la borne
  de 500 entrées tient (vérifié). Fin vue et début oubliés, vidéo déplacée retrouvée.
- `FrameLatency` (médiane sur 9, un écrivain, atomique relâché) et `Window` (sans allocation, tri mis en cache) sont
  justes et peu coûteux.
- L'en-tête et les lignes de `--stats` viennent du même tableau de colonnes.

## Questions ouvertes

Aucune : Tatoub ne veut plus de questions. Les deux choix possibles sont couverts ici.

- A13-09 : les deux corrections conviennent. Garder les pistes d'un épisode au suivant en les retrouvant par langue et
  titre (et non par index) sert les séries ; une autre vidéo ouverte ou glissée repart du choix par langue. C'est la
  correction recommandée.
- A13-02 : la session expose `ended_on_error()`. Le lecteur retente d'abord le fichier au même endroit (comme
  `GrowingFile`) une fois, puis, s'il échoue encore, s'arrête avec un message, sans enchaîner l'épisode suivant.
