# Complément de l'audit 16. Audio

Branche : `claude/project-thread-he3e8d`. Commit audité : `9be1f5a`. Date : `2026-11-09`.

Renvois traités : celui de la liste `complements/16.md`, venu de l'audit 14.

## Verdict

Le renvoi est confirmé : quand un seek exact échoue après avoir bougé le fichier et que la source revient à sa dernière
image, l'audio de la nouvelle époque reste rogné jusqu'à la cible du seek manqué. Le son se tait pendant toute la
distance du seek, l'image se fige 1,5 s, puis l'horloge de l'écran mène jusqu'au seek suivant. Nouveau défaut A16-14.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A16-14 | Mineur | `src/app/play/wiring.cpp:103-108`, `src/app/play/bridges.cpp:43-47`, `:88-91`, `src/audio/player.cpp:65`, `:226-227`, `src/media/source_loop.cpp:90-122` | La cible du rognage audio est posée par main avant le seek (`expect_seek(flushed_epoch_, target_us)`), et jamais corrigée. Si le seek échoue après avoir bougé le fichier, la source revient à `last_pts_us_` sous la **même** nouvelle époque (`recover_position`, `drop_before_ = back_to`). Le pont marque pourtant chaque paquet audio de cette époque `discard_before_us = target_us` (`bridges.cpp:90`), et `AudioPlayer::push` jette tout bloc avant la cible (`player.cpp:226-227`) jusqu'au Flush suivant. Pour un seek en avant, tout l'audio entre la position reprise et la cible est jeté. Rien n'est poussé, donc l'horloge audio de l'époque reste invalide (`player.cpp:232-238` ne publie qu'au premier bloc poussé) : la vidéo attend `audio_stall_grace_us_` (1,5 s, puis 0,4 s, `session.cpp:25`, `session.h:252`), puis `watch_clock` déclare l'audio absent (`session.cpp:439-443`) et l'horloge hôte mène jusqu'au prochain Flush (`wiring.cpp:87-94`, `:150-157`). Un seek en arrière n'est pas touché (`drop_before` sans effet sous la cible). | Lecture depuis un NAS ou un disque USB qui hoquette. Flèche droite (+5 s, seek exact, `core/keys.cpp:160-162`, `keys.h:122`) au moment d'une erreur de lecture : `seek_before` a déjà appelé `av_seek_frame` (`file_seeks` a crû), donc `recover_position` ramène la lecture à l'image affichée. L'image se fige 1,5 s, l'OSD dit « audio absent : l'horloge de l'écran mène », puis la vidéo repart **sans son pendant 5 s**. Un clic loin en avant dans la barre (seek exact, `controls.cpp:125`) : silence pendant toute la distance du clic, plusieurs minutes possibles, alors que la vidéo joue. Le son revient ensuite calé sur l'horloge hôte, et l'audio ne mène plus jusqu'au seek suivant. | Que la cible du rognage vienne de la source, qui seule sait où la lecture reprend : poser `discard_before_us` dans la source (le `drop_before_` qu'elle applique déjà à la vidéo, `source_loop.cpp:106` et `:120`) et le porter par `media::Packet`, ou faire dire au pont la position reprise (`expect_seek(epoch, back_to)`) depuis `recover_position`. Dans les deux cas, effacer la cible quand la source n'a pas de position sûre. | Lu. Test à écrire (zone 14 et 16) : source dont le seek échoue après `av_seek_frame` puis réussit au retour ; seek exact à `last + 5 s` ; attendu des paquets audio de l'époque sans `discard_before_us` au-delà de `last`, et un bloc poussé avant 1,5 s. Aujourd'hui, tous les blocs sont jetés jusqu'à `last + 5 s`. |

## Renvois traités

| Renvoi | De l'audit | Conclusion | Note |
|---|---|---|---|
| `src/audio/` : rognage audio sur la cible d'un seek refusé (MED-06) | 14 | Confirmé : A16-14 | Le cas qui compte est le seek qui échoue **après** avoir bougé le fichier et dont le retour réussit : c'est le seul où la source reprend la lecture sous l'époque du seek. Ce n'est pas un doublon : A13-08 et la moitié « source » de MED-06 (`14-sources-demux.md`, ligne MED-06) traitent le retour qui échoue, où la source reste inactive et aucun paquet audio ne part, donc le rognage ne joue pas. A14-04 traite le refus sans mouvement de fichier ; ce chemin donnerait le même silence (paquets de l'ancienne position marqués de la cible), mais il est presque inaccessible depuis la session : `Session::seek_to` borne déjà la cible à `[0, duration_us]` (`session.cpp:190-191`), les mêmes bornes que le refus de `Demuxer::seek` (`demuxer.cpp:426-429`). Il ne reste que `TrackUnavailable`, qui ne se produit pas sur la piste vidéo ouverte. La correction d'A16-14 (cible venue de la source) couvre aussi ce cas. |

## Points positifs

- Le rognage ne s'applique qu'aux seeks exacts (`wiring.cpp:105-106`) : les seeks rapides d'une touche maintenue
  (`input.cpp:83-86`) n'en dépendent pas.
- La cible est rangée par époque (`trim_epoch_`) : un paquet d'une autre époque n'est jamais rogné à tort, et
  `start_epoch` efface la cible côté audio (`player.cpp:174`).
