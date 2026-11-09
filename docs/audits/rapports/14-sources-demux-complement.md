# Complément de l'audit 14. Sources, démultiplexage et erreurs d'E/S

Branche : `claude/project-thread-921irj`. Commit audité : `2b42394`. Date : `2026-11-09`.

## Verdict

Sur quatre renvois, deux sont confirmés (A14-09, A14-10, mineurs) et deux sont des doublons (A17-05 et A17-06,
MED-14). Aucune baisse de qualité d'image faite pour la vitesse n'a été trouvée dans ces renvois.

## Cadrage retenu

Seuls les renvois de `complements/14.md` sont vérifiés, dans le code, sans question à Tatoub. Rien n'a été compilé
ni lancé (pas de dépendances dans le conteneur). Le comportement de FFmpeg cité vient de sa source
(`libavutil/mathematics.c`, `libavformat/utils.c`).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A14-09 | Mineur | `src/media/source_loop.cpp:153-159` ; `src/media/video_decoder.cpp:473-492` ; `src/media/frame_data.cpp:283-287` | Au seek exact, chaque image jetée avant la cible passe d'abord tout `VideoDecoder::receive()` : copie GPU vers hôte (`av_hwframe_transfer_data`, quand `download` est vrai et l'appareil non partagé), bwdif, puis `to_decoded_frame`, qui convertit en 4:2:0 par swscale une image hôte 4:2:2, 4:4:4 ou 12 bits. La décision `ends_before` ne vient qu'après. Sur la cible (TensorRT, NVDEC sur l'appareil partagé, `download = !tensorrt` dans `chain_setup.cpp:70`), le coût est négligeable : pas de copie, pas de conversion. Il compte hors de ce chemin. | Seek par la barre (`controls.cpp:125`, exact) ou réouverture de la chaîne (`chain.cpp:73`, `:99`, exact) sur une source 4:4:4 décodée en logiciel : 7,5 ms par image 4K perdus dans swscale (mesure de A15-06). Avec une image clé toutes les 10 s (`keyint=240`), jusqu'à environ 240 images jetées, soit jusqu'à 1,8 s de plus avant la première image. Même chose, en moindre, pour la copie hôte du repli Vulkan non partagé et pour bwdif sur une source entrelacée. | Décider du saut sur `best_effort_timestamp` juste après `avcodec_receive_frame`, avant la copie et la conversion : `receive()` prend la borne `drop_before` (ou rend une image « brute » que la source juge, puis finit). Pour bwdif, ne lui donner que les deux dernières images avant la cible (il lui faut l'image précédente). Ce n'est pas une baisse de qualité : l'image jetée n'est jamais montrée. | Lu |
| A14-10 | Mineur | `src/media/source_loop.cpp:21-25` | `ends_before` jette toute image `kNoPts`, sans borne. Si toutes les images sortent sans horodatage après un flush, la cible n'est jamais atteinte : la source décode le reste du fichier en jetant tout. Le seul chemin connu est A15-05 (bwdif perd `best_effort_timestamp`, et `next_pts_us` vaut `kNoPts` après `flush()`, `video_decoder.cpp:164-167`, `:503`). Mais la garde manque côté source, quelle que soit la cause. | Rip DVD MPEG-2 entrelacé (`.mpg`, `.vob`) en décodage logiciel, seek par la barre. Le paquet audio part dans la réserve (`packet_reserve.cpp:35-47`, 64 Mo, environ 19 min d'AC-3 à 448 kbit/s), puis l'une de deux fins : la réserve déborde, le refus `PacketBacklog` met la source au repos (`fail`, lecture figée), ou la fin du fichier arrive, la dernière image du fichier est livrée (`source_loop.cpp:167-169`) et `end_of_stream` enchaîne l'épisode suivant. Entre-temps, des dizaines de secondes sans image ; un nouveau seek coupe l'attente (la boîte est lue à chaque tour). | Borner le saut : compter les images `kNoPts` jetées de suite et, au-delà d'une durée nominale d'environ 2 s (ou d'un GOP), garder l'image et journaliser. Mieux : dater une image `kNoPts` par le pts du point clé sur lequel le seek est tombé, plus le nombre d'images × durée nominale. Corriger aussi A15-05, qui reste la cause. | Lu (avec la démonstration de A15-05) |

## Doublons

| Renvoi | Où | Conclusion |
|---|---|---|
| de 17 : pré-lecture des sous-titres | `src/media/source_loop.cpp:90-106` | Doublon de **A17-05** (pas de pré-lecture au seek) et de **A17-06** (délai négatif, MED-09). Vu côté source : `on_seek` repart de `PreviousKeyframe` (ou du point clé le plus proche, après la cible, en seek rapide) et les paquets de sous-titres antérieurs ne sont jamais relus. Précision pour la correction : la source lit le fichier dans l'ordre et attend dès qu'une file est pleine (A14-06) ; elle ne peut pas lire les sous-titres en avance sans lire aussi la vidéo en avance. La correction de A17-05 et A17-06 passe donc par un second `Demuxer` qui ne garde que la piste de sous-titres (`keep_only`), repositionné à `cible − N s` au seek et tenu `\|délai\|` + 1 s devant la lecture. |
| de 20 : `duration_to_us` et base de temps nulle | `src/media/av_util.cpp:48-51` | Doublon de **MED-14** (déjà revérifié par le rapport 14, avec une ligne fausse : `av_util.cpp:111-114` n'existe pas, le fichier fait 53 lignes ; la bonne est `48-51`). La base de temps nulle ne peut pas arriver : `avformat_new_stream` pose 1/90000 et `avpriv_set_pts_info` refuse un numérateur ou un dénominateur ≤ 0 en gardant l'ancienne base ; un numérateur nul rendrait 0, pas `kNoPts`. Le vrai chemin est le débordement de `av_rescale_q` sur une durée forgée, qui rend `INT64_MIN` (`kNoPts`). Ses effets sont tous gardés en aval : durée de piste ignorée (`pipeline.cpp:343`, `> 0`), durée d'image remplacée par la nominale (`video_decoder.cpp:166`, `> 0`), événement de sous-titre refusé seul (`subtitle_track.cpp:138`). Fichier forgé seulement ; la correction de MED-14 (borner à 0) suffit. |

## Points positifs

- La boucle lit sa boîte à chaque tour même pendant un saut d'images : un seek ou un arrêt coupe un seek exact
  trop long (`source_loop.cpp:45-53`).
- La dernière image jetée est livrée si la fin du flux arrive avant la cible : le seek exact au-delà de la
  dernière image affiche quelque chose.
