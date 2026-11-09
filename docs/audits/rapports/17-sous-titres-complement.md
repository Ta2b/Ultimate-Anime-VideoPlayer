# Complément de l'audit 17. Sous-titres

Branche : `claude/project-thread-x4fd5f`. Commit vérifié : `9be1f5a`. Date : `2026-11-09`.

## Verdict

Sur trois renvois, un défaut neuf (A17-14, Mineur : aucun plafond cumulé des paquets d'une piste intégrée) et
deux doublons (A17-02 pour les sous-titres image, A24-19 pour le rectangle positionné, dont la correction n'a
pas besoin de la donnée demandée). Aucun des trois ne baisse la qualité d'image pour la vitesse.

## Renvois vérifiés

| Renvoi | Où | Conclusion | Preuve |
|---|---|---|---|
| de 18 | `src/app/play/bridges.cpp:383-385` (aujourd'hui `:95-112`) | Doublon de **A17-02** (et de A18-15, déjà rangé sous A17-02). | `subtitle_track` n'accepte que `ass`, `ssa`, `subrip`, `text`, `srt` (`bridges.cpp:100-108`) et rend `TrackUnavailable` « sous-titres {} non pris en charge » pour tout le reste (`:109-111`) : PGS (`hdmv_pgs_subtitle`), VobSub (`dvd_subtitle`), DVB, `mov_text`, `webvtt`. Les lignes ont bougé depuis l'audit 18, le code est le même. La cause et la correction sont celles de A17-02 : pas de source image dans `TrackSource`, contrat R8 à une couleur par rectangle. |
| de 24 | `src/subs/subtitle_frame.h:59-66` | Doublon de **A24-19**, avec une correction plus simple que celle prévue. | `SubtitleRect` n'a que `src`, `dst_x`, `dst_y`, `color` : rien ne dit de quel événement vient le morceau. Mais subs ne peut pas le savoir non plus : `ass_image` de libass (lu à `subtitle_renderer.cpp:231-232`) n'a ni événement ni drapeau, et le fork `the-database/libass` (`tools/deps/linux-from-source.sh:39`) n'en ajoute pas. Ajouter la donnée demanderait de reparser chaque événement actif (`\pos`, `\move`, `\an`, style) ou de rendre en deux passes. Inutile : libass sait déjà monter seulement les dialogues. Voir la correction ci-dessous. |
| de 35 | `src/subs/subtitle_track.cpp:146-149` | Confirmé : **A17-14**. | Voir la table des défauts. |

### Correction de A24-19 côté subs

`ass_set_line_position(renderer, pourcentage)` déplace vers le haut les seuls événements sans `\pos` ni `\move`
et alignés en bas (c'est ainsi que mpv monte les sous-titres au-dessus de son OSC, `sub-pos`). Le déplacement se
fait par événement entier : une réplique de deux lignes monte d'un bloc, un panneau typographié reste sur son
enseigne. Il suffit d'ajouter à `RenderTarget` (`subtitle_renderer.h:46-56`) un champ `line_position` (0 à 100,
calculé par render depuis la hauteur de la barre), de l'appliquer dans `SubtitleRenderer` avant
`ass_render_frame`, et de retirer le déplacement des morceaux dans `placebo_overlays.cpp:62-76`. Le coût est un
rendu libass de plus quand la barre apparaît ou disparaît (le changement de cible remet déjà le rendu à jour),
aucun à image constante. `SubtitleRect` n'a pas à changer.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A17-14 | Mineur | `src/subs/subtitle_track.cpp:137-150` (ASS) et `:153-159` (texte) | Aucun plafond cumulé pour une piste intégrée. Chaque paquet est borné (1 Mio, `kMaxPacketBytes`, `subtitle_track.h:27`) mais rien ne compte ce que la piste a déjà reçu : `ass_process_chunk` copie chaque événement neuf (ReadOrder distinct) dans la piste libass, pour toute la séance. Une piste texte garde en plus une copie de chaque paquet dans `seen_text_`, bornée seulement à `INT_MAX` événements (`:153`). Un fichier externe, lui, est plafonné à 64 Mio (`kMaxScriptBytes`, `:46`, `:98`, `:109`). Les bornes de la file de paquets (`media/packet.h:68`) ne valent que pour les paquets en vol, pas pour le cumul. | MKV forgé : piste ASS dont les paquets sont compressés par zlib (ContentEncoding Matroska, que FFmpeg décompresse). Un paquet de 1 Mio de texte répété tient en quelques Ko : un fichier de 10 Mo donne environ 10 000 événements de 1 Mio, soit 10 Gio dans libass, puis `bad_alloc` ou le tueur de mémoire. En lecture normale, la mémoire grossit avec la position ; un seek en arrière ne libère rien. Pour un fichier honnête, le pire réel (karaoké d'un film de deux heures) reste sous quelques dizaines de Mo : le plafond ne touche personne. | Compter dans `SubtitleTrack` les octets et les événements acceptés ; au-delà de `kMaxScriptBytes` (64 Mio, comme un fichier externe) ou d'un nombre d'événements (par exemple 1 048 576), refuser les paquets suivants avec un seul avertissement, et laisser la piste lisible. Ne compter que les paquets neufs : un paquet revu après un seek (même ReadOrder pour l'ASS, même clé dans `seen_text_` pour le texte) ne doit pas user le plafond. Pour l'ASS, cela veut dire lire le ReadOrder en tête du paquet (premier champ avant la virgule) et garder l'ensemble des ReadOrder vus. | Lu. Test à écrire dans `tests/unit/subs/` : `SubtitleTrack::from_ass_header` puis `add_packet` en boucle avec des ReadOrder distincts et 1 Mio de texte ; aujourd'hui tous acceptés, attendu un refus après 64 Mio, et la piste toujours rendue. |

## Qualité d'image

Rien à signaler dans ces renvois : aucun ne touche à la définition ou à la fidélité du rendu des sous-titres
pour gagner du temps. La seule réduction de définition de la zone reste A17-01 (MED-05), déjà rangée.

## Questions ouvertes

Aucune.
