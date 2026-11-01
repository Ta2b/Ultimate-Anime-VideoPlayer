# Sources des images de calibration

Images 1920x1080 JPEG (q=2, 4:4:4), converties explicitement BT.709 limité -> BT.601 plein (). Aucun sous-titre incrusté.
Mouvement = écart absolu moyen (0-255, gris 480x270) avec l'image précédente et suivante, moyenné ; image choisie = la plus immobile du plan, hors coupe (écart > 10), fondu et texte. Les timecodes sont à ±1 image (la lecture des pts a dû être reconstituée).

| Fichier | Source | Timecode | Mouvement |
|---|---|---|---|
| dark_1.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:07:29.825 | 0.0000 |
| dark_2.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:07:41.668 | 0.0000 |
| dark_3.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:06:17.126 | 0.7382 |
| dark_4.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:16:21.710 | 1.3534 |
| bright_1.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:01:20.293 | 0.0085 |
| bright_2.jpg | [Erai-raws] Tensei Shitara Slime Datta Ken 4th Season - 24 [1080p CR WEB-DL AVC AAC][MultiSub][C84E0737].mkv | 00:18:38.418 | 0.1618 |
| bright_3.jpg | [Erai-raws] Tensei Shitara Slime Datta Ken 4th Season - 24 [1080p CR WEB-DL AVC AAC][MultiSub][C84E0737].mkv | 00:19:01.292 | 0.1864 |
| bright_4.jpg | [Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [1080p CR WEB-DL AVC AAC][MultiSub][AD8121A0].mkv | 00:01:37.681 | 0.0000 |

dark_3 (0,74) et dark_4 (1,35) : plans à éclats ou lueur animés, la plus immobile est retenue. dark_1 : plan large de l'escalier à 7:29-7:32 (la source donnée, 7:28, tombait sur le gros plan de Ram) ; bright_4 : Emilia à 1:37 (1:36 est un plan de main, texte du générique à 1:28-1:35).
