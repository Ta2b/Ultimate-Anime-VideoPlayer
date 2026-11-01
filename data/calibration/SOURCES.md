# Images d'exemple de la calibration du grade

Huit plans fixes, réunis dans calibration.mkv pour la calibration du grade adaptatif (Paramètres > HDR).
Le lecteur l'ouvre en pause, plan par plan, dans la même chaîne que les épisodes (src/app/play/calibration_images.cpp) :
ce que la calibration montre est ce que la lecture rendra. Livrée dans calibration/ à côté du lecteur
(tools/release/package-*.sh, package-windows.ps1).

Chaque plan tient 6 s (huit plans, 48 s, chapitres « Dark 1 » à « Bright 4 ») : 1920x1080, 24 images par
seconde, H.264 4:2:0 (crf 8, tune stillimage), BT.709 plage limitée comme un épisode, une image clé au début
de chaque plan. Dans chaque plan d'origine, l'image retenue est la plus immobile (écart moyen avec l'image
précédente et la suivante), hors coupe, fondu et texte. Aucun sous-titre incrusté.

| Plan (instant dans calibration.mkv) | Série, épisode | Instant |
|---|---|---|
| Dark 1 (0:03) | Re:Zero, saison 4, épisode 18 | 00:07:29.825 |
| Dark 2 (0:09) | Re:Zero, saison 4, épisode 18 | 00:07:41.668 |
| Dark 3 (0:15) | Re:Zero, saison 4, épisode 18 | 00:06:17.126 |
| Dark 4 (0:21) | Re:Zero, saison 4, épisode 18 | 00:16:23.042 |
| Bright 1 (0:27) | Re:Zero, saison 4, épisode 18 | 00:01:20.293 |
| Bright 2 (0:33) | Moi, quand je me réincarne en Slime (Tensei Shitara Slime Datta Ken), saison 4, épisode 24 | 00:18:38.418 |
| Bright 3 (0:39) | Moi, quand je me réincarne en Slime, saison 4, épisode 24 | 00:19:01.292 |
| Bright 4 (0:45) | Re:Zero, saison 4, épisode 18 | 00:01:37.681 |
