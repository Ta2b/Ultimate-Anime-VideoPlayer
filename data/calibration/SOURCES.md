# Images d'exemple de la calibration du grade

Onze plans fixes, réunis dans calibration.mkv pour la calibration du grade adaptatif (Paramètres > HDR).
Le lecteur l'ouvre en pause, plan par plan, dans la même chaîne que les épisodes (src/app/play/calibration_images.cpp) :
ce que la calibration montre est ce que la lecture rendra. Livrée dans calibration/ à côté du lecteur
(tools/release/package-*.sh, package-windows.ps1).

Chaque plan tient 6 s (onze plans, 66 s, un chapitre par plan) : 1920x1080, 24 images par seconde, H.264 4:2:0
(crf 8, tune stillimage), BT.709 plage limitée comme un épisode, une image clé au début de chaque plan. Les
sources sont des Blu-ray en HEVC 10 bits ; dans chaque plan d'origine, l'image retenue est la plus immobile du
plan (écart moyen avec l'image précédente et la suivante), hors coupe, fondu et texte. Aucun sous-titre incrusté.

Les cinq premiers plans sont obligatoires : ils tiennent les deux bouts de la courbe, son milieu, la peau et les
éclats d'une scène claire. Les six suivants sont des bonus.

| Plan (instant dans calibration.mkv) | Série, épisode | Instant |
|---|---|---|
| 1 · Night glow (0:03) | Akudama Drive, saison 1, épisode 1 | 00:09:20.084 |
| 2 · Dim skin (0:09) | Les Enfants du Temps (Tenki no Ko), film | 01:13:08.919 |
| 3 · Day skin (0:15) | Les Enfants du Temps, film | 00:35:15.460 |
| 4 · Sky sun (0:21) | Violet Evergarden, saison 1, épisode 1 | 00:15:05.043 |
| 5 · Bright lamps (0:27) | Akudama Drive, saison 1, épisode 1 | 00:14:06.126 |
| 6 · Fog spots (0:33), bonus | Akudama Drive, saison 1, épisode 1 | 00:19:06.920 |
| 7 · Forest (0:39), bonus | Frieren, saison 1, épisode 1 | 00:07:27.250 |
| 8 · White skin (0:45), bonus | Violet Evergarden, saison 1, épisode 1 | 00:03:18.376 |
| 9 · Fireworks (0:51), bonus | Les Enfants du Temps, film | 00:40:04.001 |
| 10 · Warm dark (0:57), bonus | Frieren, saison 1, épisode 1 | 00:22:22.628 |
| 11 · Dusk (1:03), bonus | Your Name (Kimi no Na wa), film | 01:17:17.250 |
