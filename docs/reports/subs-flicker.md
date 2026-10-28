# Clignotement des sous-titres dans « ajp play » (19 octobre 2026)

Vu par Tatoub sur son écran, pendant les essais de la version jouable (pc/vertical-slice, fenêtre
960x540, « Tensei Goblin dakedo Shitsumon Aru - 02 », 18:21–18:34 UTC). Aucun mpv ne tournait.

## Cause

La session du thread main (src/app/play/session.cpp) demande les sous-titres de chaque image à
son arrivée dans le ring, puis garde les réponses de subs dans une file bornée à 3
(kMaxKeptSubtitles = 8 atlas − 4 demandes − 1). Le ring a 4 slots : l'image à l'écran plus
3 images en avance. Quand les réponses des 3 images en avance arrivaient, la file jetait la plus
ancienne, celle de l'image à l'écran. Pendant le reste de cette image (environ 7 vsyncs à 165 Hz
pour une source à 24 im/s), `subtitles_for()` ne trouvait plus de réponse de pts ≤ pts affiché et
rendait nullptr : l'image se redessinait sans sous-titres, puis l'image suivante les retrouvait.
D'où un clignotement au rythme des images.

## Correctif

- `SubtitleQueue` (src/app/play/subtitle_queue.{h,cpp}) porte la règle : les sous-titres se
  composent sur chaque image présentée, au temps de présentation de cette image. L'image affichée
  prend la dernière réponse de pts ≤ son pts. Quand la place manque, c'est la réponse la plus en
  avance qui part, jamais celle dont l'image à l'écran a besoin : son image reprendra la précédente,
  au pire une image tôt, jamais une image sans sous-titres.
- Demandes en vol ramenées de 4 à 3, réponses gardées portées de 3 à 4 (même budget de
  kMaxAtlasesInFlight − 1 atlas), et `static_assert` dans player.cpp : le ring n'a pas plus de slots
  que de réponses gardées.
- Compteur `subtitle_gaps` : présentations sans sous-titres alors que des réponses attendaient,
  écrit dans le résumé de fin de lecture.

## Tests

- tests/unit/app/subtitle_queue_test.cpp : l'image à l'écran garde sa réponse quand le ring se
  remplit en avance ; une image sans réponse à elle prend la précédente ; réponse en retard rangée
  à sa place ; bornes. Avec l'ancienne éviction (pop_front), le premier test échoue (vérifié).
- tests/unit/app/session_test.cpp : sept vsyncs par image, ring plein en avance, l'image montrée a
  ses sous-titres à chaque vsync, subtitle_gaps = 0.

## Vérification réelle

`ajp-gpu build/linux-prefix-cuda/src/app/ajp play "<épisode>" --window=640x360 --seconds=32 --start=300`
sur la RTX, sous-titres ASS français : 753 images neuves, 4782 présentations, **0 présentation sans
sous-titres en attente**, 15 images sautées (démarrage), temps par image médian 42,5 ms.
Journal : ~/ajp-reports/vertical-slice-runs/subs-fix.log.
