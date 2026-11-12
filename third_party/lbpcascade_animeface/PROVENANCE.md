# lbpcascade_animeface, copie intacte

- Source : https://github.com/nagadomi/lbpcascade_animeface, commit
  `4433ab1ae1166ea75acfe99eb0f18709dac329a0`, fichier `lbpcascade_animeface.xml`.
- Licence : MIT, © 2011 nagadomi, écrite dans l'en-tête du fichier lui-même. Le texte est repris dans
  `THIRD_PARTY_NOTICES.md`, livré avec le lecteur.
- SHA-256 : `9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30`.
- Usage : la cascade LBP (20 étages, fenêtre de 24 × 24) est embarquée telle quelle dans la pré-analyse des plans
  et évaluée par `src/media/face_cascade.cpp`, écrit pour le lecteur (sans OpenCV) : les visages trouvés donnent
  la palette de peau du plan (`docs/hdr-anime.md`, « Peau des visages »). Le fichier n'est pas modifié.
