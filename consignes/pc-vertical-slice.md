# Consignes pour pc/vertical-slice (19 octobre, 18:40 UTC) — BUT UNIQUE DU PROJET

Jeton: vs-0419-rife-hdr

Le propriétaire (18:34 UTC) : « je veux simplement une version fonctionnelle avec rife, animejanai
et rtx hdr pour linux. J'ai constaté un bug visuel de clignotement des sous-titres. Pour l'instant
on s'en fout un peu de la version windows. »

1. BUG : les sous-titres CLIGNOTENT dans ta fenêtre de « ajp play » (vue par le propriétaire sur son
   écran pendant tes essais de 20:21 à 20:34, épisode Tensei Goblin 02). Trouve la cause et corrige.
   Règle du projet, à fixer par un test : les sous-titres se composent APRÈS l'interpolation, sur
   CHAQUE image présentée, à partir du temps de présentation de cette image (jamais du pts d'une
   source, jamais une image sur deux). Pistes : atlas libass re-téléversé ou vide une image sur
   deux (detect_change mal suivi), overlay dessiné sur certaines images seulement, atlas rendu à
   subs pendant que le GPU le lit, alternance de générations du ring. Vérifie sur au moins 30 s de
   répliques longues. Écris le diagnostic dans ~/ajp-reports/subs-flicker.md (cause, correctif,
   test) et résume-le dans vertical-slice.md.
2. BUT ÉTENDU : pc/vertical-slice devient LA version Linux fonctionnelle « ajp play » avec :
   - SPAN AnimeJaNai sur TensorRT (fait) ;
   - RIFE 4.26 sur TensorRT, interpolation calée sur l'écran 165 Hz (cadence de core), Quality ou
     Fast par option (--rife=quality|fast|off, Quality par défaut comme le profil actuel : Area 720
     → RIFE ×4 → SPAN), modèle rife_v4.26.onnx de ~/.local/share/AnimeJaNai-HDR-linux/animejanai/rife
     en lecture seule ;
   - TrueHDR (Maxine, /usr/local/VideoFX, via src/vendor_hdr ; fusionne origin/pc/vendor-hdr-real
     quand elle paraît) en option --truehdr, réglages du profil actuel (contraste, saturation, gris
     moyen, pic) ;
   - son, sous-titres, pause, seek, plein écran, quitter, OSD minimal.
   Fusionne au fil de leur parution : origin/pc/media-gpu, origin/pc/present, origin/pc/tensorrt,
   origin/pc/cuda-compute, origin/pc/gpu, origin/pc/render, origin/pc/vendor-hdr-real,
   origin/pc/app-shell. Pousse pc/vertical-slice à chaque étape verte (d'abord sous-titres corrigés,
   puis RIFE, puis TrueHDR), en notant chaque fois dans vertical-slice.md la commande exacte, le
   fichier, les temps par image et les images sautées.
3. Windows : hors sujet pour l'instant.

Quand tu as intégré ces consignes à ton plan :
  touch "$(git rev-parse --git-dir)/consignes-vs-0419-rife-hdr"
