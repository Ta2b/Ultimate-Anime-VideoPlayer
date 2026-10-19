# Consignes pour pc/vertical-slice (19 octobre, 18:35 UTC)

Jeton: vs-0419-sub

1. BUG VU PAR LE PROPRIÉTAIRE sur son écran pendant tes essais de « ajp play » : les sous-titres
   CLIGNOTENT. Trouve la cause et corrige-la avec un test qui la fixe. Pistes : atlas libass
   re-téléversé ou rendu vide une image sur deux (SubtitleFrame changé/inchangé mal suivi,
   ass_render_frame qui rend detect_change), overlay dessiné seulement sur certaines images du
   ring ou de la swapchain, horodatage des sous-titres comparé au pts d'une image interpolée ou
   répétée, double tampon de l'overlay rendu à subs trop tôt (atlas réécrit pendant que le GPU le
   lit), ou alternance entre deux générations du ring. Vérifie sur le même épisode, en fenêtre,
   sur au moins 30 s avec des répliques longues, et décris la cause dans vertical-slice.md.
2. PRIORITÉ DU PROPRIÉTAIRE : « une version fonctionnelle avec RIFE, AnimeJaNai et RTX HDR pour
   Linux ». Termine d'abord ce qui marche (agrandissement SPAN sur TensorRT, son, sous-titres sans
   clignotement, touches) et pousse pc/vertical-slice dès que c'est vert ; le coordinateur dira
   comment RIFE et TrueHDR s'y ajoutent (ne commence pas sans sa consigne). La version Windows
   n'est pas une priorité.

Quand tu as intégré ces consignes à ton plan :
  touch "$(git rev-parse --git-dir)/consignes-vs-0419-sub"
