# Consignes du coordinateur pour pc/vendor-hdr-real (19 octobre, 18:18 UTC)

1. DebandingOff : un réglage booléen de profil, débandage ACTIF par défaut (qualité d'abord), avec sa
   plage et son nom dans core/settings ; le rechargement du modèle est accepté au
   changement de réglage, jamais par image. L'interface le montrera à la vague 3. Note la mesure
   (temps avec et sans débandage) dans ~/ajp-reports/truehdr.md.
2. Pic : le lecteur passe toujours sa valeur explicite ; garde 1000 nits du plan, et, quand
   DisplayInfo est connu, la marge relative au blanc de référence de KWin
   (~/ajp-reports/display.md). Écris la valeur par défaut du SDK (650 nits) dans un commentaire, et
   un test qui vérifie qu'on n'en dépend jamais.
3. Fusionne origin/w2/fondations (371b2a2) avant la poussée : conflits triviaux à la main
   (src/vendor_hdr/maxine_runtime.cpp : garde ta version ; tests/unit/vendor_hdr/CMakeLists.txt :
   réunis en DEPS ajp::core ajp::kernels ${CMAKE_DL_LIBS}). Lis aussi
   /home/ta2b/ajp-briefs/contrats-371b2a2.md (Capabilities::true_hdr_blocker,
   vendor_hdr::first_blocker_reason()).
