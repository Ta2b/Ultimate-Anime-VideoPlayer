# Consignes du coordinateur pour pc/media-gpu (19 octobre, 18:06 UTC)

- Fusionne origin/w2/fondations (371b2a2) et lis /home/ta2b/ajp-briefs/contrats-371b2a2.md
  (FrameData : begin_vulkan_access()/end_vulkan_access() à garder dans media/frame_data.cpp ;
  ExternalImage gagne pas de ligne, plans et valeur timeline si tu en as besoin).
- Suppressions LSan : étends le fichier existant tools/ci/lsan.supp (de w2/fondations, lu par les
  préréglages de test ASan) au lieu d'en créer un autre : leak:libcuda.so, leak:libnvidia-glcore.so,
  leak:libnvidia-glsi.so, leak:libnvidia-gpucomp.so, chacun justifié en commentaire (fuites internes
  du pilote : cuInit puis dlclose ; ICD Vulkan NVIDIA). Pose detect_leaks=0 seulement pour les tests
  [vulkan] lancés sur NVIDIA (l'ICD déchargé laisse des cadres « <unknown module> » qu'aucune
  suppression par nom ne vise), et documente pourquoi.
