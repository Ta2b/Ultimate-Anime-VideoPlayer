# Consignes du coordinateur pour pc/vk-compute-nvidia (19 octobre, 18:06 UTC)

1. Fusionne origin/w2/fondations (371b2a2) dans ta branche. Conflit d'enum RefusalCode dans
   src/core/types.h : ordre tranché, ceux de w2/fondations d'abord, puis les 4 de w2/vk-compute
   à la suite (VulkanCallFailed, DeviceUnavailable, FeatureUnavailable, GpuMemoryExhausted),
   chacun avec son nom dans name(). L'intégrateur gardera cet ordre. Lis aussi
   /home/ta2b/ajp-briefs/contrats-371b2a2.md.
2. VkContext::submit ne doit plus allouer côté hôte à chaque soumission (aujourd'hui cinq
   std::vector, queue.cpp:48-69, encore dans w2/vk-compute-review-wip) : tableaux de taille bornée
   réutilisés, ou petit vecteur membre réservé une fois ; un test le fixe.
3. glslc avec -O (cmake/AjpSpirv.cmake:57). Mesure avant/après sous le verrou GPU (ajp-gpu) avec le
   banc de pc/vk-compute-bench (fusionne origin/pc/vk-compute-bench si elle est poussée), et note
   le résultat dans ~/ajp-reports/vk-compute-timings.md.
