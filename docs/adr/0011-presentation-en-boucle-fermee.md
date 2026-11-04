# 0011 — Présentation en boucle fermée : un correctif de libplacebo, pas une swapchain maison

## Contexte
Pour asservir la présentation (chantier 6), le lecteur doit savoir quand chaque image paraît à l'écran.
Vulkan le dit depuis 1.4.335 : `VK_KHR_present_id2`, `VK_KHR_present_wait2` et `VK_EXT_present_timing`. Mais
la swapchain appartient à libplacebo (`pl_vulkan_create_swapchain`), qui ne chaîne rien à la création ni à la
présentation, et n'expose pas la `VkSwapchainKHR` courante. Deux voies étaient possibles :
- un correctif de libplacebo ;
- une swapchain maison, enveloppée par `pl_vulkan_wrap` (5 à 7 jours, et tout le travail de recréation de
  libplacebo à refaire).

## Décision
1. **Un correctif du port** (`ports/libplacebo/0001-ajp-present-timing.patch`, sur v7.360.1, cee9b076). Le
   port vcpkg (`PATCHES`, port-version 1) et `tools/deps/linux-from-source.sh` appliquent le même correctif.
   Il ajoute :
   - `pl_vulkan_swapchain_params` gagne `present_flags` (drapeaux de création), `create_pnext` (chaîne de
     création) et `min_images` ;
   - `pl_vulkan_swapchain_submit(sw, const pl_vulkan_present*, VkResult*)` chaîne `VkPresentId2KHR` et
     `VkPresentTimingsInfoEXT` à la présentation ; `timing` y est un `const void *`, pour que la disposition de
     la structure ne dépende pas de la version des en-têtes Vulkan ;
   - `pl_vulkan_swapchain_current(sw, &recreations)` rend la swapchain courante et le nombre de recréations :
     l'état propre à une swapchain (taille de la file de timing, propriétés) se repose quand ce nombre change.
2. **La panne NVIDIA 615.78.08.** `VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT` compte comme une image en file :
   l'appel rend vrai, et l'appelant suspend ses demandes de timing jusqu'à une relecture. La fence de cette
   présentation est marquée orpheline :
   - au prochain acquire de l'image, elle n'est attendue que 50 ms, puis remplacée ;
   - une swapchain retirée ne l'attend pas ;
   - les fences orphelines sont détruites après `vk_wait_idle`, à la destruction.
3. **Comportement inchangé par défaut.** Sans `present_flags` et sans `pl_vulkan_swapchain_submit`, libplacebo se
   comporte comme avant. Le patch sera proposé en amont.

## Conséquences
- La construction des dépendances reconstruit libplacebo : la somme du correctif entre dans la signature.
- Le câblage côté lecteur se fait par étapes, chacune derrière `--present`, l'actuel restant `open` :
  - 6.2 : niveaux de présentation ;
  - 6.3 : retour par present timing ;
  - 6.4 : repli `wp_presentation` ;
  - 6.5 : veilleur d'écriture.
