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
1. **Un correctif du port** (`ports/libplacebo/0001-ajp-present-timing.patch`, sur v7.387.0, 275f7dea ; écrit sur v7.360.1, rebasé en novembre 2026). Le
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

## Mise en œuvre (6.2 à 6.10)
D'abord derrière des options d'essai, la boucle fermée est **active par défaut** depuis la fin du chantier 6
(`options.h`) : `--present=wait`, le veilleur d'écriture, l'attente précise, la PLL audio et le choix par
numéro de vblank. Chaque pièce se coupe seule (`--no-ring-watcher`, `--no-precise-wait`, `--no-audio-pll`,
`--no-vsync-index`, `--present=open`), ce qui rend la présentation d'avant.

| Option (défaut) | Étape | Ce qu'elle fait |
| --- | --- | --- |
| `--present=auto\|timing\|wait\|wayland` (wait) | 6.2 à 6.4, 6.8, 6.10 | Retour de présentation au plus haut niveau offert (`present_level.h`). La grille des vblanks mesurée (`core::VblankClock`) remplace la fréquence annoncée pour la synchro audio et la grille du mode Rapide ; l'état VRR lu par present timing prime sur l'ioctl. |
| `--ring-watcher` (oui) | 6.5 | Un thread attend l'écriture des slots et réveille main (`ring_watcher.h`). |
| `--precise-wait` (oui) | 6.7 | Attente de l'image suivante à la microseconde (`loop_wait.h`). |
| `--audio-pll` (oui) | 6.9 | Horloge audio filtrée par une PLL (`core/clock_pll.h`). |
| `--vsync-index` (oui) | 6.6 | Image choisie par le numéro de son vblank, avec hystérésis ; date visée en VRR au niveau TimingAbsolute. Demande `--present`. |

Les modules :
- `core/present_ledger.h` : identifiants, appariement des retours (désordre, partiels), suspension sur file
  pleine, domaine de temps recalé toutes les 2 s quand le pilote ne date pas en CLOCK_MONOTONIC ;
- `app/play/placebo_feedback.h` : la partie Vulkan (file de retour de 16, VkPresentTimingInfoEXT, relecture
  par `vkGetPastPresentationTimingEXT` sans attente, `vkWaitForPresent2KHR` sondé à délai 0 au niveau Wait) ;
- `app/play/placebo_wayland.h` : `wp_presentation` sur une file d'événements à nous, lecture sans attente ;
  compilé seulement avec wayland-client et wayland-scanner.

`--stats` gagne les colonnes `present_id`, `feedback_id`, `target_us`, `visible_us`, `vsync_k` et `level`
(`tools/bench/banc_resume.py` lit les deux formats). Au journal, à la fermeture : présentées, revenues,
perdues, file pleine, fréquence mesurée, latence de sortie. Les critères de 6.11 (σ visé − affiché, réveils de
main, p99 de l'attente, σ de `lag_us`) se mesurent sur le PC, en A/B contre `--present=open`.

## Mesures (relecture de novembre 2026, P3.47)
Ce qui fonde les défauts, tel que le code le note :
- `wait` plutôt que `auto` ou `timing` : sur NVIDIA 615 et KDE Wayland, present timing n'a rendu aucun retour
  (séance du 04/11, 0 image revenue) ; `present_wait2` répond, et le niveau descend seul quand l'appareil ne
  l'offre pas. Depuis l'audit (A10-02, A10-06), `wp_presentation` s'ouvre aussi pour `wait` et passe devant
  `present_wait2`, qui ne date rien : sans date, ni grille des vblanks ni `--vsync-index`. Un niveau `timing`
  resté muet 120 présentations retombe sur `wp_presentation`, sinon sur `wait` ;
- veilleur d'écriture et attente précise : main n'est plus réveillé toutes les 4 ms pour sonder l'anneau.

Les chiffres de 6.11 (σ visé − affiché, réveils de main, p99 de l'attente, σ de `lag_us`) n'ont pas été
consignés. Ils restent à relever sur le PC de référence, en A/B contre `--present=open --no-ring-watcher
--no-precise-wait --no-audio-pll --no-vsync-index`, et à reporter ici avant la 1.0.
