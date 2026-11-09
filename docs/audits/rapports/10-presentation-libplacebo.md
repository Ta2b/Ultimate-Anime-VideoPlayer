# Rapport de l'audit 10. Rendu et présentation par libplacebo

Branche : `claude/project-thread-ybwtt9`. Commit audité : `63289c5`. Date : `2026-11-09`.

## Verdict

La présentation tient sur le chemin nominal, mais elle ne sait pas échouer : une perte du GPU ou de la surface
fige l'image sans message et inonde le journal (GPU-06, toujours présent), un GPU bloqué bloque la fermeture
(GPU-17), et le niveau de retour par défaut (`--present=wait`) ne peut jamais retomber sur `wp_presentation`,
contrairement à ce que disent `options.h` et le test des niveaux.

## Cadrage retenu

Question posée à Tatoub le 09/11 : ce qui compte le plus dans la présentation. Réponse : **tout à égalité**. Perte
du GPU, fenêtre glissée, réduite ou redimensionnée, retour de présentation, anneau et veilleur, captures ont reçu la
même profondeur. Le rendu hors écran (`placebo_offscreen.cpp`, essais du bench) et les instantanés (`--snapshot`)
sont des outils : un défaut y est au plus Mineur. `vulkan_share.cpp` est lu pour la durée de vie de l'appareil
prêté à FFmpeg ; l'export (utilisateur de `create_ring`) est hors cible.

Le conteneur n'a ni GPU, ni en-têtes Vulkan, ni libplacebo, ni les dépendances du projet (`~/ajp-deps` absent) :
rien n'a été compilé ni lancé. Les défauts sont lus dans le code, en suivant les appels jusqu'à `session.cpp` et
`player_run.cpp` (zones 08 et 12) là où ils traversent la frontière. Ce qui demande un GPU ou un écran est marqué
« à confirmer », avec la manipulation ; les tests qui échoueraient sont décrits.

Pour se repérer : `PlaceboScreen::open` (`placebo_open.cpp:363`) crée fenêtre, instance, appareil, swapchain FIFO et
retour de présentation ; `show` (`placebo_screen.cpp:522`) acquiert l'image de la swapchain, compose l'emplacement
de l'anneau (grade, sous-titres, OSD, interface) et la présente ; `Session::present` (`session.cpp:535`) l'appelle
et, si `presented` est faux, redemande une présentation au tour suivant.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A10-01 | Majeur | `src/app/play/placebo_screen.cpp:549` | Un échec de `pl_swapchain_start_frame` (ou de `pl_swapchain_resize`, ligne 545) est pris pour « fenêtre cachée » : `presented = false`, sans erreur. `Session::present` (`session.cpp:540`) pose `present_requested_`, la boucle redemande au tour suivant après 1 à 4 ms (`session.cpp:598-616`). Rien n'appelle jamais `pl_gpu_is_failed` dans `src/app`. C'est GPU-06. | TDR sous Windows 11, Xid ou pilote rechargé sous CachyOS, câble d'écran débranché (`VK_ERROR_SURFACE_LOST_KHR`) : l'acquisition échoue à chaque tour, libplacebo écrit une erreur à chaque fois (`placebo_log`, `placebo_open.cpp:84`), soit 250 à 1 000 lignes par seconde ; l'image reste figée, l'audio continue, aucun message, la fenêtre ne se ferme pas. | Après un échec d'acquisition ou de redimensionnement : si `pl_gpu_is_failed(gpu)`, rendre un refus (`DeviceLost`) qui termine la chaîne avec un message à l'utilisateur. Sinon compter les échecs de suite et refuser au-delà d'un seuil (par exemple 2 s). Écrire l'échec au journal une fois, puis un résumé. Limiter aussi le rappel de `placebo_log` (même message répété). | Lu. À confirmer : `nvidia-smi -r` ou TDR forcé (`dxcap -forcetdr` sous Windows) pendant la lecture, compter les lignes du journal par seconde. |
| A10-02 | Majeur | `src/app/play/placebo_open.cpp:497` | `wp_presentation` n'est ouvert que pour `--present=auto` ou `--present=wayland`. Avec le défaut `--present=wait` (`options.h:171`), `wayland_presentation` reste faux et le « meilleur niveau en dessous » promis (`options.h:168`, `present_level.h:425`) ne peut être qu'`Open`. Le test `present_level_test.cpp:31` (`wayland` + `Wait` → `Wayland`) vérifie un cas que le lecteur n'atteint jamais. Au niveau `Wait`, rien n'est daté (`dated()`, `placebo_feedback.h:60`), donc `vblank_grid()` ne rend jamais rien (`placebo_screen.cpp:459`). | CachyOS, KDE Wayland, RTX : si le pilote n'offre pas `present_wait2`, le lecteur tombe en `Open` alors que KWin offre `wp_presentation`. S'il l'offre, il prend `Wait`, qui ne date rien, devant `Wayland`, qui date. Dans les deux cas, ni grille des vblanks, ni `--vsync-index` effectif, ni fréquence mesurée, sur la cible et par défaut. | Ouvrir `WaylandPresentation` aussi pour `Wait`. Revoir l'ordre de `choose_present_level` : un niveau daté (`Wayland`) passe devant un niveau qui ne fait que sonder (`Wait`), ou les deux se cumulent. Un test qui passe par `PlaceboScreen::open` (ou une fonction pure qui décide aussi de l'ouverture de `wp_presentation`). | Lu. À confirmer : la ligne « présentation : niveau … » du journal sous CachyOS au lancement par défaut, puis avec `--present=wayland`. |
| A10-03 | Mineur | `src/app/play/ring_watcher.cpp:474` | Toute erreur de `vkWaitSemaphores` autre que le délai fait sortir le thread du veilleur sans un mot au journal. Le veilleur reste branché : `wakes_on_write()` vaut toujours vrai (`placebo_screen.cpp:428`), et la boucle n'attend plus qu'au pas de 4 ms (`session.cpp:604`) au lieu de 1 ms. Ensuite, `forget_all` (ligne 441) attend un `seen_epoch_` que plus personne n'avance : `thread_.joinable()` reste vrai pour un `jthread` fini et non joint. C'est GPU-10. | `VK_ERROR_DEVICE_LOST`, ou une erreur passagère (`VK_ERROR_OUT_OF_HOST_MEMORY`) : chaque image arrive jusqu'à 4 ms après son écriture, sans trace. Si un jour l'anneau est recréé pendant qu'un veilleur vit (aujourd'hui `WriteWatch`, `player_run.cpp:186`, le détruit avant), `PlaceboRing::clear` bloque le thread de la fenêtre pour toujours (latent). | À la sortie de `run`, poser `dead_` sous le verrou, `settled_.notify_all()`, écrire le code au journal une fois. `forget_all` attend `seen_epoch_ >= epoch \|\| stopping_ \|\| dead_`, et `wakes_on_write()` rend faux pour un veilleur mort. | Lu. Test décrit : un `Api` dont `wait` rend `VK_ERROR_DEVICE_LOST` ; `forget_all()` doit rendre la main en moins de 100 ms et `wakes_on_write()` passer à faux. |
| A10-04 | Mineur | `src/app/play/placebo_open.cpp:323` | Attentes sans délai sur un GPU bloqué mais pas perdu : `pl_gpu_finish` à la fermeture, dans `PlaceboRing::clear` (`placebo_ring.cpp:242`) et `create` (`placebo_ring.cpp:189`), et `while (pl_tex_poll(...))` du rendu hors écran (`placebo_offscreen.cpp:294`). C'est GPU-17. | Un noyau TensorRT qui boucle, ou un pilote qui ne répond plus sans signaler la perte : la fenêtre est cachée en premier (ligne 320), puis `pl_gpu_finish` ne rend jamais la main. Le processus reste vivant et invisible, et l'instance unique refuse le lancement suivant. | Ne pas appeler `pl_gpu_finish` si `pl_gpu_is_failed`. À la fermeture, un chien de garde (par exemple 5 s) qui écrit l'état au journal et termine le processus. Dans `offscreen_target`, un nombre d'essais borné, puis un refus. | Lu. À confirmer : un noyau CUDA en boucle infinie lancé par un test caché, puis fermeture de la fenêtre. |
| A10-05 | Mineur | `src/app/play/placebo_screen.cpp:522` | Rien ne dessine pendant la boucle modale de Windows (glisser ou redimensionner la fenêtre) : aucun `SDL_AddEventWatch` sur `SDL_EVENT_WINDOW_EXPOSED`, aucun rappel de SDL (`SDL_AppIterate`). Le thread de la fenêtre reste dans `DefWindowProc` tant que le bouton est tenu. | Windows 11 : on déplace la fenêtre pendant la lecture. L'image se fige, l'audio continue, l'anneau se remplit et le GPU s'arrête. Au relâchement, la session rattrape l'horloge audio en jetant des images. Pendant un redimensionnement, l'image figée est étirée par le compositeur. | Un `SDL_AddEventWatch` qui, sur `SDL_EVENT_WINDOW_EXPOSED` reçu pendant la boucle modale, représente la dernière composition gardée (`compose_kept`) à la nouvelle taille. Ou passer la boucle aux rappels de SDL3. | À confirmer : sous Windows 11, glisser la fenêtre 5 s pendant la lecture ; regarder l'image et les compteurs `dropped` de `--stats`. |
| A10-06 | Mineur | `src/app/play/placebo_feedback.cpp:241` | Le niveau de retour est fixé aux capacités et ne retombe jamais quand rien ne revient. `options.h:169` note 0 retour au niveau Timing (NVIDIA 615, KDE Wayland). `PresentFeedback` continue pourtant à demander des dates, `ledger_.lost()` grossit, et rien ne le dit avant le bilan de fermeture (ligne 200). | `--present=auto` sur la cible : Timing est choisi, aucune date ne revient, pas de grille, sans repli vers `Wayland` ni `Wait`. | Après N présentations sans aucun retour (par exemple 120), retomber au niveau inférieur et l'écrire au journal. | Lu ; mesure de Tatoub citée dans `options.h:169`. |
| A10-07 | Mineur | `src/app/play/placebo_feedback.cpp:378` | Sous Windows, le domaine de l'hôte est `VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR`, qui n'y existe pas : `steady_clock` y suit QPC (`core/clock.h:14`). `choose_domain` (ligne 346) n'accepte que `CLOCK_MONOTONIC` ou un domaine local, et `calibrate` le prend comme référence. | Windows 11, pilote qui offre `VK_EXT_present_timing` : `vkGetCalibratedTimestampsKHR` reçoit un domaine non annoncé (usage invalide), le domaine n'est jamais calé, `issue.timing` reste faux et le niveau Timing ne date rien. | Sous Windows, prendre `VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR` comme domaine de l'hôte et le convertir avec la fréquence de QPC. Accepter ce domaine dans `choose_domain` et `read_back` (ligne 418). | Lu. À confirmer : `--present=timing` sous Windows avec la couche de validation, regarder « date absolue » et les retours au bilan de fermeture. |
| A10-08 | Mineur | `src/app/play/placebo_screen.cpp:569` | Les sorties en erreur après `pl_swapchain_start_frame` (lignes 569, 586, 605, 609, 616) ne soumettent jamais l'image acquise. L'erreur termine `run_chain` (`player_run.cpp:654`), mais l'écran survit : l'accueil et la réouverture appellent encore `show` sur la même swapchain. | Une image d'une ancienne génération d'anneau (`SequenceOrder`, `placebo_ring.cpp:196`) ou un `pl_render_image` refusé : une image de la swapchain reste acquise sans présentation, et le `start_frame` suivant peut échouer ou bloquer selon le pilote. | Une garde qui, sur toute sortie après l'acquisition, présente une image vide (`pl_frame_clear` puis `pl_swapchain_submit_frame`). | Lu. À confirmer : comportement de libplacebo sur deux `start_frame` sans `submit`. |
| A10-09 | Mineur | `src/app/play/placebo_ring.cpp:229` | `retire` ignore le résultat de `hand_to_cuda`. Si `pl_vulkan_hold_ex` échoue, le sémaphore du slot n'atteint jamais `ready_value + 1`, mais le slot est quand même rendu au producteur (`release_of`). | Mémoire de commandes épuisée, ou appareil en échec : l'écriture suivante de ce slot par CUDA attend une valeur qui ne viendra pas, et l'agrandissement se bloque sans message. | Faire remonter l'échec (`SlotFree` en refus, ou erreur au journal suivie de l'arrêt de la chaîne). | Lu. |
| A10-10 | Mineur | `src/app/play/placebo_screen.cpp:632` | Il reste des `path::string()` dans les instantanés (lignes 624 et 632 ; `placebo_snapshot.cpp:353`, `:360`, `:406`), contre la règle de la CI. | Windows, `--snapshot` vers un dossier dont le nom sort de la page de code : `std::filesystem::path::string()` lève une exception sur le thread de la fenêtre. | `core::to_utf8` pour le journal, comparaison de l'extension sur `path` (`u8string`). | Lu. |
| A10-11 | Mineur | `src/app/play/placebo_open.cpp:275` | Le cache des shaders est lu entier en mémoire sans plafond, alors que `max_total_size` vaut 64 Mo. Les `.part` d'une fermeture interrompue (nom unique, ligne 299) ne sont jamais purgés. | Un fichier de cache abîmé ou démesuré est lu entier à chaque lancement. Chaque plantage pendant l'écriture laisse un `.part` de plusieurs Mo dans le dossier de cache. | Refuser un fichier de plus de 64 Mo avant de le lire. À l'ouverture, effacer les `<cache>.*.part` plus vieux qu'une heure. | Lu. |
| A10-12 | Mineur | `src/app/play/present_level.h:426` | Commentaires faux ou abîmés : `present_level.h:426` et `placebo_screen.h:65` disent `open` par défaut, alors que le défaut est `wait` (`options.h:171`), et `ScreenOptions::present` vaut `Open` ; l'en-tête de `placebo_screen.h:1-6` parle d'« écran provisoire » ; le commentaire de `placebo_impl.h:166-167` est coupé en deux par clang-format. `watcher` est détruit implicitement après `pl_vulkan_destroy` (`placebo_open.cpp:350`), ce qui n'est sûr que parce que `WriteWatch` le détruit avant. | Un lecteur du code croit le retour éteint par défaut. Un appelant qui garde le veilleur jusqu'à la fermeture de l'écran détruit son sémaphore sur un `VkDevice` détruit. | Corriger les commentaires. Dans `~Impl`, `ring.set_watcher(nullptr); watcher.reset();` avant `pl_gpu_finish`. | Lu. |
| A10-13 | Mineur | `tests/unit/app/present_level_test.cpp` | Aucun test des chemins dégradés de la zone : perte du GPU dans `show` (A10-01), `PresentFeedback` (repli Timing vers Wait, absence de retours, domaines), veilleur mort (A10-03), `PlaceboRing` hors CUDA (création, repli RGB10A2 vers RGBA16, `retire`). `ring_interop_test.cpp` est caché (carte NVIDIA). | Une régression de GPU-06 ou du choix de niveau passe la CI. | `ring_watcher_test` avec un `Api` simulé. `PresentFeedback` derrière une interface de fonctions Vulkan simulables. `PlaceboRing` sur lavapipe (déjà utilisé par `ring_watcher_test`). | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-06 | Toujours présent | A10-01 : `placebo_screen.cpp:545-550` rend `presented = false`, `session.cpp:540` redemande, la boucle tourne à 1–4 ms (`session.cpp:598-616`), et aucun `pl_gpu_is_failed`. |
| GPU-10 | Toujours présent (latent pour l'attente, réel pour le pas de 4 ms) | A10-03 : `ring_watcher.cpp:474` sort sans journal ; `forget_all` (`:441`) ne voit pas un thread mort. Aujourd'hui, `WriteWatch` (`player_run.cpp:186`) détruit le veilleur avant toute destruction de l'anneau : l'attente sans fin n'est pas atteignable. |
| GPU-17 | Toujours présent | A10-04 : `placebo_open.cpp:323`, `placebo_ring.cpp:189` et `:242`, `placebo_offscreen.cpp:294`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 12 | `src/app/play/session.cpp:598-616` | Avec le niveau par défaut `Wait`, `vblank_grid()` est toujours vide : vérifier ce que `--vsync-index` (actif par défaut) et la PLL font alors sur la cible (voir A10-02). |
| 08 | `src/app/play/player_run.cpp:654` | Une erreur de `show` termine `run_chain` : quel message voit l'utilisateur, et que fait `play()` ensuite (réouverture en boucle sur un appareil perdu ?). |
| 18 | `src/app/play/placebo_open.cpp:327` | `share.reset()` libère l'appareil prêté à FFmpeg : un décodeur Vulkan Video qui garde une référence (`VulkanShare::device`) doit être détruit avant l'écran, sinon `pl_vulkan_destroy` détruit un `VkDevice` encore utilisé. Ordre à vérifier à la fermeture et à la réouverture. |
| 31 | `src/app/play/placebo_ring.cpp:218` | `ready()` rend vrai quand `vkGetSemaphoreCounterValue` échoue (appareil perdu) : la session croit l'image écrite et la compose. À relier à A10-01 et au chemin de perte du GPU de bout en bout (avec A01-01 côté CUDA). |
| 34 | `src/app/play/placebo_open.cpp:384-390` | Sous Windows seulement, la fenêtre reste cachée jusqu'à la fin de l'ouverture : si l'ouverture est refusée, vérifier que le message d'erreur s'affiche sans fenêtre. |

## Points positifs

- La fenêtre réduite ou cachée est traitée (`placebo_screen.cpp:537`) : aucune présentation FIFO qui attendrait un
  rappel qui ne vient pas, attente bornée à 50 ms.
- Les poignées exportées vers CUDA sont dupliquées côté CUDA (`ring_slot.cpp:22`) : pas de double fermeture du
  descripteur.
- La composition gardée (`compose_kept`) évite de recomposer grade et sous-titres quand l'image ne change pas.
- La fermeture cache la fenêtre avant le démontage du GPU, et le cache des shaders est écrit à côté puis renommé.
- Le choix du niveau de présentation est une fonction pure, testée.

## Questions ouvertes

- Le niveau `Wait` est-il voulu comme défaut alors qu'il ne date rien ? Sous CachyOS, `--present=wayland` donnerait
  la grille des vblanks (A10-02) : à mesurer contre `wait` sur la RTX 4070 Ti.
- Sur la cible, le pilote NVIDIA offre-t-il `present_wait2` sous Windows 11 et sous CachyOS ? Les deux lignes
  « présentation : niveau … » du journal suffisent.
- Faut-il, en cas de perte du GPU, rouvrir l'écran automatiquement, ou seulement afficher un message et fermer
  proprement ? Le rapport de l'audit 01 retient « détection et message, pas de reprise » : la même règle est
  supposée ici.
