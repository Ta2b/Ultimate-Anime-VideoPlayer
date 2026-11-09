# Rapport de l'audit 32. Budget d'une image

Branche : `claude/project-thread-gs1c4e`. Commit audité : `bec2f5b`. Date : `2026-11-09`.

## Verdict

Le chiffre qui borne la cadence dans les documents (13 à 16 ms de service par image, dont ≈ 11 ms hors réseau)
a été pris sur une chaîne qui n'existe plus (RTX HDR par image créée) : il ne se décompose pas, il se remesure,
et l'outil de mesure actuel ne voit ni l'écran ni les attentes entre CUDA et Vulkan. D'après les seuls chiffres
publiés du dépôt, la chaîne d'aujourd'hui devrait coûter 1,7 à 2,7 ms de GPU par image présentée à 1440p, pas
13 : si le lecteur plafonne encore vers 60 à 90 i/s, la cause est hors des noyaux, et rien ne permet de la
nommer sans les trois compteurs demandés en A32-02.

## Cadrage retenu

Pas de questions à Tatoub (consigne du coordinateur) : tout est audité au maximum, sur la cible RTX 4070 Ti,
écran 1440p 165 Hz, source 1080p 23,976 i/s, hauteur de travail 720p, moteur TensorRT. RIFE 4.25, RIFE 4.22
lite et DistilDRBA comptent autant ; ce qui empêche DistilDRBA d'être bien plus rapide que RIFE est cherché en
priorité. Le VRR est hors sujet. Toute baisse de qualité faite pour la vitesse est un défaut, sauf mesure qui la
justifie : elle est listée avec son gain. Deux points demandés en plus : la décomposition des ≈ 11 ms de GPU
hors réseau, et l'absence de test de qualité de bout en bout.

**Limite de cet audit** : la machine d'audit n'a ni carte NVIDIA ni les dépendances SDL (le script
`tools/deps/linux-from-source.sh` s'arrête à SDL3). Rien n'a été mesuré ici. Tout temps cité vient d'un
document du dépôt, avec sa référence ; toute somme est marquée « estimé ». Les preuves « à confirmer » donnent
la manipulation à faire sur le PC.

## Le chemin d'une image, et ce qu'on en sait

Unité : la paire de sources (41,7 ms à 23,976 i/s). À 165 Hz en mode Rapide, une paire donne ≈ 6,9
présentations, dont ≈ 5,9 images créées.

| Étape | Combien de fois par paire | Coût connu | Source du chiffre | Chronométrée aujourd'hui ? |
|---|---|---|---|---|
| Décodage NVDEC | 1 | non publié | | non (hors `gpu_profile`) |
| Ingestion (YUV vers RGB FP16, réduction à 720p, débandage) | 1 | 0,40 ms | `docs/handoff-pc.md:77` (avant le débandage) | oui, « ingestion » |
| Mesures de la paire (scores) + attente de l'hôte | 1 | 0,5 ms d'attente hôte | `docs/handoff-pc.md:78` | oui, « mesures de la paire » ; attente hôte journalisée |
| Entrée de RIFE (réduction, 7 plans) | 1 (RIFE) ; **2 par image créée** (DRBA) | 0,4 à 0,8 ms | `docs/reports/mesures-auto-4070ti-2026-10-28.md` | oui, « entrée de RIFE » |
| Réseau RIFE 4.25 fusionné, 1280 × 768 | 1 (2 si mouvement fort : une par instant) | 2,20 ms | rapport 03 complément, A03-13 | oui, « moteur RIFE fusionné » |
| Réseau RIFE 4.22 lite | 1 | **non publié à 720p** | | oui |
| Réseau DistilDRBA fusionné, 1280 × 768 | **1 par image créée** (≈ 5,9) | ≈ 0,45 ms | rapport 03 complément, A03-13 | oui |
| SPAN ×2, 720p vers 1440p, FP16 | 1 (0 sur un dessin tenu) | 4,1 ms | `docs/handoff-pc.md:77` | oui, « moteur … » |
| Warp et mélange à 1440p, écrit dans l'anneau | 1 par image créée | ≈ 0,2 ms (0,4 ms en 4K) | `docs/adr/0002` ramené aux pixels | oui, « warp et mélange » |
| Écriture de la source dans l'anneau | 1 | 0,09 à 0,12 ms | `docs/adr/0008-anneau-rgb10a2.md` | oui, « écriture du ring » |
| Passage CUDA vers Vulkan (sémaphores externes) | 2 opérations CUDA + 1 soumission Vulkan par image | **inconnu** | | **non** |
| Écran : passes libplacebo (grade, look, noir, kwin, sortie) | 1 par présentation (≈ 6,9) | 0,44 ms à 1440p pour l'ancien grade fusionné ; l'actuel, en 3 à 4 passes, **non publié** | `docs/handoff-pc.md:78` | une seule valeur globale côté écran, hors `gpu_profile` |
| Attente FIFO (`pl_swapchain_swap_buffers`) | 1 par présentation | dépend de l'écran | | non |

Somme estimée par paire, RIFE 4.25, paire calme : 0,4 + 0,6 + 2,2 + 4,1 + 5,9 × 0,2 + 0,1 + 6,9 × (0,44 à
1,5) ≈ **11,6 à 18,9 ms sur 41,7 ms**, soit 28 à 45 % d'un GPU, ou **1,7 à 2,7 ms par image présentée**.
Avec DistilDRBA : le réseau passe de 2,2 ms à 5,9 × 0,45 ≈ 2,7 ms, et l'entrée de 0,6 ms à 5,9 × 2 × (0,4 à
0,8) ≈ 4,7 à 9,4 ms : DRBA coûte **plus** que RIFE 4.25 à 165 Hz, et l'essentiel de l'écart vient de l'entrée,
pas du réseau (A32-04).

Ce que ce tableau dit du chiffre de 13 à 16 ms : il est cinq à huit fois au-dessus de la somme des étapes
connues. Les mesures d'origine (`docs/auto-libre.md:142-143,157,172`) ont été prises avec RTX HDR, une passe
Maxine par image créée, retirée depuis (`src/gpu/backend.h:60`). Aucun temps de RTX HDR n'a jamais été publié
(`docs/handoff-pc.md` : « les mesures prises avec RTX HDR ne se publient pas »). Les ≈ 11 ms hors réseau
étaient donc, très probablement, RTX HDR pour l'essentiel ; ce point est une déduction, pas une mesure.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A32-01 | Majeur | `docs/auto-libre.md:142-143,157,172` ; `src/gpu/backend.h:60` | Le « service de 13 à 16 ms » et les cadences de 60 à 67 i/s du mode Rapide viennent d'une chaîne avec RTX HDR par image créée, retirée depuis. Aucune mesure de la chaîne actuelle n'existe dans le dépôt. Les audits 03 et 21 et la fiche 32 raisonnent sur ce chiffre périmé. | Toute décision de cadence (plafond de l'Auto, choix du moteur, « DRBA plus rapide que RIFE ») s'appuie sur un coût qui n'est plus celui du lecteur. | Refaire la mesure sur le binaire livré, par moteur, et remplacer le tableau du §3 (procédure ci-dessous). Marquer l'ancien tableau « chaîne avec RTX HDR, historique ». | Lu. À confirmer : procédure « Remesurer » ci-dessous. |
| A32-02 | Majeur | `src/gpu/cuda/backend/tensorrt_backend.cpp:407-498` ; `src/gpu/cuda/backend/gpu_timing.cpp:29` ; `src/app/play/live_settings.h:164-186` | Le budget d'une image n'est pas observable. `gpu_profile` ne chronomètre que des étapes CUDA nommées ; il ne voit pas : le débandage seul, `frame_stats_start`, `flow_max`, le passage par sémaphores externes, tout le côté Vulkan (passes libplacebo une par une), l'attente FIFO, ni le délai entre « écrite » et « vue par main ». Le temps de service, lui, est un temps de mur : il additionne tout cela sans le séparer. Une étape chronométrée inclut le temps partagé avec les deux autres couloirs, et un échantillon est sauté quand une paire d'événements est déjà ouverte : les sommes ne sont pas additives et les médianes favorisent les moments calmes. | On lit « écart service 11 ms » et « warp 0,2 ms » sans pouvoir dire où sont les 10 ms restantes. C'est exactement la question posée à cet audit, et le code ne permet pas d'y répondre. | Trois compteurs, sans effet sur l'image : (1) par image, quatre instants hôte journalisés dans `<stats>` : soumise, écrite (vue par le veilleur), choisie par `main`, présentée ; (2) `pl_dispatch` par passe (libplacebo donne le temps de chaque passe, `pl_render_params.info_callback`) écrit dans `.gpu.tsv` ; (3) un événement CUDA autour de `cudaWait/SignalExternalSemaphoresAsync` (`ring_slot.cpp:154-171`). Ajouter la colonne « chevauchement » : somme des étapes ÷ temps de mur. | Lu. |
| A32-03 | Majeur | `tests/unit/` ; `tools/bench/rife_triplets.py` ; `tools/ring/banding.py` | Aucun test ne compare la sortie du lecteur à un original. Les seuls PSNR du dépôt comparent un moteur à son voisin (FP8 contre FP16, fusionné contre non fusionné) ou un ONNX seul hors du lecteur. Une baisse de qualité dans la chaîne (réduction à 720p, warp bilinéaire après SPAN, anneau 10 bits, FP8) n'a donc aucune porte chiffrée. | Un changement qui dégrade l'image mais garde « moteur A ≈ moteur B » passe toute la CI. La règle de Tatoub (ne jamais entacher la qualité sans mesure) ne peut pas être vérifiée. | Trois bancs, décrits dans « Test de qualité de bout en bout » ci-dessous. Le troisième (transparence) tourne en CI sur lavapipe ; les deux autres sur le PC, avec un seuil par moteur inscrit dans le dépôt. | Lu (recherche de PSNR, SSIM et « original » dans `tests/` et `tools/`). |
| A32-04 | Majeur | `src/app/play/upscale_drba.cpp:179-256` | DistilDRBA refait, pour **chaque image créée**, deux `rife_input` complets (réduction des deux images et assemblage des 7 plans, pour les paires 0-1 et 1-2) puis une passe réseau. Les trois images réduites ne changent pas d'un instant à l'autre : seul le plan du temps change. À 165 Hz (≈ 5,9 images créées par paire) cela fait ≈ 12 préparations d'entrée par paire là où RIFE en fait une. D'après les chiffres publiés (0,4 à 0,8 ms par entrée), l'entrée coûte alors plus que le réseau lui-même. C'est, avec A03-13 (une passe par image), ce qui empêche DRBA de battre RIFE. | Anime en ×5 à ×7 sur écran 165 Hz : DRBA paie ≈ 4,7 à 9,4 ms d'entrées par paire contre 0,6 ms pour RIFE 4.25, pour un réseau de ≈ 2,7 ms contre 2,2 ms. | Préparer une fois par paire les images réduites (elles sont déjà gardées : `small_before`, `small_after`), et n'écrire par instant que le plan du temps (un remplissage, quelques microsecondes) ; ou sortir le temps du tenseur d'entrée (entrée scalaire du moteur fusionné). Sans effet sur l'image. Puis reprendre A03-13 : exporter DRBA avec flot et masque réutilisables pour plusieurs instants, à qualité vérifiée par le banc A32-03. | Lu. À confirmer : `.gpu.tsv`, lignes « entrée de RIFE » (nombre d'appels et total par seconde) avec `--rife-model=` sur DistilDRBA puis sur RIFE 4.25, `--rife=6`. |
| A32-05 | Mineur | `src/render/grade.cpp:355` ; `src/render/shaders/anime-hdr-anime/3-anime.hook:1` ; `anime-hdr-look.hook:74` ; `anime-hdr-black.hook:97,136,208` ; `src/app/play/placebo_screen.cpp:156-181` | À chaque présentation (165 par seconde, pas 24), l'écran enchaîne jusqu'à quatre passes plein format en FP16 sur OUTPUT, toutes ponctuelles ou presque : grade (`3-anime`), look (LUT), vrai noir (quand `black_hi > black_lo`, donc sur OLED), `kwin` (sous KDE), plus la passe de sortie de libplacebo. Le commentaire `anime-hdr-look.hook:100` parle encore d'un « grade fusionné » qui lisait `ajl_look` dans la même passe : il n'existe plus. L'ancien grade en une passe coûtait 0,44 ms à 1440p (`docs/handoff-pc.md:78`) ; le coût actuel n'est pas publié. | Écran HDR OLED sous KDE (le poste de Tatoub) : 4 lectures et écritures de 3,7 Mpx FP16 par présentation, soit ≈ 2,4 Gpx/s de trafic de plus qu'en une passe. Estimé 0,5 à 1 ms par présentation, soit 80 à 165 ms de GPU par seconde. | Fusionner look, noir et kwin dans la passe `3-anime` (mêmes pixels, mêmes calculs, même résultat au bit près tant que les cibles intermédiaires sont en FP16 ; `3-anime` lit deux voisins à 2 px de l'image d'entrée, ce qui reste vrai en tête de passe). Les cartes 1/8 et 1/32 ne changent pas. Garder les passes séparées comme repli. | Lu. À confirmer : temps par passe (compteur 2 de A32-02), avant et après, et capture `--snapshot` identique. |
| A32-06 | Mineur | `src/gpu/cuda/backend/warp_kernels.cu:100-170` | Le warp lit ses entrées en plans séparés : par pixel de sortie, une vingtaine de lectures de texture (flot sur 4 plans en 2 positions, masque, 2 images × 3 plans), 2 `logf` et 1 `expf`. Des textures entrelacées (flot RGBA16F, image RGBA16F) ramèneraient cela à 5 lectures. Le noyau tourne ≈ 140 fois par seconde à 1440p à 165 Hz. | Chaque image créée ; le coût croît avec la sortie (0,4 ms en 4K selon l'ADR 0002, donc ≈ 60 ms/s en 4K 165 Hz). | Entrelacer une fois par paire (SPAN sort déjà en plans : une copie de 3,7 Mpx par source, à peser contre le gain), ou garder tel quel si la mesure donne moins de 0,05 ms de gain. Même image. | Lu. À confirmer : ligne « warp et mélange » du `.gpu.tsv`, avant et après. |
| A32-07 | Mineur | `src/app/play/placebo_ring.cpp:214-218` ; `src/app/play/session.cpp:240,315-347,393,604` | `ready()` appelle le pilote (`vkGetSemaphoreCounterValue`) à chaque question, et un tour de boucle la pose plusieurs fois pour la même image (choix, prise, attente, tête de file) : 5 à 8 appels noyau par tour, à 165 tours par seconde et plus. La valeur d'un sémaphore à compteur ne fait que monter : une image vue prête le reste. | Lecture normale ; le coût est côté hôte (fil principal), pas GPU. | Garder par emplacement la dernière valeur lue et ne rappeler le pilote que si elle est sous la valeur attendue. | Lu. À confirmer : `perf` ou ETW sur le fil principal (part de `vkGetSemaphoreCounterValue`). |
| A32-08 | Mineur | `src/core/free_rate.cpp:27-36,80` | Le pas du mode Rapide vient du temps de service lissé × 1,1 × pression. Ce temps de service inclut ce que A32-02 ne sépare pas (réveil du veilleur, attente du fil principal, FIFO). Un coût hôte y devient une « limite GPU » : la cadence baisse alors que le GPU a de la marge, et l'affichage « limite GPU 78 % » le dit à tort. | Fil principal retenu 4 ms par `pl_swapchain_swap_buffers` (FIFO pleine) pendant qu'une image finit : son « service » grossit d'autant, le pas s'allonge. | Mesurer le service côté GPU (événements CUDA du début de la première étape à la fin de l'écriture, déjà disponibles par `GpuSpan`), et garder le temps de mur comme garde-fou seulement. | Lu. À confirmer : comparer `service` journalisé et la durée `GpuSpan` de la même image. |
| A32-09 | Mineur | `docs/handoff-pc.md:77-78` ; `docs/adr/0002` ; `docs/adr/0008-anneau-rgb10a2.md` | Les temps de référence par étape datent de chaînes différentes (RIFE 4.26 non fusionné, grade fusionné, warp avant l'ADR 0002 à 0,06 ms). Aucun document ne donne le budget de la chaîne actuelle sur une même mesure. RIFE 4.22 lite, le moteur que l'Auto calibré impose (`src/app/play/auto_config.cpp:204`), n'a aucun temps publié à 720p. | Qui reprend le dépôt additionne des chiffres incompatibles (ce rapport y est contraint). | Un seul tableau daté dans `docs/`, produit par la procédure ci-dessous, et les anciens marqués historiques. | Lu. |

Aucun défaut « Bloquant ». Pas de test rouge joint : les défauts majeurs sont une mesure absente (A32-01,
A32-02), un test absent (A32-03) et un coût à confirmer sur la RTX (A32-04) ; aucun ne se prouve par un test
unitaire compilable ici.

### Remesurer (A32-01, A32-04, A32-05)

Sur le PC, binaire livré (`linux-release` avec CUDA, ou `windows-release`), écran en 165 Hz fixe, rien d'autre
sur le GPU, même extrait que `docs/auto-libre.md` §3 (Re:Zero S4 ép. 18 à 960 s, 75 s) :

1. Pour chaque moteur (RIFE 4.25, RIFE 4.22 lite, DistilDRBA) : une lecture en mode Rapide avec `--stats=<fichier>` et `--rife-model=`,
   puis une avec un multiplicateur fixe (`--rife=6`) pour comparer à nombre d'images égal.
2. Relever dans `<fichier>.gpu.tsv`, par étape : nombre d'appels par seconde, médiane, et total par seconde
   (appels × médiane). Le total par seconde est le seul chiffre comparable entre moteurs.
3. Refaire avec `--one-lane` : la différence avec les trois couloirs donne le chevauchement.
4. Relever « écart service », « limite GPU » et la cadence de l'OSD, et l'occupation GPU (`nvidia-smi dmon`).
5. Résultat attendu si l'estimation de ce rapport tient : service de 2 à 3 ms par image, GPU sous 50 % à
   165 i/s avec RIFE 4.25. Si le service reste au-dessus de 6 ms, le coût est hors noyaux : poser les compteurs
   de A32-02 avant toute optimisation.

### Test de qualité de bout en bout (A32-03)

Les briques existent : `--snapshot=<f>.pq.ppm`, `--snapshot-at`, `--snapshot-frames=N`, `--keys`, le mode hors
écran, la référence CPU `src/kernels/ref`, l'export qui rejoue la chaîne.

1. **Interpolation contre de vraies images.** Une source à mouvement réel (anime en 1 dessin par image, ou une
   séquence 48 i/s) dont on retire une image sur deux. Lecture en ×2 fixe, captures des images créées, PSNR et
   SSIM contre les images retirées, par moteur. C'est la seule mesure qui dit si DRBA « à qualité similaire »
   est vrai dans le lecteur (warp après SPAN, hauteur de travail), et non sur l'ONNX seul.
2. **Agrandissement contre un original.** Un master 1440p réduit à 720p (filtre fixé), passé dans la chaîne,
   comparé au master. Variantes : SPAN FP16 contre FP8, anneau RGBA16 contre RGB10A2 tramé, hauteur de travail
   720p contre 1080p. Chaque baisse faite pour la vitesse reçoit ainsi un chiffre.
3. **Transparence.** Chaîne sans SPAN, sans interpolation, sans grade (sortie SDR) : la capture doit valoir la
   référence CPU à 1 code près. Ce banc tourne en CI sur lavapipe avec le backend Vulkan, et attrape toute
   perte introduite dans l'ingestion, l'anneau ou la présentation.

## Baisses de qualité faites pour la vitesse

Rien de nouveau trouvé au-delà du rapport 49, mais le budget ci-dessus change la lecture de plusieurs d'entre
elles : si la chaîne coûte bien 1,7 à 2,7 ms par image, la marge existe pour les lever.

| Baisse | Où | Gain connu | Justifiée par une mesure ? |
|---|---|---|---|
| Hauteur de travail 720p pour une source 1080p (réduction avant SPAN) | ingestion ; rapport 49 | SPAN 4,1 ms contre 11,5 ms en 1080p vers 2160p (`docs/handoff-pc.md:77`), soit ≈ 7 ms par source | Gain mesuré ; perte **jamais mesurée** (banc 2 de A32-03) |
| Warp bilinéaire à la taille de sortie, après SPAN (une seule passe SPAN par source) | `docs/adr/0002` ; A49-06 | ≈ 4 ms de SPAN par image créée évitée | Gain mesuré ; perte jugée à l'œil seulement. À 165 Hz en Rapide, ≈ 6 images affichées sur 7 sont des warps : c'est le cas le plus exposé |
| Anneau RGB10A2 tramé au lieu de RGBA16 | `docs/adr/0008-anneau-rgb10a2.md` (point 5) | −50 % de mémoire d'anneau, écriture −25 % visés | **Non** : le défaut a changé sans la porte 5.5c, l'ADR le dit ; `tools/ring/banding.py` reste à passer |
| Auto calibré : SPAN en `Performance`, RIFE 4.22 lite imposé | `src/app/play/auto_config.cpp:202-204` ; A49-03 | non publié | **Non** |
| SPAN réutilisé sur un dessin tenu | `src/app/play/upscale_stage.cpp` ; A49-11 | 4,1 ms par source tenue | Sans perte si le dessin est vraiment identique ; le seuil est jugé par A49-11 |
| Source logicielle convertie en bilinéaire (`SWS_BILINEAR`) | `src/media/frame_data.cpp:257` ; A15-06 | non mesuré | **Non** |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| WIN-04 | Toujours présent | `src/app/play/player_run.cpp:397-414,534-546` : les faits d'écran sont relus toutes les 500 ms (`kDisplayPollUs`, ligne 89) sur le fil principal, `on_battery()` compris. Voir A11-02. |
| GPU-16 | Toujours présent | `src/app/play/host_facts.cpp:52,88-95` : `display_period_us` appelle `display_refresh_hz`, et la relecture appelle les deux : `QueryDisplayConfig` deux fois par relecture sous Windows. Coût hôte, non mesuré. |
| MED-03 | Toujours présent | `src/media/frame_data.cpp:247-259` : `to_chain_yuv` alloue une image (`av_frame_alloc`) par image convertie, en `SWS_BILINEAR`. Sur le fil de la source, hors budget GPU. |
| TRT-08 | Toujours présent | `src/app/play/auto_config.cpp:202-204` : l'Auto calibré pose `PrecisionTier::Performance` dès qu'un modèle SPAN est calibré, alors que le commentaire des lignes 167-169 dit que le FP8 ne se choisit qu'à la main. « FP8 non servi par l'Auto » est donc à relire avec l'audit 07. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 03 | `src/app/play/upscale_drba.cpp:179-256` | A32-04 complète A03-13 : l'entrée de DRBA, pas seulement son nombre de passes. |
| 07 | `src/app/play/auto_config.cpp:167-169,202-204` | Commentaire et code se contredisent sur le FP8 dans l'Auto (TRT-08). |
| 10 | `src/render/shaders/anime-hdr-look.hook:100` | Commentaire périmé (« passe du grade fusionné ») ; `src/app/play/placebo_grade.cpp:327` et `placebo_screen.cpp` posent `pl_render_default_params` (tramage et filtres par défaut à relire avec la sortie 10 bits). |
| 12 | `src/core/free_rate.cpp:27` | Une paire de moins de 2 images n'entre pas dans le coût de service : en ×2 le mode Rapide n'a aucune mesure et garde sa dernière estimation. |
| 21 | `src/gpu/gpu_profile.h` ; `src/gpu/kernel_ops.h` | `gpu_profile` n'a pas d'équivalent demandé au backend Vulkan pour l'écran ; le contrat ne dit pas que les durées se chevauchent entre couloirs. |
| 28 (documents) | `docs/auto-libre.md` §3 ; `docs/handoff-pc.md:77-78` | Tableaux de temps à dater et à marquer historiques (A32-01, A32-09). |
| 49 | `docs/adr/0008-anneau-rgb10a2.md` point 5 | La porte de qualité de l'anneau 10 bits n'a jamais été passée. |

## Points positifs

- Le warp écrit directement dans l'emplacement de l'anneau à la taille de sortie : aucune copie intermédiaire
  par image créée (`tensorrt_warp.cpp:142`).
- RIFE fusionné : une passe par paire avec poses, les instants suivants ne coûtent qu'un warp ; c'est la bonne
  forme, et celle que DRBA devrait rejoindre.
- Trois couloirs CUDA avec leurs dépendances déclarées (`src/gpu/lanes.h`) : SPAN d'une source ne retient pas
  l'interpolation de la précédente.
- Aucune copie hôte par image sur le chemin NVDEC ; le veilleur d'anneau réveille le fil principal au lieu d'un
  sondage serré.
- Interface, sous-titres et incrustations ne sont redessinés et renvoyés au GPU qu'à leur changement
  (`placebo_overlays.cpp`, `ui.cpp:463`) : coût nul sur une image ordinaire.
- La netteté n'est pas posée à force 0 (`sharpen.hook`, `placebo_screen.cpp:158`) : pas de passe pour rien.
- Les cartes du grade sont en 1/8 et 1/32 : leur coût est négligeable, seul le plein format compte (A32-05).

## Questions ouvertes

1. Le plafond de cadence observé aujourd'hui en mode Rapide, sans RTX HDR, n'est pas connu : aucun relevé
   postérieur au retrait. La procédure « Remesurer » y répond en une séance.
2. Le partage de temps entre le contexte CUDA et le contexte Vulkan du même processus (deux contextes sur un
   GPU) peut coûter des millisecondes de commutation par image sous charge. Hypothèse, sans preuve dans le code :
   le compteur 3 de A32-02 et `nsys` la tranchent.
3. Si la mesure confirme 1,7 à 2,7 ms par image, faut-il rendre la hauteur de travail 1080p (SPAN 1080p vers
   2160p, 11,5 ms par source) quand le GPU le permet ? C'est une décision de qualité qui revient à Tatoub, à
   prendre avec le banc 2 de A32-03.
