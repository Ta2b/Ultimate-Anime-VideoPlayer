# Complément de l'audit 32. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-hsapcw`. Commit audité : `bdfe44c`. Date : `2026-11-09`.

## Verdict

Les quatre renvois sont confirmés au moins en partie et donnent quatre défauts mineurs (A32-10 à A32-13). Aucun
ne baisse la qualité d'image : tous coûtent du temps (GPU ou hôte) sans rien rendre. Deux parties de renvois
sont des doublons (les passes HDR séparées sont A32-05 ; l'attente sur une seule voie est A21-06) et une est
réfutée (l'inverse de la matrice LMS est une expression constante, pas un calcul par pixel du look).

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les quatre renvois de `audits-docs/complements/32.md`, sans
question à Tatoub, sans toucher au code. Les ID suivent le rapport 32, qui s'arrête à A32-09. La machine
d'audit n'a pas de carte NVIDIA : tout est lu, rien n'est mesuré ; les gains sont marqués « estimé » avec leur
calcul, et la preuve « à confirmer » donne la manipulation sur la RTX 4070 Ti.

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 31 | `src/gpu/cuda/backend/tensorrt_export.cpp` | Confirmé : **A32-10** | Le fichier cité n'est pas en cause (l'export n'alloue que l'image NVENC et ses barrières) ; le défaut est dans tout le backend : aucune lecture de la mémoire libre avant de charger les moteurs ou d'allouer la chaîne, aucune alerte quand elle est pleine. |
| 30 | `src/gpu/cuda/backend/tensorrt_lanes.cpp:140-150` | Confirmé : **A32-11** | `wait_event` interroge 200 µs avec `yield`, puis `cudaEventSynchronize` ; aucun `cudaSetDeviceFlags`, aucun événement créé avec `cudaEventBlockingSync`. L'attente au-delà de 200 µs tourne à vide. |
| 30 | `src/app/play/upscale_stage.h:500` | Confirmé en partie : **A32-12** ; le reste est doublon de **A09-19** (attente non chronométrée) et de **A21-06** (une voie) | DistilDRBA relit sa mesure dans la paire même où il l'enfile ; RIFE une paire plus tard. |
| 41 | `src/render/shaders/anime-hdr-anime/3-anime.hook:60-130` | « Trois ou quatre passes » : doublon de **A32-05**. « Constantes recalculées par pixel » : confirmé, **A32-13**. « Inverse de la matrice LMS » : réfuté pour le look, voir plus bas. | |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A32-10 | Mineur | `src/gpu/cuda/backend/trt_engine.cpp:140-165` ; `src/gpu/cuda/backend/tensorrt_kernels.cpp:228-235` ; `src/app/play/chain_stats.h:54-57` ; `src/app/play/gpu_telemetry.cpp:357-358` | Sous Windows, la politique du pilote « CUDA - Sysmem Fallback Policy » (par défaut depuis le pilote 536.40) laisse `cudaMalloc` et TensorRT réussir quand la mémoire vidéo est pleine, en plaçant le surplus en mémoire système par le PCIe. Le lecteur ne lit la mémoire libre (`cudaMemGetInfo`) que pour l'OSD (`device_memory`) et pour la dérivation FP8 de fond (`tensorrt_span_derive.cpp:107-116`) ; ni le chargement des moteurs (`reserve(getDeviceMemorySizeV2())`), ni les tenseurs de la chaîne, ni l'export ne la regardent. Une chaîne qui déborde n'est donc ni refusée ni signalée : elle tourne plusieurs fois plus lentement, et la cadence Auto ou Rapide la prend pour une limite GPU (A32-08). | Carte de 6 à 8 Go (RTX 3050, 4060, 4060 portable), sortie 4K, DistilDRBA fusionné et SPAN, avec un navigateur ouvert : la VRAM atteint son total, les zones d'activation passent en mémoire système, chaque passe paie le PCIe. L'OSD affiche « VRAM 8,0 / 8,0 Go » sans autre avis ; le journal ne dit rien. Hors de la cible (12 Go à 1440p), mais c'est le cas des cartes visées par la voie unique (`tensorrt_backend.h:139`). | (1) À l'ouverture de la chaîne, après le chargement des moteurs et l'allocation des tenseurs, relire `cudaMemGetInfo` ; si le libre tombe sous une marge (256 Mio par exemple) ou si `used` ≥ `total`, journaliser « mémoire vidéo pleine : le pilote peut déborder en mémoire système » et le montrer une fois à l'écran (`visible_error`). (2) Faire la même lecture dans le fil de l'OSD à 2 Hz, qui la fait déjà, et lever l'avis au premier dépassement. (3) Facultatif : poser pour `ajp.exe` le réglage de profil « Prefer No Sysmem Fallback » par NVAPI DRS à l'installation, ce qui transforme la lenteur en refus `cudaErrorMemoryAllocation` que A31-08 sait déjà remonter. | Lu. À confirmer : sous Windows, sur une carte de 8 Go, lancer une 4K avec DistilDRBA et SPAN plus un banc qui prend 2 Go de VRAM ; comparer `--gpu-profile` avant et après, et constater qu'aucun avis ne sort. |
| A32-11 | Mineur | `src/gpu/cuda/backend/tensorrt_lanes.cpp:17-19,140-150` ; `src/gpu/cuda/backend/host_upload.cpp:43,66,114` ; `src/gpu/cuda/backend/tensorrt_rife.cpp:179,218` | `wait_event` interroge l'événement 200 µs en `yield`, puis appelle `cudaEventSynchronize`. Le contexte garde l'ordonnancement par défaut (`cudaDeviceScheduleAuto` : attente active tant que les contextes actifs ne dépassent pas les cœurs logiques) et les événements sont créés avec `cudaEventDisableTiming` seul, sans `cudaEventBlockingSync`. Au-delà de 200 µs, le fil gpu tourne donc à vide sur un cœur jusqu'à la fin du travail attendu. Même chose sans le délai pour `host_upload` (décodage logiciel) qui appelle `cudaEventSynchronize` directement. Les attentes concernées : `scores_take` à chaque paire, `flow_max_take` en mouvement fort, `hold` quand le GPU a `kHeldFrames` images de retard. | Décodage logiciel (H.264 10 bits, AV1 sans NVDEC sur une carte ancienne) sur un CPU à 6 cœurs : FFmpeg veut tous les cœurs, le fil gpu en brûle un à chaque attente longue (une voie, A21-06 ; DistilDRBA, A32-12). Le décodage ralentit, la source arrive en retard, l'Auto baisse la cadence. Sur le PC de Tatoub en NVDEC, l'effet est surtout de la chaleur et de la consommation CPU. | Créer les événements attendus par l'hôte (`PairRead::done`, `FlowRead::done`, `Held::done`, `host_upload` `copied` et `read`) avec `cudaEventDisableTiming \| cudaEventBlockingSync`. Garder les 200 µs d'interrogation de `wait_event` : le cas courant (mesure déjà finie) reste sans réveil ; seul le cas long dort. Ne pas passer par `cudaSetDeviceFlags`, qui doit précéder la création du contexte primaire (FFmpeg et le décodeur peuvent l'avoir créé avant). | Lu. À confirmer : `--one-lane`, DistilDRBA, source H.264 10 bits en décodage logiciel ; `perf top` ou le gestionnaire des tâches montrent un cœur à 100 % dans `libcuda` ; après la correction, comparer la cadence de décodage. |
| A32-12 | Mineur | `src/app/play/upscale_drba.cpp:124-157,224-229,247-252` ; `src/app/play/upscale_stage.cpp:528-536,630` ; `src/app/play/upscale_ahead.cpp:62-70,96-108` | RIFE enfile la mesure (image en cours, suivante) avant RIFE et SPAN de la paire (`look_ahead`, `upscale_stage.cpp:535`) et ne la relit qu'au début de la paire **suivante** (`pair_scores`) : il a une paire entière de marge. DistilDRBA enfile la même mesure dans `drba_next` (deux ingestions de l'image d'après, plein format et copie réduite, puis `pair_diff` au format de travail), et la relit **dans la même paire**, au premier instant du côté b (`settle_after_scores`, appelé par `drba_fate` puis `interpolate_drba`) : sa marge n'est que l'enfilage des instants du côté a, quelques dizaines de µs quand l'anneau a des cases libres. Sur trois voies, l'attente couvre les deux ingestions et la mesure (estimé 0,5 à 0,9 ms : 0,40 ms par ingestion, `docs/handoff-pc.md:77`). Sur une voie (A21-06), elle couvre aussi toute la fin de la paire précédente encore en file. Pendant ce temps le fil gpu n'enfile rien : la file se vide à chaque paire et le GPU attend le réveil de l'hôte (plus le cœur brûlé, A32-11). C'est le seul point du chemin où DistilDRBA attend plus que RIFE, et il n'est pas mesuré (A09-19 : `scores_wait_us_` ne chronomètre que `drba_scores`, pas `settle_after_scores`). | Rapide à 165 Hz, DistilDRBA, après une coupe ou un saut (anneau vide) : à chaque paire, le fil gpu attend l'ingestion de l'image d'après avant d'enfiler le côté b. Avec une seule voie (carte de 6 Go), l'attente couvre aussi le SPAN et les passes encore en file : la file ne dépasse jamais une demi-paire, et la cadence obtenue est sous ce que le GPU tiendrait. | (1) Corriger A09-19 d'abord : chronométrer `settle_after_scores` dans `scores_wait_us_`, avec la paire et le côté. (2) Si la médiane dépasse 0,2 ms sur la RTX : enfiler `drba_next` une paire plus tôt (ingérer et mesurer l'image d'après dès que la file `pending_` en tient trois), ce qui coûte une image source de latence (≈ 42 ms à 23,976 i/s), pas de débit ; DistilDRBA attend déjà la suivante (`wants_next`). (3) Mesurer sur la copie réduite quand elle existe, comme RIFE (`upscale_ahead.cpp:96`) : moins de lecture, même seuil (le score est rapporté à l'aire). | Lu. À confirmer : après (1), comparer la médiane et le p95 de l'attente entre RIFE 4.25 et DistilDRBA, trois voies contre `--one-lane`, sur la même source. |
| A32-13 | Mineur | `src/render/shaders/anime-hdr-anime/3-anime.hook:27,86,92,97,104` | La passe du grade, une fois par pixel de sortie et par présentation (165 par seconde), calcule quatre fois `pq_of` sur des valeurs qui ne dépendent que des paramètres : `pq_of(0.55 * white)`, `pq_of(0.95 * white)` (ligne 97), `pq_of(0.6 * white)`, `pq_of(white)` (ligne 104). Chaque `pq_of` passe par `nits_to_pq` en `vec3` : deux `pow` vectoriels, soit 6 `pow` scalaires pour un seul nombre utile. S'y ajoutent `pq_of(y)` (ligne 86) et `pq_to_nits(vec3(local_pq)).x` (ligne 92), deux calculs scalaires faits sur trois canaux. Les compilateurs de shaders ne sortent pas de la boucle des pixels un calcul qui ne dépend que des uniformes. Bilan estimé : 24 `pow` par pixel pour les constantes et 8 pour les canaux inutiles, sur ≈ 73 `pow` et 18 `exp` que compte la passe. | Écran HDR 1440p à 165 Hz : 3,7 Mpx × 165 = 608 Mpx/s. À deux opérations transcendantes par `pow`, la passe en demande ≈ 170 par pixel, dont ≈ 64 évitables. Sur les ≈ 2,5 Topérations/s des unités spéciales d'une 4070 Ti (60 SM × 16 × 2,6 GHz), estimé ≈ 0,25 ms par présentation pour la passe, dont ≈ 0,09 ms évitable, soit ≈ 15 ms de GPU par seconde. Aucun effet sur l'image. | Calculer les quatre seuils sur l'hôte, en double, quand les paramètres changent (`Grade::set_params`, `grade.cpp`), et les passer en `//!PARAM` (`an_gate_lo`, `an_gate_hi`, `an_spark_lo`, `an_spark_hi`) ; écrire `pq_of` et `pq_to_nits` en scalaire (`float`) pour `y` et `local_pq`. Le résultat ne bouge qu'à l'arrondi du `float` (≤ 1 ULP sur les seuils). Gagne à se faire avec la fusion de A32-05. | Lu. À confirmer : `grade_gpu_us` (`placebo_screen.cpp:184`) avant et après sur la RTX, et capture `--snapshot` identique à 1 code près. |

## Points vérifiés sans défaut

- **Inverse de la matrice LMS** (renvoi 41) : le look (`anime-hdr-look.hook:120-147`) n'a ni LMS ni `inverse()` ;
  ses matrices sont des `const mat3` écrites en clair (lignes 80-85). L'`inverse(AJK_TO_LMS)` est dans la passe
  kwin (`anime-hdr-black.hook:175`), sur une `const mat3` : une expression constante, que glslang ou le pilote
  replient à la compilation dans le cas courant. Pas de défaut retenu ; si l'on veut en être sûr, écrire la
  matrice inverse en clair ne coûte rien.
- **`skin_like(lin)` appelée deux fois** (`3-anime.hook:101,114`) : fonction pure aux mêmes arguments, que le
  compilateur fusionne. Pas de défaut.
- **Les quatre lectures voisines à 2 px** (`3-anime.hook:109-112`) passent chacune par `pq_to_nits` complet : il
  le faut, la luminance se calcule en linéaire. Pas de défaut.
- **L'export** (`tensorrt_export.cpp`) ne fait pas d'attente hôte en régime : il signale la fin par
  `cuStreamWriteValue64` ou par un rappel hôte (`cudaLaunchHostFunc`, ligne 198) ; les
  `cudaStreamSynchronize` des lignes 75, 236 et 251 sont à la préparation et à la fermeture.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A32-05 | Toujours présent | Les passes séparées (anime, look, noir, kwin) sont toujours posées en suite (`grade.cpp:355`). A32-13 s'y ajoute. |
| A21-06 | Toujours présent | `tensorrt_rife.cpp:203,218` : l'événement est posé derrière la mesure, sur la voie courante. A32-12 en est le cas le plus fréquent avec DistilDRBA. |
| A09-19 | Toujours présent | `upscale_stage.cpp:391-402` : seul `drba_scores` est chronométré. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 09 | `src/app/play/upscale_drba.cpp:95-107` | La correction de A09-19 conditionne la mesure de A32-12 : chronométrer `settle_after_scores` avec le côté (a ou b) de la paire. |
| 41 | `src/render/grade.cpp` (`set_params`) | Les seuils de A32-13 deviennent des paramètres calculés sur l'hôte : à reporter dans les tests du grade (même sortie à l'arrondi près). |

## Points positifs

- L'interrogation de 200 µs avant l'attente bloquante est le bon compromis : la correction de A32-11 ne la retire pas.
- La mesure en avance de RIFE (`upscale_ahead.cpp`) laisse une paire entière de marge ; c'est le modèle à suivre
  pour DistilDRBA.
- `device_memory` lit la VRAM sur un fil à part, hors du fil d'affichage : l'avis de A32-10 s'y ajoute sans coût.

## Questions ouvertes

Aucune.
