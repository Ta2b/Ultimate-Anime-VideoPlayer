# Temps des noyaux Vulkan du socle sur la RTX 4070 Ti

Banc `ajp_vk_kernels_bench` (branche `pc/vk-compute-bench`, `tools/bench/vk_kernels/`, option CMake
`AJP_BUILD_BENCH`), mesuré le 19 octobre 2026 vers 19 h 50, sous le verrou GPU (`ajp-gpu`).
Sorties brutes des trois passages et état de la carte : `~/ajp-reports/vk-compute-bench/`.

## En bref

- Les noyaux du socle sont petits devant le budget d'image. Une chaîne de 1080p à 4K coûte environ
  0,9 ms par source en f16 (ingestion, coupe de scène, dessins tenus, remplissage et constantes
  RIFE, recadrage, sortie RGBA16, PQ). C'est environ 2,6 % de la cible de 35,4 ms
  (plan.txt:629). Les réseaux TensorRT ne sont pas comptés.
- Le warp de S3 (images 1440p, mouvement estimé à 720p) prend 0,227 ms en f16 et 0,408 ms en
  f32 par instant. Le plan compte 1,8 ms pour « déformer et mélanger à 1440p » (plan.txt:552).
- À 4K, les noyaux à une seule passe atteignent 72 à 88 % des 504 Go/s de la carte (transfer,
  output_rgba16, rife_crop, rife_constants, warp_blend).
- Quatre noyaux restent loin du débit mémoire (19 à 37 % en octets obligatoires) :
  - ingest et output_yuv : leurs intermédiaires float32 font environ 75 % du trafic ;
  - resize : son intermédiaire de lignes est en float32, même en f16 ;
  - scene_diff et block_diff : un échantillon de 8 bits par invocation, puis une réduction par
    barrières.
  Détails et hypothèses plus bas.

## Conditions de mesure

- RTX 4070 Ti, pilote NVIDIA 615.78.08, Vulkan 1.3. Mémoire à 10501 MHz (P0), soit 504 Go/s
  théoriques. SM entre 2805 et 2835 MHz, 52 à 67 °C, 145 à 204 W. Relevés `nvidia-smi` avant,
  pendant et après : `rtx-state.txt`.
- Chaque cas : 20 passes d'échauffement écartées, puis 100 passes mesurées (le banc refuse moins
  de 50). Une exécution à la fois, avec une attente hôte entre deux passes.
- Mesure par requêtes timestamp (64 bits, 1 ns par tic), dans `GpuTimer` (gpu_timer.h:1-16) :
  1. une porte (sémaphore timeline signalé par l'hôte) retient les trois soumissions jusqu'à ce
     que toutes soient en file, pour que la latence de l'hôte reste hors de la mesure ;
  2. ouverture : vidage du L2, timestamp 0, puis une barrière ALL_COMMANDS qui retient
     l'enregistrement (pas d'attente de sémaphore entre les deux timestamps) ;
  3. l'enregistrement du noyau, inchangé ;
  4. clôture : timestamp 1 en BOTTOM_OF_PIPE.
- Le cas « plancher » (un remplissage de 4 o) mesure le coût fixe de cette mécanique :
  0,0106 ms. Il est inclus dans chaque temps, sans être retranché.
- Vidage du L2 : 64 Mio écrits avant chaque passe (`--flush-mio 64`), soit plus que les 48 Mo de
  L2. Le cache est alors plein de lignes sales, comme après l'étape précédente de la chaîne. La
  dernière colonne reprend la médiane sans vidage (`--flush-mio 0`) :
  - à 1080p, les tenseurs f16 (12 à 25 Mo) tiennent dans le L2 et le temps chute de 2 à 3 fois ;
    le coût réel dans la chaîne se situe entre ces deux colonnes ;
  - à 4K, les deux colonnes se rejoignent.
- Octets comptés (« Mo ») : entrées et sorties obligatoires seulement. Les intermédiaires et les
  tables de poids n'y sont pas. « Go/s » vaut ces octets divisés par la médiane.
- Entrées remplies une fois avant toute mesure : gris moyen pour les YUV, 0,5 pour les tenseurs,
  un mouvement uniforme de quelques pixels pour le warp (une translation, donc des accès
  cohérents comme un flot lisse). Un flot divergent serait moins cohérent.
- Reproductibilité : deux passages identiques (colonnes « Médiane » et « Médiane run 2 ») restent
  à quelques pour cent l'un de l'autre, sauf rife_constants 4K (0,226 contre 0,188 ms).

### Contention de bureau

Le verrou `ajp-gpu` n'écarte que les autres agents, pas le bureau. Pendant les mesures, d'autres
contextes tournaient sur la carte (`nvidia-smi pmon`) :

| Processus | SM |
|---|---:|
| Brave (une vidéo en NVDEC) | 7 à 39 % |
| KWin | 7 à 12 % |
| Vesktop | 6 à 13 % |

Un processus `ajp_trt_build` d'un autre agent est aussi passé brièvement.

- Le pilote partage la carte par tranches de temps. Une passe qui dépasse environ 1 ms est
  coupée et prend environ 1,2 ms de plus. La colonne « Lentes » compte les passes au-delà de
  1,5 × la médiane. warp_blend 4K f32 (1,17 ms) en a 14 sur 100 dans les trois passages : son p95
  et son σ sont gonflés, sa médiane l'est peu.
- Deux premiers passages, écartés, vidaient 256 Mio par passe (0,55 ms de plus). Toutes les
  passes à 4K dépassaient alors 1 ms, les médianes à 4K variaient de 0,65 à 1,9 ms d'un passage
  à l'autre, et le p95 montait à environ 2 ms. D'où le vidage à 64 Mio, désormais réglable.
- Pour des p95 propres, il faudrait mesurer avec le bureau au repos (vidéo de Brave fermée).

## Tableau

| Noyau | Taille | Préc. | Mo | Médiane (ms) | p95 (ms) | σ (ms) | Min (ms) | Lentes | Go/s | % de 504 | Médiane run 2 (ms) | Médiane L2 chaud (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plancher (remplissage de 4 o) | - | - | 0.0 | 0.0106 | 0.0115 | 0.0004 | 0.0101 | 0 | 0.0 | 0 % | 0.0105 | 0.0104 |
| ingest NV12 | 1920x1080 | f16 | 15.6 | 0.1275 | 0.1436 | 0.0066 | 0.1227 | 0 | 122.0 | 24 % | 0.1277 | 0.0901 |
| ingest P010 | 1920x1080 | f16 | 18.7 | 0.1295 | 0.1546 | 0.0096 | 0.0957 | 0 | 144.1 | 29 % | 0.1316 | 0.0869 |
| output_yuv P010 | 1920x1080 | f16 | 18.7 | 0.1482 | 0.1652 | 0.0083 | 0.1360 | 0 | 125.9 | 25 % | 0.1432 | 0.0866 |
| output_rgba16 | 1920x1080 | f16 | 29.0 | 0.0876 | 0.0908 | 0.0029 | 0.0794 | 0 | 331.6 | 66 % | 0.0892 | 0.0254 |
| transfer PQ EOTF | 1920x1080 | f16 | 24.9 | 0.0797 | 0.0827 | 0.0034 | 0.0724 | 0 | 312.4 | 62 % | 0.0811 | 0.0244 |
| transfer BT.709 → BT.2020 | 1920x1080 | f16 | 24.9 | 0.0786 | 0.0887 | 0.0089 | 0.0738 | 2 | 316.7 | 63 % | 0.0863 | 0.0242 |
| rife_crop tenseur | 1920x1080 | f16 | 24.9 | 0.0889 | 0.0931 | 0.0025 | 0.0853 | 0 | 279.9 | 56 % | 0.0925 | 0.0328 |
| rife_constants | 1920x1080 | f16 | 20.9 | 0.0689 | 0.0726 | 0.0066 | 0.0572 | 0 | 303.0 | 60 % | 0.0693 | 0.0227 |
| warp_blend t=0,5 | 1920x1080 | f16 | 58.1 | 0.1967 | 0.2183 | 0.0113 | 0.1702 | 0 | 295.2 | 59 % | 0.1952 | 0.1128 |
| ingest NV12 | 1920x1080 | f32 | 28.0 | 0.1531 | 0.1741 | 0.0100 | 0.1262 | 0 | 182.9 | 36 % | 0.1533 | 0.1303 |
| ingest P010 | 1920x1080 | f32 | 31.1 | 0.1551 | 0.1757 | 0.0124 | 0.1344 | 0 | 200.6 | 40 % | 0.1608 | 0.1317 |
| output_yuv P010 | 1920x1080 | f32 | 31.1 | 0.1809 | 0.1934 | 0.0094 | 0.1680 | 0 | 172.0 | 34 % | 0.1859 | 0.1255 |
| output_rgba16 | 1920x1080 | f32 | 41.5 | 0.1251 | 0.1447 | 0.0079 | 0.1140 | 0 | 331.5 | 66 % | 0.1237 | 0.0278 |
| transfer PQ EOTF | 1920x1080 | f32 | 49.8 | 0.1437 | 0.1562 | 0.0084 | 0.1292 | 0 | 346.3 | 69 % | 0.1354 | 0.0566 |
| transfer BT.709 → BT.2020 | 1920x1080 | f32 | 49.8 | 0.1426 | 0.1591 | 0.0087 | 0.1277 | 0 | 349.0 | 69 % | 0.1351 | 0.0684 |
| rife_crop tenseur | 1920x1080 | f32 | 49.8 | 0.1388 | 0.1590 | 0.0104 | 0.0827 | 0 | 358.6 | 71 % | 0.1333 | 0.0692 |
| rife_constants | 1920x1080 | f32 | 41.8 | 0.1200 | 0.1448 | 0.0164 | 0.0970 | 0 | 348.0 | 69 % | 0.1033 | 0.0966 |
| warp_blend t=0,5 | 1920x1080 | f32 | 116.1 | 0.3276 | 0.3594 | 0.0182 | 0.3124 | 0 | 354.4 | 70 % | 0.3264 | 0.2738 |
| rife_pad NV12 | 1920x1080 | - | 6.2 | 0.0326 | 0.0347 | 0.0010 | 0.0316 | 0 | 191.3 | 38 % | 0.0333 | 0.0176 |
| scene_diff NV12 | 1920x1080 | - | 4.2 | 0.0474 | 0.0484 | 0.0007 | 0.0444 | 0 | 88.1 | 17 % | 0.0475 | 0.0359 |
| block_diff NV12 | 1920x1080 | - | 4.2 | 0.0401 | 0.0423 | 0.0030 | 0.0379 | 1 | 103.6 | 21 % | 0.0399 | 0.0224 |
| ingest NV12 | 3840x2160 | f16 | 62.2 | 0.6453 | 0.6781 | 0.0238 | 0.6015 | 0 | 96.4 | 19 % | 0.6526 | 0.6224 |
| ingest P010 | 3840x2160 | f16 | 74.6 | 0.6800 | 0.7107 | 0.0248 | 0.6384 | 0 | 109.8 | 22 % | 0.6844 | 0.6442 |
| output_yuv P010 | 3840x2160 | f16 | 74.6 | 0.6507 | 0.6947 | 0.0330 | 0.6003 | 0 | 114.7 | 23 % | 0.6548 | 0.6047 |
| output_rgba16 | 3840x2160 | f16 | 116.1 | 0.2611 | 0.3022 | 0.0151 | 0.2525 | 0 | 444.7 | 88 % | 0.2604 | 0.2027 |
| transfer PQ EOTF | 3840x2160 | f16 | 99.5 | 0.2554 | 0.2848 | 0.0129 | 0.2477 | 0 | 389.7 | 77 % | 0.2614 | 0.2473 |
| transfer BT.709 → BT.2020 | 3840x2160 | f16 | 99.5 | 0.2726 | 0.2970 | 0.0148 | 0.2503 | 0 | 365.2 | 72 % | 0.2588 | 0.2477 |
| rife_crop tenseur | 3840x2160 | f16 | 99.5 | 0.2734 | 0.3011 | 0.0121 | 0.2613 | 0 | 364.1 | 72 % | 0.2688 | 0.2496 |
| rife_constants | 3840x2160 | f16 | 83.6 | 0.2263 | 0.2502 | 0.0222 | 0.1826 | 0 | 369.2 | 73 % | 0.1878 | 0.1994 |
| warp_blend t=0,5 | 3840x2160 | f16 | 232.2 | 0.5989 | 0.6305 | 0.0230 | 0.5561 | 0 | 387.8 | 77 % | 0.5780 | 0.5458 |
| ingest NV12 | 3840x2160 | f32 | 112.0 | 0.7550 | 0.8253 | 0.2175 | 0.7014 | 3 | 148.3 | 29 % | 0.7477 | 0.7337 |
| ingest P010 | 3840x2160 | f32 | 124.4 | 0.7854 | 0.8232 | 0.0265 | 0.7306 | 0 | 158.4 | 31 % | 0.7765 | 0.7708 |
| output_yuv P010 | 3840x2160 | f32 | 124.4 | 0.7813 | 0.8115 | 0.0343 | 0.7248 | 0 | 159.2 | 32 % | 0.7747 | 0.7591 |
| output_rgba16 | 3840x2160 | f32 | 165.9 | 0.3662 | 0.4034 | 0.0176 | 0.3603 | 0 | 453.0 | 90 % | 0.3890 | 0.3074 |
| transfer PQ EOTF | 3840x2160 | f32 | 199.1 | 0.4889 | 0.5415 | 0.0257 | 0.4649 | 0 | 407.2 | 81 % | 0.4976 | 0.4894 |
| transfer BT.709 → BT.2020 | 3840x2160 | f32 | 199.1 | 0.4730 | 0.5255 | 0.0235 | 0.4645 | 0 | 420.9 | 84 % | 0.5126 | 0.4884 |
| rife_crop tenseur | 3840x2160 | f32 | 199.1 | 0.4889 | 0.5415 | 0.0270 | 0.4637 | 0 | 407.2 | 81 % | 0.5155 | 0.4895 |
| rife_constants | 3840x2160 | f32 | 167.1 | 0.3892 | 0.4748 | 0.0462 | 0.3552 | 0 | 429.4 | 85 % | 0.4427 | 0.4340 |
| warp_blend t=0,5 | 3840x2160 | f32 | 464.5 | 1.1652 | 2.3517 | 0.4203 | 1.0417 | 14 | 398.6 | 79 % | 1.2995 | 1.2584 |
| rife_pad NV12 | 3840x2160 | - | 25.0 | 0.0768 | 0.0807 | 0.0029 | 0.0732 | 0 | 325.3 | 65 % | 0.0783 | 0.0350 |
| scene_diff NV12 | 3840x2160 | - | 16.7 | 0.1292 | 0.1302 | 0.0013 | 0.1186 | 0 | 129.4 | 26 % | 0.1306 | 0.1112 |
| block_diff NV12 | 3840x2160 | - | 16.6 | 0.0888 | 0.0911 | 0.0098 | 0.0873 | 1 | 187.2 | 37 % | 0.0906 | 0.0520 |
| resize Spline36 | 1280x720 → 2560x1440 | f16 | 27.6 | 0.2077 | 0.2103 | 0.0041 | 0.2055 | 0 | 133.1 | 26 % | 0.2121 | 0.1840 |
| resize Spline36 | 1920x1080 → 3840x2160 | f16 | 62.2 | 0.4995 | 0.5196 | 0.0118 | 0.4869 | 0 | 124.5 | 25 % | 0.5027 | 0.4844 |
| warp_blend t=0,5 | 2560x1440 (mouvement 1280x720) | f16 | 75.6 | 0.2267 | 0.2361 | 0.0052 | 0.2220 | 0 | 333.4 | 66 % | 0.2279 | 0.1895 |
| resize Spline36 | 1280x720 → 2560x1440 | f32 | 55.3 | 0.2530 | 0.2698 | 0.0109 | 0.2448 | 0 | 218.6 | 43 % | 0.2543 | 0.2340 |
| resize Spline36 | 1920x1080 → 3840x2160 | f32 | 124.4 | 0.5922 | 0.6220 | 0.0151 | 0.5713 | 0 | 210.1 | 42 % | 0.5959 | 0.5793 |
| warp_blend t=0,5 | 2560x1440 (mouvement 1280x720) | f32 | 151.1 | 0.4077 | 0.4366 | 0.0234 | 0.3772 | 0 | 370.7 | 74 % | 0.4004 | 0.3813 |

« Lentes » : passes au-delà de 1,5 × la médiane (sur 100). « % de 504 » : Go/s de la médiane rapportés aux
504 Go/s théoriques. Médianes en ms, L2 vidé (64 Mio) sauf la dernière colonne.

## Budgets du plan

Recherche de `ms` et de `budget` dans `plan.txt` :

| Budget du plan | Mesure | Verdict |
|---|---|---|
| S3 : déformer et mélanger à 1440p en 1,8 ms par instant (plan.txt:552) | warp_blend 2560x1440, mouvement 1280x720 : 0,227 ms en f16, 0,408 ms en f32 | Tient : 13 % du poste en f16 ; le reste du poste (agrandissement du flot, post) n'est pas un noyau du socle |
| Flot visé ≤ 3 ms par instant à 1440p (plan.txt:650) | warp_blend 1440p : 0,227 ms | Tient ; le modèle de flot (TensorRT) n'est pas mesuré ici |
| Rendu 2,0 ms par image 4K dans les totaux (plan.txt:629) ; Quality x3 exige un rendu sous 1,3 ms (plan.txt:1099) | output_rgba16 4K f16 0,261 ms + transfer PQ 4K f16 0,255 ms = 0,52 ms | La part Vulkan du socle laisse 0,8 à 1,5 ms à libplacebo |
| Cible 35,4 ms (85 % de 41,7 ms) par paire (plan.txt:565, 629) | Chaîne d'une source 1080p vers 4K en f16 : ingest 0,128 + scene_diff 0,047 + block_diff 0,040 + rife_pad 0,033 + rife_constants 0,069 + rife_crop 0,089 + output_rgba16 4K 0,261 + transfer 4K 0,255 ≈ 0,92 ms | ≈ 2,6 % de la cible ; somme de mesures unitaires L2 vidé, donc pessimiste à 1080p |
| AJG_STATE 1,4 ms par image 4K sur D3D11, « à remesurer sur Vulkan » (plan.txt:537) | Pas un noyau de src/kernels/glsl/ | Non mesuré ici |

Aucun noyau du socle n'a de budget propre dans le plan : les comparaisons portent sur les postes qui
les contiennent.

## Noyaux loin du débit de la carte (~504 Go/s) et hypothèses

Le seuil retenu est 50 % à 4K L2 vidé. À 1080p, presque tout reste sous 70 % pour une raison
commune, et pas propre aux noyaux :

- des noyaux de 0,03 à 0,2 ms portent le plancher de 0,0106 ms, la montée en charge et la queue ;
- les lignes sales laissées par le vidage doivent être réécrites en mémoire ;
- sans vidage, les mêmes noyaux dépassent le débit de la mémoire (1000 Go/s et plus), parce que
  leurs données restent dans le L2.

| Noyau 4K f16 | Go/s obligatoires | Avec intermédiaires | Hypothèse |
|---|---:|---:|---|
| ingest NV12 / P010 | 96 / 110 (19-22 %) | ≈ 405 / 402 (80 %) | Trois passes (ingest_chroma_h, resample_v, ingest_combine) et deux intermédiaires float32 : lignes w × ch × 2 (33 Mo) et plans w × h × 2 (66 Mo), chacun écrit puis relu, soit 199 Mo pour 62 Mo utiles (kernels_color.cpp:47-49). La carte tourne près de son débit ; c'est le volume qui coûte. Une passe fusionnée, avec la chroma verticale en mémoire partagée ou des intermédiaires en f16, ramènerait le noyau vers 0,15-0,2 ms. |
| output_yuv P010 | 115 (23 %) | ≈ 421 (83 %) | Même schéma à l'envers : output_matrix écrit Cb et Cr normalisés w × h × 2 en float32 (66 Mo), resample_h les réduit en cw × h × 2 (33 Mo), puis output_chroma_v (kernels_color.cpp:105-107). Même remède. |
| resize Spline36 1080p → 2160p et 720p → 1440p | 125 / 133 (25-26 %) | ≈ 324 / 346 (64-69 %) | Rééchantillonnage séparable dont l'intermédiaire de lignes est toujours en float32, même pour des tenseurs f16 : dst.width × src.height × 3 × 4 o, soit 50 Mo à 4K (kernels_color.cpp:166). En f32, le noyau monte à 42 % en octets obligatoires. Un intermédiaire f16 ou une passe 2D en tuiles partagées réduirait le trafic d'environ 60 %. |
| scene_diff NV12 | 129 (26 %) | identique | Pas d'intermédiaire. Groupes de 16 × 16 : une invocation par échantillon de 8 bits (un quart de mot), puis une réduction en mémoire partagée de 256 flottants en 8 barrières (scene_diff.comp:30-58). Le noyau est limité par la latence plus que par le débit. Lire un mot de 4 lumas par invocation et réduire par subgroupAdd viserait le débit. |
| block_diff NV12 | 187 (37 %) | identique | Un groupe par bloc de 32 × 32, la même réduction partagée, puis un atomicMax global. Même remède que scene_diff. |
| rife_pad NV12 | 325 (65 %) | identique | Au-dessus du seuil, mais court (0,077 ms), donc plancher et queue pèsent. Pas d'action proposée. |

Les noyaux à une seule passe (transfer, output_rgba16, rife_crop, rife_constants, warp_blend)
atteignent 72 à 88 % à 4K, près du maximum pratique d'une GDDR6X. Il n'y a rien à gagner sans
changer d'algorithme.

## Limites et points ouverts

- **Hôte en Debug** (préréglage linux-prefix) : sans effet sur les timestamps, puisque la porte
  sort la latence de l'hôte de la mesure.
- **Shaders compilés par glslc sans `-O`** (cmake/AjpSpirv.cmake:57). Le compilateur du pilote
  NVIDIA optimise de son côté. L'effet n'a pas été mesuré.
- **Journal du contexte** : il écrit sa ligne `[vk_compute] [info]` sur stdout, avant le tableau
  du banc.
- **Fusion de origin/w2/fondations : non faite.** Elle entre en conflit dans
  `src/core/types.h`. Les quatre RefusalCode de w2/vk-compute (VulkanCallFailed et suivants)
  occupent la même fin d'enum que les neuf de fondations. Trancher l'ordre revient au
  coordinateur ou au chantier vk-compute ; un banc ne doit pas le décider. Ordre naturel :
  fondations d'abord, puis les codes de vk-compute à la suite.
- **Hors banc, pour le chantier vk-compute** : `VkContext::submit` alloue cinq `std::vector` à
  chaque soumission (queue.cpp:48-69). C'est encore le cas dans w2/vk-compute-review-wip, et c'est
  une allocation hôte par image sur le chemin de lecture.

## Reproduire

    export PATH=~/ajp-tools/bin:~/ajp-tools/venv/bin:$PATH AJP_DEPS_PREFIX=~/ajp-deps
    cmake --preset linux-prefix -DAJP_BUILD_BENCH=ON
    ajp-build cmake --build build/linux-prefix --target ajp_vk_kernels_bench ajp_vk_bench_tests
    ajp-gpu build/linux-prefix/tools/bench/vk_kernels/ajp_vk_kernels_bench --peak-gbps 504
    ajp-gpu build/linux-prefix/tools/bench/vk_kernels/ajp_vk_kernels_bench --peak-gbps 504 --flush-mio 0
    # Validation (lavapipe, synchronisation comprise) :
    LP_NUM_THREADS=4 ajp-run with-lavapipe with-validation env VK_VALIDATION_VALIDATE_SYNC=true \
      build/linux-prefix/tools/bench/vk_kernels/ajp_vk_kernels_bench --validation --passes 50 --warmup 1 --only '1920x1080'

## Décision du coordinateur (19 octobre, 18:06 UTC)

Les optimisations de noyaux repérées plus haut ne se font pas maintenant : le socle tient déjà dans
2,6 % du budget. Elles sont notées comme travail de la parité (M3) :
- ingest et output_yuv : intermédiaires en f16, ou passe fusionnée avec la chroma en mémoire
  partagée ;
- resize Spline36 : intermédiaire de lignes en f16, ou passe 2D en tuiles partagées ;
- scene_diff et block_diff : un mot de 4 lumas par invocation et réduction par subgroupAdd.
En revanche, VkContext::submit sans allocation hôte par soumission et glslc avec -O sont confiés
maintenant à la chaîne pc/vk-compute-nvidia (mesure avant/après à ajouter ici).
