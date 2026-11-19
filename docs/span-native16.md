# SPAN FP16 natif (lot C du chantier AnimeJaNai)

Novembre 2026. Le SPAN entier en FP16 par des noyaux écrits chez nous, sans CUTLASS ni TensorRT, pour battre
mpv AnimeJaNai à précision égale : leur SPAN FP16 fait 11,9 ms (Balanced, b8f64) et 6,2 ms (Performance,
b5f48) par image 1080p → 4K sur la RTX 4070 Ti, contre 17,7 et 13,0 ms pour notre TensorRT FP16 dans le lecteur
(profil Nsight du chantier). Cibles : égaler, puis b8f64 ≤ 10 ms et b5f48 ≤ 5 ms ; ≥ 55 dB.

Source des idées (rien de recopié, règle du chantier) : libaji, `experimental/native-span/README.md` et la
description de `src/native/conv_kernels.cuh` (the-database/animejanai-inference) ; le profil Nsight de
`chantier-animejanai/profil-nsys-span-kirari-vs-animejanai.md`.

## Pourquoi les noyaux CUTLASS FP16 d'avant perdaient

Le corps CUTLASS (`--span-native` jusqu'ici) avait déjà le bon nombre de lancements (3 par bloc + 3, la porte
fondue dans la fin de bloc, DepthToSpace fondu dans to_pixel) et perdait quand même de 5 à 20 % contre
TensorRT. Deux raisons : il sommait en f32, ce que les tensor cores des GeForce font à la moitié du débit du
f16 ; et à 48 canaux, la tuile de 48 colonnes de CUTLASS tombait en NaN. TensorRT, lui, somme en f16, comme
AnimeJaNai.

## Ce que font les nouveaux noyaux

`src/gpu/cuda/backend/span_native16_kernels.cuh` (outils des noyaux), `span_native_conv.cuh` (la
convolution), `span_native16.cu` (le réseau, derrière `SpanFp8Body`, à la place de l'ancien
`span_fp8_net16.cu`). Une seule famille de convolutions :

- **mma.sync m16n8k16 FP16, sommes en f16** (`Frag<true>`) ; en f32 au cran Qualité (`SpanFp8Weights::acc32`,
  ci-dessous ; `kirari_span_fp8_run --acc32` au banc) ;
- **tuile en mémoire partagée** : un bloc de 4 warps calcule 8 × 32 pixels et toutes les colonnes ; la tuile
  d'entrée, halo compris, est copiée une fois par `cp.async` (zéros hors de l'image) au pas Cin + 8 : les huit
  lignes d'une matrice de `ldmatrix` tombent sur des banques distinctes. Moins de 48 Ko : deux blocs par SM ;
- **poids rangés une fois** dans l'ordre des fragments B : un `__ldg` de 16 octets par lane donne deux tuiles
  n8, coalescé ; le poids suivant est chargé pendant les mma du pas courant ;
- **porte fondue** : la fin de bloc (v + x) · tanh(W1·v + b1) prend la 1×1 depuis les accumulateurs (deux
  tuiles n8 voisines font un fragment A), la 1×1 et x passent par la mémoire partagée ;
- **dégroupage fondu** dans le chargement de conv_1 : l'image NCHW est lue telle quelle (un lancement et un
  tenseur de moins) ;
- **plus de concaténation** : conv_cat lit les quatre tenseurs là où leurs convolutions les ont écrits (K
  sommé sur quatre sources) ; trois écritures de 2·C octets par pixel en moins ;
- **sortie en lignes entières** : chaque tuile passe par la mémoire partagée avant d'être écrite ; to_pixel y
  fait DepthToSpace et écrit la sortie NCHW et/ou l'emplacement de l'anneau (`store_ring`, au bit près de
  `write_rgba16`).

Les 48 canaux du b5f48 sont calculés en 48 (pas de rembourrage). Rien ne change hors du backend : le paquet
« fp16c » tiré de l'ONNX, l'essai au chargement contre le calcul du processeur (≥ 50 dB, `span_fp8_check`), le
graphe CUDA et l'écriture dans l'anneau sont ceux du chemin d'avant.

## Dans le lecteur : FP16 par défaut, sommes f32 au cran Qualité

Depuis le premier banc (ci-dessous), le SPAN natif est le défaut sur RTX 40 et 50 (`PlayOptions::span_native`,
`BackendCaps::span_fp16_cutlass`), pour les SPAN d'AnimeJaNai seulement (`span_fp8_cutlass_model`) :

- **Équilibre** : FP16 natif, sommes en f16 (paquet « -fp16c- ») ;
- **Qualité** : les mêmes noyaux à sommes f32 (paquet « -fp32n- », drapeau `acc32` du paquet,
  `SpanVariantKind::Fp32Native`), puis la variante FP32 de TensorRT si `quantize_span.py` en a laissé une. En
  f32, le natif reste plus rapide que TensorRT FP16 (14,60 contre 15,39 ms au b8f64) : le cran est offert dans
  Vidéo > Agrandissement > Avancé dès que le moteur a les noyaux, sans variante sur le disque (`span_tiers`) ;
- **repli** : chaque cran finit sur le FP16 natif avant TensorRT FP16 (`span_variant_order`) ; le cran
  Performance garde le FP8 en premier.

Le paquet porte le mode des sommes : deux fichiers, deux clés d'artefact, rien à passer au backend au
chargement (un backend repris d'une vidéo à l'autre ne garde pas un mode périmé). Un ancien `play.toml` au
cran Qualité le prend tel quel. `--debug --no-span-native` revient à TensorRT pour l'ABAB ; `--span-native`
reste accepté (c'est le défaut).

## Sommes promues (essai perdu)

Le f32 du cran Qualité coûte environ 3,8 ms au banc (14,74 contre 10,98 ms au b8f64) : sur les GeForce, un mma
à sommes f32 va à la moitié du débit des tensor cores (RTX 4070 Ti : 80 TFLOPS contre 160). Le b8f64 fait
environ 1,0 TFLOP par image 1080p, soit 12,5 ms au moins en f32 : les noyaux f32 en sont à 85 %, il n'y a
presque plus rien à y prendre.

L'essai de #933 sommait chaque mma en f16 depuis zéro (débit plein), ajouté ensuite à un accumulateur f32
(quatre conversions et quatre additions par mma). Au banc : 14,39 ms et 71,51 dB, contre 14,74 ms et
77,90 dB en f32. Les conversions mangent le gain des tensor cores, et l'arrondi f16 de chaque somme de 16
produits coûte 6,4 dB. Retiré : le cran Qualité reste en f32.

## Lecture contre banc

Au banc (graphe CUDA, la carte pour lui seul), le b8f64 prend 11,8 ms en f16 ; en lecture (Spy×Family, 30 s,
SPAN seul), le rôle « agrandissement » en prend 15,0 en médiane (p95 15,1), 19,3 en f32. Ce temps va du début
du SPAN à la fin de sa queue, attente de l'emplacement de l'anneau comprise (`enqueue_span_fp8`). Le profil
de fin de séance le coupe en trois (« profil GPU · SPAN, corps », « SPAN, attente de l'emplacement de
l'anneau », « SPAN, queue (to_pixel, anneau) »), hors de `gpu_profile` dont les lecteurs somment les étapes :

- l'attente de l'emplacement : la queue attend que l'affichage rende un emplacement (sémaphore Vulkan). Un
  écart logé là n'est pas un coût du SPAN mais le rythme de l'affichage ; le corps ne l'attend déjà pas ;
- le corps plus long qu'au banc ;
- la queue : en lecture, to_pixel écrit aussi l'emplacement RGBA16 de l'anneau, absent au banc.

Mesuré au cran Qualité (Re:Zero à 16:00, SPAN seul) : corps 18,79 ms, attente 0 ms, queue 0,44 ms, contre
14,74 ms au banc. Tout l'écart est dans le corps, environ ×1,27 (×1,27 aussi en f16 à la mesure de #910).
Deux causes possibles, que les mesures n'ont pas encore séparées :

- les horloges : au banc, la carte enchaîne les images et tient sa fréquence maximale ; en lecture à 24 i/s,
  elle n'est chargée qu'à moitié et la gestion d'énergie du pilote peut baisser la fréquence entre deux
  images. Un rapport constant entre lecture et banc, en f16 comme en f32, va dans ce sens ;
- le partage avec Vulkan : le grade, la présentation et la conversion du décodage tournent dans un autre
  contexte, que le pilote fait alterner avec CUDA par tranches de temps, sans vrai recouvrement. Les
  événements CUDA comptent alors le travail Vulkan glissé au milieu du SPAN, et ce travail chasse du L2 les
  bandes que le SPAN y gardait.

La priorité d'un flux CUDA ne range que le travail du même contexte CUDA : elle ne passe pas devant Vulkan,
et ne sert à rien ici. Pour trancher :

```sh
# 1. les fréquences pendant la lecture (sm, mémoire, état d'énergie), à comparer à celles du banc
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu --format=csv -lms 200 \
    > ~/kirari-reports/horloges-lecture.csv &
kirari play --stats extrait.mkv; kill %1
# 2. la même lecture, la fréquence du GPU tenue à son maximum ; le corps retombe-t-il vers le banc ?
F=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
sudo nvidia-smi -lgc "$F,$F" && kirari play --stats extrait.mkv; sudo nvidia-smi -rgc
# 3. qui tient la carte pendant le SPAN, changements de contexte compris
nsys profile --trace=cuda,vulkan --gpuctxsw=true -o ~/kirari-reports/lecture kirari play --stats extrait.mkv
```

Si les horloges suffisent à expliquer l'écart, le levier est la gestion d'énergie (le réglage de l'utilisateur,
ou une charge qui tient la carte éveillée) ; si c'est Vulkan, il faut ranger le grade et la présentation hors
des tranches où le SPAN tourne.

## Le FP8 (mêmes techniques)

Sous `--debug --span-native-edges` (essai, pas encore mesuré), le SPAN FP8 (cran Performance) prend aussi ses
bords à ces noyaux
(`span_native16_edges.cuh`, `SpanFp8Weights::native_edges`, bit `kTrialSpanNative`) : conv_1 lit l'image et
écrit en une passe l'entrée E4M3 du premier bloc et la tranche 0 de la concaténation (le noyau de dégroupage
disparaît), conv_cat lit les quatre tranches par la tuile en mémoire partagée, to_pixel écrit la sortie et
l'anneau ; sommes en f16 quand le paquet est en acc16 (et que la politique de débordement le garde), sinon en
f32. Les convolutions des blocs restent CUTLASS FP8, déjà au plafond de la mémoire en acc16
(`span-fp8-cutlass.md`, balayage des tuiles). Le paquet et son reçu ne changent pas : les bords sont en FP16
des deux côtés, seuls l'ordre des sommes et l'arrondi E4M3 de x0 (depuis le FP16 au lieu du f32) diffèrent.

Une somme f16 débordée (infinie, ou NaN d'un débordement des deux signes) est bornée comme dans les noyaux
CUTLASS du FP8 : ±65504, NaN à 0 (`sums`). Aux bords du FP8, elle est aussi notée dans le mot de débordement
du paquet (`Params::overflow`, le même que `KP::overflow`) : la politique acc16 repasse en f32 comme avec les
bords CUTLASS.

## Bandes

Premier banc (ci-dessous) : le natif f16 ne gagne sur le f32 que 19 % (11,82 contre 14,60 ms au b8f64), loin
du double du débit des tensor cores. Il bute sur la mémoire : chaque couche lit et écrit son tenseur entier
(66 Mo à 64 canaux en 1080p), soit environ 4,2 Go par image, 8,4 ms à 504 Go/s sur la RTX 4070 Ti, quand le
calcul en f16 n'en demande que 6 à 7. Les 48 Mo de L2 ne gardent rien d'une couche à la suivante : la
suivante reprend par le haut, que la précédente a écrit en premier.

Le corps tourne donc en bandes (`SpanFp8Weights::band`, en lignes de demi-résolution,
`kirari_span_fp8_run --band N`, 0 pour l'image entière) : conv_1 fait ses lignes suivantes, puis chaque
couche aussi loin que la précédente le permet, c'est-à-dire une ligne (son halo) avant la fin de celle-ci ;
au bas de l'image, tout va jusqu'au bout. Ce qu'une couche lit vient d'être écrit et tient en L2 (7,9 Mo par
tenseur et par bande de 64 lignes) ; une ligne réécrite avant de quitter la L2 ne va même jamais en mémoire.
Rien ne change aux noyaux qu'une plage de lignes (`Params::y0`, `y1`) ; 26 lancements par bande au b8f64.

Par défaut (`SpanFp8Weights::kBandAuto`, `auto_band`) : 128 lignes là où deux tranches de 128 lignes, celle
qu'une couche lit et celle qu'elle écrit, tiennent dans les deux tiers du L2, 64 sinon. Au banc de la RTX
4070 Ti (48 Mo de L2, 15,7 Mo par tranche au b8f64) : 10,64 ms à 128 lignes, 10,98 à 64. Les cartes à
moins de L2 gardent 64 lignes faute de mesure.

Les tampons qui servent à plusieurs couches restent justes sans copie : la couche qui réécrit T1, T2 ou un
tampon de bloc (X0, X1) suit de trois lignes au moins celle qui le lit encore, qui ne revient jamais en
arrière ; conv_cat écrit sa sortie sur S0, sur place (une 1×1 ne lit que ses propres pixels, et les blocs
ont fini de lire S0 sur ces lignes), et to_pixel la lit après la dernière bande. Une simulation de
l'ordonnancement (chaque ligne marquée de la couche qui l'a écrite) l'a vérifié pour 2 à 8 blocs, des
hauteurs de 1 à 540 lignes et des bandes de 1 à 1000. Les tuiles qui bougent, qui veulent la grille de
l'image entière, se passent des bandes.

Le banc balaie la hauteur (`BANDS`, 0, 64, 192 et 270 en plus du défaut). Avec EXPORT, il dépose aussi la
sortie sans bandes, que `justesse_cpu.py` compare au bit près à celle du défaut : l'ordre des bandes ne change
aucune somme.

## Poids partagés (à 48 canaux)

Le plafond du b8f64 en f16 est de 6,3 ms (1,006 TFLOP par image à 160 TFLOPS, sommes f16) ; le natif en fait
10,64, soit 59 % du plafond, quand le f32 tient 85 % du sien (14,74 contre 12,6 ms) avec les mêmes noyaux. Ce
qui ne pèse qu'en f16 se cache derrière des sommes deux fois plus lentes en f32 : c'est la piste.

Les bandes gardent déjà les activations dans le L2. Reste les poids : chaque warp lit tous ceux de la couche,
en `__ldg`, à chaque tuile. Une 3×3 à 64 canaux en a 73,7 Ko ; quatre warps par tuile en lisent 295 Ko depuis
le L2 (le L1 qui reste à côté des deux tuiles de 43,5 Ko, environ 28 Ko, ne les garde pas), contre 43,5 Ko
pour la tuile d'entrée : 87 % de ce qu'une tuile lit vient des poids, et le même octet est lu quatre fois.
C'est l'hypothèse ; le banc la tranche.

La variante (`SpanFp8Weights::shared_weights`, `kirari_span_fp8_run --shared-weights`,
`accumulate_shared` dans `span_native_conv.cuh`) copie les poids de chaque pas k (16 canaux d'entrée × Cout,
2 Ko à 64 canaux) une fois par bloc en mémoire partagée, par `cp.async`, en double tampon : le pas suivant
arrive pendant les mma du pas courant, une barrière de bloc par pas (36 pour une 3×3 à 64 canaux). Les warps
lisent leurs fragments B dans la mémoire partagée au lieu du L2 : 74 Ko par tuile au lieu de 295.

Pour tenir deux blocs par SM sous 48 Ko avec les 4 Ko des deux tampons, la tuile d'entrée à 64 canaux perd son
rembourrage : le morceau c de 16 octets du pixel p est rangé en c ^ (p & 7) (`Tile`), et les huit lignes d'une
matrice de `ldmatrix`, huit pixels voisins, tombent encore sur des banques distinctes. 43 520 + 4 096 octets ;
à 48 canaux, le pas impair de 7 morceaux reste.

Mêmes produits, même ordre des sommes : la sortie est celle du défaut au bit près. Vérifié sur processeur
dans le cloud (l'émulateur des noyaux, chaque fil une fibre, `mma`, `ldmatrix` et `cp.async` refaits, la
mémoire partagée remplie de NaN à chaque bloc et ses bornes vérifiées) pour 48 et 64 canaux, sommes f16 et
f32, avec et sans bandes ; sur GPU, par le test « les poids partagés rendent la même sortie au bit près ».
Aucun débordement de registres sur sm 8.9 et 12.0.

Au banc (3 tours ABAB, bandes par défaut) : 5,21 → 5,02 ms au b5f48 (−3,7 %), mais 10,64 → 11,07 ms au
b8f64 (+4,1 %). L'hypothèse ne tient donc pas à 64 canaux : les poids relus n'y étaient pas la limite, et la
barrière par pas (36 par 3×3, deux blocs de quatre warps par SM) coûte plus que le L2 épargné. Pourquoi
48 canaux gagnent (27 pas, tuile au pas impair inchangée) n'est pas établi ; un profil Nsight Compute le
dirait (`span-bilan.md`). Le défaut (`SpanSharedWeights::Auto`) les prend à 48 canaux seulement ;
`kirari_span_fp8_run --shared-weights` ou `--no-shared-weights` force le choix, et le banc mesure les deux
dans les mêmes tours que le défaut.

## Mesurer

Le PC ne mesure que les temps ; la justesse se juge sur processeur, dans le cloud. Un seul banc, sur CachyOS,
binaire `linux-release` :

```sh
cmake --build --preset linux-release --target kirari_span_fp8_run
EXPORT=/mnt/project-files/span-native/$(date +%Y%m%d-%H%M) \
  flock ~/kirari-reports/gpu-bench.lock tools/bench/span_native/banc.sh build/linux-release \
    ~/modeles/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx \
    ~/modeles/…_Performance_…_b5f48_….onnx -- extrait.mkv
```

Il construit le moteur TensorRT FP16 de chaque modèle, puis donne en ABAB 3 tours la médiane de TensorRT et
des noyaux natifs (graphe CUDA tous deux, bandes par défaut), et les natifs à d'autres hauteurs de bande et en
f32. EXPORT est à donner seulement quand les noyaux ont changé : après les temps, le banc dépose pour chaque
modèle le modèle, l'entrée et une sortie de chaque mode (f16 aux bandes du lecteur, f16 sans bandes, f16 aux
poids partagés, f32), en demi-flottants, un plan par fichier (16,6 Mo en 4K). Dans le cloud :

```sh
python3 -I tools/bench/span_native/justesse_cpu.py /mnt/project-files/span-native/<date>/<modèle>
```

refait le modèle en FP32 par ONNX Runtime sur processeur (quelques secondes en 1080p), donne l'écart de
chaque sortie (PSNR, plus grand écart, valeurs non finies) et dit si les sorties sans bandes et aux poids
partagés sont celle du défaut au bit près. La référence n'est plus celle du GPU (FP16 à sommes f32,
`--compare`) mais le FP32 entier : les dB ne se comparent pas un à un à ceux de l'État ci-dessous. L'essai C0
(E/S HWC8 de TensorRT) n'y est plus : il a perdu.

Dans le lecteur : `kirari play --stats` ; le journal dit « SPAN FP16 natif chargé : …, sommes f16 » au cran
Équilibre, « …, sommes f32 » au cran Qualité, « SPAN FP8 (noyaux CUTLASS) chargé : …, bords natifs » au cran
Performance avec `--span-native-edges` ; le profil de fin coupe le SPAN en corps, attente de l'emplacement et queue
(« Lecture contre banc »).

## Décision

- natif f16 sous TensorRT et ≥ 55 dB : le natif devient le défaut. Fait en novembre 2026 pour le FP16 et le
  cran Qualité (ci-dessus) ; les bords natifs du FP8 restent à l'essai jusqu'à leur mesure. Ensuite
  seulement, les convolutions des blocs FP8 en mma m16n8k32 E4M3 sur le même gabarit, si un profil montre de
  la marge sous CUTLASS ;
- plus lent : le profil Nsight du banc dit quel noyau perd, et on règle la tuile (lignes par warp, nombre de
  warps, double tampon de la tuile suivante) avant de redécider.

## État

Premier banc (RTX 4070 Ti, 1080p → 4K, image de Re:Zero, 3 tours ABAB, sans bandes, novembre 2026) :

| Modèle | TensorRT FP16 | E/S HWC8 | Natif f16 | Natif f32 | Écart f16 | Écart f32 |
|---|---:|---:|---:|---:|---:|---:|
| b8f64 (Équilibre) | 15,39 ms | 15,96 ms | 11,82 ms (0,77×) | 14,60 ms | 62,66 dB | 77,90 dB |
| b5f48 (Performance) | 9,17 ms | 9,85 ms | 5,92 ms (0,65×) | 6,25 ms | 63,72 dB | |

Les noyaux sont justes (au-dessus des 55 dB visés) et battent TensorRT ; ils égalent AnimeJaNai (11,9 et
6,2 ms) sans marge.

Dans le lecteur (Spy×Family, 30 s, SPAN seul, b8f64) : 66,6 dB et 15,0 ms au cran Équilibre (sommes f16),
78,3 dB et 19,3 ms au cran Qualité (sommes f32) ; le cran Performance prend le FP8 CUTLASS, à 52,3 dB.

Banc de #933 (b8f64, 1080p → 4K, bandes) : TensorRT FP16 14,76 ms, natif f16 10,98 ms à 64 lignes (0,74×),
10,64 ms à 128 lignes, sous les 11,2 ms d'AnimeJaNai, à 62,66 dB ; natif f32 14,74 ms à 77,90 dB ; sommes
promues 14,39 ms à 71,51 dB (retirées).

Banc de #963 (mêmes conditions, 3 tours ABAB) : b8f64 TensorRT FP16 14,76 ms, natif f16 10,64 ms (0,72×),
bandes 0, 64, 192 et 270 à 10,98, 10,98, 10,78 et 11,60 ms, f32 14,44 ms, poids partagés 11,07 ms ; b5f48
TensorRT 8,80 ms, natif f16 5,21 ms (0,59×), bandes 5,83, 5,22, 5,30 et 5,59 ms, f32 5,95 ms, poids partagés
5,02 ms, sous les 5,8 ms d'AnimeJaNai.
