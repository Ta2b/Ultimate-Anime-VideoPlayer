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

`src/gpu/cuda/backend/span_native16_kernels.cuh` (noyaux), `span_native16.cu` (le réseau, derrière
`SpanFp8Body`, à la place de l'ancien `span_fp8_net16.cu`). Une seule famille de convolutions :

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

Le corps tourne donc en bandes (`SpanFp8Weights::band`, 64 lignes de demi-résolution par défaut,
`kirari_span_fp8_run --band N`, 0 pour l'image entière) : conv_1 fait ses 64 lignes suivantes, puis chaque
couche aussi loin que la précédente le permet, c'est-à-dire une ligne (son halo) avant la fin de celle-ci ;
au bas de l'image, tout va jusqu'au bout. Ce qu'une couche lit vient d'être écrit et tient en L2 (8 Mo par
tenseur et par bande) ; une ligne réécrite avant de quitter la L2 ne va même jamais en mémoire. Rien ne change
aux noyaux qu'une plage de lignes (`Params::y0`, `y1`) ; 26 lancements par bande au b8f64, 9 bandes en 1080p.

Les tampons qui servent à plusieurs couches restent justes sans copie : la couche qui réécrit T1, T2 ou un
tampon de bloc (X0, X1) suit de trois lignes au moins celle qui le lit encore, qui ne revient jamais en
arrière ; conv_cat écrit sa sortie sur S0, sur place (une 1×1 ne lit que ses propres pixels, et les blocs
ont fini de lire S0 sur ces lignes), et to_pixel la lit après la dernière bande. Une simulation de
l'ordonnancement (chaque ligne marquée de la couche qui l'a écrite) l'a vérifié pour 2 à 8 blocs, des
hauteurs de 1 à 540 lignes et des bandes de 1 à 1000. Les tuiles qui bougent, qui veulent la grille de
l'image entière, se passent des bandes.

Le banc balaie la hauteur (`BANDS`, 0, 32 et 128 en plus du défaut) et juge la justesse avec et sans
bandes : les deux écarts à la référence doivent être égaux, l'ordre des bandes ne change aucune somme.

## Mesurer

Un seul banc, sur CachyOS, binaire `linux-release` :

```sh
cmake --build --preset linux-release --target kirari_span_fp8_run
flock ~/kirari-reports/gpu-bench.lock tools/bench/span_native/banc.sh build/linux-release \
    ~/modeles/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx \
    ~/modeles/…_Performance_…_b5f48_….onnx -- extrait.mkv
```

Il construit le moteur TensorRT FP16 de chaque modèle, puis donne en ABAB 3 tours la médiane de TensorRT et
des noyaux natifs (graphe CUDA tous deux, bandes par défaut), les natifs à d'autres hauteurs de bande et en
f32, et l'écart de la sortie native à la référence FP16 à sommes f32 (PSNR, plus grand écart, valeurs non
finies), avec et sans bandes. L'essai C0 (E/S HWC8 de TensorRT) n'y est plus : il a perdu. Dans le
lecteur : `kirari play --stats` ; le journal dit « SPAN FP16 natif chargé : …, sommes f16 » au cran Équilibre,
« …, sommes f32 » au cran Qualité, « SPAN FP8 (noyaux CUTLASS) chargé : …, bords natifs » au cran
Performance avec `--span-native-edges`.

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
6,2 ms) sans marge. Les bandes (ci-dessus) sont écrites et compilées (sm 89 et 120, sans débordement), à
mesurer au prochain banc.
