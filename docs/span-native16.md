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

- **mma.sync m16n8k16 FP16, sommes en f16** (`Frag<true>`) ; en f32 sur demande (`SpanFp8Weights::acc32`,
  `kirari_span_fp8_run --acc32`) pour juger la précision ;
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

## Le FP8 (mêmes techniques)

Sous `--span-native`, le SPAN FP8 (cran Performance) prend aussi ses bords à ces noyaux
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

## Mesurer

Un seul banc, sur CachyOS, binaire `linux-release` :

```sh
cmake --build --preset linux-release --target kirari_span_fp8_run
flock ~/kirari-reports/gpu-bench.lock tools/bench/span_native/banc.sh build/linux-release \
    ~/modeles/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx \
    ~/modeles/…_Performance_…_b5f48_….onnx -- extrait.mkv
```

Il construit le moteur TensorRT FP16 de chaque modèle (et le même en E/S HWC8 : l'essai C0), puis donne en
ABAB 3 tours la médiane de TensorRT et des noyaux natifs (graphe CUDA tous deux), les natifs en f32, et l'écart
de la sortie native à la référence FP16 à sommes f32 (PSNR, plus grand écart, valeurs non finies). Dans le
lecteur : `kirari play --span-native --stats` ; le journal dit « SPAN FP16 natif chargé » au cran Équilibre,
« SPAN FP8 (noyaux CUTLASS) chargé : …, bords natifs » au cran Performance.

## Décision

- natif f16 sous TensorRT et ≥ 55 dB : `--span-native` devient le défaut (FP16 natif au cran Équilibre, bords
  natifs du FP8 au cran Performance) ; ensuite seulement, les convolutions des blocs FP8 en mma m16n8k32 E4M3
  sur le même gabarit, si un profil montre de la marge sous CUTLASS ;
- plus lent : le profil Nsight du banc dit quel noyau perd, et on règle la tuile (lignes par warp, nombre de
  warps, double tampon de la tuile suivante) avant de redécider.

## État

Écrit et compilé (nvcc 13, sm 89 et 120, aucun débordement de registres : 96 à 140 registres en f16), bords
du FP8 compris ; relu, jamais lancé (règle de Tatoub : tout se teste ensemble à la fin). La justesse se juge au premier banc (`--compare`) et à chaque chargement (essai contre le
processeur).
