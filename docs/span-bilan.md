# Bilan du SPAN natif (CUDA, RTX 40 et 50)

État au 19 novembre 2026, `main` à d146c35 (poids partagés de #963 compris). De quoi reprendre le chantier
sans personne : ce qui est fait, où en est la recherche, les plafonds et ce qui irait au-delà, ce qui reste,
et où tout se trouve. Le détail technique est dans [`span-native16.md`](span-native16.md) ; le FP8 CUTLASS
dans [`span-fp8-cutlass.md`](span-fp8-cutlass.md). Hors de ce bilan : les tuiles qui bougent
([`tuiles-span.md`](tuiles-span.md)) et la reprise de SPAN sur les dessins tenus (lot A), qui évitent des
images entières ou des tuiles au lieu de rendre chaque calcul plus rapide.

Toutes les mesures : RTX 4070 Ti (60 SM, 48 Mo de L2), 1080p → 4K, SPAN d'AnimeJaNai V3.1 (b8f64 au cran
Équilibre, b5f48 au cran Performance), binaire `linux-release` sur CachyOS.

## 1. Ce qui est fait

### Allumé par défaut

Pour les SPAN d'AnimeJaNai sur RTX 40 et 50 (`span_fp8_cutlass_model`, `BackendCaps::span_fp16_cutlass`) :

| Cran | Chemin | Banc (b8f64) | Lecture (b8f64) | Écart |
|---|---|---:|---:|---:|
| Équilibre | FP16 natif, sommes f16, bandes de 128 lignes | 10,64 ms | 15,0 ms | 62,66 dB au banc, 66,6 dB en lecture |
| Qualité | mêmes noyaux, sommes f32 (paquet « -fp32n- ») | 14,74 ms | 19,3 ms | 77,90 dB au banc, 78,3 dB en lecture |
| Performance | FP8 CUTLASS (acc16), bords CUTLASS | 7,98 ms de noyaux (reçu) | | 52,3 dB en lecture |

À comparer : TensorRT FP16 14,76 ms au même banc, AnimeJaNai 11,2 ms (b8f64) et 5,8 ms (b5f48) sur la même
carte. Chaque cran se replie sur le FP16 natif avant TensorRT FP16 (`span_variant_order`). Les dB du banc sont
contre la référence FP16 du GPU à sommes f32 (`--compare`) ; ceux de la lecture viennent d'un autre extrait
(Spy×Family, 30 s). Depuis #951, la justesse se juge contre ONNX Runtime FP32 sur processeur : les nouveaux dB
ne se comparent pas un à un aux anciens.

Éteints, à l'essai : les poids partagés (`--shared-weights` à l'outil, #963) et les bords natifs du FP8
(`kirari play --debug --span-native-edges`, #854).

### Les PR

| PR | Ce qu'elle a fait |
|---|---|
| #854 | Noyaux FP16 écrits sans CUTLASS (mma.sync m16n8k16, sommes f16, porte et dégroupage fondus, plus de concaténation) ; bords du FP8 par ces noyaux sous option |
| #866 | `kirari_span_fp8_run` : comparaison à la référence et banc sortis de `main` |
| #887 | Somme f16 débordée bornée et notée aux bords du FP8, NaN à 0, code mort et journaux |
| #899 | Corps en bandes : chaque couche lit ce que la précédente vient d'écrire, encore dans le L2 |
| #910 | FP16 natif par défaut, cran Qualité en sommes f32, reçus FP8 d'avant le nouveau nom relus |
| #933 | Sommes promues à l'essai ; le SPAN coupé en trois (corps, attente de l'anneau, queue) au profil de fin |
| #941 | Sommes promues retirées ; bandes de 128 lignes par défaut quand le L2 les tient |
| #951 | Le PC ne mesure que les temps ; EXPORT et `justesse_cpu.py` jugent la justesse sur processeur |
| #963 | Poids partagés à l'essai, sortie au bit près ; la convolution passe dans `span_native_conv.cuh` |

### Mesures, dans l'ordre

| Banc | TensorRT FP16 | Natif f16 | Natif f32 | Écart f16 / f32 |
|---|---:|---:|---:|---:|
| Premier banc, b8f64, sans bandes | 15,39 ms | 11,82 ms | 14,60 ms | 62,66 / 77,90 dB |
| Premier banc, b5f48, sans bandes | 9,17 ms | 5,92 ms | 6,25 ms | 63,72 dB (f16) |
| #933, b8f64, bandes de 64 lignes | 14,76 ms | 10,98 ms | 14,74 ms | inchangés |
| #933, b8f64, bandes de 128 lignes | | 10,64 ms | | inchangés |
| #933, sommes promues (retirées) | | | 14,39 ms | 71,51 dB |

Pas encore mesurés : le b5f48 en bandes, les poids partagés, les bords natifs du FP8.

## 2. Où en est la recherche

### Retenu

- **Les noyaux maison** contre CUTLASS FP16 : CUTLASS sommait en f32 (moitié du débit des tensor cores d'une
  GeForce) et sa tuile de 48 colonnes tombait en NaN à 48 canaux. Les nôtres : tuile 8 × 32 pixels, 4 warps,
  tuile d'entrée en mémoire partagée par `cp.async`, poids rangés dans l'ordre des fragments B, porte,
  dégroupage et DepthToSpace fondus, conv_cat qui lit ses quatre sources sans concaténation.
- **Les bandes** (−0,8 à −1,2 ms) : sans elles, chaque couche relit son tenseur entier depuis la mémoire
  (4,2 Go par image, 8,4 ms à 504 Go/s). En bandes de 128 lignes, ce qu'une couche lit tient encore dans le
  L2.
- **Les sommes f32 au cran Qualité** : 85 % de leur plafond (14,74 contre 12,55 ms), plus rapides que TensorRT
  FP16, à 77,9 dB.

### Écarté, et pourquoi

- **E/S HWC8 de TensorRT** (essai C0) : 15,96 contre 15,39 ms, plus lent.
- **Sommes promues** (#933 puis #941) : chaque mma sommé en f16 depuis zéro puis ajouté à un f32. 14,39 ms et
  71,51 dB contre 14,74 ms et 77,90 dB en f32 : les conversions mangent le gain, l'arrondi f16 coûte 6,4 dB.
- **Le FP8 à 48 canaux rangé par position du filtre en acc16** (CUTLASS) : 7 % de NaN, sans gain.
- **Les formes de tuile CUTLASS** : aucune ne bat celle du lecteur ; le FP8 acc16 bute sur la mémoire (environ
  300 Go/s), pas sur la forme.

### En pause, sur décision du propriétaire

- **Le FP8 natif** (les blocs du cran Performance par nos noyaux, mma m16n8k32 E4M3) : `Frag::mma8`,
  l'épilogue à l'échelle de colonne et les champs de `Params` sont écrits ; la branche E4M3 de `conv_kernel`
  et le corps `span_native8.cu` restent. Note de reprise et code dans le patch (partie 5). Attendu, sans
  mesure : 6 à 7 ms au b8f64, contre 7,98 ms au FP8 CUTLASS.
- **Le test des horloges** : en lecture, le corps prend environ 1,27 fois son temps de banc, en f16 comme en
  f32 (15,0 contre 11,8 ms). Deux causes possibles, les horloges que la gestion d'énergie baisse entre deux
  images à 24 i/s, ou le partage de la carte avec Vulkan. Les trois commandes qui tranchent sont dans
  `span-native16.md`, « Lecture contre banc ». La deuxième demande `sudo` au PC.

### À l'essai

- **Les poids partagés** (#963, attend son banc). L'hypothèse : à 64 canaux, chaque warp relit tous les poids
  d'une 3×3 (73,7 Ko) à chaque tuile ; quatre warps en tirent 295 Ko du L2 pour 43,5 Ko de tuile d'entrée. Sur
  une image, environ 14,3 Go de poids relus et 4,2 Go d'activations : 18,5 Go en 10,64 ms, soit 1,7 To/s de L2
  (débit du L2 de la 4070 Ti non mesuré ici). En f32, les sommes deux fois plus lentes cachent ce trafic, d'où
  85 % du plafond en f32 et 59 % en f16. La variante charge les poids de chaque pas une fois par bloc en
  mémoire partagée (3,6 Go par image). Sortie identique au bit près sur l'émulateur (48 et 64 canaux, f16 et
  f32, avec et sans bandes), aucun débordement de registres en sm 8.9 et 12.0. Gagnante, elle devient le
  défaut ; perdante, elle part.
- **Les bords natifs du FP8** : conv_1, conv_cat et to_pixel du cran Performance par nos noyaux. Jamais
  mesurés.

## 3. Plafonds

### Hypothèses

- Débits de la 4070 Ti, cœurs tensoriels sans creux : 160 TFLOPS en sommes f16, 80 en sommes f32 (moitié sur
  une GeForce), 321 en FP8 sommes f16. Mémoire : 504 Go/s.
- Le corps tourne à 540 × 960 (le SPAN dégroupe l'entrée par 2, puis DepthToSpace par 4 en sortie) :
  518 400 pixels. Une 3×3 à 64 canaux : 73 728 FLOP par pixel ; la porte ajoute une 1×1 (8 192).
- « Par couche » : chaque couche lit son entrée et écrit sa sortie en mémoire, en f16 ; plafond de la couche,
  le plus grand du calcul et des octets. « Fondu » : seules l'image d'entrée et la sortie 4K passent par la
  mémoire (62 Mo, 0,12 ms), le plafond est alors le calcul seul.
- Pas de queue d'onde, pas de lancement, pas de transformée : des bornes basses.

### Les chiffres

| Modèle | GFLOP par image | Calcul seul f16 | f32 | FP8 (blocs) | Par couche f16 | f32 | FP8 | Mesuré f16 / f32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b8f64 | 1 004 | 6,28 ms | 12,55 ms | 3,29 ms | 8,42 ms | 13,06 ms | 4,73 ms | 10,64 / 14,74 ms |
| b5f48 | 371 | 2,32 ms | 4,64 ms | 1,27 ms | 4,27 ms | 5,07 ms | 2,54 ms | 5,92 / 6,25 ms (sans bandes) |

Ce qu'on en tire :

- **b8f64 f16 : 59 % du plafond de calcul.** Les bandes ont déjà passé la borne « par couche » de la mémoire
  (le trafic d'activations reste dans le L2), il reste le calcul et le L2. Le f32, à 85 %, est fini.
- **b5f48 f16 : 39 % du plafond, et pas encore mesuré en bandes.** C'est le modèle où il reste le plus, en
  proportion : plus petit, il pèse moins sur les tensor cores et plus sur les octets et les lancements.
- **FP8** : deux fois le débit des tensor cores et moitié des octets dans les blocs. Plafond 3,3 ms au b8f64
  contre 6,3 en f16, mais la justesse est celle du cran Performance (52,3 dB en lecture).

### Au-delà, levier par levier

Les gains sont des estimations sans mesure, sauf les plafonds, qui sont calculés.

| Levier | Gain possible (b8f64 f16) | Risque | Où |
|---|---|---|---|
| Poids partagés (#963) | 14,3 → 3,6 Go de poids lus au L2 ; 8 à 9 ms si l'hypothèse tient | faible : bit près, éteint par défaut | fait, banc au PC |
| Réglage par couche (tuile, warps, lignes par warp, bande), choisi au chargement | 5 à 15 % | faible ; chronométrer les variantes au chargement est admis | cloud, puis temps au PC |
| Fusion d'un bloc SPAB | le trafic d'activations du L2 (4,2 Go) | moyen à fort, voir ci-dessous | cloud, puis temps au PC |
| Winograd F(2×2, 3×3) | multiplications des 3×3 ÷ 2,25 : 2,97 ms de mma au plafond | f16 : dB à mesurer ; transformées sur les cœurs ordinaires | dB dans le cloud d'abord |
| Winograd F(4×4, 3×3) | ÷ 4 : 1,82 ms de mma au plafond | fort : coefficients jusqu'à 4, l'erreur f16 grandit | dB dans le cloud d'abord |
| FP8 natif (cran Performance) | 3,29 ms au plafond, 6 à 7 ms attendus | justesse du FP8 ; en pause | cloud et PC |
| Parcimonie 2:4 (mma.sp) | tensor cores ×2 sur les blocs : 3,3 ms au plafond en f16 | fort : réentraîner le SPAN (élagage puis affinage), licence du modèle à vérifier, calcul d'entraînement | hors du cloud processeur |
| Distillation dans un SPAN plus petit | au prorata des FLOP (b5f48 = 37 % du b8f64) | fort : qualité, entraînement | hors du cloud processeur |
| Résolution | déjà bornée par la chaîne (source réduite au-dessus de 1080p) | baisser plus perd de la qualité | non proposé |
| Cache L2 (fenêtre persistante) | faible : les bandes le font déjà ; les poids pèsent 74 Ko | faible | cloud, puis temps au PC |
| Horloges en lecture | jusqu'à 15,0 → 11,8 ms en lecture, sans toucher aux noyaux | dépend de la cause ; en pause | PC (`sudo`) |

La fusion d'un bloc coûte du calcul : trois 3×3 de suite sur une tuile de sortie 8 × 32 recalculent leurs
halos (12 × 36 puis 10 × 34 pixels), 1,34 fois le calcul ; en 16 × 64, 1,16 fois, mais la sortie de la
première couche (20 × 68 × 64 canaux) pèse 174 Ko, plus que les 100 Ko de mémoire partagée d'un SM. Sur un
SPAN limité par le calcul, la fusion par tuiles perd. La forme qui tiendrait : une bande de colonnes de 64
pixels qui descend l'image et garde trois lignes de chaque couche en mémoire partagée (environ 80 Ko, recalcul
de 3 % sur les côtés seulement), découpée aussi en hauteur pour occuper les 60 SM. À chiffrer après le banc
des poids partagés : elle ne vaut que si le L2 limite encore.

## 4. Ce qui reste, dans l'ordre

1. **PC, temps seuls** (quand les chantiers qui n'ont pas passé leur porte ont fini) : le banc mesure les
   poids partagés à côté du défaut, et le b5f48 en bandes.

   ```sh
   cd ~/kirari-player && git pull && cmake --build --preset linux-release --target kirari_span_fp8_run
   EXPORT=/mnt/project-files/span-native/$(date +%Y%m%d-%H%M) \
     flock ~/kirari-reports/gpu-bench.lock tools/bench/span_native/banc.sh build/linux-release \
       ~/modeles/…b8f64….onnx ~/modeles/…b5f48….onnx -- extrait.mkv
   ```

2. **Cloud** : `python3 -I tools/bench/span_native/justesse_cpu.py
   /mnt/project-files/span-native/<date>/<modèle>` (les dB contre ONNX Runtime FP32, et le bit près des bandes
   et des poids partagés). Puis une PR : les poids partagés deviennent le défaut et l'option disparaît, ou ils
   partent.
3. **PC, si le gain est faible** : confirmer l'hypothèse du L2 sur une convolution (Nsight Compute, droits
   d'administrateur pour les compteurs ; noms de métriques à vérifier par `ncu --query-metrics`) :

   ```sh
   M=lts__t_bytes.sum,dram__bytes.sum
   M=$M,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
   sudo ncu --kernel-name regex:conv_kernel --launch-skip 40 --launch-count 3 --metrics "$M" \
     build/linux-release/src/gpu/cuda/backend/kirari_span_fp8_run MODELE.onnx 1080 1920 \
     ~/kirari-reports/span-native-<date>/entree.f32 /tmp/sortie.f32 --fp16 [--shared-weights]
   ```

4. **Cloud** : le réglage par couche. Des variantes de tuile (lignes par warp, 8 warps par bloc qui partagent
   les mêmes poids, hauteur de bande) vérifiées sur l'émulateur, puis chronométrées au chargement comme
   l'admet le propriétaire (quelques millisecondes à l'ouverture pour une vraie optimisation). Temps au PC.
5. **Cloud** : Winograd, les dB d'abord. Un script numpy (à écrire, `tools/bench/span_native/`) refait le SPAN
   en f16 avec les transformées F(2×2, 3×3) et F(4×4, 3×3) et l'arrondi de chaque somme de 16 produits, sur
   l'image exportée par le banc, contre ONNX Runtime FP32. Les noyaux seulement si la perte laisse de la marge
   au-dessus de 55 dB.
6. **Cloud puis PC** : la fusion en bande de colonnes (partie 3), si le point 3 montre que le L2 limite
   encore.
7. **PC, en lecture** : les bords natifs du FP8 au cran Performance,
   `kirari play --debug --span-native-edges --stats` contre sans l'option (le profil de fin coupe le SPAN en
   corps, attente et queue).
8. **Sur l'accord du propriétaire seulement** : le test des horloges (commandes dans `span-native16.md`), le
   FP8 natif (étapes 1 à 6 de sa note de reprise), puis la parcimonie 2:4 et la distillation, qui demandent un
   entraînement.

## 5. Point de reprise

- **Code** (`src/gpu/cuda/backend/`) : `span_native16_kernels.cuh` (outils : `Frag`, `Tile`, chargements,
  épilogues, `smem_bytes`), `span_native_conv.cuh` (`conv_kernel`, `accumulate_shared`), `span_native16.cu`
  (le corps derrière `SpanFp8Body`, bandes, `auto_band`), `span_native16_edges.cuh` (bords du FP8),
  `span_fp8_net.h` (`SpanFp8Weights` : `band`, `sums`, `shared_weights`, `native_edges`), `span_fp8_run.cpp`
  (l'outil de banc).
- **Tests** : `tests/unit/backend_span_fp8_kernels/span_fp8_48_test.cpp` (FP16 natif contre la référence,
  poids partagés au bit près ; GPU sm 8.9 seulement).
- **Banc** : `tools/bench/span_native/banc.sh` (temps en ABAB, EXPORT), `justesse_cpu.py` (dB sur processeur).
- **Émulateur processeur des noyaux** (dossier du projet, hors dépôt) :
  `/mnt/project-files/chantier-natif/span-natif-emu/` (`LISEZMOI.md`) : `./build.sh <dépôt> <sortie>`, puis
  `span_test SORTIE.bin [64|48] [f16|f32] [BANDE] [H W] [partages]` (sortie au bit près comparée par `cmp`, dB
  contre le SPAN f32 du processeur). Références de `main` en 48 × 80 : 64 canaux 71,81 dB (f16) et 83,57 dB
  (f32) ; 48 canaux 76,87 et 86,97 dB.
- **FP8 natif en pause** : `/mnt/project-files/span-native/fp8-natif-en-pause-sur-963.patch`, rejoué sur
  `main` après #963, s'applique par `git am` et se compile en sm 89 et 120 ; l'ancien
  `fp8-natif-en-pause.patch` vaut pour a96e1e5. La note de reprise entre dans `span-native16.md`.
- **Outils du cloud** (le conteneur repart de zéro) : nvcc et clang-format par pip (`nvidia-cuda-nvcc`,
  `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-cuda-crt`, `clang-format==20.1.8`) ; compiler les `.cu`
  avec `--ptxas-options=--warn-on-spills -Werror=all-warnings` en sm 89 et 120.
- **Règles** : le PC ne fait que les temps, après les chantiers qui n'ont pas passé leur porte ; la justesse
  se juge dans le cloud. Une PR qui touche `src/gpu` change l'empreinte du module TensorRT.
