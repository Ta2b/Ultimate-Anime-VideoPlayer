# DistilDRBA en noyaux natifs : fiche de reprise (en pause)

19 novembre 2026. Mis en pause à la demande implicite du tri des priorités (GMFSS d'abord, puis la mesure de
RIFE natif). Rien de fusionné. Reprendre après la séance de mesure de `docs/rife-natif.md`.

## Gain attendu

Chaîne DistilDRBA fusionnée (banc 4070 Ti, 1088×1920) : 0,92 ms par image créée, dont 0,59 ms pour les trois
blocs TensorRT (0,085 / 0,125 / 0,379), 0,11 ms d'encodeur (sauté en temps seul, D6) et 0,15 ms de colle.
À 20 % près sur les convolutions : 0,1 à 0,15 ms par image créée. Le vrai gain : ne plus construire les quatre
moteurs (38 s au premier lancement), une fois le natif tenu.

## Structure (routineLife1/DistilDRBA, distilDRBA_v2_lite.py)

- Head : cnn0 3→16 pas 2, cnn1 et cnn2 16→16, cnn3 ConvTranspose 16→16 (4×4, pas 2) ; LeakyReLU 0,2 sur les
  trois premières. Même forme que RIFE, mais 16 sorties au lieu de 4.
- IFBlock comme RIFE : conv0 (c/2 pas 2, puis c pas 2), 8 ResConv à bêta, lastconv ConvTranspose c→52.
  c = 192, 128, 96 ; entrées 58 (X_0) et 109 (X_1, X_2, flot compris).
- `prepare::rife_native_layers` lit déjà cette chaîne ; `rife_native_check(layers, 3, 16)` pour l'encodeur,
  `(58 ou 109, 52)` pour les blocs.

## Ce qui manque

- Formats en plans (kernels/spec/drba_fused.h) : X_k en NCHW (58 ou 109 plans), Y_k en NCHW (52 plans à la
  moitié de X_k), l'encodeur lit trois plans dans X_2 (plans 3i à 3i+2) et écrit 16 plans (9 + 16i) au même
  format. Dans `rife_native_kernels.cuh` :
  - `In::kPlanes` : la tranche [c0, c0+Kc) lue plan par plan (un demi-flottant par fil, fils voisins en x),
    zéros au-delà des canaux réels (paramètre in_ch) ;
  - `Out::kPhasePlanes` : la phase d'une transposée écrite plan par plan ; `reduce_kernel` aussi.
- `RifeNativeNet::create_planar(pieces, encoder h×w, blocs (canaux, h, w), acc32, error)` : encodeur kImage →
  kSame → kSame → kPhasePlanes (16 canaux) ; bloc kHalfPlanes → kHalf → 8 kSame résiduels → kPhasePlanes (52).
  out_plane = 4·oh·ow dans les deux cas.
- `DrbaFusedEngine` : native_, try_native, trial_output comme RifeFusedEngine (`rife_fused_native.cpp`) ; trois
  appels d'encode (un par image), la zone de travail au bas de l'arène (`arena->used`).
- `--drba-native`, bit d'essai suivant libre (RIFE 1<<6, GMFSS 1<<7), kBackendAbi + 1, kFlags et kTrialOptions
  recomptés, options.cpp ≤ 800 lignes.
