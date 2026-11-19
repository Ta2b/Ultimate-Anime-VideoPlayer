# GMFSS natif (`--gmfss-native`)

Novembre 2026. GMFSS Fortuna Union ([`interpolation.md`](interpolation.md)) en temps réel sur la RTX 4070 Ti : ses
petits réseaux de convolutions par nos noyaux CUDA, GMFlow restant sous TensorRT. Le chantier se fait par étapes,
chacune derrière `--gmfss-native` (éteint par défaut, essai) :

1. le tri des paires pour GMFSS (`--pair-triage` : dessins tenus, paires impossibles, panoramiques purs) ;
2. les lectures des splats sur la bibliothèque de warp ([`warp-lib.md`](warp-lib.md)) ;
3. **le GridNet de la fusion** (ce document) ;
4. FeatureNet et MetricNet de la paire ;
5. GMFSS union de retour au menu, seulement derrière le natif ;
6. GMFlow natif, s'il égale TensorRT FP8 en qualité et en vitesse ; sinon TensorRT reste.

## Le GridNet de la fusion

La fusion exportée par `tools/gmfss/export_gmfss.py` est l'IFNet de RIFE 4.26 (half0, half1, time → « rife ») suivi
du GridNet de FusionNet_u : 21 blocs (têtes, résiduels, descentes, montées par transposée 4 × 4 de pas 2, chacun
PReLU, convolution, PReLU, convolution) sur trois lignes (64, 128 et 192 canaux à la moitié, au quart et au
huitième des plans remplis), puis la queue (convolution et PReLU, convolution 64 → 256 et PixelShuffle(2),
convolution 64 → 3, bornée à [0, 1]). Environ 750 GFLOP par fusion en sortie 1280 × 768, 45 convolutions.

Sous `--gmfss-native`, le lecteur ne construit plus la fusion seule mais un manifeste « .gmfssfused »
(`prepare/gmfss_fused.h`) : la fusion entière (le même moteur qu'avant, repris du cache) et son IFNet extrait en un
petit modèle, que trtexec construit avec les réglages de RIFE (c'est un RIFE 4.26). Au chargement (`tensorrt_gmfss_fused.cpp`), les poids
du GridNet sont lus dans la fusion entière (`prepare/gmfss_native.h` : initialiseurs `gmfss.fusionnet.*`, pentes des
PReLU exportés en Relu, Neg, Relu, Mul, Sub) et le GridNet natif est essayé contre la fusion TensorRT entière.
L'étage (`upscale_gmfss.cpp`) ne voit pas la différence : mêmes entrées, même sortie.

Les noyaux sont ceux du RIFE natif ([`rife-natif.md`](rife-natif.md), `rife_native_kernels.cuh`), avec un ajout :
le **PReLU à la lecture** (`Params::in_slope`). Le graphe lit une même somme sous deux pentes (la somme A sert au
résiduel 01 et à la descente 10, chacun avec la sienne) : la pente s'applique à la tuile chargée en mémoire
partagée, sans tampon de plus. Le reste se range dans l'épilogue existant :

- le PReLU qui suit une convolution est son activation de sortie ; après une transposée (écrite par phases, sans
  épilogue), il passe à la lecture de la convolution suivante ;
- les sommes de deux blocs (A = head0 + head1, F2 = D + res11 + down11…) s'écrivent par le résiduel : le premier
  bloc écrit seul, le second l'ajoute à sa sortie ;
- le PixelShuffle de la queue est une sortie par phases : le canal c · 4 + phase de la convolution est le canal c
  de la phase, écrit au pixel (2y + py, 2x + px) ;
- les tenseurs sont placés dans une zone par durée de vie (`gmfss_conv_plan.cuh`, partagé avec la paire) : deux
  tenseurs qui vivent ensemble ne se recouvrent pas, le résiduel n'est donc jamais la sortie.

Les entrées en plans (warped*_1..3, l'image de RIFE) passent par pixel dans x0 à x3 au début, la sortie revient en
plans à la fin : deux copies, qu'un splat écrivant par pixel supprimera.

## Justesse

- Le plan (table de `gmfss_native.cpp`, index des phases, PixelShuffle, arrondis FP16 aux endroits des noyaux),
  rejoué en Python depuis le réseau lu par le C++, contre onnxruntime sur la fusion exportée : **76,6 dB** (écart
  au plus 4,9e-4), sommes en f32.
- Essai au chargement (`gmfss_fused_trial.cpp`) : des entrées de test (deux dessins décalés de 4 pixels, des
  caractéristiques en ondes) passent par la fusion TensorRT entière, puis par l'IFNet et le GridNet natif en
  sommes f16, mixtes et f32, comme le RIFE natif ([`rife-natif.md`](rife-natif.md)) ; trois passes, la dernière
  comparée. Les variantes à **45 dB** ou plus sont chronométrées contre TensorRT (`gmfss_trial.h` : onze tours
  intercalés, la médiane) ; la plus rapide n'est gardée que si elle bat TensorRT d'au moins 2 %, sinon TensorRT
  garde la fusion. Le journal dit les dB et les temps. La séance finale vise 50 dB.
- Écrit et compilé (nvcc 13.5, sm_75, 86, 89, 120, `--warn-on-spills -Werror`), jamais lancé sur une carte.

## Séance finale (sur le PC, binaire `linux-release`)

GMFSS union a quitté le menu ([`interpolation.md`](interpolation.md)) : un `play.toml` qui le nomme tombe sur
RIFE 4.25 (`interp_retired.h`). Il revient à l'étape 5, seulement derrière `--gmfss-native` ; d'ici là le natif ne
s'essaie que par les tests GPU du backend. Les commandes ci-dessous sont celles de l'étape 5 (Très élevé,
GMFSS union seul ; Élevé, le mélange avec RIFE 4.26).

```sh
# 0. Première ouverture : la fusion entière et l'IFNet se construisent ; la ligne « GMFSS natif retenu… »
#    donne les dB (50 visés), « GMFSS natif écarté… » dit pourquoi TensorRT garde la fusion.
kirari --debug play --gmfss-native --motion=gmfss --rife=2 --start=960 extrait.mkv 2>&1 | grep -i "gmfss natif"

# 1. GMFSS union seul ×2 puis ×3, avec SPAN : natif avec et sans le tri des paires, puis TensorRT seul
#    (--motion=gmfss choisit GMFSS avec ou sans le natif, pour la comparaison).
for x in 2 3; do
  kirari --debug play --gmfss-native --motion=gmfss --rife=$x --pair-triage --start=960 --stats=n$x.csv extrait.mkv
  kirari --debug play --gmfss-native --motion=gmfss --rife=$x --start=960 --stats=n$x-sans-tri.csv extrait.mkv
  kirari --debug play --motion=gmfss --rife=$x --start=960 --stats=t$x.csv extrait.mkv
done

# 2. Tests GPU du backend (l'essai natif contre TensorRT, à la taille de la chaîne).
KIRARI_TEST_VULKAN_VALIDATION=off ctest --preset linux-release -R "backend|gmfss"
```

Le paquet Linux change d'empreinte TensorRT (`tools/release/trt-fingerprint.sh`) : `src/gpu` et `src/prepare` ont
changé.
