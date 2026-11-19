# GMFSS natif (`--gmfss-native`)

Novembre 2026. GMFSS Fortuna Union ([`interpolation.md`](interpolation.md)) en temps réel sur la RTX 4070 Ti : ses
petits réseaux de convolutions par nos noyaux CUDA, GMFlow restant sous TensorRT. Le chantier se fait par étapes,
chacune derrière `--gmfss-native` (éteint par défaut, essai) :

1. le tri des paires pour GMFSS (`--pair-triage` : dessins tenus, paires impossibles, panoramiques purs) ;
2. les lectures des splats sur la bibliothèque de warp ([`warp-lib.md`](warp-lib.md)) ;
3. **le GridNet de la fusion** (ce document) ;
4. **FeatureNet et MetricNet de la paire** (ce document) ;
5. **GMFSS union de retour au menu**, seulement derrière le natif (ce document) ;
6. GMFlow natif, s'il égale TensorRT FP16 en qualité et en vitesse (GMFlow n'a pas de FP8) ; sinon TensorRT
   reste.

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

## FeatureNet et MetricNet de la paire

La paire exportée prend `frames` (les deux images, 1 × 6 × 2h × 2w, h × w la taille de la fusion) et rend half0,
half1, puis pour chaque image feature_1 (64 plans à h × w), source_2 (128 à la moitié) et source_3 (192 au quart),
enfin les six mouvements motion{k}_{l} (3 plans : le flot, puis la métrique). GMFlow y fait l'essentiel ; le
FeatureNet (trois blocs de deux convolutions 3 × 3, le premier de pas 2 sur l'image) et le MetricNet (une
convolution d'entrée sur 14 plans, trois résiduels, une de sortie) sont les petits réseaux que les noyaux prennent.

Sous `--gmfss-native`, le lecteur construit un manifeste « .gmfsspair » (`prepare/gmfss_pair_split.h`) : la paire
entière (le même moteur qu'avant, repris du cache) et son « flot », extrait par nom de la paire (GMFlow et ce qui
prépare l'entrée du MetricNet, `gmfss_pair_cut`), qui rend l'entrée du MetricNet (14 plans), les six flots, half0
et half1. Au chargement (`tensorrt_gmfss_pair.cpp`), les poids du FeatureNet et du MetricNet sont lus dans la paire
(`prepare/gmfss_pair_native.h`) et le natif est essayé contre la paire entière. À chaque paire
(`gmfss_pair_engine.cpp`) :

- le flot écrit directement dans les sorties de la paire : les deux premiers plans de chaque mouvement, half0,
  half1 ; l'entrée du MetricNet va dans un tampon à part ;
- le FeatureNet tourne sur chaque image (`In::kImage` lit ses trois plans en pas 2, sans copie) et écrit les trois
  niveaux de caractéristiques ;
- le MetricNet écrit tanh × 10 au troisième plan des mouvements : tel quel au niveau 1, en moyenne 2 × 2 au
  niveau 2, en moyenne des quatre pixels du centre de chaque 4 × 4 au niveau 3 (le Resize linéaire de la paire).

Les caractéristiques d'une image ne dépendent que d'elle : celles de B pourraient servir de A à la paire suivante
(un FeatureNet par paire au lieu de deux). Ce n'est pas encore fait, la paire garde ses deux jeux.

## Sommes et vitesse

Les noyaux font leurs sommes en f16, mixtes (chaque prise en f16, les prises entre elles en f32) ou f32, comme
pour le RIFE natif ([`rife-natif.md`](rife-natif.md)) ; la tuile de 64 canaux de sortie n'existe qu'en f16.
L'essai au chargement de la fusion comme de la paire passe chaque variante contre TensorRT ; celles qui tiennent
le seuil sont chronométrées contre lui (`gmfss_trial.h` : onze tours intercalés, la médiane), et la plus rapide
n'est gardée que si elle bat TensorRT d'au moins 2 %. Sinon TensorRT garde la fusion ou la paire : le natif ne
coûte jamais de qualité ni de vitesse. Le journal dit, pour chacune, les dB de chaque variante et les temps.

## Justesse

- Le plan (table de `gmfss_native.cpp`, index des phases, PixelShuffle, arrondis FP16 aux endroits des noyaux),
  rejoué en Python depuis le réseau lu par le C++, contre onnxruntime sur la fusion exportée : **76,6 dB** (écart
  au plus 4,9e-4), sommes en f32.
- La paire, rejouée de même (`tools/gmfss/emul_pair_native.py`, sommes en f32) contre onnxruntime : au pire
  **71,5 dB** (le plan de la métrique), chaque sortie rapportée à son étendue ; passée par les splats et la fusion,
  l'image finale est à **83 dB** de celle de la paire entière. Le flot extrait rend exactement ce que rend la paire.
- Essai au chargement de la fusion (`gmfss_fused_trial.cpp`) : des entrées de test (deux dessins décalés de
  4 pixels, des caractéristiques en ondes) passent par la fusion TensorRT entière, puis par l'IFNet et le GridNet
  natif ; trois passes, la dernière comparée. De la paire (`gmfss_pair_trial.cpp`) : les deux dessins passent par
  la paire entière, puis par le flot et les réseaux natifs ; la pire des quatorze sorties est comparée, rapportée
  à son étendue (au moins 1). Une variante n'est mesurée qu'à **45 dB** ou plus ; la séance finale vise 50 dB.
- Écrit et compilé (nvcc 13.5, sm_75, 86, 89, 120, `--warn-on-spills -Werror`), jamais lancé sur une carte.

## Séance finale (sur le PC, binaire `linux-release`)

Sous `--gmfss-native`, le menu propose de nouveau « RIFE 4.25 + GMFSS union » (l'ancien Élevé) et « GMFSS union »
(l'ancien Très élevé), et `play.toml` les garde ; sans l'option, ils tombent sur RIFE 4.25 comme avant
(`interp_retired.h`). `--motion=gmfss` joue GMFSS union seul pour une lecture, avec ou sans l'option : la
comparaison du natif et de TensorRT sur le même extrait.

```sh
# 0. Première ouverture : la fusion et la paire entières, l'IFNet et le flot se construisent ; les lignes
#    « GMFSS natif retenu pour la fusion… » et « … pour la paire… » donnent les dB (50 visés) et les temps,
#    « GMFSS natif écarté… » dit pourquoi TensorRT garde la fusion ou la paire.
kirari --debug play --gmfss-native --motion=gmfss --rife=2 --start=960 extrait.mkv 2>&1 | grep -i "gmfss natif"

# 1. Vitesse, GMFSS union seul ×2 puis ×3 avec SPAN : TensorRT (A) contre natif (B), en A, B, B, A.
for x in 2 3; do
  kirari compare extrait.mkv --a="--motion=gmfss --rife=$x --start=960" \
    --b="--gmfss-native --motion=gmfss --rife=$x --start=960" --seconds=30 --report=gmfss-x$x.txt
done
#    Le tri des paires (il demande la carte de la scène), sur le natif :
kirari compare extrait.mkv --a="--gmfss-native --motion=gmfss --rife=2 --start=960" \
  --b="--gmfss-native --motion=gmfss --rife=2 --start=960 --scene-map --pair-triage" --seconds=30 \
  --report=gmfss-tri.txt
#    Le temps de chaque image et les étages (GMFSS paire native, GMFSS fusion native) :
kirari --debug play --gmfss-native --motion=gmfss --rife=2 --start=960 --stats=n2.csv extrait.mkv

# 2. Justesse sur l'extrait : deux exports sans perte, TensorRT puis natif (l'export porte sur tout l'extrait,
#    puis la lecture reprend : fermer le lecteur), et leur PSNR (50 dB visés).
kirari --mute --motion=gmfss --rife=2 --keys=1:export-lossless=t.mkv extrait.mkv
kirari --mute --gmfss-native --motion=gmfss --rife=2 --keys=1:export-lossless=n.mkv extrait.mkv
ffmpeg -i n.mkv -i t.mkv -lavfi psnr -f null - 2>&1 | tail -1
```

Le paquet Linux change d'empreinte TensorRT (`tools/release/trt-fingerprint.sh`) : `src/gpu` et `src/prepare` ont
changé.
