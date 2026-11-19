# RIFE natif (`--rife-native`)

Novembre 2026. L'encodeur et les convolutions des blocs de RIFE fusionné ([`rife-fusionne.md`](rife-fusionne.md))
par nos propres noyaux CUDA, à la place des six moteurs TensorRT. La colle entre blocs, les warps et la fin
(`rife_fused_kernels.cu`, `warp_lib`) ne changent pas : le natif lit et écrit les mêmes tampons, dans les mêmes
formats (X et Y en HWC8), sans reformatage. L'encodage d'une image reste calculé une fois pour deux paires, le tri
des paires, l'Auto et `held_pair` ne voient aucune différence.

Cible : RIFE 4.25 ×2 sur la RTX 4070 Ti à 23 % d'occupation ou moins (environ 5,3 ms par image créée), contre 30 %
aujourd'hui (6,9 ms) et 23 % pour le RIFE natif de mpv AnimeJaNai ; 50 dB ou plus contre TensorRT FP16.

## Pourquoi pas TensorRT ici

TensorRT est le meilleur outil pour les gros calculs réguliers (un transformeur comme GMFlow, de grosses
multiplications de matrices) et reste notre référence de justesse. Les convolutions de RIFE sont l'inverse :
petites (16 à 192 canaux, de 30 × 18 à 960 × 576 pixels), limitées par la mémoire et par la latence des
lancements. TensorRT y perd sur trois points que le profil Nsight de novembre montre (4,03 ms d'IFNet par paire
contre 3,28 ms pour les noyaux faits main d'AnimeJaNai) : il choisit des noyaux de génération Turing (`sm75 stage1`)
sur une carte Ada, il garde des sommes en f32 (moitié du débit des tensor cores d'une GeForce), et il ne fond pas
le résiduel des ResConv dans la convolution. Le natif supprime aussi la construction des six moteurs au premier
lancement (47 s), dès qu'on n'a plus besoin d'eux comme référence.

Le mélange reste juste pour GMFSS : GMFlow sous TensorRT, le reste natif (`/mnt/project-files/chantier-natif/gmfss-natif.md`).

## Ce que font les noyaux

`src/gpu/cuda/backend/rife_native_kernels.cuh` (une famille de convolutions), `rife_native_net.cu` (le réseau),
`rife_fused_native.cpp` (l'essai au chargement), `prepare/rife_native.h` (les poids lus dans les morceaux ONNX).

- **mma.sync FP16**, sommes en f16 par défaut, en f32 si l'essai le demande ; m16n8k16 dès sm_80, deux m16n8k8
  sur sm_75 (Turing), mêmes fragments ;
- **tuile en mémoire partagée** de 8 × 32 pixels (4 × 32 en pas 2), halo compris, copiée par `cp.async` au pas
  Kc + 8 (les lignes d'un `ldmatrix` sur des banques distinctes en pas 1) ; les canaux d'entrée par tranches de
  Kc = 16, 32 ou 64 pour rester sous 48 Ko, ceux de sortie par tuiles de 16, 32 ou 64 (blockIdx.z), la plus large
  qui laisse deux vagues de blocs sur la carte : les blocs grossiers (192 canaux au 1/32) prennent des tuiles
  étroites ;
- **ResConv fondu** : le bêta est replié dans les poids et le biais au chargement, l'addition résiduelle et le
  LeakyRelu se font à l'écriture de la sortie ; deux tampons alternent, l'entrée reste lisible ;
- **transposée 4 × 4 de pas 2** en quatre convolutions 2 × 2, une par phase de sortie, écrites directement dans Y
  au pas 56 ; celle de l'encodeur (4 canaux) en une convolution 3 × 3 à 16 sorties (phase · 4 + canal) qui écrit
  les quatre plans d'encodage ;
- **première couche de l'encodeur** lue dans les trois plans de l'image d'entrée, sans copie.

Rien n'est propre à 4.25 : les couches se lisent dans les morceaux ONNX de la découpe (`rife_native_layers`, une
chaîne Conv, activation, Mul, Add, ConvTranspose), donc 4.22 lite, 4.26 et 4.25 lite (poids déjà permutés) passent
par le même code. Une structure inconnue est refusée et TensorRT garde la chaîne.

## Essai au chargement

Sous `--rife-native`, les six moteurs TensorRT se construisent et se chargent comme avant (en E/S HWC8), puis une
paire de test (ondes et carrés, B décalée de (5, 3) pixels, instant 0,5) passe par la chaîne avec TensorRT, puis
avec les noyaux natifs : trois passes (directe, capture du graphe CUDA, rejeu), la dernière comparée. Gardé à
45 dB ou plus (images au pic 1 ; flot au pic de son plus grand écart, masque à part), sommes f16 d'abord, f32
ensuite ; sinon TensorRT garde la chaîne. Le journal dit ce qui est retenu, avec les dB : « RIFE natif retenu
(960×576, sommes f16 : 52,3 dB contre TensorRT, seuil 45 dB) ».

## État

Écrit et compilé (nvcc 13.5, sm_75, 86, 89, 120, `--warn-on-spills -Werror`, aucun débordement de registres),
relu, jamais lancé sur une carte : la justesse se juge à l'essai au chargement, la vitesse à la séance finale.

## Séance finale (sur le PC, binaire `linux-release` puis `windows-release`)

Extrait : Re:Zero S4E18, `--start=960`, 60 s ; 1080p, sans agrandissement (comme le profil de novembre), puis
avec SPAN. RIFE 4.25 ×2, variante mouvement (le défaut), puis ×3. ABAB, trois tours, rien d'autre d'ouvert.

```sh
# 0. Première ouverture : les moteurs HWC8 se construisent, la ligne « RIFE natif retenu… » donne les dB.
kirari --debug play --rife-native --rife=2 --no-upscale --start=960 extrait.mkv 2>&1 | grep -i "rife natif"

# 1. Temps par image créée, ABAB (A = natif, B = TensorRT) ; relever la médiane « moteur RIFE fusionné »
#    du profil GPU de fin et le fichier --stats.
for tour in 1 2 3; do
  kirari --debug play --rife-native --rife=2 --no-upscale --one-lane --start=960 --stats=A$tour.csv extrait.mkv
  kirari --debug play --rife=2 --no-upscale --one-lane --start=960 --stats=B$tour.csv extrait.mkv
done

# 2. Occupation du GPU (la cible des 23 %) : Nsight Systems sur 30 s, avec et sans --rife-native.
nsys profile -o rife-natif --capture-range=none --duration=30 \
  kirari --debug play --rife-native --rife=2 --no-upscale --start=960 extrait.mkv
nsys profile -o rife-trt --capture-range=none --duration=30 \
  kirari --debug play --rife=2 --no-upscale --start=960 extrait.mkv
nsys stats --report cuda_gpu_kern_sum rife-natif.nsys-rep rife-trt.nsys-rep

# 3. Puis ×3, et avec SPAN (sans --no-upscale), mêmes commandes.
```

PSNR natif contre TensorRT : la ligne du journal (étape 0) ; 50 dB ou plus attendu. Décision : natif par défaut si
il tient 50 dB et passe sous 23 % d'occupation ; sinon le profil dit quel noyau perd (tuiles, tranches, fusion de
plusieurs ResConv d'un bloc grossier en un lancement) avant de redécider. Ensuite seulement : ne plus construire
les moteurs TensorRT quand le natif est retenu (47 s gagnées au premier lancement), et DistilDRBA.
