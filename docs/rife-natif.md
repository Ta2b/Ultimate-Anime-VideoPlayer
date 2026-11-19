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

- **mma.sync FP16**, sommes en f16, mixtes ou f32 selon l'essai ; m16n8k16 dès sm_80, deux m16n8k8 sur sm_75
  (Turing), mêmes fragments. Mixtes : chaque prise (Kc produits au plus) s'accumule en f16, au débit plein des
  tensor cores des GeForce, puis s'ajoute en f32 ; l'arrondi f16 ne porte plus sur les 9 · cin produits de la
  couche. Simulé sur une couche 3 × 3 (activations après LeakyRelu, poids N(0, 1/K)) : f16 de 56 à 67 dB contre
  la somme exacte selon cin, mixtes de 69 à 74 dB, f32 au-delà de 130 dB ;
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
- **première couche de l'encodeur** lue dans les trois plans de l'image d'entrée, sans copie ;
- **pas 2 sans conflit de banques** : la tuile d'entrée est rangée colonnes paires puis impaires, les 16 lignes
  d'un `ldmatrix` restent des pixels voisins (au pas 2 brut, deux lignes tombaient sur les mêmes banques) ;
- **fragments B préchargés** : les poids du pas k suivant se lisent pendant les mma du pas courant ;
- **K découpé sur les petites grilles** : une couche dont la grille fait moins de deux vagues (les blocs
  grossiers, 30 × 17 pixels au 1/32 de 1080p, ne donnaient que 36 blocs de calcul pour 60 SM) répartit ses prises
  × tranches entre plusieurs blocs, sommes partielles en f32, puis `reduce_kernel` ajoute le biais, le résiduel
  et l'activation.

Rien n'est propre à 4.25 : les couches se lisent dans les morceaux ONNX de la découpe (`rife_native_layers`, une
chaîne Conv, activation, Mul, Add, ConvTranspose), donc 4.22 lite, 4.26 et 4.25 lite (poids déjà permutés) passent
par le même code. Une structure inconnue est refusée et TensorRT garde la chaîne.

## Comparaison avec le RIFE natif d'AnimeJaNai

Lu dans the-database/animejanai-inference (`src/native/gconv.cuh`, `gconv.cu`, `rife.cpp`), rien recopié. Même
famille de noyaux (mma.sync FP16 en sommes f16, tuile et halo en mémoire partagée, poids rangés en fragments B,
K par tranches de 16 à 64).

Là où nous faisons mieux :

- **la transposée de fin de bloc** : nous la calculons en quatre convolutions 2 × 2 (16 prises en tout) ; eux en
  quatre convolutions 3 × 3 dont une partie des poids est nulle (36 prises). Environ 2,2 fois moins de calcul
  sur cette couche ;
- **la réutilisation des poids** : un warp calcule 2 lignes × 32 colonnes (4 fragments A par fragment B lu),
  contre 1 ligne × 16 colonnes chez eux ; quatre fois moins de lectures de poids par mma ;
- **l'encodage gardé d'une paire à l'autre** (RIFE fusionné) : eux refont l'encodeur des deux images à chaque
  instant ;
- **Turing** : nos noyaux tournent dès sm_75 ; les leurs s'arrêtent (`__trap`) sous sm_80 ;
- **les sommes f16 bornées** au plus grand fini à l'arrondi, plutôt qu'un infini qui se propage.

Là où ils faisaient mieux, repris ici (mêmes idées, notre code) : le pas 2 rangé par parité de colonne, et le K
découpé sur les petites grilles (leur `ksplit`). Restent chez eux :

- **la variante de tuile mesurée** sur la carte au chargement (8, 4 ou 2 warps, le plus rapide gardé), quand
  nous choisissons par une règle. Notre essai au chargement pourrait chronométrer deux ou trois choix de tuile
  par couche ; à faire si le profil de la séance finale montre une couche mal servie ;
- **l'occupation** : leur tuile prend moins de registres (16 accumulateurs contre 64 chez nous en CT = 64),
  donc plus de warps par SM pour cacher la latence. C'est l'envers de notre réutilisation des poids ; seul le
  profil dira lequel l'emporte sur les convolutions de RIFE ;
- **la colle** (hors de ces noyaux) : un seul rassemblement par bloc à sa propre échelle (`k_gather`), lot E
  du chantier AnimeJaNai.

## Essai au chargement

Sous `--rife-native`, les six moteurs TensorRT se construisent et se chargent comme avant (en E/S HWC8), puis une
paire de test (ondes et carrés, B décalée de (5, 3) pixels, instant 0,5) passe par la chaîne avec TensorRT, puis
avec les noyaux natifs en sommes f16, mixtes et f32 : trois passes chacune (directe, capture du graphe CUDA,
rejeu), la dernière comparée. Une variante tient si elle fait 45 dB ou plus contre TensorRT (images au pic 1 ;
flot au pic de son plus grand écart, masque à part).

Les variantes qui tiennent sont chronométrées avec TensorRT, comme le SPAN FP8 (`span_timing.h`) : une passe de
lecture ×2 (B encodée seule, graphes rejoués), onze tours intercalés (TensorRT, les natifs, puis l'inverse) d'une
passe de chauffe et deux passes mesurées, la médiane des tours. La plus rapide n'est gardée que si elle bat
TensorRT d'au moins 2 % ; sinon TensorRT garde la chaîne, même quand le natif tient le seuil. Le journal donne
tout : « RIFE natif retenu, sommes mixtes (1920×1088, contre TensorRT : f16 41.8 dB, mixtes 47.0 dB, f32 49.3 dB,
seuil 45 dB ; passe de lecture ×2 : TensorRT 4.98 ms, mixtes 4.70 ms, f32 5.60 ms) » (chiffres d'exemple).

Mesure sur 4070 Ti avant les sommes mixtes et la mesure du temps (RIFE 4.25 seul, 1080p) : f16 41,8 dB, f32
49,3 dB, retenu en f32 et 13 % plus lent que TensorRT (5,62 ms contre 4,98 ms par image créée). Le f32 natif somme
presque exactement : ses 49,3 dB contre TensorRT disent surtout l'écart entre deux calculs FP16 qui arrondissent
à des endroits différents (TensorRT peut sommer en f16, ses fusions arrondissent ailleurs). Aucune variante
native ne passera nettement au-dessus contre TensorRT ; les 50 dB de la cible se jugeraient mieux contre une
référence f32.

Une valeur non finie d'un côté ou de l'autre (référence TensorRT comprise) écarte le natif. Un essai qui fait
perdre le contexte CUDA (accès fautif d'un noyau) fait échouer le chargement au lieu de rendre une chaîne
TensorRT morte ; tout autre refus laisse TensorRT, chargé et intact. La lecture des morceaux ONNX refuse ce que la
diffusion d'ONNX lirait autrement que les noyaux : un bêta ou une pente de forme C ou 1×1×1×C (le long de la
largeur), une transposée qui porte `output_shape`.

## État

Écrit et compilé (nvcc 13.5, sm_75, 86, 89, 120, `--warn-on-spills -Werror`, aucun débordement de registres),
relu, jamais lancé sur une carte : la justesse se juge à l'essai au chargement, la vitesse à la séance finale.

Audit de novembre 2026 : un émulateur sur hôte des noyaux (ldmatrix et mma selon les tables PTX, chaque accès
borné et aligné, mémoire partagée jamais lue avant écriture) a rejoué l'encodeur et les blocs de 4.25, 4.26,
4.22 lite et 4.25 lite, de 2×2 à 1152×1920, de 1 à 170 SM, sommes f16 et f32 : sorties justes à 1e-7 près
contre une convolution en double, sans accès hors limites. Deux pistes de vitesse, à mesurer avant d'y toucher :
le split-K coupe les unités à parts égales, une part peut chevaucher deux tranches de canaux et charger deux
tuiles (couper par tranche quand il y en a assez) ; sur Turing (64 Ko par SM), les tuiles Kc 64 (jusqu'à
48 960 octets) ne laissent qu'un bloc par SM, Kc 32 en laisserait deux.

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
