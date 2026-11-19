# TemporalFix natif (`--tfix-native`)

Novembre 2026. Les convolutions de TemporalFix fusionné ([`temporalfix.md`](temporalfix.md), « Fusionné ») par nos
propres noyaux CUDA, à la place des cinq moteurs TensorRT : l'encodeur (une image), les trois blocs du flot et la
tête de confiance (lots de six appuis). La colle, les pyramides, les warps et le mélange (`tfix_fused_kernels.cu`,
`tfix_blend_kernels.cu`) restent les mêmes ; seul le rangement de leurs tampons change (par pixel au lieu de
plans). Option d'essai, éteinte par défaut ; TensorRT reste la référence et le secours.

Cible : égaler puis battre TensorRT (27,9 ms par image à pleine taille et 6,2 ms à demi-taille en 1080p sur la
RTX 4070 Ti, mesures d'avant le cache de l'encodeur), à 50 dB ou plus contre TensorRT FP16. Repère : le
TemporalFix natif d'AnimeJaNai fait 8,1 ms par image en 1080p sur une RTX 5090.

## Le réseau, vu par les noyaux

Relevé sur s1, s2 et s3 (les trois mêmes structures, poids différents) :

| Morceau | Convolutions | Sortie |
|---|---|---|
| Encodeur (une image) | 3 → 16 pas 2, 16 → 16, 16 → 16, 16 → 4, puis Resize ×2 | 4 traits à la taille du réseau |
| Bloc 0 (1/8) | 15 → 48 pas 2, 48 → 96 pas 2, 4 ResConv 96, 96 → 11 | 11 canaux au 1/32 |
| Bloc 1 (1/4) | 26 → 32 pas 2, 32 → 64 pas 2, 4 ResConv séparables 64, 64 → 11 | 11 canaux au 1/16 |
| Bloc 2 (1/2) | 26 → 24 pas 2, 24 → 48 pas 2, 4 ResConv séparables 48, 48 → 11 | 11 canaux au 1/8 |
| Tête (1/2) | 25 → 32, 32 → 32, 32 → 1 | 1 canal au 1/2 |

Toutes les convolutions sont des 3 × 3 de bord 1, LeakyRelu 0,2 ; un ResConv est `act(conv(x) · β + x)`. Un
ResConv séparable est une convolution par canal 3 × 3 suivie d'une 1 × 1, sans activation entre elles.

## Choix

- **Séparables repliées.** La convolution par canal D puis la 1 × 1 P forment une seule 3 × 3 exacte :
  W[o][prise][c] = P[o][c] · D[c][prise], b[o] = Σ P[o][c] · d[c] + p[o] (`prepare/rife_native.cpp`,
  `fold_pointwise`). Elle passe par les mêmes noyaux mma que le reste, en un lancement et une lecture de la
  mémoire au lieu de deux. Coût en 1080p pleine taille : 33 milliards de multiplications pour les blocs 1 et 2
  au lieu de 13, sur 110 pour la fenêtre (la tête en prend 72) ; sur des activations aussi petites (au plus
  6 × 135 × 240 × 64), la mémoire et les lancements pèsent plus que le calcul. Vérifié sur processeur (numpy,
  poids repliés en réels contre onnxruntime sur les morceaux de s2) : 69 à 85 dB, l'écart étant l'arrondi FP16
  d'onnxruntime. Si le profil montre les blocs 1 et 2 lents, un noyau par canal puis 1 × 1 fondu en mémoire
  partagée reste possible.
- **Six appuis en un lot** (`BatchParams`, `rife_native_kernels.cuh`) : blockIdx.z = image · (grille z d'une
  image) + reste, chaque image décalée de son pas. Le lot est un paramètre de modèle des noyaux, et ses champs
  sont hors de `Params` : les noyaux de RIFE natif gardent exactement leurs registres d'avant. Les blocs grossiers remplissent ainsi la carte (bloc 0 :
  34 × 60 points au 1/32 de 1080p, six fois) ; le K découpé ne sert que s'il manque encore des blocs.
- **Par pixel** (NHWC) : la colle écrit X0 au pas 16, X1, X2 et l'entrée de la tête au pas 32 (canaux en trop à
  zéro), et lit Y au pas 16 et la sortie de la tête au pas 8 (`kernels::tfix_in_ld`, `tfix_out_ld`,
  `TfixShape::native`, `tfix::View`). Ni copie ni transposition entre la colle et les convolutions.
- **L'encodeur** s'arrête à sa dernière convolution (`TfixSplit::encode_last`, coupée au chargement) ; le Resize
  ×2 demi-pixel du graphe est refait par `traits_kernel`, qui écrit les quatre plans de traits là où le
  moteur TensorRT les écrivait. La pyramide qui suit ne change pas.
- **Sommes f16, puis mixtes, puis f32** (`RifeNativeSums`, comme RIFE natif) : celles des tensor cores des
  GeForce à plein débit d'abord ; mixtes (chaque prise en f16, les prises entre elles en f32) puis f32 si
  l'essai le demande. AnimeJaNai garde ses 1 × 1 en f32 ; nos séparables repliées somment 9 · C termes (576 au plus)
  en f16, comme le bloc 0 (864) : l'essai tranche.

## Essai au chargement

Sous `--tfix-native`, les cinq moteurs TensorRT se construisent et se chargent comme avant. Puis :

- **encodeur** : une image de test (ondes, carrés aux bords francs, scintillement de ±3 %) passe par le moteur
  TensorRT, puis par les noyaux natifs ; les traits sont comparés ;
- **fenêtre** : sept images de test qui glissent de (2, 1) pixels et scintillent, leurs traits de test et
  leurs pyramides, passent trois fois (directe, capture du graphe CUDA, rejeu) avec TensorRT puis avec le
  natif. Sont comparés, chacun à son pic (le plus grand écart à zéro, au moins 1) : la sortie, le flot de Y2
  (canaux 0 et 1, en pixels), le reste de Y2 (masque et attributs du dernier bloc) et la sortie de la tête.
  Une sortie seule pourrait tenir le seuil même si le réseau se trompait, le mélange restant prudent ; le flot
  à part, son pic en pixels ne cache pas une erreur du masque.

Les passes natives tournent sur une zone de travail à elles : la zone de la voie ne grandit que pour un natif
retenu. Une valeur non finie, chez TensorRT ou chez nous, compte pour un échec.

Gardé à 45 dB ou plus sur le pire des quatre, sommes f16, puis mixtes, puis f32, **et seulement s'il bat
TensorRT en temps** d'au moins 2 % (tours intercalés de deux passes, médiane de onze, comme GMFSS natif :
`gmfss_trial::times`) ; sinon TensorRT garde l'étage, chargé et intact. L'encodeur et la fenêtre se décident
chacun pour soi : l'encodeur peut être natif et la fenêtre TensorRT. Le journal dit ce qui est retenu :
« TemporalFix natif retenu pour l'encodeur (960×540, sommes f16 : 64,7 dB contre TensorRT, seuil 45 dB ; une
passe : TensorRT 0,34 ms, natif 0,25 ms) ».

## Comparaison avec AnimeJaNai

Lu dans the-database/animejanai-inference (`src/native/tfix.cpp`, `tfix_kernels.cu`, `gconv.cu`), rien recopié.
Mêmes idées : NHWC, Concat écrits en place par la colle, cache de l'encodeur et des pyramides par image, toute
la fenêtre en un graphe CUDA, bêta replié dans les poids.

Différences :

- **les six appuis** : eux les font l'un après l'autre sur trois flux (un seul au-delà de 1,5 Mpx, « limité par
  la mémoire ») ; nous en un lot, un lancement par couche pour les six ;
- **les séparables** : eux en deux noyaux (par canal en f32, 1 × 1 en mma f32), nous repliées en une 3 × 3 ;
- **le mélange** : eux accumulent appui par appui dans une image f32 pleine taille relue six fois ; notre
  mélange (inchangé) lit les six appuis en un noyau, sommes en registres ;
- **la variante de tuile** : eux la chronomètrent au chargement ; nous la choisissons par une règle (même
  remarque que [`rife-natif.md`](rife-natif.md)).

## La colle et la tête (novembre 2026, après le premier profil)

Le profil par noyau (nsys `--cuda-graph-trace=node`) et les plafonds couche par couche
([`reports/tfix-natif/plafonds.md`](reports/tfix-natif/plafonds.md) : fenêtre à 1,08 ms en demi-taille, 0,22 ms tout fondu)
ont montré que les convolutions natives étaient déjà près de leur plafond, deux fois plus rapides que TensorRT,
et que l'écart venait d'ailleurs :

- **la colle par pixel** : `first_kernel`, `glue_kernel` et `conf_kernel` écrivaient leurs 15 à 32 canaux deux
  octets à la fois, au pas d'un point (64 octets) d'un fil à l'autre, et relisaient Y de même (10 à 28 fois
  leur plafond). Un point s'écrit et se lit maintenant en mots de 16 octets (`tfix::store_point`,
  `load_point`, `View::pair` dans `tfix_fused.cuh`) ; le chemin en plans de TensorRT garde ses calculs. La
  fenêtre native est passée de 3,32 à 2,02 ms par passe à l'essai ;
- **la tête** (`tfix_head.cu`) : ses trois convolutions (25 → 32 et 32 → 32 activées, puis 32 → 1) en un noyau,
  par tuiles de 16 × 8 points. L'entrée, halo de trois points compris, est copiée une fois en mémoire partagée
  ; la première convolution en tire 20 × 12 points, la deuxième 18 × 10, la dernière les 16 × 8 sur les cœurs
  CUDA (deux fils par point, sommes f32). Les deux activations du milieu (200 des 390 Mo de la fenêtre en
  demi-taille) ne quittent plus la puce. Tensor cores (mma m16n8k16, mêmes sommes f16, mixtes et f32), huit
  warps et 45 Ko par bloc : deux blocs par SM sur Ada, et la tuile tient sur Turing. Les index des fragments
  ont été vérifiés par une émulation sur processeur contre une convolution directe (écart nul, bords compris).
  Une tête d'une autre forme passe couche par couche.

Fondre aussi `conf_kernel` dans la tête serait plus lent : pour son halo de trois points, conf calculerait
22 × 14 points pour 128 utiles, et 2,4 fois ses cellules pleine taille, pour 100 Mo épargnés.

## État

Mesuré sur la RTX 4070 Ti (Re:Zero, demi-taille, Normal, `--one-lane`), une passe de l'essai au chargement :

| | justesse (f16) | natif | TensorRT | retenu |
|---|---|---|---|---|
| encodeur | 64,7 dB | 0,11 ms | 0,12 ms | natif |
| fenêtre, couche par couche | 56,2 dB | 3,32 ms | 2,14 ms | TensorRT |
| fenêtre, colle en 16 octets | 56,2 dB | 2,02 ms | 2,05 ms | TensorRT (1,5 %, sous la marge de 2 %) |
| fenêtre, tête fondue (8 warps) | 56,2 dB | 2,01 ms | 2,16 ms | natif |

En lecture, la fenêtre native fait 7,33 ms en médiane (p95 7,53) contre 7,71 en TensorRT, l'encodeur 0,29 ms.
Les noyaux y durent environ trois fois leur temps d'essai, pour TensorRT comme pour le natif : la carte, peu
chargée par un épisode à 24 i/s, ne monte sans doute pas à ses fréquences hautes. Le profil de lecture donne
la tête fondue à 2,91 ms (32 % du GPU), `conf_kernel` à 1,03 ms : la tête reste le premier poste (plafond
d'environ 0,26 ms à l'essai, halo compris), à regarder avec ncu.

## Mémoire

Par pixel, les tampons de la fenêtre grossissent (26 → 32 et 25 → 32 canaux) ; la zone de travail des noyaux
garde deux tampons de la plus grande activation (la tête couche par couche : 6 × 32 canaux à demi-résolution
du réseau), soit environ 400 Mo à pleine taille en 1080p et 100 Mo en demi-taille (le défaut). La tête fondue
n'en a pas besoin ; la zone reste réservée pour une tête qui passerait couche par couche.

## Séance finale (sur le PC, binaire `linux-release` puis `windows-release`)

Le code touche `src/gpu`, `src/kernels` et `src/prepare` : il demande un nouveau paquet TensorRT Linux
(`tools/release/trt-fingerprint.sh`) avant d'arriver chez un utilisateur.

Extrait : Re:Zero S4E18, `--start=960`, 60 s, 1080p ; stabilisation Normale, demi-taille (le défaut) puis
pleine taille (`temporal_fix_full`), sans interpolation ni agrandissement d'abord. ABAB, trois tours.

```sh
# 0. Première ouverture sous compute-sanitizer (code jamais lancé sur une carte : un accès hors des tampons se
#    voit ici, pas dans les dB), puis sans : les lignes « TemporalFix natif… » donnent les dB.
compute-sanitizer --tool memcheck --target-processes all \
  kirari --debug play --tfix-native --no-upscale --start=960 --seconds=5 extrait.mkv
kirari --debug play --tfix-native --no-upscale --start=960 extrait.mkv 2>&1 | grep -i "temporalfix natif"

# 1. Temps par image, ABAB (A = natif, B = TensorRT) : médianes « TemporalFix fusionné fenêtre » et
#    « … encodeur » du profil GPU de fin, et le fichier --stats.
for tour in 1 2 3; do
  kirari --debug play --tfix-native --no-upscale --one-lane --start=960 --stats=A$tour.csv extrait.mkv
  kirari --debug play --no-upscale --one-lane --start=960 --stats=B$tour.csv extrait.mkv
done

# 2. Noyaux : Nsight Systems sur 30 s, avec et sans --tfix-native. La fenêtre tourne en graphe CUDA
#    (fused_graphs.cpp) : --cuda-graph-trace=node en détaille les noyaux.
nsys profile -o tfix-natif --capture-range=none --duration=30 --cuda-graph-trace=node \
  kirari --debug play --tfix-native --no-upscale --start=960 extrait.mkv
nsys profile -o tfix-trt --capture-range=none --duration=30 --cuda-graph-trace=node \
  kirari --debug play --no-upscale --start=960 extrait.mkv
nsys stats --report cuda_gpu_kern_sum tfix-natif.nsys-rep tfix-trt.nsys-rep
```

Décision : natif par défaut s'il tient 50 dB et va plus vite que TensorRT ; sinon le profil dit quel morceau
perd (la tête, les séparables repliées, les tuiles) avant de redécider. Ensuite seulement, et seulement si le
natif est meilleur en qualité et en vitesse : ne plus construire les moteurs TensorRT quand il est retenu
(79 s gagnées au premier lancement en 1080p), avec une référence f32 maison pour l'essai. Tant que la mesure
ne l'a pas montré, la construction reste : elle est la référence de justesse et le secours.
