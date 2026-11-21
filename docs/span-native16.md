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
- **porte finale et conv_cat dans un lancement**, essai `KIRARI_AB_SPAN_GATE_CAT=1` pour le Balanced
  b8f64 et le Performance b5f48 directs en sommes FP16. Le dernier bloc garde les codes FP16 de la
  porte en mémoire partagée.
  Le banc ABBA vérifie la sortie entière au bit près en 1080p et sur un recadrage 642×358. Les médianes
  du Balanced vont de 11,26–11,66 à 10,95–11,05 ms, les minima de 10,50 à 10,37–10,39 ms. Les médianes
  du Performance à poids partagés vont de 5,03–5,04 à 4,90–4,91 ms, les minima de 4,98 à 4,86 ms.
  En lecture 4K avec RIFE, le test du Balanced donne 156,23 et 156,26 i/s après composition sur
  le même extrait de 7 s. Pour le Performance, le test de 6 s donne 162,99 et 163,16 i/s,
  tandis que la médiane GPU de SPAN passe de 6,61 à 6,38 ms ;
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
hauteurs de 1 à 540 lignes et des bandes de 1 à 1000. Avec la zone pleine, les tuiles qui bougent se
passent des bandes pour garder la grille de l'image entière.

Le banc balaie la hauteur (`BANDS`, 0, 64, 192 et 270 en plus du défaut). Avec EXPORT, il dépose aussi la
sortie sans bandes, que `justesse_cpu.py` compare au bit près à celle du défaut : l'ordre des bandes ne change
aucune somme.

### Repli compact après un refus de mémoire

Si l'allocation d'essai ou le redimensionnement de l'arène partagée échoue par manque de mémoire, le
chargeur peut reprendre un SPAN FP16 à 64 canaux et sommes FP16 avec des bandes de 32 lignes. S0 reste
entier pour la queue du réseau. Les cinq autres tampons utilisent 64 lignes chacun, avec le halo. Le
masque de tuiles reste aligné sur la grille de l'image entière et les mêmes convolutions valident le
résultat au chargement.

Sur Balanced 1920×1080, l'arène passe de 379,69 à 100,78 Mio. Les deux points de refus ont été testés dans
le vrai chargeur avec un pool CUDA privé de 192 Mio. Les deux replis et le retour sans plafond gardent
exactement les mêmes 24 883 200 valeurs. Le chronométrage sans pression ne montre pas de gain stable.

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

## Winograd par paires de lignes, 48 et 64 canaux

Le noyau transforme les 3×3 à 48 ou 64 canaux dans le sens vertical, sur deux lignes de demi-résolution à la fois.
Il garde les trois prises horizontales directes. Pour chaque prise, quatre fragments A restent dans les
registres et se multiplient aux quatre lignes de poids transformées, contre six produits sur la même paire
en direct. Les poids sont préparés en double puis arrondis en FP16. Les sommes des MMA restent en FP16 et
la combinaison des quatre résultats se fait en FP32. La troisième convolution de chaque bloc applique
dans la même tuile la 1×1 de la porte, le résidu et tanh.

Sur la RTX 4070 Ti, Balanced 1920×1080 → 3840×2160 passe de 10,244 à 8,980 ms en graphe CUDA, en bandes
de 128 lignes sur une image Re:Zero. C'est 12,3 % de temps réseau en moins. Le banc mesure le réseau seul.
Sur trois autres recadrages Re:Zero de 642×358, les deux modèles Balanced et Balanced Sharp1 gagnent de
0,086 à 3,018 dB contre ONNX Runtime FP32. Le 99,9ᵉ percentile de l'erreur est meilleur dans chacun des six
cas. Le plus grand écart augmente de 0,003326 à 0,003513 dans un cas Sharp1 et baisse ou reste égal ailleurs.
Toutes les valeurs sont finies. Les sorties des bandes de 35 lignes et sans bandes sont identiques au bit.
Le masque partiel a été vérifié sur une zone de travail initialisée à NaN, coins et centre compris.

Sur la mire saturée à 1080p, les deux résultats GPU ne se ressemblent qu'à 53,4 dB dans la pire bande.
La comparaison indépendante à ONNX Runtime FP32 tranche : pour Sharp1, cette bande passe de 54,074 à
55,679 dB et son erreur maximale de 0,011253 à 0,010622. Les deux autres bandes gagnent respectivement
0,207 et 3,399 dB. Un seuil de 60 dB contre le direct rejetait cette amélioration.

La grille par tuiles conserve ses 32 × 8 pixels de demi-résolution. Le halo vertical compte deux lignes par
3×3 transformée, car l'arrondi empêche d'annuler exactement le dernier terme de la paire. En bandes, le
corps avance de paire en paire. Un panoramique de deux pixels d'entrée en horizontal et quatre en vertical
conserve la phase du modèle et peut réemployer sa sortie. Un décalage vertical de deux pixels est refusé
pour la reprise.

Le chargeur essaie le noyau sur Ada pour Balanced C64 à huit blocs en 1080p avec des sommes FP16.
Il le retient si les trois bandes de l'image d'essai restent à au moins 50 dB du direct, si le temps médian
gagne au moins 3 % et si le contrôle contre le processeur passe. L'outil `kirari_span_fp8_run` permet
de mesurer les deux voies avec `--winograd-vertical` et `--no-winograd`. Le noyau CUDA est compilé en
SM89 et SM120. Au vrai chargement Ada, Sharp1 passe de 10,18 à 9,00 ms et Balanced de 9,92 à 8,82 ms. Le
contrôle indépendant du coin avec le processeur donne 66,920 dB pour Sharp1 (contre 67,189 en direct)
et 66,617 dB pour Balanced (contre 66,568 en direct).

Avec Performance C48, trois warps de 16 canaux traitent chaque paire sans padding. Le chronométrage réseau
ABBA donne 5,096 → 4,057 ms en 1080p, 2,326 → 1,764 ms en 720p et 1,326 → 1,100 ms en 540p. Sharp1 gagne
également aux trois tailles. Sur les six recadrages des deux modèles, le PSNR gagne de 0,062 à 2,817 dB
contre ONNX Runtime FP32. Il y a une hausse du p99,9 de 0,002853 à 0,002860 avec Performance et une hausse
du maximum de 0,006148 à 0,006640 avec Sharp1, sur des recadrages différents. Les autres valeurs de ces
deux métriques baissent ou restent égales. Toutes les sorties sont finies. Bandes, masques et phases
autorisées gardent leurs bits sur les deux variantes.

En lecture avec Sharp1 Performance à 720p et composition HDR à 240 Hz, le temps GPU de SPAN passe
d'environ 2,67 à 2,13 ms. Sur les deux alternances, l'affichage passe de 224,27 à 221,54 puis de
225,84 à 222,54 nouvelles images par seconde. Sharp1 Performance garde le calcul direct à cette taille.

En 1080p sur Ada, le chargeur essaie une variante du paquet officiel Sharp1 Performance FP16 qui garde
la deuxième convolution de chaque bloc directe. Les mêmes contrôles de parité, de vitesse et de référence
CPU décident du repli. Le réglage expérimental Sharp1 Performance Winograd permet de revenir au direct.
Chaque masque de couche majore le halo des convolutions restantes. Une convolution directe de Sharp1
ajoute une ligne verticale, une convolution Winograd en ajoute deux. À cinq blocs, ce calcul retire une
rangée de halo en haut et en bas sur six masques internes, là où l'image laisse cette marge.

Sur le paquet Performance C48 officiel en 1080p, le chargeur départage les sommes FP16 avec un
calcul en sommes FP32 si leur parité descend sous 50 dB. Le candidat doit
égaler ou améliorer l'erreur quadratique et le 99,9ᵉ centile d'erreur dans chaque bande. Les tests
de vitesse et de parité CPU restent actifs. Le temps au chargement passe de 5,02 à 4,07 ms. Avec
RIFE 4.25 et le compositeur HDR 4K à 240 Hz, le corps SPAN passe de 8,51 à 5,85 ms et Wayland
confirme 209,55 → 211,97 images nouvelles/s.

Le pas vertical agrandit `SpanTileGrid`, renvoyé par le backend au lecteur. L'ABI passe à 62 et l'empreinte
inclut la taille de cette grille. Le lecteur demande donc un backend recompilé.

## Winograd F(2×2, 3×3), 64 canaux

Les 3×3 carrées des blocs (trois par bloc, 85 % du calcul du b8f64) passent par Winograd F(2×2, 3×3) :
Y = Aᵀ [U ⊙ V] A, seize produits par sortie 2×2 au lieu de trente-six, soit 2,25 fois moins de mma. Le
plafond des seuls mma du b8f64 tombe de 6,3 à 3,0 ms. `span_native_wino.cuh` (`wino_kernel`), derrière
`SpanFp8Weights::winograd` (`kirari_span_fp8_run --winograd`), éteint par défaut : il a perdu au banc
(ci-dessous) ; 64 canaux et sommes f16 seulement (le cran Qualité garde la convolution directe ; à 48
canaux, les 12 sorties par warp ne tombent pas sur les tuiles n16).

- U = G g Gᵀ se calcule à l'envoi, en double, arrondi en f16 (`pack_wino`), et se range comme les poids
  directs avec 16 points à la place des 9 positions : 128 Ko par couche au lieu de 72.
- Le bloc garde la tuile 8 × 32 de `conv_kernel` (mêmes bandes, même masque des tuiles, même épilogue :
  `finish`), soit 4 × 16 tuiles 2×2 en deux moitiés de 32. Le calcul réserve quatre points V d'une
  même ligne transformée. Les fils relisent deux lignes de l'entrée, puis calculent les quatre V
  avec les mêmes sommes f32 et l'arrondi f16 de la version précédente. Après une barrière, chaque
  warp multiplie V par U pour ses 16 canaux de sortie. Chaque M_ξ est borné à ±65 504 avant
  l'accumulation f32 selon Aᵀ, dans le même ordre. Deux barrières par groupe au lieu de deux par
  point ramènent le coût des deux moitiés de 64 à 16 barrières. La zone partagée du noyau passe
  de 48 128 à 45 056 octets.
- Les tuiles 2×2 partent des lignes paires de l'image quelle que soit la bande (une bande qui commence à une
  ligne impaire calcule la ligne d'avant sans l'écrire) : en bandes, la sortie est celle de l'image entière
  au bit près.

Justesse : sur les vrais modèles (`tools/bench/span_native/winograd_db.py`, numpy avec les arrondis des
noyaux, contre ONNX Runtime FP32), Winograd gagne même un peu (b8f64 65,93 → 65,98 dB, b5f48 66,68 → 70,11)
parce que ses sommes f16 sont plus courtes. Sur l'émulateur des noyaux (48×80, poids tirés au hasard,
contre le processeur en f32) : 72,13 dB contre 71,81 au chemin direct, aucune valeur non finie, bornes de la
mémoire partagée tenues, même sortie au bit près en bandes de 5 et de 8 lignes et sans bandes. Sur GPU, le
test « Winograd suit le chemin direct » demande 50 dB contre le direct et la même sortie en bandes.

Au banc (#980, RTX 4070 Ti, 1080p → 4K, Re:Zero, 3 tours ABAB) : **13,72 ms au b8f64 contre 10,64 en
direct (+29 %)**, à 62,96 dB contre 62,83 (export jugé sur processeur contre ONNX Runtime FP32 : juste sur
GPU aussi). Au b5f48 (48 canaux), sans effet comme prévu : 5,03 ms. La voie F(2×2, 3×3) reste donc
éteinte par défaut. La version groupée donne 13,92–14,34 ms contre 14,57–14,60 ms pour l'ancienne dans le même aller-retour
(RTX 4070 Ti, graphe CUDA, 100 lancements par passage, entrée Re:Zero, bande de 128 lignes). Le direct
est à 10,61–11,50 ms. Le résultat Winograd reste identique à l'octet sur cette image 4K.
Sur une convolution d'une bande profilée avec Nsight Compute, l'indicateur d'attente de barrière
passe de 0,28 à 0,20 cycle par instruction émise. L'occupation reste autour de 17 % et cette convolution
isolée n'accélère pas. Il faut réduire les transferts et la pression sur les registres avant d'évaluer
F(4×4, 3×3), dont la borne des seuls mma est de 1,8 ms et dont les arrondis sont différents.

Les convolutions sans porte appliquent maintenant le biais et le ReLU directement depuis les fragments
Winograd et écrivent chacune des deux demi-tuiles. L'aller-retour des fragments par la disposition de
la convolution directe disparaît à cet endroit. Avec la même entrée, en bandes de 128 lignes et en ABBA
de 70 relectures de graphe, on mesure 13,24–13,72 ms contre 13,89–14,46 ms auparavant. Les quatre sorties
sont identiques à l'octet. Le direct garde l'avantage sur le modèle Balanced.

## 21 novembre 2026 — Tuiles du Winograd vertical

Sur deux images en 1080p du Balanced C64, le banc rejoue 80 graphes par passage en ABBA. Le calcul
partagé des transformées verticales coûte 9,93–10,23 ms, contre 9,11–9,62 ms sur les références appariées.
Les sorties sont identiques. Les premières captures Nsight diffaient par la grille de lancement
(30×32 contre 30×63), donc leurs compteurs bruts ne départagent pas les deux codes.

La tuile de 16×4 conserve les arrondis mais double le nombre de blocs. Elle coûte 9,87 et 9,91 ms sur les
deux images, contre 8,86 et 9,16 ms en 32×4. La tuile de 16×8 conserve le nombre de blocs et réduit de
12 % le halo partagé. Elle gagne sur la première image (9,23 → 8,74 ms) et perd sur la seconde
(8,85 → 9,20 ms). Écrire la transformée inverse directement en mémoire globale donne un résultat
variable et un minimum plus lent sur les deux images. Toutes ces variantes donnent les mêmes octets
que la référence et restent hors du lecteur.

La tuile 16×8 à huit warps donne également les mêmes octets. Elle passe de 8,87 à 9,37 ms et de
9,25 à 9,36 ms sur les deux images. Même les minima reculent. Elle est écartée.

Sur les mêmes images, les déroulages de 1, 2 et 4 pas ne gagnent pas de façon répétable sur les 3 pas
du lecteur. La sortie reste identique. Dérouler 6 pas déverse des registres sur Ada. Sur les quatre ONNX,
aucun groupe contigu de quatre poids d'une 3×3 de bloc n'a deux zéros FP16. La sparsité matérielle 2:4
demanderait de modifier les poids.

Sur la première convolution ReLU d'un bloc, les seize produits séparés en GEMM groupées cuBLAS
avec Winograd F(2×2, 3×3) prennent 1,050 ms en bandes de 32 lignes, contre 0,341 ms pour le noyau
Winograd vertical. Les poids sont ceux du modèle Balanced et les activations du premier filtre
proviennent d'une image réelle. Les transformées et la réécriture des résultats coûtent au moins
30 Mio de temporaires par bande dans ce banc. Le test de 8 192 sorties face à la convolution directe
en double donne une erreur RMS de 0,000064 contre 0,000077 pour le Winograd vertical. L'essai porte
sur une couche et la variante reste hors du lecteur.

Dans le noyau F(2×2, 3×3) qui garde les transformées sur la tuile, éliminer les contributions
structurellement nulles de la reconstruction réduit la médiane de cette couche de 428,7 à 415,1 µs.
Les 33 177 600 codes FP16 sont égaux sur la même image. Le temps reste supérieur à la référence
verticale mesurée à 341,4 µs dans le banc précédent.

La zone de travail Winograd verticale compacte passe de 398 à 145 Mo en bandes de 64 lignes,
et à 224 Mo en bandes de 128 lignes. Le banc Ada conserve les octets de sortie sur deux images
dans les deux configurations. Les bandes de 64 lignes rallongent les médianes sur les deux
images. Les bandes de 128 lignes donnent un résultat variable. Le candidat reste au banc.

Pour une entrée 1920×1080, le décompte sur les convolutions des ONNX officiels, à 160 TFLOP/s en
accumulation FP16, donne les bornes arithmétiques suivantes. Il compte le premier filtre avec un K
rembourré de 12 à 16 canaux, les portes 1×1 et les autres convolutions à leur taille réelle. Le calcul
des deux Winograd sur les deux axes porte sur les 3×3 des blocs. Ces bornes excluent les transformées,
le trafic entre couches et la présentation.

| Modèle | Direct | Vertical F(2,3) | F(2×2,3×3) | F(4×4,3×3) |
| --- | ---: | ---: | ---: | ---: |
| Balanced C64, avec ou sans Sharp1 | 6,290 ms | 4,379 ms | 3,105 ms | 1,991 ms |
| Performance C48, avec ou sans Sharp1 | 2,329 ms | 1,657 ms | 1,209 ms | 0,817 ms |

La voie Performance Sharp1 qui garde la seconde 3×3 de chaque bloc en direct a une borne de 1,881 ms.
Les 90 % du débit de calcul du Balanced vertical correspondent à une passe en 4,866 ms.
Le [calculateur SPAN](../tools/bench/span_native/plafonds.py) lit ces formes dans les ONNX officiels.
Il donne aussi un modèle qui compte un transfert VRAM par tenseur et par étage, ainsi que les
transformées et la fusion de la dernière porte avec la concaténation. Avec cette hypothèse, le
Balanced vertical arrive à 8,160 ms et F(4×4, 3×3) à 8,169 ms malgré la baisse du nombre de MMA.
Le traitement en bandes peut servir une partie de ces transferts depuis le L2. Un ratio fondé
sur ce modèle de trafic demande d'abord une mesure des octets réellement transférés.

À 165 Hz, une lecture de 5 s à partir d'une source 1080p à 23,976 i/s demande environ 192 i/s avec
RIFE 4.25 à ×8. Le Balanced FP16 natif produit le 3840×2160. Wayland confirme 693 images successives en
fenêtre SDR, puis 694 en plein écran HDR10, à 165,00 i/s dans les deux cas. Les intervalles confirmés
couvrent environ 4,2 s par passage. Les journaux ne signalent aucune perte GPU. Le plein écran HDR
mesure 6,612 ms de médiane pour le SPAN et 2,062 ms pour RIFE. Le profil isolé utilise le grade sans LUT
personnelle. Le plafond de l'écran et les temps d'exécution des deux réseaux se mesurent séparément.

Un rejeu prolongé en plein écran HDR10 couvre 20,21 s confirmées par Wayland avec la même demande
de cadence et le même mode du lecteur. Sans carte NVOF, il donne 3 300 images nouvelles à 163,41 i/s.
Avec la carte et l'option d'éclats, il donne 3 296 images à 163,22 i/s. Les deux journaux comptent zéro
perte GPU. La carte traite 504 paires, à 0,34 ms de médiane sur le moteur NVOFA. Ce rejeu du même
extrait mesure la cadence présentée et le coût de la carte, indépendamment de l'évaluation visuelle
des éclats.

Sur une seconde lecture en fenêtre SDR, le lecteur met à l'échelle une source 1920×1080 à 120 i/s
vers 3840×2160 avec chacun des quatre modèles et l'interpolation coupée. Les intervalles confirmés
couvrent environ trois secondes, après le démarrage. Le lecteur signale huit pertes GPU sur chaque
Balanced et aucune sur les Performance. La réutilisation par tuiles s'applique à 58,2 % des images
calculées du Balanced, à 59,3 % de celles du Balanced Sharp1 et à 66,2 % pour chaque Performance.
Les temps associent donc des passes entières et partielles.

Pour les grilles de 64 colonnes ou moins, le lecteur et le moteur calculent les halos en
combinant des lignes de bits. Les mêmes tuiles sont marquées. À 30×68 et 10 % de tuiles
modifiées, le microbanc du code final passe de 52,8 à 17,4 µs pour cinq masques. Avec 25 % de
tuiles modifiées, il passe de 87,0 à 19,2 µs. Ces durées concernent la préparation des masques.
Le plan du lecteur combine les deux dilatations du halo en une seule et reprend le premier compte
de la région après l'élargissement du cœur. Les masques et les compteurs restent identiques.

Les rectangles de tuiles couverts par les blocs de mesure sont calculés quand la géométrie change.
Le banc du plan complet passe de 270 à 102 µs de médiane sur une mesure 120×68 et de 166 à 67 µs
avec 90×51 blocs de côté fractionnaire. Le rejeu compare les décisions et masques octet pour octet
sur 15 600 images, avec changements de mesure, pertes de mesure et redémarrages, sous UBSan.

| Modèle FP16 | Cadence Wayland | Passes SPAN, médiane | Passes SPAN, p95 |
| --- | ---: | ---: | ---: |
| Balanced | 118,72 i/s | 7,298 ms | 10,819 ms |
| Balanced Sharp1 | 118,33 i/s | 7,282 ms | 10,855 ms |
| Performance | 120,03 i/s | 3,625 ms | 5,165 ms |
| Performance Sharp1 | 119,94 i/s | 3,590 ms | 6,219 ms |

### Trafic des quatre calculs denses

Une image de Re:Zero 1920×1080, mêmes recettes Winograd qu'en lecture, bandes de 128 lignes et
aucune tuile omise. Nsight Compute relève les octets DDR lus et écrits, les octets au L2 et la
somme des durées des noyaux. Le banc laisse le cache et les fréquences sous le contrôle du pilote.

| Modèle | Durée CUDA | DDR lue + écrite | Trafic L2 | Temps optimiste pour ces octets |
|---|---:|---:|---:|---:|
| Balanced | 8,171 ms | 2,464 Go | 16,991 Go | 6,18 ms |
| Balanced Sharp1 | 8,101 ms | 2,397 Go | 17,028 Go | 6,20 ms |
| Performance | 4,062 ms | 1,240 Go | 6,563 Go | 2,46 ms |
| Performance Sharp1 | 4,676 ms | 1,437 Go | 5,929 Go | 2,85 ms |

Le temps optimiste prend le plus grand des deux transferts estimés. La DDR utilise le pic de
504 Go/s. Le L2 utilise 2,748 To/s, mesurés par un noyau de lecture et écriture sur deux tampons
de 16 Mio. Ces repères supposent un trafic identique, un recouvrement idéal et aucune latence
supplémentaire. Pour Balanced, les convolutions Winograd comptent pour environ 90 % du trafic
L2. Un regroupement de tuiles qui relit moins de poids peut donc rapprocher le lecteur de la
borne après fusion. Le scénario Winograd 2D nécessite ses propres compteurs.

La somme des bornes par lancement affine ce calcul. Pour chaque lancement, elle prend le
maximum entre le temps du trafic DDR, celui du trafic L2 et celui des MMA FP16 exécutées à
160,4 TFLOP/s. Le quotient rapporte la somme au temps CUDA de la capture dense.

| Modèle | Borne agrégée | Part de la borne atteinte | Temps correspondant à 90 % |
|---|---:|---:|---:|
| Balanced | 6,804 ms | 83,3 % | 7,560 ms |
| Balanced Sharp1 | 6,724 ms | 83,0 % | 7,471 ms |
| Performance | 3,090 ms | 76,1 % | 3,433 ms |
| Performance Sharp1 | 3,308 ms | 70,8 % | 3,676 ms |

Ces chiffres dépendent des noyaux et du trafic de cette capture. Une nouvelle fusion change
la borne. Les ratios de ce tableau portent sur le réseau en calcul dense.

Pour Performance et Performance Sharp1 à 1920×1080 sur Ada, la dernière porte Winograd peut
enchaîner la concaténation en mémoire partagée. Les six warps de la convolution restent en place.
Quatre suffisent à la concaténation, un par ligne. L'image complète garde les mêmes codes sur
deux images et chacun des deux modèles. Par alternance ABBA, les minima des passes Performance
sont de 3,76–3,80 ms sans fusion et de 3,63–3,69 ms avec elle. Pour Sharp1, ils passent de
4,05–4,12 à 3,96–3,97 ms. Les passes avec masque de tuiles gardent leur chemin séparé.
En fenêtre, sur la source à 120 i/s avec Sharp1, le p95 de l'ensemble des passes SPAN passe
de 5,936 à 5,308 ms sur un A/B. Les médianes sont de 3,579 et 3,603 ms, environ deux tiers
des passes utilisent un masque, et Wayland confirme 363/363 images nouvelles dans les deux cas.

Le réemploi des poids entre deux paires de lignes de la convolution verticale C64 a aussi
été testé sur l'image entière. Les deux sorties Balanced restent identiques au bit près,
mais les minima du réseau passent de 8,173–8,233 à 8,179–8,314 ms. Cette forme reste au banc.
Un double tampon partagé de 16 Kio pour les poids de la convolution verticale par CTA
garde également les mêmes codes. Les minima du réseau Balanced passent de 7,97–8,89
à 9,88–9,92 ms. Cette forme ajoute des barrières, et la taille du tampon la limite à
deux blocs résidents par SM.

Sur Ada en 1080p dense, les deux Balanced laissent les halos voisins dans le L1 lors de la
copie asynchrone de la convolution verticale déroulée. Les poids gardent leur chemin. Les
deux images du banc Balanced gardent les mêmes codes sur toute la sortie 4K. Les minima
passent de 8,025–8,229 à 7,911–7,952 ms. Donner une priorité L1 aux poids seuls donne un
résultat variable. Les registres et le nombre de blocs résidents ne changent pas.

En fenêtre, source 120 i/s, sans interpolation ni réemploi SPAN, les deux passages témoins
et les deux passages avec le L1 donnent les résultats suivants. La colonne CUDA agrège les
médianes des passes. Wayland confirme toutes les images nouvelles dans le segment mesuré.

| Modèle | CUDA témoin | CUDA avec L1 | Wayland témoin | Wayland avec L1 |
|---|---:|---:|---:|---:|
| Balanced | 10,414–10,470 ms | 10,099–10,136 ms | 95,18–95,55 i/s | 96,85–96,91 i/s |
| Balanced Sharp1 | 10,419–10,471 ms | 10,091–10,116 ms | 94,29–94,49 i/s | 97,21–98,27 i/s |

Deux convolutions Winograd ReLU enchaînées dans un anneau partagé gardent tous les codes sur
la couche 960×540 et deux recadrages. En C64, elles passent d'environ 0,57–0,63 à 1,01–1,33 ms.
En C48, elles passent de 0,39–0,42 à 0,72–1,04 ms. Le calendrier C64 double la construction du
fragment A pour la même sortie et réduit la résidence de trois à deux blocs. L'anneau reste au banc.

Le Winograd vertical F(4,3) emploie six coefficients pour quatre lignes de sortie, au lieu de
huit coefficients en deux paires. Avec les points ±0,75 et ±1,5, l'erreur RMS baisse face à un
direct FP64 sur les sorties échantillonnées à 960×540. À C64, elle passe de 7,97 à 7,07 × 10⁻⁵
et la médiane augmente de 337 à 346 µs. À C48, la forme à six warps passe de 254 à 249 µs et
l'erreur RMS de 8,01 à 7,05 × 10⁻⁵. Le maximum d'erreur augmente sur un recadrage malgré la
baisse de la RMS. Le réseau complet et la cadence composée restent à qualifier pour cette forme.

Un banc du Performance complet a depuis comparé le front en deux et quatre lignes, avec la
dernière porte et la sortie RGB inchangées. Le changement de front seul garde tous les codes.
Avec les ReLU F4 et le front de quatre lignes, la sortie 4K conserve 73,38 dB face à ONNX Runtime,
comme la référence verticale F2. Avec les portes F4 aussi, elle atteint 73,64 dB. Les médianes
respectives sont de 3,964, 4,073 et 3,977 ms pour F2, F4 ReLU et F4 ReLU avec portes. Cette
géométrie reste au banc. Le passage des portes de N16 à N8 garde les codes du F4 sur les
24 883 200 valeurs RGB. Quatre passages par forme donnent 3,862 ms de médiane avec les
portes F4 en N16, 3,902 ms en N8 et 3,950 ms avec le témoin F2. Les plages se recouvrent,
de 3,824 à 3,927 ms pour F4 N16 et de 3,882 à 4,225 ms pour F2. Le lecteur et les autres
scènes restent à vérifier avant de retenir cette variation de géométrie.

Quatre warps N24 sur les deux ReLU de chaque bloc réutilisent la prise A entre trois groupes
de huit sorties. Sur six images de trois scènes, le corps jusqu'au RVB passe de 3,980 ms en F2
à 3,856 ms, contre 3,971 ms pour le F4 à six warps. Les deux F4 donnent les mêmes bits sur les
six images. Face à ONNX Runtime FP32, la PSNR et la SSIM moyenne progressent sur les six images.
Le pire écart couleur passe de 0,003917 à 0,004027 sur [0, 1], et celui de la variation entre deux
images de 0,005180 à 0,005697. La RMS de cette variation diminue sur les trois paires.

Avec une source recadencée à 180 i/s, sans réemploi ni masques, le moteur du lecteur passe de
5,038–5,051 à 4,635–4,670 ms en médiane, et de 5,686–5,772 à 5,254–5,477 ms au p95. Wayland
confirme 165,00 puis 160,08 i/s avec F2 et 165,00 i/s dans les deux passages F4. Sur la source
à 120 i/s, les quatre passages se maintiennent autour de 120 i/s après composition.

Sur une image de banc et 76 lancements du corps jusqu'au RVB, la capture Nsight Compute passe
de 262,04 à 207,98 GFLOP Tensor exécutés et de 1,282 à 1,117 Go DDR. Le trafic L2 passe de
6,253 à 6,965 Go. La somme des bornes par lancement, avec les mêmes débits que le tableau
ci-dessus, passe de 2,969 à 3,070 ms. La capture exclut l'entrée conv_1 et la composition.

Avec réemploi, le chemin tuilé F4 garde les 24 883 200 codes du F4 dense sur chacune des six
images rejouées, y compris des translations et des bords de régions. En lecture, les quatre
passages alternés sur la source à 180 i/s atteignent 165,00 i/s confirmées par Wayland.
Le halo vertical augmente de quatre à huit rangs de tuiles. Le lecteur recalcule par tuiles
66,2–66,3 % des images avec F2 et 56,4 % avec F4. Le temps médian du moteur passe de
3,085–3,529 à 3,635–3,695 ms, et le p95 reste dans la même plage de 4,873–5,165 ms.
Le F4 reste au banc.

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
des noyaux natifs (graphe CUDA tous deux, bandes par défaut), les poids partagés et Winograd forcés dans les
mêmes tours, et les natifs à d'autres hauteurs de bande et en
f32. EXPORT est à donner seulement quand les noyaux ont changé : après les temps, le banc dépose pour chaque
modèle le modèle, l'entrée et une sortie de chaque mode (f16 aux bandes du lecteur, f16 sans bandes, f16 aux
poids partagés, f16 par Winograd, f32), en demi-flottants, un plan par fichier (16,6 Mo en 4K). Dans le cloud :

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

Banc de #980 (mêmes conditions, 3 tours ABAB, export jugé sur processeur contre ONNX Runtime FP32) : b8f64
TensorRT FP16 14,70 ms, natif f16 10,64 ms (0,72×) à 62,83 dB, poids partagés 11,11, Winograd 13,72 à
62,96 dB, bandes 0, 64, 192 et 270 à 11,01, 10,97, 10,76 et 11,62 ms, f32 14,45 ms à 76,93 dB ; b5f48
TensorRT 8,76 ms, natif f16 5,03 ms (0,57×) à 63,93 dB, sans poids partagés 5,22, bandes 5,54, 5,16, 4,88 et
5,15 ms, f32 6,00 ms à 77,15 dB. Les bandes de 192 lignes (4,88 ms au b5f48) battent le défaut ici mais
perdaient au banc de #963 (5,30 contre 5,21) : à remesurer avant de changer auto_band.
