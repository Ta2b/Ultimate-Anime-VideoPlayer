# GMFSS natif

GMFSS Fortuna Union ([`interpolation.md`](interpolation.md)) sur la RTX 4070 Ti. Le GridNet (45 convolutions),
FeatureNet, MetricNet et l'IFNet d'Union tournent par nos noyaux CUDA sur les formes retenues lors de la
reprise du 20 novembre. L'essai au chargement juge la paire, la fusion, l'IFNet et GMFlow contre les moteurs
TensorRT correspondants. GMFlow a une chaîne CUDA complète qui doit encore franchir ce contrôle de fidélité.
`--no-gmfss-native` laisse GMFSS à TensorRT et le retire du menu. Les mesures de reprise et les anciennes
estimations sont dans [`gmfss-bilan.md`](gmfss-bilan.md).

Le chantier s'est fait par étapes, chacune derrière `--gmfss-native` :

1. le tri des paires pour GMFSS (`--pair-triage` : dessins tenus, paires impossibles, panoramiques purs) ;
2. les lectures des splats sur la bibliothèque de warp ([`warp-lib.md`](warp-lib.md)) ;
3. **le GridNet de la fusion** (ce document) ;
4. **FeatureNet et MetricNet de la paire** (ce document) ;
5. **GMFSS union de retour au menu**, seulement derrière le natif (ce document) ;
6. GMFlow natif, s'il égale TensorRT FP16 en qualité et en vitesse (GMFlow n'a pas de FP8) ; sinon TensorRT
   reste.

## Le GridNet de la fusion

La fusion exportée par `tools/gmfss/export_gmfss.py` est l'IFNet d'union (half0, half1, time → l'image de l'IFNet),
remplacé par RIFE 4.26 dans celle de `tools/gmfss/ifnet_rife.py` (l'image s'y appelle « rife »), suivi
du GridNet de FusionNet_u : 21 blocs (têtes, résiduels, descentes, montées par transposée 4 × 4 de pas 2, chacun
PReLU, convolution, PReLU, convolution) sur trois lignes (64, 128 et 192 canaux à la moitié, au quart et au
huitième des plans remplis), puis la queue (convolution et PReLU, convolution 64 → 256 et PixelShuffle(2),
convolution 64 → 3, bornée à [0, 1]). Environ 750 GFLOP par fusion en sortie 1280 × 768, 45 convolutions.

Sous `--gmfss-native`, le lecteur ne construit plus la fusion seule mais un manifeste « .gmfssfused »
(`prepare/gmfss_fused.h`) : la fusion entière (le même moteur qu'avant, repris du cache) et son IFNet extrait en un
petit modèle, que trtexec construit avec les réglages de RIFE. L'image de l'IFNet se trouve par la structure, pas
par son nom : la deuxième des trois images du Concat que lit la première convolution du GridNet
(`gmfss_ifnet_tensor`) ; la coupe est exacte sur les deux exports (écart nul sous ONNX Runtime). Jusqu'au
19 novembre 2026, seule la fusion d'`ifnet_rife.py` se coupait : celle du 20 octobre du PC restait entière à
TensorRT (« « rife » n'est ni une entrée du morceau… »). Au chargement (`tensorrt_gmfss_fused.cpp`), les poids
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

Sur Ada à 960×576 dans le GridNet Union de 45 convolutions, les deux groupes résiduels suivis d'une
remontée et d'une addition peuvent enchaîner leurs quatre couches par bandes. Les trois anneaux
réutilisent le tampon du résiduel. Les entrées des deux branches restent vivantes jusqu'à la fin de
l'addition pour éviter d'écraser un halo par une sortie écrite en avance. Le plan garde l'ordre
d'accumulation du réglage FP16, y compris lorsqu'il choisit deux tranches de 32 canaux.

Le chargeur choisit d'abord les précisions et l'IFNet avec le plan normal. Il compare ensuite l'image
FP16 complète avec et sans anneaux. Un écart conserve le plan normal. Sur deux groupes réels à
960×576, la somme des médianes des groupes passe de 3,94 à 3,47 ms avec K32, à sortie bit à bit et à
taille d'arène inchangée.

Les transposées de ces jonctions partagent maintenant un halo entre les deux phases horizontales.
Chaque phase garde ses poids, son ordre de produits et son arrondi FP16. Le chargeur vérifie toujours
la sortie native entière. `KIRARI_AB_GMFSS_JOIN_PHASES=0` reprend les phases séparées pour un essai
de comparaison sur la même arène.

À cette même taille, certaines formes de la première convolution sur x0 lisent directement les neuf
plans FP16 et appliquent le PReLU dans la tuile. Le message au chargement confirme un deuxième contrôle
bit à bit de l'image entière. Sur les formes mesurées, la préparation et cette convolution gagnent
0,056 à 0,082 ms, sans changer l'ordre des produits ni les arrondis.

Les entrées en plans (warped*_1..3, l'image de RIFE) passent par pixel dans x0 à x3. Pour les caractéristiques,
chaque tuile lit 32 pixels consécutifs par plan et écrit 32 canaux contigus par pixel. Les trois niveaux
sont identiques au bit près à l'ancien rangement. La sortie revient en plans à la fin.

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

Les caractéristiques d'une image ne dépendent que d'elle. Le chemin natif reprend les trois
niveaux de B quand la prochaine paire prend cette même source pour A. Le lecteur vérifie le numéro d'ingestion,
y compris lorsqu'il adopte une paire calculée en avance. Les nouvelles sorties prennent les banques libres,
et A référence directement la banque de B. Le backend ordonne ces trois entrées en lecture seule, y compris
sur la voie des splats. Une paire disjointe recalcule les deux côtés. L'essai `--no-gmfss-feature-reuse`
refait le calcul complet sur toutes les paires.
Le banc sur la RTX à 1920×1152 trouvait les mêmes codes FP16 en copiant B. Ce transfert coûtait 0,60–0,65 ms
contre 2,71–2,78 ms pour refaire FeatureNet d'un côté. En Instantané sur Re:Zero, le journal compte 51 paires
réemployées sans copie (21,06 ms de médiane) et 45 paires avec deux côtés calculés (24,48 ms de médiane).
Ces durées comprennent les autres travaux de la paire et proviennent de groupes de paires différents.

## Splats de la fusion

Sur Ada, les huit splats de la fusion préparent un index des sources par destination. Si deux
travaux consécutifs partagent le mouvement, sa grille et l'instant, le second reprend l'index. Chaque fil
rassemble ses contributions en FP32 et écrit le résultat normalisé directement en FP16. Les caractéristiques
des niveaux à 128 et 192 canaux peuvent écrire un groupe C8 aligné en une fois. L'ordre des additions FP32
peut changer, comme entre deux ordres d'exécution du splat atomique.

Les destinations qui reçoivent plus de 128 contributions passent par une accumulation FP32 dédiée à ces
pixels. Le nombre de contributions borne la taille de ce tampon. Le choix est fait sur le GPU et les autres
destinations conservent le rassemblement. Le splat par étapes reste disponible pour les autres chemins,
les dimensions dépassant 552 960 pixels ou si le tampon d'index manque de mémoire.

## Sommes et vitesse

Les noyaux font leurs sommes en f16, mixtes (chaque prise en f16, les prises entre elles en f32) ou f32, comme
pour le RIFE natif ([`rife-natif.md`](rife-natif.md)) ; la tuile de 64 canaux de sortie n'existe qu'en f16.
Chaque réseau natif est coupé en morceaux qui prennent chacun leurs sommes (« Essai par morceaux et réglage »
ci-dessous) ; le mélange le plus rapide qui tient le seuil est chronométré contre TensorRT (`gmfss_trial.h` :
onze tours intercalés, la médiane) et n'est gardé que s'il le bat d'au moins 2 %. Sinon TensorRT garde la
fusion ou la paire. Le seuil et le chronométrage portent sur les entrées de cet essai. Les différences
d'arrondi et de temps sur les scènes réelles se vérifient séparément.

## Essai par morceaux et réglage

L'ancien essai passait chaque réseau entier en f16, en mixtes ou en f32, chaque convolution à une forme fixe
(« Résultats du PC » : la paire et la fusion retenues en f16, 12 % et 14 % plus rapides que TensorRT). L'essai
coupe maintenant chaque réseau en morceaux, comme le mélange du RIFE natif, et chaque convolution a sa forme
réglée au banc.

**Les morceaux** (`gmfss_grid_pieces` de `prepare/gmfss_native.cpp` pour la fusion, `gmfss_pair_native_net.cu`
pour la paire) :

| Unité | Morceau | Convolutions | Tête |
|---|---|---|---|
| fusion | ligne 0 | celles qui écrivent un tenseur de 64 canaux (la moitié des plans remplis) | aucune |
| fusion | ligne 1 | 128 canaux (le quart) | aucune |
| fusion | ligne 2 | 192 canaux (le huitième) | aucune |
| fusion | queue | la convolution avant le PixelShuffle, celle du PixelShuffle, `conv_last` | `conv_last` |
| paire | FeatureNet | les six, sur chaque image | les trois sorties (`block{b}.3`) |
| paire | MetricNet | les cinq | `metric_out` |

Une convolution va à la ligne du tenseur qu'elle écrit : une descente à la ligne d'en dessous, une montée à celle
d'au-dessus. Les trois lignes ont chacune leur largeur et leur taille, donc leur coût et leur erreur ; la queue
écrit l'image, là où le f16 coûte le plus (l'émulation ci-dessous). Un découpage plus fin multiplierait les
passes de l'essai (chaque option de chaque morceau est mesurée sur l'unité entière) pour des morceaux de deux
convolutions sur 45. La coupe ne lit aucun nom de tenseur de l'export : la queue part de la convolution qui
précède le PixelShuffle, la ligne de chaque tenseur est celle du plan. Sur les deux exports de la fusion (IFNet
d'union, IFNet de RIFE 4.26), elle donne 14, 16 et 12 convolutions aux lignes 0, 1 et 2, et 3 à la queue
(`tests/unit/prepare/gmfss_native_test.cpp` sur la fusion du dossier des modèles).

**Les crans** (`gmfss_mix.h`) : f16, « f16 et tête f32 » (un morceau à tête seulement), mixtes, f32. La tête
d'un morceau est la couche qui écrit une sortie de l'unité. TensorRT n'est pas une option d'un morceau : la
fusion et la paire sont chacune un seul moteur. Il reste la référence, et le repli de l'unité entière.

**Le choix** (`gmfss_trial.cpp`, sur `native_mix.h` ; le RIFE natif a son propre choix, `rife_mix_choice.h`) :

1. une passe TensorRT (la référence) et une passe native f32 partout (la base) ; une valeur non finie de l'une
   ou de l'autre, et TensorRT garde l'unité ;
2. le temps seul de chaque option de chaque morceau, les autres au f32 (cinq tours intercalés, la médiane) ;
3. l'effet de chaque option, ce morceau seul hors de la base, mesuré sur l'unité entière contre la base ; les
   crans f16 et mixtes de tout le réseau (les variantes de l'ancien essai) sont mesurés tels quels, le mélange
   ne fait donc jamais moins bien qu'eux ;
4. le mélange le plus rapide (la somme des temps seuls) dont l'erreur prévue tient 45 dB contre TensorRT,
   vérifié sur l'unité. Un raté recale la corrélation des erreurs ; quatre essais au plus, le dernier ne prend
   qu'un mélange déjà mesuré. Chaque sortie (l'image ; chaque caractéristique, chaque mouvement) est jugée à
   part, l'écart rapporté à son étendue (au moins 1) ;
5. le mélange contre TensorRT, onze tours intercalés, la médiane : gardé s'il le bat d'au moins 2 %, les
   poids des tuiles que ses formes n'emploient pas alors libérés.

**Le réglage** (`gmfss_conv_tune.cu`, à la place du choix fixe de la tuile) : chaque convolution passe au banc
du RIFE natif (`rife_native_bench.cu`) pour chacune des trois sommes, puisque l'essai en change sans rien
reconstruire. La forme de référence (tuile de 16, K entier, sans avance, la plus petite tranche, deux lignes par
warp, une en pas 2) d'abord, puis chaque forme que les noyaux font, sur une sortie remplie de NaN, comparée à la
référence aux tolérances du RIFE (1/16 du pic en f16, 1/32 en mixtes, 1/128 en f32 : `native_sums.h`) et
chronométrée ; la plus rapide de celles qui rendent la sortie de la référence est gardée. Une convolution à une
seule forme n'est pas mesurée ; une convolution jumelle d'une autre déjà réglée (même genre, même géométrie,
même résiduel, même activation : les blocs d'une ligne) en reprend les formes. Le banc ne lance une grande
couche que sur une bande de ses lignes de sortie (`band_of`), 64 Mio au plus par tampon (entrée, sortie,
référence) : il prend ainsi 192 Mio au plus pendant le chargement, au lieu de 810 Mio pour des plans de la paire
à 1920 × 1152 (la queue de la fusion : 270 Mio par tampon) et de 3 Gio à 3840 × 2176. Une couche dont une forme
découpe K (moins de deux vagues de blocs) passe entière ; une bande garde plusieurs vagues, où les formes se
classent comme sur la couche entière, à la traîne de la dernière vague près. La convolution du PixelShuffle a
son genre dans le choix des noyaux du RIFE (`Kind::kShuffle`, `rife_native_pick.cu`, tranche de 64 seulement) :
12 instances de plus, celles du RIFE inchangées (mêmes registres sur sm_75, 86, 89 et 120).

**Émulé contre ONNX Runtime** (`tools/gmfss/emul_mix_native.py`, sur processeur : les arrondis f16 aux endroits
des noyaux, les sommes dans leur ordre, la tranche de l'ancien choix ; le réglage peut en prendre une autre, ce
qui ne change que l'ordre des arrondis f16). Images de la paire à 1024 × 576, la fusion à 512 × 288 (celle
d'`export_gmfss.py`, à l'IFNet d'union ; les entrées du GridNet se trouvent par la structure). Chaque morceau
seul, les autres au f32 : la pire sortie de l'unité en dB contre ORT, entre parenthèses contre le natif f32.

| Morceau | f16 | f16 et tête f32 | mixtes | f32 |
|---|---|---|---|---|
| ligne 0 | 73,4 (73,3) | | 78,0 (78,3) | 78,3 |
| ligne 1 | 78,1 (79,5) | | 78,3 (80,8) | 78,3 |
| ligne 2 | 78,2 (80,0) | | 78,3 (81,3) | 78,3 |
| queue | 67,2 (67,2) | 74,6 (74,9) | 76,4 (77,2) | 78,3 |
| FeatureNet | 72,0 (72,0) | 73,3 (73,4) | 73,3 (76,5) | 73,3 |
| MetricNet | 65,2 (64,8) | 65,8 (65,3) | 71,1 (70,2) | 73,3 |

En mélange : la fusion toute en f16 66,5 (66,5), en f16 et tête f32 72,8 (72,7), toute en mixtes 76,2 (76,3) ; la
paire toute en f16 65,2 (64,8), le FeatureNet en f16 et le MetricNet en mixtes 71,1 (70,2), le MetricNet en
f32 72,0 (72,0). Le pire de la paire est la métrique dès que le MetricNet n'est pas en f32 ; celui de la fusion, la
queue, puis la ligne 0. Tous les mélanges tiennent 45 dB de loin, et 50 dB contre le natif f32 : le choix se fera
sur les temps seuls, et le mélange ne gagne sur le f16 partout que si un morceau est plus rapide à un autre cran, ce
que seul le PC dira ; le gain à attendre tient surtout aux formes réglées par couche. Le f16 de la paire émulé
(65,2 dB contre ORT) rejoint celui du PC (64,8 dB contre TensorRT) ; celui de la fusion (66,5) passe au PC à 62,3 dB
contre TensorRT, qui n'est lui-même qu'à 65,3 dB du natif f32 : les deux écarts s'ajoutent (66,5 et 65,3 dB font
environ 62,8 dB). La fusion d'`ifnet_rife.py` (l'IFNet de RIFE 4.26, images de 512 × 256) donne les mêmes chiffres à
moins de 1 dB : f16 partout 66,5, f16 et tête f32 72,6, mixtes 76,1.

**Queue composée.** Pour le fragment officiel de `models-interp-1` vérifié à 1920 × 1152 sur Ada, les
deux dernières convolutions sont combinées sur la grille à 960 × 576. Les quatre phases produisent les
pixels RGB directement. Le mélange des poids se fait en FP64, les produits de la zone intérieure sont
accumulés en FP32 et les bords gardent leurs filtres et biais propres. L'essai compare l'image au natif
et à TensorRT, borne le pic d'écart et mesure le temps en direct puis sous graphe. L'anneau PixelShuffle
sert de repli.

**Ce que le réglage peut gagner.** Le plafond du GridNet pour des plans de la paire à 1920 × 1152 (la fusion
à 960 × 576), couche par couche le plus lent du calcul (tensor cores f16 de la RTX 4070 Ti, 160 Tflop/s) et des
octets (chaque tenseur lu et écrit une fois, 504 Go/s) : 1 690 Gflop, 11,7 ms (calcul 10,8 ms, octets 8,6 ms ;
canaux complétés à 16 et PixelShuffle à 256 sorties compris). La fusion f16 passe en 21,90 ms, dont environ 20 ms de
GridNet : près de deux fois le plafond, autant que la somme du calcul et des octets. Les 14 convolutions de la
ligne 0 (64 canaux à 960 × 576) sont à l'équilibre, 0,255 ms de calcul contre 0,28 ms d'octets chacune :
recouvertes, 0,28 ms ; l'un après l'autre, 0,54 ms. Or `conv_kernel` charge une tranche de l'entrée (`cp.async`,
attendu aussitôt), puis fait ses mma : un bloc ne recouvre jamais son propre chargement, seuls les autres blocs du
même multiprocesseur le peuvent. La forme fixe (tranche et tuile de 64, deux lignes par warp) prend 192 registres
par fil et 49 Ko de mémoire partagée : deux blocs par multiprocesseur (sm_89). Le réglage peut prendre une ligne par
warp (151 registres, 29 Ko) ou une tranche de 32 (156 registres, 27 Ko) : trois blocs, au prix d'un bord relu plus
grand ou de deux tranches ; une tuile de 32 (jusqu'à six blocs en tranche de 32 et une ligne, 84 registres),
l'entrée alors lue deux fois ; ou les poids à l'avance (la latence des poids, pas celle des images). Il fait donc
mieux recouvrir d'un bloc à l'autre, jamais dans un bloc ; et il ne prend jamais une forme plus lente au banc que la
forme fixe, qui est parmi les siennes. Attendu, sans mesure : 1,5 à 3 ms sur les 21,9 de la fusion (la
ligne 0 de 7,5 vers 5,5 à 6 ms au mieux ; les lignes 1 et 2, au calcul, par les poids à l'avance), 0,3 à 1 ms sur
les 19,7 de la paire, dont les réseaux natifs ne font que 3,5 ms au plafond (FeatureNet 1,15 ms par image,
MetricNet 1,18 ms) : GMFlow, sous TensorRT, en fait l'essentiel. Le reste demande des noyaux nouveaux : un
chargement en double tampon (la bande suivante en vol pendant les mma de la courante, ce qui épargne aussi le bord
relu) : c'est la forme en flux ci-dessous, qui ramène la ligne 0 vers 0,28 ms par convolution (0,42 ms avec un
résiduel). Le modèle de coût lui donne 2,3 à 2,8 ms de moins, et non 3,5 : les résiduels lus comptent, et seules
les 11 convolutions de 64 vers 64 en pas 1 passent en flux.

**La forme en flux** (`conv_stream_kernels.cuh`, calendrier dans `conv_stream.h`). Un bloc descend une colonne de
32 pixels, bande de 4 lignes après bande, toute l'entrée (64 canaux, une seule tranche) dans un anneau de 10 lignes
en mémoire partagée (48 960 octets : deux blocs par multiprocesseur, un seul sur sm_75). La fenêtre d'une bande en
occupe 6 ; les 4 autres reçoivent par `cp.async`, pendant les mma de la bande, les nouvelles lignes de la
suivante ; les deux lignes du bord sont déjà là. L'entrée est lue 1,08 à 1,12 fois (du L2 aux multiprocesseurs),
contre 1,33 fois en deux lignes par warp et 1,59 en une. La sortie se range dans les 4 premières lignes de la
fenêtre, finies, avant de partir. Sommes, PReLU de l'entrée, biais et fin sont ceux de `conv_kernel<64, CT, 3, 1, 1>`
dans le même ordre : la sortie est la sienne au bit près (`tests/unit/gpu/conv_stream_test.cpp` rejoue le noyau sur
l'hôte, sommes arrondies en FP16, contre la tuile d'une tranche), celle qu'émule déjà `emul_mix_native.py` pour
64 canaux en pas 1 (tableau ci-dessus). Trois noyaux : tuile de 64 en sommes f16, de 32 en f16 et en mixtes
(registres sur sm_75, 86, 89 et 120 : 140, 136, 136, 164 ; 88, 88, 88, 118 ; 120, 124, 124, 154 ; sans
débordement). Ce sont des formes de plus au réglage du GMFSS natif, et seulement là (`rn::stream_configs`, 3 × 3 de
pas 1 à 64 entrées, bandes par bloc pour une, deux ou quatre vagues de deux blocs par multiprocesseur) :
chronométrées sur la bande du banc, avec autant de colonnes de blocs que la couche entière, gardées seulement plus
rapides et à la sortie de la référence, jamais forme de référence. Le RIFE natif ne les voit pas : `pick_for`,
`configs` et ses noyaux sont inchangés (les 159 noyaux des fichiers qui lisent ces en-têtes ont le même SASS,
instruction par instruction, sur les quatre architectures).

Le modèle de coût (sm_89, 160 Tflop/s, 504 Go/s) : une convolution de la ligne 0 fait 40,8 Gflop (0,254 ms), un
tenseur 70,8 Mo (0,140 ms). Les 11 convolutions 64 → 64 de pas 1 de la ligne 0, dont 6 lisent un résiduel, font
6,7 ms quand le calcul et les octets se suivent (la forme fixe, à peu près : un bloc ne recouvre pas son propre
chargement), 3,9 ms recouverts au mieux, 4,5 ms à 85 % du calcul et 88 % de la bande. D'où 2,3 à 2,8 ms de moins
sur la fusion, de 21,9 vers 19,1 à 19,6 ms, ce que le réglage des tuiles gagnait déjà sur la ligne 0 compris. Les
3,9 ms visés sont le plafond de ces convolutions dans le tableau des plafonds (3,1 ms), plus la lecture des six
résiduels, que le tableau suppose fondus (0,84 ms) : compté ainsi, le plafond de la fusion est d'environ 12 ms et
non 11,2. Le reste de l'écart tient aux lignes 1 et 2, au calcul, et à la queue.

**Le bloc résiduel fondu n'est pas écrit.** Pour les blocs résiduels de la ligne 0 (r01, r04, r05 : PReLU,
convolution, PReLU, convolution, résiduel), un seul noyau garderait l'image du milieu en mémoire partagée :
141,6 Mo épargnés par bloc (le milieu écrit puis relu, 0,28 ms). Mais la première convolution se refait sur le
halo : une sortie de 4 × 32 demande un milieu de 6 × 34 pixels (13 fragments de 16, la première convolution
1,625 fois), une sortie de 8 × 32 un milieu de 10 × 34 (22 fragments, 1,375 fois). Le bloc passe alors au calcul :
0,67 ou 0,60 ms, contre 0,70 ms recouverts pour ses deux convolutions en flux. L'entrée et le milieu à 64 canaux
prennent 70 848 octets de mémoire partagée en 4 × 32, 111 168 en 8 × 32 au pas des noyaux (ne tient pas), 98 816
sans marge (permutée) : un seul bloc par multiprocesseur, quatre ou huit warps sur 48, et aucun sur sm_75 (64 Ko).
Les registres ne bornent pas (ceux de deux lignes par warp, 192 sur sm_89, pour 13 fragments sur 4 warps), mais
les fragments du milieu ne se partagent pas également entre les warps. Gain idéal : 0,04 à 0,10 ms par bloc, 0,1 à
0,3 ms pour les trois, et 0,26 ms pour head0 (16 → 64 puis 64 → 64) ; à un seul bloc par multiprocesseur, à 65 %
des tensor cores, il perd 0,13 à 0,23 ms par bloc. Au mieux 0,5 ms des 21,9, au pire une perte : pas un noyau de
plus. Il redeviendrait intéressant si la forme en flux tenait la bande (88 %) en laissant les tensor cores sous
60 % (étape 4 de la séance).

**Ce qui reste au PC** : les temps (seul de chaque option, le mélange contre TensorRT) ; les dB contre TensorRT
(l'émulation juge contre ONNX Runtime sur processeur ; au PC, TensorRT FP16 n'était qu'à 67,7 dB du natif f32
sur la paire, 65,3 dB sur la fusion) ; le temps du réglage, et ce qu'il gagne (le temps d'une passe contre les
21,90 et 19,70 ms de la forme fixe ; le temps de chaque morceau par Nsight Systems, étape 3 de la séance
finale) ; et que tout se lance : l'essai par morceaux et le réglage n'ont jamais tourné sur une carte. Pour la
forme en flux (étape 4 de la séance) : si le réglage la prend sur la ligne 0, ce qu'elle y gagne contre la
meilleure tuile, son occupation et ses octets.

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
  à son étendue (au moins 1). Chaque mélange de l'essai (`gmfss_trial.cpp`) passe ainsi sur l'unité entière ; il
  n'est retenu qu'à **45 dB** ou plus ; la séance finale vise 50 dB.
- Compilé par nvcc 13.5 (sm_75, 86, 89, 120, `--warn-on-spills -Werror`), lancé sur la RTX 4070 Ti le
  19 novembre 2026 (ci-dessous) avec l'ancien essai et la forme fixe de chaque convolution. L'essai par
  morceaux, le réglage au banc et sa bande, la forme en flux sont compilés de même, jamais lancés sur une
  carte.
- La justesse se vérifie dans le cloud (émulation contre onnxruntime sur processeur, coupe comparée à la fusion
  ou à la paire entière) ; le PC ne donne que ce qui demande la carte : les temps et les dB contre TensorRT.

## Résultats du PC (RTX 4070 Ti, 19 novembre 2026)

Extrait de Re:Zero, GMFSS union seul (`--motion=gmfss`), SPAN derrière, travail en 1080p (plans remplis de la
paire à 1920×1152).

- **Paire native retenue**, sommes f16 : contre la paire TensorRT, f16 64,8 dB au pire (mixtes 66,0, f32 67,7) ;
  une passe en 19,75 ms contre 22,55 ms (mixtes 21,83, f32 22,76), 12 % de moins. En lecture (`--stats`, 30 s) :
  19,73 ms en médiane (p95 20,02) ; les splats 7,14 ms, le décor fixe et l'entrée 0,08 ms chacun.
- **Image** : export sans perte de 5 s, ×2, natif contre TensorRT : **63,15 dB** en moyenne (Y 62,98, U 61,23,
  V 68,66 ; la pire image 57,84). Au-dessus des 50 dB visés.
- **Cadence** (`kirari compare`, A TensorRT, B natif) : ×2 tenu des deux côtés (1,1 % et 1,4 % d'images perdues,
  p95 42,4 et 36,4 ms) ; ×3 tenu par aucun (35,6 % et 28,4 % perdues, p95 54,5 ms des deux) ; ×2 avec le tri des
  paires (`--scene-map --pair-triage`) : 0,7 % perdues des deux côtés, p95 42,4 et 30,3 ms.
- **Fusion native retenue**, sommes f16 (fusion de l'export d'origine, coupée par la structure ; zone de
  337 Mio) : contre la fusion TensorRT, f16 62,3 dB (mixtes 65,2, f32 65,3) ; une passe en 21,90 ms contre
  25,57 ms (mixtes 28,44, f32 32,04), 14 % de moins. Lors de la première séance, elle restait entière à TensorRT :
  la coupe ne connaissait que la fusion d'`ifnet_rife.py`.
- **Paire et fusion natives ensemble** (Re:Zero, 30 s, ×2, SPAN FP8, paire au demi-flot) : paire 19,69 ms en
  médiane (p95 19,91), fusion 21,76 ms (p95 22,20). Les deux passent de 48,0 à 41,6 ms par image interpolée
  (13 % de moins). C'est encore trop lent pour le temps réel : en 1080p avec SPAN, GMFSS union n'a pris que
  180 des 671 paires fortes, et RIFE a gardé les 491 autres faute de temps.
- Première ouverture : 149 s pour construire la paire, 180 s pour la fusion entière ; 54,6 s pour la fusion
  coupée (IFNet seul sous TensorRT).

## Plafonds théoriques en 1080p

Calculés sur processeur par `tools/gmfss/plafond_gmfss.py`, à partir des graphes de l'export d'origine, aux formes
d'une lecture en 1080p (plans remplis 1920×1152, paire au demi-flot). Hypothèses :
- débits de la RTX 4070 Ti : 160 Tflop/s en sommes f16, 80 en sommes f32, 321 en FP8 ;
- bande passante : 504 Go/s ;
- chaque couche lit son entrée une fois, écrit sa sortie une fois et lit ses poids, en f16, avec PReLU, additions
  et PixelShuffle fondus ;
- le plafond d'une couche est le plus lent du calcul et des octets, celui d'un morceau la somme de ses couches.

Les canaux ne sont pas complétés à 16 : le plafond du GridNet est de 10,9 ms ici, contre 11,7 ms avec les canaux
complétés (paragraphe précédent).

| Morceau | Taille | Gflop | Mo minimaux | Plafond f16 | Plafond FP8 | Mesuré au PC |
|---|---|---|---|---|---|---|
| paire : FeatureNet | 1152×1920 | 284 | 959 | 2,3 ms | 1,9 ms | |
| paire : MetricNet | 1152×1920 | 133 | 584 | 1,2 ms | 1,2 ms | |
| paire : GMFlow (demi-flot) | 288×480 | 363 | 3 283 | 2,3 à 6,6 ms | 1,1 à 6,5 ms | |
| **paire entière** | | **780** | | **5,7 à 10,1 ms** | | **19,70 ms** (TensorRT 22,44) |
| fusion : IFNet | 576×960 | 44 | 171 | 0,4 ms | 0,3 ms | |
| fusion : GridNet | 576×960, sortie 1152×1920 | 1 560 | 3 987 | 10,9 ms | 8,2 ms | |
| **fusion entière** | | **1 604** | | **11,2 ms** | **8,5 ms** | **21,90 ms** (TensorRT 25,57) |

Le plafond de GMFlow est une fourchette :
- 2,3 ms si l'attention est fondue (ses scores ne quittent pas la puce) ;
- 6,6 ms si chaque produit de matrices écrit sa sortie, scores compris.

La fusion est à environ deux fois son plafond, la paire à deux à trois fois et demie. L'IFNet, à 0,4 ms, ne
vaut pas d'être travaillé plus bas.

En FP8, le plafond du GridNet ne passe que de 10,9 à 8,2 ms tant que les activations restent en f16 : les octets
prennent alors le dessus.

La porte du temps réel en 1080p, ×2 avec SPAN, compte deux passes de SPAN par image source (GMFSS crée son image
avant SPAN) et les splats (7,14 ms) : avec le SPAN d'aujourd'hui, il ne reste qu'environ 10,6 ms à la paire et à la
fusion quand chaque paire passe par GMFSS, sous leurs plafonds additionnés (17 à 21 ms en f16). Le budget, les
paliers en 1080p et en 720p et les leviers au-delà du plafond sont dans [`gmfss-bilan.md`](gmfss-bilan.md) ; le
profil Nsight Systems (étape 3 de la séance) dit où sont les écarts.

## Séance finale (sur le PC, binaire `linux-release`)

Avec le natif (le défaut), le menu propose de nouveau « RIFE 4.25 + GMFSS union » (l'ancien Élevé) et « GMFSS
union » (l'ancien Très élevé), et `play.toml` les garde ; sous `--no-gmfss-native`, ils tombent sur RIFE 4.25 comme
avant (`interp_retired.h`). `--motion=gmfss` joue GMFSS union seul pour une lecture, natif ou non : la comparaison
du natif et de TensorRT sur le même extrait. `kirari compare` passe `--debug` à ses deux variantes.

```sh
# 0. Première ouverture : la fusion et la paire entières, l'IFNet et le flot se construisent. Pour chacune
#    (« fusion », « paire »), dans l'ordre (chiffres à lire) :
#    « GMFSS natif, fusion : 45 convolutions réglées au banc … en … s (… formes mesurées, N écartées ; … sans
#      choix ; … reprises d'une jumelle) » : le temps du réglage ; N écartées doit rester à 0, sinon une ligne
#      « GMFSS natif, fusion, réglage : convolution i (…), sommes … : N formes écartées » dit laquelle ;
#    « GMFSS natif, fusion, ligne 0 seul : f16 … ms (… dB) ; mixtes … ; f32 … ms (base) », une par morceau :
#      le temps seul de chaque cran, son effet sur l'unité contre le natif f32 ;
#    « GMFSS natif, fusion, tout le réseau contre TensorRT : f32 … dB, f16 … dB, mixtes … dB » : l'ancien essai ;
#    « GMFSS natif retenu pour la fusion (…, ligne 0 f16, … ; contre TensorRT … dB (prévus …), seuil 45 dB ;
#      contre le natif f32 : mélange … dB, TensorRT … dB ; une passe : TensorRT … ms, mélange … ms ; …) »,
#      les dB (50 visés) et les temps ; « GMFSS natif écarté, … » dit pourquoi TensorRT garde l'unité ;
#      « GMFSS natif refusé, TensorRT garde la fusion : … », une erreur avant l'essai (mémoire, lancement).
#    Les deux graphes : dans un dossier de RIFE, son voisin gmfss (~/kirari/models/gmfss) ou --gmfss-dir DOSSIER.
kirari play --motion=gmfss --rife=2 --start=960 extrait.mkv 2>&1 | grep -i "gmfss natif"

# 1. Vitesse, GMFSS union seul ×2 puis ×3 avec SPAN : TensorRT (A) contre natif (B), en A, B, B, A.
for x in 2 3; do
  kirari compare extrait.mkv --a="--no-gmfss-native --motion=gmfss --rife=$x --start=960" \
    --b="--motion=gmfss --rife=$x --start=960" --seconds=30 --report=gmfss-x$x.txt
done
#    Le tri des paires (il demande la carte de la scène), sur le natif :
kirari compare extrait.mkv --a="--motion=gmfss --rife=2 --start=960" \
  --b="--motion=gmfss --rife=2 --start=960 --scene-map --pair-triage" --seconds=30 \
  --report=gmfss-tri.txt
#    Le temps de chaque image et les étages (GMFSS paire native, GMFSS fusion native) :
kirari play --motion=gmfss --rife=2 --start=960 --stats=n2.csv extrait.mkv

# 2. Justesse sur l'extrait : deux exports sans perte, TensorRT puis natif (l'export porte sur tout l'extrait,
#    puis la lecture reprend : fermer le lecteur), et leur PSNR (50 dB visés).
kirari --debug --mute --no-gmfss-native --motion=gmfss --rife=2 --keys=1:export-lossless=t.mkv extrait.mkv
kirari --mute --motion=gmfss --rife=2 --keys=1:export-lossless=n.mkv extrait.mkv
ffmpeg -i n.mkv -i t.mkv -lavfi psnr -f null - 2>&1 | tail -1

# 3. Le temps GPU de chaque morceau : Nsight Systems, plages NVTX du domaine kirari autour de l'envoi de chaque
#    morceau (KIRARI_TRACE, src/core/trace.h) : gmfss.paire.gmflow (le moteur TensorRT du flot, avec ce qui
#    prépare l'entrée du MetricNet), gmfss.paire.featurenet, gmfss.paire.metricnet, ou gmfss.paire.tensorrt
#    quand TensorRT garde la paire entière ; gmfss.fusion.ifnet, gmfss.fusion.gridnet, ou gmfss.fusion.tensorrt ;
#    gmfss.splats (les huit splats d'un instant) ; gmfss.cle (une clé du mélange, hors --motion=gmfss).
#    --cuda-graph-trace=node : les moteurs TensorRT rejouent un graphe CUDA, dont chaque noyau compte alors
#    dans sa plage. L'essai au chargement passe par les mêmes plages : --delay le laisse passer (les moteurs
#    sont construits depuis l'étape 0).
nsys profile -o gmfss --trace=cuda,nvtx --cuda-graph-trace=node --delay=20 --duration=30 \
  kirari --debug play --motion=gmfss --rife=2 --start=960 extrait.mkv
nsys stats -r nvtx_kern_sum gmfss.nsys-rep

# 4. La forme en flux, contre la seule tuile : deux binaires linux-release, celui-ci et celui du commit « le
#    réglage au banc dans son propre fichier », juste avant la forme en flux ; mêmes commandes, même extrait.
#    a. Au réglage (première ouverture, étape 0), une ligne par convolution réglée où une forme en flux passe
#       (les jumelles reprennent la forme de leur aînée sans ligne) :
#       « GMFSS natif, fusion, réglage : convolution i (3×3 64→64 sur 960×576), sommes f16 : en flux au mieux
#         … µs (… bandes par bloc, tuile …), en tuile au mieux … µs : flux retenu » (ou « tuile gardée »).
#       Attendu, sur la bande du banc (544 lignes des 576) : 270 à 300 µs en flux contre 500 au plus en tuile
#       sans résiduel, 400 à 450 contre 640 avec. Une tuile gardée sur la ligne 0 dit que le flux n'y recouvre
#       pas ; aucune ligne en flux pour une convolution de la ligne 0, mais « N formes écartées », qu'il n'y
#       rend pas la sortie de la référence (à corriger avant toute mesure).
kirari play --motion=gmfss --rife=2 --start=960 extrait.mkv 2>&1 | grep -E "réglage : convolution|retenu pour"
#    b. Le GridNet entier : la plage gmfss.fusion.gridnet de nvtx_kern_sum (étape 3), un rapport par binaire ;
#       attendu 2,3 à 2,8 ms de moins que la forme fixe (21,90 ms la fusion), moins contre la seule tuile réglée.
#       Le temps de chaque noyau, stream_kernel contre conv_kernel, nombre d'appels et médiane :
nsys stats -r cuda_gpu_kern_sum --format csv gmfss.nsys-rep | grep -E "stream_kernel|conv_kernel"
#    c. Une convolution de la ligne 0 au détail : Nsight Compute, les noyaux lancés dans la plage du GridNet
#       (domaine kirari ; le réglage au banc se fait hors de la plage, à la construction). Si rien ne se
#       profile, ôter --nvtx et --nvtx-include : le banc et la paire passent alors aussi, que la grille distingue.
ncu --nvtx --nvtx-include "kirari@gmfss.fusion.gridnet/" -k regex:stream_kernel -c 12 \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis -o flux \
  kirari --debug play --motion=gmfss --rife=2 --start=960 extrait.mkv
ncu --import flux.ncu-rep --page details --csv > flux.csv
#       À relever pour chaque lancement de la ligne 0 (grille 30 × 4, 8 ou 16 × 1 en tuile de 64 ; 30 × 2, 4
#       ou 8 × 2 en tuile de 32) : la durée ; les octets lus en mémoire (70,8 Mo, 141,6 avec un résiduel) et
#       écrits (70,8 Mo) ; le débit mémoire (88 % de la crête visés) et l'activité des tensor cores (85 %
#       visés) ; l'occupation atteinte (8 warps sur 48 : deux blocs). La bande tenue avec des tensor cores sous
#       60 % rouvrirait le bloc résiduel fondu, qui échange des octets contre du calcul ; sinon il reste non
#       écrit. Les dB contre TensorRT (ligne « retenu pour la fusion ») ne bougent pas au-delà du bruit : la
#       forme en flux rend la sortie de la tuile en tranche de 64.
```

Le paquet Linux change d'empreinte TensorRT (`tools/release/trt-fingerprint.sh`) : `src/gpu` et `src/prepare` ont
changé.
