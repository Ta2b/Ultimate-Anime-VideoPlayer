# RIFE natif (le défaut sur NVIDIA, `--no-rife-native` pour l'éteindre)

Novembre 2026. L'encodeur et les convolutions des blocs de RIFE fusionné ([`rife-fusionne.md`](rife-fusionne.md))
par nos propres noyaux CUDA, à la place des six moteurs TensorRT, ou de ceux d'entre eux qu'ils battent : l'essai au
chargement choisit morceau par morceau. La colle entre blocs, les warps et la fin
(`rife_fused_kernels.cu`, `warp_lib`) ne changent pas : le natif lit et écrit les mêmes tampons, dans les mêmes
formats (X et Y en HWC8), sans reformatage. Seule exception, la colle du dernier bloc en pleine résolution : sa
première couche native peut la refaire à la volée (plus bas). L'encodage d'une image reste calculé une fois pour
deux paires, le tri des paires, l'Auto et `held_pair` ne voient aucune différence.

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

Le mélange reste juste pour GMFSS : GMFlow sous TensorRT, le reste natif
(`/mnt/project-files/chantier-natif/gmfss-natif.md`).

## Ce que font les noyaux

`src/gpu/cuda/backend/rife_native_kernels.cuh` (une famille de convolutions), `rife_native_pick.cu` (les formes
instanciées et les formes à essayer), `rife_native_weights.cu` (poids repliés, rangés en fragments B),
`rife_native_layer.h` et `rife_native_bench.cu` (une couche, le banc du réglage), `rife_native_glue.cu` (la
première couche du dernier bloc, entrée calculée à la volée par la colle de `rife_glue.cuh`),
`rife_native_finish.cu` (la dernière couche du dernier bloc, fin comprise, par les fonctions de `rife_finish.cuh`),
`rife_native_net.cu`
(le réseau), `rife_native_trial.cpp` (l'essai morceau par morceau), `rife_native_mix.cpp` (le choix du mélange),
`rife_fused_native.cpp` (la chaîne pour l'essai), `prepare/rife_native.h` (les poids lus dans les morceaux ONNX).

- **mma.sync FP16**, sommes en f16, mixtes ou f32 selon l'essai ; m16n8k16 dès sm_80, deux m16n8k8 sur sm_75
  (Turing), mêmes fragments. Mixtes : chaque prise (Kc produits au plus) s'accumule en f16, au débit plein des
  tensor cores des GeForce, puis s'ajoute en f32 ; l'arrondi f16 ne porte plus sur les 9 · cin produits de la
  couche. Simulé sur une couche 3 × 3 (activations après LeakyRelu, poids N(0, 1/K)) : f16 de 56 à 67 dB contre
  la somme exacte selon cin, mixtes de 69 à 74 dB, f32 au-delà de 130 dB ;
- **tuile en mémoire partagée** de 8 × 32 ou 4 × 32 pixels (deux lignes par warp ou une ; une seule en pas 2),
  halo compris, copiée par `cp.async` au pas Kc + 8 (les lignes d'un `ldmatrix` sur des banques distinctes en pas
  1) ; les canaux d'entrée par tranches de Kc = 16, 32 ou 64 pour rester sous 48 Ko, ceux de sortie par tuiles de
  16, 32 ou 64 (blockIdx.z ; 64 en sommes f16 seulement). La forme de chaque couche est réglée au chargement
  (plus bas) ;
- **ResConv fondu** : le bêta est replié dans les poids et le biais au chargement, l'addition résiduelle et le
  LeakyRelu se font à l'écriture de la sortie ; deux tampons alternent, l'entrée reste lisible ;
- **transposée 4 × 4 de pas 2** en quatre convolutions 2 × 2, une par phase de sortie, écrites directement dans Y
  au pas 56 ; celle de l'encodeur (4 canaux) en une convolution 3 × 3 à 16 sorties (phase · 4 + canal) qui écrit
  les quatre plans d'encodage ;
- **première couche de l'encodeur** lue dans les trois plans de l'image d'entrée, sans copie ;
- **entrée du dernier bloc à la volée** (`rife_native_glue.cu`). En pleine résolution (d = 1), X du dernier bloc
  pèse 64 octets par pixel, soit 134 Mo en 1920 × 1088 : la colle l'écrivait, sa première couche le relisait
  (0,363 ms à elle seule sur la 4070 Ti, à 75 % de la bande passante). Cette couche, une 3 × 3 de pas 2 de 32
  canaux (28 utiles) vers 16 (ou 32 selon le modèle), peut calculer X sur sa tuile d'entrée (65 × 9 pixels pour
  32 × 4 sorties) :
  - elle relève les Y des blocs précédents comme la colle, puis appelle les mêmes fonctions qu'elle
    (`rife_glue.cuh`), donc les mêmes arrondis ;
  - elle somme ensuite comme la forme 16/16 d'une ligne par warp : même tuile, même ordre ;
  - X tient en deux tranches de 16 canaux, calculées l'une après l'autre ; la seconde refait le flot, peu
    coûteux. Cela donne 28 Ko de tuile et au plus 15 Ko de relèvement (des échelles d'au moins 2 avant le
    dernier bloc, sinon la colle reste à part), pour deux blocs par multiprocesseur ; un seul sur Turing (64 Ko)
    sauf pour 4.25 lite ;
  - seul le flot accumulé (8 octets par pixel) part dans X, pour la fin ; chaque pixel l'est une seule fois,
    par la tuile qui le possède (vérifié hors carte de 2 × 2 à 300 × 257).

  Le trafic attendu tombe de 268 Mo (écrire X, puis le relire) à environ 75 Mo (les images et les encodages warpés,
  le flot, la sortie de la couche). La colle du dernier bloc ne passe plus quand un réseau natif fait ce bloc à la
  volée. L'essai mesure aussi chaque réseau avec la colle à part puis sa première couche réglée, et garde la plus
  rapide des deux (le journal donne les deux temps) : sur une carte où la colle à la volée perd, le dernier bloc
  reste natif. Le relèvement réserve, par source d'échelle s, au plus `glue_capacity(s)` pixels de Y. Cette borne
  est vérifiée sur 11 millions de tuiles (tailles de 32 à 2176 lignes, échelles 1 à 32) et atteinte sans être
  dépassée. La colle de TensorRT appelle les mêmes fonctions : sa SASS ne change que par l'arithmétique d'adresses,
  avec les mêmes opérations flottantes. Il existe 12 noyaux (sommes × 3 ou 4 blocs d'avant × tuile 16 ou 32), à 160
  registres au plus (sm_120), sans débordement ;
- **fin dans la dernière couche** (`rife_native_finish.cu`). La transposée du dernier bloc écrivait Y (52 canaux,
  56 rangés, à la moitié de la pleine résolution : 58 Mo en 1920 × 1088), dont fused_finish ne relisait que les
  20 premiers (le flot du bloc et le masque, en quatre parités). Quand la forme réglée de cette couche n'a pas de
  K découpé et qu'un noyau existe pour sa tranche et ses lignes par warp (sommes f16 ou f32), elle ne calcule plus
  que sa première tuile de 32 canaux et fait la fin dans son épilogue, depuis la mémoire partagée où elle a rangé
  sa sortie : mêmes sommes, même arrondi FP16 qu'à l'écriture de Y, puis les fonctions de fused_finish
  (`rife_finish.cuh`). Sortie identique au bit à « transposée puis fused_finish » (émulateur sur processeur, sept
  cas : sommes, tranches, lignes, image et mouvement, fenêtre et Catmull-Rom, grille qui ne tombe pas juste) ; Y
  n'est pas écrit. L'essai par morceau écrit toujours Y. L'essai chronomètre la passe retenue avec la fin fondue
  et avec la fin à part, et garde la plus rapide (ligne « fin du dernier bloc » du journal) ; la passe comparée
  part de sorties à NaN, pour qu'un pixel oublié ne garde pas la valeur de TensorRT. Neuf noyaux, sans
  débordement de sm_75 à sm_120 ;
- **pas 2 sans conflit de banques** : la tuile d'entrée est rangée colonnes paires puis impaires, les 16 lignes
  d'un `ldmatrix` restent des pixels voisins (au pas 2 brut, deux lignes tombaient sur les mêmes banques) ;
- **fragments B préchargés** : les poids du pas k suivant se lisent pendant les mma du pas courant ;
- **K découpé sur les petites grilles** : une couche dont la grille fait moins de deux vagues (les blocs
  grossiers, 30 × 17 pixels au 1/32 de 1080p, ne donnaient que 36 blocs de calcul pour 60 SM) peut répartir ses
  prises × tranches entre plusieurs blocs, sommes partielles en f32, puis `reduce_kernel` ajoute le biais, le
  résiduel et l'activation ;
- **précision par couche** : le corps d'un morceau et sa tête (la dernière couche : flot et masque d'un bloc,
  encodage de l'encodeur) peuvent sommer différemment, la tête plus finement.

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

Là où ils faisaient mieux, repris ici (mêmes idées, notre code) : le pas 2 rangé par parité de colonne, le K
découpé sur les petites grilles (leur `ksplit`), et la forme mesurée sur la carte au chargement (eux : 8, 4 ou 2
warps par convolution ; nous : tranche, tuile, une ou deux lignes par warp, K découpé ou non, jusqu'à 36 formes
par couche). Restent chez eux :

- **les sommes f16 partout**, sans porte : ils ne comparent pas leur sortie à une référence. La nôtre (45 dB
  contre TensorRT sur la chaîne) refusait le f16 de bout en bout (41,8 dB) et retombait sur le f32, à moitié du
  débit des tensor cores d'une GeForce : c'est surtout ce qui rendait notre natif plus lent que le leur. D'où la
  précision par morceau et par couche (plus bas) ;
- **l'occupation** : leur tuile prend moins de registres (16 accumulateurs contre 64 chez nous en CT = 64),
  donc plus de warps par SM pour cacher la latence. Les tuiles d'une ligne par warp (4 × 32, deux fois moins
  d'accumulateurs) s'en approchent ; le réglage choisit couche par couche ;
- **la colle** (hors de ces noyaux) : un seul rassemblement par bloc à sa propre échelle (`k_gather`), lot E
  du chantier AnimeJaNai.

## Essai au chargement

Par défaut sur NVIDIA depuis la séance PC du 19 novembre 2026 (4.25 ×2 en 2,51 ms par passe contre 3,40 pour
TensorRT, 4.22 lite en 2,95 contre 3,56 ; [`rife-bilan.md`](rife-bilan.md)) ; `--debug --no-rife-native` rend toute
la chaîne à TensorRT, `--rife-native` reste accepté. Les six moteurs TensorRT se construisent et se chargent comme
avant (en E/S HWC8).

**Verdict gardé** (`rife_native_verdict.h`) : l'essai complet coûte environ 4,5 s au chargement. Son verdict
(l'option de chaque morceau, la fin fondue ou à part, la chaîne contre TensorRT en dB) se garde dans le dossier des
moteurs, un fichier `rife-natif-<empreinte>.verdict` par identité : carte (nom, architecture, UUID), version CUDA
du pilote, clé d'artefact de la chaîne (taille, variante, warp, plan, noms des moteurs TensorRT), empreinte des
sources du RIFE natif (`rife_*` et `warp_lib*` du backend, prise à la configuration) et seuil. Au chargement
suivant, seuls les réseaux du mélange sont réglés, puis la chaîne passe une fois sur la paire de test et doit tenir,
contre TensorRT, au plus 1 dB sous la valeur du verdict ; sinon la chaîne revient à TensorRT et l'essai complet
se refait. Un verdict « TensorRT garde tout » se garde aussi. Effacer le fichier refait l'essai.

L'essai complet :

1. **quatre réseaux natifs**, du plus rapide au plus juste : sommes f16 ; f16 et tête f32 ; mixtes et tête f32 ;
   f32. Chaque couche de chacun prend sa forme au banc (`rife_native_bench.cu`) : sur une entrée de valeurs
   variées, chaque forme qu'un noyau fait est lancée une fois sur une sortie remplie de NaN (une valeur non
   écrite ne passe pas), sa sortie comparée à celle de la forme la plus simple (tranche et tuile de 16, sans
   découpe de K) et écartée si elle s'en écarte de plus que l'ordre des sommes ne l'explique : 1/16 du pic en
   f16, 1/32 en mixte, 1/128 en f32 (un défaut de rangement ne passe pas : il se voit au journal, « formes
   écartées »), puis chronométrée en graphe (trois lancements, le meilleur de trois rejeux). Les poids partent
   sur la voie du banc, qui les attend. Une couche de mêmes sommes reprend la forme déjà réglée pour un réseau
   d'avant. Les ResConv (3 × 3 de pas 1, tranche de 32 ou 64, hors f32) ont en plus des formes « poids à
   l'avance » : les fragments de poids de toute la prise suivante se chargent dès le début de la prise courante,
   dans un second jeu de registres. Les poids d'une couche de 128 ou 192 canaux ne tiennent pas dans le L1 et
   viennent du L2, dont la latence dépassait les mma d'un seul pas ; le réglage choisit, couche par couche, entre
   cette distance et l'occupation que les registres en plus coûtent ;
2. une paire de test (ondes et carrés, B décalée de (5, 3) pixels, instant 0,5) passe par la chaîne avec
   TensorRT : la référence ;
3. **chaque morceau seul** (l'encodeur sur l'image A, chaque bloc sur le X_k que TensorRT a laissé dans la zone),
   par TensorRT et par chaque réseau (`rife_native_trial.cpp`) : son temps (un graphe de deux passes, cinq tours
   intercalés, la médiane ; ces passes lancées une à une si la capture est refusée, ce que le journal dit), puis
   sa sortie contre celle de TensorRT, en dB. Le dernier bloc, quand un réseau natif sait prendre son entrée à la
   volée, se mesure colle comprise pour chaque option (TensorRT : la colle puis le moteur ; chaque tel réseau
   avec sa colle à la volée et avec la colle à part, la plus rapide gardée, ce que le journal dit) ; sa justesse,
   puis la chaîne, se mesurent avec la colle gardée. La sortie de TensorRT de chaque morceau reprend sa place
   après ses réseaux : la colle en relit les Y. Sur la chaîne, X du dernier bloc part de NaN quand son réseau le
   refait à la volée. En variante mouvement, un pixel oublié sort NaN dans le flot ; en variante image, la fin
   le ramènerait au bord de la fenêtre, donc le flot est relu après la passe, et un pixel non fini rend toute
   la sortie NaN : l'option est écartée. La justesse d'un morceau se lit rôle par rôle (flot, masque, traits
   de Y, chacun à son pic ; `rife_mix::piece_db`) : un défaut du seul masque ne se dilue pas dans les 52 canaux.
   Le profil couche par couche du réseau f16 va au journal ; un profil qui échoue n'y laisse qu'une ligne,
   l'essai continue ;
4. **la référence du mélange** (`rife_mix_choice.cpp`, testé sans carte). À travers des blocs FP16, tout
   changement d'un morceau, même d'un ulp, redistribue les arrondis de TensorRT : jugé contre sa chaîne, un
   encodeur plus juste que le sien se lisait à 43,6 dB sur le 4.22 lite, le bruit propre de TensorRT face au
   calcul exact (émulé contre onnxruntime en f32 : notre chaîne f32 en est à 60,3 dB sur le masque, 64,7 sur
   4.25). La référence est donc la chaîne native f32 quand une garde la reconnaît : chaque morceau f32 à 48 dB
   au moins du sien de TensorRT sur la même entrée, rôle par rôle (une comparaison sans le bruit des blocs
   suivants ; 48 et non 60 : les blocs FP16 de TensorRT sortent eux-mêmes à 50,9 à 62,4 dB de nos morceaux
   f32 sur le PC, et 60 refusait un natif juste), et la chaîne de TensorRT à 42 dB au moins de la chaîne f32. La barre d'un groupe est alors 45 dB,
   ou ce que TensorRT y tient s'il n'y arrive pas : le mélange reste au moins aussi proche que lui du calcul
   exact. Sinon, l'ancienne règle : 45 dB contre TensorRT. Un natif faux ne devient jamais sa propre
   référence, un refus à tort ne fait que revenir à l'ancienne règle (le journal dit laquelle et pourquoi). Les
   effets se mesurent alors le reste en f32, TensorRT compris comme une option ;
5. **le mélange** (`rife_native_mix.cpp`, `rife_mix_choice.cpp`). Un morceau peut prendre TensorRT ou un réseau
   plus rapide que lui. L'effet de chacune de ces options est mesuré sur la chaîne entière, ce morceau seul hors
   de la référence : l'erreur
   quadratique de la sortie contre la référence (images au pic 1 ; flot au pic de son plus grand écart, masque à
   part), celle que les morceaux suivants font de l'erreur du morceau. Les erreurs de morceaux différents
   s'ajoutent à peu près : l'erreur prévue d'un groupe va de la somme des effets (erreurs indépendantes) au
   carré de la somme de leurs racines (erreurs de même signe aux mêmes pixels), selon une corrélation qui part
   de 0. Parmi toutes les combinaisons (au plus 5⁶), le mélange retenu est le plus rapide (la somme des temps
   seuls) dont l'erreur prévue tient la barre (45 dB, ou moins sur la référence native, étape 4) ; un mélange
   d'un seul morceau hors de la référence n'a rien à prévoir, son effet est sa mesure. Il est vérifié sur la
   chaîne. Sous la barre, ce mélange est écarté, la corrélation de chaque groupe se recale sur l'erreur mesurée
   (sans jamais descendre) et les mélanges de plusieurs morceaux doivent tenir 0,5 dB de plus ; au quatrième
   essai, seul un mélange d'un seul morceau hors de la référence peut être pris, sinon
   tout revient à TensorRT. Essayé hors carte sur des erreurs synthétiques (400 tirages par cas, six
   morceaux, corrélation de 0 à 1) : jamais tout à TensorRT quand un mélange tient, 0,06 à 0,08 ms de l'optimum
   en moyenne. La première version de #942 (un écart unique ajouté à la barre de toutes les combinaisons) en
   restait à 0,10 à 0,23 ms et, sur des erreurs très corrélées, rendait 1 % des tirages à TensorRT. L'ancien
   mélange montait d'un cran le morceau le moins juste jugé seul, sans jamais redescendre : jugés contre
   TensorRT, les crans plus justes ne le sont pas toujours (le bloc 4 de 4.25 : 73,2 dB en f16, 69,3 en
   mixtes), et la justesse d'un morceau seul ne dit pas son poids sur la chaîne (l'encodeur décidait des
   45 dB). Le bloc 4, plus rapide en f16 que TensorRT, finissait ainsi à TensorRT ;
6. le mélange est chronométré contre TensorRT seul comme le SPAN FP8 (`span_timing.h`) : une passe de lecture ×2
   (B encodée seule, graphes rejoués), onze tours intercalés d'une passe de chauffe et deux passes mesurées, la
   médiane. Il n'est gardé que s'il bat TensorRT d'au moins 2 %.

Le moteur aiguille chaque morceau vers son réseau ou vers son moteur TensorRT (`set_route`, l'aiguillage est
dans la clé des graphes) ; les réseaux que le mélange n'emploie pas sont libérés. Le journal donne tout : le
temps du réglage, une ligne par morceau (« RIFE natif, bloc 2 seul : TensorRT 0.412 ms ; f16 0.301 ms (52.3 dB)
; … »), une ligne par couche du réseau f16 (temps, forme retenue), l'effet de chaque option sur la chaîne
(« RIFE natif, effet de chaque option sur la chaîne (le reste à TensorRT) ; encodeur f16 47.1 dB ; … »), puis le
verdict : « RIFE natif retenu (1920×1088, encodeur f16, bloc 0 f16, bloc 1 f16 et tête f32, … ; chaîne contre
TensorRT 46.1 dB (prévus 46.4), seuil 45 dB ;
contre le natif f32 : chaîne 47.0 dB, TensorRT 49.3 dB ; passe de lecture ×2 : TensorRT 4.98 ms, mélange 4.40 ms ;
essai 3.2 s) » (chiffres d'exemple). L'écart au natif f32, le calcul le plus juste, dit si un mélange refusé l'est
pour une vraie perte ou pour un arrondi différent de celui de TensorRT.

Mesure sur 4070 Ti avant ce travail (RIFE 4.25 seul, 1080p, chaîne entière en une seule précision) : f16
41,8 dB, f32 49,3 dB, retenu en f32 et 13 % plus lent que TensorRT (5,62 ms contre 4,98 ms par image créée). Le
f32 natif somme presque exactement : ses 49,3 dB contre TensorRT disent surtout l'écart entre deux calculs FP16
qui arrondissent à des endroits différents. Aucune variante native ne passera nettement au-dessus contre
TensorRT ; les 50 dB de la cible se jugent contre le natif f32 (la ligne du verdict les donne).

Remesure avec les sommes mixtes et la mesure du temps (#897, même PC, même extrait) : f16 41,8 dB, mixtes
43,8 dB, f32 49,3 dB contre TensorRT ; seul le f32 tenait les 45 dB, et il restait plus lent (passe de lecture
×2 : TensorRT 3,64 ms, f32 4,03 ms). Le natif était donc écarté, TensorRT gardait la chaîne à la même cadence
(138,6 i/s).
D'où ce travail : têtes en f32 et précision par morceau pour la justesse, réglage des formes pour la vitesse.

Mesure de #919 (même PC, main 7b14e78) : **retenu en mélange, 9 % plus rapide**. Encodeur f32 (0,215 ms contre
0,524 pour TensorRT), blocs 0 à 2 TensorRT, bloc 3 f16, bloc 4 TensorRT ; chaîne 49,1 dB contre TensorRT, 50,0 dB
contre le natif f32 (TensorRT : 49,3) ; passe de lecture ×2 : TensorRT 3,40 ms, mélange 3,10 ms ; essai 2,7 s
(réglage 1,1 s). ABAB sur Re:Zero, ×2 sans SPAN, `--one-lane`, 3 tours : 5,85 ms par passe en médiane contre
6,42 ms, stable. Le mélange est donc plus proche du calcul exact que TensorRT. Trois défauts que le journal
montrait, repris ensuite : le bloc 4, plus rapide en f16 (0,761 ms contre 0,870), laissé à TensorRT par
l'ancien mélange (corrigé, point 4) ; les couches de 60 × 34 pixels des blocs 0 à 2 à 17 µs contre environ 13
pour TensorRT (formes « poids à l'avance », point 1) ; la première convolution du bloc 4 relit X_4 en pleine
taille (1920 × 1088 × 32 canaux, 134 Mo, 0,363 ms, environ 75 % de la bande passante), que la colle vient
d'écrire : la calculer à la volée depuis les sources de la colle épargnerait les deux transferts, de l'ordre de
0,4 ms par passe (fait ensuite : « entrée du dernier bloc à la volée », plus haut).

Une valeur non finie d'un côté ou de l'autre (référence TensorRT comprise) écarte le natif. Un essai qui fait
perdre le contexte CUDA (accès fautif d'un noyau) fait échouer le chargement au lieu de rendre une chaîne
TensorRT morte ; tout autre refus laisse TensorRT, chargé et intact. La lecture des morceaux ONNX refuse ce que la
diffusion d'ONNX lirait autrement que les noyaux : un bêta ou une pente de forme C ou 1×1×1×C (le long de la
largeur), une transposée qui porte `output_shape`.

## État

L'entrée du dernier bloc à la volée est vérifiée hors carte, avant toute mesure sur le PC :

- nvcc sm_75, 86, 89 et 120 sans débordement ; g++ strict, clang-tidy, format, complexité et règles de la CI ;
- un émulateur processeur du code CUDA tel quel (une fibre par fil, mma, ldmatrix et cp.async émulés ; sources
  et mode d'emploi dans `/mnt/project-files/chantier-natif/rife-natif-emu/`) : sortie et flot écrit dans X
  identiques au bit à l'ancien chemin (colle puis convolution 16/16), sommes f16, mixtes et f32, 8 cas
  synthétiques et les vrais poids et tenseurs de 4.22 lite et 4.25 ; trois mutations du noyau sont détectées ;
- contre onnxruntime en f32 (paire de l'essai, 1024 × 576) : première convolution du dernier bloc à 76,9 à
  81,4 dB, X à 70 dB ou plus par groupe de canaux, la grille du graphe donnée en f32 exact. Avec la grille FP16
  de l'entrée v1, les warps du graphe tombent à 51-55 dB du calcul exact : l'écart de la grille FP32 de la colle
  (A03-20), pas une erreur.

Les plafonds théoriques d'une passe ×2 en 1080p sur la 4070 Ti
(`/mnt/project-files/chantier-natif/rife-plafonds.md`) : 2,28 ms couche par couche, 1,58 ms morceaux fondus, contre
3,00 ms mesurées pour le mélange de 4.25 (#942). La suite vise ces plafonds : la colle et la fin fondues (environ
1,2 ms mesurées contre 0,82 de plafond), puis la passe entière en un noyau. Sur 4.22 lite, natif f32 et TensorRT
ne sont qu'à 43,5 dB l'un de l'autre : l'encodeur seul en f32 sort à 79,6 dB de TensorRT mais la chaîne en fait
43,8 dB ; la référence f32 n'est donc pas fautive : c'est TensorRT qui est à 43,6 dB du calcul exact sur cette
chaîne (le masque de la variante mouvement est le plus sensible), d'où la référence native de l'étape 4.

Écrit et compilé (nvcc 13.5, sm_75, 86, 89, 120, `--warn-on-spills -Werror`, aucun débordement de registres),
relu, jamais lancé sur une carte : la justesse se juge à l'essai au chargement, la vitesse à la séance finale.
Les formes « poids à l'avance » ajoutent 20 noyaux (116 convolutions en tout) : elles lisent les mêmes fragments
dans le même ordre, plus tôt, et passent par le même contrôle au réglage. La plus grande (tranche et tuile de 64,
deux lignes par warp, f16) prend 252 registres sur sm_75 et 246 à 250 ailleurs, sans débordement : la marge est
mince, un changement qui la fait déborder casse la construction (`--warn-on-spills`). Le réglage par couche et le
mélange par morceau (novembre 2026) : les 96 noyaux instanciés (contre 66) passent l'émulateur ci-dessous dans
216 cas (sommes f16 et f32 ; tranches, tuiles, lignes par warp, K découpé ou non ; grilles à bords incomplets et
grilles de quelques blocs), à 1e-7 près ; le module grossit d'environ 3 Mo par
architecture (9 Mo au lieu de 6 pour sm_89). Quand le réglage aura tourné sur le PC, les formes jamais retenues
pourront quitter le module. L'essai dure plus longtemps (le journal donne la durée du réglage et de l'essai) :
si c'est trop, garder les formes réglées d'un chargement à l'autre par carte et par taille.

RIFE 4.22 lite passe par le même chemin : ses cinq morceaux (encodeur, blocs de 192, 128, 64 et 32 canaux aux
réductions 8, 4, 2, 1) se lisent et passent la vérification des couches (contrôlé sur le modèle du catalogue,
même empreinte que `tools/release/fetch-models.sh`) ; `--rife-native --rife-model=rife_v4.22_lite.onnx` l'essaie.

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

PSNR natif contre TensorRT et contre le natif f32 : la ligne du verdict (étape 0) ; 50 dB ou plus attendu contre
le natif f32. Garder aussi les lignes par morceau et par couche : elles disent où le natif perd. Décision : natif
par défaut s'il tient 50 dB et passe sous 23 % d'occupation ; sinon le profil dit quel noyau perd (tuiles,
tranches, fusion de plusieurs ResConv d'un bloc grossier en un lancement) avant de redécider. Ensuite seulement :
ne plus construire les moteurs TensorRT quand le natif est retenu (47 s gagnées au premier lancement), et
DistilDRBA.
