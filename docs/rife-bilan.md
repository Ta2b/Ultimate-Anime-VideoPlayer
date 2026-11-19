# Bilan : RIFE natif et DistilDRBA (19 novembre 2026)

Mis à jour le 19 novembre 2026 à 21 h 00 : #961 fusionnée sur main (520f4632), le RIFE natif est le défaut
de la rc.1. Sur la branche de #987, pour après la rc.1 : garde des morceaux de la porte à 48 dB (partie 4, b),
colle de l'avant-dernier bloc à la volée (d), outil nsys pour les blocs grossiers (c).

Pour reprendre ce travail sans son auteur. Tout est mesuré en 1080p (1920 × 1088 rempli) sur la RTX 4070 Ti du PC
de Tatoub, sauf mention contraire. Une « passe » est une passe de lecture ×2 de RIFE fusionné : l'image B seule
encodée (celle de A est reprise de la passe d'avant), tous les blocs, les colles et la fin. Elle crée une image.
Documents de détail : `docs/rife-natif.md` (le code et l'essai), `docs/drba-fusionne.md`, `docs/fp8-interpolation.md`,
[`rife-plafonds.md`](rife-plafonds.md) (le calcul des plafonds), journaux du PC
`/mnt/project-files/rife-natif.log` (#919), `rife-natif-425.log` et `rife-natif-422.log` (#942).

## En bref

- RIFE 4.25 : **2,51 ms par passe en natif contre 3,40 pour TensorRT (−26 %)**, chaîne à 48,5 dB de TensorRT et
  49,6 dB du natif f32 (séance du 19 novembre, e05a3a5). Plafond 2,28 ms couche par couche, 1,58 ms morceaux fondus.
- RIFE 4.22 lite : **2,95 ms contre 3,56 (−17 %)**, 47,6 dB. La porte contre le calcul exact n'a pas joué (partie
  0) : l'encodeur reste à TensorRT ; avec elle, environ 2,6 à 2,7 ms attendues.
- En lecture, avec ou sans SPAN, ×2 et ×3 : rien de perdu des deux côtés (0,1 à 0,3 % d'images).
- Natif par défaut avec le verdict de l'essai gardé sur le disque (PR après #961) : l'essai de 4,5 s ne passe
  qu'une fois par carte, pilote, modèle et taille.
- DistilDRBA v2 lite : chaîne fusionnée 0,917 ms par image créée (TensorRT pour les convolutions), allumée par
  défaut ; aucun noyau natif encore. Plafond 0,37 ms couche par couche, environ 0,27 ms fondu.

## 0. Séance PC du 19 novembre 2026 (e05a3a5, RTX 4070 Ti, Re:Zero S4E18 à 960 s)

Sorties brutes : `/mnt/project-files/rife-natif-seance.tar.gz`.

**Verdict de l'essai** (passe ×2 chronométrée par l'essai, carte seule) :

| Modèle | Mélange retenu | Natif | TensorRT | Gain | Contre TensorRT | Contre natif f32 |
|---|---|---|---|---|---|---|
| 4.25 | encodeur f32, blocs 0 à 2 TensorRT, blocs 3 et 4 f16 | 2,51 ms | 3,40 ms | −26 % | 48,5 dB | 49,6 dB |
| 4.22 lite | encodeur et bloc 0 TensorRT, blocs 1 à 3 f16 | 2,95 ms | 3,56 ms | −17 % | 47,6 dB | 43,5 dB |

- **Entrée du dernier bloc à la volée** : gardée sur les deux modèles et tous les crans ; bloc 4 de 4.25 en f16 à
  0,944 ms colle comprise contre 1,242 à part (−0,30 ms), bloc 3 du lite 0,894 contre 1,230 (−0,34 ms).
- **Fin fondue** : gardée, 2,511 contre 2,716 ms à part sur 4.25 (−0,21 ms), 2,950 contre 3,131 sur le lite
  (−0,18 ms) ; `finish_conv_kernel` remplace la transposée du dernier bloc et `fused_finish` dans le profil.
- **Porte contre le calcul exact** : elle n'a pas joué. « Référence du mélange : TensorRT (bloc 0 du natif f32 à
  55,7 dB de TensorRT, moins de 60) » sur 4.25, 59,3 dB sur le lite. Les morceaux f32 sont à 50,9 à 62,4 dB de
  TensorRT, et nos morceaux f16 sont plus près de TensorRT que nos f32 (bloc 1 de 4.25 : 56,6 contre 50,9 dB) :
  c'est TensorRT qui s'écarte, ses blocs tournent en FP16. Le plancher de 60 dB par morceau est trop haut pour le
  bruit propre de TensorRT ; à reprendre (partie 4). Sur le lite, la chaîne TensorRT est à 43,6 dB du natif f32
  (au-dessus du plancher de chaîne de 42) : avec une garde des morceaux qui passe, l'encodeur natif entrerait.
- **Lecture, ABAB** (`kirari compare`, 60 s, une voie) : A et B tiennent tous deux, 18,2 ms par image en ×2,
  12,1 ms en ×3 avec SPAN, 0,1 à 0,3 % d'images perdues des deux côtés : le lecteur suit la cadence de l'écran,
  le gain se voit dans le temps du moteur, pas dans les images.
- **Temps du moteur en lecture** (profil GPU, « moteur RIFE fusionné ») : 4.25 médiane 6,59 ms contre 7,23,
  minimum 2,64 contre 3,62 ; lite 6,74 contre 7,13, minimum 2,93 contre 3,60. Les minimums sont la passe de
  l'essai (2,51 et 2,95). La médiane mesure, entre deux horodatages GPU sur la voie de RIFE, la même passe pendant
  que la carte fait autre chose sur ses autres voies (ingestion, entrée de RIFE, warp et mélange, écriture du
  ring, composition de l'écran) et à des horloges basses : la carte est occupée moins d'un cinquième du temps,
  elle redescend en fréquence entre deux images. Les noyaux le montrent dans nsys : `glue_conv_kernel` en
  médiane 1,02 ms en lecture contre 0,52 ms à l'essai, `fused_glue<1,3>` 1,01 ms des deux côtés. L'écart de 4 ms
  n'est donc pas du travail en plus : c'est la passe étirée par le partage et par l'horloge. Le gain relatif s'y
  tasse (−9 % au lieu de −26 %) parce que les noyaux qui ne changent pas (colles, warp) s'étirent autant que
  les autres.
- **Occupation, nsys** (30 s) : 18,7 % de noyaux en natif contre 16,2 % en TensorRT, mais la capture native
  contient l'essai au chargement (154 noyaux distincts, dont les quatre crans, la colle à part, les passes f32) :
  508 passes de lecture dans la capture native contre 630 dans celle de TensorRT. La comparaison n'est pas juste ;
  à refaire avec `nsys profile --delay=15` (après l'essai, ou avec le verdict gardé qui le saute).

## 1. Ce qui est fait

### RIFE 4.25

PR fusionnées, toutes derrière `--rife-native` (éteint par défaut) :

| PR | Ce qu'elle apporte |
|---|---|
| #869 | L'encodeur et les convolutions des blocs par nos noyaux CUDA (mma.sync FP16), lus dans les morceaux ONNX |
| #871 | Pas 2 sans conflit de banques, fragments de poids préchargés, K découpé sur les petites grilles |
| #891 | Corrections de l'audit de #869 et #871 |
| #897 | Sommes mixtes (f16 par prise, f32 entre prises), mesure du temps contre TensorRT à l'essai |
| #919 | Réglage de la forme de chaque couche au chargement, précision par couche (tête plus fine), mélange morceau par morceau avec TensorRT |
| #942 | Mélange choisi d'après l'effet de chaque option sur la chaîne, recalé sur la corrélation des erreurs ; poids une prise à l'avance |

Mesures sur le PC (journaux cités plus haut) :

| Étape | Passe (TensorRT → natif) | Justesse |
|---|---|---|
| Avant #897, chaîne en une précision | 4,98 → 5,62 ms (f32, +13 %) | f16 41,8 dB, f32 49,3 dB contre TensorRT |
| #897 | 3,64 → 4,03 ms (f32), natif écarté | mixtes 43,8 dB : seul le f32 tenait 45 dB |
| #919 | 3,40 → 3,10 ms (−9 %), retenu | chaîne 49,1 dB contre TensorRT, 50,0 dB contre le natif f32 |
| #942 | 3,39 → 3,00 ms (−11 %), retenu | chaîne 49,0 dB contre TensorRT (47,4 prévus), 50,0 dB contre le natif f32 |

Mélange retenu par #942 : encodeur f32, blocs 0 à 2 TensorRT, blocs 3 et 4 f16. Dans le lecteur (#919, ABAB,
Re:Zero, ×2 sans SPAN) : 5,85 ms par passe en médiane contre 6,42 ms. Essai au chargement : 3,0 s, dont 1,4 s de
réglage.

Temps de chaque morceau seul (#942, TensorRT ; natif f16, dB contre TensorRT) :

| Morceau | TensorRT | Natif f16 | Plafond f16 couche par couche | Fondu |
|---|---|---|---|---|
| Encodeur (une image) | 0,520 ms | 0,207 ms (71,0 dB) ; f32 0,216 ms (79,2 dB) | 0,26 ms | 0,06 ms |
| Bloc 0 (1/8) | 0,143 | 0,169 (65,2 dB) | 0,08 | 0,08 |
| Bloc 1 (1/8) | 0,103 | 0,118 (67,7 dB) | 0,04 | 0,04 |
| Bloc 2 (1/4) | 0,197 | 0,208 (67,7 dB) | 0,11 | 0,09 |
| Bloc 3 (1/2) | 0,394 | 0,375 (73,6 dB) | 0,27 | 0,16 |
| Bloc 4 (1/1) | 0,849 | 0,765 (73,2 dB) | 0,71 | 0,34 |
| Colle et fin | environ 1,2 (la passe moins les morceaux) | mêmes noyaux | 0,82 | |

Dans le bloc 4 natif (profil f16 de #942) : la première convolution 0,361 ms (elle relit X_4 que la colle vient
d'écrire, 134 Mo), la dernière transposée 0,150 ms (elle écrit Y_4, 58 Mo, dont la fin ne relit que 20 canaux sur
52), les huit ResConv 0,027 ms chacune.

### RIFE 4.22 lite

Même code : ses blocs 1 à 3 ont les formes des blocs 2 à 4 de 4.25. #942 : TensorRT 3,31 ms, mélange 3,20 ms
(−3 %) avec encodeur TensorRT, bloc 0 TensorRT, blocs 1 à 3 f16 ; chaîne 48,1 dB contre TensorRT. Son encodeur
natif ne passe pas sur la chaîne : f16 37,4 dB, f32 43,8 dB, sous les 45 dB. Pour le f32, ce n'est pas son erreur
mais celle de TensorRT (partie 2, « L'encodeur du lite »).

### DistilDRBA v2 lite

Rien de natif. Ce qui tourne est DistilDRBA fusionné (`docs/drba-fusionne.md`) : quatre moteurs TensorRT
(l'encodeur en lot de trois images, trois blocs) reliés par nos noyaux CUDA, allumé par défaut quand la découpe est
valide. Banc : 0,917 ms par image créée contre 2,099 ms pour le réseau entier (−56 %) ; dans le lecteur, ×3 :
0,96 ms contre 3,61 à 3,88 ms. Moteurs : encodeur 0,106 ms, blocs 0,085, 0,125 et 0,379 ms ; nos noyaux 0,152 ms
(dont la fin 0,085). Qualité contre ORT FP16 entier : 57,9 à 67,1 dB. Construction des quatre moteurs : 38 s au
premier lancement. Le FP8 a été essayé (`docs/fp8-interpolation.md`) : −11 % sur le réseau entier, mais il coupe le
fusionné, deux fois plus rapide ; sur les sous-moteurs du fusionné, aucun gain.

Le natif de DistilDRBA est conçu sur le papier ([`drba-natif.md`](drba-natif.md)) : la lecture
des couches (`prepare::rife_native_layers`) le reconnaît déjà ; il manque les entrées et sorties en plans dans les
noyaux et le branchement dans `DrbaFusedEngine`.

## 2. Où en est la recherche

### Pistes retenues

- **Précision par morceau et par couche.** Le f16 de bout en bout tombait à 41,8 dB sur la chaîne (porte : 45 dB
  contre TensorRT). Le corps d'un morceau somme en f16, sa tête (flot et masque) peut sommer en f32, et chaque
  morceau choisit TensorRT ou un réseau natif. C'est ce qui a rendu le natif plus rapide que TensorRT (#919).
- **Mélange d'après l'effet sur la chaîne** (#942). Juger un morceau seul contre TensorRT trompait : l'encodeur
  décidait des 45 dB, et un cran plus juste ne l'est pas toujours contre TensorRT (bloc 4 : 73,2 dB en f16, 69,3 en
  mixtes). On mesure l'effet de chaque option sur la sortie de la chaîne, on prévoit l'erreur d'un mélange entre la
  somme des effets et le carré de la somme des racines, puis on vérifie sur la chaîne.
- **Réglage de la forme de chaque couche au chargement** (tranche, tuile, lignes par warp, K découpé, poids à
  l'avance) : jusqu'à 36 formes par couche, chacune contrôlée sur une sortie remplie de NaN.
- **Entrée du dernier bloc à la volée** (#961, ouverte). La première convolution du bloc 4 calcule X_4 sur sa tuile
  d'entrée depuis les sources de la colle, au lieu de le lire : 268 Mo de trafic deviennent environ 75 Mo. Vérifiée
  dans le cloud au bit près contre l'ancien chemin (émulateur), 77 à 81 dB contre onnxruntime en f32. L'essai
  chronomètre chaque réseau avec cette entrée à la volée et avec la colle à part, et garde le plus rapide.

### Pistes écartées, et pourquoi

- **FP8** (`docs/fp8-interpolation.md`) : RIFE 4.25 entier 31,3 dB (porte 38 dB au pire), sous-moteurs du fusionné
  plus lents ; RIFE est limité par les octets, pas par le calcul.
- **Sommes mixtes** : 43,8 dB, pas assez justes, et pas plus rapides que f16 avec tête f32.
- **Juger chaque morceau seul** (premier mélange de #919) : laissait le bloc 4, plus rapide en f16, à TensorRT.
- **Un écart unique ajouté à la barre de tous les mélanges** (première version de #942) : 0,10 à 0,23 ms de
  l'optimum et 1 % des tirages rendus à TensorRT sur des erreurs très corrélées, contre 0,06 à 0,08 ms et jamais
  tout à TensorRT avec la corrélation recalée.

### Résolu : l'écart de l'encodeur du lite

Seul, notre encodeur f32 sort à 79,6 dB de TensorRT, mais sa chaîne n'en faisait que 43,8 dB (4.25 : 49,1), sous
la porte de 45 dB. Mesuré dans le cloud avec onnxruntime en f32 (le calcul exact), sur la paire de l'essai
(`/mnt/project-files/chantier-natif/lite-encodeur/`, scripts et résultats) :

- la sortie la plus sensible est le masque de la variante mouvement (17 à 20 dB sous le flot ; l'image créée est
  au-dessus de 75 dB partout) ;
- notre encodeur f32, émulé (sommes f32, rangé en FP16), y fait 60,3 dB contre le calcul exact (4.25 : 64,7) ;
  l'encodeur f16 émulé, 38,7 dB (4.25 : 42,4), ce qui retrouve le PC à 1 dB près (37,4 et 41,9) : l'émulation est
  fidèle, et l'erreur du f16 est réelle ;
- les 43,8 et 49,1 dB du PC sont ceux de TensorRT lui-même contre notre natif f32 (43,6 et 49,3 dB). Un
  changement de l'encodeur, même d'un ulp sur 1 % de ses valeurs, redistribue les arrondis des blocs FP16 qui le
  suivent : à travers des blocs exacts il se lit à 78 dB, à travers des blocs FP16 au niveau de leur propre bruit.
  Juger l'encodeur contre TensorRT, c'est mesurer le bruit des blocs de TensorRT. Sur le lite, ce bruit est à
  43,6 dB : aucun encodeur différent du sien ne pouvait tenir 45 dB, même exact.

Les blocs f32 natifs sont d'ailleurs plus loin de TensorRT que nos blocs f16 (4.25, bloc 1 : 61,9 contre 67,7 dB) :
TensorRT somme lui aussi en f16 dans les blocs.

La correction est une porte jugée contre le calcul exact (partie 4, point 2). Gain attendu sur le lite : son
encodeur natif (0,213 ms au lieu de 0,521), soit environ 2,9 ms par passe au lieu de 3,20.

### Questions ouvertes

- **Blocs 0 à 2 de 4.25** : TensorRT y reste plus rapide (0,143 contre 0,169 ms au bloc 0). Ces blocs ne remplissent
  pas les 60 SM (grilles de 34 × 60 à 136 × 240) : chaque couche coûte surtout sa latence, 10 à 17 µs, onze couches
  par bloc. Régler les formes n'y gagnera plus ; il faut fondre plusieurs couches en un lancement.
- **Turing (sm_75)** : jamais mesuré. Le noyau de l'entrée à la volée n'y a qu'un bloc par SM (37 Ko de mémoire
  partagée sur 64 Ko) ; l'essai garde alors la colle à part si elle est plus rapide.

### DistilDRBA : étudié ou non

- Étudié : la structure (encodeur 3 → 16 canaux en lot de trois images au quart de la taille, trois IFBlock de 192,
  128 et 96 canaux aux réductions 16, 8 et 4, entrées 58 et 109 canaux), les formats en plans du fusionné, ce qui
  manque aux noyaux, le gain attendu.
- Pas étudié : la justesse d'un DistilDRBA natif contre TensorRT (aucun noyau n'a tourné sur ses couches), la
  sensibilité de sa chaîne aux sommes f16, le temps de ses blocs en natif.

## 3. Plafonds théoriques et au-delà

### Le calcul

Hypothèses ([`rife-plafonds.md`](rife-plafonds.md)) : 4070 Ti à 160 TFLOPS en sommes f16, 80 en f32, 321 en FP8, 504 Go/s. Chaque
couche lit son entrée une fois, écrit sa sortie une fois, lit ses poids, en FP16, aux canaux utiles. Plafond d'une
couche : le plus grand de « calcul / débit » et « octets / bande passante » ; d'un morceau, la somme de ses couches.
Borne fondue : seules l'entrée et la sortie d'un morceau passent par la mémoire. La colle et la fin : leurs octets
dans le découpage d'aujourd'hui. Formes tirées des graphes du catalogue par l'inférence de formes d'onnx (scripts
dans `/mnt/project-files/chantier-natif/plafonds-scripts/`).

| Modèle | Mesuré | Couche par couche (f16) | Fondu | Écart à combler |
|---|---|---|---|---|
| RIFE 4.25 | 3,00 ms (mélange) ; TensorRT 3,39 | 2,28 ms | 1,58 ms | 1,3× le plafond, 1,9× la borne fondue |
| RIFE 4.22 lite | 3,20 ms ; TensorRT 3,31 | 2,28 ms | 1,59 ms | 1,4×, 2,0× |
| DistilDRBA v2 lite | 0,917 ms (fusionné) | 0,37 ms | environ 0,27 ms | 2,5×, 3,4× |

DistilDRBA en détail (calculé pour ce bilan, mêmes hypothèses) : encodeur des trois images 1,8 GFLOP, 0,067 ms ;
bloc 0 (68 × 120) 0,023 ms ; bloc 1 (136 × 240) 0,053 ms ; bloc 2 (272 × 480) 0,150 ms ; noyaux de colle et de fin
0,075 ms de borne mémoire (mesurés 0,152). Ses moteurs TensorRT sont à 1,6× (encodeur) et 2,4 à 3,7× (blocs) leur
plafond : beaucoup de marge relative, peu en absolu.

En images par seconde : une passe crée une image. RIFE 4.25 à 3,39 ms permet 295 passes par seconde, à 3,00 ms 333,
au plafond par couche 439, à la borne fondue 633. Pour une source 1080p à 23,976 i/s en ×2 (24 images créées par
seconde), le moteur RIFE occupe le GPU 8,1 % du temps avec TensorRT, 7,2 % avec le mélange, 5,5 % au plafond par
couche et 3,8 % à la borne fondue ; le double en ×3. Ces parts ne comptent que le moteur, pas le reste de la chaîne
du lecteur (le chiffre de 6,9 ms par image créée et 30 % d'occupation de la fiche est une mesure du lecteur entier,
d'une autre nature).

### Ce qui reste sous le plafond par couche

- **Le dernier bloc est déjà à 1,08× son plafond** (0,765 contre 0,705 ms) : limité par les octets, il ne gagne
  plus qu'en fondant.
- **La colle et la fin pèsent environ 1,2 ms contre 0,82 de plafond.** Le plus gros est la colle de X_4 (0,39 ms de
  plafond), puis la colle de X_3 (0,18 ms) et la fin (0,18 ms).
- **Les blocs grossiers sont à 1,7 à 2,8× leur plafond**, TensorRT comme le natif : latence des lancements.
- **L'encodeur natif est à 3,6× sa borne fondue** (0,21 contre 0,06 ms).

### Les leviers, du plus sûr au plus risqué

Gains pour RIFE 4.25 par passe, estimés sur le papier, jamais mesurés sauf mention.

| Levier | Gain attendu | Risque |
|---|---|---|
| Entrée du dernier bloc à la volée (#961) | 0,4 ms (plafond 0,46) | Faible : au bit près, l'essai garde la colle à part si elle gagne |
| Fin fondue dans la dernière transposée (écrite) | 0,15 à 0,2 ms : Y_4 non écrit (58 Mo), moitié du calcul de la transposée (32 canaux sur 64), Z non relu | Faible : mêmes arrondis, vérifiable au bit près |
| Flot recalculé dans la fin plutôt que lu dans X_4 | environ 0,05 ms (33 Mo) | Faible |
| Colle de X_3 à la volée (bloc 3, d = 2) | environ 0,12 ms (93 Mo) | Moyen : variante réduite de la colle (quatre points par sortie) |
| Couches d'un bloc grossier en un lancement (halo recalculé) | environ 0,2 ms pour les blocs 0 à 2 | Moyen : registres et mémoire partagée |
| Encodeur fondu par tuiles | jusqu'à 0,15 ms | Moyen : halo de quatre couches |
| Rétention dans le L2 (fenêtre d'accès persistante, ordre des tuiles) | quelques centièmes de ms | Faible, mais ne se juge qu'au PC |
| Passe entière en un noyau persistant | jusqu'à la borne fondue, 1,58 ms | Élevé : complexité, registres, Turing |
| Activations rangées en FP8 (stockage, calcul en FP16) | jusqu'à 0,3 ms sur les couches limitées par les octets | Élevé : 3 bits de mantisse, le flot ne les supporte pas ; seulement les attributs, porte de 45 dB obligatoire |
| Calcul en FP8 | moins de 0,05 ms (RIFE est limité par les octets) | Élevé : 31,3 dB mesurés sous TensorRT |
| Sparsité 2:4 des poids | moins de 0,05 ms (même raison) | Hors périmètre : demande d'élaguer puis de réentraîner, le modèle change |
| Dernier bloc en demi-résolution (le « scale 0,5 » de RIFE) | environ 1 ms (bloc 4, colle et fin divisés par 4 environ) | Élevé pour la qualité : l'image change, petits mouvements et contours ; à juger au banc de qualité, peut-être en 4K seulement ou plan par plan |

Si les cinq premiers tiennent, RIFE 4.25 passerait d'environ 3,0 à 2,1 ms par passe (476 passes par seconde), sous
le plafond par couche d'aujourd'hui, qui suppose chaque couche écrite et relue.

DistilDRBA : ses blocs natifs au rendement du natif de RIFE (1,4× le plafond au bloc 3) feraient 0,59 ms de blocs
TensorRT en environ 0,32 ms ; avec l'encodeur, environ 0,3 ms de moins par image créée sur 0,92. L'autre gain est de
ne plus construire les quatre moteurs (38 s), mais seulement avec une référence indépendante de nos noyaux à l'essai
(calcul au processeur, `principes.md`).

## 4. Ce qui reste à écrire et à tester, dans l'ordre

Règle de Tatoub : tout ce qui se vérifie hors carte se fait dans le cloud ; le PC ne sert qu'à chronométrer, et
seulement après les chantiers qui n'ont pas encore passé leur porte.

**Après la séance du 19 novembre** (dans l'ordre du gain) :

a. **Fusion de #961** : faite le 19 novembre (520f4632), natif par défaut et verdict gardé compris, sans
   attendre son ci.yml de branche (choix de Tatoub pour la rc.1 : un seul ci.yml sur main ensuite ; le
   dernier ci.yml complet de la branche, sur e05a3a5, était vert). Nouveau paquet TensorRT Linux à faire.
b. **La garde des morceaux de la porte** : 60 dB contre TensorRT ne passait jamais (TensorRT FP16 est lui-même à
   51 à 59 dB de nos morceaux f32). Abaissée à 48 dB (`kPieceFloorDb`), le plancher de chaîne gardé à 42 :
   un défaut du natif qui pèse autant que le bruit propre de TensorRT fait encore tomber le morceau dessous
   (test du natif faux à 46 dB, et une mutation qui retire la garde le fait échouer). Le témoin indépendant
   (le morceau f32 par la forme de référence du banc) reste possible s'il faut remonter la garde. Gain
   attendu sur le lite : l'encodeur natif (0,21 contre 0,52 ms), environ −0,3 ms ; à vérifier au PC, le
   verdict gardé se refait seul (l'empreinte des sources change).
c. **Les blocs 0 à 2 de 4.25** restent à TensorRT (0,143, 0,104, 0,200 ms seuls contre 0,170, 0,119, 0,210 en
   f16) : un lancement par bloc grossier entier (noyau persistant, barrière de grille entre les couches, les
   sommes partielles de K découpé finies dans la même grille). Pas encore écrit : le gain dépend de ce qui
   coûte entre deux couches, et en lecture la carte tourne à moins d'un cinquième d'occupation, horloges
   basses. La séance PC qui suit la rc.1 le mesure ([`rife-seance-pc.md`](rife-seance-pc.md), étape 1) avec
   `tools/rife/nsys_passe.py` : temps des petites grilles et écart juste avant elles, natif contre TensorRT.
   Si l'écart pèse au moins le tiers, le noyau persistant ; sinon la forme des couches.
d. **Les noyaux lourds du profil** : `glue_conv_kernel<4,16>` (0,52 ms à l'essai), `fused_glue<1,3>` (la colle de
   X_3), `warp_blend` (0,90 ms, hors de la passe). **La colle de X_3 à la volée est écrite** (sur la branche de
   #987) : la première couche de l'avant-dernier bloc (X_3 de 4.25, X_2 du lite, à moitié) la refait comme celle
   du dernier, quatre points pleine résolution par point de X dans l'ordre de `fused_glue`. Identique au bit sur
   l'émulateur processeur, 24 noyaux sans débordement de sm_75 à sm_120 ; l'essai la chronomètre bloc par bloc,
   le verdict gardé retient les blocs restés à part (format 2). Gain attendu de l'ordre de 0,1 ms par passe
   (33 Mo écrits puis relus en moins, la colle refaite sur un halo de 1/8) ; à mesurer à la séance d'après la
   rc.1. `glue_conv_kernel<4,16>` est à son plafond d'octets ; `warp_blend` reste à voir.
e. **nsys refait** avec `--delay=15` pour l'occupation, après l'essai.

1. **#961, l'entrée du dernier bloc à la volée** (cloud). Deux relectures sévères, points corrigés (422c6a6 : le
   flot du dernier bloc relu après la passe de l'essai, car une variante image ne montrait pas un flot oublié ; la
   justesse des morceaux mesurée avec la colle que chaque réseau garde). ci.yml en cours ; fusion quand il est
   vert. Nouveau paquet TensorRT Linux (src/gpu touché).
2. **La porte contre le calcul exact** (cloud, pour le lite). La référence du mélange devient la chaîne native
   f32 au lieu de TensorRT ; l'effet de chaque option, TensorRT compris, se mesure le reste en natif f32. Un
   mélange passe s'il tient 45 dB contre elle ou, quand TensorRT lui-même n'y arrive pas, s'il en est au moins
   aussi proche que TensorRT. Garde, pour qu'un natif faux ne devienne jamais sa propre référence : chaque morceau
   du réseau f32 doit sortir à 60 dB au moins de TensorRT sur la même entrée (une comparaison sans bruit des blocs
   suivants : 61,9 à 79,6 dB mesurés), et les deux chaînes à 42 dB au moins l'une de l'autre ; sinon l'ancienne
   règle (45 dB contre TensorRT) s'applique. La logique du choix sort du code CUDA pour être testée sans GPU : un
   natif faux retombe sur l'ancienne règle, un TensorRT bruité laisse passer un natif juste. Le choix de 4.25 ne
   peut que rester le même ou devenir plus rapide en tenant 45 dB contre le calcul exact ; DistilDRBA n'a pas de
   natif. Vérification au PC dans la séance du point 8.
   **Fait** (arbre `wip-porte`, commits fa6ba5f et 1ede233, PR après #961) : `rife_mix_choice.h/.cpp` (module
   `cuda_rife_mix`, C++ pur) et `tests/unit/gpu/rife_mix_choice_test.cpp` (8 cas : TensorRT bruité et natif juste,
   natif faux que seule la garde des morceaux voit, 4.25 à 44 dB refusé, NaN, variante mouvement à deux barres,
   rôle le plus bas, tailles incohérentes). Une mutation qui retire la garde des morceaux fait échouer le test du
   natif faux. La relecture sévère a trouvé que la justesse d'un morceau, mesurée sur tout le tenseur, diluait un
   défaut du seul masque d'environ 11 dB : elle se mesure maintenant par rôle (flot, masque, traits), chacun à son
   pic. Le plancher de la chaîne est monté de 40 à 42 dB (43,6 mesurés sur le lite). Risque connu : les morceaux
   par rôle sont plus sévères que les 61,9 dB mesurés sur tout le tenseur ; un refus à tort ne fait que revenir à
   l'ancienne règle, la séance PC le dira (ligne « référence du mélange » du journal).
3. **La fin fondue** (cloud). Écrit : `rife_finish.cuh` (la fin au point, partagée par fused_finish et le nouveau
   noyau, SASS de fused_finish inchangée), `rife_native_finish.cu` (neuf noyaux, 80 à 163 registres, sans
   débordement), `conv_sums` sorti de `conv_kernel` (mêmes opérations). Reste : le branchement (réseau, moteur,
   essai : les fusions du dernier bloc en un masque, l'essai chronomètre chaque jeu), le test au bit près sur
   l'émulateur contre « transposée puis fused_finish », les dB contre onnxruntime, relecture, PR.
   **Branché et vérifié** (arbre `wip-fin-fondue`, sur b68466c) : `Layer::finish` (la couche du dernier bloc,
   posée au réglage si la forme retenue n'a pas de K découpé et qu'un noyau existe pour sa tranche et ses lignes),
   `RifeNativeNet::block(…, finish)` et `can_finish`, `RifeFusedEngine::run` qui saute fused_finish quand le
   dernier bloc natif la fait ; l'essai par morceau (`run_piece`) écrit toujours Y. Émulateur (`finish_test.cpp`) :
   7 cas, sommes f16 et f32, tranches 16, 32 et 64, une ou deux lignes par warp, image et mouvement, fenêtre et
   Catmull-Rom, grille qui ne tombe pas juste : image, flot et masque identiques au bit, Y jamais écrit.
   `glue_test` rejoué sur les noyaux remaniés : identique au bit. `conv_kernel` remanié : SASS sm_89 qui ne
   diffère que par l'allocation des registres. Les dB contre onnxruntime sont donc ceux du chemin d'avant.
   Relecture sévère faite et corrigée (essai à NaN, A/B fondue contre à part, forme vérifiée, 48 Ko). PR après #961.
4. **La colle de X_3 à la volée** (cloud), puis le flot recalculé dans la fin.
5. **Les blocs grossiers fondus**, puis l'encodeur fondu (cloud pour la justesse, PC pour le temps).
6. **DistilDRBA natif** (cloud) : `In::kPlanes` et `Out::kPhasePlanes` dans `rife_native_kernels.cuh`,
   `RifeNativeNet::create_planar`, `DrbaFusedEngine::try_native`, `--drba-native` (fiche `drba-natif.md`).
7. **La passe entière en un noyau** (cloud), une fois les morceaux fondus.
8. **Une seule séance PC, chronométrage seulement**, après #961, la porte et la fin fondue, et après les chantiers
   qui n'ont pas passé leur porte (avant GMFSS) : le verdict de l'essai sur 4.25 et le lite, l'ABAB dans le
   lecteur et l'occupation du GPU (séance finale de `docs/rife-natif.md`). Ses commandes partent au fil de
   coordination quand la fin fondue est fusionnée.
9. **Le natif allumé par défaut**, si cette séance passe (50 dB contre le natif f32, occupation sous 23 %) : une PR
   courte, `--no-rife-native` pour revenir à TensorRT, l'ancien `--rife-native` toujours lu, ci.yml sur la branche
   (Windows compris) avant la fusion. Rien ne s'y oppose en justesse : l'essai au chargement garde déjà TensorRT
   si le mélange ne tient pas la porte ou ne le bat pas de 2 % sur une passe chronométrée. Avec elle : chercher à
   mettre le verdict de l'essai (3 s) en cache par carte, pilote et modèle, comme le cache des moteurs, pour ne le
   payer qu'une fois.

Commandes du cloud :

```sh
# nvcc sans kit CUDA : python -m venv nvenv && nvenv/bin/pip install nvidia-cuda-nvcc nvidia-cuda-cccl \
#   nvidia-cuda-runtime nvidia-cuda-cuobjdump ; onnx et onnxruntime dans un autre venv pour les plafonds
# Construction d'un fichier CUDA pour une architecture, débordements interdits
NV=nvenv/lib/python3.13/site-packages/nvidia/cu13
$NV/bin/nvcc -std=c++20 -I src -I $NV/include --ptxas-options=--verbose,--warn-on-spills -Werror=all-warnings \
  -Xcompiler=-Wall,-Wextra -arch=sm_89 -c src/gpu/cuda/backend/rife_native_finish.cu -o /tmp/f.o
# Règles de rules.yml
for s in check-comments check-file-size check-no-getenv check-path-string check-log-paths check-no-atoi \
         check-utilities; do bash tools/ci/$s.sh; done
clang-format-20 --dry-run --Werror src/gpu/cuda/backend/*.cpp src/gpu/cuda/backend/*.h
# Émulateur processeur des noyaux : patch.py, construction par g++ et tests dans rife-natif-emu/LISEZMOI.md
cd /mnt/project-files/chantier-natif/rife-natif-emu && python3 patch.py <dépôt> out
# Plafonds d'un modèle (venv avec onnx)
python -I plafonds-scripts/shapes.py rife_v4.25.onnx rife_v4.25.json
python -I plafonds-scripts/plafonds.py rife_v4.25
```

Commandes du PC (binaire `linux-release`) :

```sh
# Le verdict de l'essai (mélange, dB, temps de la passe), l'entrée à la volée gardée ou non
kirari --debug play --rife-native --rife=2 --no-upscale --start=960 extrait.mkv 2>&1 | grep -i "rife natif"
kirari --debug play --rife-native --rife-model=rife_v4.22_lite.onnx --rife=2 --no-upscale --start=960 \
  extrait.mkv 2>&1 | grep -i "rife natif"
# ABAB dans le lecteur : voir « Séance finale » dans docs/rife-natif.md
```

## 5. Point de reprise

- **Dépôt** : `Ta2b/kirari-player`, branche `claude/project-thread-i22zy2`. PR #961 ouverte : 78bfe89 (entrée à la
  volée), 8577c18 (fusion de main), cb71272 (doc), f75f66f (points de relecture : colle à la volée ou à part au
  chrono, X du dernier bloc à NaN dans la passe comparée, borne de mémoire partagée), 422c6a6 (seconde relecture :
  flot du dernier bloc relu après la passe, justesse mesurée avec la colle gardée).
- **#961** : b68466c (troisième relecture : format de glue_layer, flot relu en variante image seulement).
- **Porte et fin fondue** : sur la branche de #961 depuis e05a3a5 (3f812cc, cabb619, af27407 la porte,
  0c549e5 la fin fondue, e05a3a5 la fusion de main). #961 se fusionne quand ci.yml est vert sur e05a3a5.
- **Fichiers** : `src/gpu/cuda/backend/rife_native_*.cu|h|cuh` (noyaux, réseau, réglage, essai, mélange),
  `rife_glue.cuh` (la colle au point), `rife_fused_native.cpp` (la chaîne de l'essai), `rife_fused_engine.*`
  (aiguillage), `src/prepare/rife_native.h` (lecture des poids) ; DistilDRBA : `drba_fused_engine.*`,
  `kernels/spec/drba_fused.h`, `prepare/drba_split.h`.
- **Émulateur** : `/mnt/project-files/chantier-natif/rife-natif-emu/` (le code CUDA du dépôt tel quel sur
  processeur, mma, ldmatrix et cp.async émulés ; tests contre l'ancien chemin et contre onnxruntime).
- **Scripts des plafonds** : `/mnt/project-files/chantier-natif/plafonds-scripts/` (formes par onnx, plafonds par
  morceau ; `shapes_drba.py` pour DistilDRBA).
- **Fiches** : [`rife-natif.md`](rife-natif.md), [`rife-plafonds.md`](rife-plafonds.md),
  [`drba-natif.md`](drba-natif.md), [`rife-seance-pc.md`](rife-seance-pc.md) dans `docs/` ; les notes de travail
  (`principes.md`, émulateur, scripts) restent dans `/mnt/project-files/chantier-natif/`.
- **Règles** : fichiers sous 800 lignes, clang-format-20, pas de getenv, commentaires en français sans nom ni date,
  aucune ligne de libaji ni de mpv ; toucher `src/gpu` demande un nouveau paquet TensorRT Linux ; relecture sévère
  de chaque PR, ci.yml sur la branche avant de fusionner une PR CUDA ; pas d'étiquette ni de release sans Tatoub.
