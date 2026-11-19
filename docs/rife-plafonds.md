# RIFE 4.25 et 4.22 lite en 1080p : plafond théorique de chaque morceau sur la RTX 4070 Ti

Calculé dans le cloud le 19 novembre 2026, à partir des graphes du catalogue (`rife_v4.25.onnx`,
`rife_v4.22_lite.onnx`), formes inférées par onnx pour une entrée remplie de 1920×1088. Une passe ×2 de lecture :
un encodeur (l'image B ; celle de A est reprise de la passe d'avant), tous les blocs, les colles et la fin.
Comparé aux mesures du PC : #919 (`rife-natif.log`) et #942 (`rife-natif-425.log`, `rife-natif-422.log`).

## Hypothèses

- Débits de la 4070 Ti (cœurs tensoriels sans creux) :
  - sommes f16 (accumulation FP16) : 160 TFLOPS ;
  - sommes f32 (accumulation FP32, deux fois plus lente sur une GeForce) : 80 TFLOPS ;
  - FP8 (accumulation FP16) : 321 TFLOPS.
- Bande passante : 504 Go/s.
- Octets minimaux d'une couche : l'entrée lue une fois, la sortie écrite une fois et les poids, en f16, aux canaux
  utiles (15, 28, 52), pas au pas HWC8 (16, 32, 56). LeakyRelu, bêta des ResConv, additions résiduelles et
  DepthToSpace sont supposés fondus dans les convolutions.
- Plafond d'une couche : le plus grand de « FLOP / débit » et de « octets / bande passante ». Plafond d'un
  morceau : la somme de ceux de ses couches.
- En plus, comme borne basse : le **morceau fondu**, dont seules l'entrée et la sortie passent par la mémoire,
  les activations du milieu restant sur la puce ou dans le L2 (48 Mo sur la 4070 Ti).
- La colle et la fin ne calculent presque rien : leur plafond, ce sont leurs octets dans le découpage
  d'aujourd'hui (X_k écrit par la colle puis relu par le bloc). Leur temps n'est pas mesuré à part : c'est la
  passe moins la somme des morceaux.

## RIFE 4.25 (réductions 8, 8, 4, 2, 1)

| Morceau | Taille de travail | GFLOP | Mo minimaux | Plafond f16 | Plafond f32 | Plafond FP8 | Fondu f16 | Mesuré au PC (#919 ; #942) |
|---|---|---|---|---|---|---|---|---|
| Encodeur (une image) | 544×960, sortie 1088×1920 | 6,3 | 130 | 0,26 ms | 0,26 ms | 0,26 ms | 0,06 ms | TensorRT 0,524 ; natif f16 0,211, f32 0,215 (retenu) |
| Bloc 0 | 136×240 (1/8) | 12,4 | 25 | 0,08 ms | 0,16 ms | 0,05 ms | 0,08 ms | TensorRT 0,143 (retenu) ; f16 0,169 |
| Bloc 1 | 136×240 (1/8) | 5,8 | 17 | 0,04 ms | 0,08 ms | 0,03 ms | 0,04 ms | TensorRT 0,103 (retenu) ; f16 0,118 |
| Bloc 2 | 272×480 (1/4) | 13,6 | 47 | 0,11 ms | 0,18 ms | 0,09 ms | 0,09 ms | TensorRT 0,197 (retenu) ; f16 0,208 |
| Bloc 3 | 544×960 (1/2) | 26,0 | 136 | 0,27 ms | 0,38 ms | 0,27 ms | 0,16 ms | TensorRT 0,394 ; f16 0,372 (retenu) |
| Bloc 4 | 1088×1920 (1/1) | 31,6 | 355 | 0,71 ms | 0,71 ms | 0,71 ms | 0,34 ms | TensorRT 0,870 (retenu en #919) ; f16 0,761 (retenu en #942) |
| Colle et fin | | 0 | 414 | 0,82 ms | 0,82 ms | 0,82 ms | | environ 1,2 ms (la passe moins les morceaux) |
| **Passe ×2** | | **95,8** | **1 123** | **2,28 ms** | **2,58 ms** | **2,23 ms** | **1,58 ms** | **mélange 3,10 ; 3,00** (TensorRT 3,40 ; 3,39) |

Colle et fin en détail : entrée X_0 4,6 Mo, colle X_1 6,3, X_2 23,6, X_3 92,8, **X_4 194,1 Mo (0,39 ms)**, fin
(flot et masque) 91,9 Mo (0,18 ms).

## RIFE 4.22 lite (réductions 8, 4, 2, 1)

Ses blocs 1 à 3 ont les formes des blocs 2 à 4 de 4.25, et les mêmes temps mesurés.

| Morceau | Taille de travail | GFLOP | Mo minimaux | Plafond f16 | Plafond f32 | Plafond FP8 | Fondu f16 | Mesuré au PC (#942) |
|---|---|---|---|---|---|---|---|---|
| Encodeur (une image) | 544×960, sortie 1088×1920 | 6,3 | 130 | 0,26 ms | 0,26 ms | 0,26 ms | 0,06 ms | TensorRT 0,521 (retenu) ; f16 0,208, f32 0,213 |
| Bloc 0 | 136×240 (1/8) | 12,4 | 25 | 0,08 ms | 0,16 ms | 0,05 ms | 0,08 ms | TensorRT 0,143 (retenu) ; f16 0,170 |
| Bloc 1 | 272×480 (1/4) | 23,3 | 59 | 0,17 ms | 0,30 ms | 0,12 ms | 0,15 ms | TensorRT 0,278 ; f16 0,270 (retenu) |
| Bloc 2 | 544×960 (1/2) | 26,0 | 136 | 0,27 ms | 0,38 ms | 0,27 ms | 0,16 ms | TensorRT 0,394 ; f16 0,374 (retenu) |
| Bloc 3 | 1088×1920 (1/1) | 31,6 | 355 | 0,71 ms | 0,71 ms | 0,71 ms | 0,34 ms | TensorRT 0,848 ; f16 0,763 (retenu) |
| Colle et fin | | 0 | 405 | 0,80 ms | 0,80 ms | 0,80 ms | | environ 1,1 ms (la passe moins les morceaux) |
| **Passe ×2** | | **99,6** | **1 109** | **2,28 ms** | **2,61 ms** | **2,20 ms** | **1,59 ms** | **mélange 3,20** (TensorRT 3,31) |

## Ce qu'on en tire

- **La passe est à 1,3× (4.25) ou 1,4× (lite) son plafond par couche, et à 1,9× ou 2,0× de la borne fondue.** Le mélange
  de 4.25 fait 3,00 ms contre 2,28 ms de plafond par couche et 1,58 ms de borne fondue.
- **Le modèle par couche est pessimiste dès que les activations tiennent dans le L2.** L'encodeur natif (0,21 ms)
  passe sous son propre plafond par couche (0,26 ms) : ses activations du milieu (16,7 Mo chacune) restent dans
  les 48 Mo du L2. Le vrai plafond est entre les deux colonnes, plus près de la borne fondue pour les blocs
  grossiers et l'encodeur, plus près du plafond par couche pour le dernier bloc (X_4 pèse 117 à 134 Mo).
- **Le dernier bloc est déjà à 1,08× son plafond par couche** (0,761 ms contre 0,705). Il ne gagnera plus qu'en
  fondant : il est limité par les octets.
  - Premier levier : l'entrée calculée à la volée (sur la branche, en pause). Elle retire l'écriture de X_4 par la
    colle et sa relecture par la première couche, soit environ 234 Mo, **0,46 ms de plafond par passe**. Le gain
    attendu, 0,4 ms, en est proche.
  - Ensuite, la borne fondue du bloc (0,34 ms) ne serait approchée qu'en gardant ses 272×480×32 canaux entre
    couches (8,4 Mo, dans le L2), ce qu'ils font sans doute déjà en partie.
- **Les blocs grossiers (0 à 2 de 4.25) sont à 1,7× à 2,8× leur plafond, TensorRT comme le natif.** Ils sont
  limités par le calcul sur le papier, mais leurs grilles (34×60 à 136×240) n'occupent pas les 60 SM : chaque
  couche coûte surtout sa latence (10 à 17 µs mesurés, 11 couches par bloc). Le levier est de fondre plusieurs
  couches d'un bloc dans un seul lancement, pas de régler les formes. Le plus à gagner : environ 0,2 ms pour les
  trois blocs.
- **L'encodeur natif est à 3,6× sa borne fondue** (0,21 ms contre 0,06). Fondre ses quatre couches par tuiles
  gagnerait jusqu'à 0,15 ms par passe.
- **La colle et la fin pèsent environ 1,2 ms mesurées, contre 0,82 ms de plafond.** Le plus gros est la colle de
  X_4 (0,39 ms), qui disparaît avec l'entrée à la volée, puis la colle de X_3 (0,18 ms) et la fin (0,18 ms).
- **Ni le FP8 ni le f32 ne changent grand-chose.** La passe est limitée par les octets : le FP8 ne retire que
  0,05 à 0,08 ms de plafond. Les sommes f32 partout n'ajouteraient que 0,3 ms, surtout sur les blocs 0 à 3.

## À mesurer au PC (si le chantier reprend)

- Le profil nsys d'une passe : le temps de chaque colle et de la fin, que l'essai ne mesure pas à part, et les
  octets lus et écrits par noyau. Il dirait si les activations du milieu restent bien dans le L2.
