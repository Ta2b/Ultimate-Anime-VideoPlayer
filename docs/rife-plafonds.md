# RIFE 4.25 et 4.22 lite en 1080p : plafond théorique de chaque morceau sur la RTX 4070 Ti

Le [recalcul du 21 novembre](#recalcul-du-21-novembre) inclut les graphes exportés par le lecteur, Winograd
et la fusion actuelle. Les premières sections décrivent le découpage et les mesures du 19 novembre.

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

## Recalcul du 21 novembre

Le lecteur a exporté les morceaux RIFE 4.25 (`fusion3`, `de0fb71ae50fe082`), RIFE 4.22 Lite
(`fusion3`, `8b30b93ef8ab0b3c`) et DistilDRBA v2 Lite (`fusion2`, `40bda3486348dd1a`) en
1920×1088. Les tailles sont inférées à partir des poids, strides, paddings et dilatations de chaque ONNX.
Une convolution compte deux opérations par produit accumulé. Une convolution transposée distribue ses produits
depuis la grille d'entrée. Les 20 canaux utiles du dernier bloc suffisent aux quatre composantes de flot et au
masque avec leurs quatre phases. La tête seule économise ainsi 4,278 GFLOP pour chaque RIFE et 0,802 pour
DistilDRBA par rapport à l'export 52 canaux.

Pour chaque convolution dense 3×3 de stride 1, Winograd F(2×2,3×3) emploie 16 produits par tuile 2×2 et par
couple de canaux, contre 36 en direct. L'hypothèse suppose les poids transformés d'avance et les transformées
d'activation fusionnées. Elle conserve les formats d'entrée, de somme et de sortie du calcul, avec un ordre
d'opérations différent. La fidélité à précision égale reste à valider pour chaque bloc. Le calcul des convolutions
inclut les zéros de halo et exclut le remplissage des canaux MMA, le warp, les fonctions élémentaires et les
accès mémoire. Les transformées d'activation 2D représentent par ailleurs environ 1,10 milliard d'additions
pour chaque RIFE avec un encodage et 129 à 172 millions pour DistilDRBA selon l'état de son cache.
[Débits denses NVIDIA pour la RTX 4070 Ti, tableau 5](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf#page=51).

| Modèle et état | Direct, GFLOP | Winograd 2D, GFLOP | Temps Tensor optimiste, ms | Médiane GPU du moteur, ms | Quotient calcul seul |
|---|---:|---:|---:|---:|---:|
| RIFE 4.25, encodage déjà prêt | 85,157 | 49,060 | 0,306–0,412 | 2,162 | 14,1–19,1 % |
| RIFE 4.25, encodage de B | 91,491 | 52,720 | 0,352–0,458 | 2,162 | 16,3–21,2 % |
| RIFE 4.22 Lite, encodage déjà prêt | 88,992 | 50,890 | 0,317–0,435 | 2,165 | 14,7–20,1 % |
| RIFE 4.22 Lite, encodage de B | 95,326 | 54,550 | 0,340–0,474 | 2,165 | 15,7–21,9 % |
| DistilDRBA, trio déjà encodé | 23,544 | 13,421 | 0,102–0,137 | 0,636 | 16,0–21,6 % |
| DistilDRBA, nouveau trio | 26,687 | 16,063 | 0,127–0,170 | 0,636 | 19,9–26,7 % |

La fourchette applique 160,4 TFLOP/s aux sommes FP16 et 80,2 aux sommes FP32. Elle garde la précision actuelle
des morceaux natifs, y compris l'encodeur FP32 de 4.25 et la projection FP16 de l'encodeur hybride Lite. Elle
affecte alternativement les deux débits aux morceaux TensorRT, dont le mélange de tactiques est inconnu dans
cette capture. Pour DistilDRBA, les trois convolutions
de bord du dernier bloc restent en FP32. Le cas chaud réemploie l'encodeur et les sommes FP32 des 48 premiers
canaux de sa première convolution de stride 2. Le quotient divise chaque temps théorique par la médiane de
toutes les passes du moteur correspondant. La médiane par état de cache n'est pas disponible. RIFE a réutilisé
les deux encodages sur 186/305 et 192/311 passes respectivement. Ces quotients donnent un repère sur le calcul
des convolutions. Un palier pour tout le moteur doit aussi modéliser les noyaux entre les morceaux.

Les mesures `rife425-24x8`, `rife422-24x8` et `drba-24x8` viennent de la série 24p ×8 en fenêtre sans SPAN
du 21 novembre. Pour chacun des trois modèles, la jointure des nouvelles images avec les retours
`wp_presentation` par identifiant donne 165,00 i/s et 693/693 nouvelles images confirmées sur le segment
central d'environ 4,2 s. Les médianes de composition dans le lecteur sont 0,156, 0,150 et 0,114 ms dans le
même ordre. Le p95 du moteur est de 2,861, 2,689 et 1,023 ms. Le warp et mélange RIFE intervient après
le moteur et coûte 0,384 et 0,392 ms en médiane. Le temps moteur DistilDRBA inclut son warp final RGB.
Le rapprochement a été effectué avec `bilan-corrige.py` de la série.

### Transferts aux frontières des blocs

- RIFE produit l'entrée des deux blocs fins à la volée. L'aller-retour de `X_D2` en HWC32 (66,85 Mo) et
  celui de `X_D1` (267,39 Mo) cèdent la place à un seul aller-retour du flot pleine résolution HWC4
  entre l'entrée et la tête de D1 (33,42 Mo), soit une économie nette de 300,81 Mo.
- La tête D1 fusionnée retire encore jusqu'à 79,38 Mo d'échanges logiques en évitant une écriture de
  `Y_D1` au pas 56 et sa lecture sur 20 canaux. Les 20,89 Mo de flot et masque pleine résolution
  restent la sortie du moteur en mode mouvement. La liaison HWC4 compacte actuelle représente à elle seule
  0,066 ms dans un scénario où son aller-retour passerait entièrement en DDR à 504 Go/s. Les quelques
  étapes grossières du moteur font encore circuler leurs `X_k` et `Y_k` globaux.
- DistilDRBA conserve les frontières planaires `X_0` et `X_1` (16,12 Mo aller-retour pour les deux).
  Le bloc fin travaille sur `X_2` groupé au pas 112. À chaud, la colle réécrit 56 canaux et la
  convolution relit 64 canaux avec le remplissage (31,33 Mo logiques). À froid, les 112 canaux
  sont écrits et relus (58,49 Mo). La reprise à chaud relit au moins 6,27 Mo de sommes intermédiaires
  FP32. La tête écrit `Y_2` à basse résolution (pas 24) et le warp final lit 20 canaux de cette sortie.
  Le warp lit deux sources RGB selon le flot et écrit une sortie RGB, soit un volume de référence
  de 37,60 Mo pour des lectures couvrant les deux sources une fois. Les déplacements et la localité
  des lectures changent ce volume.

Ces volumes sont des tailles logiques aux frontières des noyaux. Le débit DDR réel dépend du L2, de la disposition,
du warp et des accès concurrents. Une passe à froid peut en plus inclure la préparation des images et les
écritures des sommes réemployées. Une mesure des octets L2/DDR permettrait de chiffrer le coût résiduel de chaque
frontière et le gain d'une fusion supplémentaire.

### Gains à vérifier

- Pour RIFE, le gain Tensor brut de Winograd 2D atteint 0,242 ms pour 4.25 et 0,254 ms pour Lite avec un
  encodage, à 160,4 TFLOP/s. L'essai F(2×2,3×3) du ResConv D2 améliore l'erreur mais augmente la durée
  de 25,5–25,8 à 33,8–34,0 µs. Il faut gagner sur les transformations ou sur leur trafic entre couches
  pour que la réduction du nombre de produits se traduise dans le lecteur. [Détail du banc](rife-natif.md).
- DistilDRBA applique déjà Winograd vertical sur huit résidus de la queue. Passer de cette queue verticale
  et de blocs grossiers directs à Winograd 2D sur tous les 3×3 éligibles ne retire que 0,041 à 0,044 ms de
  calcul Tensor au pic FP16. Une préparation de `X_2` lue directement par la première convolution et la
  conservation de ses sommes intermédiaires ont un potentiel du même ordre. [Voie actuelle](drba-fusionne.md).
