# TemporalFix en 1080p : plafond théorique de chaque couche sur la RTX 4070 Ti

Calculé dans le cloud le 19 novembre 2026, à partir des formes de s2 (`docs/tfix-natif.md`, « Le réseau, vu par
les noyaux » ; s1 et s3 ont les mêmes). Script :
[`tools/bench/tfix_native/plafonds.py`](../../../tools/bench/tfix_native/plafonds.py).

## Hypothèses

- Mêmes débits que les plafonds de GMFSS natif : 160 TFLOPS en sommes f16, 80 en f32, 321 en FP8 ;
  504 Go/s.
- Octets minimaux d'une couche : l'entrée lue une fois, la sortie écrite une fois, les poids, en f16. Activation
  et résiduel fondus dans la convolution.
- FLOP du vrai réseau : une ResConv séparable compte sa convolution par canal et sa 1 × 1 (pas la 3 × 3 repliée
  que lancent nos noyaux, qui en fait 8 fois plus, sans effet ici : ces couches sont limitées par les octets).
- La fenêtre : trois blocs et la tête, en lots de six appuis. La « colle » : X1, X2 et l'entrée de la tête
  écrites, les pyramides et les images lues (glue, conf, mélange).
- Plafond d'une couche : max(FLOP / débit, octets / bande passante).

## Résultat

| Morceau | Taille du réseau | GFLOP | Mo minimaux | Plafond f16 | Plafond f32 |
|---|---|---|---|---|---|
| Encodeur (une image) | 960×540 | 1,5 | 29 | 0,06 ms | 0,06 ms |
| Fenêtre : convolutions | 960×540 | 35,0 | 390 | 0,78 ms | 0,79 ms |
| Fenêtre : colle | 960×540 | | 152 | 0,30 ms | 0,30 ms |
| **Fenêtre, couche par couche** | 960×540 | | | **1,08 ms** | **1,09 ms** |
| Fenêtre tout fondue | 960×540 | 35,0 | 42 | 0,22 ms | 0,44 ms |
| Encodeur (une image) | 1920×1080 | 5,8 | 116 | 0,23 ms | 0,23 ms |
| Fenêtre : convolutions | 1920×1080 | 140 | 1 555 | 3,10 ms | 3,15 ms |
| Fenêtre : colle | 1920×1080 | | 608 | 1,21 ms | 1,21 ms |
| **Fenêtre, couche par couche** | 1920×1080 | | | **4,30 ms** | **4,36 ms** |
| Fenêtre tout fondue | 1920×1080 | 140 | 166 | 0,88 ms | 1,75 ms |

« Tout fondu » : les pyramides et les sept images lues une fois, la sortie écrite une fois, rien entre les deux ;
seul le calcul compte alors. Le FP8 n'apporte rien couche par couche : presque tout est limité par les octets
(seules les ResConv du bloc 0, à 96 canaux, le sont par le calcul, 3 µs chacune).

En demi-taille, la tête pèse les deux tiers du plafond des convolutions (0,18 + 0,20 + 0,10 ms) ; la première
convolution du bloc 2 (26 → 24, 240×135, 50 Mo) vient ensuite avec 0,10 ms ; les blocs 0 et 1 entiers font
0,10 ms.

## Contre le mesuré (demi-taille)

Deux mesures, qui ne se comparent pas entre elles : en lecture, les temps sont environ trois fois ceux de
l'essai au chargement pour TensorRT comme pour le natif (fenêtre TensorRT 7,74 ms en lecture, 2,14 ms par passe
à l'essai). La carte, peu chargée en lecture, ne tourne sans doute pas à ses fréquences hautes.

| | Plafond | Natif | TensorRT |
|---|---|---|---|
| Fenêtre, essai au chargement (une passe) | 1,08 ms | 3,32 ms (3,1×) | 2,14 ms (2,0×) |
| Fenêtre, lecture (profil du lecteur) | 1,08 ms | 10,55 ms | 7,85 ms |

Par noyau, en lecture (nsys `--cuda-graph-trace=node`, 0bd175e4, médianes) :

| Noyau | Natif | TensorRT | Plafond | Natif / plafond |
|---|---|---|---|---|
| conf (entrée de la tête) | 3,40 ms | 0,92 ms | env. 0,12 ms | ≈ 28× |
| single (tête 32 → 1) | 1,72 ms | (dans les xmma) | 0,10 ms | ≈ 17× |
| glue (X1 et X2), par niveau | 0,33 ms | 0,13 ms | env. 0,03 ms | ≈ 10× |
| tête 25 → 32 et 32 → 32, chacune | 0,33 ms | | 0,18 et 0,20 ms | ≈ 1,7× |
| convolutions, toutes | ≈ 1,8 ms par image | ≈ 3,4 ms | 0,78 ms | ≈ 2,3× |

## Ce qu'on en tire

- **Les convolutions natives sont déjà près de leur plafond** (environ 2× en lecture, la fréquence comprise) et
  deux fois plus rapides que TensorRT. Le réglage des tuiles par couche n'y gagnera que peu.
- **La colle et la dernière couche de la tête sont 10 à 28× leur plafond.** Par pixel, chaque fil écrivait ou
  lisait ses 25 à 32 canaux deux octets à la fois, au pas de 64 octets d'un point à l'autre. C'est là que va
  l'effort : écritures et lectures de 16 octets par point (fait, #964), et la dernière couche à quatre fils par
  point (lectures contiguës d'une chaîne).
- **Au-delà** : fondre la tête (conf, 25 → 32, 32 → 32, 32 → 1) par tuiles, pour que ses 32 canaux à 480×270 ×
  6 ne quittent pas la puce. C'est 240 Mo sur les 390 de la fenêtre en demi-taille ; la borne tout fondue
  (0,22 ms en f16) montre la marge.

## Suite (19 novembre, #964)

La colle en 16 octets et la tête fondue par tuiles sont sur main ([`docs/tfix-natif.md`](../../tfix-natif.md),
« La colle et la tête »). À l'essai, la fenêtre native fait 2,01 ms par passe contre 2,16 ms pour TensorRT
(56,2 dB), retenue ; en lecture, 7,33 ms en médiane contre 7,71 ms. Le profil de lecture après la tête fondue
(4 warps, [`kern-tete-fondue-4warps.csv`](kern-tete-fondue-4warps.csv)) donne `head_kernel` à 3,21 ms
(2,91 ms à 8 warps), `conf_kernel` à 1,02 ms, `glue_kernel` à 96 µs (335 avant). La tête reste loin de son
plafond (environ 0,26 ms à l'essai, halo compris) : ncu le dira, après la première release.

Profils de lecture bruts, nsys `cuda_gpu_kern_sum` sur 30 s en demi-taille, `--no-upscale --rife=0
--one-lane` : [`kern-natif-0bd175e4.csv`](kern-natif-0bd175e4.csv) (avant la colle en 16 octets),
[`kern-trt-0bd175e4.csv`](kern-trt-0bd175e4.csv) (TensorRT), [`kern-tete-fondue-4warps.csv`](kern-tete-fondue-4warps.csv).
