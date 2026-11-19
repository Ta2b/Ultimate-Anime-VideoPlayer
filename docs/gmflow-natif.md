# GMFlow natif : étude

Étude ; le squelette CNN ([PR 1](#squelette-natif-pr-1)), l'attention du transformeur ([PR 2](#attention-native-pr-2)) et son
MLP fondu ([PR 3](#mlp-natif-pr-3)) sont écrits, pas encore branchés. Question : calculer GMFlow (le flot du graphe « paire » de GMFSS Fortuna
union) par nos propres noyaux CUDA, comme RIFE ([`rife-natif.md`](rife-natif.md)) et, en cours, FeatureNet,
MetricNet et GridNet de GMFSS, pour qu'aucun moteur TensorRT ne soit plus construit pour GMFSS. Outils :
`tools/gmfss/gmflow_natif_couts.py` (coûts et temps estimés), `tools/gmfss/gmflow_natif_reseau.py` (GMFlow découpé
comme le natif, en PyTorch) et `tools/gmfss/gmflow_natif_emule.py` (émulation contre ONNX Runtime). Pas de GPU
pour cette étude : les temps du natif sont des estimations, les erreurs sont mesurées (émulation CPU).

## Ce qui tourne aujourd'hui

- Le graphe paire d'union (`gmfss_fortuna_union_pair.onnx`, ou `_s05_pair` au demi-flot) tourne sous TensorRT **en
  FP16**. Le FP8 de GMFSS a été retiré le 30/10 (mauvaise image dans les gros mouvements) et union est revenu
  sans lui ([`interpolation.md`](interpolation.md), « ni la base, ni le fusionné, ni le FP8 »). Même le graphe FP8
  d'alors laissait GMFlow en FP16 : `/flownet/backbone/` exclu, et les Q/DQ n'étaient placés que devant les
  convolutions, jamais devant les MatMul du transformeur ([`fp8-interpolation.md`](fp8-interpolation.md)). **Il
  n'existe donc pas de « GMFlow sous TensorRT FP8 » mesuré** : la référence de vitesse et de qualité est TensorRT
  FP16.
- Temps mesurés sur la RTX 4070 Ti, travail 720p (paire 1280x768) : paire **35 ms** au flot entier, **12 ms** au
  demi-flot (`gmfss_cost`, `chain_gmfss.cpp` ; 9,4 à 9,9 ms au trtexec pour le demi-flot). Construction de la paire :
  179 s. Aucun profil par couche ne sépare GMFlow de FeatureNet et MetricNet. Avec le rapport des coûts
  ci-dessous (flot entier ≈ 4,4 fois le demi-flot), on **déduit** GMFlow ≈ 29 à 30 ms au flot entier et ≈ 6 à 7 ms
  au demi-flot, le reste de la paire ≈ 5 à 6 ms. À mesurer (PR 0).

## Architecture de GMFlow dans ce graphe

Lue dans la roue (`vsgmfss_fortuna/gmflow/`, poids `flownet.pkl`, 4,72 M paramètres, partagés par base et union)
et dans l'export (`gmfss_graph.bidirectional_reuse` : un seul passage, `pred_bidir_flow=True`). GMFlow reçoit les
deux images réduites de moitié (640x384 pour la paire 1280x768 ; 320x192 au demi-flot), normalisées ImageNet.

| Étape | Détail | Taille (paire 1280x768, flot entier) |
|---|---|---|
| Squelette CNN (lot 2 : les deux images) | conv 7x7/2 3→64 (sans biais), InstanceNorm, Relu ; layer1 : 2 blocs résiduels 64 ; layer2 : 2 blocs 96, le premier de pas 2 avec raccourci 1x1/2 (biais) + InstanceNorm ; layer3 : 2 blocs 128, pas 1, raccourci 1x1 ; conv2 1x1 128→128 (biais) ; « trident » 3x3 128→128 aux poids partagés, pas 1 et pas 2 | 1/2 : 320x192 ; 1/4 : 160x96 ; 1/8 : 80x48. Bloc résiduel : conv 3x3, IN, Relu, conv 3x3, IN, Relu, + raccourci, Relu. InstanceNorm sans paramètres partout (15 : la 7x7, les 12 convolutions des blocs, les 2 raccourcis), 18 convolutions |
| Échelle 1/8 | position sinus par fenêtre (40x24), transformeur 6 blocs, puis appariement global, puis propagation globale | 3 840 jetons par image |
| Transformeur (mêmes poids aux deux échelles) | bloc = self-attention (sans MLP) puis cross-attention + MLP ; d = 128, **une tête** ; projections q, k, v, fusion 128x128 sans biais ; LayerNorm après la fusion ; MLP sur [source, message] 256 → 1024 (GELU exacte, par erf) → 128, sans biais, LayerNorm ; résiduel ; la cible de la cross-attention est l'autre moitié du lot telle qu'elle entrait dans le bloc ; blocs impairs en fenêtres décalées d'une demi-fenêtre (Swin, masque −100 entre régions) | 1/8 : lot 2, fenêtres 2x2 de 960 jetons, décalage (12, 20) |
| Appariement global | corrélation f0·f1ᵀ/√128 (3 840 x 3 840), softmax des lignes (0→1) et des colonnes (1→0), flot = Σ p·coordonnée − coordonnée | |
| Propagation globale | self-attention 3 840 x 3 840 sur les caractéristiques, q = Wq·f + bq, **k = Wk·q + bk** (projections enchaînées, une seule matrice en fait), v = le flot (2 canaux) ; sur les deux cartes | |
| Échelle 1/4 | les deux sens en lot (f0 = [f0, f1], f1 = [f1, f0]), flot ×2 (bilinéaire, coins alignés), f1 déformé par le flot, position (fenêtre 20x12), transformeur 6 blocs | 15 360 jetons par carte, lot 4 dans le transformeur, fenêtres 8x8 de 240 jetons, décalage (6, 10) |
| Appariement local | rayon 4 : 81 produits par pixel, hors image masqués, softmax, flot résiduel ajouté | |
| Propagation locale | fenêtre 3x3, q = Wq·f + bq, k = Wk·f + bk ; les voisins hors image valent 0 (score 0, flot 0), non masqués | |
| Agrandissement convexe ×4 | conv 3x3 [flot, f0] 130 → 256 (biais), Relu, conv 1x1 256 → 144 (biais), softmax sur 9 voisins pour chacun des 16 sous-pixels, somme pondérée de 4·flot | sortie 640x384, 2 canaux, deux sens |

Au demi-flot (`_s05`), tout est deux fois plus petit en largeur et en hauteur : 960 jetons à 1/8 (fenêtres de 240),
3 840 à 1/4 (fenêtres de 60). À 1920x1152 (travail 1080p), 8 640 jetons à 1/8 et des fenêtres de 2 160.

## Coûts par opération

`python3 tools/gmfss/gmflow_natif_couts.py [--width 1280 --height 768] [--scale 1|0.5]`. FLOP : 2 par
multiplication-addition. « Mo natif » : ce que lit et écrit le noyau fondu prévu ; « Mo graphe » : le graphe ONNX
non fondu (matrices de scores, couche cachée du MLP, fenêtrages et décalages en copies). Les tailles sont celles de
la paire remplie (1280x768 pour 720p de travail ; 1280x720 n'existe pas dans la chaîne, la paire est remplie à un
multiple de 128).

Flot entier (GMFlow sur 640x384) :

| Opération | × | GFLOP | Mo natif | Mo graphe |
|---|---|---|---|---|
| conv1 7x7/2 3→64 | 1 | 2,3 | 19 | 19 |
| layer1 : conv 3x3 64→64 (320x192, lot 2) | 4 | 36,2 | 126 | 126 |
| layer2 : conv 3x3/2 64→96, raccourci 1x1/2, 3 conv 3x3 96→96 | 5 | 19,1 | 79 | 79 |
| layer3 : conv 3x3 96→128, raccourci 1x1, 3 conv 3x3 128→128 | 5 | 34,7 | 75 | 75 |
| conv2 1x1, trident /1 et /2 | 3 | 12,3 | 41 | 41 |
| fins de bloc, InstanceNorm | — | 0 | 236 | 472 |
| 1/8, transformeur : QKV (GEMM 128→384), self et cross | 12 | 9,1 | 94 | 94 |
| 1/8, attention fenêtrée (960 jetons) + fusion + LayerNorm | 12 | 48,3 | 106 | 967 |
| 1/8, MLP 256→1024→128 + LayerNorm | 6 | 36,2 | 35 | 260 |
| 1/8, appariement global (deux sens) | 1 | 7,6 | 4 | 356 |
| 1/8, propagation globale (q, k, softmax, deux cartes) | 2 | 8,1 | 10 | 364 |
| 1/4, transformeur : QKV | 12 | 72,5 | 755 | 755 |
| 1/4, attention fenêtrée (240 jetons) + fusion + LayerNorm | 12 | 114,8 | 849 | 3 492 |
| 1/4, MLP 256→1024→128 + LayerNorm | 6 | **289,9** | 283 | 2 076 |
| 1/4, déformation, appariement local r = 4, propagation 3x3 | 3 | 2,7 | 55 | 226 |
| 1/4, agrandissement (conv 3x3 130→256, 1x1 256→144, convexe) | 2 | 20,7 | 44 | 85 |
| **Total** | | **714** | **2 811** | **9 487** |

Demi-flot : **150 GFLOP**, 703 Mo natif (1 841 Mo graphe). Travail 1080p (paire 1920x1152) : 1 990 GFLOP au flot
entier, 365 au demi-flot. Les deux tiers du calcul sont le transformeur à 1/4, et son MLP à lui seul en fait 40 % :
c'est là que se joue la vitesse.

## Plan natif

Format des activations : NHWC FP16 (celui de `rife_native_kernels.cuh`), le flot en f32 (2 canaux, quelques Mo :
le FP16 l'arrondirait à 1/16 de pixel au-delà de 64 px, puis ×4 et ×8 à l'agrandissement). Environ 95 lancements
par paire, capturés dans un graphe CUDA comme les autres moteurs natifs.

**Squelette** : la famille de `rife_native_kernels.cuh` (mma.sync m16n8k16, tuile et halo en mémoire partagée,
`cp.async`, K par tranches, `ksplit`). À ajouter :

- la première couche 7x7/2 sur trois plans : en `In::kImage` elle compterait 16 canaux par prise (K = 784 pour
  147 utiles, 5,3 fois trop) ; un im2col en mémoire partagée à K = 147 → 160 ;
- les 1x1 (raccourcis de pas 2, conv2) : la même famille en KH = 1, ou une GEMM ;
- **InstanceNorm** : l'épilogue de chaque convolution ajoute Σx et Σx² de ses pixels par (image, canal) en f32
  (réduction dans le bloc, puis une addition atomique par bloc, ou une réduction en deux temps pour rester
  déterministe) ; le noyau suivant normalise et applique Relu au chargement de sa tuile, comme `prelu_tile`
  ajouté pour le GridNet (commit 261814d) ;
- une fin de bloc élément par élément (deux normalisations, somme, Relu, une écriture FP16) ;
- la position sinus ajoutée dans l'épilogue du trident de pas 2 au huitième (table FP16 de 40x24 x 128) ; au
  quart, elle s'ajoute après la déformation de la cible par le flot, dans le noyau de déformation (voir
  « Attention native (PR 2) »).

**Transformeur** : trois noyaux par couche, 60 en tout.

1. **QKV** : une GEMM 128 → 384 par couche (les trois projections d'un coup ; en cross-attention, k et v se lisent
   dans l'autre moitié du lot par l'indice, sans copie de `concat1`).
2. **Attention fenêtrée fondue**, façon FlashAttention-2 : un bloc de 4 warps par (fenêtre, 64 requêtes) ; K et V
   de la fenêtre par tuiles de 64 clés en double tampon (`cp.async`) ; S = QKᵀ en mma FP16 à sommes f32 ; maximum
   et somme courants en registres ; P arrondi en FP16 pour le mma P·V ; le décalage Swin par arithmétique
   d'adresses (modulo, aucun `roll` matérialisé) et le masque par numéros de région calculés à la volée. Dans
   l'épilogue : O/l, puis la **fusion 128x128** (poids 32 Ko en mémoire partagée), la LayerNorm (la ligne entière
   est dans le bloc) et, en self-attention, le résiduel. Ni les scores, ni la sortie de l'attention ne passent par
   la mémoire.
3. **MLP fondu** (cross-attention seulement ; écrit, [plus bas](#mlp-natif-pr-3)) : un bloc de 64 ou 128 jetons
   charge [x, message] (256 canaux) ; pour chacune des 16 tranches de 64 de la couche cachée, H = GELU(X·W1)
   reste sur la puce (FP16), puis O += H·W2 ; à la fin LayerNorm et résiduel. La couche cachée (126 Mo par couche
   à 1/4) n'est jamais écrite. W1 et W2 (768 Ko en FP16) se lisent dans la L2 une fois par bloc, par un anneau en
   mémoire partagée.

Les GEMM de 1 et 3 peuvent venir de CUTLASS (`third_party/cutlass`, déjà là pour SPAN FP8) ou de mma.sync écrits à
la main ; le MLP fondu n'existe pas tel quel dans CUTLASS (deux GEMM chaînées en mémoire partagée) : à la main.

**Appariement global et propagation globale** : le même noyau à softmax fondu, deux passages pour l'appariement
(lignes puis colonnes : S est recalculé, 3,8 GFLOP de plus, au lieu de stocker 3 840² scores). V n'a que deux
colonnes : les coordonnées (calculées depuis l'indice de la clé, rien à lire) ou le flot (f32) ; P·V en f32 sur
les cœurs CUDA, P exact (59 M multiplications-additions, négligeable). Pour la propagation, une GEMM 128 → 256
donne q et k d'un coup (k = Wk·q + bk se réécrit k = (Wk·Wq)·f + (Wk·bq + bk), matrices fondues au chargement).

**Fin à 1/4** : déformation bilinéaire de f1 (+ position) ; appariement local r = 4 (tuile + halo de 4, produits
en half2 ou en mma sur une bande ; 0,6 GFLOP) ; propagation 3x3 (GEMM q, k puis pochoir) ; agrandissement : conv
3x3 130 (→ 144) → 256 par la famille de RIFE, puis la 1x1 256 → 144 avec, dans son épilogue, le softmax des 9
poids et la somme convexe : le flot ×4 sort directement, les 144 canaux ne sont jamais écrits.

## Temps estimés sur la RTX 4070 Ti

Modèle (`gmflow_natif_couts.py`) : pics des tensor cores denses de la 4070 Ti (60 SM, 2,61 GHz) : FP16 80 TFLOPS
à sommes f32, 160 à sommes f16 ; E4M3 160 à sommes f32, 320 à sommes f16 ; 504 Go/s. Rendements supposés :
convolutions 50 %, GEMM 70 % (SPAN FP8 en CUTLASS tient 76 à 83 %), attention fondue 40 % ; mémoire à 75 % de la
bande ; 4 µs par lancement. Temps d'un noyau = max(calcul, mémoire) + lancement.

| Bloc (paire 1280x768) | FP16, sommes f32 | FP16, sommes f16 | GEMM du transformeur en E4M3 |
|---|---|---|---|
| Squelette | 3,4 ms | 2,2 ms | 3,4 ms |
| 1/8 : transformeur | 2,5 ms | 2,2 ms | 2,1 ms |
| 1/8 : appariement et propagation globaux | 0,5 ms | 0,5 ms | 0,5 ms |
| 1/4 : transformeur | 10,9 ms | 8,3 ms | 7,8 ms |
| 1/4 : appariement local, propagation, agrandissement | 0,7 ms | 0,5 ms | 0,7 ms |
| **GMFlow, flot entier** | **18,0 ms** | **13,6 ms** | **14,5 ms** |
| **GMFlow, demi-flot** | **4,1 ms** | **3,0 ms** | **3,2 ms** |
| TensorRT FP16 (déduit, voir plus haut) | ≈ 29 à 30 ms / ≈ 6 à 7 ms | | |

La colonne E4M3 n'est là que pour mémoire : l'émulation l'écarte pour la qualité (plus bas). Lecture : sur le
papier, le natif FP16 à sommes f32 ferait GMFlow en 0,6 fois le temps déduit de TensorRT ; la
paire entière passerait de 35 à ≈ 23 ms au flot entier et de 12 à ≈ 9,5 ms au demi-flot (FeatureNet et MetricNet
inchangés). Mais **ces chiffres sont fragiles** : le rendement de l'attention fondue à une tête de 128 sur sm_89
(pression sur les registres : O fait 64 flottants par fil) n'est pas mesuré, TensorRT fond peut-être déjà
l'attention (Myelin reconnaît le motif MatMul-Softmax-MatMul), et le temps de GMFlow sous TensorRT est déduit, pas
mesuré. À 1080p de travail, le flot entier reste hors de portée (≈ 53 ms natif FP16), le demi-flot ≈ 9 ms.

## Émulation

`python -I tools/gmfss/gmflow_natif_emule.py ROUE/vsgmfss_fortuna --work DOSSIER [--width W --height H] [--scale
1|0.5] [--images] [--recettes …]`.

- **Référence** : le graphe paire union exporté en FP32 pur, exécuté par ONNX Runtime (CPU). **Actuel** : le même
  graphe en FP16 tel que le lecteur le donne à TensorRT, exécuté par ONNX Runtime (ce n'est pas TensorRT : ordre
  des sommes et choix de noyaux différents, mais mêmes arrondis de stockage).
- **Émulation** : `gmflow_natif_reseau.py` refait GMFlow noyau par noyau depuis `flownet.pkl` : stockage FP16 entre
  les noyaux, statistiques d'InstanceNorm et LayerNorm en f32, attention, appariement global et propagation globale
  par tuiles de 64 clés avec softmax en ligne (jamais la matrice entière), P arrondi en FP16 avant P·V, flot en
  f32. Sommes f16 émulées par paquets de 16 (32 en FP8) arrondis après chaque mma ; E4M3 par `float8_e4m3fn`
  saturé à 448, échelle d'activation par tenseur calibrée (maximum × 1,25 / 448 sur trois paires hors test,
  comme SPAN) ou par jeton, poids par canal de sortie.
- **Paires** : six images réelles de `data/calibration/calibration.mkv` (Re:Zero, Frieren, Black Clover), la
  seconde image obtenue par un mouvement connu : translation (6, −3), panoramique de 60 px, zoom 1,03 + translation,
  objet découpé qui traverse le décor, rotation de 4° + zoom, mouvement fort (40, −25, 2°) + objet. Mouvements de
  caméra synthétiques : pas de vraies paires d'anime avec du mouvement dessiné (limite, voir les risques).
- **Mesures**, sur les deux flots de sortie (motion0_1 et motion1_1, à la demi-taille de la paire) contre la
  référence FP32 : EPE (erreur de point final, px), 99,9e centile et pire pixel ; PSNR du flot = 10·log10(crête² /
  EQM), crête = plus grand |flot| de la référence. Avec `--images` : l'image du milieu (t = 0,5) de toute la suite
  de GMFSS en FP32 PyTorch, seul le flot changeant, contre celle du flot FP32 (PSNR de l'image, l'unité des gardes
  de [`fp8-interpolation.md`](fp8-interpolation.md) : 42 dB en moyenne, 38 au pire).

### Résultats, paire 640x384 au flot entier (GMFlow sur 320x192, la charge du demi-flot en 720p)

| Recette | EPE moyen | p99,9 pire | PSNR flot moyen / pire | PSNR image moyen / pire |
|---|---|---|---|---|
| **actuel** (graphe FP16, ONNX Runtime) | 0,0200 | 0,364 | 58,5 / 46,8 | 60,0 / 54,9 |
| natif émulé en FP32 (contrôle du découpage) | 0,0000 | 0,000 | 120,6 / 109,0 | 112,9 / 76,9 |
| **natif FP16, sommes f32** | **0,0057** | **0,149** | **68,9 / 54,7** | **69,6 / 61,7** |
| natif FP16, sommes f16 (convolutions et GEMM) | 0,0092 | 0,234 | 64,1 / 49,6 | 64,9 / 58,3 |
| natif FP16, P·V du flot en FP16 (tensor cores) | 0,0059 | 0,130 | 68,7 / 55,3 | 70,0 / 61,3 |
| GEMM du transformeur en E4M3, échelles par tenseur | 0,2203 | 5,343 | 37,5 / 24,5 | 43,1 / **35,1** |
| idem, sommes f16 (marge de SPAN) | NaN : débordements | | | 25,5 / 23,1 |
| tout en E4M3 (squelette et agrandissement compris) | 0,3014 | 6,794 | 34,2 / 21,2 | 41,2 / 33,5 |

Ablation du FP8 (une seule famille de GEMM en E4M3, le reste en FP16 natif ; mêmes paires) :

| GEMM en E4M3 | EPE moyen | PSNR flot moyen / pire | PSNR image moyen / pire |
|---|---|---|---|
| QKV seules | 0,103 | 43,4 / 29,1 | 47,7 / 40,6 |
| fusion seule | 0,085 | 45,2 / 32,9 | 49,9 / 40,8 |
| MLP, première couche seule | 0,149 | 40,9 / 27,7 | 45,2 / 35,6 |
| MLP, seconde couche seule | 0,097 | 43,7 / 30,6 | 47,2 / 39,8 |
| toutes, échelle par jeton (et par tranche de 128 de K) | 0,209 | 37,8 / 25,3 | 43,5 / 35,8 |
| MLP seul, échelle par jeton | 0,160 | 40,3 / 28,5 | 45,4 / 36,8 |

- **Le découpage est exact** : en FP32, l'émulation rend le graphe FP32 à 1e-4 px près (EPE moyen < 1e-5 ; vérifié
  aussi contre le GMFlow PyTorch d'origine : 7e-6 px). Fenêtres décalées, masque, cible de la cross-attention,
  projections enchaînées de la propagation, zéros non masqués de la fenêtre 3x3 : tout est à sa place.
- **Le natif FP16 est plus juste que le graphe FP16 actuel** : EPE 3,5 fois plus petit, image +9,6 dB en moyenne et
  +6,8 dB au pire. La raison principale : le flot reste en f32 (le graphe le stocke en FP16 entre l'appariement, la
  propagation et l'agrandissement) et le softmax en ligne garde S et la somme en f32.
- **Les sommes f16** coûtent ≈ 5 dB sur l'image mais restent au-dessus de l'actuel (64,9 / 58,3 contre 60,0 / 54,9).
- **E4M3 ne passe pas** les gardes (35,1 dB au pire sur l'image), et ce n'est pas une affaire d'échelle : chaque
  famille de GEMM seule en E4M3 coûte déjà 10 à 25 dB sur l'image, et l'échelle par jeton ne fait pas mieux que
  l'échelle par tenseur. C'est la mantisse de 3 bits (≈ 6 % d'erreur relative par valeur) qui ne passe pas douze
  couches de transformeur suivies d'un softmax global très pointu : une petite erreur sur les caractéristiques
  déplace le maximum de la corrélation. Même conclusion que la recherche par groupes de l'ancien FP8 TensorRT
  (squelette de GMFlow en FP8 : −12 dB). À sommes f16, la marge de SPAN (poids à 448/16) déborde presque partout.
  **GMFlow reste en FP16** ; le FP8 n'achète rien ici, ni en natif ni sous TensorRT.

### Résultats à la taille de la lecture (paire 1280x768)

Demi-flot (`_s05`, GMFlow sur 320x192, ce que le lecteur joue par défaut en 720p), six paires, image du milieu
comprise (l'image de l'actuel n'est pas calculée au demi-flot : le graphe ne sort que le flot déjà agrandi) :

| Recette | EPE moyen | p99,9 pire | PSNR flot moyen / pire | PSNR image moyen / pire |
|---|---|---|---|---|
| **actuel** (graphe FP16, ONNX Runtime) | 0,0423 | 1,020 | 55,6 / 47,4 | — |
| natif émulé en FP32 | 0,0000 | 0,000 | 120,0 / 114,9 | 109,8 / 98,8 |
| **natif FP16, sommes f32** | **0,0094** | **0,160** | **68,0 / 63,1** | **71,9 / 62,6** |
| natif FP16, sommes f16 | 0,0152 | 0,326 | 63,5 / 57,7 | 67,7 / 58,1 |
| MLP en E4M3, échelle par jeton | 0,2427 | 5,749 | 41,0 / 34,2 | 47,9 / 37,5 |

Flot entier (GMFlow sur 640x384), flot seulement (l'image du milieu à cette taille est trop longue sur ce poste) :

| Recette | EPE moyen | p99,9 pire | PSNR flot moyen / pire |
|---|---|---|---|
| **actuel** (graphe FP16, ONNX Runtime) | 0,0207 | 0,211 | 60,2 / 45,0 |
| natif émulé en FP32 | 0,0000 | 0,000 | 120,8 / 107,0 |
| **natif FP16, sommes f32** | **0,0057** | **0,109** | **71,9 / 57,7** |
| MLP en E4M3, échelle par jeton | 0,1615 | 1,173 | 43,3 / 27,4 |

Au demi-flot sur la petite paire 640x384 (GMFlow sur 160x96, plus petit qu'aucun cas de la lecture), le FP32 lui-même
trouve des appariements ambigus (pics de 52 px sur le zoom) ; le natif FP16 et l'actuel s'y trompent autant (EPE
0,20 et 0,21, PSNR image du natif 47,1 / 34,2) : cette taille ne départage rien, elle n'est pas retenue.

## Squelette natif (PR 1)

Écrit sans GPU : compilé pour sm_75, 86, 89 et 120 sans débordement de registres, jamais lancé. Rien ne l'appelle encore.

**Lecture** (`src/prepare/gmflow_native.{h,cpp}`, types sans `std::expected` dans `gmflow_backbone.h` pour nvcc).
`gmflow_backbone(paire)` lit les poids du squelette par leur nom dans la paire ONNX (`gmfss.flownet.backbone.*`,
nœuds `/flownet/backbone/…`) et refuse (FieldMissing) ce que les noyaux ne font pas : noyau, pas ou bord autres
que ceux attendus, groupes, dilatation, InstanceNorm à échelle ou décalage appris, ε différents d'une couche à
l'autre, biais en trop ou en moins, raccourci absent ou en trop, Add dans l'autre ordre, normalisations
différentes des deux images, trident qui ne partage pas ses poids. Elle rend un **plan** : 17 convolutions (le
trident une fois), 25 tenseurs, 24 opérations (18 convolutions, 6 fins de bloc), 15 statistiques ;
`gmflow_backbone_check` le revérifie contre ce que savent les noyaux (canaux d'entrée par 32, de sortie par 16
jusqu'à 128, normalisation à la lecture des seules 3x3 de pas 1, chaque tenseur écrit une fois avant d'être lu).
Le réseau CUDA et l'émulation exécutent ce plan à la lettre : l'indexation prouvée par l'émulation est celle du
lecteur.

**Normalisation de l'image** rangée dans la 7x7 : poids × 1/écart de leur plan, biais −Σ poids · pad, où pad est
la moyenne d'ImageNet arrondie en FP16 (dans la paire FP16, elle est déjà en FP16 : pad est exact). L'image FP16
entre telle quelle et un pixel hors de l'image vaut pad, soit zéro une fois normalisé, comme le bord du graphe.
Arrondir l'image normalisée en FP16 coûtait 5 dB sur les sorties (émulation).

**Noyaux** (`src/gpu/cuda/backend/gmflow_backbone_kernels.cuh`, `gmflow_backbone_net.{h,cu}`) : NHWC FP16, les deux
images en lot, mma.sync m16n8k16 et sommes f32 (`Acc::kF32`) de `rife_native_kernels.cuh` (fragments, `accumulate`,
`stage`, `cp.async`). La 7x7/2 déplie un morceau d'image de 13 × 69 pixels en mémoire partagée (K = 160 pour 147
produits) ; les 1x1 passent par le noyau des 3x3 en KH = 1, celle de pas 2 lit un pixel sur deux. L'épilogue de chaque
convolution suivie d'une InstanceNorm ajoute Σx et Σx² (f32, valeurs FP16 écrites) par image et par canal, une
addition atomique par bloc et par canal ; la 3x3 suivante normalise et applique Relu à sa tuile au chargement
(`In::kNorm`). Les raccourcis et la sortie de chaque bloc se lisent dans un noyau de fin de bloc élément par
élément (une écriture FP16). Les tenseurs sont placés dans une zone selon leur durée de vie ; les statistiques
sont remises à zéro au début du passage (capturable en graphe CUDA).

**Émulation** (`tools/gmfss/gmflow_squelette_emule.py`, plan et poids vidés par `tools/gmfss/gmflow_squelette`,
qui appelle la lecture du lecteur) : le plan rejoué avec les arrondis des noyaux, contre le sous-graphe du
squelette de la paire FP32 sous ONNX Runtime, sur 4 paires de `calibration.mkv`. Moyenne des quatre ; PSNR sur la
crête de la référence, moyen / pire.

| Images | Sortie | Contrôle f32 (plan de la paire FP32) | Natif FP16 | ONNX Runtime sur la paire FP16 | Stockage FP16 de chaque couche (façon TensorRT, `gmflow_natif_reseau.py`) |
|---|---|---|---|---|---|
| 320x192 | 1/4 | 1,13e-6, 133,3 dB | 9,9e-4, 74,5 / 74,0 dB | 6,3e-4, 78,3 / 77,4 dB | 2,1e-3, 68,7 / 64,2 dB |
| 320x192 | 1/8 | 1,13e-6, 132,7 dB | 9,8e-4, 73,9 / 73,6 dB | 6,3e-4, 77,8 / 77,1 dB | 2,1e-3, 68,1 / 64,1 dB |
| 640x384 | 1/4 | 1,52e-6, 131,0 dB | 1,12e-3, 73,7 / 72,1 dB | 6,9e-4, 77,8 / 76,9 dB | 2,9e-3, 66,3 / 61,6 dB |
| 640x384 | 1/8 | 1,52e-6, 130,8 dB | 1,12e-3, 73,4 / 71,8 dB | 6,9e-4, 77,5 / 76,6 dB | 2,9e-3, 66,0 / 61,6 dB |

(écart relatif : √EQM / √(moyenne des carrés de la référence)). Le contrôle f32, sans aucun arrondi, rend ONNX
Runtime FP32 à l'ordre des sommes près (avec au bord la moyenne FP32 exacte, `pad_exact`) : poids rangés, prises,
dépliage de la 7x7, normalisation rangée, statistiques, raccourcis et trident sont justes. Le natif FP16 est
≈ 4 dB sous ONNX Runtime sur la paire FP16, qui n'est pas TensorRT : sur le CPU il garde des poids FP16 mais
calcule et stocke les activations en f32. Face à un stockage FP16 de chaque couche, plus proche de ce que fait
TensorRT FP16, le natif gagne 6 à 7 dB (10 dB au pire cas). La comparaison qui compte, contre le moteur TensorRT
de la paire sur la carte, reste à faire.

**Ce qui manque avant de brancher** : la position sinus dans l'épilogue du trident (PR 2 ou à l'assemblage), le
choix des tuiles et un `ksplit` pour les petites grilles (1/8 au demi-flot), les variantes de sommes f16 et
mixtes chronométrées contre TensorRT comme RIFE et GMFSS (`gmfss_trial.h`), le premier lancement sur la carte
(sorties contre le sous-graphe ONNX, déterminisme : les additions atomiques changent les statistiques à l'ulp
près d'un lancement à l'autre). Le 1/8 est exactement le 1/4 pris un pixel sur deux (même trident, bord 1) :
la seconde passe du trident pourrait se réduire à cette lecture.

## Attention native (PR 2)

Écrite sans GPU : compilée pour sm_75, 86, 89 et 120 sans débordement de registres, jamais lancée. Rien ne
l'appelle encore ; le MLP fondu (PR 3) et le reste de GMFlow restent dans la paire TensorRT.

**Ce que dit le graphe** (paire union exportée, initialiseurs `gmfss.flownet.transformer.*`, nœuds
`/flownet/transformer/…`), là où l'étude restait vague :

- les nœuds des couches sont les mêmes aux deux échelles, suffixés `_1` au quart (`layers.<b>/self_attn/q_proj_1/
  MatMul`), et lisent les mêmes poids ; ceux de la géométrie sont sous `/flownet/transformer/` (huitième) et
  `/flownet/transformer_1/` (quart) ;
- fenêtres : `Div` de la taille par 2 (huitième) et par 8 (quart) ; le décalage est d'une demi-fenêtre arrondie en
  bas (`Div_2`, `Div_3`), soit (12, 20) et (6, 10) à 640x384 ; torch.roll s'écrit en deux `Slice` et un `Concat`
  (vers la gauche avant l'attention, vers la droite après) ;
- masque : régions de l'image décalée, trois par axe, comptées au bord moins une fenêtre et au bord moins un
  décalage, région = ligne · 3 + colonne ; −100 entre deux régions différentes, 0 dans une même région, ajouté aux
  scores déjà divisés par √d (`Div` par `Pow(Shape(q)[2], 0.5)`) ;
- la paire FP16 divise par √128 arrondi en FP16 (11,3125 au lieu de 11,3137) ; le natif prend 1/√128 exact, comme la
  paire FP32 ;
- self-attention : sortie = source + LayerNorm(fusion(attention)) ; cross-attention : q à la sortie de la self, k et
  v à l'entrée du bloc dans l'autre moitié du lot (au premier bloc `Concat(f1, f0)`, ensuite deux `Slice` par la
  moitié du lot lue dans la forme), sortie = LayerNorm seule, le message que le MLP lit avec la source ;
- **la position sinus** est calculée dans le graphe à partir des formes (sin et cos d'une somme cumulée
  normalisée), pas lue dans une constante. Au huitième elle s'ajoute aux deux sorties du trident de pas 2 ; au
  quart, elle s'ajoute **après** la déformation de la cible par le flot (`feature_add_position` suit `flow_warp`),
  et la source sans position sert à la déformation : elle ne peut pas aller dans l'épilogue du trident de pas 1.
  Elle n'est pas dans cette PR : la table (calculée au chargement, vérifiée contre la position du graphe sous ONNX
  Runtime) ira à l'assemblage, dans le trident de pas 2 au huitième et dans le noyau de déformation au quart
  (PR 5).

**Lecture** (`src/prepare/gmflow_attention_native.{h,cpp}`, plan revérifié par `gmflow_attention_check.cpp`, types
sans `std::expected` dans `gmflow_attention.h` pour nvcc ; parcours du graphe dans `gmflow_onnx_view.{h,cpp}`).
`gmflow_attention(paire)` lit, aux deux échelles, les projections q, k, v et la fusion par le nom de leur `MatMul`
(poids 128x128 constants, rangés [sortie][entrée]), la LayerNorm `norm1` par ses initialiseurs, et vérifie tout ce
qui les relie : fenêtres de chaque `Reshape` à six axes et `Transpose` (0, 1, 3, 2, 4, 5), chemins de q, k, v jusqu'au
`MatMul` des scores (une découpe en fenêtres, kᵀ, décalés sur les deux axes si la couche l'est, rien d'autre),
`Softmax` sur le dernier axe, facteur, masque, chemin retour de la sortie, ε commun, résiduel `Add(source, norm1)`,
`Concat(source, message)` du MLP, cible de la cross et sortie `Add(source, norm2)` de chaque bloc. Le quart doit
avoir les mêmes poids et les mêmes décalages que le huitième. Tout écart est refusé (FieldMissing, avec le nœud).
Elle rend 12 couches (self du bloc b en 2b, cross en 2b + 1, blocs impairs décalés) et un **plan** de 18 opérations
sur 19 tenseurs (self, cross, MLP par bloc) que le banc, l'émulation et plus tard l'assemblage exécutent à la
lettre. Elle lit la paire FP16 du lecteur, celle du demi-flot et la paire FP32.

**Noyaux** (`src/gpu/cuda/backend/gmflow_attention_kernels.cuh`, `gmflow_attention_net.{h,cu}`), jetons
[lot][h · w][128] FP16, deux lancements par couche :

1. `qkv_kernel` : la GEMM 128 → 384, un bloc de 4 warps par (64 jetons, q | k | v, image) ; en cross-attention k
   et v se lisent au lot (b + lot/2) mod lot, sans copie. Sortie [lot][jetons][384] FP16 (5,9 Mo au huitième,
   47 Mo au quart à 640x384, un seul tampon réutilisé d'une couche à l'autre).
2. `attention_kernel` : un bloc de 4 warps par (64 requêtes, fenêtre, image), 16 requêtes par warp. K et V passent
   par tuiles de 64 clés, chacun dans son tampon (`cp.async` : V de la tuile se charge pendant S = QKᵀ, K de la
   suivante pendant P · V ; 35 Ko de mémoire partagée). S en mma FP16 à sommes f32, facteur et masque en base 2
   (`exp2`), maximum et somme courants en registres, P arrondi en FP16 directement dans les fragments A de P · V.
   Le décalage est une arithmétique d'adresses (jeton t de la fenêtre (wy, wx) en ((wy · wh + t / ww + sh) mod H,
   (wx · ww + t mod ww + sw) mod W)), la sortie s'écrit à la place de sa requête, les régions du masque se
   calculent au chargement des clés. Épilogue sans mémoire : O / l arrondi en FP16 devient le fragment A de la
   fusion (poids rangés pour le mma, lus en L1 : la mémoire partagée garde K et V), LayerNorm sur la ligne tenue
   par les quatre fils d'un groupe (moyenne, puis variance des écarts, en f32), résiduel en self, une écriture FP16.

Sommes : S et P · V toujours en f32 ; la GEMM QKV et la fusion suivent `Acc` (`kF16`, `kMixed`, `kF32`) comme les
noyaux de RIFE natif, et le réseau ne lance que `kF32`, le plus sûr, tant que le banc n'a pas parlé. Registres :
attention 168 (181 sur sm_75), QKV 128 (182 sur sm_75), aucun débordement. L'attention tiendrait 3 blocs par SM
par ses registres, mais 2 sur sm_86 et sm_89 par sa mémoire partagée (3 × 35 Ko dépassent les 100 Ko d'un SM).

**Émulation** (`tools/gmfss/gmflow_attention_emule.py`, plan et poids vidés par `tools/gmfss/gmflow_attention`, qui
appelle la lecture du lecteur) : chaque couche rejouée avec les arrondis des noyaux (q, k, v en FP16, tuiles de 64
clés, softmax en ligne en base 2, P et O / l en FP16, fusion et LayerNorm en f32, une écriture FP16), à partir de
son entrée FP32 arrondie en FP16, contre sa sortie dans la paire FP32 sous ONNX Runtime ; 4 paires de
`calibration.mkv` (15, 39, 69 et 99 s, deux images à 2/24 s), paire 1280x768, GMFlow sur 640x384 : fenêtres de 960
jetons au huitième (lot 2), de 240 au quart (lot 4). Couches des blocs 0 (sans décalage), 1 et 5 (décalés).
Pire écart, écart relatif moyen (√EQM / √moyenne des carrés de la référence), PSNR sur la crête de la référence
moyen / pire.

| Échelle | Couche | Contrôle f32 | Natif FP16 | FP16 nœud par nœud |
|---|---|---|---|---|
| 1/8 | 0 self | 1,4e-6, 152,3 dB | 3,2e-3, 4,1e-4, 84,1 / 83,7 dB | 4,3e-3, 5,3e-4, 81,8 / 81,4 dB |
| 1/8 | 1 cross | 1,4e-6, 153,9 dB | 2,1e-3, 5,1e-4, 82,5 / 82,3 dB | 1,1e-2, 6,2e-4, 80,7 / 80,6 dB |
| 1/8 | 2 self décalée | 1,2e-6, 151,5 dB | 2,1e-3, 3,7e-4, 82,0 / 81,7 dB | 3,6e-3, 4,7e-4, 80,0 / 79,7 dB |
| 1/8 | 3 cross décalée | 1,9e-6, 154,0 dB | 2,0e-3, 3,7e-4, 87,1 / 86,9 dB | 2,9e-3, 4,8e-4, 84,9 / 84,6 dB |
| 1/8 | 10 self décalée | 1,9e-6, 150,7 dB | 3,6e-3, 4,0e-4, 82,3 / 81,4 dB | 5,8e-3, 5,0e-4, 80,4 / 79,5 dB |
| 1/8 | 11 cross décalée | 1,7e-6, 152,2 dB | 2,2e-3, 4,6e-4, 84,9 / 84,6 dB | 3,6e-3, 7,3e-4, 80,8 / 80,4 dB |
| 1/4 | 0 self | 2,1e-6, 150,7 dB | 3,3e-3, 4,4e-4, 83,5 / 83,2 dB | 4,7e-3, 5,7e-4, 81,3 / 81,2 dB |
| 1/4 | 1 cross | 1,4e-6, 150,5 dB | 2,8e-3, 5,3e-4, 82,4 / 82,1 dB | 1,3e-2, 8,5e-4, 78,4 / 77,5 dB |
| 1/4 | 2 self décalée | 1,7e-6, 151,2 dB | 2,2e-3, 3,8e-4, 82,3 / 82,0 dB | 4,0e-3, 4,7e-4, 80,3 / 80,1 dB |
| 1/4 | 3 cross décalée | 1,9e-6, 152,9 dB | 2,8e-3, 3,9e-4, 86,4 / 86,3 dB | 4,1e-3, 5,1e-4, 84,0 / 83,8 dB |
| 1/4 | 10 self décalée | 2,9e-6, 149,9 dB | 3,9e-3, 3,7e-4, 81,7 / 81,5 dB | 5,2e-3, 4,5e-4, 79,9 / 79,7 dB |
| 1/4 | 11 cross décalée | 1,9e-6, 152,8 dB | 2,6e-3, 4,4e-4, 85,2 / 85,0 dB | 3,8e-3, 5,7e-4, 83,1 / 82,9 dB |

Le contrôle f32, sans aucun arrondi, rend ONNX Runtime FP32 à l'ordre des sommes près (≤ 3e-6) aux deux tailles de
fenêtres, décalées ou non, en self comme en cross : poids rangés, fenêtres, décalage par adresses, régions et
valeur du masque, cible de la cross et résiduel sont justes. Le natif FP16 reste à 3,7 à 5,3e-4 en relatif, l'ordre d'un
arrondi FP16 (2⁻¹¹ ≈ 4,9e-4) : la sortie FP16 domine, le softmax en ligne et P en FP16 n'ajoutent presque rien. La
recette « nœud par nœud » arrondit en FP16 la sortie de chaque nœud du graphe (q, k, v, scores, scores / √d avec √d
en FP16, masque, softmax, P · V, fusion, LayerNorm, somme), ce que ferait un moteur FP16 qui ne fond rien : le natif
est 2 à 4 dB au-dessus, et jusqu'à 5 fois plus près au pire écart (cross du premier bloc). Ces écarts sont ceux d'une couche à
entrée exacte ; ce qu'ils deviennent sur douze couches et six MLP se mesurera à l'assemblage du transformeur.

**Banc** (pour le PC ; rien n'est conclu ici sur la vitesse). Le natif : `gmflow_attention_banc` (dans
`tools/gmfss/gmflow_attention`, avec `-DKIRARI_BANC_CUDA=ON`), une couche lue dans la paire du lecteur, capturée en
graphe CUDA (QKV puis attention), 21 mesures de 200 lancements, médiane et minimum en µs, et le débit sur les FLOP
de la couche. TensorRT : la même couche seule en ONNX par `gmflow_attention_onnx.py` (poids du plan, mêmes
opérations que l'export de la paire, FP16, taille fixe), vérifiée sous ONNX Runtime contre le contrôle f32
(≤ 1e-5 : 2,7e-6 en self décalée au quart, 8,9e-6 en cross au quart), construite par trtexec avec la recette de la
paire (`Recipe::Fp32Islands` : `--stronglyTyped`, entrées et sorties FP16, sans cuDNN ni cuBLAS, tuilage 3).

```sh
cmake -S tools/gmfss/gmflow_attention -B build/gmflow_attention -DCMAKE_BUILD_TYPE=Release -DKIRARI_BANC_CUDA=ON
cmake --build build/gmflow_attention
P=<dossier des modèles du lecteur>/gmfss_fortuna_union_pair.onnx
build/gmflow_attention/gmflow_attention_vide $P attn16
# couches 0 (self), 1 (cross), 2 (self décalée), 3 (cross décalée), au huitième (0) et au quart (1)
for c in "0 0" "1 0" "2 0" "3 0" "0 1" "1 1" "2 1" "3 1"; do set -- $c
  python -I tools/gmfss/gmflow_attention_onnx.py --plan attn16 --couche $1 --echelle $2 --sortie couches/c$1_e$2.onnx
done
# en alternance ABAB, même séance : TensorRT puis natif, pour chaque couche (keys en plus en cross)
trtexec --onnx=couches/c2_e1.onnx --stronglyTyped --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
  --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --tilingOptimizationLevel=3 --useCudaGraph --noDataTransfers \
  --warmUp=500 --duration=10
build/gmflow_attention/gmflow_attention_banc $P temps 1 2
trtexec --onnx=couches/c3_e1.onnx --stronglyTyped --inputIOFormats=fp16:chw,fp16:chw --outputIOFormats=fp16:chw \
  --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --tilingOptimizationLevel=3 --useCudaGraph --noDataTransfers \
  --warmUp=500 --duration=10
build/gmflow_attention/gmflow_attention_banc $P temps 1 3
```

Comparer la médiane « GPU Compute Time » de trtexec à celle du banc. Avant de chronométrer, le premier lancement se
vérifie : `gmflow_attention_emule.py … --paires 1 --dump verif` écrit l'entrée, la sortie émulée et la sortie FP32
des couches mesurées, puis `gmflow_attention_banc $P verifier verif` les rejoue sur la carte (attendu : à un ulp
FP16 près de l'émulation, et les PSNR du tableau contre la paire FP32). La paire FP32 vient de
`gmflow_natif_emule.py --work`.

**À surveiller au banc, et ce qui manque avant de brancher** : au huitième, l'attention ne lance que 120 blocs
(15 tranches de 64 requêtes, 4 fenêtres, lot 2), moins d'une vague sur les 60 SM de la RTX 4070 Ti, et 32 au
demi-flot : si le banc le montre, couper les clés d'une fenêtre entre plusieurs blocs (maximum, somme et O
partiels fusionnés ensuite). Les variantes de sommes `kF16` et `kMixed` de QKV et de la fusion, chronométrées
contre TensorRT comme RIFE et GMFSS. La position sinus (voir plus haut). Le MLP fondu (PR 3), puis l'assemblage du
transformeur en graphe CUDA, et l'écart sur ses douze couches.

## MLP natif (PR 3)

Écrit sans GPU : compilé pour sm_75, 86, 89 et 120 sans débordement de registres, jamais lancé. Rien ne l'appelle :
le transformeur reste dans la paire TensorRT.

**Ce que fait le graphe** (lu nœud par nœud dans la paire exportée ; le plan plus haut est corrigé là où il s'en
écartait). Dans chaque couche de cross-attention, message = norm1(fusion(attention)) ; le MLP lit Concat([x,
message]) sur le dernier axe, où x est l'entrée de la couche (la sortie de la self-attention du même bloc, qui sert
aussi au résiduel) ; `mlp.0` est un MatMul 256 → 1024 **sans biais**, la GELU est **exacte** (Div par √2, Erf, + 1,
× x, × 0,5), `mlp.2` un MatMul 1024 → 128 sans biais, `norm2` une LayerNorm à échelle et décalage appris (ε = 1e-5,
statistiques en f32), puis Add(x, norm2) : la LayerNorm d'abord, le résiduel ensuite. Les deux appels du
transformeur (1/8 et 1/4) partagent les initialiseurs (un Identity au second). Dans la paire FP16 du lecteur, √2 est
stocké en FP16 (1,4140625 : 1,1e-4 d'écart relatif, moins que l'arrondi relatif du FP16, 2⁻¹¹ = 4,9e-4) ; le noyau
multiplie par le 1/√2 exact.

**Lecture** (`src/prepare/gmflow_mlp.{h,cpp}`, types sans `std::expected` dans `gmflow_mlp_plan.h` pour nvcc).
`gmflow_mlp(paire)` trouve les nœuds de chaque couche par leurs noms
(`/flownet/transformer/layers.<i>/cross_attn_ffn/mlp/mlp.0/MatMul`, suffixe `_1` au second appel) et suit le reste
d'arête en arête. Elle refuse (FieldMissing, avec le nom du nœud) une GELU autre (constantes √2, 1 et 0,5, ordre
des opérandes), un Concat sur un autre axe ou dans l'autre ordre, une couche cachée lue hors du MLP, un biais, des
poids non constants ou d'une autre forme, une LayerNorm sur un autre axe ou sans statistiques f32, un résiduel dans
l'autre ordre, des poids ou un ε différents d'une échelle à l'autre. Elle rend un **plan** : 6 couches (W1 rangé
[caché][256], W2 [128][caché], γ, β, ε) et 12 emplois (échelle, couche, noms ONNX de x, du message et de la sortie,
que l'émulation lit). `gmflow_mlp_check` le revérifie contre le noyau : 128 canaux, couche cachée multiple de 64
jusqu'à 4 096, poids finis et représentables en FP16, ε positif, chaque emploi valide et unique. Tests
(`tests/unit/prepare/gmflow_mlp_test.cpp`) : un faux transformeur aux noms et nœuds de la paire, lu et rangé, 14
fautes refusées, les fautes du plan, puis la vraie paire quand elle est là (sautée en CI, plafonds relevés d'un).

**Noyau** (`src/gpu/cuda/backend/gmflow_mlp_kernels.cuh`, `gmflow_mlp_net.{h,cu}`) : un bloc de 4 warps (64 jetons,
45 Ko de mémoire partagée, deux blocs par SM) ou de 8 warps (128 jetons, 78 Ko, un bloc par SM, pas sur Turing) ;
chaque warp tient 16 jetons. [x, message] se charge une fois en mémoire partagée (`cp.async`). Pour chacune des 16
tranches de 64 canaux cachés : H = X·W1 (mma.sync m16n8k16 de `rife_native_kernels.cuh`), GELU en f32 arrondie en
FP16 directement dans les fragments A du mma suivant (la sortie de deux tuiles m16n8 voisines est un fragment A
m16k16 : H ne passe ni par la mémoire partagée ni par la mémoire globale), puis O += H·W2. O (16 × 128 en f32 par
warp) reste en registres jusqu'à la fin : LayerNorm de chaque ligne en deux passes (moyenne, puis variance des
écarts, réduites entre les quatre fils de la ligne par `__shfl_xor`), résiduel lu dans la tuile x, un seul arrondi
FP16, écriture par lignes entières ; out peut être x. Les poids passent par un anneau de trois étages de 4 Ko,
rangés au chargement en un flux dans l'ordre où le noyau les prend (fragments B prêts, un uint4 par fil) : la L2 les
donne une fois par bloc, 768 Ko pour 64 ou 128 jetons (65 ou 131 FLOP par octet lu). Pour les petites grilles (le
1/8 au demi-flot : 30 blocs de 64 jetons), la couche cachée se partage entre 2 à 8 blocs (`blockIdx.y`) qui écrivent
leurs parts de O en f32 ; un second noyau les somme dans un ordre fixe (déterministe), puis fait LayerNorm et
résiduel. Registres : 203 (sm_75), 212 (sm_86 et 89), 165 à 168 (sm_120), aucun débordement. Capturable dans un
graphe CUDA.

Sommes : seul `Acc::kF32` est instancié, le départ le plus sûr (l'étude donne aux sommes f16 −5 dB sur l'image) ;
le paramètre `Acc` reste pour `kMixed` et `kF16` (PR 7), à chronométrer après le banc ci-dessous.

**Émulation** (`tools/gmfss/gmflow_mlp_emule.py`, plan et poids vidés par `tools/gmfss/gmflow_mlp/gmflow_mlp_vide`,
qui appelle la lecture du lecteur) : le plan rejoué avec les arrondis du noyau, contre le même MLP dans la paire
FP32 sous ONNX Runtime (CPU), sur les vraies entrées de chaque couche : paire 1280x768 au flot entier (GMFlow sur
640x384 : 3 840 jetons par image à 1/8 en lot 2, 15 360 à 1/4 en lot 4), quatre paires de `calibration.mkv` (deux
images à 2/24 s d'écart, à 15, 39, 69 et 99 s), six couches aux deux échelles. Écart relatif √EQM / √(moyenne des
carrés de la référence), moyen sur les 24 (paire, couche) ; PSNR sur la crête de la référence, moyen / pire.
« Sortie » : la sortie de la couche, après le résiduel (ce que lit la suite) ; « branche » : la sortie de norm2, ce
que le MLP ajoute.

| Échelle | Recette | Sortie : écart relatif | Sortie : PSNR (dB) | Branche : écart relatif | Branche : PSNR (dB) |
|---|---|---|---|---|---|
| 1/8 | contrôle f32 (plan de la paire FP32) | 2,0e-7 | 148,9 / 146,5 | 2,3e-7 | 147,6 / 145,2 |
| 1/8 | **natif FP16** | 3,55e-4 | **83,8 / 81,8** | 3,24e-4 | **84,9 / 81,1** |
| 1/8 | ONNX Runtime sur la paire FP16 | 3,46e-4 | 84,0 / 82,1 | 4,01e-4 | 83,0 / 79,8 (*) |
| 1/8 | stockage FP16 de chaque nœud (façon TensorRT) | 4,69e-4 | 81,4 / 79,3 | 4,85e-4 | 81,3 / 78,0 |
| 1/4 | contrôle f32 (plan de la paire FP32) | 2,7e-7 | 146,1 / 142,4 | 3,3e-7 | 144,8 / 141,1 |
| 1/4 | **natif FP16** | 3,63e-4 | **83,6 / 80,6** | 3,41e-4 | **84,5 / 80,8** |
| 1/4 | ONNX Runtime sur la paire FP16 | 3,52e-4 | 83,8 / 80,9 | 4,12e-4 | 82,8 / 79,7 (*) |
| 1/4 | stockage FP16 de chaque nœud (façon TensorRT) | 4,80e-4 | 81,1 / 78,3 | 5,05e-4 | 81,1 / 77,9 |

(*) sortie moins x : l'arrondi FP16 de la sortie y compte.

- **L'indexation est juste** : sans aucun arrondi, le plan rend le graphe FP32 à l'ordre des sommes près (≈ 147 dB) :
  rangement de W1 et W2, GELU, LayerNorm, résiduel.
- **Sur la sortie**, le natif FP16 est à 0,2 ou 0,3 dB d'ONNX Runtime sur la paire FP16, qui garde des poids FP16
  mais calcule le MLP en f32 sur le CPU (le natif arrondit H en FP16 avant le second produit) : tous deux sont
  dominés par l'arrondi FP16 de la sortie. Face à un stockage FP16 de chaque nœud, plus proche de TensorRT FP16, le
  natif gagne 2,4 dB en moyenne et 2,3 à 2,5 dB au pire.
- **Sur la branche**, que le natif n'arrondit jamais, il gagne 3,4 dB sur le stockage FP16 de chaque nœud. Le MLP
  natif ne perd rien ; la comparaison qui compte, contre le moteur TensorRT, se fait sur la carte.
- Le demi-flot n'est pas émulé à part : le MLP agit jeton par jeton, le demi-flot n'en change que le nombre.

**Banc sur le PC** (RTX 4070 Ti ; rien n'a tourné sur une carte). D'abord les entrées réelles du contrôle, sur
n'importe quel poste avec PyTorch et ONNX Runtime (≈ 10 min de CPU, 11 Go de mémoire au plus). La première commande
n'est là que pour la paire FP32 : elle l'exporte avant tout, la suite de son émulation peut s'interrompre.

```sh
python -I tools/gmfss/gmflow_natif_emule.py ROUE/vsgmfss_fortuna --work W --recettes fp32  # W/paire_fp32.onnx
cmake -S tools/gmfss/gmflow_mlp -B build/gmflow_mlp && cmake --build build/gmflow_mlp --target gmflow_mlp_vide
build/gmflow_mlp/gmflow_mlp_vide W/paire_fp32.onnx W/mlp32
build/gmflow_mlp/gmflow_mlp_vide ~/kirari/models/rife/gmfss_fortuna_union_pair.onnx W/mlp16
python -I tools/gmfss/gmflow_mlp_emule.py --fp32 W/paire_fp32.onnx \
    --fp16 ~/kirari/models/rife/gmfss_fortuna_union_pair.onnx --plan32 W/mlp32 --plan16 W/mlp16 --work W \
    --banc W/entrees
```

Puis sur le PC, `W/entrees` copié (170 Mo) :

```sh
flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_mlp_banc.sh \
    ~/kirari/models/rife/gmfss_fortuna_union_pair.onnx ~/kirari-reports/gmflow-mlp W/entrees
```

Le script construit `gmflow_mlp_vide` et `gmflow_mlp_banc` (CMake, nvcc du PC, architecture de la carte), extrait
de la paire le MLP de la couche 0 seul (entrées x et message [lot, jetons, 128] FP16, sortie out), passe le
contrôle (`gmflow_mlp_banc --check`), puis pour chacune des huit charges (paires 1280x768 et 1920x1152, flot entier
et demi-flot, 1/8 au lot 2 et 1/4 au lot 4) : le moteur TensorRT du sous-graphe construit comme le lecteur construit
la paire (`--stronglyTyped`, niveau 5, sans cuDNN ni cuBLAS quand trtexec prend encore `--tacticSources`),
chronométré en graphe CUDA (`--useCudaGraph --noDataTransfers --useSpinWait`, médiane), et le noyau natif aux deux
blocs, chronométré de même (graphe CUDA, médiane de 300) ; deux tours ABAB, moyenne des deux médianes. Il écrit
`~/kirari-reports/gmflow-mlp/banc-mlp.md` : par charge les trois temps, le rapport du meilleur natif à TensorRT et
son seuil ; puis la somme par paire, 6 × (1/8 + 1/4). Les profils de trtexec (`mlp-*.profile.json`) disent si
TensorRT a fait du MLP une seule région Myelin.

**Ce que la mesure doit montrer pour continuer** (règle du propriétaire : TensorRT ne s'enlève que si c'est
vraiment évitable, sans perte de qualité ni de vitesse) :

1. **Juste d'abord** : au contrôle, sur les quatre entrées (couches 0 et 5, 1/8 et 1/4), un PSNR contre le FP32
   égal à celui de l'émulation à 0,1 dB près, et au plus 0,1 % des valeurs à plus d'un ulp FP16 de l'émulation
   (l'ordre des sommes du mma). Sinon le noyau est faux et ses temps ne valent rien.
2. **Nettement plus rapide là où se joue la vitesse** : meilleur bloc natif ≤ **0,80 ×** TensorRT au 1/4 en 720p,
   au flot entier (61 440 jetons) et au demi-flot (15 360), et ≤ 1,00 × sur les six autres charges (1/8, 1080p).
   Le MLP à 1/4 fait 40 % du calcul de GMFlow et c'est le bloc le plus favorable au natif (deux GEMM, la couche
   cachée gardée sur la puce) ; l'étude n'attend GMFlow natif à 0,6 × TensorRT qu'avec des rendements que
   l'attention et le reste tiendront moins bien. Sans 20 % de gagné ici, le reste ne fera pas la marge.

Décision : tout tenu, continuer (PR 2 mesurée de même, puis PR 4 et 5) ; une charge à 1/4 de 720p au-dessus de
1,00, **arrêter** : TensorRT garde GMFlow (la conclusion de l'étude) ; entre les deux (1/4 de 720p entre 0,80 et
1,00, ou une autre charge au-dessus de 1,00), le MLP seul ne tranche pas et le banc de la PR 2 décide, sur la couche
entière du transformeur à 1/4 (attention et MLP) : ≤ 0,80 × TensorRT, ou l'arrêt.

Le banc mesure le MLP isolé : dans la paire, TensorRT peut le fondre avec ses voisins (norm1, la fusion de
l'attention) ; le profil par couche de la paire (PR 0) dira si le temps isolé est le bon repère. **Rien n'est conclu
ici sur la vitesse** : aucun temps de ce noyau n'a été mesuré.

**Ce qui manque** : le premier lancement sur la carte (contrôle, puis banc) ; le choix du bloc (64 ou 128 jetons) et
des parts d'après la mesure ; les sommes `kMixed` et `kF16` ; le branchement dans le transformeur natif, avec la PR
2, et la lecture des poids au chargement du backend (aujourd'hui, seul l'outil de banc les lit).

## Découpage en PR proposé

| PR | Contenu | Test |
|---|---|---|
| 0 | **Mesure** : profil par couche du moteur paire TensorRT (`trtexec --dumpProfile --separateProfileRun`, 1280x768, flot entier et demi-flot) pour séparer GMFlow, FeatureNet et MetricNet, et le temps de chaque bloc de GMFlow ; ces outils d'étude | les chiffres de « Temps estimés » remplacés par des mesures |
| 1 | Squelette natif : 7x7/2 en im2col, 1x1, statistiques d'InstanceNorm à l'épilogue et normalisation au chargement, fins de bloc, trident + position | sorties 1/4 et 1/8 contre un sous-graphe ONNX exporté (FP32, ONNX Runtime) et contre TensorRT |
| 2 | GEMM QKV, attention fenêtrée fondue (décalage, masque, fusion, LayerNorm, résiduel) | une couche contre PyTorch FP32 aux deux tailles de fenêtres ; banc contre la même couche sous TensorRT |
| 3 | MLP fondu (256 → 1024 → 128, GELU, LayerNorm, résiduel) | idem ; c'est le bloc qui décide de la vitesse : banc avant de continuer |
| 4 | Appariement global et propagation globale (softmax fondu, deux passages, V en f32) | contre `global_correlation_softmax` et `FeatureFlowAttention` |
| 5 | Fin à 1/4 (déformation, appariement local, propagation 3x3, agrandissement convexe fondu), assemblage de GMFlow, graphe CUDA | flot entier contre ONNX FP32 sur les paires de l'émulation et de vraies paires ; banc ABAB contre la paire TensorRT |
| 6 | Paire native entière (avec FeatureNet et MetricNet natifs) derrière un réglage de `play.toml`, TensorRT par défaut | voir la conclusion : la bascule ne se fait que sur mesure |
| 7 (facultatif) | Sommes f16 dans les GEMM et convolutions, avec le garde-fou de débordement de SPAN et repli f32 | perte ≤ 5 dB sur l'image contre le natif f32, et toujours au-dessus de TensorRT |

Pas de PR FP8 : l'ablation l'exclut pour GMFlow.

## Risques

- **Vitesse non prouvée** : tout repose sur des rendements supposés (40 % pour l'attention, 70 % pour les GEMM) et
  sur un temps de TensorRT déduit. Si Myelin fond déjà l'attention et le MLP, l'écart peut fondre. La PR 0 et le
  banc de la PR 3 tranchent tôt, avant l'essentiel du code.
- **Attention à une tête de 128** : le bloc de sortie (64 requêtes × 128 canaux en f32) pèse 64 registres par
  fil ; avec S, P et les fragments, l'occupation tombe à un bloc par SM. Les fenêtres de 960 (1/8) et 2 160
  (1080p) jetons se parcourent en tuiles, sans limite de taille, mais avec 15 à 34 tuiles par requête.
- **Sommes f16** : 5 dB de moins sur l'image ; débordement possible sur une image atypique (la leçon de SPAN :
  garde-fou et repli f32 à prévoir). Le FP8 à sommes f16 tel quel déborde.
- **Paires de test** : mouvements de caméra synthétiques sur de vraies images. Le retrait du FP8 de GMFSS le 30/10
  est venu de mouvements forts que les paires de test ne couvraient pas : avant une bascule, mesurer sur de vraies
  paires d'anime à mouvement fort (panoramiques rapides, rotations, effets).
- **InstanceNorm** : une réduction sur toute l'image par couche ; des additions atomiques rendent les sommes non
  déterministes à l'ulp près (réduction en deux temps si l'on veut des sorties identiques d'un lancement à l'autre).
- **Maintenance** : ≈ 3 000 lignes de CUDA de plus, Turing (sm_75 : m16n8k8, pas de FP8) à garder ou à exclure.
- **Supprimer la construction** de TensorRT pour GMFSS demande aussi FeatureNet et MetricNet natifs (chantier
  voisin) : GMFlow natif seul ne retire que la plus longue part des 179 s.

## Conclusion

Règle posée par le propriétaire : la construction TensorRT de GMFlow ne se supprime que si elle est vraiment
évitable. L'attente de construction est acceptable si elle achète de la vitesse ou de la sûreté.

**Critère de bascule.** Le natif ne devient le défaut que si les deux conditions sont **mesurées** sur la RTX
4070 Ti et tenues :

1. **qualité au moins égale** à TensorRT : sur les paires de l'émulation et sur de vraies paires d'anime à
   mouvement fort, EPE moyen et 99,9e centile du flot contre le FP32, et PSNR moyen et pire de l'image du milieu,
   au moins aussi bons que ceux du moteur TensorRT FP16 de la paire, mesurés sur la carte et non en émulation ;
2. **vitesse au moins égale** : médiane ABAB (trtexec contre le banc natif, graphe CUDA, même séance) de la paire
   entière inférieure ou égale à celle de TensorRT, au flot entier et au demi-flot, à 1280x768 et 1920x1152.

Si une condition manque, même de peu, **TensorRT reste le défaut** et le natif reste un réglage facultatif de
`play.toml`, pour qui ne veut pas attendre la construction.

**Ce que l'étude permet d'attendre.** Sur la qualité, l'émulation donne au natif FP16 à sommes f32 un flot 3,5 à
4,5 fois plus proche du FP32 que le graphe FP16 actuel (EPE 0,006 contre 0,021 au flot entier, 0,009
contre 0,042 au demi-flot à 1280x768), et ≈ 10 dB de mieux sur l'image (paire 640x384) : la condition 1 a de
bonnes chances de tenir, à confirmer sur la carte. Sur la vitesse, le modèle donne GMFlow natif en ≈ 18 ms au flot
entier et ≈ 4 ms au demi-flot, contre ≈ 29 et ≈ 6,5 ms déduits pour TensorRT : un avantage sur le papier, mais
reposant sur des rendements supposés et un temps de TensorRT non mesuré. La condition 2 n'est pas acquise. Le FP8
est exclu pour GMFlow (−10 à −25 dB sur l'image par famille de GEMM) : la seule voie plus rapide que le FP16 à
sommes f32 est l'accumulation f16 (−5 dB, encore au-dessus de l'actuel).

**Suite proposée** : la PR 0 (profil par couche de la paire TensorRT), puis les PR 1 à 3 et un banc du MLP fondu
contre TensorRT avant d'écrire le reste. Si le transformeur à 1/4 natif n'est pas nettement plus rapide que celui
de TensorRT à ce stade, arrêter : TensorRT garde GMFlow, et l'étude aura servi à mesurer sa part.
