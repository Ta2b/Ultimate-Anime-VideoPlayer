# GMFSS union en temps réel : bilan du chantier natif

> **Mis à jour le 19 novembre 2026, en fin de journée** :
> - GridNet : la forme en flux (chargement en double tampon) est écrite et entrée dans #959 comme candidate du
>   réglage, avec son étape à la séance PC ; le bloc résiduel fondu est écarté pour l'instant (parties 2, 3.4, 4) ;
> - GMFlow natif : attention fondue (PR 2) et MLP fondu (PR 3), audités et entrés dans #959 ; leurs bancs et leur
>   règle de décision ajoutés à la séance PC (partie 4) ;
> - avant : le budget du ×2 en 1080p corrigé (partie 3.2).

Le but : GMFSS Fortuna Union en temps réel sur la RTX 4070 Ti de Tatoub. Ses
petits réseaux (le GridNet de la fusion, FeatureNet et MetricNet de la paire) passent par nos noyaux CUDA ; GMFlow
et l'IFNet restent sous TensorRT pour l'instant. Le détail technique est dans [`gmfss-natif.md`](gmfss-natif.md) et
[`gmflow-natif.md`](gmflow-natif.md). Ce document dit ce qui est fait, ce qui a été essayé, jusqu'où l'on peut
aller, ce qui reste, et où reprendre.

Décision de Tatoub : la séance PC de GMFSS passe en dernier, après les autres chantiers. Tout ce qui peut se faire
dans le cloud s'y fait d'ici là.

## 1. Ce qui est fait

### Les PR

| PR | Contenu | État |
|---|---|---|
| #870 | Étape 1 : le tri des paires dans l'étage GMFSS (`--pair-triage` : dessins tenus, paires impossibles, panoramiques purs) | fusionnée |
| #872 | Étape 2 : les lectures des splats sur la bibliothèque de warp | fusionnée |
| #902 | Étape 3 : le GridNet de la fusion par nos noyaux (`--gmfss-native`) | fusionnée |
| #923 | Étape 4 : FeatureNet et MetricNet de la paire par nos noyaux | fusionnée |
| #927 | Étape 5 : GMFSS union de retour au menu, derrière le natif | fusionnée |
| #930 | GMFlow natif, PR 1 : étude et squelette CNN (pas encore branché) | fusionnée |
| #931 | Commandes de la séance PC | fusionnée |
| #950 | La fusion se coupe sur les deux exports (IFNet d'union et de RIFE 4.26), recherche des graphes élargie | fusionnée |
| #959 | Le natif par défaut ; mélange par morceau et réglage des formes par couche ; plages NVTX ; une paire périmée cède à l'autre paire du dossier, dans les deux sens ; GMFlow natif, attention (PR 2) et MLP (PR 3), pas encore branchés ; ce bilan | ouverte : fusion après une CI verte sur sa tête et l'étape 0 de la séance PC |

### Les mesures du PC

Mesures faites sur la RTX 4070 Ti, sur un extrait de Re:Zero. Réglages : GMFSS union seul, SPAN derrière, plans
de la paire à 1920×1152 (lecture en 1080p), paire au demi-flot.

| Mesure | TensorRT | Natif | Écart |
|---|---|---|---|
| Une passe de la paire (essai au chargement) | 22,55 ms | **19,75 ms** (f16) | −12 %, 64,8 dB au pire contre TensorRT |
| Une passe de la fusion (essai au chargement) | 25,57 ms | **21,90 ms** (f16) | −14 %, 62,3 dB au pire contre TensorRT |
| Paire et fusion par image interpolée, en lecture (30 s, ×2, SPAN FP8) | 48,0 ms | **41,6 ms** | −13 % |
| Splats d'un instant (huit), en lecture | | 7,14 ms | |
| Export sans perte de 5 s, ×2, natif contre TensorRT | | **63,15 dB** en moyenne, 57,84 au pire | 50 dB visés |

En lecture, par `kirari compare`, avec TensorRT en A et le natif en B :
- en ×2, la cadence tient des deux côtés : 1,1 % et 1,4 % d'images perdues ;
- en ×3, elle ne tient ni d'un côté ni de l'autre : 35,6 % et 28,4 % perdues ;
- en ×2 avec le tri des paires, 0,7 % perdues des deux côtés.

En 1080p avec SPAN, GMFSS n'a pris que 180 des 671 paires fortes. RIFE a gardé les 491 autres, faute de temps.

À la première ouverture, la construction prend 149 s pour la paire, 180 s pour la fusion entière et 54,6 s pour la
fusion coupée, où l'IFNet reste seul sous TensorRT.

TensorRT FP16 lui-même n'est qu'à 67,7 dB du natif f32 sur la paire, et à 65,3 dB sur la fusion. Les écarts contre
TensorRT mêlent donc deux arrondis.

### Ce qui marche par défaut

- **Sur main** : GMFSS union est au menu derrière `--gmfss-native`. La paire au demi-flot
  (`gmfss_fortuna_union_s05_pair.onnx`) est prise par défaut quand elle est là.
- **Avec #959** :
  - le natif devient le défaut. La paire et la fusion ont chacune leur essai au chargement, et TensorRT reprend
    l'unité qui perd. `--no-gmfss-native` laisse tout GMFSS à TensorRT et le retire du menu ;
  - l'essai coupe chaque réseau en morceaux et règle la forme de chaque convolution au banc ;
  - une paire périmée, d'avant le format 2, cède à l'autre paire du dossier, avec un avertissement au journal.
    C'était le cas du PC : sa paire au flot entier est périmée, celle au demi-flot est à jour.

## 2. La recherche : pistes essayées, gardées ou écartées

| Piste | Verdict | Pourquoi |
|---|---|---|
| GridNet de la fusion par nos noyaux | **gardé**, f16 | −14 % contre TensorRT à 62,3 dB, mesuré au PC |
| FeatureNet et MetricNet par nos noyaux | **gardé**, f16 | −12 % sur la paire à 64,8 dB, mesuré au PC |
| Demi-flot (paire `s05`) | **gardé par défaut** | Paire environ −65 %, GMFSS environ −40 % en tout. Dans l'ancien lecteur, il coûtait 0,5 dB. Au flot entier, même ×2 en 720p ne tient pas. |
| FP8 de la paire (TensorRT, hors squelette de GMFlow) | **retiré** le 30/10 | 46,9 dB sur les paires de test, mais une très mauvaise image dans les gros mouvements, que ces paires ne couvraient pas |
| FP8 de la fusion | **écarté** | 36,9 dB sur un cas de mouvement fort |
| GMFlow en E4M3 (émulé) | **écarté** | 35,1 dB au pire sur l'image : chaque famille de GEMM seule en E4M3 coûte déjà 10 à 25 dB |
| Mélange par morceau (`native_mix.h`, commun avec le RIFE natif) | **dans #959**, jamais lancé sur une carte | En émulation, tous les crans tiennent 45 dB de loin. Fusion : f16 partout 66,5 dB, tête f32 72,8, mixtes 76,2. Paire : 65,2 à 72,0. Le choix se fera donc sur les temps. TensorRT ne peut pas servir un seul morceau, car la paire et la fusion sont chacune un seul moteur. |
| Formes réglées par couche | **dans #959** | Attendu sans mesure : 1,5 à 3 ms de moins sur la fusion, 0,3 à 1 ms sur la paire |
| GMFlow par nos noyaux | **en cours** | La PR 1 est fusionnée. Les PR 2 (attention fondue) et 3 (MLP fondu) sont écrites, auditées et dans #959 avec leurs bancs, sans être branchées. En émulation, l'image gagne 9,6 dB sur le graphe actuel, parce que le flot reste en f32. La vitesse n'est pas prouvée : le banc du PC décide de la suite. |
| IFNet de la fusion | **reste sous TensorRT** | Plafond 0,4 ms : rien à gagner en vitesse. La coupe accepte les deux exports. |
| Tri des paires (`--pair-triage --scene-map`) | **gardé** | 0,7 % d'images perdues en ×2 |
| Forme en flux du GridNet (chargement en double tampon) | **dans #959**, candidate du réglage | Écrite et vérifiée au bit près sur l'hôte ; gardée par le banc du chargement seulement si elle est plus rapide. Prévu : fusion de 21,9 vers 19,1 à 19,6 ms. |
| Bloc résiduel fondu de la ligne 0 | **écarté** pour l'instant | 0,5 ms au mieux, une perte au pire : le halo refait la première convolution 1,4 à 1,6 fois et la mémoire partagée ne laisse qu'un bloc par multiprocesseur |

## 3. Plafonds, et au-delà

### 3.1 Le plafond de chaque morceau en 1080p

Calcul fait sur processeur par `tools/gmfss/plafond_gmfss.py`, à partir des graphes de l'export d'origine. Les
formes sont celles d'une lecture en 1080p : plans 1920×1152, paire au demi-flot.

Hypothèses :
- débits de la 4070 Ti, avec 60 SM à 2,61 GHz :
  - 160 Tflop/s en sommes f16 ;
  - 80 Tflop/s en sommes f32 ;
  - 321 Tflop/s en FP8 ;
- bande passante de 504 Go/s ;
- chaque couche lit son entrée une fois, écrit sa sortie une fois et lit ses poids, en f16. PReLU, additions et
  PixelShuffle sont fondus ;
- le plafond d'une couche est le plus lent du calcul et des octets. Celui d'un morceau est la somme de ses couches.

| Morceau | Gflop | Mo minimaux | Plafond f16 | Plafond FP8 | Mesuré |
|---|---|---|---|---|---|
| Paire : FeatureNet (1152×1920) | 284 | 959 | 2,3 ms | 1,9 ms | |
| Paire : MetricNet (1152×1920) | 133 | 584 | 1,2 ms | 1,2 ms | |
| Paire : GMFlow, demi-flot (288×480) | 363 | 3 283 | 2,3 à 6,6 ms | (exclu) | probablement 12 à 15 ms (déduit, pas mesuré) |
| **Paire entière** | **780** | | **5,7 à 10,1 ms** | | **19,75 ms** |
| Fusion : IFNet (576×960) | 44 | 171 | 0,4 ms | 0,3 ms | |
| Fusion : GridNet (576×960, sortie 1152×1920) | 1 560 | 3 987 | 10,9 ms | 8,2 ms | environ 20 ms |
| **Fusion entière** | **1 604** | | **11,2 ms** | **8,5 ms** | **21,90 ms** |
| Splats d'un instant (estimés à la main, voir plus bas) | | environ 1 500 | environ 3 ms | | **7,14 ms** |

Notes sur le tableau :
- **GMFlow** : le plafond est une fourchette.
  - 2,3 ms si l'attention est fondue, c'est-à-dire que ses scores ne quittent pas la puce.
  - 6,6 ms si chaque produit de matrices écrit sa sortie en mémoire.
- **Splats** : le calcul est fait pour ce bilan. Il compte les deux côtés, l'image et les caractéristiques de 64,
  128 et 192 canaux à 576×960, 288×480 et 144×240. Chaque source est lue une fois en f16, les sommes f32 sont
  écrites puis relues une fois, et la sortie f16 est écrite une fois.

**Où va le temps aujourd'hui** (hypothèses ; le profil Nsight de la séance les tranchera) :
- **Fusion, deux fois son plafond.** `conv_kernel` ne recouvre pas son propre chargement : il charge une tranche
  par `cp.async`, l'attend, puis calcule. Le calcul (9,7 ms) et les octets (7,9 ms) s'additionnent alors en
  17,6 ms, près des quelque 20 ms du GridNet. Les convolutions de la ligne 0 (64 canaux à 960×576) sont à
  l'équilibre : 0,255 ms de calcul contre 0,28 ms d'octets. Elles coûtent 0,28 ms recouvertes, 0,54 ms l'un
  après l'autre. Onze d'entre elles sont des 3×3 de 64 vers 64 au pas 1, dont six lisent un résiduel : comptés,
  ces résiduels mettent le plafond de la ligne 0 à 3,9 ms et celui de la fusion vers 12 ms plutôt que 11,2.
- **Paire, deux à trois fois et demie son plafond.** FeatureNet et MetricNet natifs ont un plafond de 3,5 ms à eux
  deux. Le reste est GMFlow sous TensorRT. L'étude de GMFlow déduit 6 à 7 ms en travail 720p au demi-flot, soit
  environ 13 à 15 ms en 1080p.
- **Splats, 2,4 fois leur plafond.** Les sommes f32 font l'aller-retour en mémoire. Les entrées en plans du GridNet
  ajoutent deux copies.

### 3.2 Le budget du temps réel, corrigé

À 23,976 i/s, une période source dure 41,7 ms. Elle paie :
- la paire, une fois ;
- pour chaque image que GMFSS crée : ses splats, sa fusion et **une passe de SPAN**. GMFSS crée son image avant
  SPAN. Seuls RIFE et DistilDRBA savent warper après SPAN (en Auto au-dessus de 1080p ou dès ×3), GMFSS ne le
  peut pas ;
- le SPAN de l'image source ;
- environ 2 ms pour le reste de la chaîne : grade, présentation, mesures. C'est à lire au profil.

**Correction.** `gmfss-natif.md` disait que la porte du ×2 en 1080p laissait « environ 28 ms » à la paire et à la
fusion. Ce compte ne prenait qu'un SPAN par période source et oubliait les splats. Le budget réel est le suivant :
- **avec le SPAN d'aujourd'hui** (b8f64 natif au cran Équilibre, environ 11 ms par image 1080p vers 4K) :
  41,7 − 2 × 11 − 7,1 − 2, soit
  environ **10,6 ms** pour la paire et la fusion, quand chaque paire passe par GMFSS. C'est sous la somme de leurs
  plafonds f16, 17 à 21 ms ;
- **avec un SPAN à 6 ms** (le modèle plus léger b5f48 fait 6,2 ms chez AnimeJaNai ; le lecteur ne l'a pas
  mesuré) : il reste environ 20,6 ms, à portée des plafonds.

**Vérification sur la mesure du PC.** Avec SPAN FP8 à 14,5 ms par image :
- le modèle donne environ 80 ms pour une paire prise par GMFSS ;
- il donne environ 20 ms pour une paire laissée à RIFE, en supposant 4 ms pour RIFE lui-même ;
- le crédit de temps ne peut donc donner à GMFSS qu'au plus 36 % environ des paires fortes.

Le PC en a donné 27 % (180 sur 671). C'est cohérent avec un crédit prudent.

### 3.3 Ce que donne chaque palier

Le tableau donne le coût d'une période source quand **chaque** paire passe par GMFSS seul. C'est le pire cas, celui
de l'extrait de Re:Zero : 671 paires fortes sur environ 720. Une cadence tient si le coût reste sous 41,7 ms. Une
entrée en gras tient. Les images par seconde montrées sont 48 en ×2, 72 en ×3, 96 en ×4 et 120 en ×5.

Coût d'une période source, en ms :

| Sortie, SPAN | Palier | ×2 | ×3 | ×4 | ×5 |
|---|---|---|---|---|---|
| 1080p vers 4K, SPAN d'aujourd'hui (11 ms) | aujourd'hui | 72,7 | 112,7 | | |
| | plafonds f16 de GMFSS | 46,1 | 71,3 | | |
| | au-delà | **37,0** | 55,3 | | |
| 1080p vers 4K, SPAN à 6 ms | aujourd'hui | 62,7 | 97,7 | | |
| | plafonds f16 de GMFSS | **36,1** | 56,3 | | |
| | au-delà | **27,0** | **40,3** | 53,6 | |
| Travail 720p vers 1440p, SPAN d'aujourd'hui (4,9 ms) | aujourd'hui | **33,4** | 51,2 | | |
| | plafonds f16 de GMFSS | **21,6** | **32,8** | 44,0 | |
| | au-delà | | **25,7** | **33,8** | 42,0, à la limite |

Les trois paliers :
- **Aujourd'hui** : les mesures du PC. Paire 19,7 ms, splats 7,1 ms, fusion 21,9 ms.
- **Plafonds f16 de GMFSS** : paire 7,9 ms (le milieu de sa fourchette), splats 3 ms, fusion 11,2 ms.
- **Au-delà** :
  - paire 5,7 ms, avec l'attention de GMFlow fondue, toujours en f16 ;
  - splats 1,5 ms, avec la normalisation fondue dans le GridNet ;
  - fusion 5,8 ms, en FP8 avec des couches fondues : le plafond du calcul seul en FP8 est de 4,9 ms pour le
    GridNet.

Le travail 720p met tout à l'échelle de ses pixels, soit 0,444 fois ceux du 1080p. SPAN y vaut le même rapport.

Ce qu'on en tire :
- **En 1080p vers 4K avec le SPAN d'aujourd'hui, le mur est SPAN.** En ×2, deux passes prennent 22 ms sur 41,7. Aux
  plafonds f16, GMFSS seul ne tient pas le ×2 sur chaque paire. Il faut :
  - soit les leviers au-delà du plafond ;
  - soit un SPAN plus léger ;
  - soit laisser le tri et le crédit lui donner les paires fortes que le temps permet, ce que fait déjà le lecteur.
- **En ×3 et plus, la voie est le mélange :**
  - une clé GMFSS au milieu de la paire, passée une fois par SPAN ;
  - RIFE warpe les autres instants après SPAN, pour environ 0,4 ms par warp ;
  - le coût reste alors proche de celui du ×2 de GMFSS seul.
- **En travail 720p** :
  - le ×2 tient déjà sur chaque paire, au modèle ;
  - le ×3 tient aux plafonds f16 ;
  - le ×4 tient au-delà.

  La séance PC (étape 2) mesure le ×2 et le ×3 d'aujourd'hui, en 1080p comme en 720p.

### 3.4 Les leviers au-delà du plafond

Le plafond par couche suppose que chaque couche lit et écrit la mémoire. On peut aller plus bas de quatre façons :
- garder les tenseurs sur la puce d'une couche à l'autre ;
- faire moins de multiplications (Winograd, creux 2:4) ;
- passer en FP8 ;
- faire travailler GMFlow plus bas.

| Levier | Morceau | Gain attendu | Risque | Où |
|---|---|---|---|---|
| Forme en flux : un bloc descend une colonne, l'entrée dans un anneau de 10 lignes, la bande suivante en vol pendant les mma | GridNet, ligne 0 | ligne 0 de 6,7 vers 3,9 à 4,5 ms ; fusion de 21,9 vers 19,1 à 19,6 ms (modèle) | faible : le banc ne la garde que plus rapide et juste ; occupation de 8 warps, sm_75 sans doute perdant | **écrite, dans #959** ; temps au PC |
| Bloc résiduel fondu par tuiles (conv, PReLU, conv, addition), comme SPAN natif | GridNet, ligne 0 | 0,28 ms d'octets épargnés par bloc, mais le halo refait la première convolution 1,4 à 1,6 fois : +0,5 ms au mieux pour les trois blocs, une perte à 65 % des tensor cores | moyen à élevé : 70 à 99 Ko de mémoire partagée, un bloc par multiprocesseur, rien sur sm_75 | **écarté** ; à rouvrir si Nsight Compute montre le flux à 88 % de la bande avec des tensor cores sous 60 % |
| Queue fondue : la convolution du PixelShuffle écrit en sortie de phases et `conv_last` lit sa tuile sur la puce | GridNet, queue | octets de 650 à environ 85 Mo ; plafond de la queue de 1,6 à environ 1,1 ms | moyen | cloud, à faire |
| Couches enchaînées par bandes dans le L2 (48 Mo) | GridNet | plafond du calcul seul : 9,7 ms au lieu de 10,9 ; plus au réel, où la ligne 0 ne recouvre pas ses octets | moyen à élevé : ordre des bandes, halo qui grandit | cloud, après le double tampon |
| Winograd F(2,3) sur les couches limitées par le calcul (surtout les lignes 1 et 2 et la convolution du PixelShuffle : 6,3 ms au plafond) | GridNet | −2 à −2,5 ms au plafond. Utile seulement une fois les octets réglés : ces couches deviennent alors limitées par les octets. | moyen : précision f16 des transformées | émulation dans le cloud, puis noyaux |
| FP8 (E4M3) dans le GridNet, ligne par ligne | GridNet | plafond de 10,9 à 8,2 ms avec les poids seuls ; fusion vers 5 à 6 ms avec des activations FP8 et des couches fondues | **élevé** : le FP8 de GMFSS a été retiré pour les gros mouvements | émulation d'abord, sur de vraies paires à mouvement fort ; lignes 1 et 2 d'abord, la queue en f16 |
| Attention fenêtrée fondue (PR 2) et MLP fondu (PR 3) de GMFlow | paire | GMFlow de 6,6 à 2,3 ms au plafond. Modèle : environ 9 ms natif contre 13 à 15 sous TensorRT en 1080p. | moyen : vitesse non prouvée, Myelin fond peut-être déjà l'attention | cloud, en cours (`gmflow-attn-mlp`) ; banc au PC |
| Sommes f16 dans GMFlow | paire | débit doublé sur les GEMM ; −5 dB sur l'image, encore au-dessus du graphe actuel | moyen : débordement, il faut un garde-fou | après les PR 4 à 6 de GMFlow |
| Flot au quart (GMFlow sur 144×240 en 1080p) | paire | GMFlow divisé par environ 4 | **élevé** : le demi-flot coûtait déjà 0,5 dB | émulation d'abord, sur de gros mouvements |
| Splats : normalisation fondue dans la lecture du GridNet, entrées par pixel | splats et fusion | de 7,1 à environ 2 à 3 ms ; deux copies en moins | faible à moyen | cloud, à faire |
| Creux 2:4 (sparsity) | GridNet | ×2 sur les couches limitées par le calcul, environ −3 ms | **élevé** : demande un réentraînement et une requalification | déconseillé pour l'instant |
| IFNet par les noyaux du RIFE natif | fusion | temps : presque rien ; construction : −54,6 s | faible | cloud, facultatif |
| Hors GMFSS : un SPAN plus léger pour l'image créée | chaîne | 5 ms par image créée, avec un SPAN à 6 ms au lieu de 11 | qualité de SPAN | fil SPAN |

Ce que ces leviers donnent ensemble, d'après le tableau 3.3 :

| Sortie | GMFSS seul aujourd'hui | Aux plafonds f16 | Au-delà |
|---|---|---|---|
| **1080p vers 4K** | ×2 au crédit, environ 27 % des paires fortes | ×2 avec un SPAN à 6 ms | ×2 avec le SPAN d'aujourd'hui ; ×3 avec un SPAN à 6 ms |
| **Travail 720p** | ×2 | ×3 | ×4 ; ×5 à la limite |

Avec le mélange, une clé GMFSS par paire et RIFE pour les autres instants, chaque case gagne un ou deux paliers :
le coût reste à peu près celui du ×2 de GMFSS seul, plus les flots et les warps de RIFE.

## 4. Ce qui reste, dans l'ordre

### Dans le cloud, maintenant

1. Ce bilan, relu et fusionné. Il voyage avec #959 et corrige le budget de `gmfss-natif.md`.
2. GridNet : la forme en flux est faite et dans #959. Ensuite, par ordre de gain : la queue fondue, la
   normalisation des splats dans la lecture du GridNet, puis les couches enchaînées par bandes dans le L2 ; les
   lignes 1 et 2, limitées par le calcul, attendent l'émulation de Winograd et du FP8.
3. GMFlow : les PR 2 et 3 sont dans #959, auditées, avec leurs bancs. La suite attend le banc du PC (règle
   ci-dessous). Si elle continue :
   - PR 4 : appariement global et propagation globale au 1/8 ;
   - la position sinus, calculée au chargement ;
   - PR 5 : le chemin du 1/4 (déformation, transformeur, appariement local, propagation, agrandissement convexe) ;
   - l'assemblage en un `GmflowNet` (un espace de travail, un graphe CUDA par forme), le partage des clés entre
     blocs pour les petites grilles du 1/8, les sommes f16 sous la porte de qualité ;
   - la préparation de l'entrée du MetricNet en natif, pour sortir TensorRT de la paire ;
   - le branchement dans `GmfssPairEngine`, avec un essai au chargement contre TensorRT, coupé par défaut
     jusqu'à la mesure.

   Gain attendu en 1080p au demi-flot : GMFlow natif vers 9 ms en sommes f32 (6,5 à 7 ms en f16) contre 13 à 15 ms
   déduits pour TensorRT, soit 4 à 8 ms de moins par paire.
4. Nouveaux leviers de ce bilan :
   - la queue fondue ;
   - la normalisation des splats fondue dans le GridNet.
5. Émulations, avant d'écrire du code :
   - le FP8 du GridNet ligne par ligne, sur de vraies paires à mouvement fort ;
   - le flot au quart ;
   - l'erreur de Winograd F(2,3) en f16.
6. Facultatif : l'IFNet par les noyaux du RIFE natif, pour retirer la construction TensorRT de la fusion.

### Sur le PC, en dernier, une seule séance

À la tête de #959 (branche `claude/project-thread-1t05iy`), avec le binaire de lecture CUDA, pas `linux-prefix` en
Debug :

```sh
git fetch origin claude/project-thread-1t05iy && git checkout --detach FETCH_HEAD
cmake --preset linux-prefix-cuda-release && cmake --build --preset linux-prefix-cuda-release
G="--gmfss-dir $HOME/ajp/models/rife"   # <extrait> : l'extrait habituel de Re:Zero

# 0. Essai au chargement en 1080p (la première ouverture construit les moteurs).
kirari --debug play --motion=gmfss --rife=2 --start=960 $G <extrait> 2>&1 | grep -i "gmfss natif"

# 1. Profil Nsight Systems en 1080p ×2 : le temps de chaque plage gmfss.* et les 15 noyaux les plus lourds.
nsys profile -o gmfss --trace=cuda,nvtx --cuda-graph-trace=node --delay=20 --duration=30 \
  kirari --debug play --motion=gmfss --rife=2 --start=960 $G <extrait>
nsys stats -r nvtx_kern_sum,cuda_gpu_kern_sum gmfss.nsys-rep

# 2. Portes du temps réel : 1080p puis travail 720p, ×2 puis ×3, sans (A) puis avec (B) le tri des paires.
for w in "" "--work-height=720"; do for x in 2 3; do
  kirari compare <extrait> --a="--motion=gmfss --rife=$x --start=960 $G $w" \
    --b="--motion=gmfss --rife=$x --start=960 $G $w --scene-map --pair-triage" \
    --seconds=30 --report="gmfss${w:+-720}-x$x.txt"
done; done
```

À relever :
- **Étape 0** :
  - « N écartées » doit valoir 0 ;
  - une ligne « … seul : » par morceau ;
  - « retenu » ou « écarté », avec les dB et le temps d'une passe, à comparer aux 21,90 ms de la fusion et aux
    19,75 ms de la paire.
- **Étape 1** :
  - les plages `gmfss.paire.gmflow`, `paire.featurenet`, `paire.metricnet`, `fusion.ifnet`, `fusion.gridnet` et
    `splats` ;
  - les noyaux les plus lourds.

  Cette étape confirme ou corrige la partie 3.
- **Étape 2** :
  - les images perdues et le p95 ;
  - les paires prises par GMFSS et celles laissées à RIFE.

Ensuite, dans la même séance, les bancs de GMFlow natif (le détail est dans `gmflow-natif.md`, « Séance sur le PC
(attention et MLP) »). Il faut cmake, nvcc, trtexec, ffmpeg et un python avec torch, onnx et onnxruntime. `ROUE`
est le dossier `vsgmfss_fortuna` extrait de la roue de vs-gmfss_fortuna.

```sh
# 3. Entrées du contrôle (processeur, une vingtaine de minutes).
W=~/kirari-reports/gmflow-natif-entrees; P=$HOME/ajp/models/rife/gmfss_fortuna_union_pair.onnx
python -I tools/gmfss/gmflow_natif_emule.py ROUE/vsgmfss_fortuna --work $W --recettes fp32
cmake -S tools/gmfss/gmflow_attention -B build/gmflow_attention -DCMAKE_BUILD_TYPE=Release
cmake --build build/gmflow_attention --target gmflow_attention_vide
cmake -S tools/gmfss/gmflow_mlp -B build/gmflow_mlp -DCMAKE_BUILD_TYPE=Release
cmake --build build/gmflow_mlp --target gmflow_mlp_vide
build/gmflow_attention/gmflow_attention_vide $W/paire_fp32.onnx $W/attn32
build/gmflow_attention/gmflow_attention_vide $P $W/attn16
build/gmflow_mlp/gmflow_mlp_vide $W/paire_fp32.onnx $W/mlp32
build/gmflow_mlp/gmflow_mlp_vide $P $W/mlp16
python -I tools/gmfss/gmflow_attention_emule.py --fp32 $W/paire_fp32.onnx --plan32 $W/attn32 --plan16 $W/attn16 \
    --paires 1 --dump $W/attention
python -I tools/gmfss/gmflow_mlp_emule.py --fp32 $W/paire_fp32.onnx --fp16 $P --plan32 $W/mlp32 --plan16 $W/mlp16 \
    --work $W --banc $W/mlp

# 4. Bancs ABAB contre TensorRT, deux fois chacun (moteurs gardés ; le premier banc de l'attention construit
#    64 moteurs, environ une heure).
for passage in 1 2; do
  flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_mlp_banc.sh $P ~/kirari-reports/gmflow-mlp $W/mlp
  flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_attention_banc.sh $P ~/kirari-reports/gmflow-attention \
      $W/attention
done
```

Si la lecture des plans refuse la paire au flot entier du PC (export d'avant le format 2), la réexporter par
`tools/gmfss/export_gmfss.py`.

À relever pour GMFlow :
- `banc-mlp.md` et `banc-attention.md` de chaque passage, et l'écart entre les deux (au-delà de 3 %, refaire) ;
- `nvidia-smi -q -d CLOCK,TEMPERATURE` avant et après ;
- dans `trt/*.profile.json`, si TensorRT a pris un noyau d'attention fondu (`fmha`, `mha`).

La règle de décision : les deux contrôles tenus, puis le rapport de la couche entière du transformeur au quart de
720p, (attention native + 6 × MLP natif) sur la même somme sous TensorRT. **0,80 ou moins : on continue** (PR 4 et
5) ; au-dessus, TensorRT garde GMFlow.

Puis la forme en flux du GridNet (détail dans `gmfss-natif.md`, « Séance finale », étape 4) :

```sh
# 5a. Au réglage (première ouverture, étape 0) : une ligne par convolution où une forme en flux passe.
kirari play --motion=gmfss --rife=2 --start=960 $G <extrait> 2>&1 | grep -E "réglage : convolution|retenu pour"
# 5b. Le temps de chaque noyau, dans le profil de l'étape 1.
nsys stats -r cuda_gpu_kern_sum --format csv gmfss.nsys-rep | grep -E "stream_kernel|conv_kernel"
# 5c. Une convolution de la ligne 0 au détail (si rien ne se profile, ôter --nvtx et --nvtx-include).
ncu --nvtx --nvtx-include "kirari@gmfss.fusion.gridnet/" -k regex:stream_kernel -c 12 \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis -o flux \
  kirari --debug play --motion=gmfss --rife=2 --start=960 $G <extrait>
ncu --import flux.ncu-rep --page details --csv > flux.csv
```

À relever pour le GridNet :
- « flux retenu » ou « tuile gardée » pour chaque convolution ; attendu sur la bande du banc : 270 à 300 µs en
  flux contre 500 au plus en tuile sans résiduel, 400 à 450 contre 640 avec ;
- « N formes écartées » doit rester à 0 ;
- la plage `gmfss.fusion.gridnet` (étape 1) : attendu 2,3 à 2,8 ms de moins que les 21,90 ms de la fusion ;
- dans Nsight Compute : la durée, les octets lus et écrits, le débit mémoire (88 % visés), les tensor cores (85 %
  visés) et l'occupation (8 warps).

Pour un ABAB du GridNet entier contre la seule tuile, un second binaire se construit au premier parent de la
fusion de `gridnet-noyaux` dans #959 (le commit « Fusion de gridnet-noyaux », `^1`) ; il n'est utile que si le
réglage garde le flux et que le gain reste flou.

La séance se termine par la fusion de #959.

Toute modification de `src/core`, `src/gpu`, `src/kernels` ou `src/prepare` demande un nouveau paquet TensorRT
Linux. Il ne se fait que sur demande de Tatoub, comme toute release.

## 5. Point de reprise

| Branche | Commit | Contenu |
|---|---|---|
| `claude/project-thread-1t05iy` (#959) | sa tête | Le natif par défaut, le mélange, GMFlow PR 2 et 3, ce bilan, à jour de main (#962) |
| `gmfss-defaut` | `6532362a` | Le natif par défaut seul, à jour de main |
| `gmfss-melange-2` | `4f1df2bf` | Le mélange, le réglage, les plages NVTX, la doc de la fusion mesurée |
| `gridnet-noyaux` | `aa81450b` | La forme en flux du GridNet, fusionnée dans #959 |
| `gmflow-attn-mlp` | `e7e5cba0` | Les PR 2 et 3 de GMFlow auditées, fusionnées dans #959 |

| Fichier | Rôle |
|---|---|
| `src/gpu/cuda/backend/gmfss_native_net.cu`, `gmfss_pair_native_net.cu` | GridNet, FeatureNet et MetricNet natifs |
| `src/gpu/cuda/backend/gmfss_conv_plan.cu`, `.cuh` | Les formes des convolutions, le réglage au banc, les zones par durée de vie |
| `src/gpu/cuda/backend/gmfss_trial.cpp`, `gmfss_mix.*`, `native_mix.*` | L'essai au chargement et le mélange par morceau, ce dernier commun avec RIFE |
| `src/gpu/cuda/backend/gmfss_kernels.cu` | Les splats |
| `src/prepare/gmfss_native.cpp` | La coupe de la fusion et les morceaux du GridNet |
| `src/app/play/gmfss_model.cpp`, `chain_gmfss.cpp` | Le choix des graphes, la paire périmée qui cède |
| `tools/gmfss/` | `export_gmfss.py`, les émulations (`emul_mix_native.py`, `emul_pair_native.py`, `gmflow_natif_*.py`) et `plafond_gmfss.py` |
| [`gmfss-natif.md`](gmfss-natif.md), [`gmflow-natif.md`](gmflow-natif.md) | Le détail technique et les résultats |

Une copie de ce bilan est rangée dans le dossier partagé du projet, sous `chantier-natif/bilan-gmfss.md`, à côté de
la séance (`seance-pc-gmfss.md`) et des plafonds (`gmfss-plafonds.md`).
