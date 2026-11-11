# Banc de qualité des réseaux

Ce banc donne un chiffre à toute affirmation sur la qualité de SPAN, ArtCNN, RIFE 4.25, RIFE 4.22 lite et DistilDRBA :
chaque réseau livré est comparé à sa référence FP32 et à la vraie image. Il tourne sans carte graphique (ONNX
Runtime sur processeur), ici comme en CI (`.github/workflows/qualite.yml`).

Règle qu'il sert : une baisse de qualité faite pour la vitesse est un défaut, sauf mesure. Une correction de
qualité donne sa mesure avant et après avec ce banc.

## Le lancer

```sh
tools/bench/qualite.sh                                  # tout, environ 2 min sur 4 cœurs
tools/bench/qualite.sh --md rapport.md --json r.json    # avec le rapport par cas
tools/bench/qualite.sh --video episode.mkv --scenes 60  # mouvement réel d'un épisode
tools/bench/qualite.sh --seulement mouvement --reseau nouveau_rife.onnx
python3 -I tools/bench/qualite/test_banc.py             # les tests du banc, sans modèle
```

Il faut `python3` avec `numpy`, `onnx` et `onnxruntime`, `ffmpeg`, et `7z` pour le premier téléchargement des
modèles (`tools/release/fetch-models.sh`, gardés dans `~/.cache/ajp-release-models`).

## Ce qu'il mesure

| Mesure | Question | Comment |
|---|---|---|
| Fidélité | Le graphe FP16 livré rend-il l'image de sa référence FP32 ? | Même entrée dans les deux graphes ; PSNR, pire écart et 99,9e centile en codes 8 bits, ΔE_ITP |
| Vérité, interpolation | L'image créée vaut-elle la vraie ? | Le réseau reçoit les images k et k+2 (DistilDRBA aussi k+4, comme `drba_call(0,5)`), sa sortie se compare à k+1 |
| Vérité, agrandissement | L'image agrandie vaut-elle l'original ? | Un plan 1080p de `calibration.mkv` réduit de moitié (moyenne 2 × 2), agrandi ×2, comparé au plan ; le bicubique sert de repère |

La référence FP32 (`tools/bench/qualite/fp32.py`) est le graphe livré où tout tenseur passe en float32 : mêmes
poids, tout le calcul en FP32. Avec `--origine NOM=FICHIER`, le graphe FP32 d'origine la remplace (l'arrondi des
poids à la conversion entre alors dans la mesure).

Les réseaux sont nourris comme dans le lecteur : fenêtre centrée au multiple de 64 et grille du v1
(`src/kernels/spec/rife.h`), paires `pair01` et `pair12` (`src/kernels/spec/distil_drba.h`).

Le mouvement par défaut est de synthèse (`tools/bench/qualite/sources.py`) : un plan fixe glisse en fond, une
ellipse découpée dans un autre plan glisse devant lui ou tient la pose. Les déplacements sont entiers, donc la
vraie image du milieu est exacte, et le personnage découvre le fond comme un cellulo. Quatre allures : lent
(4 px entre deux entrées), moyen (12 px), rapide (32 px), pose tenue devant un fond qui glisse.

## Les seuils

`tools/bench/qualite/seuils.json` fixe par réseau le PSNR de fidélité le plus bas, le pire écart de fidélité le
plus haut et le PSNR de vérité moyen le plus bas. Le banc sort en erreur au-delà, et la CI avec lui. Ils valent
pour la configuration par défaut (1280 × 720, 8 scènes, 6 plans) ; une autre configuration se mesure sans porte.

Les seuils décrivent l'état mesuré, avec une marge (1 dB, 25 % sur l'écart, 0,3 dB sur la vérité). Ils ne
disent pas que cet état est bon : le pire écart admis de RIFE 4.25 est de 52 codes parce que c'est ce qu'il fait
aujourd'hui. Une correction qui améliore un réseau les resserre avec `--ecrire-seuils` ; les relâcher demande la
mesure du gain de vitesse qui le justifie.

## Première mesure

1280 × 720, 8 scènes de synthèse, 6 plans, ONNX Runtime 1.29 sur processeur. Écarts en codes 8 bits.

| Réseau | Fidélité (dB) | Pire cas (dB) | Pire écart | Écart p99,9 | ΔE_ITP p99,9 | Vérité FP16 (dB) | Vérité FP32 (dB) | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPAN V3.1 Sharp1 Balanced (b8f64) | 79,55 | 75,29 | 0,42 | 0,15 | 0,16 | 41,39 | 41,39 | 0,9875 |
| SPAN V3.1 Sharp1 Performance (b5f48) | 79,72 | 75,40 | 0,30 | 0,14 | 0,16 | 41,24 | 41,24 | 0,9873 |
| SPAN V3.1 Balanced (b8f64) | 79,89 | 75,63 | 0,56 | 0,14 | 0,16 | 41,80 | 41,80 | 0,9897 |
| SPAN V3.1 Performance (b5f48) | 80,02 | 75,76 | 0,41 | 0,13 | 0,16 | 41,67 | 41,67 | 0,9892 |
| RIFE 4.25 | 56,09 | 50,28 | 41,43 | 9,29 | 14,34 | 47,24 | 48,09 | 0,9958 |
| RIFE 4.22 lite | 56,29 | 50,55 | 25,26 | 9,07 | 13,86 | 47,56 | 48,48 | 0,9969 |
| DistilDRBA v2 lite | 77,51 | 73,96 | 1,19 | 0,42 | 0,60 | 35,76 | 35,76 | 0,9691 |
| Bicubique ×2 (repère de SPAN) | | | | | | 42,13 | | 0,9892 |

En 1792 × 1024 (4 scènes), l'interpolation donne : RIFE 4.25 47,94 dB en FP16 contre 49,11 en FP32, RIFE 4.22
lite 48,12 contre 49,26, DistilDRBA 37,92 contre 37,98.

Ce que ces chiffres disent :

- **SPAN en FP16 est transparent** : moins d'un code d'écart au FP32, aucune perte contre l'original.
- **Les deux RIFE perdent au FP16** : 0,9 dB contre la vraie image en 720p, 1,2 dB en 1024 lignes, et des
  pixels à 25 ou 41 codes de leur référence. C'est ce que l'audit A03-20 attribue à la grille du v1 arrondie
  en FP16 ; le banc en donne la mesure, la correction donnera la sienne.
- **DistilDRBA est 11 à 12 dB sous RIFE 4.22 lite sur ce mouvement** (35,7 contre 47,6 dB), pour un temps cinq
  fois plus court sur processeur. Il place bien le mouvement (un panoramique pur tombe au bon endroit) mais le
  rend moins net. « Plus rapide à qualité proche » n'est donc pas tenu sur ces scènes ; la mesure sur un
  épisode réel suit.
- **SPAN ne bat pas le bicubique en PSNR sur ce banc** : il renforce les traits au lieu de rendre l'original,
  et la réduction par moyenne n'est pas la dégradation qu'il a apprise. La colonne sert à comparer deux états
  de SPAN (FP16, FP8, hauteur de travail), pas SPAN au bicubique.

### ArtCNN C4F32 et C4F16 (P6.7)

Les deux modèles légers des petites cartes, mesurés le 11 novembre 2026 dans la même configuration (ONNX Runtime
1.29, processeur à 4 cœurs ; le temps n'y sert qu'à comparer les réseaux entre eux) :

| Réseau | Fidélité (dB) | Pire cas (dB) | Pire écart | Vérité FP16 (dB) | SSIM | ms par plan 960 × 540 |
|---|---:|---:|---:|---:|---:|---:|
| SPAN V3.1 Sharp1 Performance (b5f48) | 79,72 | 75,40 | 0,30 | 41,24 | 0,9873 | 706 |
| ArtCNN C4F32 | 79,82 | 75,40 | 0,44 | 45,14 | 0,9936 | 326 |
| ArtCNN C4F16 | 79,79 | 75,38 | 0,50 | 44,97 | 0,9934 | 86 |
| Bicubique ×2 | | | | 42,13 | 0,9892 | |

- **Les deux sont transparents en FP16** : l'enveloppe RGB (`tools/upscale/wrap_luma_rgb.py`) calcule en FP32,
  seules l'entrée et la sortie sont en FP16.
- **C4F16 rend presque l'image de C4F32** (0,17 dB de moins contre la vraie image) pour un quart du calcul
  (16 canaux au lieu de 32 : 86 ms contre 326 sur processeur). Sur carte graphique le rapport sera moindre
  (les petites couches remplissent mal le GPU) : c'est au banc des petites cartes de le dire
  (`tools/bench/banc_leger.sh`).
- **ArtCNN dépasse SPAN et le bicubique sur cette colonne**, ce qui ne dit pas qu'il est plus beau : il est
  entraîné à rendre l'original depuis une réduction, ce que mesure ce banc, et SPAN d'AnimeJaNai à renforcer
  les traits (voir plus haut). À l'œil, sur un épisode, la comparaison reste à faire.

### DistilDRBA, grille en FP32 (C05, A03-18)

Le graphe publié jusqu'au 10 novembre 2026 (`models-drba-1`, SHA-256 7048367e…) arrondissait en FP16 la grille
de ses dix GridSample. Le graphe livré depuis (`models-interp-1`, 40bda348…) est exporté avec `DRBA_GRID_OPS`,
qui garde cette chaîne en FP32, mêmes poids :

| réseau | fidélité dB | pire dB | pire écart | écart p99,9 | ΔE_ITP p99,9 | vérité dB | ms (processeur) |
|---|---|---|---|---|---|---|---|
| distil_drba_v2_lite d'avant, grille en FP16 | 57,01 | 51,98 | 20,10 | 7,33 | 10,69 | 35,72 | 189,0 |
| distil_drba_v2_lite livré, grille en FP32 | 77,51 | 73,96 | 1,19 | 0,42 | 0,60 | 35,76 | 180,5 |

20 dB de fidélité gagnés, sans coût. La vérité ne bouge pas : sur ce mouvement de synthèse l'écart au vrai
milieu vient du réseau, pas de son FP16. Sur l'extrait de la section « Mesure sur un épisode » (`--video`,
60 fenêtres, 1280 × 720), le graphe passe de 60,08 à 81,13 dB de fidélité, et son pire écart de 23,1 à
2,3 codes ; la comparaison à RIFE sur cet extrait est dans cette section. Le demi-flot (`_s05`) donne 77,43 dB
en synthèse et 80,78 dB sur l'extrait. `seuils.json` porte les chiffres du graphe livré : un export qui
remettrait la grille en FP16 ferait échouer la CI.

### DistilDRBA, demi-flot (C05, A09-24)

`distil_drba_v2_lite_s05.onnx` estime le mouvement à la moitié de la taille. Le lecteur le prenait dès 1440
lignes, sans chiffre. Mesuré avec `--reseau`, le flot entier d'abord, le demi-flot ensuite :

| Images | Vérité, flot entier (dB) | Vérité, demi-flot (dB) | Écart | ms entier | ms demi-flot | Temps |
|---|---:|---:|---:|---:|---:|---:|
| synthèse 1280 × 720, 8 scènes | 35,76 | 31,92 | −3,84 | 167 | 96 | −42 % |
| synthèse 1600 × 900, 8 scènes | 36,67 | 32,30 | −4,37 | 300 | 197 | −34 % |
| extrait 1280 × 720, 60 fenêtres | 24,58 | 24,68 | +0,10 | 162 | 90 | −44 % |
| extrait 1920 × 1080, 12 fenêtres | 23,87 | 23,97 | +0,10 | 366 | 215 | −41 % |
| extrait agrandi à 2560 × 1440, 8 fenêtres | 24,63 | 24,86 | +0,23 | 713 | 419 | −41 % |

Opérations (multiplications-additions des convolutions, comptées sur le graphe) : 6,5 milliards contre 1,6 à
1280 × 768, 24,8 contre 6,5 à 1440 lignes, 55,0 contre 13,7 à 2160 lignes. Le demi-flot fait quatre fois
moins de convolutions ; sur processeur le temps ne baisse que de 34 à 44 %, le reste (warps, mélange) se fait
à la taille pleine.

Ce que ces chiffres disent :

- **Sur un mouvement simple** (les scènes de synthèse : translation, zoom, où tout tient au placement), le
  demi-flot perd 3,8 à 4,4 dB, et la perte ne diminue pas quand la taille monte.
- **Sur l'extrait d'action**, les deux se valent, le demi-flot un peu devant (0,1 à 0,2 dB) : sur un grand
  mouvement, estimer à la moitié de la taille élargit ce que le réseau voit.
- Le seuil passe donc à **2160 lignes** (`kDrbaHalfFlowHeight`), le conseil de l'auteur : à 1440 lignes rien ne
  justifie de perdre 4 dB sur les panoramiques.

Limites : pas de source native en 1440 ou 2160 lignes ici, l'extrait est agrandi (ce qu'est presque tout anime
en 4K) et les scènes de synthèse s'arrêtent sous 1080 lignes ; les temps sont ceux d'ONNX Runtime sur
processeur, pas ceux de la RTX. À 3840 × 2160 le banc n'a pas tenu dans les 15 Go de la machine de mesure : le
demi-flot en 4K reste sur le conseil de l'auteur, sans chiffre d'ici.

## Mesure sur un épisode

Un extrait de 31 s d'un épisode en 1080p (743 images, surtout de l'action), avec `--video`. Le banc y prend des
fenêtres où k, k+1 et k+2 diffèrent, et compare l'image créée à k+1. PSNR contre la vraie image, graphes FP16.

| Taille, fenêtres | RIFE 4.25 | RIFE 4.22 lite | DistilDRBA v2 lite | DistilDRBA en tout FP32 |
|---|---:|---:|---:|---:|
| 1280 × 720, les 60 | 26,41 | 26,39 | 24,57 | 24,58 |
| 1280 × 720, les 13 faciles (RIFE au-dessus de 35 dB) | 43,77 | 43,54 | 36,79 | 36,85 |
| 1280 × 720, les 8 entre 28 et 35 dB | 31,38 | 31,44 | 29,71 | |
| 1280 × 720, les 39 sous 28 dB | 19,61 | 19,64 | 19,45 | |
| 1920 × 1080, les 16 | 24,78 | 24,74 | 23,66 | 23,66 |
| 1920 × 1080, les 2 faciles | 44,87 | 44,43 | 38,99 | 38,98 |

- **DistilDRBA vaut RIFE sur l'action et perd sur le mouvement simple** : égalité sous 28 dB, où tous échouent ;
  6,8 dB de moins sur les fenêtres faciles en 720p, 5,4 dB en 1080p. En moyenne 1,8 dB de moins en 720p
  (0,2 dB en médiane, il gagne 25 fenêtres sur 60) et 1,1 dB en 1080p. Les 11 à 12 dB de la synthèse sont donc
  le cas le plus défavorable, pas la moyenne d'un épisode.
- **L'écart vient du réseau, pas du FP16** : avec tout le calcul en FP32, grilles comprises, DistilDRBA ne gagne
  que 0,06 dB sur les fenêtres faciles.
- **RIFE 4.25 et RIFE 4.22 lite ne se distinguent pas** : 0,02 dB en 720p, 0,04 dB en 1080p.
- **Le FP16 des RIFE s'écarte davantage sur un épisode** : pire écart de 105 codes (4.25) et 193 codes (4.22 lite)
  en 720p, 61 et 54 en 1080p, contre 23 et 34 pour DistilDRBA ; le PSNR moyen contre la vraie image n'en bouge
  pas (26,41 contre 26,45 en FP32).
- Temps sur processeur : DistilDRBA 140 ms par image créée en 720p contre 776 (4.22 lite) et 906 (4.25) ; 288 ms
  en 1080p contre 1861 et 1953.

Ce que cette mesure ne dit pas : un seul extrait ; **les poses tenues et les coupes ne sont pas mesurées** (une
fenêtre dont le dessin du milieu est tenu est écartée, et une coupe entre k et k+2 n'est pas détectée). Les poses
tenues sont l'argument de DistilDRBA.

## Chroma d'une source 4:4:4

`tools/bench/qualite/chroma444.py` mesure ce que perd une source 4:4:4 réduite en 4:2:0 avant SPAN (audits A15-07
et A49-13). Même échelle ½ : la moitié d'un plan 1080p de `calibration.mkv` tient lieu de source 4:4:4 10 bits.
La réduction est celle de swscale lui-même, la remontée Spline36 celle de l'ingestion.

```sh
python3 -I tools/bench/qualite/chroma444.py --modeles ~/.cache/ajp-release-models/modeles --md chroma444.md
```

Six plans, SPAN b8f64 FP16, ONNX Runtime 1.29 sur processeur.

| Mesure | Chemin | PSNR (dB) | Pire plan (dB) | SSIM | ΔE_ITP p99,9 |
|---|---|---:|---:|---:|---:|
| ingestion, contre la source | 4:2:0 d'avant | 44,44 | 35,74 | 1,0000 | 22,76 |
| ingestion, contre la source | 4:2:0 spline | 47,39 | 39,83 | 1,0000 | 12,72 |
| ingestion, contre la source | 4:4:4 | identique | identique | 1,0000 | 0,00 |
| après SPAN ×2, contre le plan | 4:2:0 d'avant | 40,17 | 34,90 | 0,9869 | 31,90 |
| après SPAN ×2, contre le plan | 4:2:0 spline | 41,01 | 36,18 | 0,9874 | 27,73 |
| après SPAN ×2, contre le plan | 4:4:4 | 41,37 | 36,30 | 0,9874 | 27,09 |

- « 4:2:0 d'avant » est le lecteur avant la correction : swscale réduisait la chroma en bilinéaire et la posait
  au centre en largeur, alors que l'image l'annonçait à gauche et que l'ingestion la lisait là.
- « 4:2:0 spline » est ce que garde l'ingestion TensorRT, qui ne lit pas le 4:4:4 en mémoire hôte : réduction
  en spline, arrondi exact, chroma posée à gauche. 0,84 dB de mieux après SPAN.
- « 4:4:4 » est le chemin Vulkan (et Core ML) après la correction : la chroma entière arrive à l'ingestion.
  1,20 dB de mieux après SPAN, et l'ingestion rend la source au code près.

Le SSIM se lit sur la luma, que ces chemins ne touchent pas avant SPAN.

## Hauteur de travail et modèle de SPAN

`tools/bench/qualite/travail.py` mesure ce que coûte une source réduite avant SPAN, et ce que gagne le modèle
b8f64 sur le b5f48 (audits A49-01 et A49-02). Il joue tout à l'échelle ½, comme la mesure « vérité,
agrandissement » : un plan 1080p de `calibration.mkv` est la vérité, sa moitié tient lieu de source 1080p.
Chaque chaîne réduit la source à une hauteur de travail (Hermite), l'agrandit par SPAN, puis la ramène à l'écran
comme libplacebo par défaut (Lanczos vers le haut, Hermite vers le bas). Les hauteurs sont dites à l'échelle du
lecteur.

```sh
python3 -I tools/bench/qualite/travail.py --modeles ~/.cache/ajp-release-models/modeles --md travail.md
```

Six plans, modèles Sharp1 en FP16, ONNX Runtime 1.29 sur processeur. PSNR contre la vérité de l'écran.

| Écran | Travail | SPAN | PSNR (dB) | Pire plan (dB) | SSIM | ΔE_ITP p99,9 |
|---|---|---|---:|---:|---:|---:|
| 2160p | 1080p | b8f64 | 41,39 | 36,30 | 0,9875 | 27,09 |
| 2160p | 1080p | b5f48 | 41,24 | 36,52 | 0,9873 | 28,34 |
| 2160p | 720p | b8f64 | 39,09 | 33,31 | 0,9798 | 37,53 |
| 2160p | 720p | b5f48 | 38,82 | 33,16 | 0,9789 | 38,68 |
| 2160p | 540p | b8f64 | 36,59 | 31,15 | 0,9655 | 50,21 |
| 2160p | 540p | b5f48 | 36,26 | 31,02 | 0,9646 | 52,34 |
| 1440p | 1080p | b8f64 | 45,28 | 40,25 | 0,9959 | 16,34 |
| 1440p | 1080p | b5f48 | 44,99 | 40,46 | 0,9956 | 17,39 |
| 1440p | 720p | b8f64 | 41,94 | 35,94 | 0,9896 | 26,35 |
| 1440p | 720p | b5f48 | 41,68 | 35,91 | 0,9893 | 27,38 |
| 1440p | 540p | b8f64 | 38,72 | 33,21 | 0,9800 | 39,39 |
| 1440p | 540p | b5f48 | 38,32 | 32,49 | 0,9797 | 41,89 |
| 1080p | 1080p | b8f64 | 45,56 | 42,57 | 0,9977 | 14,39 |
| 1080p | 1080p | b5f48 | 45,38 | 42,41 | 0,9974 | 14,73 |
| 1080p | 720p | b8f64 | 40,49 | 35,67 | 0,9903 | 30,33 |
| 1080p | 720p | b5f48 | 40,21 | 35,47 | 0,9897 | 31,54 |
| 1080p | 540p | b8f64 | 38,16 | 32,52 | 0,9812 | 40,40 |
| 1080p | 540p | b5f48 | 37,75 | 31,80 | 0,9808 | 43,17 |

Ce que ces chiffres disent :

- **Réduire la source 1080p avant SPAN coûte cher.** Garder la 1080p plutôt que travailler en 720p rend 2,3 dB
  sur un écran 4K, 3,3 dB sur un écran 1440p et 5,1 dB sur un écran 1080p. Contre un travail en 540p : 4,8, 6,6
  et 7,4 dB. C'est la raison de la source gardée aux niveaux Qualité et Ultra, et des essais de la calibration
  jusqu'à 1080p.
- **Le b8f64 gagne peu sur le b5f48** : 0,15 à 0,41 dB, davantage quand la source est réduite, et un ΔE_ITP au
  p99,9 plus bas partout. Sur un des six plans à 1080p, le b5f48 est devant de 0,2 dB.
- Sans SPAN, la source portée à l'écran par libplacebo donne 42,74 dB (4K) et 48,70 dB (1440p) : au-dessus de
  SPAN, pour la raison dite plus haut (SPAN renforce les traits, la moyenne 2 × 2 n'est pas la dégradation qu'il
  a apprise). Ce banc classe des chaînes avec SPAN entre elles ; il ne dit pas s'il faut SPAN.

### Sur un vrai épisode

La même mesure sur cinq plans d'un extrait réel de 30 s (H.264 1080p, 23,976 i/s, hors du dépôt : contenu
protégé), par `--calibration extrait.mkv --plans 5` :

| Écran | Travail | b8f64 (dB) | b5f48 (dB) | Sans SPAN (dB) |
|---|---|---:|---:|---:|
| 2160p | 1080p | 43,36 | 43,34 | 46,35 |
| 2160p | 720p | 42,44 | 42,33 | |
| 2160p | 540p | 39,75 | 39,48 | |
| 1440p | 1080p | 47,51 | 47,47 | 52,45 |
| 1440p | 720p | 44,75 | 44,83 | |
| 1440p | 540p | 41,86 | 41,57 | |
| 1080p | 1080p | 47,81 | 47,86 | |
| 1080p | 720p | 43,37 | 43,29 | |
| 1080p | 540p | 41,42 | 41,12 | |

- Garder la 1080p plutôt que travailler en 720p rend 0,9 dB sur un écran 4K, 2,8 dB en 1440p et 4,4 dB en
  1080p ; contre 540p, 3,6, 5,7 et 6,4 dB. Le sens est celui de la séquence de calibration, l'écart en 4K est
  plus petit (0,9 dB au lieu de 2,3).
- Le b8f64 et le b5f48 sont à égalité quand la source est gardée (−0,05 à +0,04 dB) ; le b8f64 ne gagne que sur
  une source réduite à 540p (0,3 dB). Sur cet extrait, le b8f64 des niveaux Qualité et Ultra ne s'achète donc
  par aucun gain mesurable en PSNR : son temps sur la carte décidera (A49-01).
- Sans SPAN, la source portée à l'écran reste devant (3,0 dB en 4K), avec la même réserve que plus haut.

Le temps de ces chaînes ne se mesure que sur la carte : voir `docs/audits/A-CONFIRMER.md` (A49-01, A49-02).

## Limites

- Sur processeur, ONNX Runtime range les tenseurs FP16 en demi-précision mais fait les sommes de chaque couche
  en FP32. La fidélité mesurée est un plancher : un moteur qui somme aussi en FP16 perd davantage.
- Le banc mesure les graphes, pas le lecteur : ni les moteurs TensorRT, ni le moteur Vulkan, ni les chaînes
  fusionnées, ni le warp après SPAN, ni la hauteur de travail.
- La référence FP32 a les poids du fichier livré, déjà arrondis en FP16.
- Les temps sont ceux du processeur : ils classent à peine les réseaux et ne valent pas ceux du lecteur.
- Le mouvement de synthèse est une translation de calques : ni déformation, ni rotation, ni animation « sur
  deux ».

## Ce qui demande la RTX

| Mesure | Pourquoi la RTX |
|---|---|
| Sortie des moteurs TensorRT FP16 contre la référence FP32 (A49-16, A49-14) | TensorRT ne tourne que sur NVIDIA |
| SPAN FP8 contre FP16 et FP32 (cran Performance) | noyaux CUDA de la RTX 40 |
| RIFE et DistilDRBA fusionnés contre le réseau entier, variante mouvement (A03-02, A03-03) | noyaux CUDA |
| Temps par image créée de chaque réseau, FP16 contre FP32 | seul le temps de la carte compte |

Le même banc tourne sur la carte avec `--provider cuda` (paquet `onnxruntime-gpu`), ce qui donne le FP16 d'un
vrai GPU, mais pas celui de TensorRT.
