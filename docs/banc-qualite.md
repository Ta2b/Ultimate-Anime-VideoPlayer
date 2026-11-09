# Banc de qualité des réseaux

Ce banc donne un chiffre à toute affirmation sur la qualité de SPAN, RIFE 4.25, RIFE 4.22 lite et DistilDRBA :
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
| DistilDRBA v2 lite | 57,01 | 51,98 | 20,10 | 7,33 | 10,69 | 35,72 | 35,76 | 0,9688 |
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
  rend moins net. « Plus rapide à qualité proche » n'est donc pas tenu sur ces scènes ; un épisode réel
  (`--video`), où ses poses tenues comptent, reste à mesurer avant de conclure.
- **SPAN ne bat pas le bicubique en PSNR sur ce banc** : il renforce les traits au lieu de rendre l'original,
  et la réduction par moyenne n'est pas la dégradation qu'il a apprise. La colonne sert à comparer deux états
  de SPAN (FP16, FP8, hauteur de travail), pas SPAN au bicubique.

### DistilDRBA, grille en FP32 (C05, A03-18)

Le graphe livré reconverti par `tools/rife/convert_rife_fp16.py` avec `DRBA_GRID_OPS` (la chaîne de la grille
des dix GridSample en FP32), mêmes poids, mesuré avec `--reseau` :

| réseau | fidélité dB | pire dB | pire écart | écart p99,9 | ΔE_ITP p99,9 | vérité dB | ms (processeur) |
|---|---|---|---|---|---|---|---|
| distil_drba_v2_lite (livré) | 57,01 | 51,98 | 20,10 | 7,33 | 10,69 | 35,72 | 189,0 |
| le même, grille en FP32 | 77,51 | 73,96 | 1,19 | 0,42 | 0,60 | 35,76 | 180,5 |

20 dB de fidélité gagnés, sans coût. La vérité ne bouge pas : sur ce mouvement de synthèse l'écart au vrai
milieu vient du réseau, pas de son FP16. `seuils.json` garde les chiffres du fichier livré tant que le graphe
réexporté n'est pas publié ; à ce moment, `--ecrire-seuils`.

### Sur un épisode réel (C05, décision 8)

`--video` sur un extrait de 30 s en 1080p (Re:Zero saison 4, épisode 18, de 17 min 10 à 17 min 40 ; fourni par
Tatoub, hors du dépôt), 60 fenêtres, 1280 × 720 :

| réseau | fidélité dB | pire dB | pire écart | vérité dB | SSIM | ms (processeur) |
|---|---|---|---|---|---|---|
| rife_v4.25 | 58,35 | 45,48 | 105,2 | 26,41 | 0,8322 | 1228 |
| rife_v4.22_lite | 58,68 | 48,81 | 193,0 | 26,39 | 0,8345 | 1379 |
| distil_drba_v2_lite (livré) | 60,08 | 54,22 | 23,1 | 24,57 | 0,8152 | 241 |
| le même, grille en FP32 | 81,13 | 71,12 | 2,3 | 24,58 | 0,8150 | 374 |

- **L'écart de 11 à 12 dB du mouvement de synthèse ne se retrouve pas.** En moyenne DistilDRBA est 1,8 dB sous
  RIFE 4.22 lite ; l'écart médian par fenêtre est de 0,23 dB et DistilDRBA est devant sur 25 fenêtres sur 60.
- L'écart dépend du plan. Là où RIFE dépasse 40 dB (10 fenêtres, mouvement lent et propre), DistilDRBA est à
  38,4 dB contre 45,6 (SSIM 0,980 contre 0,994) : il rend ces plans moins nets, comme sur la synthèse. Entre 20
  et 30 dB (21 fenêtres), les deux se valent (24,1 contre 24,0 dB). Sous 20 dB (19 fenêtres : coupes, dessin
  « sur deux », mouvement que personne ne retrouve), aucun des deux n'est juste et la mesure ne les départage
  pas.
- La grille en FP32 donne 21 dB de fidélité à DistilDRBA ici aussi, et son pire écart tombe de 23 à 2 codes.
- Les RIFE livrés ont un pire écart de 105 et 193 codes sur 255 entre leur FP16 et leur FP32 : sur un
  mouvement fort, l'arrondi fait basculer une zone entière (le PSNR moyen reste à 58 dB). La cause n'est pas
  cherchée ici ; les deux graphes reçoivent la même grille, ce n'est donc pas son arrondi d'entrée (A03-20).
- Un seul extrait de 30 s : assez pour écarter « 11 à 12 dB », pas pour classer les deux réseaux. Le temps est
  celui du processeur.

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
