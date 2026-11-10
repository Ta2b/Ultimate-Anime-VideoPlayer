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
