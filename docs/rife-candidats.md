# RIFE 4.25 lite et 4.26 face aux livrés

Question de Tatoub (novembre 2026) : sans se fier aux rumeurs, RIFE 4.25 lite vaut-il mieux que 4.22 lite, et 4.26
mieux que 4.25, en temps par image et en qualité ? Ce document donne ce qui se mesure sans carte graphique, puis le
protocole ABAB à passer sur la RTX. Sur ces mesures, Tatoub a choisi les niveaux du menu (docs/interpolation.md) :
Très rapide 4.25 lite, Rapide 4.22 lite, Équilibré 4.26 (le défaut sur une RTX) ; 4.25 et DistilDRBA en réglages
avancés. Le protocole ABAB reste celui qui le vérifie sur la RTX.

## Les fichiers

`tools/rife/candidats.sh <dossier>` les télécharge et les vérifie par SHA-256 :

| Fichier | Origine | Conversion FP16 |
|---|---|---|
| `rife_v4.26.onnx` | archive `models-rife-fp16-1` d'animejanai-inference, celle des RIFE livrés | la même que `rife_v4.25.onnx` livré : seuls les poids diffèrent |
| `rife_v4.25_lite.onnx` | export officiel de vs-mlrt, release `external-models`, dossier `rife/` (11 plans, FP32) | `tools/rife/convert_rife_fp16.py` (Resize en FP32 : TensorRT 11.3 sort des NaN sur ce réseau sinon) |

Aucun code d'un autre lecteur n'est repris (TheAnimeScripter est sous AGPL 3, ajp sous MPL 2) : ce sont les fichiers
des auteurs des exports. 4.25 lite se remplit au multiple de 128 (`rife_padding_multiple`), les autres au multiple
de 64.

## Structure et RIFE fusionné

| Modèle | Blocs | Réduction d'entrée des blocs | Calcul à 960 × 576 (travail 540) | à 1920 × 1088 |
|---|---|---|---:|---:|
| 4.22 lite | 4 | 8, 4, 2, 1 | 14,0 GMAC | 53,0 GMAC |
| 4.25 lite | 5 | 32, 16, 8, 4, 1 | 7,0 GMAC (rempli en 1024 × 640) | 23,7 GMAC (1920 × 1152) |
| 4.25 | 5 | 8, 8, 4, 2, 1 | 13,5 GMAC | 51,1 GMAC |
| 4.26 | 5 | 16, 8, 4, 2, 1 | 12,3 GMAC | 46,4 GMAC |

Multiplications-additions des convolutions, comptées sur le graphe (encodeur compris). Ce n'est pas un temps :
le temps du RIFE fusionné compte aussi les noyaux de colle en pleine résolution, que les quatre partagent. C'est
ce que la RTX doit départager.

Avant ce travail, la découpe (`prepare/rife_split.h`) refusait les deux candidats, et la lecture gardait le RIFE
entier : 4.26 pour son premier bloc au 1/16, 4.25 lite pour son 1/32 et sa colle autrement rangée (les deux images
warpées d'un seul GridSample : image puis encodage de A, image puis encodage de B, au lieu des deux images puis des
deux encodages). La découpe prend maintenant les réductions jusqu'à 32, ignore les Cast que pose
`convert_rife_fp16.py`, et pour 4.25 lite permute les poids de la première convolution de chaque morceau de bloc :
les noyaux CUDA et Vulkan gardent leur disposition et ne changent pas.

Vérifié sur processeur (ONNX Runtime) en enchaînant les morceaux dérivés avec les formules de la référence
(`kernels/ref/rife_fused.cpp`), face au réseau entier, 384 × 256, panoramique de 12 × 6 pixels sur un plan de
Frieren : 4.26 57,5 dB, 4.25 lite 56,7 dB, 4.25 57,4 dB (pire écart 10, 11 et 9 codes). Le même écart que les RIFE
livrés : il vient de la grille FP32 des noyaux (A03-20) et de l'ordre des sommes.

## Qualité sur processeur

`tools/bench/qualite.sh --seulement mouvement` (docs/banc-qualite.md) : l'image du milieu est retirée, le réseau la
refait depuis ses voisines, et la sortie se compare à la vraie. 1280 × 720, scènes de synthèse tirées des plans de
`calibration.mkv` (un fond glisse, un personnage découpé glisse devant ou tient la pose), graphes FP16 sous ONNX
Runtime, références FP32 des exports de vs-mlrt.

16 scènes (4 allures × 4 tirages), moyenne des PSNR et des SSIM contre la vraie image :

| Réseau | Vérité FP16 (dB) | Vérité FP32 (dB) | SSIM | Fidélité au FP32 (dB) |
|---|---:|---:|---:|---:|
| RIFE 4.22 lite (livré) | 48,20 | 49,23 | 0,9972 | 56,84 |
| RIFE 4.25 lite | 45,33 | 45,97 | 0,9940 | 56,34 |
| RIFE 4.25 (livré) | 47,66 | 48,70 | 0,9958 | 56,50 |
| RIFE 4.26 | 47,93 | 49,10 | 0,9961 | 56,27 |

Par paire, scène par scène (écart de B contre A en PSNR, « gagne » au-delà de 0,05 dB) :

| B contre A | Moyenne | Médiane | Pire, meilleur | Gagne, perd | Lent | Moyen | Rapide | Pose tenue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.25 lite contre 4.22 lite | −2,87 dB | −2,44 | −9,78, −0,42 | 0, 16 | −2,57 | −2,52 | −2,26 | −4,13 |
| 4.26 contre 4.25 | +0,27 dB | +0,02 | −0,89, +1,55 | 7, 6 | −0,51 | +0,08 | +0,77 | +0,75 |

- **4.25 lite perd partout face à 4.22 lite** : les 16 scènes, 2,9 dB en moyenne, 4,1 dB sur les poses tenues, et
  déjà en FP32 (−3,3 dB) : c'est le réseau, pas la conversion. En échange, deux fois moins de calcul.
- **4.26 et 4.25 se valent** : +0,27 dB en moyenne, médiane nulle, 7 scènes contre 6. 4.26 gagne un peu sur le
  rapide et les poses, perd un peu sur le lent, pour 9 % de calcul en moins.
- Les fidélités au FP32 (56 à 57 dB) sont celles des RIFE livrés : la conversion des candidats ne coûte rien de
  plus.

Limites : des scènes de synthèse (déplacements entiers, un seul plan de fond par scène), pas un épisode ; les
temps d'ONNX
Runtime sur processeur ne disent rien de la RTX et ne sont pas repris. Rapports complets : dossier
`bench/rife-candidats/` des fichiers du projet.

## Protocole ABAB sur la RTX

Sur le poste NVIDIA, GPU sans autre charge, binaire livré (`linux-release` ou `windows-release` ; `linux-prefix`
est en Debug et ses temps ne valent rien). Les fichiers de Re:Zero restent sur le poste, hors du dépôt.

1. Les candidats : `tools/rife/candidats.sh ~/ajp-rife-candidats`. Les livrés sont dans le dossier `rife/` du
   lecteur (ou `tools/release/fetch-models.sh`).
2. Les extraits, `~/ajp-reports/rife/extraits.tsv` (fichier, tabulation, début en secondes) : Re:Zero 18 à 16:00
   (`960`, l'extrait de docs/rife-fusionne.md), un panoramique lent et une scène d'action du même épisode.
3. Temps par image, 4.22 lite (A) contre 4.25 lite (B), cinq tours (banc_resume ne conclut qu'à partir de cinq) :

   ```sh
   L=~/ajp-rife-candidats
   flock ~/ajp-reports/gpu-bench.lock tools/bench/banc.sh --tours=5 --secondes=30 \
     --a="--rife=3 --one-lane --rife-model=rife_v4.22_lite.onnx" \
     --b="--rife=3 --one-lane --rife-model=$L/rife_v4.25_lite.onnx" \
     --extraits=~/ajp-reports/rife/extraits.tsv --sortie=~/ajp-reports/rife/lite
   python3 -I tools/bench/banc_gpu_resume.py ~/ajp-reports/rife/lite
   ```

   puis 4.25 (A) contre 4.26 (B), `--rife-model=rife_v4.25.onnx` et `--rife-model=$L/rife_v4.26.onnx`, sortie
   `~/ajp-reports/rife/426`. `banc_gpu_resume.py` donne la médiane du moteur RIFE de chaque passage (avec
   `--one-lane`, son coût propre) et l'écart de B contre A ; `resume.txt` (banc_resume) donne les images en retard
   et manquées. Le premier passage de chaque variante construit ses moteurs (une minute environ) : le préchauffage
   de banc.sh l'absorbe. Le journal de chaque passage dit si la chaîne fusionnée a tourné (« RIFE fusionné ») ;
   un repli sur le RIFE entier fausse la comparaison et se signale.
4. Qualité sur l'épisode, PSNR et SSIM des images retirées puis refaites, en CUDA pour aller vite :

   ```sh
   tools/bench/qualite.sh --seulement mouvement --provider cuda --video ReZero-18.mkv --scenes 60 \
     --taille 1920x1080 --reseau $L/rife_v4.25_lite.onnx --reseau $L/rife_v4.26.onnx --md ~/ajp-reports/rife/qualite.md
   ```

   Le banc ne garde que des fenêtres où k, k+1 et k+2 diffèrent ; poses tenues et coupes n'y sont pas.
5. À l'œil, sur Re:Zero, après les chiffres : jouer chaque paire A, B, A, B sur le panoramique et sur l'action
   (`ajp play ReZero-18.mkv --start=960 --rife-model=…`), plein écran, en regardant les traits fins et les cheveux
   qui bougent, le texte et les bords de l'image (fantômes, traits dédoublés, cassures), les poses tenues.

Ce qui ferait revenir sur les niveaux : 4.26 nettement plus lent que 4.25 (banc_resume : « gain acquis » en
faveur de 4.25), ou moins bon au banc sur l'épisode ou à l'œil ; 4.25 lite sans gain de temps sur 4.22 lite qui
paie ses 2,9 dB. Les deux sont livrés depuis ce choix (`tools/release/fetch-models.sh`) : `--rife-model=` les
trouve par leur nom dans le dossier `rife/` du lecteur, `candidats.sh` ne sert plus qu'hors d'un paquet.
