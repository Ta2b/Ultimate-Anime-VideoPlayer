# Les tuiles qui bougent

Novembre 2026, phase 3 du plan du flux optique matériel NVIDIA (suite du lot A, « Sauter les images tenues »).
Quand une image n'est pas reprise entière, SPAN ne recalcule que les tuiles qui ont changé ; le reste de sa
sortie reprend celle de l'image d'avant, décalée telle quelle sur un panoramique d'un nombre entier de pas.
Aucun modèle nouveau : SPAN tel quel, la carte de la scène (`docs/carte-de-la-scene.md`) pour décider.

## Ce qu'il faut pour que ça tourne

- **SPAN FP16 natif** (`--span-native`, `docs/span-native16.md`) : nos noyaux savent sauter une tuile d'un bloc
  entier. TensorRT (forme fixe) et le corps FP8 CUTLASS ne calculent que l'image entière : leur grille est vide
  (`gpu::Engine::span_tile_grid`) et rien ne change pour eux.
- **La carte de la scène** (`--scene-map`, RTX 30 et après) : sans elle, l'image entière se calcule, image pour
  image comme avant.
- **SPAN avant le warp** (RIFE ou DistilDRBA en variante mouvement, `warp_after_span`) : la carte de la paire
  (N − 1, N) arrive avant le SPAN de N. En SPAN seul, elle n'arrive qu'à l'image suivante : pas de tuiles. Le
  chemin SPAN après RIFE sait faire les tuiles sur place (sans décalage) quand une carte est là.
- **En lecture** : jamais à l'export (la carte arrive quand elle peut ; l'export rejoue la chaîne au bit
  près), ni avec TemporalFix (la carte ne voit pas sa sortie), ni avec « Lisser les dessins tenus » (SPAN fait
  en avance), ni avec le grade avant le warp (la sortie d'avant est gradée sur place).

`--no-span-tiles` (avec `--debug`) les éteint pour l'ABAB. La ligne de fin du journal dit ce qu'elles ont
fait : « SPAN par tuiles : t images sur n calculées (p %), x % des tuiles recalculées (y % avec le halo), d
décalées sur un panoramique ; images entières : … sans carte, … région trop large, … refus du moteur ».

## Comment

- **Grille** (`kernels/spec/span_tiles.h`) : la tuile des noyaux, 8 × 32 pixels de demi-résolution, soit 64 × 16
  pixels de travail. Le masque d'un lancement tient 4 096 tuiles (de quoi couvrir 2560 × 1080) ; au-delà,
  l'image entière.
- **Halo** : le champ réceptif calculé sur les couches lancées, un pixel de demi-résolution par 3×3 sur le plus
  long chemin (conv_1, trois par bloc, to_pixel) : 26 pour un SPAN à huit blocs, soit une colonne et quatre
  lignes de tuiles.
- **Exactitude** : toutes les couches se calculent sur le cœur dilaté du halo. Hors de cette région, les
  tampons intermédiaires tiennent n'importe quoi, mais une valeur fausse n'avance que d'un rayon par couche :
  elle n'atteint jamais le cœur. Seul le cœur est écrit (`to_pixel` masqué). Le calcul des tuiles est donc
  celui d'un calcul complet, au flottant près.
- **Décision** (`app/play/upscale_tiles.{h,cpp}`) : une tuile se recalcule si une tuile de la carte dessous a
  changé (le signal `changed` de la carte, qui ignore le grain), ainsi que là où son contenu va ; si elle a
  dérivé de plus d'un demi-pixel de travail depuis son dernier calcul (mouvements lents additionnés) ; si
  elle a été reprise 24 fois de suite. L'image entière se recalcule à chaque coupe, fondu, saut, image
  abîmée, toutes les 120 images, et quand la région dilatée dépasse 70 % de l'image.
- **Décalage** : caméra ajustée sans zoom ni rotation, résidu sous un demi-pixel, translation à moins de 1/8 de
  pixel d'un multiple de 2 pixels de travail (le pas du dégroupage). La sortie d'avant (l'autre tampon de
  SPAN) se copie décalée dans la nouvelle ; ne se reprend que le décor sûr qui suit, la bande qui entre se
  recalcule, la part fractionnaire va dans la dérive.
- **Moteur** (`gpu/cuda/backend/span_fp8_tiles.cpp`) : `next_span_tiles` avant l'appel, puis l'appel lance le
  corps masqué en direct (pas le graphe de l'image entière) ; les masques passent par une mémoire épinglée
  tournante. Le chemin qui écrit l'anneau depuis SPAN abandonne les tuiles.

ABI du backend : 48 (`Engine::span_tile_grid`, `Engine::next_span_tiles`). Nouvelle empreinte TensorRT,
nouveau paquet Linux.

## Porte (plan : 48 dB ou plus d'un calcul complet, et l'outil 0.1 sans hausse)

À mesurer sur CachyOS, binaire `linux-release`, à la séance de test commune de la fin (consigne de Tatoub du 19 novembre : pas de banc par fonction), avec la carte de la scène et le SPAN natif :

1. **Justesse** : `--debug --span-native --scene-map --span-tiles-check` sur l'extrait unique de 60 s
   (`rezero-s4e18-1710.mkv`), jamais des épisodes entiers ; la ligne « SPAN par tuiles » du journal donne le
   pire bloc 16 × 16 et le plus grand écart moyen contre un calcul complet. Porte : pire bloc sous 1 niveau
   8 bits (environ 48 dB), écart moyen sous 0,05 niveau.
2. **Stabilité** : `tools/bench/qualite/scintillement.py` sur 20 s du même extrait, avec les tuiles face à
   `--no-span-tiles` : pas plus de scintillement avec les tuiles (porte `--porte`).
3. **Gain** : la ligne « SPAN par tuiles » et le temps GPU de SPAN (`--stats`) sur le même extrait.

## État

Écrit sans carte ni GPU (lavapipe ne fait pas tourner nos noyaux CUDA) ; jamais lancé. Le test
`tests/unit/app/upscale_tiles_test.cpp` couvre la dilatation, les masques et la décision sur des cartes
synthétiques.
