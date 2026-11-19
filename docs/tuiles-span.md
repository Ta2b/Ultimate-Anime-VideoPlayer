# Les tuiles qui bougent

Novembre 2026, phase 3 du plan du flux optique matériel NVIDIA (suite du lot A, « Sauter les images tenues »).
Quand une image n'est pas reprise entière, SPAN ne recalcule que les tuiles qui ont changé ; le reste de sa
sortie reprend celle de l'image d'avant, décalée telle quelle sur un panoramique d'un nombre entier de pas.
Aucun modèle nouveau : SPAN tel quel, la carte de la scène (`docs/carte-de-la-scene.md`) pour décider.

## Ce qu'il faut pour que ça tourne

- **SPAN natif** (le défaut sur RTX 40 et 50, `docs/span-native16.md`) : nos noyaux savent sauter une tuile
  d'un bloc entier, à sommes f16 (Équilibre, et repli du cran Performance) comme f32 (Qualité). TensorRT
  (forme fixe) et le corps FP8 CUTLASS ne calculent que l'image entière : leur grille est vide
  (`gpu::Engine::span_tile_grid`) et rien ne change pour eux. Avec la carte ou `--span-tiles-check` et un de
  ces moteurs, le journal le dit au démarrage (« ce moteur ne calcule que l'image entière »).
- **La carte de la scène** (`--scene-map`, RTX 30 et après) : sans elle, l'image entière se calcule, image pour
  image comme avant.
- **La carte à l'heure** : avec RIFE ou DistilDRBA en variante mouvement (`warp_after_span`), la carte de la
  paire (N − 1, N), lancée pendant l'image N − 1 (`look_ahead`, `drba_next` sous DistilDRBA), arrive avant le
  SPAN de N. En SPAN seul aussi depuis le lot A (#901) : la suivante est ingérée et sa paire lancée en avance
  (`look_ahead`). Le chemin SPAN après RIFE et celui de SPAN seul font les tuiles sur place (sans décalage).
- **En lecture** : jamais à l'export (la carte arrive quand elle peut ; l'export rejoue la chaîne au bit
  près), ni avec TemporalFix (la carte ne voit pas sa sortie), ni avec « Lisser les dessins tenus » (SPAN fait
  en avance), ni avec la plaque de décor.
- **Sous le grade avant le warp** (le cas du PC : SPAN avant le warp en 4K, grade HDR anime) : le grade
  réécrit sur place le tampon de SPAN que le warp lit. Juste avant, la sortie de SPAN telle quelle est copiée
  (`SpanReuse::ungraded`, une copie exacte de l'image de sortie, environ 0,2 ms en 4K sur une 4070 Ti) et
  l'image suivante part de cette copie. La copie ne se fait que si les tuiles peuvent servir (moteur à
  grille, carte, chaîne en lecture) et que le grade a écrit l'image d'avant : un grader qui ne grade pas ne
  coûte rien. Sans elle (refus de place), l'image se calcule entière. Jusqu'au 19 novembre, rien n'était
  copié : sur le PC, aucune image ne passait par les tuiles (passage du 19 novembre, Spy×Family).

`--no-span-tiles` et `--span-tiles-check` (avec `--debug`, options d'essai) : l'ABAB et le contrôle. La ligne de
fin du journal sort dès que les tuiles sont actives, même sans aucune image par tuiles : « SPAN par tuiles : t
images sur n calculées (p %), x % des tuiles recalculées (y % avec le halo), d décalées sur un panoramique ;
images entières : … ». Chaque image entière y compte sous une seule raison (seules les raisons non nulles
sont écrites) : sans carte, région trop large, refus du moteur, sortie d'avant gradée sur place (pas de
copie), sortie d'avant écrasée (image créée par-dessus, source sautée), chaîne sans tuiles (export, YUV,
TemporalFix, Smooth, plaque), début, saut ou image abîmée, entrée de SPAN changée, rafraîchissement, coupe ou
fondu, caméra sans décalage entier.

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
- **Décision** (`app/play/upscale_tiles.{h,cpp}`) : ce qui a changé, c'est une tuile dont une tuile de la
  carte dessous a changé (le signal `changed` de la carte, qui ignore le grain), ainsi que là où son contenu
  va ; une tuile qui a dérivé de plus d'un demi-pixel de travail depuis son dernier calcul (mouvements lents
  additionnés), ou glissé de plus d'un code de luminosité (`SceneTile::shift`, signé : le grain s'y annule,
  une lueur ou un fondu local s'additionne) ; une tuile reprise 24 fois de suite. Le **cœur** recalculé est
  ce qui a changé dilaté de la portée (une sortie voit l'entrée jusqu'à 26 pixels de demi-résolution autour
  d'elle), plus le rayon du débandage (16 pixels de travail) quand l'entrée est débandée ; la **région**
  calculée est le cœur dilaté encore de la portée. L'image entière se recalcule à chaque coupe, fondu, saut,
  image abîmée, toutes les 120 images, quand la recette de l'entrée de SPAN change (débandage, filtre de
  réduction) et quand la région dépasse 70 % de l'image.
- **Filet sous la carte** : la carte juge des tuiles de 32 × 32 pixels de travail sur une image réduite de
  moitié, et ne dit « a changé » qu'à 6 codes d'écart moyen ou sur un vecteur net ; une petite bouche ou un
  clignement dans un plan large y passent. L'écart de chaque bloc 16 × 16 de la mesure de la paire (celle des
  dessins tenus, relue par `pair_diff` avec son pire bloc, `gpu::PairScores::blocks`) les voit : un bloc au-delà
  de 2 niveaux (`SpanTiles::kBlockChange`) recalcule les tuiles qu'il couvre. Le grain n'y arrive pas
  (écart moyen d'environ 1,13 σ par bloc ; les paires sans mouvement de Spy×Family 13 BD restent sous un
  niveau au pire bloc), une bouche si (une cinquantaine de pixels qui changent de 80 codes : 15 niveaux). Pas
  sous un décalage : la mesure compare les mêmes pixels, pas le contenu suivi. La ligne de fin dit combien de
  tuiles le filet a ajoutées, sur combien d'images : beaucoup, c'est que la carte en laisse passer.
- **Décalage** : caméra ajustée sans zoom ni rotation, résidu sous un demi-pixel, translation à moins de 1/8 de
  pixel d'un multiple de 2 pixels de travail (le pas du dégroupage). La sortie d'avant (l'autre tampon de
  SPAN) se copie décalée dans la nouvelle ; ne se reprend que le décor sûr qui suit. Les deux bords de l'axe
  décalé sont dans ce qui a changé : la bande qui entre n'a pas de source, et près du bord d'où le contenu
  sort, la sortie voit maintenant le bourrage de zéros. La part fractionnaire va dans la dérive. Le
  débandage tire ses points sur une grille fixe de l'écran : sous un décalage, un aplat débandé repris peut
  s'écarter d'un seuil du débandage (environ un niveau) de son calcul complet ; la porte le mesure.
- **Moteur** (`gpu/cuda/backend/span_fp8_tiles.cpp`) : `next_span_tiles` avant l'appel, puis l'appel lance le
  corps masqué en direct (pas le graphe de l'image entière) ; les masques passent par une mémoire épinglée
  tournante. Le chemin qui écrit l'anneau depuis SPAN abandonne les tuiles. La sortie d'avant lue pour le
  décalage est déclarée aux voies comme une lecture (`SpanFp8Engine::tile_read`). Un appel refusé avant le
  moteur annule la demande (`next_span_tiles` avec un cœur vide) : elle ne vaut pas pour l'appel suivant.

ABI du backend : 50 (`Engine::span_tile_grid`, `Engine::next_span_tiles`), 56 (`PairScores::blocks`, le
filet). Nouvelle empreinte TensorRT, nouveau paquet Linux.

## Porte (plan : 48 dB ou plus d'un calcul complet, et l'outil 0.1 sans hausse)

À mesurer sur CachyOS, binaire `linux-release`, à la séance de test commune de la fin (consigne de Tatoub du 19 novembre : pas de banc par fonction), avec la carte de la scène et le SPAN natif :

1. **Justesse** : `--debug --span-native --scene-map --span-tiles-check` sur l'extrait unique de 60 s
   (`rezero-s4e18-1710.mkv`), jamais des épisodes entiers ; la ligne « SPAN par tuiles » du journal donne le
   pire bloc 16 × 16 et le plus grand écart moyen contre un calcul complet. Porte : pire bloc sous 1 niveau
   8 bits (environ 48 dB), écart moyen sous 0,05 niveau.
2. **Stabilité** : `tools/bench/qualite/scintillement.py` sur 20 s du même extrait, avec les tuiles face à
   `--no-span-tiles` : pas plus de scintillement avec les tuiles (porte `--porte`).
3. **Gain** : la ligne « SPAN par tuiles » et le temps GPU de SPAN (`--stats`) sur le même extrait, et sur
   le dialogue en SPAN seul (Spy×Family 13 BD à 17:06, 60 s), où la reprise entière ne prend que 0,8 % des
   images : le grain et les bouches y font changer presque chaque paire. Le cran Équilibre (ou Qualité) :
   au cran Performance, le FP8 CUTLASS, quand il se charge, calcule l'image entière.

## État

Écrit sans carte ni GPU (lavapipe ne fait pas tourner nos noyaux CUDA). Premier essai sur le PC le
19 novembre (SPAN natif Équilibre, `--scene-map`, 30 s de Spy×Family) : aucune image par tuiles et aucune
ligne au journal, la sortie d'avant gradée sur place sans que le bilan le dise ; corrigé par la copie
d'avant le grade et la ligne de fin toujours écrite (plus haut), à mesurer de nouveau. Audit du
19 novembre : voisines d'une tuile changée, bord sortant d'un panoramique, lueur locale, recette de l'entrée,
grade réellement passé, lecture déclarée aux voies, demande annulée après un refus. Puis la mesure du
contrôle (`scores_now`) qui, sous DistilDRBA après une image sautée, périmait le ticket du pont : `scores_now`
a son propre emplacement, hors de l'anneau des tickets (`slice_ops_test.cpp`). Le test
`tests/unit/app/upscale_tiles_test.cpp` couvre la dilatation, les masques, la décision sur des cartes
synthétiques et les raisons du bilan ; `source_grade_stage_test.cpp`, l'étage sous le grade avant le warp
(tuiles dès la troisième source, toujours depuis la copie d'avant le grade).
