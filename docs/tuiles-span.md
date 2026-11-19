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
fondu, caméra sans décalage entier. La région trop large dit ce qui la remplissait : part des tuiles en cause
et de la région une fois les deux portées ajoutées, puis la part de chaque cause (bords d'un décalage, carte,
filet des blocs, âge, dérive, caméra, luminosité). En fin de ligne, la caméra de la carte sur les images où
elle est ajustée : tenues (sous 0,025 pixel aux quatre coins et au centre), en partie (zoom, rotation) et qui
bougent partout, la médiane du déplacement au centre et la cohérence de sa direction (|Σ d| / Σ |d| sur
chaque suite d'images entre deux coupes : près de 100 % sous un panoramique, près de 0 pour une caméra
fantôme, du bruit de l'ajustement, qu'une médiane des normes ne distingue pas). Sans tuiles actives, la ligne
dit pourquoi (`--no-span-tiles`, moteur sans grille), et le démarrage donne la grille.

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
  va ; une tuile qui a dérivé de 0,025 pixel de travail ou plus depuis son dernier calcul (mouvements lents
  additionnés), qui a reçu autant de la caméra (plus bas), ou glissé de plus d'un code de
  luminosité (`SceneTile::shift`, signé : le grain s'y annule, une lueur ou un fondu local s'additionne) ; une
  tuile reprise 24 fois de suite. Le **cœur** recalculé est ce qui a changé dilaté de la portée (une sortie
  voit l'entrée jusqu'à 26 pixels de demi-résolution autour d'elle), plus le rayon du débandage (16 pixels de
  travail) quand l'entrée est débandée ; la **région** calculée est le cœur dilaté encore de la portée.
  L'image entière se recalcule à chaque coupe, fondu, saut, image abîmée, toutes les 120 images, quand la
  recette de l'entrée de SPAN change (débandage, filtre de réduction) et quand la région dépasse 70 % de
  l'image.
- **Caméra sans décalage** : chaque tuile additionne ce que la caméra de la carte lui fait (l'affine au coin
  de la tuile qui bouge le plus, moins le décalage appliqué), signé ; à 0,025 pixel de travail, elle se
  recalcule. Le bruit de l'ajustement sur un plan fixe s'annule, une caméra lente s'additionne, un zoom ou une
  rotation gardent les tuiles près de leur centre. L'image n'est entière d'emblée que si la caméra bouge
  d'autant aux quatre coins et au centre. Avant le 19 novembre, toute caméra que la carte ne disait pas tenue
  rendait l'image entière ; or la carte lit son seuil au coin haut gauche, partie linéaire comprise (un zoom
  de 0,03 % y fait déjà un quart de pixel en 1080p).
- **Budget de glissement** (`SpanTiles::kStillDrift`, 0,025 pixel de travail, pour la dérive, la caméra et
  l'accroche d'un décalage) : une sortie reprise d'une entrée qui a glissé de δ n'est pas celle du calcul
  complet. Mesure du 19 novembre (`tools/bench/qualite/tuiles_panoramique.py`, ArtCNN C4F32 2× à la place de
  SPAN, trois images de Spy×Family en 720p, fenêtre de 512 pixels) : la reprise sans décalage s'écarte au pire
  bloc de 0,6 à 0,7 niveau à δ = 0,02, de 1,4 à 1,7 à 0,05, de 7 à 8,5 à 0,25 et de 14 à 17 à 0,5, environ
  34 niveaux par pixel. La porte en tolère 1 : 0,025 en laisse 0,85. Jusque-là, la dérive tolérait un
  demi-pixel et la caméra un quart.
- **Filet sous la carte** : la carte juge des tuiles de 32 × 32 pixels de travail sur une image réduite de
  moitié, et ne dit « a changé » qu'à 6 codes d'écart moyen ou sur un vecteur net ; une petite bouche ou un
  clignement dans un plan large y passent. L'écart de chaque bloc 16 × 16 de la mesure de la paire (celle des
  dessins tenus, relue par `pair_diff` avec son pire bloc, `gpu::PairScores::blocks`) les voit : un bloc au-delà
  de 2 niveaux (`SpanTiles::kBlockChange`) recalcule les tuiles qu'il couvre. Le grain n'y arrive pas
  (écart moyen d'environ 1,13 σ par bloc ; les paires sans mouvement de Spy×Family 13 BD restent sous un
  niveau au pire bloc), une bouche si (une cinquantaine de pixels qui changent de 80 codes : 15 niveaux). Pas
  sous un décalage : la mesure compare les mêmes pixels, pas le contenu suivi. La ligne de fin dit combien de
  tuiles le filet a ajoutées, sur combien d'images : beaucoup, c'est que la carte en laisse passer.
- **Décalage** : caméra ajustée sans zoom ni rotation, résidu sous un demi-pixel, translation à moins de 0,025
  pixel d'un multiple de 2 pixels de travail (le pas du dégroupage). La sortie d'avant (l'autre tampon de
  SPAN) se copie décalée dans la nouvelle ; ne se reprend que le décor sûr qui suit. Les deux bords de l'axe
  décalé sont dans ce qui a changé : la bande qui entre n'a pas de source, et près du bord d'où le contenu
  sort, la sortie voit maintenant le bourrage de zéros. La part fractionnaire va dans la dérive et dans ce
  que la tuile a reçu de la caméra. Le débandage tire ses points sur une grille fixe de l'écran : sous un
  décalage, un aplat débandé repris peut s'écarter d'un seuil du débandage (environ un niveau) de son calcul
  complet ; la porte le mesure.
- **Panoramique sous-pixel** : rien ne s'y reprend, et c'est juste. Décaler la sortie d'avant d'une fraction de
  pixel (même mesure, l'entrée suivante étant l'image décalée en Lanczos comme par le compositeur) s'écarte du
  calcul complet de 1 à 3,7 niveaux au pire bloc et de 0,05 à 0,36 en moyenne pour δ de 0,05 à 0,5, en
  bicubique comme en Lanczos : le réseau ne commute pas avec un rééchantillonnage, alors qu'un décalage entier
  de l'entrée est exact (au pas de 2 pour SPAN, à cause du dégroupage). Un panoramique de 0,8 pixel par image
  n'atteint un multiple de 2 qu'après plusieurs images (12 pour 0,825), avec un reste de phase : il faudrait
  garder autant de sorties 4K, pour rien de sûr. Les tuiles ne servent qu'aux plans à caméra tenue.
- **Moteur** (`gpu/cuda/backend/span_fp8_tiles.cpp`) : `next_span_tiles` avant l'appel, puis l'appel lance le
  corps masqué en direct (pas le graphe de l'image entière) ; les masques passent par une mémoire épinglée
  tournante. Le chemin qui écrit l'anneau depuis SPAN abandonne les tuiles. La sortie d'avant lue pour le
  décalage est déclarée aux voies comme une lecture (`SpanFp8Engine::tile_read`). Un appel refusé avant le
  moteur annule la demande (`next_span_tiles` avec un cœur vide) : elle ne vaut pas pour l'appel suivant.

ABI du backend : 50 (`Engine::span_tile_grid`, `Engine::next_span_tiles`), 57 (`PairScores::blocks`, le
filet). Nouvelle empreinte TensorRT, nouveau paquet Linux.

## Rejeu sur processeur

Le PC ne garde que le contrôle final et les temps : les décisions se rejouent dans le cloud sur les vecteurs
NVOF qu'il a exportés une fois (`flux-carte-*` du dossier partagé, dix extraits de 160 images en 960 × 540,
diviseur 2 : six panoramiques de Re:Zero et de Spy×Family, deux dialogues de Spy×Family, une lueur, une aura).

- `tools/bench/qualite/tuiles_rejeu/` (construit à part, comme `carte_rejeu`) lit les tâches de `carte_rejeu`,
  tire la carte de chaque paire (`kernels::ref`), mesure la paire et l'écart à l'ancre sur les lumas du flux
  (blocs de 16 pixels de travail), puis fait tourner `SpanReuse::decide` et `SpanTiles::plan` compilés tels
  quels. Il écrit les deux lignes de fin du lecteur, et la part d'une image entière que le corps de SPAN a
  calculée. Écart au lecteur : la mesure lit la luma du flux, réduite de moitié, et non la copie de RIFE.
- `--ancre-composee`, `--somme-des-paires`, `--somme-bornee` : à côté, l'essai d'`ancre_tuiles.h`, des tuiles
  décidées sur les pixels, sans NVOF (plus bas) ; `--masques` écrit ses décisions image par image.
- `tools/bench/qualite/tuiles_controle.py` rejoue `--span-tiles-check` sur ces décisions : un réseau 2× (ArtCNN
  C4F32 à la place de SPAN) agrandit chaque luma, la sortie composée suit les tuiles calculées et reprises, et
  se compare au calcul complet de la même image (pire bloc 16 × 16, écart moyen).

Premier rejeu (19 novembre, `main` e52f6d26) : aucune image par tuiles sur les dix extraits, comme sur le PC.
Les deux dialogues et la lueur sont des panoramiques réels (cohérence de direction 99 à 100 %, de 0,66 à
1,05 pixel par image au centre, ce que confirme `tuiles_panoramique.py`) : pas de caméra fantôme. Sur les
plans à caméra tenue des panoramiques, la région déborde par la dérive et la carte : la moyenne des vecteurs
de NVOF d'une tuile de la carte y vaut 0,12 pixel de travail en médiane (0,55 au 90e centile, 14 % de
vecteurs nuls ; Spy×Family 13:16, images 81 à 126), cinq fois le budget de glissement. NVOF ne peut pas
certifier un glissement de 0,025 pixel : la dérive passe dès la première paire sur 90 % des tuiles.

Essai sur les pixels (`ancre_tuiles.h`) : la règle du lot A tuile par tuile. Une tuile se reprend si aucun de
ses blocs ne s'écarte de plus de 0,70 niveau de l'entrée qui a produit sa sortie ; un glissement sur un bord
net s'y voit (0,025 pixel y fait déjà 0,85 niveau), un aplat qui glisse ne coûte rien. Trois façons de tenir
cet écart : l'image composée de ce qui a produit la sortie (exacte ; au lecteur, un tampon et une copie
masquée), la somme des écarts des paires depuis le dernier calcul de la tuile (une majoration : le grain s'y
additionne ; au lecteur, rien de plus que `PairScores::blocks`), et cette somme resserrée par l'écart à la
dernière image calculée. Part d'une image entière que calcule le corps de SPAN :

| extrait | lecteur | image composée | somme des paires | somme bornée |
|---|---|---|---|---|
| Re:Zero, panoramique3 | 86,8 % | 49,1 % | 49,9 % | 49,9 % |
| Re:Zero, panoramique | 57,2 % | 43,5 % | 44,2 % | 43,6 % |
| Re:Zero, panoramique2 | 69,2 % | 57,0 % | 57,1 % | 57,1 % |
| Re:Zero, panoramique4 | 69,8 % | 56,4 % | 56,6 % | 56,5 % |
| Re:Zero, aura | 77,4 % | 67,1 % | 67,8 % | 67,2 % |
| Spy×Family 13:16, panoramique6 | 95,0 % | 56,8 % | 75,9 % | 75,3 % |
| Spy×Family 4:53, panoramique5 | 99,4 % | 78,7 % | 86,6 % | 86,6 % |
| dialogues, lueur (caméra qui bouge) | 99,4 à 100 % | inchangé | inchangé | inchangé |

Contrôle sur processeur de l'image composée (seuil 0,70) : pire bloc de 0,70 (panoramique), 0,76
(panoramique3) et 1,03 au plus (Spy×Family 13:16, médiane 0,88), au niveau des reprises entières du lot A
sur les mêmes extraits (0,67 à 0,79). L'écart moyen, lui, est celui du grain : 0,12 à 0,41 niveau, comme les
reprises entières du lot A (0,04 à 0,40). Une sortie reprise garde le grain de son image ; aucune reprise,
entière ou par tuiles, ne tient l'écart moyen de 0,05 sur une source granuleuse. Au seuil de 0,50 :
panoramique3 à 51,7 % (pire bloc 0,54), mais Spy×Family 13:16 à 98,7 %.

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
d'avant le grade et la ligne de fin toujours écrite (plus haut). Deuxième essai, même extrait : 0 image sur
716 par tuiles, dont 595 « caméra sans décalage entier » et 104 « région trop large » (sous DistilDRBA, 591
sur 715) ; temps de SPAN identique avec ou sans tuiles. D'où la caméra additionnée par tuile (plus haut) et le
détail de la région trop large et de la caméra au bilan. Troisième essai (#945) : toujours 0 sur 715, dont
593 « caméra sans décalage entier », la caméra bougeant partout sur 604 images sur 701 (médiane 0,825 pixel au
centre). D'où la mesure du panoramique sous-pixel et du budget (plus haut), le budget ramené à 0,025 pixel, et
la cohérence de direction au bilan ; `tuiles_panoramique.py` vérifie la caméra sans la carte (corrélation de
phase sur les images décodées). Rejeu sur processeur des dix
extraits du PC (plus haut) : même verdict, la dérive tirée de NVOF ne peut pas descendre au budget ; une règle
sur les pixels, celle du lot A tuile par tuile, en fait tomber le corps de SPAN de 87 % à 50 % sur Re:Zero et
de 95 % à 57 % sur Spy×Family 13:16. Audit du
19 novembre : voisines d'une tuile changée, bord sortant d'un panoramique, lueur locale, recette de l'entrée,
grade réellement passé, lecture déclarée aux voies, demande annulée après un refus. Puis la mesure du
contrôle (`scores_now`) qui, sous DistilDRBA après une image sautée, périmait le ticket du pont : `scores_now`
a son propre emplacement, hors de l'anneau des tickets (`slice_ops_test.cpp`). Le test
`tests/unit/app/upscale_tiles_test.cpp` couvre la dilatation, les masques, la décision sur des cartes
synthétiques et les raisons du bilan ; `source_grade_stage_test.cpp`, l'étage sous le grade avant le warp
(tuiles dès la troisième source, toujours depuis la copie d'avant le grade).
