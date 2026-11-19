# Les tuiles qui bougent

Novembre 2026, phase 3 du plan du flux optique matériel NVIDIA (suite du lot A, « Sauter les images tenues »).
Quand une image n'est pas reprise entière, SPAN ne recalcule que les tuiles dont l'entrée a bougé ; le reste de
sa sortie reprend celle de l'image d'avant. Aucun modèle nouveau : SPAN tel quel, et la règle du lot A tuile par
tuile, sur l'écart des blocs de la mesure de la paire. Jusqu'au 19 novembre, la carte de la scène décidait
(plus bas, « La carte de la scène ») ; elle ne le pouvait pas. Allumé par défaut depuis le 19 novembre, une fois
le scintillement mesuré sur processeur (plus bas) ; `--no-span-tiles` (avec `--debug`) le coupe, pour l'ABAB des
temps du PC.

## Ce qu'il faut pour que ça tourne

- **Par défaut** ; `--no-span-tiles` (avec `--debug`) les coupe, pour mesurer avec et sans. `--span-tiles`,
  l'option de l'essai, reste acceptée et ne change rien ; `--span-tiles-check` les rallume, même après
  `--no-span-tiles`.
- **SPAN natif** (le défaut sur RTX 40 et 50, `docs/span-native16.md`) : nos noyaux savent sauter une tuile
  d'un bloc entier, à sommes f16 (Équilibre, et repli du cran Performance) comme f32 (Qualité). TensorRT
  (forme fixe) et le corps FP8 CUTLASS ne calculent que l'image entière : leur grille est vide
  (`gpu::Engine::span_tile_grid`) et rien ne change pour eux, pas même la mesure des paires ; le journal le
  dit au démarrage et à la fin (« ce moteur ne calcule que l'image entière »).
- **La mesure de la paire, bloc par bloc** (`gpu::PairScores::blocks`, relue par le backend TensorRT, comme le
  SPAN natif) : quand les tuiles peuvent servir (moteur à grille), chaque paire (N − 1, N) se mesure avant le
  SPAN de N, à ×1 compris, en
  avance (`look_ahead`, `drba_next` sous DistilDRBA) ; en SPAN seul comme depuis le lot A (#901), avec ou sans
  `--no-span-reuse`. Sans mesure, l'image se calcule entière.
- **En lecture** : jamais à l'export (l'export rejoue la chaîne au bit près), ni avec TemporalFix (la mesure
  ne voit pas sa sortie), ni avec « Lisser les dessins tenus » (SPAN fait en avance), ni avec la plaque de
  décor.
- **Sous le grade avant le warp** (le cas du PC : SPAN avant le warp en 4K, grade HDR anime) : le grade
  réécrit sur place le tampon de SPAN que le warp lit. Juste avant, la sortie de SPAN telle quelle est copiée
  (`SpanReuse::ungraded`, une copie exacte de l'image de sortie, environ 0,2 ms en 4K sur une 4070 Ti) et
  l'image suivante part de cette copie. La copie ne se fait que si les tuiles peuvent servir (moteur à
  grille, tuiles non coupées, chaîne en lecture) et que le grade a écrit l'image d'avant : un grader qui ne grade
  pas ne coûte rien. Sans elle (refus de place), l'image se calcule entière.

La ligne de fin du journal sort sous SPAN, sauf avec `--no-span-tiles`, même sans aucune image par tuiles :
« SPAN par tuiles : t images sur n calculées (p %), x % des tuiles recalculées (y % avec le halo) ; images
entières : … ». Chaque image entière y compte sous une seule raison (seules les raisons non nulles sont
écrites) : sans mesure des blocs, région trop large, refus du moteur, sortie d'avant gradée sur place (pas de
copie), sortie d'avant écrasée (image créée par-dessus, source sautée), chaîne sans tuiles (export, YUV,
TemporalFix, Smooth, plaque), début, saut ou image abîmée, entrée de SPAN changée, rafraîchissement. La région
trop large dit ce qui la remplissait : part des tuiles en cause et de la région une fois les deux portées
ajoutées, puis la part des blocs et de l'âge. Sur un moteur sans grille, la ligne le dit ; le démarrage donne
la grille. Les tuiles n'arrêtent aucune série du lot A : la ligne « SPAN repris » est celle de
`--no-span-tiles` (plus bas, « lot A d'abord »).

## Comment

- **Grille** (`kernels/spec/span_tiles.h`) : la tuile des noyaux, 8 × 32 pixels de demi-résolution, soit 64 × 16
  pixels de travail. Le masque d'un lancement tient 4 096 tuiles (de quoi couvrir 2560 × 1080) ; au-delà,
  l'image entière.
- **Halo** : le champ réceptif calculé sur les couches lancées, un pixel de demi-résolution par 3×3 sur le plus
  long chemin (conv_1, trois par bloc, to_pixel) : 26 pour un SPAN à huit blocs, soit une colonne et quatre
  lignes de tuiles. C'est la région que le moteur calcule autour du cœur.
- **Portée réelle** (`SpanTiles::kCoreReachPx`, soir du 19 novembre) : ce qu'une entrée changée fait à la sortie
  s'éteint bien avant le champ théorique. SPAN Équilibre (b8f64) sur processeur, un carré de 32 pixels de
  travail changé de 20 à 70 niveaux sur trois images de Re:Zero : la sortie bouge encore de 0,1 à 0,6 niveau
  à 16 pixels, de 0,01 à 0,1 à 20, et de moins de 0,003 à 32 ; rien du tout à 26 pour b5f48. Le cœur se
  dilate donc de 32 pixels de travail (une colonne, deux lignes) et non plus de la portée du réseau (quatre
  lignes) : une tuile isolée coûte 5 × 13 tuiles de région au lieu de 5 × 17.
- **Exactitude** : toutes les couches se calculent sur le cœur dilaté du halo. Hors de cette région, les
  tampons intermédiaires tiennent n'importe quoi, mais une valeur fausse n'avance que d'un rayon par couche :
  elle n'atteint jamais le cœur. Seul le cœur est écrit (`to_pixel` masqué). Le calcul des tuiles est donc
  celui d'un calcul complet, au flottant près.
- **Décision** (`app/play/upscale_tiles.{h,cpp}`) : la règle du lot A, tuile par tuile. La sortie d'une tuile
  se reprend tant qu'aucun bloc de son entrée ne s'écarte de plus de 0,70 niveau (`kernels::kHeldThreshold`)
  de l'entrée qui l'a produite. L'écart se lit sur la mesure de la paire, l'écart moyen de chaque bloc 16 × 16
  entre la précédente et l'image (relu avec le pire bloc par `pair_diff`, sur la copie de RIFE quand elle
  existe). Par bloc, la somme des écarts des paires majore l'écart à l'entrée d'avant (l'écart moyen d'un bloc
  est une norme : inégalité triangulaire ; chaque paire compte 1/256 de niveau de plus, ce que la mesure
  tronque). La somme se resserre par l'écart à l'ancre du lot A (la dernière image calculée), mesuré bloc par
  bloc (`AnchorGap::blocks`) : écart à l'ancre plus la majoration du bloc à l'ancre. Juste après une image
  entière, c'est la règle du lot A ; dans une série de reprises, du grain que l'ancre voit au même niveau ne
  s'additionne plus. Un bloc au-delà du seuil fait recalculer ses tuiles ; le **cœur** recalculé est ce qui
  a changé dilaté de la portée réelle (plus haut), plus le rayon du débandage (16 pixels de travail) quand
  l'entrée est débandée ; la **région** calculée est le cœur dilaté encore du halo. Toute sortie dont le
  champ tient dans la région est exacte : elle s'écrit aussi (l'érosion de la région), sans rien calculer de
  plus, et repart de cette image (deux cœurs voisins, la colonne du bord).
- **Remise à zéro** : la somme d'un bloc ne repart de zéro que quand toute sortie qui le voit est recalculée,
  ses tuiles et tout ce qui est à leur portée (l'érosion du cœur). Recalculer ses seules tuiles ne suffit
  pas : une voisine reprise, à quatre lignes, garde la sortie de son entrée d'avant. Une tuile reprise 24 fois
  de suite se recalcule. L'image entière se recalcule toutes les 120 images, à chaque saut, image abîmée ou
  paire sans mesure, quand la recette de l'entrée de SPAN change (débandage, filtre de réduction) et quand la
  région dépasse 70 % de l'image (une coupe et un fondu y passent).
- **Reprise entière après une image par tuiles** (lot A d'abord) : la sortie mêle alors des tuiles calculées à
  des images différentes, et l'écart à l'ancre du lot A ne mesure que la dernière. La reprise entière suit
  quand même la seule règle du lot A, sans garde des blocs : les tuiles s'ajoutent à ses séries sans en
  arrêter aucune. Chaque reprise vieillit chaque tuile et donne son écart à chaque bloc (`SpanTiles::held`) ;
  la prochaine image calculée recalcule ce qui a passé le seuil ou 24 images, ou se calcule entière si
  l'image des 120 est due. Le contrôle avec le vrai SPAN tient ces reprises au pire bloc des images par
  tuiles (plus bas). Une reprise imposée (copie d'une tenue fondue, Smooth) ne mesure pas les blocs : la
  prochaine image calculée est entière.
- **Ce que la règle voit** : un glissement sur un bord net (0,025 pixel y fait déjà 0,85 niveau au pire bloc,
  plus bas), une bouche, un clignement, une lueur ou un fondu local qui s'additionnent ; un aplat qui glisse
  ne coûte rien, sa sortie ne bouge pas non plus. Le grain s'additionne lui aussi : la majoration est
  prudente, et sur une source granuleuse les blocs texturés passent le seuil en quelques images.
- **Budget de glissement** : une sortie reprise d'une entrée qui a glissé de δ n'est pas celle du calcul
  complet. Mesure du 19 novembre (`tools/bench/qualite/tuiles_panoramique.py`, ArtCNN C4F32 2× à la place de
  SPAN, trois images de Spy×Family en 720p, fenêtre de 512 pixels) : la reprise s'écarte au pire bloc de 0,6 à
  0,7 niveau à δ = 0,02, de 1,4 à 1,7 à 0,05, de 7 à 8,5 à 0,25 et de 14 à 17 à 0,5, environ 34 niveaux par
  pixel. La porte en tolère 1 : environ 0,025 pixel, que NVOF ne sait pas certifier (plus bas) et que les
  pixels voient.
- **Panoramique** : rien ne s'y reprend, et c'est juste. Décaler la sortie d'avant d'une fraction de pixel
  (même mesure, l'entrée suivante étant l'image décalée en Lanczos comme par le compositeur) s'écarte du
  calcul complet de 1 à 3,7 niveaux au pire bloc et de 0,05 à 0,36 en moyenne pour δ de 0,05 à 0,5, en
  bicubique comme en Lanczos : le réseau ne commute pas avec un rééchantillonnage, alors qu'un décalage entier
  de l'entrée est exact (au pas de 2 pour SPAN, à cause du dégroupage). Un panoramique de 0,8 pixel par image
  n'atteint un multiple de 2 qu'après plusieurs images (12 pour 0,825), avec un reste de phase ; aucun
  extrait du PC n'en a montré d'un nombre entier de pas. Le décalage de la sortie d'avant n'est plus demandé
  (`TilePlan::shift_x` toujours nul) ; le moteur le garde. Les tuiles servent aux plans à caméra tenue.
- **Moteur** (`gpu/cuda/backend/span_fp8_tiles.cpp`) : `next_span_tiles` avant l'appel, puis l'appel lance le
  corps masqué en direct (pas le graphe de l'image entière) ; les masques passent par une mémoire épinglée
  tournante. Le chemin qui écrit l'anneau depuis SPAN abandonne les tuiles. La sortie d'avant lue sur l'autre
  tampon est déclarée aux voies comme une lecture (`SpanFp8Engine::tile_read`). Un appel refusé avant le
  moteur annule la demande (`next_span_tiles` avec un cœur vide) : elle ne vaut pas pour l'appel suivant.

ABI du backend : 50 (`Engine::span_tile_grid`, `Engine::next_span_tiles`), 57 (`PairScores::blocks`). Le
passage de la carte aux blocs ne touche que `src/app` : ni ABI, ni empreinte TensorRT.

## La carte de la scène (jusqu'au 19 novembre)

La première décision lisait la carte de la scène (`--scene-map`, `docs/carte-de-la-scene.md`) : une tuile
changeait si une tuile de la carte dessous changeait, si elle avait dérivé de 0,025 pixel de travail ou plus
depuis son dernier calcul (vecteurs de NVOF additionnés), reçu autant de la caméra ou glissé d'un code de
luminosité ; un filet sur l'écart des blocs (2 niveaux) rattrapait les petites bouches ; sur un panoramique
d'un nombre entier de pas, la sortie d'avant se copiait décalée. Trois essais sur le PC (19 novembre,
Spy×Family, SPAN natif Équilibre) et le rejeu sur processeur des dix extraits (plus bas) : aucune image par
tuiles. Sur les plans à caméra tenue, la moyenne des vecteurs de NVOF d'une tuile de la carte vaut 0,12 pixel
de travail en médiane (0,55 au 90e centile, 14 % de vecteurs nuls ; Spy×Family 13:16, images 81 à 126), cinq
fois le budget de glissement : la dérive y passait dès la première paire sur 90 % des tuiles. Le détail est
dans l'historique de ce fichier (#945, #949, #956).

## Rejeu sur processeur

Le PC ne garde que le contrôle final et les temps : les décisions se rejouent dans le cloud sur les vecteurs
NVOF qu'il a exportés une fois (`flux-carte-*` du dossier partagé, dix extraits de 160 images en 960 × 540,
diviseur 2 : six panoramiques de Re:Zero et de Spy×Family, deux dialogues de Spy×Family, une lueur, une aura).

- `tools/bench/qualite/tuiles_rejeu/` (construit à part, comme `carte_rejeu`) mesure la paire et l'écart à
  l'ancre sur les lumas du flux (blocs de 16 pixels de travail), puis fait tourner `SpanReuse::decide` et
  `SpanTiles::plan` compilés tels quels, comme le lecteur. Il écrit les deux lignes de fin
  du lecteur, et la part d'une image entière que le corps de SPAN a calculée ; `--masques` écrit ses
  décisions image par image. Écart au lecteur : la mesure lit la luma du flux, réduite de moitié, et non la
  copie de RIFE. La carte de chaque paire (`kernels::ref`) ne sert plus qu'aux colonnes de `--detail` et aux
  coupes de l'essai.
- `--ancre-composee`, `--somme-des-paires`, `--somme-bornee` : à côté, l'essai d'`ancre_tuiles.h`, les règles
  sur les pixels essayées avant d'en brancher une (plus bas) ; `--masques` écrit alors leurs décisions.
- `tools/bench/qualite/tuiles_controle.py` rejoue `--span-tiles-check` sur ces décisions : un réseau 2× (ArtCNN
  C4F32 à la place de SPAN) agrandit chaque luma, la sortie composée suit les tuiles calculées et reprises, et
  se compare au calcul complet de la même image (pire bloc 16 × 16, écart moyen).

Trois façons de tenir l'écart d'un bloc à l'entrée qui a produit sa sortie ont été essayées : l'image composée
de ce qui a produit la sortie (exacte ; au lecteur, un tampon et une copie masquée), la somme des écarts des
paires (une majoration : le grain s'y additionne ; au lecteur, rien de plus que `PairScores::blocks`), et cette
somme resserrée par l'écart à la dernière image calculée. Le lecteur a pris la somme des paires, avec la remise
à zéro stricte (plus haut). Part d'une image entière que calcule le corps de SPAN :

| extrait | carte (avant) | lecteur | image composée | somme bornée |
|---|---|---|---|---|
| Re:Zero, panoramique3 | 86,8 % | 51,1 % | 49,1 % | 49,9 % |
| Re:Zero, panoramique | 57,2 % | 44,2 % | 43,5 % | 43,6 % |
| Re:Zero, panoramique2 | 69,2 % | 57,3 % | 57,0 % | 57,1 % |
| Re:Zero, panoramique4 | 69,8 % | 56,9 % | 56,4 % | 56,5 % |
| Re:Zero, aura | 77,4 % | 67,5 % | 67,1 % | 67,2 % |
| Spy×Family 13:16, panoramique6 | 95,0 % | 75,9 % | 56,8 % | 75,3 % |
| Spy×Family 4:53, panoramique5 | 99,4 % | 86,6 % | 78,7 % | 86,6 % |
| dialogues, lueur (caméra qui bouge) | 99,4 à 100 % | inchangé | inchangé | inchangé |

Les deux dialogues et la lueur sont des panoramiques réels (cohérence de direction 99 à 100 %, de 0,66 à
1,05 pixel par image au centre, ce que confirme `tuiles_panoramique.py`) : rien ne s'y reprend, et c'est
juste. L'image composée ne gagne qu'au grain de Spy×Family (la somme y additionne le grain des blocs
texturés) ; elle coûterait au lecteur un tampon et une copie masquée par image.

Contrôle sur processeur des décisions du lecteur (`tuiles_controle.py`, ArtCNN C4F32, 159 images par extrait),
face aux reprises entières du lot A sur les mêmes images :

| extrait | images par tuiles | pire bloc (max) | pire bloc (médiane) | écart moyen (max) | lot A : pire bloc (max) |
|---|---|---|---|---|---|
| Re:Zero, panoramique3 | 104 | 0,76 | 0,50 | 0,16 | 0,71 |
| Re:Zero, panoramique | 37 | 0,70 | 0,61 | 0,12 | 0,66 |
| Re:Zero, panoramique2 | 25 | 0,69 | 0,52 | 0,08 | 0,69 |
| Re:Zero, panoramique4 | 29 | 0,69 | 0,55 | 0,12 | 0,69 |
| Re:Zero, aura | 26 | 0,78 | 0,50 | 0,19 | 0,66 |
| Spy×Family 13:16, panoramique6 | 37 | 0,98 | 0,85 | 0,35 | 0,91 |
| Spy×Family 4:53, panoramique5 | 32 | 1,17 | 0,92 | 0,30 | (une seule reprise) |

Sur Re:Zero, le pire bloc des images par tuiles reste au niveau des reprises entières du lot A. Sur Spy×Family
en BD, le grain le pousse vers 1 : 0,98 sur 13:16, et sur 4:53, 7 images sur 32 au-delà de 1 niveau (1,17 au
plus) : la porte du pire bloc n'y est pas tenue. Le seuil de 0,70 porte sur l'entrée, et le réseau amplifie
le grain d'une sortie reprise ; la reprise entière du lot A y monte de même (0,91 sur 13:16). L'écart moyen
est celui du grain, 0,08 à 0,35 niveau, comme les reprises entières du lot A (0,02 à 0,39) : une sortie
reprise garde le grain de son image, et aucune reprise, entière ou par tuiles, ne tient 0,05 sur une source
granuleuse. Au seuil de 0,50 (essai de l'image composée), le pire bloc tombe vers 0,55 sur Re:Zero
(panoramique3 à 51,7 %), mais Spy×Family ne gagne plus rien (98,7 % sur 13:16).

**Scintillement** (`tuiles_controle.py --scintillement`) : le PC ne peut pas le mesurer, l'export (seule sortie
que lit `scintillement.py`) ne passant jamais par tuiles. Les deux suites du contrôle, calculée entière et
composée par les tuiles, s'y mesurent comme deux exports (`scintillement.run`) : la source est la luma agrandie,
le flux celui que le PC a exporté (`k.flux`, l'aller seul). Porte : pas plus de scintillement avec les tuiles,
plan par plan, à 1 % près. Tenue sur les 16 plans des sept extraits :

| extrait | plans | entières | tuiles |
|---|---|---|---|
| Re:Zero, panoramique3 | 1 | 0,192 | 0,191 |
| Re:Zero, panoramique | 4 | 0,337 · 0,120 · 0,015 · 0,026 | 0,337 · 0,120 · 0,013 · 0,023 |
| Re:Zero, panoramique2 | 2 | 0,220 · 0,037 | 0,220 · 0,035 |
| Re:Zero, panoramique4 | 1 | 0,165 | 0,164 |
| Re:Zero, aura | 4 | 0,367 · 0,035 · 0,252 · 0,283 | 0,367 · 0,031 · 0,250 · 0,283 |
| Spy×Family 13:16, panoramique6 | 2 | 0,486 · 0,365 | 0,486 · 0,260 |
| Spy×Family 4:53, panoramique5 | 2 | 0,908 · 0,516 | 0,908 · 0,469 |

Une tuile reprise est la même d'une image à l'autre : les tuiles égalent le calcul entier ou scintillent
moins. Le grain de 4:53, qui pousse le pire bloc à 1,17, ne fait pas monter le scintillement (0,469 contre
0,516) : la règle reste telle quelle.

## Mesure du PC (19 novembre) et règle revue

Au PC (#962, Re:Zero 17:10, 60 s, SPAN natif Équilibre et RIFE 4.25) : pire bloc 0,75 niveau, écart moyen
0,125 au plus (justesse tenue, tolérance du lot A) ; 195 images par tuiles sur 1229 calculées (15,9 %, 21 %
de région) ; 1031 images entières pour région trop large (82 % des tuiles en cause, 99 % de région, blocs
99 %) ; moteur SPAN 11 456 ms contre 12 228 sans tuiles (−6,3 %). Les tuiles arrêtent 71 séries du lot A
(reprise entière 14,9 % au lieu de 19,8 %).

Ce que disent ces chiffres :
- une image par tuiles coûte sa région, plus environ 6 % d'une image entière (534 ms pour 195 images à 21 %
  de région, corps à 10,26 ms l'image entière) ;
- les 70 images que le lot A perd passent par tuiles, pas entières (1034 images entières contre 1159 sans
  tuiles) : environ 190 ms, 1,6 % du temps de SPAN ;
- la région trop large ne vient pas de la règle : sur une seule paire, 82 % des tuiles ont un bloc au-delà
  du seuil (panoramiques, mouvement d'ensemble), et la région dilatée en couvre 99 %. Aucun plafond n'y fait
  gagner. Sur les extraits du rejeu, une paire qui n'est pas tenue met en cause 25 à 93 % des tuiles en
  médiane, et un seuil de 1 niveau au lieu de 0,70 n'y change presque rien (93 à 90 % sur panoramique, 25 à
  24 % sur panoramique3) : ce sont de vrais changements.

La règle revue garde ce qui gagne sans rien relâcher : la portée réelle (deux lignes au lieu de quatre),
l'écart à l'ancre bloc par bloc (moins d'arrêts du lot A) et les sorties exactes de la région écrites (plus
haut). Rejeu, coût de SPAN (corps, plus 6 % d'une image entière par image par tuiles ; `tuiles_rejeu` l'écrit,
`--sans-tuiles` donne le lot A seul) :

| extrait | lot A seul | tuiles (#962) | règle revue | lot A d'abord | arrêts du lot A (#962, revue) |
|---|---|---|---|---|---|
| Re:Zero, panoramique3 | 86,8 % | 55,0 % | 53,5 % | 53,2 % | 8, 5 |
| Re:Zero, panoramique | 57,2 % | 45,6 % | 44,9 % | 44,3 % | 10, 6 |
| Re:Zero, panoramique2 | 69,2 % | 58,2 % | 57,9 % | 57,8 % | 4, 3 |
| Re:Zero, panoramique4 | 69,8 % | 58,0 % | 57,4 % | 57,3 % | 5, 3 |
| Re:Zero, aura | 77,4 % | 68,4 % | 67,1 % | 66,8 % | 6, 5 |
| Spy×Family 13:16, panoramique6 | 95,0 % | 77,3 % | 76,3 % | 75,9 % | 2, 3 |
| Spy×Family 4:53, panoramique5 | 99,4 % | 87,8 % | 86,9 % | 86,9 % | 0, 0 |
| moyenne des sept | 79,3 % | 64,3 % | 63,4 % | 63,2 % | 35, 25 |

Essayé et laissé : un plafond de région à 0,85 ou 0,90 (plus d'images par tuiles, mais plus d'arrêts du lot
A : 63,4 % dans les deux cas). Le gain des tuiles dépend du contenu : sur un plan à caméra tenue il est grand
(panoramique3, 87 % à 53 %), sur un panoramique nul.

Contrôle avec le vrai SPAN Équilibre (b8f64) sur processeur, `tuiles_controle.py --modele` (aura, 159 images) :
pire bloc des images par tuiles 0,96 niveau avant comme après (médiane 0,59 avant, 0,73 après, sur 26 et 31
images), écart moyen 0,24 au plus ; reprises entières du lot A 0,77 ; scintillement tenu sur les quatre plans
(0,391 contre 0,390 au pire).

Au PC, la règle revue (9c40844a) : pire bloc 0,79 niveau, écart moyen 0,144 au plus (0,125 sur #962) ; moteur
SPAN 11 399 ms contre 12 210 sans tuiles (−6,6 %), médiane 10,48 ms, p95 10,59 contre 10,96 ms. Ce qui
retient encore le gain : la reprise entière à 14,7 % au lieu de 19,8 % (73 arrêts aux tuiles, série la plus
longue 6 au lieu de 19), et 1028 images entières pour région trop large.

**Lot A d'abord** (retenu le soir du 19 novembre) : la reprise entière ne dépend plus des tuiles (plus haut).
Le lot A garde toutes ses séries, aux mêmes taux que seul (13,1 % à 42,5 % de reprises sur les cinq extraits
de Re:Zero), et les tuiles ne jouent que sur les images qu'il calcule : 63,2 % au rejeu, aucun arrêt. Contrôle
avec le vrai SPAN : sur aura, reprises entières du lot A au pire bloc 0,96 (comme les images par tuiles, écart
moyen 0,25 au plus), sur panoramique3 0,79 (images par tuiles 0,82) ; scintillement tenu (aura 0,391 contre
0,390, panoramique3 0,212 contre 0,215).

Au PC (4e719097), la porte passe : pire bloc 0,79 niveau, écart moyen 0,165 au plus ; moteur SPAN 11 241 ms
contre 12 244 sans tuiles (−8,2 %), p95 10,68 contre 11,07 ms ; reprise entière 19,8 % des deux côtés, série
la plus longue 19. Reste le gros levier : 1028 images entières pour région trop large (82 % des tuiles en
cause).

## Porte (plan : 48 dB ou plus d'un calcul complet, et l'outil 0.1 sans hausse)

La stabilité est tenue sur processeur (plus haut). Au PC, à la séance de test commune de la fin (consigne de
Tatoub du 19 novembre : pas de banc par fonction), sur CachyOS, binaire `linux-release`, SPAN natif au cran
Équilibre (ou Qualité : au cran Performance, le FP8 CUTLASS, quand il se charge, calcule l'image entière), sur
l'extrait unique de 60 s (`rezero-s4e18-1710.mkv`), jamais des épisodes entiers :

1. **Justesse** (peut partager le PC) : `kirari --debug --mute --span-tiles-check --seconds=60
   rezero-s4e18-1710.mkv`. La ligne « SPAN par tuiles » du journal donne la part des images par tuiles et des
   tuiles recalculées, puis le pire bloc 16 × 16 et le plus grand écart moyen contre un calcul complet. Porte :
   pire bloc sous 1 niveau 8 bits (environ 48 dB). L'écart moyen sous 0,05 niveau ne se tient sur aucune
   source granuleuse, pas même par les reprises entières du lot A (plus haut) : la porte est celle du lot A,
   l'écart moyen de ses reprises entières sur le même extrait (tolérance du lot A, choisie le 19 novembre,
   Tatoub laissant le choix au fil).
2. **Temps** (le PC seul) : `kirari --debug --mute --seconds=60 --stats=tuiles.tsv rezero-s4e18-1710.mkv`, puis
   la même avec `--no-span-tiles --stats=sans-tuiles.tsv`. Le temps GPU de SPAN se lit sur la ligne « profil
   GPU : moteur … » du journal (médiane et total ; `*.gpu.tsv` à côté), avec « SPAN repris » et « SPAN par
   tuiles ». Le dialogue de Spy×Family 13 BD à 17:06 est un panoramique : les tuiles n'y gagnent rien. Sur
   #962 : −6,3 %, sur la règle revue −6,6 %, sur le lot A d'abord −8,2 % (plus haut, « Mesure du PC »).
   Porte du temps : le moteur SPAN pas plus lent qu'avec `--no-span-tiles`, et la reprise entière (« SPAN
   repris ») pas plus basse.

Une ligne ratée : ce fil corrige la règle, ou coupe les tuiles par défaut.

## État

Écrit sans GPU (lavapipe ne fait pas tourner nos noyaux CUDA). Trois essais sur le PC le 19 novembre, sur la
carte de la scène (SPAN natif Équilibre, `--scene-map`, Spy×Family) : aucune image par tuiles ; le premier a
mené à la copie d'avant le grade et à la ligne de fin toujours écrite, les suivants au budget de 0,025 pixel
et au bilan de la caméra. Le rejeu sur processeur des dix extraits du PC a montré que NVOF ne peut pas
descendre à ce budget : la décision lit maintenant l'écart des blocs de la paire. Sur les mêmes extraits, le
corps de SPAN tombe de 87 % à 51 % sur Re:Zero (panoramique3) et de 95 % à 76 % sur Spy×Family 13:16 ; le
contrôle sur processeur tient le pire bloc au niveau du lot A sur Re:Zero, pas sur Spy×Family 4:53 (1,17
niveau sur 7 images sur 32), et le scintillement n'augmente sur aucun plan. Allumé par défaut le 19 novembre
sur ce dernier point.

Audits du 19 novembre : voisines d'une tuile changée, lueur locale, recette de l'entrée, grade réellement
passé, lecture déclarée aux voies, demande annulée après un refus, mesure du contrôle (`scores_now`) hors de
l'anneau des tickets (`slice_ops_test.cpp`) ; puis, sur la règle des blocs, la remise à zéro stricte, l'âge
et l'image entière due sous la reprise entière, la troncature de la mesure, la raison au bilan d'une reprise
sans mesure et la mesure de chaque paire à ×1. Restent à reprendre avec le prochain paquet TensorRT (un
changement de `src/gpu` ou de `src/kernels` le demande) : les commentaires de `PairScores::blocks`
(`gpu/kernel_ops.h`), de `tensorrt_rife.cpp` et de `kernels/spec/scene_flow.h`, qui disent encore que les
tuiles lisent la carte, et le nom du profil GPU du SPAN natif, « moteur FP8 … » alors que le FP16 natif passe
par le même moteur (`tensorrt_span_fp8.cpp`, `stage_named`).

Le test `tests/unit/app/upscale_tiles_test.cpp` couvre la dilatation, les masques, la décision sur des écarts
de blocs synthétiques (remise à zéro stricte comprise), la reprise entière qui suit la seule règle du lot A
(l'image calculée d'après rattrape les blocs et l'âge), et les raisons du bilan ; `source_grade_stage_test.cpp`, l'étage sous le grade avant le warp (tuiles
dès la troisième source, toujours depuis la copie d'avant le grade) et en SPAN seul, puis (soir du 19 novembre) la
portée réelle, les sorties exactes de la région et l'écart à l'ancre bloc par bloc. Reste au PC : la justesse
(`--span-tiles-check`) et les temps du lot A d'abord, mesurés le 19 novembre : porte passée (plus haut).
