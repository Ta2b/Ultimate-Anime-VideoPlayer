# La carte de la scène (flux optique de NVIDIA)

Étapes 2.2 et 2.3 du plan « Usages du flux optique matériel NVIDIA » (19 novembre 2026) : le flux de NVOF sert
tel quel à décider, jamais à déplacer des pixels. Il est imprécis sur les mouvements rapides de l'anime (11 à
45 pixels d'écart au-delà de 45 pixels de mouvement, mesure du 17 novembre) ; la carte ne dit donc que « ça
bouge ou pas », « ça correspond ou pas » et le mouvement d'ensemble de la caméra. Aucun modèle, aucun
entraînement.

## Où, et pour qui

- Cartes : NVIDIA RTX 30 et après (la carte de coût de NVOF n'existe pas sur Turing). AMD, Intel, Turing, un
  pilote sans `libnvidia-opticalflow.so.1` (ou `nvofapi64.dll`) : une ligne au journal, puis la chaîne
  exactement comme sans la carte.
- La lecture lance la carte lorsque les calques, la plaque, le tri des paires ou les éclats qui utilisent
  ses tuiles sont activés. L'essai `--scene-map` la lance aussi pour le banc et `--no-scene-map` la coupe.
  Les éclats
  sur ses effets (classe Effet) restent éteints par défaut (`--effect-highlights`, essai, les rallume) ; les
  lueurs vues dans l'image par le grade n'en dépendent pas (`docs/hdr-anime.md`).
  Chantier en pause depuis le 19 novembre (Tatoub) : la classe Effet reste calculée, mais rien ne la rallume
  sans son accord. Pour reprendre : étiqueter des lueurs qui changent, lire la table « Lumières » de
  `carte_lecteur.py`, régler `kLightKept` sans rouvrir les faux effets, puis juger sur `--effect-mask-view`.
- **Défaut connu (v0.13.0-rc.2, 20 novembre 2026)** : des traits fantômes, ceux d'une image précédente, restent
  dans l'image. Exemple : Re:Zero saison 4 épisode 18 à 1:59, des contours violets et une forme rose dans le ciel
  bleu à gauche d'Emilia. Le défaut touche tous les moteurs d'interpolation (RIFE 4.25, 4.22 lite et DistilDRBA)
  avec SPAN ×2 et le grade HDR. Au PC de Tatoub, `--no-scene-map` le fait disparaître ; `--no-span-tiles` et
  `--no-span-reuse` ne changent rien. Cause trouvée le même jour : la **plaque de décor**, qui dépend de la carte
  et s'applique avec tous les moteurs ; `background_plate = false` seul le fait partir, la carte restant
  allumée ; correctif : la garde de la plaque compare les trois canaux au lieu de la luma (détail :
  [`calques.md`](calques.md)).
- Journal de fin : `carte de la scène : N paires lancées, N relues, N posées sur leur paire ; N sautées (NVOF
  occupé), N relues trop tard, N perdues ; caméra ajustée sur N, affinée sur N (N affinages trop tard, dont N
  relus ensuite, N pas lancés ; N lancés derrière le flux depuis la caméra du plan, N écartés), N coupes
  probables, N fondus ; hôte : N µs pour lancer une paire, N µs pour tirer une carte, en moyenne`. « Posées »
  est ce qui compte : avec RIFE, presque toutes les paires lancées. « Trop tard » : la carte n'était pas relue
  quand l'étage a décidé de sa paire ; « affinages trop tard » : elle l'était, mais pas sa caméra affinée (la
  carte part avec celle de NVOF ; l'affinage, relu ensuite, va aux calques par `SceneMap::last_refined`).
  « Lancés derrière le flux » : l'affinage parti au lancement de la paire (plus bas) ; « écartés » : de
  ceux-là, ceux qui ont dû repartir de la caméra de NVOF. Les temps de l'hôte sont ceux du thread du lecteur.
- Profil GPU de fin, quatre étapes. « carte de la scène : lumas », sur la voie, « carte de la scène :
  tuiles », la réduction sur les SM, et « carte de la scène : caméra », l'affinage sur les SM : avec les
  temps de l'hôte, c'est tout ce que la carte ajoute à chaque image. « carte de la scène : flux (NVOFA) » : le bloc matériel du flux optique, qui tourne à côté des SM ;
  c'est la latence de la carte (le temps avant qu'elle soit relue), pas un coût par image. Les deux mesures
  de la file de NVOF comptent aussi l'attente de SM libres : sa priorité est la plus haute, mais un bloc de
  SPAN ou de RIFE déjà lancé finit d'abord. Passage 10 (19 novembre, une seule étape « NVOF et tuiles » à
  l'époque) : 0,62 ms en médiane, 1,15 au 95e centile, lumas 0,01 ms. La porte se juge au temps par image
  avec et sans `--scene-map` (banc A/B, `tools/bench/banc.sh`).

## Le chemin d'une paire

1. `UpscaleStage::look_ahead` (avec RIFE ou GMFSS, l'image suivante ingérée en avance), et `drba_next` avec
   DistilDRBA, lancent la paire (N, N+1) par `KernelOps::scene_flow_start`, une image entière avant que
   l'étage en décide. Sinon (sans interpolation, image suivante pas encore décodée), `scene_pair` lance
   (N−1, N) à l'image N, et sa carte n'arrive qu'à l'image suivante (`SceneMap::latest`, pas dans
   `PairFacts`). Avec l'image suivante déjà là, `scene_pair` ne lance rien en retard : au passage 9, chaque
   paire lancée en retard prenait NVOF à la paire lancée en avance, qui sautait, et ainsi de suite (460
   sautées sur 473, 38 posées). Une paire n'est lancée qu'une fois, jamais d'une époque à l'autre.
2. Backend TensorRT (`src/gpu/cuda/backend/scene_flow.h`) : trois tampons de luma tournent. Une paire qui
   suit la précédente reprend la luma de b de celle-ci comme a, et le noyau `scene_luma` n'écrit, sur la voie
   courante, que la luma 8 bits de b (BT.709, boîte d × d, d = `scene_flow_divisor`, au plus 720 lignes),
   directement dans le tampon que l'avant-dernière paire a lu : aucun pas de ligne de NVDEC, et NVOF n'est
   attendu que deux paires en arrière. Si même celle-là n'est pas finie, la paire n'est pas lancée (ticket 0,
   « sautée » au journal de fin) et rien n'attend. Sur la file de NVOF (priorité la plus haute, pour que la
   réduction ne fasse pas la queue derrière SPAN et RIFE), qui attend la luma : le flux (grille 4, réglage
   moyen, coût 8 bits), la réduction par tuile (`scene_tiles`, sommes entières), la copie vers une mémoire
   hôte épinglée. Aucune attente, ni de l'hôte ni de la voie. La bibliothèque de NVOF n'est jamais
   déchargée (le pilote garde ses fils) ; sous Windows, elle n'est cherchée que dans System32.
3. Dès que les tuiles sont relues (`scene_flow_take`, sans attente, à chaque image de sortie par
   `SceneMap::advance` et quand l'étage décide), la caméra est ajustée sur leurs vecteurs
   (`kernels::ref::scene_camera_of`), puis affinée sur la file de NVOF (`scene_camera_start`, voir « La
   caméra »). L'affinage relu (`scene_camera_take`), `kernels::ref::scene_motion_of` tire la carte avec la
   caméra affinée.
4. `UpscaleStage::scene_pair`, quand l'étage décide de la paire : la carte, posée dans `PairFacts::scene`.
   Tuiles pas relues à temps : `scene` reste vide pour cette paire. Caméra pas encore affinée : la carte
   part avec celle de NVOF.

Les indices temporels de NVOF repartent à chaque paire qui ne suit pas la précédente (début, saut, nouvelle
époque, début d'export), après une coupe jugée par la chaîne et après une coupe probable de la carte. Une
autre taille d'image refait la session.

## Ce que lisent les clients

`src/kernels/spec/scene_flow.h` : `SceneMotion`, ses tuiles, la caméra et les drapeaux ; les seuils dans
`kernels::scene`, que règle la porte 2.3 (premier passage plus bas). `drawing_held` ne se dit que sur preuve : un effet
compte comme ce qui bouge, et sans caméra ajustée (aplat), toute tuile changée l'éteint. Règle absolue :
`scene` absente donne exactement le comportement d'aujourd'hui, image pour image.

## La caméra

Méthode de la porte 2.1 (banc `tools/bench/qualite/nvof_mesures.py` : `camera`, `refine_affine`,
`photo_residual`), sur 391 paires sur 407 sous 0,5 pixel d'écart à 720p avec au moins 40 % de décor.

- **Ajustement** (`ref::scene_camera_of`, constantes `kFit*`) : affine à 6 paramètres sur les vecteurs moyens
  des tuiles candidates (texture d'au moins `kFitTexture`, trois quarts de vecteurs sûrs, vecteurs cohérents),
  pondérées par `1 / (1 + coût / coût médian)` et par leur texture. Départ sur la translation médiane, 8 tours ;
  à chaque tour, rejet des tuiles à plus de 3 écarts médians (estimés sur celles à moins de 2 pixels), seuil
  borné entre 0,75 et 2 pixels à 720p. Le décor, ce sont les tuiles gardées.
- **Affinage** (`kernels/spec/scene_camera.h`, référence `ref::scene_camera_refine`, port
  `scene_camera_kernels.cu`) : NVOF est biaisé sous le pixel sur les caméras lentes (sur un panoramique lent du
  banc, il invente un cisaillement d'environ 2 pixels à 720p). Lucas-Kanade global sur les pixels du décor
  retire ce biais : Gauss-Newton sur les 6 paramètres, poids de Huber, 3 pas sur les lumas du flux floutées en
  7 × 7, puis 3 en 3 × 3, un pixel sur deux en x et en y. La caméra de NVOF reste si H est mal conditionné ou si
  le décor a moins de 1000 pixels. Tout se fait sur la file de NVOF, sans retour à l'hôte entre les pas : les
  lumas floutées, un histogramme des écarts (la médiane du poids de Huber), les sommes pondérées, et un fil qui
  résout. Les lumas de la paire restent dans leurs tampons jusqu'à l'affinage : une paire plus récente qui les
  réécrirait attend (« relâché » reposé après l'affinage), et un affinage demandé trop tard n'est pas lancé.
- **Affinage à l'heure** (`app/play/scene_map.cpp`) : dans un plan, l'affinage part dès le lancement de la
  paire, enfilé derrière son flux sur la file de NVOF, depuis la dernière caméra du plan (affinée si elle est
  relue, sinon celle de NVOF de la paire d'avant) : il n'attend plus que l'hôte relise les tuiles, et finit
  avec elles. Lancé seulement après la relecture, il arrivait trop tard sur un tiers des paires au passage du
  19 novembre (177 sur 473 sans calques). Il est gardé si sa vérification sur l'image (`SceneCamera::check`)
  reste sous 0,05 pixel du flux, ou sous deux fois celle de la dernière caméra affinée du plan sans passer
  0,15, et s'il reste à 16 pixels du flux de la caméra de NVOF de sa paire ; sinon il repart des tuiles
  relues, depuis la caméra de NVOF, comme la première paire d'un plan. Rejeu de dix extraits (`carte_rejeu`,
  ligne `tot`, 1 590 paires, 1 479 départs) : 1 346 gardés, dont 3 vérifiés à plus de 0,05 pixel au-delà de
  l'affinage parti de NVOF. Garder ceux qui restent à 2 pixels de NVOF, sans regarder l'image, en gardait
  1 416 dont 46 ainsi faux (un départ qui glisse vers une autre solution, souvent une image sur trois sur un
  dessin en trois), et écartait des départs justes là où NVOF biaise la caméra de 2 à 9 pixels. Un affinage
  pas fini quand l'étage décide de sa paire est relu plus tard (`SceneMap::last_refined`, pixels de travail) :
  les calques reprennent la caméra d'une paire voisine. Sur la RTX 4070 Ti (19 novembre, RIFE 4.25 Auto et
  SPAN, trois tours de 60 s) : caméra affinée sur toutes les paires ajustées (Re:Zero 16:00, 1 339 sur 1 339 ;
  panoramique, 424 sur 424), aucun affinage perdu (les 57 et 22 arrivés trop tard sont tous relus ensuite),
  calques sur 74 à 83 % des paires. L'affinage coûte 0,85 ms en médiane (p95 1,95 ms) sur la file de NVOF,
  l'hôte 69 µs par paire lancée et 214 µs par carte tirée ; la lecture avec la carte reste à 131 i/s contre
  130 sans (p95 du retard 2,0 à 2,5 ms contre 2,4 à 2,5).
- **Vérification**, celle que mesure la porte : sur chaque tuile (décor ou non), deux pas de Lucas-Kanade sur le
  décalage qui reste ; une tuile texturée dans un seul sens se mesure en travers du bord. `SceneMotion::residual`
  est alors la médiane de ces décalages sur les tuiles du décor (`residual_tiles` tuiles), sinon l'écart des
  vecteurs. Sur les autres tuiles, la même mesure sépare le dessin du décor (« Par l'image », plus bas).

## Les classes des tuiles

Décor : la tuile suit le modèle de la caméra. Dessin : elle s'en écarte, ou ses vecteurs s'éparpillent. Effet :
elle ne correspond plus et change fort, ou une lumière y change sur place. Incertain : le flux ne tranche pas
(vecteurs sans correspondance sans effet, mouvement sans caméra ajustée, et les aplats ci-dessous).

- **Par l'image**, quand la caméra est affinée (le cas courant) : l'écart après la caméra de chaque tuile tranche,
  quoi que disent ses vecteurs. Deuxième passage de la porte 2.3 : sur les dialogues, les personnages finissaient
  en décor, parce qu'une bouche ou des yeux qui changent ne déplacent pas le vecteur moyen de NVOF. Au premier
  pas de la vérification, sur le modèle seul, un pixel bouge si son écart |e| dépasse ce qu'un décalage d'un
  demi-pixel à 720p explique (`kMovingCodes` + 0,5 × |∇|, `spec/scene_camera.h`). La tuile bouge hors de la
  caméra si au moins `kMovingShare` (10 %) de ses pixels bougent, ou si Lucas-Kanade la recale d'au moins
  `kShiftMoving` (1 pixel à 720p) dans un sens mesurable. Elle bouge : effet si ses vecteurs coûtent (coût moyen
  d'au moins `kEffectCost`) sur au moins `kEffectShare` de pixels qui bougent, dessin sinon. Elle ne bouge pas :
  décor, aplat compris. Une tuile qui bouge compte aussi pour « changée » : sous une caméra tenue, la paire où la
  bouche change n'est plus une paire sans preuve. Un personnage entièrement immobile reste décor : rien n'y
  bouge, la chaîne le traite comme le décor sans rien perdre. Seuils réglés sans mesure, à revoir sur la table
  « Écart après la caméra » de `carte_lecteur.py` (le décor y donne le bruit). Sans affinage relu (affinage trop
  tard, trop peu de décor, coupe), les règles des vecteurs ci-dessous restent.

- **Lumières** (effets lumineux, avant toute autre règle, jamais sur un fondu). Instantanés de Re:Zero à 16:20
  et 16:43 (une lueur, une aura) : aucune tuile en effet, alors que les éclats HDR en dépendent
  (`app/play/effect_book.cpp`). Une lueur coûte peu à NVOF et ne déplace pas son vecteur ; seule la règle du coût
  la voyait, presque jamais. La vérification compte aussi, parmi les pixels qui bougent, ceux qui
  s'éclaircissent et ceux qui changent d'un bord net (|∇a − ∇b| d'au moins `kSharpEdge`, 8 codes par pixel), et
  l'écart qui reste au dernier pas, après le recalage de la tuile (`settled`). Une tuile est une lumière (effet,
  de confiance l'accord) si :
  - au moins `kMovingShare` de ses pixels bougent, et presque tous dans le même sens : l'accord
    |éclaircis − assombris| / qui bougent atteint `kLightAgree` (0,6, 80 % d'un côté). Un dessin qui se
    déplace assombrit d'un côté ce qu'il éclaircit de l'autre ;
  - ni elle ni ses huit voisines ne changent d'un bord net (plus de `kLightSharp`, 25 %, de leurs pixels qui
    bougent) : une bouche qui s'ouvre ou un œil qui cligne vont aussi d'un seul sens, sur place, mais d'un bord
    net ; le cœur d'une grande bouche change sans bord, les tuiles de son bord le trahissent ;
  - le recalage n'explique pas le changement (au moins `kLightKept`, la moitié, de l'écart reste) : un dégradé
    qui glisse va lui aussi d'un seul sens sur une tuile ;
  - son vecteur suit la caméra : un objet clair qui entre dans une tuile sombre l'éclaircit aussi, mais il
    entraîne ses vecteurs.

  Une lueur tout à fait immobile ne se voit pas ainsi : rien n'y change d'une image à l'autre. C'est une limite
  tenue, pas un défaut à corriger : sur les captures étiquetées du 19 novembre, la corne et le halo de Re:Zero
  sont parfaitement fixes, et la bougie se recale (1,7 pixel) assez pour que le test `kLightKept` l'écarte.
  Relâcher les règles pour les attraper faisait environ 2 400 faux effets sans voir la bougie. Les éclats
  fixes, c'est le grade qui les trouve lui-même (`docs/hdr-anime.md`) ; la carte ne donne que les lumières qui
  changent. Seuils réglés sans mesure, à revoir sur la table « Lumières » de `carte_lecteur.py`.

- **Coupes** (`cut_likely`). Une paire qui enjambe une coupe ne dit rien de la scène : ses vecteurs relient deux
  plans sans rapport. La carte la juge coupée si la luminosité saute partout d'au moins `kCutJump` (32 codes, ce
  qu'aucun fondu ne fait d'une image à l'autre) ou si, hors fondu, au moins `kCutUnmatched` des vecteurs sont
  sans correspondance ; la chaîne peut aussi la dire coupée (`SceneMap::cut`, depuis `UpscaleStage`).
  `ref::mark_cut` pose alors une carte neutre : pas de caméra, toutes les tuiles incertaines et changées, aucun
  drapeau tenu. Avant, une coupe de Re:Zero passait pour un fondu (la luminosité glissait d’un seul sens) et ses
  vecteurs inventaient un effet vert sur un menton, que le masque des éclats gardait 700 ms ; corrigé, le PC
  ne le voit plus (commit 9fff1e32). Au rejeu, 5 coupes trouvées sur les extraits (aucune avant), et les
  effets de la lueur et de l'aura passent de 600 à 374 et de 330 à 183 tuiles : le reste venait des coupes.

- **Aplats** (texture sous `kTextureFloor`). NVOF n'y voit rien, mais rien n'y bouge de visible : un aplat qui
  ne contredit pas la caméra est décor, pleinement sûr sous `kStillDiff` d'écart |b − a|, plus du tout à
  `kChangeDiff`. Un aplat qui change plus, ou dont le vecteur contredit la caméra, reste incertain : ce vecteur
  vient des bords voisins, il ne prouve pas un dessin.
- **Lissage** (`ref::SceneHistory`). Une tuile ne change de classe que vue deux paires de suite ; entre les deux,
  la classe tenue reste, à 0,4 fois sa confiance (`kContradicted`) : jamais sûre pour un client (seuil 0,5).
  Une paire où la tuile ne change pas sous une caméra tenue ne compte pas : les images doublées d'une animation
  en deux ou en trois ne ramènent pas un dessin au décor une paire sur deux. Une coupe vide l'historique.

Les drapeaux (`drawing_held`, `changed`) restent bruts, non lissés.

## La porte (groupe de tests 2, sur CachyOS)

Les commandes des mesures du PC (banc A/B, captures des éclats, carte sur les extraits) :
[`reports/carte-de-la-scene-mesures-pc-2026-11-19.md`](reports/carte-de-la-scene-mesures-pc-2026-11-19.md).

Pendant une lecture SPAN + RIFE, le temps par image ne monte pas de plus de 0,3 ms avec `--scene-map` (preuve
que le bloc NVOFA tourne à côté). Sur les extraits du banc 2.1 (porte 2.3) : classe décor contre dessin juste
sur 95 % des tuiles sûres, stable sur les plans fixes.

Le banc 2.1 (`tools/bench/qualite/nvof_scene.py`) classe ses tuiles avec son propre classeur : ses parts
« incertain » et ses bascules ne mesurent pas la carte du lecteur. La porte 2.3 se mesure avec
`tools/bench/qualite/carte_lecteur.py`, qui lit chaque extrait comme le lecteur (taille du flux, toutes les
paires voisines, doublons compris, réglage moyen, indices temporels) et rejoue la carte par `carte_rejeu`
(`kernels::ref`, lissage compris, construit à part) : parts des classes et des tuiles sûres, bascule sur toutes
les paires et sur les plans fixes, diagnostic de l'ajustement de la caméra (tuiles candidates contre le minimum,
tuiles écartées par l'aplat, le coût ou l'éparpillement, texture), queue des coûts de NVOF (pour régler
`kSceneCostHigh`), vues PNG pour juger la classe des tuiles sûres.

Premier passage (19 novembre, 6 panoramiques et 2 dialogues, avant les changements ci-dessous) : aucune caméra
ajustée sur 4 panoramiques sur 6, d'où 83 à 97 % d'incertain, ni dessin ni effet. Deux causes. Une tuile
n'entrait dans l'ajustement qu'avec une texture d'environ un trait net entier (6 codes par pixel), et il en
fallait 15 % de l'image (306 tuiles), alors que ces scènes sombres (luma moyenne 31 à 49) n'en ont que 2 à 355
au-dessus de 3 codes ; le banc 2.1 y ajuste sa caméra sur tout bloc où la luma varie d'un code. L'ajustement
prend maintenant les tuiles dès 1,5 code (`kFitTexture`) sur 3 % de l'image, l'aplat est sous 3 codes. Et aucun
vecteur n'était « sans correspondance » : NVOF ne donne presque jamais un coût au-dessus de 32 (99e centile 20,
99,9e 30), le seuil était à 160 ; il est à 24.

Passage du 19 novembre au soir (RTX 4070 Ti, commit 9baf5c9c, les 8 extraits plus la lueur de 16:17 et l'aura
de 16:40) : 80 à 99,6 % de tuiles sûres (23 à 67 % au passage précédent), 0 à 19 % d'incertain, bascule de 0 à
2,1 % ; dessin 25 % sur le premier panoramique, 17 % sur le cinquième, 0 sur les dialogues ; caméra affinée
sur toutes les paires ajustées, écart sur l'image de 0,00 à 0,12 pixel. Les effets sont là, mais sur peu de
l'image : 0,1 à 0,3 % des tuiles, sur 17 paires sur 159 de l'aura et 34 de la lueur au rejeu des mêmes
vecteurs ; le tableau de `carte_lecteur.py` les arrondissait à 0, il les donne maintenant au dixième, avec
la part des paires qui en ont. Reste, pour la porte, à juger sur les vues la classe des tuiles sûres.
