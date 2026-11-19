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
- Interrupteur : `--scene-map` (essai). Éteint par défaut tant que la porte 2.2 n'est pas passée.
- Journal de fin : `carte de la scène : N paires lancées, N relues, N posées sur leur paire ; N sautées (NVOF
  occupé), N relues trop tard, N perdues ; caméra ajustée sur N, N coupes probables, N fondus`. « Posées » est
  ce qui compte : avec RIFE, presque toutes les paires lancées. « Trop tard » : la carte n'était pas relue
  quand l'étage a décidé de sa paire. Le profil GPU de fin montre deux étapes, « carte de la scène : lumas »
  (sur la voie, le seul coût ajouté au temps par image) et « carte de la scène : NVOF et tuiles » (sur la
  file de NVOF : surtout le bloc NVOFA, qui tourne à côté des SM ; c'est la latence de la carte, pas un coût
  qui s'ajoute). La porte se juge au temps par image avec et sans `--scene-map`.

## Le chemin d'une paire

1. `UpscaleStage::look_ahead` (avec RIFE ou GMFSS, l'image suivante ingérée en avance) lance la paire
   (N, N+1) par `KernelOps::scene_flow_start`, une image entière avant que l'étage en décide. Sinon (sans
   interpolation, DistilDRBA, image suivante pas encore décodée), `scene_pair` lance (N−1, N) à l'image N, et
   sa carte n'arrive qu'à l'image suivante (`SceneMap::latest`, pas dans `PairFacts`). Avec l'image suivante
   déjà là, `scene_pair` ne lance rien en retard : au passage 9, chaque paire lancée en retard prenait NVOF à
   la paire lancée en avance, qui sautait, et ainsi de suite (460 sautées sur 473, 38 posées). Une paire
   n'est lancée qu'une fois, jamais d'une époque à l'autre.
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
3. `UpscaleStage::scene_pair`, quand l'étage décide de la paire : relecture sans attente
   (`scene_flow_take`), puis `kernels::ref::scene_motion_of` tire la carte, posée dans `PairFacts::scene`.
   Pas finie à temps : `scene` reste vide pour cette paire.

Les indices temporels de NVOF repartent à chaque paire qui ne suit pas la précédente (début, saut, nouvelle
époque, début d'export), après une coupe jugée par la chaîne et après une coupe probable de la carte. Une
autre taille d'image refait la session.

## Ce que lisent les clients

`src/kernels/spec/scene_flow.h` : `SceneMotion`, ses tuiles, la caméra et les drapeaux ; les seuils dans
`kernels::scene`, non mesurés, à régler sur le banc 2.1. `drawing_held` ne se dit que sur preuve : un effet
compte comme ce qui bouge, et sans caméra ajustée (aplat), toute tuile changée l'éteint. Règle absolue :
`scene` absente donne exactement le comportement d'aujourd'hui, image pour image.

## Les classes des tuiles

Décor : la tuile suit le modèle de la caméra. Dessin : elle s'en écarte, ou ses vecteurs s'éparpillent. Effet :
elle ne correspond plus et change fort. Incertain : le flux ne tranche pas (vecteurs sans correspondance sans
effet, mouvement sans caméra ajustée, et les aplats ci-dessous).

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
