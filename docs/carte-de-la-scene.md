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
- Journal de fin : `carte de la scène : N paires lancées, N relues, N posées sur leur paire, N abandonnées ;
  caméra ajustée sur N, N coupes probables, N fondus`. Le profil GPU de fin montre deux étapes, « carte de
  la scène : lumas » (sur la voie) et « carte de la scène : NVOF et tuiles » (sur la file de NVOF).

## Le chemin d'une paire

1. `UpscaleStage::look_ahead` (avec RIFE ou GMFSS, l'image suivante ingérée en avance) lance la paire
   (N, N+1) par `KernelOps::scene_flow_start`. Sinon (sans interpolation, DistilDRBA, image suivante pas
   encore décodée), `scene_pair` lance (N−1, N) à l'image N, et sa carte n'arrive qu'à l'image suivante
   (`SceneMap::latest`, pas dans `PairFacts`). Une paire n'est lancée qu'une fois, jamais d'une époque à
   l'autre.
2. Backend TensorRT (`src/gpu/cuda/backend/scene_flow.h`) : si NVOF n'a pas fini la paire d'avant, la
   paire n'est pas lancée (ticket 0, « abandonnée » au journal de fin) et rien n'attend. Sinon, sur la voie
   courante, le noyau `scene_luma` écrit les deux lumas 8 bits du flux depuis le RGB de
   travail (BT.709, boîte d × d, d = `scene_flow_divisor`, au plus 720 lignes), directement dans les tampons
   de NVOF : aucun pas de ligne de NVDEC. Sur la file de NVOF, qui attend ces lumas : le flux (grille 4,
   réglage moyen, coût 8 bits), la réduction par tuile (`scene_tiles`, sommes entières), la copie vers une
   mémoire hôte épinglée. Aucune attente, ni de l'hôte ni de la voie. La bibliothèque de NVOF n'est jamais
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

## La porte (groupe de tests 2, sur CachyOS)

Pendant une lecture SPAN + RIFE, le temps par image ne monte pas de plus de 0,3 ms avec `--scene-map` (preuve
que le bloc NVOFA tourne à côté). Sur les extraits du banc 2.1 (porte 2.3) : classe décor contre dessin juste
sur 95 % des tuiles sûres, stable sur les plans fixes.
