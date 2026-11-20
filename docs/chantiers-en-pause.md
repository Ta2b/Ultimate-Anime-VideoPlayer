# Chantiers mis en pause par Tatoub

On ne relance aucun de ces chantiers sans son accord explicite. Ce qui est déjà fusionné dans main reste en place.

Repris le 19/11 à 16:35 (Tatoub : « on vise tout ça pour RIFE aussi ») : la suite du RIFE natif (78bfe89 sur `claude/project-thread-i22zy2`), le GMFlow natif avec attention fondue (#930) et Winograd sur SPAN et GridNet (toujours en dernier). Objectifs et leviers dans la section « Viser les plafonds théoriques » du plan. Le PC sert d'abord aux chantiers dont la porte n'est pas passée.

## Reprise du 20 novembre

- **Cadence 4K**. Sur Re:Zero à 16:00, SPAN Balanced FP16 et RIFE 4.25 avec HDR passent de 110,7–114,0 à
  116,7 i/s en mouvement avec le warp Catmull-Rom optimisé. La médiane du warp passe de 1,61–1,62 à 1,35 ms.
  Le calcul partage les coefficients et les indices entre les trois plans RGB. L'avance SPAN expérimentale
  donne 119,1 i/s, avec 52 pertes à l'affichage contre 31 dans le passage précédent. Avec la grille NVOF
  compacte et une marge VRR à zéro, un passage atteint 121,4 i/s en mouvement, 107,5 i/s sur l'extrait.
  Le warp fusionné vers l'anneau puis la composition PQ sans seconde mesure du pic ramènent le service vers
  5,4 ms. Sur 36 s, le dernier témoin comparable donne 136,6 i/s en mouvement et l'essai 154,1. Une
  profondeur de présentation de 3 donne 162,5–162,9 i/s confirmées par Wayland sur la fenêtre en mouvement
  de 17 s, et jusqu'à 164,5 sur 2 s sans saut. Voir [`auto-libre.md`](auto-libre.md#rejeu-4k-hdr-du-20-novembre).
- **GMFSS**. L'IFNet d'Union a maintenant quatre blocs de convolutions CUDA. L'essai à 960×576 les retient
  à 1,35 ms contre 1,47 ms et 73,19 dB sur sa sortie. GridNet transcrit ses six entrées de caractéristiques
  par tuiles contiguës, bit pour bit. La fusion passe de 22,18 à 20,16 ms de médiane en lecture.
  Les groupes de plans qui tiennent dans le L2, le dénominateur FP32 partagé et la marge de tuile plus courte
  ramènent la médiane des splats de 8,16–8,41 à 4,00 ms au meilleur passage. GMFlow possède une chaîne CUDA
  complète, qui tourne vers 8 ms une fois chaude à 960×576. Le contrôle de fidélité garde encore son flot TensorRT.
- **NVOF**. L'essai aller-retour compact obtient 352 résultats pour 360 paires lancées, dont 288
  rendus à la paire correspondante, avec 0,43 ms de latence moyenne. La relecture passe de 259 200 à
  12 240 octets à la taille interne 960×540. Le contrôle recense 72 tuiles de bord suspectes ; huit sont
  corroborées par le suivi LK. L'option `--nvof-layer-guard` rend 90 tuiles de bord à RIFE sur le passage
  SPY×FAMILY. Elle exige une corroboration sous la caméra affinée de la même paire.
- **Couleurs de l'export**. Le grade attend du RGB. L'export l'applique désormais au stade
  `PREOUTPUT` de libplacebo avant le codage YCbCr. Sur une mire avec un look, l'écart maximal
  face au témoin RGB converti séparément passe de 76 à 1 code 10 bits. Les exports de la mire
  en FFV1 et HEVC Vulkan décodent tous 48 images BT.2020/PQ en plage limitée.
- **Éclats et NVOF**. Le [rejeu des lueurs](hdr-anime.md#rejeu-des-lueurs-fixes-sur-la-source-sdr)
  et le banc des paires aller-retour dans [`tools/nvof_interp`](../tools/nvof_interp/README.md)
  documentent les témoins qui empêchent d'élargir le masque HDR ou de remplacer des passes de RIFE.
- **SPAN Performance FP8**. Les deux convolutions ReLU de chaque bloc C48 utilisent les poids E4M3 plats et
  reproduisent les groupes de sommes de CUTLASS. Sur une mire saturée et une image 1080p, le réseau donne les
  mêmes codes, en sommes FP16 et FP32. L'essai du lecteur retient 5,73 ms contre 7,70 ms à sommes FP32.
  Dans le graphe CUDA du banc d'image, les sommes FP16 donnent 4,92 ms contre 6,19 ms.

## Points d'arrêt du 19 novembre

| Depuis | Chantier | Fil | État au moment de la pause | Pour reprendre |
|---|---|---|---|---|
| 19/11, 14:44 | SPAN FP8 en noyaux natifs (cran Performance) | « Battre AnimeJaNai sur SPAN en FP16 » | Convolutions FP8 (mma m16n8k32 E4M3) en cours d'écriture, rien de fusionné. Le FP8 CUTLASS reste en service | Commit 41a4a2f sur `claude/project-thread-avpwom` (sans PR), copie dans `/mnt/project-files/span-native/fp8-natif-en-pause.patch` ; note de reprise dans `docs/span-native16.md`, section « FP8 natif (en pause) », avec les six étapes qui restent |
| 19/11, 14:47 | Flux NVIDIA, perfectionnement après la porte 2.1 | « Évaluer le flux optique NVOF hors lecteur » | Porte 2.1 passée à 99 % (lk_bords). 4 paires ratées sur Spy 4:53 | Note dans `tools/nvof_interp/README.md`, section « Reprise » (#955) : paires 7, 12, 14 et 16, données, pistes dans l'ordre ; reste : extraire les 4 paires sur le PC, les étudier dans le cloud |
| 19/11, 14:54 | Couleurs fausses de l'export (pas de la lecture) | « Corriger les couleurs des exports » | Bug trouvé par le PC à 14:46 : chroma fausse en FFV1 comme en HEVC (fond bleu-gris en brun-orange), captures du lecteur justes. Tatoub : l'export n'est pas une priorité | Note de reprise du fil ; aucune demande au PC |
| 19/11, 15:37 | Test des horloges de SPAN (#941) | « Battre AnimeJaNai sur SPAN en FP16 » | Impossible : `sudo nvidia-smi -lgc` demande le mot de passe. Tatoub : « on verra ça plus tard » | Tatoub pose la règle sudoers limitée à `nvidia-smi -lgc`/`-rgc` (commande donnée dans le fil du plan, 15:36), puis le PC lance le test |
| 19/11, 18:04 | Éclats HDR sur les effets et lueurs | « Faire briller les effets en HDR » | Ce qui est fusionné reste : la carte de la scène nourrit le masque des effets (#954, #967, #969) et les lueurs vues dans l'image (#974). Essai du blanc des surfaces à 60 % (#977) annulé, retour à 100 % (#979). Cause : la corne et le halo sont des blancs purs de la source (code 255), comme un mur blanc. Ils sont déjà au plafond de sortie, et les lueurs de #974 ne s'y déclenchent pas au PC (écart bloc-zone trop faible, surtout sur le halo) | `docs/hdr-anime.md`, section des lueurs (« Chantier en pause ») : rejouer la passe anime sur la source de 16:23 et 16:45, mesurer l'écart bloc-zone dans la source, puis reconnaître une lueur à sa forme (petit cœur blanc dans un dégradé coloré). Captures PQ : `/mnt/project-files/eclats-effets3` et `eclats-effets4.tar.zst.part*` |
| 19/11, 18:04 | Reconnaissance des effets par la carte de la scène | « Intégrer NVOF et la carte de la scène » | #939 livre la carte allumée par défaut (caméra, décor, affinage à l'heure, coupes ; calques et plaque s'en servent) avec les éclats sur ses effets éteints (`PlayOptions::map_effects`, `--effect-highlights` en essai). La classe Effet reste calculée : le tri des paires et les calques la lisent comme un dessin. Le faux effet vert sur la coupe est corrigé (vérifié au PC sur 9fff1e32). Les effets restent rares (0,1 à 0,3 % des tuiles sur la lueur et l'aura), et une lueur immobile n'est pas visible pour la carte par construction | `docs/carte-de-la-scene.md`, règle des lumières et section des coupes. Pistes dans l'ordre : 1) étiqueter des lueurs qui changent (pas la corne ni le halo, fixes) et lire la table « Lumières » de `carte_lecteur.py` sur ces extraits ; 2) régler `kLightKept` sur des cas comme la bougie (recalée de 1,7 px) sans rouvrir les faux effets (relâcher la règle en faisait environ 2 400) ; 3) captures `--effect-highlights --effect-mask-view` aux moments à voir du journal de fin, puis rallumer par défaut si le vert tombe sur des effets |
| 19/11, 18:10 | GMFSS union natif vers les plafonds | « Rendre GMFSS union temps réel » | Fusionné dans main avec #959 : le natif par défaut (paire et fusion retenues à l'étape 0 du PC), le mélange, la forme en flux du GridNet, GMFlow PR 2 et 3 (pas branchées). Arrêtés : les splats par étapes dans le L2 (écrits, pas lancés sur une carte, gain modélisé 7,1 → 2 à 3 ms) et l'émulation de Winograd et du FP8 du GridNet (bases seules mesurées : natif f16 à 66,8 dB au pire). Queue fondue à écarter ; attention fondue de GMFlow au-delà des PR 2 et 3 pas commencée | `docs/gmfss-bilan.md`, partie 4, « En pause » : état de chaque levier et ordre de reprise (au PC d'abord le test `[gmfss]` des splats et Nsight). Patchs dans `/mnt/project-files/chantier-natif/en-pause/` ; copie du bilan dans `chantier-natif/bilan-gmfss.md` |
