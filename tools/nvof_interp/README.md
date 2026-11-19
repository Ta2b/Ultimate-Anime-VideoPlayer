# FlowJaNai : interpolation d'anime guidée par le flux optique NVIDIA (expérimental)

Le flux optique matériel des RTX (NVOF) suit très bien les grands déplacements et ne coûte presque
rien au GPU, mais ses vecteurs, un par bloc de 4 × 4, sont trop grossiers pour être utilisés tels
quels sur des traits d'anime. FlowJaNai part de ce flux au lieu de l'estimer de zéro : le réseau ne
fait qu'affiner le mouvement et estimer le masque de mélange, sur trois étages (1/4, 1/2, 1/1).
Voir `model.py` pour l'architecture.

La sortie est le contrat de la variante « mouvement » de RIFE dans le lecteur
(`src/kernels/spec/warp.h`) : le warp, le mélange, « warp après SPAN » et la passe unique au
milieu de la paire servent tels quels. Côté lecteur il restera à brancher la session NVOF (le code
retiré par le commit 97feba0, `nvof_session.cpp`) devant le moteur, dans le backend TensorRT.

## Étapes (sur le PC, RTX)

1. **Données.** ATD-12K (dépôt AnimeInterp, licence de recherche) : `train_10k/` et
   `test_2k_540p/`, un dossier par triplet `frame1`, `frame2`, `frame3`.
2. **Outil NVOF.** Les en-têtes de l'Optical Flow SDK de NVIDIA (licence acceptée au
   téléchargement, par toi) :
   ```sh
   cmake -S tools/nvof_interp/nvof_dump -B build/nvof_dump -DNVOF_SDK_INCLUDE=<sdk>/NvOFInterface
   cmake --build build/nvof_dump
   ```
3. **Flux.** Une fois pour toutes, `nvof.npy` à côté de chaque triplet :
   ```sh
   python -I tools/nvof_interp/precompute_nvof.py atd12k/train_10k --tool build/nvof_dump/nvof_dump
   python -I tools/nvof_interp/precompute_nvof.py atd12k/test_2k_540p --tool build/nvof_dump/nvof_dump
   ```
4. **Entraînement.**
   ```sh
   python -I tools/nvof_interp/train.py --train atd12k/train_10k --val atd12k/test_2k_540p \
       --out runs/flowjanai
   ```
   `--resume` reprend depuis `runs/flowjanai/last.pt`. Le PSNR de validation se compare à celui
   de RIFE 4.25 sur les mêmes triplets (référence à mesurer avant de conclure).
5. **Export.** `export_onnx.py runs/flowjanai/best.pt --out modeles/flowjanai.onnx` : FP16, vérifié
   contre le modèle PyTorch à deux tailles.

## Banc de la porte 2.1 (hors lecteur)

Le flux NVOF sépare-t-il le décor des personnages assez bien pour un rendu par calques ?
`tools/bench/qualite/nvof_scene.py` passe des extraits typés dans `nvof_dump` (grilles 4, 2, 1 en 540p,
720p et 1080p, carte de coût, flux CUDA propres à la session) et rend le temps GPU par appel, le gain
du recalage, le coût sur aplats, traits et grain, l'ajustement de la caméra, les tuiles décor contre
dessin avec des vues PNG, et le verdict de la porte. L'écart du décor au modèle de caméra, celui de la
porte, se mesure sur l'image (Lucas-Kanade sur les tuiles de décor, `nvof_mesures.py`) : les vecteurs
NVOF sont bruités d'environ 0,3 px sur un décor fixe et un personnage lent les tire, ce qui faisait
rater les caméras lentes (4070 Ti, 19 novembre). Une tuile texturée dans un seul sens (un bord droit,
un décor rayé que la caméra longe) se mesure en travers du bord, le seul écart visible ; l'image ne
passe la main aux vecteurs qu'à moins de 3 tuiles mesurables. Le banc juge cinq pistes du modèle de
caméra au même critère : nvof, traits (blocs de traits seuls), lk (l'affine NVOF affinée par
Lucas-Kanade global sur l'image, qui corrige le biais sous-pixel de NVOF sur les caméras lentes),
traits_lk et lk_bords (lk, avec un décor où un bloc texturé dans un seul sens se juge en travers de son
bord et où les blocs que la caméra sort de l'image ne comptent pas). `--garder` sauve images, flux et coûts
des panoramiques dans `SORTIE/flux` pour régler le banc hors ligne. Sur les panoramiques, le mode pyramide
(`--pyramide 360:720`, par défaut) passe la caméra d'une passe en 360p en indices externes
(`nvof_dump --hint-grid 8`) à la passe 720p, et compare avec et sans par vitesse de caméra. Les en-têtes de NVIDIA sont dans
`third_party/nvidia-optical-flow` : rien à télécharger.

```sh
cmake -S tools/nvof_interp/nvof_dump -B build/nvof_dump -DCMAKE_BUILD_TYPE=Release
cmake --build build/nvof_dump
python3 -I tools/bench/qualite/nvof_scene.py --outil build/nvof_dump/nvof_dump --sortie ~/nvof-banc \
    --extrait panoramique=pan.mkv@12:30 --extrait dialogue=dlg.mkv@3:10 --extrait sakuga=sak.mkv@0:41 \
    --extrait fondu=... --extrait flash=... --extrait zoom=... --extrait parallaxe=... --extrait pluie=...
```

Un seul banc à la fois sur la carte (rien d'autre sur le GPU), sur la 4070 Ti puis la 4050.

Résultat (4070 Ti, 19 novembre, grille 4 × 4 en 720p, 7 panoramiques, 407 paires où la caméra bouge) : la
porte passe. Un appel NVOF coûte 1,0 ms en 720p (0,35 ms en 360p).

| Piste | Paires bonnes | Caméras lentes (0,5 à 15 px) |
|---|---:|---:|
| nvof | 335 (82 %) | 75 % |
| traits | 368 (90 %) | 88 % |
| traits_lk | 381 (94 %) | 96 % |
| lk | 393 (97 %) | 98 % |
| lk_bords | 403 (99 %) | 100 % |

lk_bords est la méthode retenue pour le lecteur (`docs/carte-de-la-scene.md`, « La caméra »). Restent 4
paires de Spy 4:53 : un panoramique de 87 px à 39 % de décor, et trois paires (54, 17 et 8 px) dont l'affine
part en rotation et cisaillement, sans doute des images de traîne ou un tremblé plutôt qu'un décor rigide.
Le mode pyramide n'aide pas : les indices font baisser les caméras lentes (62 % contre 75 % sans) et ne
sauvent pas les panoramiques de plus de 45 px ; il reste dans le banc pour mémoire.

## Ce qui reste ouvert

- Les flux sont calculés sans indices temporels (paires indépendantes) ; le lecteur devra faire de
  même pour retrouver la distribution d'entraînement, ou réentraîner avec des séquences.
- ATD-12K n'a que t = 1/2 ; d'autres instants demandent des séquences plus longues (à ajouter au
  jeu de données), sinon le lecteur garde la passe au milieu et la loi de mélange du warp.
- Le modèle n'a jamais tourné : les tailles des étages (`FlowJaNai.STAGES`) sont un point de
  départ, à régler sur le PC selon le coût mesuré face à RIFE 4.25 lite.
