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

## Mesure préalable : le flux NVOF vaut-il un guide ? (étape 1, novembre 2026)

Avant tout entraînement, `tools/bench/qualite/nvof_warp.py` fabrique l'image du milieu par warp seul avec le flux
NVOF (grilles 4, 2 et 1, calculé à 720p et à 1080p), et la compare à la vraie image, au même warp par le flux de
GMFlow (GMFSS union) et de RIFE 4.26, et à RIFE 4.26 tel que le lecteur le joue. Seuil : sur les paires de plus de
30 px, la meilleure configuration NVOF reste à 1 dB ou moins de GMFlow ; sinon la piste s'arrête.

```sh
# En-têtes publics de NVIDIA (nvOpticalFlowCommon.h, nvOpticalFlowCuda.h), ou ceux du SDK 5.0 téléchargé :
git clone https://github.com/NVIDIA/NVIDIAOpticalFlowSDK ~/nvof-sdk
cmake -S tools/nvof_interp/nvof_dump -B build/nvof_dump -DNVOF_SDK_INCLUDE=$HOME/nvof-sdk
cmake --build build/nvof_dump
python3 -I tools/bench/qualite/nvof_warp.py --video rezero-s4e18-1710.mkv --reseau rife_v4.26.onnx \
    --outil build/nvof_dump/nvof_dump --gmfss gmfss/roue/vsgmfss_fortuna --device cuda --lpips \
    --md nvof-ecart3.md --json nvof-ecart3.json
# puis la même chose avec --ecart 6 (paires espacées, grands mouvements) : --md nvof-ecart6.md
```

`nvof_dump` prend `--grid 1|2|4` et `--preset slow|medium|fast` (par défaut 4 et slow, comme avant) et écrit le
temps de chaque appel au matériel sur la sortie d'erreur.

## Ce qui reste ouvert

- Les flux sont calculés sans indices temporels (paires indépendantes) ; le lecteur devra faire de
  même pour retrouver la distribution d'entraînement, ou réentraîner avec des séquences.
- ATD-12K n'a que t = 1/2 ; d'autres instants demandent des séquences plus longues (à ajouter au
  jeu de données), sinon le lecteur garde la passe au milieu et la loi de mélange du warp.
- Le modèle n'a jamais tourné : les tailles des étages (`FlowJaNai.STAGES`) sont un point de
  départ, à régler sur le PC selon le coût mesuré face à RIFE 4.25 lite.

## Étape 2 : RIFE 4.26 guidé par le flux NVOF (novembre 2026)

L'étape 1 bis a montré que le flux NVOF (grille 4 à 720p) vaut celui de GMFlow comme guide sur les grands mouvements
(125 paires, +0,03 dB). L'étape 2 part de RIFE 4.26 tel quel et lui donne ce flux en entrée, avec des poids nouveaux
à zéro : au pas 0, le modèle est RIFE 4.26. Critère : battre RIFE 4.26 en LPIPS sur les paires de plus de 30 px des
passages du banc (0,175 en écart 3), qui restent hors de l'entraînement.

1. **Suites animées « sur un ».** Une douzaine de fenêtres de 30 s par vidéo, réparties sur toute sa durée ; au plus
   25 suites par vidéo et 150 par série ; les vidéos à une autre cadence que 23,976 i/s sont écartées ; les passages
   du banc (`holdout.json`) ne donnent rien. Images brutes (`frames.npy`), lues en mmap :
   ```sh
   python -I tools/nvof_interp/sequences.py --liste videos.txt --sortie /ssd/ajp-dataset/suites \
       --exclure holdout.json --travaux 4
   ```
2. **Flux de toutes les paires** d'écart 2 à 8 (`flows.npy`, `pairs.npy`, `mag.npy`) :
   ```sh
   python -I tools/nvof_interp/precompute_pairs.py /ssd/ajp-dataset/suites --tool build/nvof_dump/nvof_dump
   ```
3. **Échantillons** (`seq_dataset.py`) : t quelconque, la moitié tirée parmi les paires de plus de 16 px.
