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

## Ce qui reste ouvert

- Les flux sont calculés sans indices temporels (paires indépendantes) ; le lecteur devra faire de
  même pour retrouver la distribution d'entraînement, ou réentraîner avec des séquences.
- ATD-12K n'a que t = 1/2 ; d'autres instants demandent des séquences plus longues (à ajouter au
  jeu de données), sinon le lecteur garde la passe au milieu et la loi de mélange du warp.
- Le modèle n'a jamais tourné : les tailles des étages (`FlowJaNai.STAGES`) sont un point de
  départ, à régler sur le PC selon le coût mesuré face à RIFE 4.25 lite.
