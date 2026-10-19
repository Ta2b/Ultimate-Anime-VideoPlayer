# w2/fondations 205f7d1 → 371b2a2 : à fusionner maintenant

La relecture cloud a avancé origin/w2/fondations de 205f7d1 à 371b2a2 (672/672, ASan compris).
Toute branche qui a fusionné w2/fondations la refusionne : `git fetch origin w2/fondations &&
git merge origin/w2/fondations`, puis s'adapte et relance ses tests.

## Changements de contrat

1. Backends ABI 3 : `enqueue()`, `signal()`, `wait()` rendent `std::expected` (refus nommés) ;
   points d'entrée `extern "C"` en `noexcept`, types `BackendAbiFn` / `BackendLayoutFn` /
   `BackendCreateFn` / `BackendDestroyFn` ; chaque backend exporte `ajp_backend_layout()` qui rend
   `kBackendLayout` ; le chargeur de gpu appelle `check_backend_abi(abi, layout)` avant
   `ajp_backend_create` (refus `AbiMismatch`). Un `Engine` meurt avant son backend.
2. FrameData : un lecteur d'image Vulkan passe par `begin_vulkan_access()` / `end_vulkan_access()`
   (verrou AVVkFrame, attente, signal, disposition et file enregistrées, comme hwdec_vulkan de
   mpv), plus par `ready_value` seul. Implémenté dans media/frame_data.cpp : media-gpu le garde.
3. Ring (core/ring.h, inclut messages.h) : toute image reçue rend son slot (`release_of`), même
   jetée pour une ancienne époque ; règle de retrait d'une ancienne génération écrite ; passage de
   slot fixé par `kRingLayout` (GENERAL, file EXTERNAL vers CUDA, partage concurrent, usage
   STORAGE).
4. kernel_ops.h : un noyau qui lit une DecodedFrame en garde une copie jusqu'à la fin de sa
   lecture GPU ; `write_ring` relâche les imports des anciennes générations.

## Décisions de conception (à ajouter au contrat, avec test, dans la chaîne indiquée)

- gpu possède l'instance de Backend et charge les moteurs d'un Program ; prepare ne fait que
  préparer (construction et cache des moteurs) via sa propre instance, jamais partagée entre
  threads (chaîne gpu).
- render envoie les `RingSlotDesc` à gpu par un message de core/messages.h à chaque création de
  génération (chaînes render et gpu, d'accord entre elles ; l'intégrateur tranche).
- `ExternalImage` gagne pas de ligne, plans et valeur timeline si media-gpu ou present en ont
  besoin (chaîne media-gpu).
- `BackendInit` gagne le chemin de trtexec (chaîne tensorrt).
- Repli Yuv444p16 multi-plans : reporté, hors de cette vague ; `write_ring` reste RGBA16.
