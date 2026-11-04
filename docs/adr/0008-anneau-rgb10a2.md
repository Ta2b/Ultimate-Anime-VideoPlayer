# 0008 — Anneau RGB10A2, et TensorRT qui n'écrit pas dans l'anneau

## Contexte
L'anneau entre le backend et l'écran est fait d'images RGBA16 UNORM : 8 octets par pixel, soit 66 Mo par
emplacement en 4K, une douzaine d'emplacements en Rapide. Sa précision dépasse ce que l'écran affiche : la
sortie HDR10 est en 10 bits. Deux questions se posaient : faut-il réduire l'anneau, et TensorRT peut-il écrire
directement dans un emplacement, sans le noyau `write_rgba16` ?

## Décision
1. **Non à l'écriture directe par TensorRT (étape 5.1).** `setTensorAddress` exige un pointeur linéaire aligné
   sur 256 octets ; un emplacement est une `VkImage` en tuilage optimal, importée en `cudaMipmappedArray`
   (`ring_slot.cpp`), sans adresse linéaire. Le contourner (image linéaire, sortie HWC à 4 canaux, anneau
   rgba16f) coûterait plus à l'affichage que les 0,09 à 0,12 ms gagnées par image. `write_rgba16` reste sur le
   chemin TensorRT FP16 ; il disparaît par l'épilogue du SPAN FP8 (étape 3.4) et le chantier 10.
2. **Une seule fonction d'écriture (étape 5.5b).** `store_ring(RingSurface, x, y, r, g, b)` (`pixel_kernels.cuh`)
   écrit tous les emplacements : RGBA16 au bit près d'avant, ou RGB10A2 = r | g << 10 | b << 20 | 3 << 30
   (`kernels/spec/ring_pack.h`), tramage TPDF fixe par pixel en option. C'est aussi là que le grade dans
   l'anneau (chantier 2) s'insérera.
3. **RGB10A2 de bout en bout, derrière `--ring-rgb10a2` (étape 5.5a).** `PlaceboRing` le prend si l'appareil
   l'inscrit et le rend en `VK_FORMAT_A2B10G10R10_UNORM_PACK32`, CUDA l'importe en 1010102 ; sinon RGBA16, sans
   message. Le backend Vulkan reste en RGBA16 ; l'export aussi.
4. Le plan veut RGB10A2 **par défaut sous TensorRT** : le défaut ne change qu'après la porte de l'étape 5.5c,
   mesurée sur la RTX (test caché `[.ring_interop]` vert, `tools/ring/banding.py` ≤ 1,10 × la référence sur
   les plans sombres de `calibration.mkv`, export P010 ≥ 50 dB contre RGBA16, VRAM de l'anneau −50 %, noyaux
   d'écriture −25 %). Si le banding dépasse, on essaie le tramage forcé ; s'il dépasse encore, RGBA16 reste le
   défaut et cette page le dit.
5. **Novembre 2026 : RGB10A2 est le défaut, et son écriture est tramée.** Le défaut a changé sans la porte de
   l'étape 5.5c ; l'audit (constat F9) a relevé que la pile de LUT et l'expansion HDR étirent alors les paliers de
   10 bits. `ImportedSlot::surface()` pose `dither` pour tout emplacement RGB10A2 (TPDF fixe par pixel, ±1 code).
   Le contrôle de la porte reste à faire sur la RTX : `tools/ring/banding.py` avec `--no-ring-rgb10a2` en
   référence.

## Conséquences
- `kBackendAbi` passe à 26 (`BackendCaps::ring_rgb10a2`).
- Moitié moins de mémoire pour l'anneau et pour chaque écriture, une fois le défaut changé.
- Une image de l'anneau perd ses 6 bits de poids faible : sans effet sur une sortie 10 bits, à surveiller sur
  les dégradés sombres d'une sortie SDR 8 bits tramée par libplacebo.
