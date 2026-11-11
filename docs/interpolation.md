# Interpolation

Le lecteur crée des images entre deux images de la source, pour suivre la fréquence de l'écran. Le réseau
qui les calcule s'appelle ici le « moteur du mouvement » (`MotionEngine`, `src/app/play/settings.h`). Le moteur
de calcul qui l'exécute (TensorRT, Vulkan, Core ML) est un autre choix : voir `docs/moteurs.md`.

## Moteurs du mouvement

| Moteur | Clé `play.toml` | Lecture | Export | Code |
|---|---|---|---|---|
| RIFE (4.22 lite, 4.25…) | `rife` | oui, par défaut | oui | `src/app/play/rife_*`, `kernels/spec/rife*` |
| DistilDRBA | `distil_drba` | oui | oui | `kernels/spec/distil_drba.h` |
| Apple (VideoToolbox) | `apple` | macOS seulement | non | `gpu/system_interpolator.h` |

## Ce que montre le menu

Le menu ne nomme pas les modèles, il propose des niveaux (`kInterpLevels`, `settings.h`). Le nom du modèle ne
s'affiche qu'en mode avancé, à côté du niveau.

| Niveau | Moteur, modèle |
|---|---|
| Très rapide | DistilDRBA (garde lui-même les poses, une image de retard) |
| Rapide | RIFE 4.22 lite (le défaut hors RTX) |
| Équilibré | RIFE 4.25 (le défaut sur une RTX) |
| Système (macOS) | interpolation d'Apple, sur un Mac seulement |

Un niveau dont le modèle manque n'est pas proposé. Le multiplicateur offre « Arrêt » et « Auto » en lecture.
Les facteurs fixes (×1,5 à ×8) ne sont proposés qu'à l'export. En lecture, ils restent possibles par
`--rife=N` ou par `play.toml`, pour le diagnostic.

## Warp du backend TensorRT : lecture filtrée

Sous TensorRT, le warp et le mélange des images intermédiaires (`warp_blend_filtered`,
`src/gpu/cuda/backend/warp_kernels.cu`) lisent les images, le flot et le masque par le filtre bilinéaire de
la texture. Sa fraction est sur 8 bits : l'écart avec le warp exact (`warp_blend`) va jusqu'au gradient local
divisé par 512, soit environ un demi-niveau 8 bits sur un trait noir net. C'est une perte de qualité consentie
pour la vitesse, dont le gain est **non mesuré** (audit B03-03). Mesure à faire sur la RTX :
`ncu --kernel-name regex:warp_blend` sur 30 s de lecture ×3 en 1080p et en 4K, chaque noyau forcé tour à tour,
plus le temps par image de `--stats=`. Si le gain est sous 0,1 ms par image, la lecture exacte redevient le
défaut ; sinon le chiffre s'écrit ici.

## GMFSS retiré

GMFSS Fortuna (base et union) donnait les images intermédiaires les plus propres sur l'animation, mais coûtait
plusieurs fois RIFE : la plupart des machines ne le tenaient pas en direct, et en lecture il avait des défauts.
Débranché de la lecture puis de l'export, il est retiré des sources en 0.10.0 (lecteur, prepare, noyaux,
backends Vulkan et TensorRT, scripts d'export). Un `play.toml` qui nomme `gmfss` ou `gmfss_union` se lit en
RIFE ; `gmfss_fused`, `gmfss_quality` et `gmfss_precision` sont ignorés. L'historique git garde le code d'avant
0.10.0 pour qui voudrait le reprendre, après une mesure en direct face à RIFE (`ajp bench`).
