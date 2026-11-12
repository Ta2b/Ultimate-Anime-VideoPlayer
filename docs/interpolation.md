# Interpolation

Le lecteur crée des images entre deux images de la source, pour suivre la fréquence de l'écran. Le réseau
qui les calcule s'appelle ici le « moteur du mouvement » (`MotionEngine`, `src/app/play/settings.h`). Le moteur
de calcul qui l'exécute (TensorRT, Vulkan, Core ML) est un autre choix : voir `docs/moteurs.md`.

## Moteurs du mouvement

| Moteur | Clé `play.toml` | Lecture | Export | Code |
|---|---|---|---|---|
| RIFE (4.25 lite, 4.22 lite, 4.26, 4.25) | `rife` | oui, par défaut | oui | `src/app/play/rife_*`, `kernels/spec/rife*` |
| DistilDRBA | `distil_drba` | oui | oui | `kernels/spec/distil_drba.h` |
| Apple (VideoToolbox) | `apple` | macOS seulement | non | `gpu/system_interpolator.h` |

## Ce que montre le menu

Le menu ne nomme pas les modèles, il propose des niveaux (`kInterpLevels`, `settings.h`). Le nom du modèle ne
s'affiche qu'en mode avancé, à côté du niveau. Les niveaux suivent le choix de Tatoub de novembre 2026, sur les
mesures de `docs/rife-candidats.md`.

| Niveau | Moteur, modèle |
|---|---|
| Très rapide | RIFE 4.25 lite (deux fois moins de calcul que 4.22 lite, 2,9 dB de moins au banc ; repli hors RTX si 4.22 lite manque) |
| Rapide | RIFE 4.22 lite (le défaut hors RTX) |
| Équilibré | RIFE 4.26 (le défaut sur une RTX, et l'Auto en Qualité et Ultra) |
| RIFE 4.25, réglages avancés | l'ancien Équilibré : même qualité que 4.26, environ 9 % de calcul en plus |
| DistilDRBA, réglages avancés | garde lui-même les poses, une image de retard ; ni plus rapide ni plus propre que RIFE à ce jour, hors de l'Auto (`kDistilDrbaInAuto`) |
| Système (macOS) | interpolation d'Apple, sur un Mac seulement |

En Personnalisé, un ancien `play.toml` qui nomme RIFE 4.25 ou DistilDRBA les garde : la liste montre alors leur
nom, même hors du mode avancé. Une calibration faite avec RIFE 4.25 le garde aussi jusqu'à la suivante, qui joue 4.26.
Un niveau dont le modèle manque n'est pas proposé. Le multiplicateur offre « Arrêt » et « Auto » en lecture.
Les facteurs fixes (×1,5 à ×8) ne sont proposés qu'à l'export. En lecture, ils restent possibles par
`--rife=N` ou par `play.toml`, pour le diagnostic.

## Lisser les dessins tenus

Case « Lisser les dessins tenus » (Vidéo, sous le multiplicateur ; `[interpolation] smooth_drawings` de
`play.toml`, éteinte par défaut ; `--smooth-drawings --debug` pour une lecture). Un dessin tenu de 2 à 6 images
de la source (animation en deux, en trois…), suivi d'un dessin nouveau, glisse vers lui au lieu de rester figé
puis de sauter : la paire de rang k d'une tenue de n images prend l'instant t à la position (k + t) / n entre le
dessin et le suivant (`smooth_hold_u`, `src/app/play/rife_plan.h`). Une tenue de plus de 6 images garde la
copie (une pose), une coupe reste une coupe, et la tenue s'arrête si la lecture saute (seek, image en retard).

- RIFE mouvement seulement ; l'option remplace « Garder les poses », qui fait l'inverse (`poses_kept`).
- L'étage lit jusqu'à 5 images en avance pour trouver la fin de la tenue (`frames_ahead`, `ring_stage.h`) : 4
  images décodées de plus en vol, 4 surfaces NVDEC de plus sous TensorRT (`held_decoded_frames`).
- Chaque image de la tenue se compare à la première copie : un fondu lent, tenu paire par paire, n'est pas pris
  pour un seul dessin.
- L'idée vient du `--smooth_dedup` de TheAnimeScripter (AGPL) ; aucun code n'en est repris.
- Le journal de fin compte les tenues fondues par longueur (« Smooth : … tenues fondues »).

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
