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

## Warp de Catmull-Rom (essai)

Case « Warp de l'interpolation plus net (Catmull-Rom) » (Vidéo, sous le multiplicateur ; `[interpolation]
cubic_warp` de `play.toml`, éteinte par défaut). RIFE déplace chaque pixel des deux images source selon son flot,
puis les mélange ; le réseau lit la source en bilinéaire, qui adoucit un trait fin déplacé d'une fraction de
pixel. L'option lit les images source par un filtre de Catmull-Rom (Keys, a = −0,5), 4 × 4 points en FP32, la
position bornée à l'image, les points du bord répétés et la valeur bornée à [0, 1]
(`catmull_rom_weights`, `src/kernels/spec/warp.h`). Le flot et le masque restent lus en bilinéaire.

- RIFE seulement : DistilDRBA et le moteur système n'ont pas d'étage de warp. Le RIFE entier (non fusionné) fait
  son warp dans le graphe et l'ignore.
- Deux endroits portent le warp final, selon la chaîne :
  - la variante mouvement du fusionné (le défaut de la lecture) : l'étage de warp, `warp_blend.comp` (Vulkan) et
    `warp_kernels.cu` (TensorRT ; toujours en lecture exacte, même sur le chemin à texture) ;
  - la variante image du fusionné : le noyau de fin, sortie `image-cubic` du manifeste `.rifefused`
    (`RifeFusedOutput::ImageCubic`, `src/prepare/rife_split.h`), un moteur de clé distincte.
- L'export rejoue la même chaîne et suit l'option.
- La référence CPU (`src/kernels/ref/warp.cpp`, `rife_fused.cpp`) et les tests des noyaux Vulkan et TensorRT
  comparent chaque backend à elle, comme pour le bilinéaire.
- Coût attendu : 16 lectures au lieu de 4 par canal et par image source, sur l'étage de warp seulement (les six
  morceaux du réseau ne changent pas). Le gain sur les traits n'est **pas mesuré** : l'option reste éteinte
  jusqu'à la mesure ci-dessous.

### Mesure sur la RTX (165 Hz)

Binaire `linux-release` ou `windows-release`, TensorRT, RIFE 4.26 (Équilibré) puis 4.22 lite (Rapide), ×2 en
1080p et ×2 vers 165 Hz (cadence auto) :

1. **Coût** : 60 s de lecture d'un même passage, option éteinte puis allumée, avec `--stats=` ; relever le temps
   moyen et le 99ᵉ centile par image de l'étage de mouvement, et les images en retard. Si la carte le permet,
   `ncu --kernel-name regex:warp_blend` sur 30 s, chaque variante.
2. **Fidélité** : le banc de qualité (`docs/banc-qualite.md`) lit les réseaux entiers et ne passe pas par
   l'étage de warp ; la mesure se fait donc par l'export. Un extrait de 24 images/s réduit à une image sur deux
   (`ffmpeg -vf "select=not(mod(n\,2))",setpts=N/FRAME_RATE/TB -r 12`), exporté en RIFE ×2 sans
   agrandissement, une fois par mode (deux `--data-dir`, `cubic_warp` éteint puis allumé dans leur `play.toml`) ;
   les images créées se comparent aux images retirées (`ffmpeg -lavfi "[0][1]psnr;[0][1]ssim"`). Le
   Catmull-Rom ne doit pas perdre plus de 0,3 dB de PSNR.
3. **Traits fins** : un panoramique lent avec des traits d'un pixel (décor, cheveux), capture `--snapshot` des
   mêmes instants dans les deux modes, comparées à l'œil à 200 %.

L'option devient le défaut si le coût reste sous 0,3 ms par image en 1080p, sans image en retard de plus à
165 Hz, et si les traits sont plus nets sans halo ni perte de PSNR. Sinon elle reste un essai, ou se retire.

## Échelle par paire (essai)

Case « RIFE allégé sur les mouvements rapides (demi-taille) » (Vidéo, sous le multiplicateur ; `[interpolation]
pair_scale` de `play.toml`, éteinte par défaut). Une paire qui bouge fort, où RIFE mouvement passe à chaque
instant au lieu d'une seule fois au milieu, estime ces passes sur a et b réduites de moitié (filtre Area,
`pair_scale_side`, `src/app/play/upscale_stage.h`), par un second moteur RIFE à cette taille. Le warp met le
mouvement à l'échelle des images, comme pour l'estimation en 720p d'une sortie 4K. Le sens est « moins cher »
seulement : rien ne passe jamais à plus grand que la taille de RIFE.

- Une paire est forte selon la mesure de la précédente (mouvement maximal au-dessus de `single_run_motion`),
  ou selon celle de son propre milieu, comme avant (`upscale_motion.cpp`). Le milieu mesuré et les paires calmes
  restent à pleine taille : la décision ne se prend que sur un mouvement estimé en entier.
- Le mouvement maximal du premier instant d'une paire forte se mesure sur la demi-taille, ramené en pixels de
  RIFE (×2) avant le seuil.
- Le second moteur se prépare au chargement (`chain_half.cpp`) : le fusionné si le RIFE principal l'est, sinon
  la variante mouvement entière. La première fois, il se construit pour cette taille. Un refus laisse la lecture
  à pleine taille, avec la raison dans le journal.
- RIFE mouvement seulement : ni RIFE image (`--rife-image`), ni DistilDRBA, ni le moteur système.
- Le journal de fin compte les passes à demi-taille (« échelle par paire : … passes à demi-taille sur … »).
- Le gain et la perte ne sont **pas mesurés** : l'option reste éteinte jusqu'à la mesure ci-dessous.

### Mesure sur la RTX (165 Hz)

Binaire `linux-release` ou `windows-release`, TensorRT, RIFE 4.26 (Équilibré) puis 4.22 lite (Rapide), ×2 et
cadence auto vers 165 Hz, en 1080p puis en 4K (RIFE en 720p) :

1. **Gain de temps** : 60 s d'un passage riche en panoramiques rapides et en combats, option éteinte puis
   allumée, avec `--stats=`. Relever le temps moyen et le 99ᵉ centile par image de l'étage de mouvement, les
   images en retard, et la part des passes à demi-taille du journal de fin.
2. **Qualité** : un extrait de 24 images/s réduit à une image sur deux
   (`ffmpeg -vf "select=not(mod(n\,2))",setpts=N/FRAME_RATE/TB -r 12`), exporté en RIFE ×2 sans agrandissement,
   une fois par mode (deux `--data-dir`, `pair_scale` éteint puis allumé). Les images créées se comparent aux
   images retirées (`ffmpeg -lavfi "[0][1]psnr;[0][1]ssim"`) sur les seules paires fortes : panoramiques rapides,
   puis petits objets et traits fins en mouvement, le risque principal.
3. **Faux déclenchements** : un flash et un fondu enchaîné ; la part des passes à demi-taille ne doit pas y
   monter.
4. **Scintillement** : un mouvement qui accélère puis ralentit, regardé à 165 Hz. Aucun saut de netteté ne doit
   se voir quand l'échelle change d'une paire à l'autre.
5. **Coût** : le temps de la première construction du second jeu de moteurs, et la VRAM en plus
   (`nvidia-smi --query-gpu=memory.used`, option éteinte puis allumée, même passage).

L'option devient le défaut si le gain dépasse 1 ms par image sur les paires fortes en 1080p, sans perte visible
sur les traits fins ni scintillement, et pour moins de 300 Mo de VRAM en plus. Sinon elle reste un essai, ou se
retire.

## Warp du backend TensorRT : lecture exacte, lecture filtrée à l'essai

Sous TensorRT, le warp et le mélange des images intermédiaires lisent les images, le flot et le masque
exactement (`warp_blend`, `src/gpu/cuda/backend/warp_kernels.cu` : quatre chargements et poids en float).
La lecture par le filtre bilinéaire de la texture (`warp_blend_filtered`) a une fraction sur 8 bits : l'écart
au warp exact va jusqu'au gradient local divisé par 512, soit environ un demi-niveau 8 bits sur un trait noir
net. Son gain n'a jamais été mesuré : c'était une perte de qualité consentie pour une vitesse inconnue, donc
elle n'est plus le défaut. Elle reste derrière l'option d'essai `--filtered-warp` (éteinte), pour la mesure.

Mesure à faire sur la RTX : `ncu --kernel-name regex:warp_blend` sur 30 s de lecture ×3 en 1080p et en 4K,
une fois sans option, une fois avec `--filtered-warp`, plus le temps par image de `--stats=`. Si le gain dépasse
0,1 ms par image, il s'écrit ici avec son chiffre et la question de rallumer le chemin filtré se pose ; sinon
le chemin filtré se retire.

## GMFSS retiré

GMFSS Fortuna (base et union) donnait les images intermédiaires les plus propres sur l'animation, mais coûtait
plusieurs fois RIFE : la plupart des machines ne le tenaient pas en direct, et en lecture il avait des défauts.
Débranché de la lecture puis de l'export, il est retiré des sources en 0.10.0 (lecteur, prepare, noyaux,
backends Vulkan et TensorRT, scripts d'export). Un `play.toml` qui nomme `gmfss` ou `gmfss_union` se lit en
RIFE ; `gmfss_fused`, `gmfss_quality` et `gmfss_precision` sont ignorés. L'historique git garde le code d'avant
0.10.0 pour qui voudrait le reprendre, après une mesure en direct face à RIFE (`ajp bench`).
