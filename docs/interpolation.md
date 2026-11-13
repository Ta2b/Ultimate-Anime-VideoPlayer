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

## Cadences interpolées (11/2026)

Demande de Tatoub : l'interpolation ne tourne qu'à une cadence d'anime, 23,976, 24, 25, 29,97 ou 30 i/s, à 0,5 %
près (`anime_rate`, `src/app/play/source_fit.h`). Une source à 50, 59,94 ou 60 i/s (jeu, captation, vidéo déjà
interpolée) ou à 12 ou 15 i/s joue sans interpolation, en lecture comme à l'export : `prepare_chain` la retire de
la chaîne jouée, un avis le dit à l'ouverture, Ctrl+I le redit sans rouvrir la chaîne, et l'onglet Fluidité
l'écrit. `play.toml` garde le choix : la vidéo suivante, à 23,976, est de nouveau interpolée. Une cadence
inconnue (0) ne change rien.

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
position bornée à l'image, les points du bord répétés et la valeur bornée au minimum et au maximum des quatre points centraux (les sur-blancs de la source passent, le ringing d'un trait net est coupé ; C50-08)
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

## Une seule passe RIFE par paire calme (C50-02)

Sous le mouvement maximal `kSingleRunMotion720` (15 px à 720p, mis à l'échelle de la hauteur de RIFE par
`single_run_motion`, `src/kernels/spec/warp.h`), RIFE ne passe qu'une fois par paire, au milieu ; les autres
instants se déduisent par `warp_blend_at`, le mouvement en ligne droite et la loi de mélange. C'est un compromis
**de qualité contre du temps**, consenti par défaut (poses gardées) : un mouvement courbe ou accéléré sous le
seuil (bras qui tourne, caméra qui freine) suit une droite, et les occlusions gardent le masque du milieu. Au ×2
(t = ½ seul) rien ne change.

- Le seuil vient d'animejanai-inference-rtx. Le **gain de temps et la perte de qualité ne sont pas mesurés** ici :
  la mesure de la loi de mélange (`warp.h`) porte sur le mélange, pas sur le seuil.
- Le journal de fin dit la part de paires servies par une passe (« RIFE mouvement : N passes ; N paires à une
  passe »).
- Mesure à faire sur la RTX (`linux-release` ou `windows-release`, TensorRT, RIFE 4.26 puis 4.22 lite) : réduire
  des extraits à 24 images/s à une image sur trois, interpoler à t = ⅓ et ⅔, comparer aux vraies images (PSNR,
  pire bloc), une fois avec le seuil à 15 px et une fois à 0 (RIFE à chaque instant) ; relever le temps par image
  de l'étage avec `--stats=`. Le seuil se règle ou s'annonce d'après ces chiffres.

## Instants déduits sur les paires fortes (essai)

Case « Moins de passes RIFE sur les mouvements rapides » (Vidéo, sous le multiplicateur ; `[interpolation]
strong_middle` de `play.toml`, **éteinte par défaut**, en direct). Une paire forte (mouvement maximal au-dessus de
`single_run_motion`) fait aujourd'hui passer RIFE à chaque instant : en 24 i/s vers 144 Hz, 5 passes au lieu
d'une. Un long panoramique n'a que des paires fortes, et RIFE y coûte environ 5 fois plus pendant tout le plan
(retour de Tatoub sur une ville médiévale en 3D : les détails n'y sont pour rien, un réseau fait le même calcul
quel que soit le contenu).

Avec l'option, l'instant t d'une paire forte se déduit du milieu (`warp_blend_at`, comme sur une paire calme)
tant que le milieu prolongé jusqu'à t déplace les points de moins que `single_run_motion` :
2·|t − ½| × mouvement < 15 px à 720p (`middle_reaches`, `src/app/play/rife_plan.h`). C'est la même tolérance
qu'une paire calme, qui prolonge déjà son milieu jusqu'à 15 px vers ses bords. Au-delà, RIFE repasse à
l'instant exact. En 24 vers 144 Hz : sous 45 px, les instants ⅓ et ⅔ (2/6, 4/6) se déduisent et seuls 1/6 et
5/6 repassent (3 passes au lieu de 5) ; sous 22,5 px, une seule passe.

- Avec les poses gardées (le défaut) ; sans elles, une paire forte prend le mode direct (pas de passe au
  milieu), que l'option ne change pas.
- Le journal de fin compte les instants déduits (« paires fortes : … instants déduits du milieu »), et les
  passes RIFE (« RIFE mouvement : … passes »).

### Mesure sur processeur (13 novembre 2026)

`tools/bench/qualite/rife_instants.py` : 40 suites de quatre images toutes différentes de Re:Zero (1080p,
animées en uns), paire (k, k + 3), vérités k + 1 et k + 2 à t = ⅓ et ⅔ ; mouvement estimé à 720p comme le
lecteur, warp en 1080p. « Déduit − RIFE » : PSNR contre la vraie image de l'instant déduit du milieu moins celui
de RIFE à l'instant.

**RIFE 4.22 lite** (le défaut)

| Mouvement (720p) | prolongé à t = ⅓ | Instants | RIFE (dB) | Déduit − RIFE : moyenne | pire | Pire bloc, écart moyen |
|---|---:|---:|---:|---:|---:|---:|
| 0 à 15 px | 0 à 5 px | 32 | 34,38 | -0,26 | -8,04 | -0,64 |
| 15 à 30 px | 5 à 10 px | 24 | 25,26 | -0,03 | -1,37 | -0,01 |
| 30 à 45 px | 10 à 15 px | 8 | 25,16 | -0,21 | -0,58 | -0,33 |
| 45 à 60 px | 15 à 20 px | 6 | 20,59 | +0,46 | -0,01 | +0,04 |
| 60 à 90 px | 20 à 30 px | 6 | 20,54 | +0,40 | -0,29 | +0,37 |
| 90 px et plus | 30 px et plus | 4 | 19,40 | -0,27 | -0,66 | -0,15 |

**RIFE 4.26**

| Mouvement (720p) | prolongé à t = ⅓ | Instants | RIFE (dB) | Déduit − RIFE : moyenne | pire | Pire bloc, écart moyen |
|---|---:|---:|---:|---:|---:|---:|
| 0 à 15 px | 0 à 5 px | 36 | 33,19 | -0,06 | -1,83 | +0,07 |
| 15 à 30 px | 5 à 10 px | 22 | 24,16 | +0,01 | -0,42 | +0,11 |
| 30 à 45 px | 10 à 15 px | 12 | 23,84 | -0,09 | -0,53 | -0,01 |
| 45 à 60 px | 15 à 20 px | 6 | 21,86 | +0,01 | -0,35 | +0,16 |
| 60 à 90 px | 20 à 30 px | 4 | 19,64 | +0,21 | -0,22 | +0,20 |

Lecture : déduire du milieu coûte peu (de −0,26 à +0,46 dB en moyenne selon la tranche), et pas plus sur les
paires fortes (15 à 45 px, celles que l'option déduit à ⅓ et ⅔) que sur les paires calmes, où le lecteur le
fait déjà par défaut (pire instant : −8,0 dB calme contre −1,4 dB fort en 4.22 lite). Au-delà de 45 px, RIFE à
l'instant n'est pas meilleur que le milieu prolongé : les deux sont loin de la vraie image (vers 20 dB). Peu de
paires très fortes dans cet extrait : la mesure oriente, l'œil de Tatoub juge (la même scène de panoramique,
ABAB avec et sans l'option, et le nombre de passes du journal). L'option ne devient le défaut que sans perte
visible.

## Échelle par paire (essai)

Case « RIFE allégé sur les mouvements rapides (demi-taille) » (Vidéo, sous le multiplicateur ; `[interpolation]
pair_scale` de `play.toml`, **éteinte par défaut** : Tatoub la juge nettement moins belle en lecture, voir
plus bas). Une paire qui bouge fort, où RIFE mouvement passe à chaque
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
- Au ×2, RIFE ne passe qu'au milieu, toujours à pleine taille : l'option ne change rien.

### Mesure sur processeur (13 novembre 2026)

`tools/bench/qualite/rife_taille.py` : 30 fenêtres k, k+1, k+2 de l'extrait 1080p de Re:Zero (saison 4,
épisode 18, `bench/rezero-s4e18-1710.mkv` des fichiers du projet), graphes FP16 livrés sous ONNX Runtime. RIFE
estime flot et masque sur k et k+2 réduits à la hauteur donnée (moyenne de surface), le flot remonte en
bilinéaire, le warp et le mélange se font en 1080p comme l'étage de warp ; la sortie se compare à la vraie k+1.
Une paire est « forte » au-dessus de 15 px de mouvement à 720p (99,9e centile de |flot|, comme `k_flow_max`).
Écarts de PSNR contre l'estimation en 720p (le lecteur aujourd'hui), en dB :

| Estimé à | 4.26, fortes (19) | 4.26, calmes (11) | 4.22 lite, fortes (19) | 4.22 lite, calmes (11) |
|---|---:|---:|---:|---:|
| 1080p | +0,03 (pire −0,87) | −0,08 (pire −0,60) | +0,02 (pire −0,43) | −0,24 (pire −1,19) |
| 540p | −0,02 (pire −0,26) | −0,82 (pire −3,07) | −0,06 (pire −0,54) | −1,25 (pire −3,77) |
| 360p (demi-taille) | −0,03 (pire −0,38) | −4,28 (pire −11,88) | −0,09 (pire −0,67) | −4,47 (pire −13,14) |

Ce que ça dit :

- **Sur les paires fortes, image par image, la demi-taille ne perd presque rien** : 0,03 à 0,09 dB en moyenne,
  le pire bloc de 64 × 64 bouge de 0,1 à 0,2 dB. Mais l'œil la voit en lecture (plus bas).
- **Sur les paires calmes, elle se verrait** : jusqu'à 12 dB de moins sur un plan lent et détaillé. La porte de
  15 px est donc au bon endroit, et les paires calmes doivent rester à pleine taille, comme le code le fait.
- **Estimer au-dessus de 720p ne gagne rien** (1080p : −0,01 dB sur les 30 paires) ; **en dessous**, 540p
  coûte 0,8 à 1,3 dB sur les paires calmes. `kRifeMotionHeight` = 720 est le bon réglage (audit A49-10, C9).
- Limite : une paire calme qui suit une paire forte prend la demi-taille (la décision lit la paire précédente).
  C'est le cas d'un mouvement qui s'arrête net ; il ne dure qu'une paire.

### À l'œil (Tatoub, 13 novembre 2026)

Sur sa RTX 4070 Ti, l'interpolation est « de bien moins bonne qualité » avec l'option allumée. Le banc ci-dessus
ne le voit pas, parce qu'il juge chaque image seule, au milieu d'une paire. En lecture, deux choses lui
échappent :

- dans une paire forte, le milieu reste à pleine taille et les autres instants passent à la demi-taille : la
  netteté du mouvement change d'une image affichée à l'autre (scintillement à 165 Hz) ;
- la décision suit la paire précédente : un mouvement qui ralentit garde la demi-taille une paire de trop, là où
  elle coûte jusqu'à 12 dB.

L'option reste donc un essai éteint. La reprendre demanderait de juger la suite des images (tous les instants
d'une paire, et les paires qui suivent un changement d'allure), pas l'image du milieu seule.

### Mesure sur la RTX (165 Hz)

Binaire `linux-release` ou `windows-release`, TensorRT puis Vulkan, RIFE 4.26 (Équilibré) puis 4.22 lite
(Rapide), cadence auto vers 165 Hz, en 1080p puis en 4K (RIFE en 720p) :

1. **Gain de temps** : 60 s d'un passage riche en panoramiques rapides et en combats, option éteinte puis
   allumée, avec `--stats=`. Relever le temps moyen et le 99ᵉ centile par image de l'étage de mouvement, les
   images en retard, et la part des passes à demi-taille du journal de fin.
2. **Faux déclenchements** : un flash et un fondu enchaîné ; la part des passes à demi-taille ne doit pas y
   monter.
3. **Scintillement** : un mouvement qui accélère puis ralentit, regardé à 165 Hz. Aucun saut de netteté ne doit
   se voir quand l'échelle change d'une paire à l'autre.
4. **Coût** : le temps de la première construction du second jeu de moteurs, et la VRAM en plus
   (`nvidia-smi --query-gpu=memory.used`, option éteinte puis allumée, même passage).

L'option ne deviendrait le défaut que sans perte visible en lecture (le verdict de Tatoub l'a refusée), avec un gain
sur les paires fortes, et pour moins de 300 Mo de VRAM en plus.

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
