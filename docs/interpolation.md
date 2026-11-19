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

Depuis novembre 2026 (choix de Tatoub après le vote à l'aveugle et Kill Blue 4:08), le menu ne propose plus de
niveaux : il liste trois modèles par leur nom (`kInterpLevels`, `settings.h`), chacun avec une infobulle courte.
Plus rien n'est caché en mode avancé.

| Modèle | Ce qu'il fait de mieux, son coût |
|---|---|
| DistilDRBA v2 lite | garde le rythme de l'animation (poses tenues) ; juste sur les barreaux de Kill Blue (4:08,6) où 4.26 et 4.25 lite se trompent ; un réseau environ 5 fois plus léger que 4.22 lite (140 ms contre 776 par image en 720p sur processeur), mais refait à chaque image créée : plus léger que RIFE jusqu'à ×5, plus lourd au-delà ; une image de retard ; hors de l'Auto (`kDistilDrbaInAuto`) |
| RIFE 4.22 lite | le RIFE le plus léger, traits fins ; le défaut hors RTX |
| RIFE 4.25 | le RIFE le plus précis (premier bloc au 1/8 : juste sur les barreaux) ; environ 15 % de calcul en plus que 4.22 lite (906 ms contre 776 sur processeur) ; le défaut sur une RTX, et l'Auto en Qualité et Ultra |
| Système (macOS) | interpolation d'Apple, sur un Mac seulement |

Retirés du menu : RIFE 4.25 lite (Très rapide), RIFE 4.26 (Équilibré), GMFSS au milieu des paires fortes (Élevé+)
et GMFSS union seul. Un `play.toml` ou une calibration qui les nomme se relit et tombe sur le plus proche
(`src/app/play/interp_retired.h`) : 4.26 et GMFSS sur 4.25, 4.25 lite sur 4.22 lite. Leurs moteurs restent dans le
code ; `--rife-model` charge encore 4.26 ou 4.25 lite pour un essai.
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

## Paire forte avec les poses gardées

Une paire forte n'applique les poses à aucun instant : ses instants hors du milieu sont RIFE à l'instant exact
(`strong_half`), et son milieu est le milieu de RIFE tel quel (`interpolate_by_motion`,
`src/app/play/upscale_motion.cpp`). Avant, le milieu gardait les poses : après un dessin tenu, il montrait déjà b
pendant que 1/4 et 3/4 restaient linéaires. L'image à 3/4 reculait donc derrière le milieu, une paire sur deux
sur un dessin en deux qui bouge vite. Tatoub l'a vu au ralenti dans un export en Équilibré ×4 ; la mesure image
par image a trouvé 19 reculs sur 425 images, aux rangs multiples de 8. La copie de b après un dessin tenu
(`--skip-held`) ne change pas : toutes ses images valent a puis b, sans recul.

## Taille de l'estimation du mouvement

Case « Estimer le mouvement en 720p » (Vidéo > Interpolation ; `[interpolation] motion_720p` de `play.toml`),
**décochée par défaut** : au-delà de 720 lignes de travail, RIFE mouvement estime le flot à la taille de travail.
Cochée, il l'estime sur une copie réduite à 720 lignes (`kRifeMotionHeight`, `motion_size` dans
`src/app/play/chain_setup.cpp`), plus rapide. Recharge la chaîne. Avant ce réglage, la copie 720p était
toujours faite ; Tatoub a choisi la taille de travail par défaut après la mesure ci-dessous.

Origine : un panoramique lent (environ 1,3 px par image source en 1080p) ondule en 4.25 lite et pas en
4.22 lite. Mesure sur le PC de Tatoub (×4, médiane du flot du rang 0 vers chaque instant, en part du pas
d'une source ; régulier : 0,25 / 0,50 / 0,75) :

| Modèle et mode | ¼ | ½ | ¾ |
|---|---|---|---|
| 4.25 lite, mouvement estimé en 720p | 0,03 | 0,11 | 0,35 |
| 4.22 lite, mouvement estimé en 720p | 0,12 | 0,34 | 0,69 |
| 4.25 lite, RIFE image en 720p | 0,02 | 0,14 | 0,93 |
| 4.25 lite, RIFE image en 1080p | 0,01 | 0,47 | 0,96 |
| 4.22 lite, RIFE image en 1080p | 0,07 | 0,41 | 0,86 |

Le réseau 4.25 lite n'est pas régulier sur un mouvement de moins d'un pixel, même en 1080p ; en 720p, son milieu
colle à la source (0,14) et le mode mouvement, qui prolonge ce milieu en ligne droite, garde l'écart à tous les
instants. En 1080p, le milieu revient à 0,47 : le mode mouvement doit alors répartir les instants presque
régulièrement. Mesuré ensuite, même passage, mode mouvement estimé à la taille de travail : 4.25 lite
0,22 / 0,47 / 0,69, 4.22 lite 0,22 / 0,45 / 0,72, 4.26 0,21 / 0,45 / 0,72 ; le même paquet en 720p redonne
0,03 / 0,11 / 0,35 pour 4.25 lite.

Observation à l'œil (Tatoub, 17 novembre 2026) : avec le mouvement d'ensemble ci-dessous (« Panoramiques lents
stables »), l'estimation en 720p reste très bien sur ses panoramiques lents. Le défaut de `motion_720p` ne change
pas sans son accord.

## Mouvement d'ensemble (« Panoramiques lents stables »)

Case « Panoramiques lents stables » (Vidéo > Interpolation ; `[interpolation] pan_lock` de `play.toml`),
**décochée par défaut** tant qu'elle n'a pas été vérifiée à l'écran sur la RTX ; RIFE et DistilDRBA (variante
mouvement), recharge la chaîne. Contrat : `src/kernels/spec/warp_pan.h` ; référence `src/kernels/ref/warp_pan.cpp` et `ref/warp.cpp` ;
GLSL `pan_global.comp` et `warp_blend.glsl` ; CUDA `warp_kernels.cu` (`pan_global`, `warp_pixel`).

Au warp de la variante mouvement, une petite passe (un seul groupe de 16 × 16) estime la translation de la paire
par Lucas-Kanade sur la luma (32 × 32 points, 3 itérations, au centième de pixel), dans un tampon de 3 flottants
relu par tous les pixels du warp. Là où le flot de RIFE est petit (moins de 2 px) et proche de cette translation
(moins de 1 px d'écart), il est remplacé par elle ; puis, sous un demi-pixel de déplacement, le mélange glisse
vers un fondu linéaire (poids 1 − t), pleinement au-dessus de 2 px. Garder les poses n'est pas touché.

La translation est prise à l'instant du flot lu : au milieu (± D/2) pour le flot du milieu de RIFE, à l'instant de la
sortie quand le flot en vient (`Motion::at_instant`) : DistilDRBA, dont le warp reçoit l'image du centre en a et
l'instant s depuis elle (t avant ½, 1 − t après), et RIFE estimé directement à t (mode direct, paires fortes). Le
masque appris de DistilDRBA est ramené de même vers le fondu (poids 1 − s). Ses poses ne sont pas touchées : un
personnage tenu sur un fond qui défile n'a pas le flot de la translation, et au-delà de 2 px le masque reste le sien.

Pourquoi. Sur des panoramiques lents de synthèse (vérité exacte), RIFE voit mal le sous-pixel : à 1 px par image,
4.26 estime 0,2 px vers chaque image au lieu de 0,5, et rien à 0,3 px ; son flot varie de 0,13 à 0,24 px d'un
trait à l'autre. La loi de mélange (`kBlendLawB`) colle alors les instants à la source la plus proche : à 0,3 px
par image, ×4, l'instant ¼ tombe à 0,05 du pas et l'instant ¾ à 0,85 (saccade), et les traits ondulent.

Mesures (×4, Python, chemin du lecteur ; « irrégularité » : écart-type du pas d'une image affichée à la
suivante, par pavé de 32 px riche en traits, rapporté au pas moyen ; panoramique parfait 0,02 à 0,04 ;
« écart » : différence moyenne entre pavés voisins, rapportée au pas) :

| Plan | Modèle | lecteur actuel | fondu seul | mouvement d'ensemble + fondu |
|---|---|---|---|---|
| synthèse 1080p, 0,6 px/image (irrégularité, PSNR) | 4.26 | 0,67 (57,1 dB) | 0,06 (65,1 dB) | 0,09 (64,3 dB) |
| synthèse 1080p, 1 px/image | 4.26 | 0,44 (54,4 dB) | 0,12 (59,0 dB) | 0,07 (62,2 dB) |
| synthèse 1080p, 1,5 px/image | 4.26 | 0,19 (55,4 dB) | 0,06 (58,0 dB) | 0,04 (60,9 dB) |
| Re:Zero 18, 21:22, 720p, 0,65 px/image (irrégularité) | 4.22 lite / 4.26 | 0,78 / 0,74 | 0,17 / 0,21 | 0,24 / 0,22 |
| Nia Liston 07, 2:10–2:20, travelling ≈ 1,7 px/image (irrégularité ; écart) | 4.22 lite | 0,31 ; 0,064 | 0,30 ; 0,049 | 0,05 ; 0,035 |
| même plan | 4.26 | 0,14 ; 0,066 | 0,11 ; 0,050 | 0,06 ; 0,035 |

Le mouvement d'ensemble gagne nettement dès 1 px par image et sur le travelling de Nia, où il réduit l'écart d'un
pavé à l'autre (l'ondulation du mur) d'environ 45 % sans perte de netteté ; le fondu seul reste un peu meilleur
sur le panoramique très lent de Rem (0,65 px). Le plan de Nia vient d'une source dont la compression fait pomper la
netteté d'une image sur trois (+60 à +75 %) : ce scintillement est dans la source, pas dans l'interpolation.
À l'œil, Tatoub a préféré la variante avec mouvement d'ensemble sur ses panoramiques (version de test).

Coût attendu : une passe d'un groupe par warp (environ 50 000 lectures demi-précision, servies par le cache),
de l'ordre de 20 µs par image créée ; à mesurer sur la RTX avant de l'allumer par défaut.

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

## Blocs du RIFE fusionné en E/S HWC8 (par défaut ; `--no-rife-hwc8` en essai)

Sous TensorRT, chaque bloc du RIFE fusionné recevait X en plans (NCHW) et rendait Y en plans ; le moteur
recopiait l'un et l'autre vers son rangement interne (les « Reformatting CopyNode » du profil). Sur la RTX
4070 Ti, RIFE 4.26 en 1280 × 768, ces copies prenaient 41,5 % du dernier bloc (0,279 ms à l'entrée, 0,063 ms
à la sortie, sur 0,825 ms) et 10 % de l'encodeur.

Les blocs se construisent avec `--inputIOFormats=hwc8 --outputIOFormats=hwc8`
(`TrtexecConfig::hwc8_io`, haché dans la clé : un moteur en plans du cache ne sert pas à sa place), et les
noyaux de colle écrivent X et lisent Y par pixel, canaux complétés de zéros jusqu'au multiple de 8 (15 → 16,
28 → 32, 52 → 56 ; `rife_fused_kernels.h`). Le calcul est le même, au rangement près : le test des noyaux
contre la référence CPU passe dans les deux rangements. `TrtEngine` lit le rangement de chaque E/S au
chargement et `RifeFusedEngine` refuse des blocs rangés différemment. L'encodeur reste en plans (2 % de la
passe).

Mesure de trtexec sur les morceaux du 4.26 en 1280 × 768 (graphe CUDA, deux tours ABAB, médianes) :

| Morceau | En plans | HWC8 |
|---|---:|---:|
| encodeur | 0,2064 ms | 0,2023 ms |
| bloc 0 | 0,0625 ms | 0,0605 ms |
| bloc 1 | 0,0686 ms | 0,0654 ms |
| bloc 2 | 0,1085 ms | 0,0994 ms |
| bloc 3 | 0,1808 ms | 0,1558 ms |
| bloc 4 | 0,8121 ms | 0,4331 ms |
| passe entière | 1,4388 ms | 1,0164 ms (−29 %) |

Aucun reformatage ne reste au profil des moteurs HWC8. Dans le lecteur (RTX 4070 Ti, ×8 en 4K, deux fois
60 s, SPAN en FP16) : RIFE passe de 3,0 à 2,38 ms par passe (−19 %), les images présentées de 142,6 à 146,0 par
seconde, les images manquées d'environ 520 à 312 ; l'image reste dans le bruit de la lecture (36,9 à 37,1 dB
pour un plancher de 37,4 entre deux lectures de main) et à 68 dB de la référence CPU. Allumé par défaut depuis
cette mesure ; `--no-rife-hwc8` (avec `--debug`) revient aux E/S en plans. Le premier lancement reconstruit les
blocs une fois (autre clé de moteur). `tools/trt/nhwc_pieces.py` refait les blocs
en E/S NHWC dans l'ONNX même, pour un trtexec qui refuserait `--inputIOFormats`.

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

### Lectures du flot groupées par point

Chaque pixel de la sortie lit le flot à quelques points seulement, mais canal par canal : avec les poses gardées
(le défaut), 41 lectures bilinéaires de quatre chargements, dont les quatre canaux en (u, v) relus à chaque
itération de `pose_time_at`. `Exact::sample_n` (`bilinear.cuh`) lit les canaux d'un même point en une fois
(position, voisins et poids calculés une fois) et le flot en (u, v) se lit une seule fois par pixel : 33 lectures
au lieu de 41, résultat identique au bit près (mêmes opérations dans le même ordre). Le gain dépend de ce que
nvcc regroupait déjà. Mesuré par nsys sur la RTX 4070 Ti (×8 en 4K) : médiane de warp_blend de 529 à 440 µs
(−17 %), moyenne de 630 à 472 µs.

### Mouvement par tuile (`--tiled-warp`) : retiré

L'idée : calculer le mouvement du milieu (second prélèvement du flot, instant réglé des poses, poids de
mélange) une fois par point du flot de la tuile d'un bloc, en mémoire partagée, puis l'interpoler à chaque
pixel de la sortie (neuf pixels par point en 4K pour un flot en 1280). Elle ne payait que tant que chaque pixel
relisait le flot canal par canal ; depuis les lectures groupées ci-dessus, le warp exact ne lit plus le flot
qu'une fois par pixel et il n'y a plus rien à partager. Mesuré sur la RTX 4070 Ti (×8 en 4K, deux lectures
ABAB) : warp et mélange à 0,62 ms avec l'essai contre 0,44 ms sans (+41 %), 154 i/s dans les deux cas, image
à 46,4 dB du warp exact. L'interpolation elle-même s'écartait du warp exact de 0,08 à 0,11 sur un flot qui
tourne vite, et encore de 0,025 sur un flot dix fois plus lent, même en laissant les pixels calculer
eux-mêmes là où le mouvement varie trop entre les points (instant, tx, ty, poids). L'essai, son noyau et ses
tests sont retirés ; le bit `1U << 2U` des essais du backend reste libre.

## GMFSS Fortuna Union : les niveaux Élevé et Très élevé

Retiré en 0.10.0, GMFSS revient en novembre 2026 à la demande de Tatoub, après la mesure de qualité
([relevé du 14 novembre](reports/gmfss-union-qualite-2026-11-14.md)) : RIFE 4.26 égale ou bat union en PSNR et en
LPIPS, mais union tient mieux les rotations et panoramiques rapides à l'œil. Seul union revient, flot entier
(le demi-flot de l'ancien lecteur coûtait 0,5 dB) ; ni la base, ni le fusionné, ni le FP8.

- **Élevé** (`gmfss_middle = true`, RIFE 4.26) : une paire dont le mouvement mesuré atteint `gmfss_motion`
  (15 px à 720p par défaut, de 4 à 60) a son image du milieu M par une fusion union ; chaque autre instant est
  RIFE mouvement sur la demi-paire (a, M) ou (M, b), puis le warp de l'étage. Les autres paires restent à RIFE
  seul. Mesuré : 30,12 dB / 0,0739 contre 30,40 / 0,0734 pour RIFE seul et 29,92 / 0,0745 pour union seul.
- **Très élevé** (GMFSS union seul, `engine = "gmfss_union"`) : la fusion à chaque instant, sans RIFE. Proposé
  hors du mode avancé depuis le 15 novembre 2026, à la demande de Tatoub : l'utilisateur choisit entre le mélange
  (Élevé) et union seul (Très élevé).

Code : `src/app/play/upscale_gmfss.cpp` (l'étage), `chain_gmfss.cpp` (les deux moteurs),
`src/kernels/spec/gmfss.h` et `ref/gmfss.cpp` (le softmax splatting), `src/gpu/cuda/backend/gmfss_kernels.cu`
(les huit splats d'un passage). Deux graphes exportés par `tools/gmfss/export_gmfss.py` :
`gmfss_fortuna_union_pair.onnx` (FeatureNet, GMFlow, MetricNet, une fois par paire) et
`gmfss_fortuna_union_fusion.onnx` (IFNet et GridNet, qui reçoit l'instant), construits à la taille de la chaîne.

**IFNet remplacé par RIFE 4.26** (essai du 15 novembre 2026) : `tools/gmfss/ifnet_rife.py` écrit une fusion dont
l'IFNet d'union (un RIFE ancien, environ 3 ms de la fusion à 1280x768 sur la 4070 Ti, un quart de son temps) est
remplacé par le graphe `rife_v4.26.onnx` du lecteur (1,5 ms à la même demi-taille), branché devant GridNet dans
le même ONNX. Mêmes entrées et même sortie que la fusion d'origine, même nom de fichier : le lecteur la charge sans
changement ; la métadonnée `ajp.gmfss.ifnet` nomme le RIFE. GridNet n'est pas réentraîné : à l'œil, Tatoub ne
voit pas de différence (vidéos de Re:Zero S4E18). RIFE travaillant à la demi-taille au 1/64, la taille remplie de
GMFSS est un multiple de 128 (`kGmfssAlign`), ce que sont déjà 1280x768 et 1920x1152.

**Demi-flot** (`gmfss_half_flow`, case « Mouvement de GMFSS à demi-taille » sous le niveau, ajoutée le
15 novembre 2026 à la demande de Tatoub pour juger à l'œil) : la paire `gmfss_fortuna_union_s05_pair.onnx`
(`export_gmfss.py --scale 0.5`, mêmes entrées et sorties), rangée à côté des deux autres, remplace celle au flot
entier ; la fusion reste la même. Sur la 4070 Ti, en travail 720p, la paire au flot entier coûte 34 ms par image
source et la fusion 11 à 13 ms par instant : Très élevé ne tient qu'en ×2 (environ 40 i/s sur 48 au flot
entier), le demi-flot doit l'amener à 48. Au-delà de ×2, GMFSS seul ne tient pas en temps réel sur cette carte ;
Élevé ×8 tient (GMFSS sur une paire sur dix environ, au seuil de 15 px). Sans le graphe, la case est grisée.
Activé par défaut depuis la refonte des infobulles (paire environ −65 %, GMFSS environ −40 % en tout, perte
mineure) ; un `play.toml` qui l'a éteint le garde éteint.
Bord de la paire : `gmfss_frames` recopie le bord de l'image dans le remplissage (720 → 768 lignes) au lieu du
noir. Au demi-flot, le noir passait dans le flot près du bord : chaque clé avait une bande sombre ondulée en
haut (environ 5 lignes à 720p, jusqu'à 40 niveaux) et en bas, et le bord clignotait à chaque clé, ce qui
semblait changer la taille de l'image (Tempal 08, 2:46).
Décor fixe des clés (`gmfss_still`, spec `kernels/spec/gmfss.h`) : la fusion de GMFSS rend une image plus douce
que les sources, et la composition à la taille de travail n'en rend pas tout le détail. Mesuré en lecture image
par image (travail 1080p, façade immobile) : une clé perdait 56 % du détail fin sans composition, 9 % avec, et la
couleur de ses traits s'écartait de ses voisines 3,5 fois plus qu'en Équilibré ; les traits changeaient
d'épaisseur et de teinte une image sur quatre. Chaque clé reprend maintenant les pixels des sources là où elles
ne diffèrent pas (écart du voisinage 3 × 3 sous 3/255, fondu jusqu'à 8/255). Avec `gmfss_key_floor = 0`, Élevé
pose une clé sur toute paire qui bouge un peu (une fontaine suffit) : il se comporte alors comme Très élevé.
Splat tiré (`SplatJob::pull`) : sur ce qui bouge, la clé composée restait environ 20 % moins nette que ses
voisines (l'eau de la fontaine de Tempal 08, 5:06), parce que le splat poussé étale chaque pixel sur quatre
voisins. La composition pousse maintenant le flot, puis lit chaque source en Catmull-Rom là d'où vient le pixel,
comme le warp de RIFE : même occultation, détail de la source intact.

**Clés GMFSS du mélange** (novembre 2026) : une paire qui bouge reçoit une à trois images union, les clés, aux
instants k/(n+1) ; RIFE mouvement fait chaque autre instant entre les deux voisines (source ou clé), une passe
par image créée. Une clé de plus à chaque doublement du mouvement au-delà de 30 px à 720p (`gmfss_keys`) ;
aucune sous `gmfss_key_floor` (0 par défaut). En lecture, un crédit de temps (0,75 de la durée de chaque
source, au plus 120 ms) les paie ; l'export, sans horloge, les prend toutes. En Auto instantané, la grille libre
place la première image de la paire après le temps de ses clés (`gmfss_lead_us`, qui ne prend jamais le dernier
instant de la paire) : sans lui, les premiers instants d'une paire à clés paraissaient en retard, la
contre-pression montait à ×2 pour toutes les paires (au banc, Très élevé : 1 407 images manquées en 90 s) et
plan_gaps rendait à RIFE les paires fortes dont le premier instant aurait été en retard (573 sur 90 s). Le coût
des clés est un coût fixe de la paire, hors du temps de service par image (`LiveSettings::auto_fixed`).

**Infobulles** : chaque réglage de l'interpolation dit ce qu'il fait en une phrase, puis « Vitesse » (en %, + = plus
rapide), « Impact sur la qualité » (imperceptible, mineur, modéré, élevé) et « Recommandé », qui est la valeur
par défaut ; aucun nom de carte ni de configuration. Les chiffres sont des ordres de grandeur des bancs de
novembre 2026.

**Construction** : les deux graphes sans pavage L2 ni niveau d'optimisation 5 (`gpu::ModelSpec::tuned`). Mesuré
sur la 4070 Ti : la paire en 3 minutes ; la fusion à 640x384 en 218 s au lieu de 692 s, et 6 % plus rapide ; à
960x576 (travail 1080p), le pavage faisait planter Myelin et la construction dépassait le délai.

**TensorRT seulement** (`BackendCaps::gmfss`) : le compilateur de Vulkan n'a ni MatMul ni Softmax. Sans les
deux graphes ou sans TensorRT, le niveau n'est pas proposé, et un `play.toml` qui le demande garde RIFE (raison
au journal). Les poids d'union sont MIT ; ils ne sont pas livrés sans l'accord de Tatoub. Un `play.toml` qui
nomme `gmfss` se lit en RIFE ; `gmfss_fused`, `gmfss_quality` et `gmfss_precision` restent ignorés. Les temps
sont à mesurer sur une carte NVIDIA (`tools/gmfss/banc_temps.sh`).

## Tri des paires par la carte de la scène (`--pair-triage`, essai)

La carte de la scène (`kernels/spec/scene_flow.h`, flux optique matériel de NVIDIA, `--debug --scene-map`) ne
déforme rien : elle décide. Avec `--pair-triage`, sur RIFE mouvement seulement (ni GMFSS, ni DistilDRBA), chaque paire où RIFE
tournerait est triée (`src/app/play/rife_triage.h`) :

- **dessin tenu** : ni le dessin ni la caméra ne bougent, et presque aucune tuile ne change hors du grain.
  RIFE ne tourne pas, b sort seul, comme un dessin tenu reconnu par le pire bloc 16 × 16, que le grain d'une
  source granuleuse fait souvent manquer (`pair_held`, `rife_plan.h`) ;
- **paire impossible** : le flux ne trouve presque aucune correspondance (part `unmatched` de 0,45 ou plus,
  ou coupe probable que RIFE n'a pas vue). Le mouvement de RIFE est remplacé par un fondu enchaîné de a et b,
  sans fantômes ;
- **zone douteuse** : les tuiles d'effet qui changent, et celles qui changent avec une confiance basse,
  passent au fondu, dilatées d'une tuile et adoucies ; RIFE reste ailleurs. La moitié du doute d'une paire
  passe à la suivante, pour qu'une zone ne clignote pas ; une coupe ou un saut l'oublie.

Le fondu se fait sur le mouvement, pas sur l'image : `KernelOps::motion_doubt` (ABI 48,
`kernels/spec/motion_doubt.h`, référence `kernels/ref/motion_doubt.cpp`, noyau
`gpu/cuda/backend/doubt_kernels.cu`) tire flot et masque vers « déplacement nul, masque 0,5 » (1 − t pour un
mouvement lu à l'instant) dans des tampons à part ; le warp habituel fait le reste, avec la loi de mélange de
RIFE. Vulkan ne l'a pas : il n'a pas non plus de carte.

Sans carte (AMD, Intel, Turing, NVOF refusé) ou sans l'option : la décision d'avant, image pour image. La
ligne de fin `tri des paires par le flux : …` compte les paires lues, tenues, fondues et fondues par endroits.
Les seuils (`triage::` dans `rife_triage.h`) ne sont pas mesurés ; la porte : le test à l'aveugle gagné sur
les passages de rife-fantomes, sans hausse du scintillement (outil 0.1).
