# Bibliothèque de warp CUDA (lot E)

Tout ce qui déplace une image par un flot s'écrit une fois, ici, et ses clients la reprennent : la colle de RIFE
fusionné, TemporalFix fusionné (lot B, étape 2 : `tfix_fused.cuh`, `docs/temporalfix.md`) et les lectures de
GMFSS (`gmfss_kernels.cu`, 19 novembre 2026 : mouvement sur une autre grille, splat tiré en Catmull-Rom,
composition à la taille de travail, une prise par point pour tous ses plans) aujourd'hui ; la composition par
calques, le décalage des tuiles et RIFE guidé par le flot NVOF demain. Le splat poussé de GMFSS reste à part :
c'est une dispersion (chaque source ajoute à quatre cibles), pas une lecture.

## Les pièces

- `src/gpu/cuda/backend/warp_lib.cuh` : les briques de périphérique.
  - La prise : position `x + flot` en FP32 (`__fadd_rn`, la grille de A03-20), bornée à une fenêtre, lue en
    bilinéaire ou en Catmull-Rom borné (C50-08).
  - Le flot à toute échelle : un tenseur grossier agrandi en bilinéaire demi-pixel (le Resize de RIFE), évalué aux
    seuls points lus (`coarse_axis`, `lerp2`).
  - Le relèvement par bloc : `coarse_span` donne, avec la même arithmétique que la lecture, la région d'un tenseur
    grossier que lit un bloc de fils ; `stage_region` la copie une fois en mémoire partagée, par 16 octets quand le
    tenseur est rangé par pixel (HWC8).
  - La réduction aux points d'échantillonnage : les 2 × 2 points `d x + d/2 − 1 + {0, 1}` portés par quatre fils
    voisins, sommés par `sum_quad` (échange de registres).
- `src/gpu/cuda/backend/warp_lib.h` : `launch_warp_planes`, le warp générique. 1 à 16 plans FP16 ; flot tensoriel
  (en plans ou par point, canal de départ au choix, échelle et gain : une sortie de RIFE, un flot NVOF en grille 4,
  un flot vers un autre instant) ou mouvement affine (caméra) ; confiance facultative sur la grille du flot, avec
  image de repli `c · warp + (1 − c) · repli` et sortie de la confiance ; réduction d ; sortie en plans ou par
  pixel. Pas de relèvement en mémoire partagée : ce point d'entrée ne sert pas le chemin le plus chaud.

## La colle de RIFE réécrite (`rife_fused_kernels.cu`)

Le profil Nsight du 19 novembre (4070 Ti, RIFE 4.25, ×2) donnait 3,31 ms par paire à la colle : `fused_glue`
2,24 ms (4 appels), `fused_finish` 0,66 ms, `warp_blend` 0,41 ms ; 1,53 ms chez AnimeJaNai. Ce qui coûtait :

1. chaque fil de `fused_glue` relisait les flots de tous les blocs précédents par lectures de 2 octets éparses dans
   les Y rangés en HWC8 (une ligne de cache par fil ou presque), plus le masque et les attributs : une centaine de
   lectures par point ;
2. aux échelles grossières (d = 8 ou 16), un fil par sortie qui enchaîne ses quatre points : trop peu de fils pour
   occuper la carte, chacun long ;
3. 128 registres imposés et le débordement qui menaçait sm_120.

La nouvelle colle : chaque bloc de 256 fils relève une fois la région des Y précédents qu'il lit (16 octets à la
fois), puis agrandit les flots depuis la mémoire partagée. Réduite, un fil par point d'échantillonnage (quatre fois
plus de fils), réduction par échange de registres, et chacun des quatre fils écrit un quart du pixel de sortie
(16 octets). Spécialisée par forme : pleine résolution (tuile 32 × 8) ou réduite (tuile de 16 × 4 sorties), et
nombre de flots (1 à 4). 108 à 124 registres sous `__maxnreg__(128)`, sans débordement avec nvcc 13.5 de sm_75 à sm_120 (`__launch_bounds__` laissait nvcc 13 déborder de quelques octets). La fin lit le flot de `x_last` en 8 octets et chaque
canal de Z (ses quatre parités) en 8 octets.

Les calculs et leurs arrondis FP16 sont ceux de l'ancienne colle, au même endroit ; seul l'ordre de la somme des
quatre points change ((a + b) + (c + d) au lieu de ((a + b) + c) + d), un écart de l'ordre du dernier bit FP32.

Une contrainte nouvelle : chaque source de flot doit être au moins aussi grossière que la sortie (s ≥ d), ce que
suivent tous les RIFE découpés (réductions décroissantes). Sinon `launch_fused_glue` refuse (erreur à l'essai du
chargement) et le RIFE entier reprend. La place réservée par source (17 × 5 pixels de Y en pleine résolution,
10 × 4 réduite) est vérifiée pour d de 1 à 32 et s ≥ d.

`warp_blend` (le warp de la variante mouvement, 0,41 ms) n'est pas touché : il porte les poses, le mouvement
d'ensemble et l'écriture dans l'anneau. Il passera sur les briques communes quand TemporalFix les prendra.

Méthode reprise de la description publique de libaji (rassemblement à l'échelle du bloc, warp aux points
d'échantillonnage, spécialisation par forme) ; aucune ligne de son code (sans licence).

## Tests

- `kirari_backend_tensorrt_tests "[warp_lib]"` : un seul test, le warp générique contre une référence CPU en double
  (flot grossier avec confiance, repli, réduction 4, sortie par pixel ; caméra en Catmull-Rom dans une fenêtre).
- La colle et la fin gardent leurs tests : `"[rife_fused]"` (noyaux contre la référence CPU, 65 à 83 dB, 4.25 et
  4.22 lite, en plans et en HWC8) et le test caché `"[.rife_fused]"` (chaîne contre le RIFE entier, ≥ 50 dB).

## Porte (séance PC du groupe 1, CachyOS)

Binaire `linux-release`, Re:Zero 18 comme `docs/rife-fusionne.md`, ×2, `--one-lane`. Nouveau paquet Linux :
`src/gpu` change l'empreinte TensorRT.

1. `ctest -L nvidia` puis `kirari_backend_tensorrt_tests "[warp_lib],[rife_fused],[.rife_fused]"` : tout vert.
2. Profil `nsys` d'une lecture de 30 s, branche contre `main`, ABAB 3 tours : somme `fused_glue` + `fused_finish`
   + `warp_blend` par paire. Porte : sous 1,6 ms, sortie à 50 dB ou plus de l'actuelle (le test caché).
3. Si la porte manque : le rapport `nsys` dit lequel des trois noyaux reste lourd ; `ncu` sur lui
   (`--kernel-name regex:fused_glue`) dit s'il est limité par la mémoire ou par la latence.
