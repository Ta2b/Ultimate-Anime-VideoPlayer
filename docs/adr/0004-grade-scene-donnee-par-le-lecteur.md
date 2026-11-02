# 0004 — Le grade en trois passes, la scène mesurée par le lecteur

## Contexte
Le grade HDR mesurait la scène dans le shader : neuf passes d'analyse (réductions, histogrammes) à chaque image
de sortie, en 4K, en plus de la pré-analyse des plans qui mesurait déjà la même chose sur le CPU.

## Décision
Une seule définition de l'analyse (`kernels/spec/frame_stats.h`, référence CPU dans `kernels/ref`, noyau GLSL
lu sans attente à l'ingestion) nourrit le lecteur : la pré-analyse écrit un `ShotMeta` par plan, l'analyse à
l'ingestion couvre la première lecture et les sauts (`media/live_scenes.h`). Le lecteur passe au grade l'APL et
la part d'aplats clairs de la scène (`fs_apl`, `fs_bright`) ; le shader (`anime-hdr-fused`) n'a plus que trois
passes : carte locale, aplats lissés avec mémoire temporelle, grade. L'ancien corps reste une version derrière
`--legacy-grade`.

## Conséquences
- Les neuf passes d'analyse disparaissent du rendu.
- La décision de scène est la même pour le grade, la peau et les coupes ; elle se teste sans GPU.
