# Stabilisation temporelle (TemporalFix)

À jour, novembre 2026. Réglage : Vidéo > Agrandissement > Stabilité, « Stabilisation temporelle »
(`[image] temporal_fix` de `play.toml` : `off`, `light`, `normal`, `strong`). Éteinte par défaut, l'Auto n'y
touche pas.

## Ce qu'elle fait

TemporalFix (pifroggi, [vs_temporalfix](https://github.com/pifroggi/vs_temporalfix), Apache 2.0) retire le
scintillement d'une image à l'autre : grain, bruit de compression, aplats et traits qui bougent sans raison
sur un plan fixe. Le réseau (un demi-million de poids, un flot à la RIFE puis un mélange) lit sept images, trois
avant et trois après, et rend l'image du milieu stabilisée. Ce qui bouge vraiment reste net : le flot aligne les
voisines avant de les mélanger.

| Force | Modèle | Pour |
|---|---|---|
| Légère | `temporalfix_s1_v1.1_op18_fp16.onnx` | une source propre, un peu de grain |
| Normale | `temporalfix_s2_v1_op18_fp16.onnx` | le défaut conseillé par pifroggi |
| Forte | `temporalfix_s3_v1_op18_fp16.onnx` | une source très bruitée ou bloquée |

s0, plus faible que s1, n'est pas livré.

## Où elle est dans la chaîne

Avant tout le reste, à la taille de travail (au plus 1080p, `chain_work_size`) : l'image ingérée (et débandée)
passe par TemporalFix, puis SPAN, RIFE, DistilDRBA, les mesures et le grade lisent l'image stabilisée sans le
savoir. Stabiliser avant l'interpolation aide aussi RIFE : moins de bruit, des flots plus propres.

- `src/app/play/temporal_fix.h` : les forces et leurs modèles.
- `src/app/play/temporal_fix_model.*` : le modèle livré (`models/temporalfix/`) dérivé à la taille de travail
  (`with_fixed_dims`), son entrée 1 × 21 × H × W coupée en sept entrées 1 × 3 × H × W
  (`src/prepare/onnx_frames.h`) : un Concat en tête du graphe refait l'entrée d'origine, et l'étage gpu passe
  ses sept tampons sans copie ni noyau de plus. Le dérivé est rangé avec les autres (`derived_model.h`), une
  taille à la fois.
- `src/app/play/chain_temporal.cpp` : la construction et le chargement (TensorRT, voie Upscale). Un refus laisse
  la lecture sans elle, retenu pour la séance.
- `src/app/play/temporal_window.*` : les tampons bruts gardés par pts et le choix des sept images.
- `src/app/play/upscale_temporal.cpp` : `ingest_source` passe par TemporalFix quand il est chargé ; ses
  ingestions brutes restent dans `ingest_plain`.

## Avance et bords

L'étage attend trois images de plus derrière la plus lointaine qu'il lit déjà (`frames_ahead`, et la suivante
de `look_ahead` qu'il ne compte pas) : `temporal_extra_frames` en vol en plus, autant de surfaces NVDEC
(`held_decoded_frames`, sous les 32 de NVDEC). Chaque image n'est ingérée qu'une fois : la fenêtre garde de
quoi couvrir trois images avant la tête jusqu'à la plus lointaine lue en avance.

Au début d'une époque (ouverture, saut) et à la fin du flux, les voisines manquent : la fenêtre se reflète comme
`gen_shifts` de vs_temporalfix, l'image d'avant manquante k prend l'image d'après k et inversement ; sans
aucune des deux, la plus proche déjà prise, puis l'image elle-même. Le journal de fin compte les passes, celles
qui se sont reflétées et les ingestions brutes.

Une image sans pts, ou d'une autre taille que la source, n'est pas stabilisée (et n'entre pas dans les
fenêtres). Les images réduites à la taille de RIFE (`small_`) restent celles de la source : RIFE mesure son
mouvement sur elles.

## Coût

Mesures de pifroggi sur une RTX 4090 en TensorRT : environ 80 i/s en 1440 × 1080, 20 i/s en 2880 × 2160. À la
taille de travail d'une source 1080p, c'est l'ordre de grandeur d'un second SPAN. À mesurer sur une RTX avec
`linux-release` avant d'en dire plus. Le budget de l'Auto ne la compte pas encore : sur une carte juste, l'Auto
peut choisir une interpolation qui ne tient plus avec elle.

## Ce qui manque

- **Vulkan** : l'import du graphe (`src/gpu/ir/ops.cpp`) ne connaît pas ConstantOfShape, Equal, Where
  (calculs de forme, repliables à la taille fixée), Expand, CumSum, ReduceSum (la grille de base, construite
  depuis `flow * 0 + 1`), ReduceL2 et Abs. Tant qu'ils manquent, un autre moteur que TensorRT lit sans elle
  (une ligne au journal, une note dans le menu).
- **Core ML** : pas essayé.
- **Auto** : compter son coût dans le budget (`auto_config`).
- La première construction TensorRT (annoncée 120 s, `kTemporalFixFirstBuildS`) est à mesurer.
