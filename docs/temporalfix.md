# Stabilisation temporelle (TemporalFix)

À jour, 19 novembre 2026. Réglage : Vidéo > Agrandissement > Stabilité, « Stabilisation temporelle »
(`[image] temporal_fix` de `play.toml` : `off`, `light`, `normal`, `strong`), et dans Avancé « Stabilisation à
pleine taille » (`temporal_fix_full`, éteinte : demi-taille). Éteinte par défaut, l'Auto n'y touche pas.

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
- `src/app/play/temporal_fix_model.*` : le modèle livré (`models/temporalfix/`) dérivé une fois pour toutes les
  tailles (`derived_model.h`) : son entrée 1 × 21 × H × W coupée en sept entrées 1 × 3 × H × W
  (`src/prepare/onnx_frames.h`), un Concat en tête du graphe refaisant l'entrée d'origine ; l'étage gpu passe
  ses sept tampons sans copie ni noyau de plus. H et W restent libres (voir « Construction »).
- **Demi-taille** (le défaut) : le même graphe moyenne d'abord les sept images 2 × 2 (AveragePool), fait
  tourner le réseau sur elles, retire de sa sortie l'image du milieu moyennée, agrandit cette correction
  (Resize bilinéaire) et l'ajoute à l'image du milieu à pleine taille. Le réseau coûte un quart ; le détail de
  l'image reste entier, seule la correction est basse fréquence (le grain le plus fin reste tel quel). Tout se
  fait dans le graphe : TensorRT le fond avec le reste. Vérifié sur processeur (onnxruntime) : le graphe coupé
  à pleine taille rend exactement celui de pifroggi.
- **Cache de l'encodeur** (`src/prepare/onnx_cache.h`, lot B du chantier AnimeJaNai) : le réseau encode ses
  sept images (quatre convolutions image par image, puis un Resize ×2 : 21 % des multiplications du réseau
  en 1080p) avant de les aligner. Une image lue par sept passes de suite était encodée sept fois. Le modèle
  livré donne maintenant deux moteurs : l'**encodeur** (une image 1 × 3 × H × W → ses traits 1 × 4 × H × W),
  lancé une fois par image ingérée, et la **fenêtre**, le graphe d'origine où la sortie de l'encodeur
  (`/encode/Resize_output_0`) est remplacée par la concaténation de sept entrées de traits gardés
  (`features_0` à `features_6`), à côté des sept images. Vérifié sur le graphe : l'encodeur ne lit que son
  image (lot de sept, convolutions sans mélange entre images), dans s1, s2 et s3. Vérifié sur processeur
  (onnxruntime, s2) : à pleine taille, encodeur puis fenêtre rendent au bit près le graphe en un morceau ; à
  demi-taille, 75 dB. En demi-taille, l'encodeur moyenne son image 2 × 2 et range ses traits à demi-taille
  en DepthToSpace (1 × 1 × H × W), que la fenêtre remet en place par SpaceToDepth : toutes les entrées
  gardent les dimensions H et W, et le profil de construction les donne à toutes (`ModelSource::twins`, au
  nombre de canaux près). Les traits sont gardés à côté de leur image dans `temporal_window` : le reflet aux
  bords et les époques s'appliquent aux deux à la fois. Idée reprise de libaji (`src/native/tfix.cpp`,
  commit `1fea490`), écrite sur nos propres découpes ; aucune ligne reprise. Retour en arrière : `--debug --no-tfix-cache`
  reconstruit le graphe en un seul moteur, chaque image encodée à chaque passe (même sortie).
- `src/app/play/chain_temporal.cpp` : la construction et le chargement des deux moteurs (TensorRT, voie
  Upscale). Un refus laisse la lecture sans elle, retenu pour la séance.
- `src/app/play/temporal_window.*` : les tampons bruts gardés par pts et le choix des sept images.
- `src/app/play/upscale_temporal.cpp` : `ingest_source` passe par TemporalFix quand il est chargé ; ses
  ingestions brutes restent dans `ingest_plain`, et chaque image ingérée pour une passe est encodée juste
  avant elle, sur la même voie.

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

## Construction

Le premier essai figeait la taille de travail dans le graphe (`with_fixed_dims`, comme DistilDRBA). TemporalFix
calcule ses grilles de base depuis la forme (ConstantOfShape → CumSum → Expand, 10 CumSum et 23 Expand) :
figées, elles deviennent des constantes que TensorRT range comme des poids. Mesuré sur la RTX 4070 Ti de Tatoub
(TensorRT 11.3, 1080p) : moteur de 6,2 Go, 295 ms par image, la mémoire vidéo saturée à l'ouverture.

Maintenant :

- H et W restent libres dans le graphe, et le profil laisse la largeur descendre de 8 pixels sous celle du
  travail (`EngineLabel::loose_width`, comme vs_temporalfix) : la forme n'est jamais une constante. Les sept
  entrées sont jumelles (`ModelSource::twins`) : trtexec reçoit le même profil pour chacune.
- Niveau d'optimisation 3, sans pavage L2 (`ModelSpec::tuned = false`, recette Fp32Islands, `--stronglyTyped`).

| Réglage (1080p, RTX 4070 Ti) | Moteur | Par image | Construction |
|---|---|---|---|
| Taille figée (premier essai) | 6,2 Go | 295 ms | 377 s |
| Libre, niveau 5 + pavage 3 (réglages de SPAN) | 5,4 Mo | 42,4 ms | 482 s |
| Libre, niveau 3, sans pavage (pleine taille) | 4,4 Mo | 27,9 ms | 79 s |
| Le même réseau en 540p (ce que tourne la demi-taille) | | 6,2 ms | |

Ces temps sont ceux du graphe en un morceau, avant le cache de l'encodeur : à mesurer de nouveau sur la RTX
(cible du lot B, étape 1 : 20 ms ou moins à pleine taille en 1080p).

À 24 i/s, la demi-taille prend environ 15 % de la carte, la moitié de SPAN FP8 (12,5 ms) ; la pleine taille
environ 67 %. Le graphe à demi-taille complet (moyennes, Resize, addition à pleine taille) reste à mesurer sur
la RTX.

## Fusionné (lot B, étapes 2 et 3)

Sous TensorRT, le lecteur essaie d'abord TemporalFix fusionné (`src/app/play/chain_temporal.cpp`), et ne
retombe sur le graphe à cache de l'encodeur que s'il est refusé (la raison au journal, le refus retenu pour la
séance) ; « Annuler et regarder » ou une vidéo glissée pendant sa construction ne lancent pas ce graphe. `--debug --no-tfix-fused`
garde ce graphe, pour la mesure ABAB ; `--debug --no-tfix-cache` prend le graphe en un morceau.

- **Découpe** (`src/prepare/tfix_split.*`) : le modèle est reconnu (noms des tenseurs aux frontières, Concat
  d'entrée, temps des appuis, constantes du mélange, attributs des Resize et des GridSample) puis coupé en cinq
  morceaux de convolutions : l'encodeur (une image), les trois blocs du flot et la tête de confiance, en lots
  de six appuis à forme fixe. Le manifeste `.tfixfused` et les morceaux vont dans les modèles dérivés, sous un
  nom qui porte la dérivation (`fusion1`), le rôle, la taille du réseau et l'empreinte du modèle.
- **Encodeur fusionné** : pour chaque image ingérée, en demi-taille la moyenne 2 × 2, puis les traits
  (TensorRT), puis les trois niveaux de pyramide (flou binomial 5 × 5 et Resize, 7 canaux) écrits une fois
  dans le tampon gardé (`kernels::tfix_packed`). Une image sert sept fenêtres : la pyramide n'est plus refaite.
- **Fenêtre fusionnée** (`src/gpu/cuda/backend/tfix_fused_engine.*`) : X0 depuis les pyramides, puis pour
  chaque niveau le Resize du bloc, le flot doublé, le warp des pyramides de l'appui (bibliothèque de warps,
  `docs/warp-lib.md`) et le Concat en un seul noyau (`tfix_fused_kernels.cu`) ; l'entrée de la tête (warp de
  l'image entière, moyennes, différences) en un noyau, puis le mélange des six appuis en un noyau
  (`tfix_blend_kernels.cu`). Toute la fenêtre est un seul graphe CUDA ; ses adresses passent par une table en
  mémoire de l'appareil, posée avant chaque lancement.
- **Écart** : la référence numpy de la même décomposition (`tools/temporalfix/fused_reference.py`) donne
  78 dB face au graphe entier FP16 (s1, s2, s3), le pixel (0, 0) mis à part : le graphe y ajoute un terme de
  2^-11 que la fusion laisse de côté.

Cible du lot B : 12 ms ou moins par image à pleine taille en 1080p sur la RTX 4070 Ti (27,9 ms pour le graphe
entier), à 52 dB ou plus face au graphe entier. Pas encore mesuré : porte du groupe de tests 2, sur le PC.

## Coût dans l'Auto

Le choix de la chaîne par l'Auto (préréglages, calibration) ne la compte pas : la calibration l'a mesurée sans
elle. Pendant la lecture, le palier de l'Auto baisse le multiplicateur de RIFE si la chaîne prend du retard,
ce qui absorbe la demi-taille. Une place dans le budget de la calibration demande une mesure sur la RTX.

## Ce qui manque

- **Vulkan** : l'import du graphe (`src/gpu/ir/ops.cpp`) ne connaît pas ConstantOfShape, Equal, Where
  (calculs de forme, repliables à la taille fixée), Expand, CumSum, ReduceSum (la grille de base, construite
  depuis `flow * 0 + 1`), ReduceL2 et Abs. Tant qu'ils manquent, un autre moteur que TensorRT lit sans elle
  (une ligne au journal, une note dans le menu).
- **Core ML** : pas essayé.
- **Auto** : compter son coût dans la calibration (`auto_config`, `machine_calibration`).
- **Dessins tenus** : une image identique à la précédente (un dessin tenu en deux ou en trois) pourrait reprendre
  la sortie de la précédente au lieu d'une passe : jusqu'à deux passes sur trois de moins en anime. Il faut
  pour cela une copie de tenseur dans KernelOps et la mesure du dessin tenu avant la passe, sans attente hôte.
