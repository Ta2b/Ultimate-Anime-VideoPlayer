# TemporalFix en noyaux natifs (CUDA) : fiche du chantier

Historique : la fiche de départ du chantier, gardée telle quelle. L'état actuel est dans
[`docs/tfix-natif.md`](../../tfix-natif.md).

19 novembre 2026, demandé par Tatoub. Dépôt `Ta2b/kirari-player`, branche partie de `main` à jour.

## But

TemporalFix (`docs/temporalfix.md`) sans moteur TensorRT : plus de construction (79 s aujourd'hui à 1080p) et
plus rapide. Aujourd'hui sous TensorRT, RTX 4070 Ti, 1080p : 27,9 ms par image en pleine taille, 6,2 ms en
demi-taille (mesures d'avant le cache de l'encodeur, à refaire). Repère : le TemporalFix natif d'AnimeJaNai fait
8,1 ms par image à 1080p sur une RTX 5090, cache d'encodage compris (environ 16 à 20 ms attendus sur une 4070 Ti).
Cible : égaler TensorRT, puis le battre ; 50 dB ou plus contre TensorRT FP16.

## Sources

- Notre code : `docs/temporalfix.md` (forces, modèles, cache de l'encodeur, demi-taille), `chain_temporal.cpp`,
  la découpe et le moteur TensorRT existants.
- Nos noyaux à réutiliser : `src/gpu/cuda/backend/rife_native_kernels.cuh` (convolutions 3 × 3 mma.sync FP16,
  sommes f16 ou f32, pas 2 par parité, K découpé, poids rangés en fragments B), la bibliothèque de warp
  (`warp_lib.cuh`, `docs/warp-lib.md`), le modèle de l'essai au chargement (`rife_fused_native.cpp`,
  `docs/rife-natif.md`).
- Pour les idées seulement (ne copier aucune ligne, règle des chantiers) : the-database/animejanai-inference,
  `CLAUDE.md` sections « Native kernels » et `vk/tfix` (native/tfix.cpp, native/vidnet) : convolutions 3 × 3
  sur leur conv générale, convolutions par canal et 1 × 1 en sommes f32, rééchantillonnages, warps et mélange
  dans quelques noyaux, cache par image d'entrée choisi sur le GPU selon la règle des coupes de plan.

## Étapes (chacune fusionnable seule, derrière `--tfix-native`, éteint par défaut)

1. Lire le graphe ONNX livré (trois forces) : inventaire des opérateurs et des formes ; dire ce qui manque à
   nos noyaux (attendu : convolutions par canal, 1 × 1, rééchantillonnages, GridSample, mélange).
2. Noyaux manquants (convolutions par canal et 1 × 1 fondues avec leur activation ; sommes f32 là où la
   profondeur l'exige), puis le réseau natif, encodeur gardé par image comme le chemin TensorRT.
3. Essai au chargement contre le moteur TensorRT (≥ 45 dB), sinon TensorRT garde la chaîne ; journal en dB.
4. Doc `docs/tfix-natif.md` avec les commandes de la séance finale (ABAB, `--stats`, nsys).
5. Ensuite seulement : ne plus construire TensorRT quand le natif est retenu (référence f32 maison).

## Règles du jour

Écrire à l'aveugle (pas de banc ni de test GPU ici) ; compiler CUDA avec nvcc 13.5 des paquets pip de NVIDIA,
`-Xptxas -warn-spills -Werror all-warnings`, sm_75/86/89/120. Fichiers sous 800 lignes, clang-format-20, pas de
getenv, commentaires et journaux en français. Nouveau bit d'essai (RIFE 1<<6, GMFSS 1<<7 : prendre le suivant
libre), kBackendAbi = main + 1, kFlags et kTrialOptions recomptés, options.cpp ≤ 800 lignes. Relire sa PR, la
rebaser, la fusionner soi-même ; ni étiquette ni release. Touche `src/gpu` : nouveau paquet TensorRT Linux.
Écrire à Tatoub en français.
