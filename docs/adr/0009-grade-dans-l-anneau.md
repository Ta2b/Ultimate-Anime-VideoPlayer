# 0009 — Le grade exact dans l'anneau ; la LUT réservée au chantier 8

## Contexte
Après le chantier 2a, le grade fusionné tient en une passe d'OUTPUT de libplacebo, précédée de trois cartes
(locale 1/8, zone 1/32 lissée, max local du vrai noir). Il tourne à chaque présentation, sur l'image à la taille
de l'écran : en 4K, c'est la plus grosse passe de l'affichage. Les images du ring sont déjà sur le GPU de CUDA,
à leur taille de travail, une seule fois par image produite. Deux voies : appliquer une LUT 3D qui approche le
grade, ou refaire le grade exact dans le ring.

## Décision
1. **Le grade exact, pas une LUT.** Le grade dépend de cartes locales (isolement, voisinage clair, peau) qu'une
   LUT de couleur ne voit pas. La LUT d'approximation est laissée au chantier 8, s'il en a l'usage.
2. **Les formules écrites une fois** : `kernels/spec/grade_pixel.h` porte ligne à ligne `anime-hdr-funcs.glsl`,
   `1-local.hook`, `2-area.hook`, `2b-localmax.hook`, le look, le vrai noir et KWin, en fonctions
   `__host__ __device__` (macro `AJP_GRADE_FN`), avec les lectures d'image passées par l'appelant. La référence
   CPU (`kernels/ref/grade.{h,cpp}`) et les noyaux CUDA (chantier 2.8) les incluent toutes les deux : le port CUDA
   n'a plus que les lectures et la mise en mémoire partagée du flou à écrire.
3. **Le contrat** (`kernels/spec/grade.h`) : `GradeUniforms`, les //!PARAM du grade sous leurs noms
   (`kGradeUniformFields`, relus dans le hook posé par `render::Grade::param`) ; la taille des cartes
   (`roundf` comme libplacebo) ; les constantes du flou et du lissage, α = 1 − 0,7^(48 Δt) par l'écart de pts.
4. **L'entrée telle que libplacebo la fait** : `render::capture_grade_input` rend une rampe grise de 4 096
   points sans hook vers la sortie PQ et la relit (une table par canal), plus la matrice des primaires de
   `pl_get_color_mapping_matrix`. La table capte ce que la formule BT.1886 ignore, comme le noir relevé des
   sources SDR.
5. Le shader reste la vérité de l'aspect et le chemin par défaut ; le ring ne gradera que derrière `--grade-ring`
   (chantier 2.9), et `FusedSplit` ne part qu'après la porte des 30 plans (2.11).

6. **La version par image** (chantier 2.9) : main poste une `GradeVersion` (`LiveSettings::post_grade`), qui
   vaut pour les images de son époque de pts ≥ `from_pts_us` et pour les époques suivantes ; les 8 dernières
   sont gardées (`gpu::kGradeOrders`). L'étage la choisit au pts de chaque image écrite (`grade_at`), la confie
   au backend à sa première image (`KernelOps::set_grade`) et la nomme dans `RingWrite::grade` ; le backend
   grade l'emplacement juste avant son signal, et `RingFrame::grade` dit à l'écran que l'image l'est. Sans
   source débandée, en export YUV ou sur un backend sans `grades_ring()`, la version reste 0 et l'écran grade.
   L'historique de la zone repart à une nouvelle époque, à un pts qui recule ou après plus d'une seconde.
7. **L'écran** (chantier 2.10, `--grade-ring`) : `PlaceboScreen` poste une version à chaque changement de ce
   que le grade applique (paramètres, looks, KWin, cible), pour les images qui suivent la dernière présentée ;
   un choix posé par `set_grade_after` part tout de suite, pour les images qui suivent `after`, comme l'écran
   l'appliquera. Les valeurs viennent de `Grade::param_for` et `look_stack_for`, qui disent ce que
   `set_params` poserait sans toucher aux hooks de l'écran. Seul le grade en trois passes vers une sortie PQ
   est porté ; ailleurs l'écran poste « off ». Une image gradée se déclare dans la couleur de la cible de sa
   version (BT.2020, PQ, son max_luma) : libplacebo n'a rien à convertir ; ni hooks ni détection de pic, et le
   débandage seulement s'il est demandé. En pause, une image gradée par une version dépassée est récrite par
   un seek exact sur elle (`Session::regrade_paused`, au plus une fois par 150 ms) : un réglage changé en
   pause se voit. Le backend libère la plus ancienne version derrière l'événement de sa dernière lecture,
   sans vider les voies.
8. **La porte** (chantier 2.11) : `tools/grade/shots30.tsv` (les 18 plans de calibration.mkv et 12 plans
   d'épisodes découpés par `make_stills.sh`), `compare_shots.sh A B SORTIE` (deux instantanés PQ par plan,
   ΔE_ITP par `de_itp.py`, verdict par plan). `FusedSplit` ne part qu'une fois la porte passée sur le PC.

## Mesures (chantier 2.7, lavapipe)
- Référence contre le shader, même image 256 × 256 PQ, six réglages : ΔE_ITP p50 ≤ 0,37, p99 ≤ 1,0 sauf
  l'aspect TrueHDR adaptatif (3,7) et vibrance + saturation au bord du gamut (2,0). L'entrée SDR captée puis
  gradée : p99 0,91, max 0,99.
- Le critère du plan, |ΔPQ| p99,9 ≤ 2⁻¹² par canal, ne tient pas sur lavapipe : ses textures demi-flottantes
  sont arrondies vers zéro (≈ 64 codes 16 bits au grade neutre, où la référence est l'identité à 1e-4), son
  exp/log de l'aspect TrueHDR s'écarte de ~2 % (em_pow = 0 : aucun écart ; la référence suit une transcription
  en double à 0,1 code près), et un canal presque noir d'une couleur saturée est mal conditionné en PQ. Le test
  `[render][gpu]` juge donc en ΔE_ITP ; le |ΔPQ| ≤ 2⁻¹¹ revient au test CUDA contre la référence (2.8), en fp32
  sans stockage intermédiaire.

## Conséquences
- `render` expose `Grade::param`, `Grade::look_stack` et `Grade::skin_mask` ; il ne dépend toujours pas de
  `kernels` (la correspondance vers `GradeUniforms` se fait au-dessus).
- Tests : `tests/unit/kernels/grade_test.cpp` ([kernels][grade]), `tests/unit/render/grade_ring_ref_test.cpp`
  ([render][gpu][grade_ring]).

## Statut (novembre 2026)
`--grade-ring` est allumé par défaut (`options.h`), mais le grade de l'anneau ne porte que le corps `Fused` :
avec le corps par défaut (`AnimeGuided`, la méthode « HDR anime »), la version « off » est postée et l'écran
garde les hooks de libplacebo à chaque présentation (audit, constats F6 et F7). Porter le corps anime dans
`kernels/spec/grade_pixel.h` reste à faire ; le point 5 ci-dessus ne décrit plus le défaut.
