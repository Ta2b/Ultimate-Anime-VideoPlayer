# Essais à faire sur le PC de Tatoub

Ce que le cloud a intégré sans pouvoir l'essayer : il n'a ni RTX, ni écran QD-OLED, ni CUDA
(construit et vérifié seulement jusqu'à la compilation, sur lavapipe pour Vulkan). Chaque ligne dit
quoi lancer et ce qu'on attend. L'extrait de référence reste Re:Zero S4E18, 16:00 → 18:00.

## Construire

Le backend CUDA a de nouveaux fichiers (`nvof_session.cpp`, `tensorrt_optical_flow.cpp`,
`optical_flow_kernels.cu`) : reconfigurer `linux-prefix-cuda` et `linux-prefix-cuda-release`, puis
`ajp-gpu ctest --test-dir build/linux-prefix-cuda -j8`. `kBackendAbi` vaut toujours 3 alors que
`KernelOps` a encore grandi (`open_optical_flow`, `optical_flow`) : à relever à la fusion.

## Flux optique matériel (Fluidité → Moteur du mouvement)

- `ajp_backend_tensorrt_tests "[nvidia]"` : la translation de 6 pixels doit passer (mouvement médian
  à ±0,5 pixel, warp au milieu à moins de 0,02 d'écart moyen). Un SKIP « absente » veut dire que
  `libnvidia-opticalflow.so.1` manque au pilote.
- Lecture 1440p ×4 et 4K ×3 avec `interpolation.engine = "optical_flow"` et `--stats` : relever
  « flux optique » dans le profil GPU (attendu bien sous les 7,6 ms de RIFE), les images sautées
  et les paires en retard, contre les mesures de référence de `handoff-pc.md`.
- À l'œil, contre RIFE, sur 16:00 → 18:00 : contours des personnages en mouvement (halos,
  déchirures), panoramiques, et le fond découvert derrière un objet qui passe.
- Si la qualité déçoit : `kPreferredFlowGrid` (2 → 1) et `kFlowConsistency` dans
  `kernels/spec/optical_flow.h`.

## Smooth Motion

- RIFE ×3 + Smooth Motion (case de Fluidité, puis relancer) : lecture normale attendue, comme le
  premier essai (4 images sautées sur 1312).
- Sans RIFE : la lecture avait presque calé (16 images neuves en 8 s). Comprendre pourquoi avant de
  retirer l'avertissement du menu ; piste : la cadence à la demande de la session contre la
  métrologie du pilote (essayer avec « Cadence fixe » cochée).

## Écran QD-OLED

- Couleurs étendues (Grade → « Couleurs étendues (large gamut) ») : rouges, verts et cyans saturés plus purs, peau
  et gris inchangés ; comparer 0, 0,5 (Éclatant) et 1 avec `--snapshot=X.pq.ppm`.
- Limiteur OLED : la plage vient maintenant de la moyenne d'image de l'EDID ; vérifier dans le
  journal les valeurs tirées pour le MAG 272U E16 (80 et 250 nits attendus pour 265 nits).
- « Cadence fixe » (Soins OLED) : plus de scintillement du VRR dans les scènes sombres, sans images
  sautées en plus.
- Barre de lecture : en pause, elle disparaît après 20 s quand « Effacer la barre en pause » est coché.
