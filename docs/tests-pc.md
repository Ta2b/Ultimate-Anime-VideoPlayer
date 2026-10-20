# Essais à faire sur le PC de Tatoub

Ce que le cloud a intégré sans pouvoir l'essayer : il n'a ni RTX, ni écran QD-OLED, ni CUDA
(construit et vérifié seulement jusqu'à la compilation, sur lavapipe pour Vulkan). Chaque ligne dit
quoi lancer et ce qu'on attend. L'extrait de référence reste Re:Zero S4E18, 16:00 → 18:00.

## Construire

Le flux optique matériel NVIDIA a été retiré (fichiers `nvof_session`, `tensorrt_optical_flow`,
`optical_flow_kernels` supprimés) et `kBackendAbi` passe à 4 : reconfigurer `linux-prefix-cuda` et
`linux-prefix-cuda-release`, puis `ajp-gpu ctest --test-dir build/linux-prefix-cuda -j8`.

## Garder les poses (DRBA)

- `ajp_backend_tensorrt_tests "warp et mélange"` : le warp à poses gardées doit suivre la
  référence CPU (paire précédente mobile, puis tenue, à t = 0,25, 0,5 et 0,75).
- Lecture ×3 puis ×4 avec « Garder les poses (DRBA) » coché, RIFE 4.25 puis 4.22 lite, sur
  16:00 → 18:00 : un personnage animé en deux images sur un fond qui défile doit rester net et
  tenir sa pose pendant que le fond glisse, au lieu de fondre entre deux poses. Le coût attendu
  pour RIFE : une seule passe au milieu par paire.
- À vérifier aussi : pas de saut à la reprise après une coupe (la première paire après une coupe
  utilise le warp simple).

## GMFSS Fortuna (Fluidité → Moteur de l'interpolation)

- Exporter les deux graphes (en tête de `tools/gmfss/export_gmfss.py`) dans le dossier de RIFE ;
  le script vérifie que le réseau coupé rend ce que rend GMFSS. Le menu doit cesser de dire
  « Graphes introuvables ».
- `ajp_backend_tensorrt_tests "[gmfss]"` : l'entrée de la paire et le softmax splatting CUDA
  contre la référence CPU (la même que sur lavapipe).
- La première lecture construit les deux moteurs (plusieurs minutes, journal « gpu »). Puis
  720p ×1,5, ×2 et ×3 avec `--stats` : relever le coût de la paire et de chaque image
  intermédiaire, les paires en retard, et ce que la chaîne laisse sans interpolation.
- À l'œil, contre RIFE 4.26 sur 16:00 → 18:00 : traits fins, fond découvert, objets rapides.
- Si un moteur est refusé (forme, nom de tenseur), le lecteur repasse sur RIFE avec la note
  « GMFSS indisponible : RIFE » : le journal dit pourquoi.

## Versions de RIFE et ×1,5

- Fluidité → Modèle RIFE : 4.26, 4.25, 4.13 et 4.6, changés en pleine lecture (chacun construit
  son moteur la première fois). Comparer coût et rendu sur le même extrait.
- ×1,5 (`--rife=1.5`, ou le menu) : 23,976 → 35,964 im/s, une paire sur deux donne une image à
  t = 2/3, l'autre une à t = 1/3 puis la source. Lecture régulière attendue, sans saccade à
  période de deux images, et pts croissants dans `--stats`.

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
