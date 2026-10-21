# Passation du PC de Tatoub (pc/vertical-slice)

État de « ajp play », la tranche verticale jouable, au moment où le travail repart sur le cloud.
Machine : CachyOS, RTX 4070 Ti (12 Go, Ada, compute 8.9), pilote 615.78.08, écran MSI
« MAG 272U E16 » (QD-OLED 4K, HDR10), bureau KDE Wayland à 165 Hz, échelle 2. Rapports complets :
[docs/reports/](reports/) (copie de `~/ajp-reports/*.md`, sans les images).

## État de chaque fonction

| Fonction | État | Où |
|---|---|---|
| Chaîne NVDEC → réduction → RIFE → SPAN ×2 → RTX HDR → libplacebo HDR10 | fait | `src/app/play/`, backend `src/gpu/cuda/backend/` |
| Réductions fusionnées à l'ingestion : Area, Bilinear, Bicubic, Mitchell, Catmull-Rom, Spline36 | fait | `IngestSpec.reduce`, `slice_kernels.cu` |
| RIFE par le mouvement (variante ONNX dérivée, warp + mélange, une passe par paire calme) | fait | `motion_model`, `onnx_outputs`, `warp_kernels.cu` |
| 4K sans réduction (mouvement à 720p, warp à 1080p) | fait ; ×2 tient, ×3 rattrape l'horloge | `upscale_stage` |
| Rattrapage de l'horloge (paire non interpolée si son premier instant est passé) | fait | `VideoStageInit.clock` |
| Barre de lecture ImGui, pistes, préparation des moteurs (pourcentage) | fait | `controls*`, `ui`, `build_progress` |
| Menu des paramètres : Profils, Image, Fluidité, RTX HDR, Grade, Sous-titres, Son, Stats, Moteurs | fait, en direct ou par réouverture | `settings_panel`, `settings` |
| 9 profils (base TOML v5 + user.toml), Ctrl+1 à Ctrl+9 | fait | `profiles` |
| Écran de bienvenue (moteurs, badges, présélection, profil de départ) | fait ; installation des moteurs absents : non | `first_run` |
| Grade HDR « Éclatant » / « Naturel » sur les routes RTX HDR et native | fait | `src/render/grade*`, hooks dans `src/render/shaders/` |
| Route HDR native (SDR en PQ par libplacebo, aspect TrueHDR émulé), repli si Maxine manque | fait | `hdr_native` |
| Vrai noir OLED : détection EDID (DisplayID 0x21, table des modèles, heuristique), seuils calés sur une rampe de 32 marches | fait | `display_edid`, `true_hdr_ramp`, `grade_ramp` |
| Taille des sous-titres en direct ; karaoké sans recomposition en trop | fait | `subs/subtitle_renderer`, `session` |
| Statistiques à la demande, profil GPU par étape, `--stats` | fait | `session`, `gpu_timing` |
| Réduction « Perceptuelle » (Öztireli–Gross, écrite par nous) | à faire | — |
| SPAN en FP8 ou INT8 (Ada), comparé à FP16 en PSNR/SSIM | à faire | — |
| Mesures de la paire sans attente hôte (`scores_now`, `flow_max_now`) | à faire (≈ 0,5 ms par paire) | — |
| Smooth Motion du pilote, en complément de RIFE | fait sur le cloud, à essayer ([tests-pc.md](tests-pc.md)) | `settings`, `placebo_screen` |
| Choix du modèle RIFE (catalogue 4.25, 4.25 lite, 4.22 lite, 4.26, 4.18, 4.13, 4.6 ; heavy déconseillé) | fait, en direct ; défaut 4.25 sur une RTX, 4.22 lite ailleurs (le premier présent) ; heavy coûte 34 ms par passe à 1280x768 | `model_choice` |
| Interpolation ×1,5 (grille de cadence, `core/cadence.h`) | fait sur le cloud, à essayer | `upscale_stage`, `options` |
| Garder les poses (DRBA causal, MIT, routineLife1/DRBA) | fait sur le cloud, à essayer | `kernels/spec/poses.h`, `warp_kernels.cu` |
| GMFSS Fortuna en temps réel | fait sur le cloud (TensorRT), à essayer : graphe « paire » une fois par paire, six softmax splats et graphe « fusion » par image ; visé 720p ×1,5 ou ×2 ; Vulkan attend les opérations de GMFlow dans le moteur réseau | `upscale_gmfss`, `gmfss_model`, `kernels/spec/gmfss.h`, `tools/gmfss/export_gmfss.py` |
| FSR 3 FI, XeSS-FG, VTFrameProcessor | à noter au plan (autres systèmes), rien de codé | — |

## Construire et lancer

Pas de sudo sur ce PC : outils dans `~/ajp-tools`, dépendances dans `~/ajp-deps`, TensorRT dans
`~/ajp-sdk/tensorrt`.

```sh
export PATH=$HOME/ajp-tools/bin:$HOME/ajp-tools/venv/bin:$PATH AJP_DEPS_PREFIX=$HOME/ajp-deps
cd ~/ajp-work/vertical-slice
cmake --preset linux-prefix-cuda -DAJP_TENSORRT_ROOT=$HOME/ajp-sdk/tensorrt          # Debug, tests
cmake --preset linux-prefix-cuda-release -DAJP_TENSORRT_ROOT=$HOME/ajp-sdk/tensorrt  # RelWithDebInfo, lecture
ajp-build cmake --build build/linux-prefix-cuda
ajp-build cmake --build build/linux-prefix-cuda-release
ajp-gpu ctest --test-dir build/linux-prefix-cuda -j8        # verrou GPU
```

Lecture : `~/ajp-tools/bin/ajp-anime "épisode.mkv"` (plein écran, build optimisé). Réglages dans
`~/.local/share/ajp/ajp/` : `play.toml` (menu), `user.toml` (profils), `settings.toml` (moteurs).

Essais sans toucher aux fichiers de Tatoub : un dossier de données à part, avec des liens vers
`engines`, `sdk` et `modeles` :

```sh
XDG_DATA_HOME=/tmp/claude-1000/ajp-data ajp-gpu build/linux-prefix-cuda-release/src/app/ajp play \
  "$HOME/Desktop/[Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 [...].mkv" \
  --window=1280x720 --start=960 --seconds=120 --stats=stats.tsv
```

Options d'essai : `--keys=S:touche,…` (aussi `mouse=XxY`, `click=XxY`, `ctrl+2`), `--snapshot=X.ppm`
(SDR) ou `X.pq.ppm` (sortie PQ 16 bits, grade compris), `--snapshot-at=S`, `--snapshot-frames=N`,
`--keys=0:space` (première image figée, identique d'un essai à l'autre), `--first-run`,
`--rife=N`, `--work-height=N`, `--rife-image`.

## Mesures de référence

Extrait de référence choisi par Tatoub : Re:Zero saison 4 épisode 18, 16:00 → 18:00, build optimisé,
fenêtre 1280x720 (2560x1440 pixels), verrou GPU, bureau chargé (~40 % de GPU hors lecture). Les
mesures prises avec RTX HDR ne se publient pas (licence de NVIDIA) : elles restent sur le PC.

GPU par appel (médiane) : RIFE 4.26 à 720p 7,6 ms ; SPAN 720p → 1440p 4,1 ms ; SPAN 1080p → 2160p
11,5 ms ; warp et mélange 0,06 ms ; ingestion + Area 0,40 ms ;
grade HDR 0,44 ms par image présentée en 2560x1440 ; attente hôte des mesures de paire 0,5 ms.
Démarrage moteurs en cache : 0,9 s (16,7 s en Debug). Grade Éclatant contre aucun, à 16:43 :
p99 839 → 949 nits, pixels sous 0,5 nit 6 → 15 %, saturation +8 à +20 % selon le plan. Vrai noir :
TrueHDR met déjà les codes 16 et 17 à 0 nit et le code 18 à 0,0003 nit.

Smooth Motion du pilote (`NVPRESENT_ENABLE_SMOOTH_MOTION=1`, couche implicite `VK_LAYER_NV_present`,
pilote ≥ 580.82.07 sur RTX 40) : la couche se charge et accepte la swapchain HDR10 de libplacebo
(R10G10B10A2, « HW Flip Metering »). RIFE ×3, 20 s : lecture normale, 1312 images neuves, 4 sautées,
nos présentations à 14,0 ms ; les images ajoutées par le pilote ne se mesurent pas d'ici (journal
muet). Sans RIFE, la lecture a presque calé (16 images neuves en 8 s contre 192) : à comprendre
avant d'en faire une option.

## Tests qui exigent la RTX ou l'écran

Sans GPU NVIDIA ils se sautent (SKIP) ou n'existent pas dans la CI ; `[.]` et `[.bench]` sont cachés.

- `tests/unit/backend_tensorrt/backend_test.cpp` (CUDA) : ABI et refus nommés ; ingestion NV12 BT.709
  contre la référence ; réduction Area fusionnée (1080p → 720p) ; réductions triangle et cubiques
  contre la référence ; warp et mélange par le mouvement ; entrée de RIFE (11 plans, recadrage) ;
  mesures d'une paire ; RTX HDR refusé tant qu'il n'est pas ouvert ; vrai SPAN d'AnimeJaNai construit
  et exécuté (modèle lu en place) ; `[.bench]` coût des réductions.
- `tests/unit/vendor_hdr/real_true_hdr_test.cpp` (`[nvidia][cuda]`, SDK Maxine importé) : du magasin
  au GPU ; rampe de 32 marches quasi noires ; entrée 8 bits tramée avec la vraie matrice ; lignes à
  marge ; le pic passé fait foi ; débandage coupé, modèle rechargé. `real_true_hdr_bench.cpp`
  (`[.][bench]`) : temps de TrueHDR en 1080p, 1440p, 4K.
- `tests/unit/vendor_hdr/real_sdk_test.cpp` (`[nvidia][sdk]`) : SDK installé, import réduit.
- `tests/unit/media/gpu_decode_test.cpp` (`[nvidia]`) : Vulkan Video H.264, HEVC, AV1 sur le GPU ;
  `hw_test.cpp` (`[vulkan]`) : appareil Vulkan choisi par l'UUID (lavapipe en CI).
- `tests/unit/render/grade_gpu_test.cpp` (`[render][gpu]`, tout appareil Vulkan) : garde-fou du grade
  sur une rampe orange ; gris et blanc inchangés.
- Écran : aucun test automatique. Le lecteur entier (fenêtre, HDR10, superposition, vrai noir calé)
  se vérifie à la main avec `--snapshot*.pq.ppm`, `--keys` et `--stats` ; `display_edid_test.cpp`
  rejoue l'EDID réel de l'écran sans écran.

## Contrats modifiés depuis 371b2a2

`kBackendAbi` vaut toujours 3 alors que `KernelOps` a grandi : à relever à la fusion (l'empreinte `kBackendLayout` de
`gpu/backend.h` couvre la forme des structures, pas la table virtuelle).

- `gpu/kernel_ops.h` :
  - `IngestSpec.reduce` (`Reduction{kind, b, c}`, `ReduceKind {Area, Triangle, Cubic, Spline36}`,
    jusqu'à `kMaxReduceTaps` = 32 prises dans le backend) ;
  - `rife_input`, `scores_now` ;
  - `warp_blend(a, b, Motion, t, out)` et `Motion{flow, mask}`, `flow_max_now` (attente hôte) ;
  - `open_true_hdr`, `set_true_hdr`, `write_ring_true_hdr`, `true_hdr_ramp(levels)` (rampe RGBA8 →
    PQ, attente hôte) ;
  - `device_memory()`.
- `gpu/cuda/backend/trt_engine.h` : une entrée et 1 à `kMaxOutputs` (4) sorties FP16, noms dans
  `output_names()`.
- `prepare/onnx_outputs.h` (nouveau) : `with_outputs`, `onnx_node_outputs` (protobuf réécrit sans
  dépendance onnx) ; sert à `app/play/motion_model` (`<data>/modeles/<stem>-mouvement-<sha16>.onnx`).
- `render/grade.h`, `grade_params.h`, `shaders/*.hook` repris de pc/render, plus :
  - `//!PARAM guard` dans anime-hdr-grade.hook ; à 0, calcul identique à l'original ;
  - `GradeSettings.guard`, `area_rtx`, `GradePreset {Eclatant, Naturel, Custom}`, `grade_preset`,
    `black_thresholds`, `kBlackRampSteps`, `grade_limits::kGuard/kAreaRtx/kBlackSteps` ;
  - refus `VendorCallFailed` et `FieldMissing` au lieu de `RenderFailed` et `FeatureUnavailable`,
    absents d'ici.
- `subs/subtitle_renderer.h` : `RenderTarget.font_scale` (0,25 à 4).
- `core/profile.h` : `DownscaleFilter` avec Spline36 et les noms partagés avec play.toml.
- `media/` (packet_reserve, source, source_loop, hw_api…) et `vendor_hdr/` (maxine_abi, runtime,
  sdk_layout, sdk_store, sdk_version, true_hdr) : fusions de pc/media-gpu et pc/vendor-hdr-real
  (583e97b, a8455e5, 8cca1ce, cdbf1bb).

## Modèles et SDK sur ce PC (lus en place, jamais copiés dans le dépôt)

- SPAN AnimeJaNai : `~/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/` (Performance par défaut,
  Balanced pour le profil 6), sinon
  `~/Documents/mpv-AnimeJaNai-HDR/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/onnx/`.
- RIFE 4.26 : `~/.local/share/AnimeJaNai-HDR-linux/animejanai/rife/rife_v4.26.onnx` ; variante
  mouvement dérivée : `~/.local/share/ajp/ajp/modeles/rife_v4.26-mouvement-fd6b06538898c9a9.onnx`.
  RIFE 4.25, 4.13 et 4.6 vont à côté (`rife_v4.25.onnx`, `rife_v4.13.onnx`, `rife_v4.6.onnx`) :
  les fichiers v1 de vs-mlrt, convertis en FP16 par `tools/convert_rife_fp16.py`
  d'animejanai-inference-rtx. Le menu grise ceux qui manquent.
- GMFSS Fortuna : `gmfss_fortuna_base_pair.onnx` et `gmfss_fortuna_base_fusion.onnx` dans le même
  dossier, exportés par `tools/gmfss/export_gmfss.py` (mode d'emploi en tête du script).
- Moteurs TensorRT construits : `~/.local/share/ajp/ajp/engines/aji-v3-*.engine` (+ journaux de
  construction, caches de minutage).
- Maxine VideoFX : source `/usr/local/VideoFX`, copie importée `~/.local/share/ajp/ajp/sdk/`.
- TensorRT 11.3.0.99 et trtexec : `~/ajp-sdk/tensorrt`. Dépendances (FFmpeg, libplacebo 7.360, SDL3,
  ImGui 1.92.9, libass…) : `~/ajp-deps`.
- Épisodes d'essai : `~/Downloads/[Erai-raws] Tensei Goblin dakedo Shitsumon Aru - 02 [1080p …].mkv`,
  `~/Desktop/[Erai-raws] Re Zero kara Hajimeru Isekai Seikatsu 4th Season - 18 […].mkv`.
