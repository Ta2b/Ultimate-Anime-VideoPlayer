# Anime4K : l'agrandissement des petites cartes

SPAN demande un moteur de réseau (TensorRT, ou Vulkan de calcul) et, sur un iGPU, ne tient pas la cadence :
la tranche Économe joue alors la source telle quelle. Anime4K (https://github.com/bloc97/Anime4K, licence MIT)
est un agrandissement ×2 en shaders de mpv : libplacebo les lit comme les hooks du grade, sans moteur, sans
construction, sur toute carte qui affiche déjà la vidéo. C'est le plancher des petites configurations.

## Ce que le lecteur fait

| Élément | Où |
|---|---|
| Les shaders, copiés tels quels de l'amont (commit `7684e95`), avec leur licence | `third_party/anime4k/` (`PROVENANCE.md`, vérifiés par `tools/deps/anime4k-copy.sh --check`) |
| Leur chargement, le mode, les jeux de passes | `src/render/anime4k.h` |
| Les deux entrées de Vidéo > Modèle, leur nom dans `play.toml` | `src/app/play/anime4k_choice.h` |
| La chaîne : Anime4K à la place de SPAN | `prepare_chain_once` (`chain_setup.cpp`) |
| L'écran : les shaders devant la netteté et le grade | `PlaceboScreen::set_anime4k`, `compose` (`placebo_screen.cpp`) |
| L'export : le même ×2 au rendu de chaque image | `run_export` (`export_run.cpp`), `FrameExportSetup::hooks` |
| Le palier Économe et l'Auto | `kAnime4kInAuto` (`auto_config.h`), **éteint** tant que le banc n'a pas parlé |

Anime4K suit les règles de SPAN : la même hauteur de travail, la même réduction, le même « restaurer les traits »
(une source gardée telle quelle n'est pas agrandie), le même ×2. La différence est l'endroit : SPAN tourne sur le
thread gpu et remplit l'anneau d'images doublées ; avec Anime4K l'anneau garde l'image de travail et l'écran pose
les shaders sur MAIN à la présentation. La composition gardée (`compose_key.h`) les fait tourner une fois par
image distincte, pas à chaque rafraîchissement. L'interpolation, elle, travaille à la taille de travail, plus
petite : elle coûte moins qu'avec SPAN, et chaque image créée passe aussi par Anime4K.

Deux tailles :

- **Anime4K Fast** (`anime4k-fast`) : les réseaux M, le jeu qu'Anime4K conseille aux petites cartes ;
- **Anime4K HQ** (`anime4k-hq`) : les réseaux VL, environ quatre fois le temps de M selon Anime4K.

Chaque jeu suit la chaîne « Mode X » d'Anime4K pour mpv, réduite à un seul ×2 (comme SPAN ; le second ×2 de mpv
et ses AutoDownscalePre ne sont pas repris, libplacebo ajuste à la fenêtre) :

| Mode | Passes | Choisi quand |
|---|---|---|
| A | Clamp_Highlights, Restore, Upscale ×2 | source gardée de 900 lignes et plus (la plupart des 1080p, souvent agrandies d'une 720p) ou sous 600 (les SD) |
| B | Clamp_Highlights, Restore Soft, Upscale ×2 | source gardée de 600 à 899 lignes (720p), ou réduite de moins de 1,75 fois (1080p vers 720p) |
| C | Clamp_Highlights, Upscale Denoise ×2 | source réduite d'au moins 1,75 fois (1080p vers 540p ou 480p : une image nette, sans défaut) |

C'est la table « Optimized for » d'Anime4K (`md/GLSL_Instructions_Advanced.md`), lue d'après ce que le lecteur
sait de l'image de travail. Le ×2 ne tourne que si la fenêtre fait au moins 1,2 fois l'image (condition `//!WHEN`
des shaders) : en dessous, il serait réduit aussitôt, et seule la restauration (A, B) reste.

Sans moteur de réseau (ni TensorRT ni Vulkan cochés à l'accueil), Anime4K garde son agrandissement
(`within_engines`) ; RIFE, lui, s'éteint.

## Le banc des petites cartes

`tools/bench/petits_gpu.py` décide `kAnime4kInAuto` (et éclaire `kLightSpanInAuto`, ArtCNN) :

```sh
python3 -I tools/bench/petits_gpu.py --video EPISODE.mkv --sortie banc-igpu \
  --span ~/.local/share/ajp/models/onnx/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx \
  --artcnn ~/.local/share/ajp/models/onnx/2x_ArtCNN_C4F32_rgb_fp16.onnx
```

- **Temps** : chaque variante (Économe sans agrandissement, Anime4K Fast et HQ, SPAN Performance, ArtCNN, et
  tout `--modele NOM=FICHIER`) joue 30 s du même épisode en plein écran, sans interpolation, deux fois ; le
  tableau donne les images perdues, le temps GPU de l'affichage (où Anime4K tourne), celui des moteurs et celui
  des seules passes d'Anime4K (journal : « profil GPU · Anime4K »).
- **Qualité** : chaque variante exporte un plan 1080p de 4 s réduit à 540 lignes par le lecteur, puis agrandi ×2 ;
  PSNR et SSIM contre le plan d'origine. Économe (540 lignes agrandies par Lanczos) est le repère. Une image du
  milieu du plan par variante va dans `qualite/` pour le jugement à l'œil, qui prime : le PSNR punit la netteté
  inventée d'Anime4K.

Critère pour `kAnime4kInAuto = true` : sur un iGPU (Intel Xe ou Radeon 680M) en 1080p, Anime4K Fast ne perd
aucune image de plus qu'Économe et donne une image meilleure à l'œil, sans recul au PSNR sur Lanczos. ArtCNN
qui ferait mieux au même coût l'emporte (son interrupteur est `kLightSpanInAuto`).

## Mesures

Aucune sur une vraie petite carte encore. Sous lavapipe (Vulkan logiciel, processeur), les tests
`[anime4k]` vérifient que chaque jeu se lit, agrandit plus net que le filtre seul et ne déborde pas des niveaux
de l'image ; ils ne disent rien du temps.
