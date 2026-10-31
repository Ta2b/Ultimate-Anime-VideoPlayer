# Avis des composants tiers

Le lecteur (Ultimate Anime VideoPlayer) est publié sous la **Mozilla Public License 2.0** (fichier `LICENSE`). Les composants ci-dessous gardent leur propre licence. Cette liste décrit ce que les paquets livrent ou chargent ; chaque licence amont fait foi.

## Modèles de réseaux livrés dans `models/`
Source de chaque fichier : `tools/release/fetch-models.sh` (SHA-256 vérifiés).

| Modèle | Auteur | Licence | Remarque |
|---|---|---|---|
| AnimeJaNai HD V3.1 (SPAN), dossier `onnx/` | the-database | **CC BY-NC-SA 4.0** | Attribution, usage non commercial, partage à l'identique des dérivés (dont les poids quantifiés par `tools/span/quantize_span.py`). https://github.com/the-database/mpv-upscale-2x_animejanai |
| RIFE 4.25 et 4.22 lite, dossier `rife/` | hzwer (Practical-RIFE) | MIT, © 2021 hzwer | Conversions ONNX fp16 de vs-mlrt, reprises de la release `models-rife-fp16-1` de `the-database/animejanai-inference`. |
| DistilDRBA v2 lite, dossier `rife/` | neko (routineLife1) | MIT, © 2025 neko | Converti en ONNX fp16 par `tools/drba/export_distil_drba.py`. Le dépôt DRBA d'origine est aussi sous MIT. |

À noter : l'outil de conversion vs-mlrt (AmusementClub) est sous GPL-3.0 pour son code ; il n'est pas livré, et seuls les fichiers de poids convertis le sont. Le dépôt `animejanai-inference`, d'où vient l'archive RIFE, ne publie aucun fichier de licence : la licence de RIFE reste celle de ses auteurs (MIT).

## Bibliothèques liées
| Composant | Licence | Version ou source |
|---|---|---|
| FFmpeg | LGPL-2.1 ou plus récent | n9.0.2, construit sans `--enable-gpl` ni `--enable-nonfree` (`tools/deps/linux-from-source.sh`, `macos-from-source.sh`). https://ffmpeg.org |
| libplacebo | LGPL-2.1 ou plus récent | 7.360.1 (`ports/libplacebo`). https://code.videolan.org/videolan/libplacebo |
| libass (fork the-database) | ISC | 0.17.5-the-database-896614fa (`ports/libass`) |
| dav1d | BSD-2-Clause | via vcpkg |
| SDL3 | zlib | via vcpkg |
| Dear ImGui, spdlog, toml++ | MIT | via vcpkg |
| NVIDIA CUTLASS | **BSD-3-Clause**, © 2017-2026 NVIDIA CORPORATION & AFFILIATES | 4.8.1, en-têtes seulement, copiés dans `third_party/cutlass/` avec leur `LICENSE.txt` (les 259 fichiers que les noyaux du SPAN FP8 incluent, `docs/span-fp8-cutlass.md`) ; compilés dans le backend TensorRT. https://github.com/NVIDIA/cutlass |
| FreeType (FTL), HarfBuzz, FriBidi (LGPL-2.1), Fontconfig, lcms2 | licences de leurs projets | via libplacebo et libass |

FFmpeg, libplacebo et FriBidi sont des bibliothèques partagées (`lib/`) : vous pouvez les remplacer par une version modifiée de la même interface. Leurs sources sont disponibles aux adresses et versions ci-dessus.

## LUT d'HDR (data/luts)

55 LUT 3D `.cube` (« HDR LUT ver.9.1 ») ajoutées au dépôt à la demande de son propriétaire ; auteur et licence d'origine non indiqués dans l'archive reçue, voir `data/luts/ATTRIBUTION.md`.

## Moteurs facultatifs téléchargés depuis l'écran de bienvenue
OpenVINO (Apache-2.0), MIGraphX et ROCm (MIT, d'après AMD), ONNX Runtime (MIT), Windows ML (licence de redistribution de Microsoft jointe au moteur), MoltenVK (Apache-2.0). TensorRT, CUDA, cuDNN et le SDK Video Effects de NVIDIA (RTX HDR) sont sous les licences propriétaires de NVIDIA, que l'utilisateur accepte à l'écran avant le téléchargement.

## État de ce relevé
Licences des modèles et configuration de FFmpeg : vérifiées dans les dépôts amont et dans les scripts de construction. Les autres lignes sont celles annoncées par leurs projets, à relire au prochain relevé complet avec les fichiers de licence installés par vcpkg (`share/<port>/copyright`).
