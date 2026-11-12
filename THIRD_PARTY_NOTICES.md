# Avis des composants tiers

Le lecteur (Ultimate Anime VideoPlayer) est publié sous la **Mozilla Public License 2.0** (fichier `LICENSE`). Les composants ci-dessous gardent leur propre licence. Cette liste décrit ce que les paquets livrent ou chargent ; chaque licence amont fait foi.

## Modèles de réseaux livrés dans `models/`
Source de chaque fichier : `tools/release/fetch-models.sh` (SHA-256 vérifiés).

| Modèle | Auteur | Licence | Remarque |
|---|---|---|---|
| AnimeJaNai HD V3.1 (SPAN), dossier `onnx/` | the-database | **CC BY-NC-SA 4.0** | Attribution, usage non commercial, partage à l'identique des dérivés (dont les poids quantifiés par `tools/span/quantize_span.py`). https://github.com/the-database/mpv-upscale-2x_animejanai |
| RIFE 4.25, 4.26, 4.22 lite et 4.25 lite, dossier `rife/` | hzwer (Practical-RIFE) | MIT, © 2021 hzwer | Conversions ONNX fp16 de vs-mlrt (4.25, 4.26, 4.22 lite), reprises de la release `models-rife-fp16-1` de `the-database/animejanai-inference`. 4.25 lite (niveau « Très rapide ») est converti par `tools/rife/convert_rife_fp16.py` de ce dépôt. |
| DistilDRBA v2 lite, dossier `rife/` | neko (routineLife1) | MIT, © 2025 neko | Converti en ONNX fp16 par `tools/drba/export_distil_drba.py`. Le dépôt DRBA d'origine est aussi sous MIT. |

À noter : l'outil de conversion vs-mlrt (AmusementClub) est sous GPL-3.0 pour son code ; il n'est pas livré, et seuls les fichiers de poids convertis le sont. Le dépôt `animejanai-inference`, d'où vient l'archive RIFE, ne publie aucun fichier de licence : la licence de RIFE reste celle de ses auteurs (MIT).

## Bibliothèques liées
| Composant | Licence | Version ou source |
|---|---|---|
| FFmpeg | LGPL-2.1 ou plus récent | n9.0.2, construit sans `--enable-gpl` ni `--enable-nonfree` (`tools/deps/linux-from-source.sh`, `macos-from-source.sh`). https://ffmpeg.org |
| libplacebo | LGPL-2.1 ou plus récent | 7.387.0 (`ports/libplacebo`). https://code.videolan.org/videolan/libplacebo |
| libass, avec le rendu sur plusieurs fils de the-database/libass | ISC | 0.17.5 et `ports/libass/rendu-sur-plusieurs-fils.patch` |
| dav1d | BSD-2-Clause | via vcpkg |
| SDL3 | zlib | via vcpkg |
| Dear ImGui, spdlog, toml++ | MIT | via vcpkg |
| NVIDIA CUTLASS | **BSD-3-Clause**, © 2017-2026 NVIDIA CORPORATION & AFFILIATES | 4.8.1, en-têtes seulement, copiés dans `third_party/cutlass/` avec leur `LICENSE.txt` (les 259 fichiers que les noyaux du SPAN FP8 incluent, `docs/span-fp8-cutlass.md`) ; compilés dans le backend TensorRT. https://github.com/NVIDIA/cutlass |
| wayland-protocols, `presentation-time.xml` | MIT, © 2013-2014 Collabora, Ltd. | 1.38, copié dans `third_party/wayland-protocols/` (licence dans le fichier) ; code client produit par wayland-scanner à la compilation (`--present=wayland`). https://gitlab.freedesktop.org/wayland/wayland-protocols |
| wayland-client | MIT | bibliothèque du système, liée sous Linux quand elle est là |
| FreeType (FTL), HarfBuzz, FriBidi (LGPL-2.1), Fontconfig, lcms2 | licences de leurs projets | via libplacebo et libass |
| libunibreak | zlib | 7.0, via libass |

## Code repris

| Composant | Licence | Où |
|---|---|---|
| TweetNaCl (D. J. Bernstein, W. Janssen, T. Lange, P. Schwabe, S. Smetsers, B. van Gastel) | **domaine public** | Arithmétique de la vérification Ed25519 des mises à jour, réécrite en C++ (`src/app/play/ed25519.cpp`) ; le SHA-512 vient de libavutil (FFmpeg). https://tweetnacl.cr.yp.to |
| Mesures de luminance d'une dalle QD-OLED 27" 4K (RTINGS, PG27UCDM) | faits mesurés, cités | Courbe du limiteur du modèle d'écran (`src/render/display_model.cpp`). https://www.rtings.com |

FFmpeg, libplacebo et FriBidi sont des bibliothèques partagées (`lib/`) : vous pouvez les remplacer par une version modifiée de la même interface. Leurs sources sont disponibles aux adresses et versions ci-dessus.

## LUT d'HDR (data/luts)

4 LUT 3D `.cube` (« HDR LUT ver.9.1 », générées par DaVinci Resolve) ajoutées au dépôt à la demande de son propriétaire, qui répond de leur redistribution ; auteur et licence d'origine non indiqués dans l'archive reçue (ni readme ni licence), voir `data/luts/ATTRIBUTION.md`. À compléter dès qu'ils sont connus, ou à retirer si l'auteur s'y oppose.

## Plans de test du grade (data/calibration)

`calibration.mkv` : dix-huit plans fixes de 6 s (H.264 1920×1080, 108 s, extraits de Blu-ray et de WEB-DL), tirés d'Akudama Drive (saison 1, épisode 1), des Enfants du Temps (film), de Violet Evergarden (saison 1, épisode 1), de Frieren (saison 1, épisode 1), de Your Name (film), de Re:Zero kara Hajimeru Isekai Seikatsu (saison 4, épisode 18, © Tappei Nagatsuki, KADOKAWA / Re:Zero 4 Partners), de Tensei Shitara Slime Datta Ken (saison 4, épisode 24, © Fuse, Micro Magazine / Tensura Project) et de Black Clover (saison 2, épisode 5). Ajoutés à la demande du propriétaire du dépôt, qui répond de leur redistribution, pour les seuls tests et comparaisons du grade (test de bout en bout, banc de qualité, `tools/ring/banding.py`, `tools/grade/compare_shots.sh`). Le lecteur n'embarque pas cette vidéo et ne s'en sert plus pour calibrer : la calibration du grade adaptatif est retirée. Les images restent la propriété de leurs ayants droit ; liste et instants dans `data/calibration/SOURCES.md`.

## ArtCNN C4F32 (modèle livré depuis la 0.10.2) et C4F16
ArtCNN, © 2024 João Chrisóstomo, licence MIT (texte dans `tools/upscale/models/ArtCNN-LICENSE.txt`, à joindre à toute redistribution des modèles `2x_ArtCNN_C4F32_rgb_fp16.onnx` et `2x_ArtCNN_C4F16_rgb_fp16.onnx`).

## Anime4K (shaders embarqués)
Anime4K, © 2019-2021 bloc97, licence MIT. Neuf shaders GLSL de mpv (commit `7684e95`, `third_party/anime4k/`),
embarqués tels quels dans le lecteur pour l'agrandissement Anime4K (Vidéo > Modèle). https://github.com/bloc97/Anime4K

```
MIT License

Copyright (c) 2019 bloc97

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## lbpcascade_animeface (cascade de visages embarquée)
lbpcascade_animeface, © 2011 nagadomi, licence MIT (texte dans l'en-tête du fichier). Le fichier
`lbpcascade_animeface.xml` (commit `4433ab1`, `third_party/lbpcascade_animeface/`) est embarqué tel quel ; le
lecteur l'évalue avec son propre code (`src/media/face_cascade.cpp`), sans OpenCV, pour trouver la peau des
visages (option `face_skin`, éteinte par défaut). https://github.com/nagadomi/lbpcascade_animeface

```
The MIT License (MIT)

Copyright (C) 2011 by nagadomi@nurs.or.jp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Moteurs facultatifs téléchargés depuis l'écran de bienvenue
MoltenVK (Apache-2.0) sur Mac. TensorRT, CUDA et cuDNN sont sous les licences propriétaires de NVIDIA, que l'utilisateur accepte à l'écran avant le téléchargement.

## État de ce relevé
Licences des modèles et configuration de FFmpeg : vérifiées dans les dépôts amont et dans les scripts de construction. Les autres lignes sont celles annoncées par leurs projets, à relire au prochain relevé complet avec les fichiers de licence installés par vcpkg (`share/<port>/copyright`).
