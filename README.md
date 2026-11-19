<img src="tools/release/linux/kirari.png" alt="" width="96" align="right">

# Kirari Player

[![Licence : MPL 2.0](https://img.shields.io/badge/licence-MPL%202.0-blue.svg)](LICENSE)
![Version 0.11.0](https://img.shields.io/badge/version-0.11.0-informational.svg)
![Linux · Windows](https://img.shields.io/badge/plateformes-Linux%20%C2%B7%20Windows-lightgrey.svg)

Un lecteur vidéo fait **uniquement pour l'anime**. Il agrandit l'image, ajoute des images intermédiaires et
convertit l'anime SDR en HDR, **en temps réel**, pendant la lecture, sur la carte graphique de votre PC. Il
s'ouvre comme n'importe quel lecteur : double-clic sur une vidéo, et c'est tout.

## Ses points forts

- **Spécialisé anime.** Chaque étape est choisie et réglée pour l'animation : modèles entraînés sur l'anime,
  traits nets et aplats propres, dessins tenus et coupes de scène respectés, interpolation seulement aux
  cadences d'anime (23,976 à 30 i/s).
- **Upscale et interpolation en temps réel, haute performance.** Agrandissement ×2 par les modèles
  **AnimeJaNai** (SPAN), interpolation par **RIFE** ou **DistilDRBA** jusqu'à la fréquence de votre écran.
  Le mode Auto règle la chaîne selon la carte, l'écran et la vidéo, et redescend tout seul quand la carte ne suit
  plus. Moteurs coupés sur mesure, noyaux CUDA maison et FP8 sur les RTX récentes pour tenir la cadence.
- **Auto HDR spécialisé anime.** L'anime SDR devient du HDR10 avec un grade pensé pour l'animation : surfaces
  à luminosité stable d'un plan à l'autre, éclats (lumières, effets) qui montent vers le pic de l'écran, noirs
  intacts. La sortie HDR ou SDR suit votre système, sans réglage ; sur un écran SDR, le même grade s'applique en
  SDR.
- **Stabilisation temporelle** (TemporalFix, facultative) : retire le scintillement du grain et de la
  compression avant l'agrandissement.
- **Export** : la même chaîne rejouée image par image vers un fichier MKV, MP4 ou MOV.
- Et ce qu'on attend d'un lecteur : sous-titres (libass), pistes par langue préférée, épisode suivant, passage du
  générique.

## Cartes graphiques

**Le lecteur est surtout testé et optimisé sur NVIDIA RTX**, avec le moteur TensorRT : c'est là qu'il donne
le meilleur et que tout a été mesuré.

Les autres cartes (AMD, Intel, NVIDIA sans TensorRT) passent par le moteur **Vulkan**, livré avec le lecteur.
Elles ont l'agrandissement et l'interpolation et **devraient fonctionner**, mais elles n'ont pas encore été
essayées sur du vrai matériel. Quelques fonctions restent propres à TensorRT pour l'instant (stabilisation
temporelle, GMFSS). Si quelque chose ne va pas sur votre carte, ouvrez une issue avec le journal (`kirari.log`) :
c'est ainsi que ces voies seront corrigées. Pour les petites cartes et les graphiques intégrés, Anime4K
(shaders) prend la place de SPAN.

| Carte | Moteur | Livraison |
|---|---|---|
| NVIDIA RTX | TensorRT | proposé au premier lancement, téléchargé à part |
| AMD, Intel, autres NVIDIA | Vulkan | livré avec le lecteur |

## Installation

Téléchargez la dernière version sur la page des [releases](https://github.com/Ta2b/kirari-player/releases) :

| Système | Fichier |
|---|---|
| Windows 10 et 11 (x64) | `kirari-windows-x64-setup.exe` (installeur), ou `kirari-windows-x64.zip` |
| Linux (x64) | `kirari-linux-x86_64.AppImage`, ou `kirari-linux-x64.tar.gz` |

Mettez à jour le pilote de votre carte graphique avant le premier lancement. Kirari Player s'appelait ajp jusqu'à la
0.11 : il reprend tout seul vos réglages et vos moteurs déjà construits. macOS n'est plus publié pour
l'instant.

Au premier lancement, un court assistant demande vos langues, propose TensorRT sur une NVIDIA RTX, puis peut
calibrer le lecteur sur votre machine. Ensuite, **`p`** ouvre les paramètres, **Tab** les statistiques, et
`kirari --help` liste les options de la ligne de commande.

## Construire depuis les sources

CMake 3.28+, Ninja et un compilateur C++23 (GCC 13+, Clang 20+, MSVC 17.10+) :

```sh
tools/deps/linux-from-source.sh ~/kirari-deps
export KIRARI_DEPS_PREFIX=~/kirari-deps
cmake --preset linux-prefix && cmake --build --preset linux-prefix
ctest --preset linux-prefix
```

Windows se construit avec vcpkg (`cmake --preset debug`). Pour reprendre le code :
[`CLAUDE.md`](CLAUDE.md) (guide de reprise), puis [`docs/README.md`](docs/README.md) (carte des documents).

## Remerciements

[the-database](https://github.com/the-database) pour **AnimeJaNai** et
[mpv-AnimeJaNai](https://github.com/the-database/mpv-AnimeJaNai), dont ce lecteur est parti ;
[hzwer](https://github.com/hzwer/Practical-RIFE) pour **RIFE** ; routineLife1 pour **DistilDRBA** ;
[pifroggi](https://github.com/pifroggi/vs_temporalfix) pour **TemporalFix** ;
[bloc97](https://github.com/bloc97/Anime4K) pour **Anime4K** ;
[AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt) ;
[libplacebo](https://code.videolan.org/videolan/libplacebo), [FFmpeg](https://ffmpeg.org),
[libass](https://github.com/libass/libass), [SDL3](https://libsdl.org) et
[Dear ImGui](https://github.com/ocornut/imgui).

## Licence

Le code est sous [Mozilla Public License 2.0](LICENSE). Les modèles gardent leur licence : AnimeJaNai est sous
**CC BY-NC-SA 4.0** (usage non commercial), RIFE, DistilDRBA et Anime4K sous MIT, TemporalFix sous Apache 2.0.
Le détail est dans [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
