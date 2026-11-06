# Ultimate Anime VideoPlayer

Lecteur vidéo écrit en C++ pour l'anime, en temps réel :

- **agrandissement** AnimeJaNai (SPAN ×2) ;
- **interpolation** (RIFE 4.22 lite par défaut, DistilDRBA au choix) en mode Auto, qui suit ce que le GPU tient ;
- **grade HDR anime** : l'anime SDR passe en HDR10 avec un grade guidé par plan. Le HDR suit le système, sans
  bouton : sortie HDR quand Windows ou le compositeur envoient du HDR à un écran assez lumineux, sinon SDR gradée.

L'export rejoue exactement le même traitement, sans horloge.

Le lecteur s'installe léger : il embarque son moteur **Vulkan**, qui tourne sur tout GPU récent (AMD, Intel,
NVIDIA). Sur une carte NVIDIA RTX, **TensorRT** se propose au premier lancement et se télécharge à part ; sous
macOS, les réseaux passent par **Core ML**. MIGraphX et OpenVINO restent dans les sources mais sont débranchés du
lecteur (voir [`docs/moteurs.md`](docs/moteurs.md)).

## Utilisation

`ajp` sans fichier ouvre l'écran d'accueil ; `ajp video.mkv` lit la vidéo. Le menu des paramètres (touche `p`) a
trois catégories :

| Catégorie | Pages |
|---|---|
| Vidéo | Post-traitement (niveau Économe à Ultra, agrandissement, interpolation), Couleurs, Affichage |
| Audio et lecture | Son, Lecture (langues, sous-titres, épisode suivant) |
| Général | Application (langue, mises à jour, lecteur par défaut), Moteurs, Export, Statistiques, À propos |

Le niveau **Auto** choisit la chaîne selon la carte, l'écran et la vidéo ; un réglage fait à la main passe en
Personnalisé. Les réglages vivent dans `play.toml` (dossier des données du lecteur), les moteurs choisis dans
`settings.toml`, les raccourcis dans `user.toml` (`[keys]`). `ajp --help` liste les options et les touches.

## État

Version 0.9.1. Lecture, agrandissement, interpolation, grade HDR anime et export sous Linux, Windows et macOS.
Pour comprendre le code : [`CLAUDE.md`](CLAUDE.md) (guide de reprise, pour un humain ou une IA), puis
[`docs/README.md`](docs/README.md) (carte des documents, vivants et historiques),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (modules et règles) et [`docs/adr/`](docs/adr/README.md) (décisions).

## Construire

Prérequis : CMake 3.28+, Ninja, un compilateur C++23 (GCC 13+, Clang 20+, MSVC 17.10+).

Avec vcpkg (Windows, CI, versions livrées) :

```sh
export VCPKG_ROOT=/chemin/vers/vcpkg
cmake --preset debug          # lecteur complet
cmake --build --preset debug
ctest --preset debug
```

`cmake --preset core` ne construit que le cœur et ses tests.

Sous Linux sans vcpkg, les mêmes versions se construisent depuis leurs dépôts :

```sh
tools/deps/linux-from-source.sh ~/ajp-deps      # --core pour le cœur seul
export AJP_DEPS_PREFIX=~/ajp-deps
cmake --preset linux-prefix && cmake --build --preset linux-prefix
ctest --preset linux-prefix
```

Sous macOS (Apple Silicon, macOS 14+), Vulkan passe par MoltenVK, et les réseaux par Core ML
(Neural Engine et GPU, au travers d'ONNX Runtime) avec `ajp_backend_coreml` :

```sh
tools/deps/macos-from-source.sh ~/ajp-deps
AJP_DEPS_PREFIX=~/ajp-deps tools/release/package-macos.sh   # build/package/ajp-macos-arm64.zip
```

## Paquets

- **Linux** : `tools/release/package-linux.sh` produit `ajp-linux-x64.tar.gz` (lecteur, Vulkan, `engines.lock`).
  Le paquet TensorRT se construit sur un poste NVIDIA (`tools/release/trt-package.sh`).
- **Windows** : vcpkg (`windows-release`), puis `tools/release/package-windows.ps1` produit
  `ajp-windows-x64.zip` et, avec `AJP_WITH_CUDA=ON`, le paquet TensorRT.
- **macOS** : `AJP.app`, ci-dessus, ou le workflow « Release macOS ».

Aucun fichier de NVIDIA n'est dans nos archives : le paquet TensorRT ne porte que notre module et `trtexec`, et
le lecteur télécharge les bibliothèques de NVIDIA après l'accord de licence ([`docs/moteurs.md`](docs/moteurs.md)).

## Licence

Mozilla Public License 2.0 (fichier `LICENSE`). Les composants tiers et les modèles gardent leur licence : voir
`THIRD_PARTY_NOTICES.md`.
