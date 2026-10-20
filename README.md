# Ultimate Anime VideoPlayer

Lecteur vidéo écrit en C++ pour l'anime : agrandissement AnimeJaNai (SPAN),
interpolation (RIFE et au-delà), grade HDR adaptatif et RTX HDR, en temps réel.
L'export rejoue exactement le même traitement, sans horloge.

Le lecteur s'installe léger : il embarque notre moteur Vulkan, qui tourne sur
tout GPU récent. TensorRT, MIGraphX, OpenVINO, DirectML et les SDK RTX HDR sont
facultatifs et se choisissent au premier lancement, avec une présélection
adaptée à la machine.

## État

Jalon M1 en cours : squelette, cœur (`core`) et tests. Voir
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) pour les modules et leurs règles.

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

Sous macOS (Apple Silicon, macOS 14+), Vulkan passe par MoltenVK ; le paquet `AJP.app` se fait
ainsi, ou par le workflow « Release macOS » lancé à la main :

```sh
tools/deps/macos-from-source.sh ~/ajp-deps
AJP_DEPS_PREFIX=~/ajp-deps tools/release/package-macos.sh   # build/package/ajp-macos-arm64.zip
```

Sur les GPU AMD et Intel, les réseaux peuvent passer par le runtime du fabricant tandis que la
chaîne reste sur Vulkan : `-DAJP_WITH_MIGRAPHX=ON` (ROCm installé) construit
`ajp_backend_migraphx`, `-DAJP_WITH_OPENVINO=ON` construit `ajp_backend_openvino`. Chacun ne
s'essaie que choisi à l'écran de bienvenue, et refuse les autres fabricants (Vulkan prend alors
le relais).

## Licence

À décider (MPL-2.0 ou permissive, jamais GPL, car le lecteur charge TensorRT).
