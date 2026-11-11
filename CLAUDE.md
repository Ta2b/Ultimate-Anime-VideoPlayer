# Guide de reprise

Pour qui reprend ce dépôt, humain ou IA. Lire ceci d'abord, puis [`docs/README.md`](docs/README.md) qui dit quels
documents décrivent le code actuel et lesquels sont historiques.

## Le produit

`ajp`, un lecteur vidéo C++23 pour l'anime : agrandissement SPAN ×2 (AnimeJaNai), interpolation (RIFE 4.22 lite,
DistilDRBA au choix) en mode Auto, grade HDR anime guidé par plan, export qui rejoue la même chaîne. Linux,
Windows ; macOS est délaissé pour l'instant (plus construit par la CI ni par les releases, novembre 2026). Le propriétaire est Tatoub ; il écrit en français et décide des releases.

Principes tenus depuis le tri de novembre 2026 ([ADR 0012](docs/adr/0012-lecteur-simple.md)) :

- **un seul HDR** : le grade HDR anime. La sortie HDR ou SDR suit le système, sans bouton (`system_hdr`,
  `src/app/play/host_facts.h`) ;
- **pas de profils** : `play.toml` garde la dernière chaîne, `user.toml` ne garde que `[keys]` ;
- **menu en trois catégories** : Vidéo, Audio et lecture, Général (`kPages`, `src/app/play/settings_panel.cpp`) ;
- **moteurs** : Vulkan pour tout GPU (le seul pour AMD et Intel), TensorRT sur NVIDIA RTX, Core ML sur Mac.
  MIGraphX, OpenVINO et GMFSS sont retirés des sources : n'en remettre aucun sans une mesure face à Vulkan (ou
  à RIFE pour GMFSS) (`docs/moteurs.md`, `docs/interpolation.md`).

## Où est quoi

| Dossier | Contenu |
|---|---|
| `src/core` | cadence, choix des moteurs (`engines.h`), raccourcis (`keys.h`), écriture atomique (`file_sync.h`), chemins UTF-8 (`utf8_path.h`), écran et source (`display.h`) ; le profil v5 ne garde que ses types |
| `src/app/play` | le lecteur : boucle (`player.cpp`, `player_run.cpp`), chaîne en direct (`chain*.cpp`), menu (`settings_*.cpp`), réglages et `play.toml` (`settings.h`, `settings_toml.cpp`), assistant (`wizard.*`, `first_run.cpp`), traductions (`i18n_fr_*.cpp`) |
| `src/render` | libplacebo, grade (`grade_params.*`), shaders `.hook` (`src/render/shaders/`), LUT (`cube_lut.*`) |
| `src/gpu` | backends chargés à l'exécution : `vulkan`, `cuda` (TensorRT), `coreml` |
| `src/av`, `src/media`, `src/audio`, `src/subs`, `src/export`, `src/prepare`, `src/kernels` | voir `docs/ARCHITECTURE.md` |
| `tests/unit/<module>` | Catch2, un exécutable par module |
| `tools/` | dépendances (`deps`), paquets (`release`), bancs, scripts de modèles |

## Construire et tester

```sh
tools/deps/linux-from-source.sh ~/ajp-deps
export AJP_DEPS_PREFIX=~/ajp-deps
cmake --preset linux-prefix && cmake --build --preset linux-prefix
ctest --preset linux-prefix
```

Pour reconstruire plus vite en local : `-DAJP_PCH=ON` (en-têtes précompilés, environ un quart de moins sur un module
qui journalise ; la CI les laisse éteints pour ccache et pour qu'une inclusion oubliée reste une erreur).

`linux-prefix` est en Debug : ses temps ne valent rien. Un banc ou une mesure se fait sur le binaire livré,
`linux-release` (Linux), `windows-release`, `macos-release`.

Sans GPU, Vulkan logiciel (lavapipe) suffit : `ajp --software-gpu`. Les tests GPU se lancent avec
`AJP_TEST_VULKAN_VALIDATION=off` quand la couche de validation manque. CUDA ne se construit que sur un poste
NVIDIA (`linux-prefix-cuda`).

Voir l'interface sans écran : `xvfb-run ajp --software-gpu --mute --no-upscale --keys=1:p
--snapshot=capture.ppm --snapshot-at=3 video.mkv` (la touche `p` ouvre le menu).

## Règles que la CI vérifie (`.github/workflows/rules.yml`)

- `clang-format-20` sur tous les `.cpp` et `.h` (`--dry-run --Werror`) ;
- un fichier vise moins de 500 lignes, la CI échoue au-delà de 800 ;
- aucun `getenv` : une option vient de `play.toml` ou de la ligne de commande (`options.cpp`) ;
- les `path::string()` de `src/` ne peuvent que diminuer : un chemin passe par `core::to_utf8` (`src/core/utf8_path.h`) ;
- les shaders se compilent ; la construction est en `-Werror`.

## Conventions et pièges

- **Langue** : commentaires, journaux et documents en français. Les textes de l'interface sont en anglais dans
  `i18n::tr("…")`, avec leur traduction dans `src/app/play/i18n_fr_*.cpp`. `tests/unit/app/i18n_test.cpp`
  échoue si un littéral de `tr()` n'a pas d'entrée, ou si un journal français sort d'un fichier qu'il ne liste
  pas.
- **Shaders `.hook`** : le lecteur de libplacebo prend toute suite `//!` pour une directive, même dans un
  commentaire. N'écrire `//!` qu'en début de directive.
- **Compatibilité des fichiers** : la lecture de `play.toml` et `user.toml` ignore les clés inconnues ; une
  fonction retirée doit laisser un ancien fichier lisible (ajouter un test qui le relit).
- **TensorRT** : toute modification de `src/core`, `src/gpu`, `src/kernels` ou `src/prepare` change l'empreinte
  du module (`tools/release/trt-fingerprint.sh`) et demande un nouveau paquet Linux.
- **Releases** : aucun fichier de NVIDIA dans nos archives (le module et `trtexec` seulement). Pas de release
  ni d'étiquette sans demande explicite de Tatoub.

## Ce qui reste

[`docs/a-faire-1-0.md`](docs/a-faire-1-0.md) : paquets TensorRT, banc Windows contre Linux, code mort repéré.
