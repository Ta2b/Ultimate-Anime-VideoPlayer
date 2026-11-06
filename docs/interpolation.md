# Interpolation

Le lecteur crée des images entre deux images de la source, pour suivre la fréquence de l'écran. Le réseau
qui les calcule s'appelle ici le « moteur du mouvement » (`MotionEngine`, `src/app/play/settings.h`). Le moteur
de calcul qui l'exécute (TensorRT, Vulkan, Core ML) est un autre choix : voir `docs/moteurs.md`.

## Moteurs du mouvement

| Moteur | Clé `play.toml` | Lecture | Export | Code |
|---|---|---|---|---|
| RIFE (4.22 lite, 4.25…) | `rife` | oui, par défaut | oui | `src/app/play/rife_*`, `kernels/spec/rife*` |
| DistilDRBA | `distil_drba` | oui | oui | `kernels/spec/distil_drba.h` |
| Apple (VideoToolbox) | `apple` | macOS seulement | non | `gpu/system_interpolator.h` |
| GMFSS Fortuna, Union | `gmfss`, `gmfss_union` | **débranché** | oui | `src/app/play/gmfss_*`, `src/gpu/cuda/backend/gmfss_*` |

## Ce que montre le menu

Le menu ne nomme pas les modèles, il propose des niveaux (`kInterpLevels`, `settings.h`). Le nom du modèle ne
s'affiche qu'en mode avancé, à côté du niveau.

| Niveau | Moteur, modèle |
|---|---|
| Très rapide | DistilDRBA (garde lui-même les poses, une image de retard) |
| Rapide | RIFE 4.22 lite (le défaut hors RTX) |
| Équilibré | RIFE 4.25 (le défaut sur une RTX) |
| Système (macOS) | interpolation d'Apple, sur un Mac seulement |

Un niveau dont le modèle manque n'est pas proposé. Le multiplicateur offre « Arrêt » et « Auto » en lecture.
Les facteurs fixes (×1,5 à ×8) ne sont proposés qu'à l'export. En lecture, ils restent possibles par
`--rife=N` ou par `play.toml`, pour le diagnostic.

## GMFSS débranché de la lecture

GMFSS donne les images intermédiaires les plus propres sur l'animation, mais coûte plusieurs fois RIFE. La
plupart des machines ne le tiennent pas en direct, et en lecture il avait des défauts. Le code reste entier.
Seule la lecture ne le propose plus :

- `offered()` (`settings.h`) dit non pour `Gmfss` et `GmfssUnion` : le menu ne les montre plus ;
- `run_chain` (`player_run.cpp`) remplace un `play.toml` qui les nomme par RIFE, avec la ligne de journal
  « débranchée de la lecture, RIFE à sa place » ;
- l'export (`export_run.cpp`) ne passe pas par `run_chain` : un `play.toml` qui nomme GMFSS (écrit à la main,
  `[interpolation] engine = "gmfss"`) l'utilise encore pour l'export.

Pour le rebrancher, il faut retirer la condition de `offered()` et le remplacement de `run_chain`. Ne le faire
qu'après une mesure en direct sur la carte visée, face à RIFE (`ajp bench`).
