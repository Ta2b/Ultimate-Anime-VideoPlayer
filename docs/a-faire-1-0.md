# À faire avant la 1.0

Suivi après le grand nettoyage (PR #530 à #533, [ADR 0012](adr/0012-lecteur-simple.md)).

## Avant une release

- **Paquet TensorRT Linux 0.9.1** : le module a changé, son empreinte (`tools/release/trt-fingerprint.sh`) aussi ;
  à reconstruire sur le poste NVIDIA Linux (`tools/release/trt-package.sh`). Aucune release sans demande de Tatoub.
- **TensorRT Windows** : `release-windows.yml` avec `tensorrt=true`, puis `tensorrt=false` sur la même étiquette
  (voir `docs/moteurs.md`).
- **Banc Windows contre Linux** sur le même PC, mêmes réglages.

## Code mort à retirer (sans effet sur l'image)

- `src/render/curve.*` : les fonctions de courbe et de cuisson qui ne servaient qu'au HDR adaptatif.
- `contract_metadata` et la branche `look_clean` du shader, si la mesure confirme qu'elles ne changent rien.
- Les clés `grade.look` et `grade.look_mix` de `play.toml`, écrites mais plus lues par le grade anime.
- `Priority` (Fluidité, Équilibre, Image) : absente du menu, encore lue dans `play.toml` par `apply_preset`.
  À retirer ou à rattacher au niveau de traitement.

## Reporté

- **GMFSS Fortuna Union : artefacts** (fond vert saturé, formes en miroir, traînées), en FP16 comme en FP32. Seule
  différence avec Base : le RIFE intégré du graphe fusion (`tools/gmfss/gmfss_graph.py`, `Fusion.forward`). GMFSS
  est débranché de la lecture (`docs/interpolation.md`) ; mesure GPU à lancer sur ordre de Tatoub.

## Audit 0.9.1, phase 5 : ce qui reste

La phase 5 du plan d'audit a livré une règle par contrôle de CI (complexité, `path::string()`, commentaires,
version unique) et les corrections ponctuelles. Restent les chantiers d'architecture, chacun une PR à lui :

- **P5.1** code mort : GMFSS. L'import `.ini`, `resolve()`, le `Plan`, `pace_rule`, `Capabilities`, la pile de
  LUT morte (`curve`, `looks`, `look_mix*`), MIGraphX, OpenVINO et WinML sont retirés (l'écran et la source sont
  dans `core/display.h`).
- **P5.4** LUT et `calibration.mkv` hors de git (attend la licence des LUT, P7.7).
- **P5.5** options fantômes et champs TOML morts ; **P5.13** utilitaires réécrits (refus, lecture de
  fichier, minuscules…) ; **P5.30** seuils en ligne ; **P5.31** duplications de l'application.
- **P5.8** clang-tidy sur Windows, macOS et TensorRT.
- **P5.15** et **P5.16** : découper `run_chain`, `play`, `UpscaleStage::outputs`… (les 40 `NOLINTNEXTLINE` de
  complexité les nomment) et une interface `Interpolator` par moteur.
- **P5.19** ABI des moteurs stable par extensions ; **P5.24** cycle `gpu` ⇄ `prepare` ; **P5.25** en-têtes
  précompilés ; **P5.26** graphe des dépendances vérifié en CI ; **P5.27** état global mutable ; **P5.32** un
  module de chemins.
- **P5.40** langue des journaux : attend la décision P7.5.
