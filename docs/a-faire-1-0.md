# À faire avant la 1.0

Suivi au 06/11/2026, après le grand nettoyage (PR #530 à #533, [ADR 0012](adr/0012-lecteur-simple.md)).

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
