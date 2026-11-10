# À faire avant la 1.0

Suivi après le grand nettoyage (PR #530 à #533, [ADR 0012](adr/0012-lecteur-simple.md)).

## Avant une release

- **Release** : une étiquette poussée suffit (`release.yml`, `docs/signature-des-releases.md`) ; les paquets TensorRT
  Linux et Windows se construisent en CI s'ils manquent. Aucune release sans demande de Tatoub.
- **Banc Windows contre Linux** sur le même PC, mêmes réglages.

## Code mort à retirer (sans effet sur l'image)

- `src/render/curve.*` : les fonctions de courbe et de cuisson qui ne servaient qu'au HDR adaptatif.
- `contract_metadata` et la branche `look_clean` du shader, si la mesure confirme qu'elles ne changent rien.
- Les clés `grade.look` et `grade.look_mix` de `play.toml`, écrites mais plus lues par le grade anime.
- `Priority` (Fluidité, Équilibre, Image) : absente du menu, encore lue dans `play.toml` par `apply_preset`.
  À retirer ou à rattacher au niveau de traitement.

## Audit 0.9.1, phase 5 : ce qui reste

La phase 5 a livré ses contrôles de CI (complexité avec `tools/ci/complexity-baseline.txt`, graphe des modules,
utilitaires réécrits, `path::string()`, commentaires, version unique), le code mort, les options fantômes, les
utilitaires communs (`core/text.h`, `av/av_error.h`, `ajp::refuse`), les seuils nommés, le module de chemins
(`paths.h`), l'état du menu hors des `static`, les en-têtes précompilés (`AJP_PCH`), clang-tidy sur TensorRT et
Core ML, et l'empreinte de la frontière des moteurs (`cmake/BoundaryHash.cmake`). Les journaux restent en français
(P5.40). Restent, chacun une PR à lui :

- **P5.1** les additions atomiques de float du moteur Vulkan, sans noyau depuis GMFSS : à retirer après une mesure
  sur un vrai GPU.
- **P5.4** LUT et `calibration.mkv` hors de git (attend la licence des LUT, P7.7).
- **P5.8** clang-tidy sur Windows.
- **P5.15** suite du découpage : `run_chain` (la surveillance du mode Auto en est sortie, `auto_watch.h`), `play`,
  `UpscaleStage::outputs` ; la base de complexité ne peut que diminuer.
- **P5.16** une interface `Interpolator` par moteur : à faire avec un GPU sous la main, la chaîne d'interpolation
  ne se vérifie pas sur lavapipe seul.
- **P5.19** la moitié « extensions » : une interface de base stable et des extensions par identifiant. L'empreinte
  refuse déjà un module bâti sur une autre frontière.

## Recette : ce que la CI joue déjà (P7.12)

Le job lavapipe lance les cas `[e2e]` de `tests/unit/app/e2e_play_test.cpp` : la lecture d'un clip et son image,
le scénario 14 pour une source 853×480, et le scénario 15 (fichier tronqué lu jusqu'à sa fin, fichier illisible
refusé avec un message). Les tests unitaires couvrent l'ordre des épisodes du scénario 19 (`episodes_test`), la
fenêtre unique du scénario 21 (`single_instance_test`) et la lecture de `[keys]` du scénario 25
(`keys_test`, `text_file_test`). Les autres scénarios demandent un vrai GPU, un écran HDR ou VRR, ou un réseau : ils se
jouent sur les machines de Tatoub à la fin.
