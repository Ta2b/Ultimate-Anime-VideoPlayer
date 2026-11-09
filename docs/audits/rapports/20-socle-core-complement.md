# Complément de l'audit 20. Socle commun (core)

Branche : `claude/project-thread-shueno`. Commit audité : `0cfcdee`. Date : `2026-11-09`.

## Verdict

Les deux renvois reçus décrivent des défauts réels, mais déjà inscrits : le premier est l'effet (1) d'A18-01, le
second est A06-11. Seul le commentaire d'en-tête de `model_index.h`, devenu faux, est nouveau (A20-14, mineur).

## Renvois traités

| De | Où | Conclusion | Preuve |
|---|---|---|---|
| 18 | `src/core/log.cpp:82-95` | **Doublon d'A18-01** (effet 1). | Lu. `close_log()` (`log.cpp:163-165`) n'appelle que `spdlog::shutdown()`, qui remet le journal par défaut à nul. Ensuite, `log_channel` lit `source == nullptr` (`:83`) : aucune entrée du cache ne lui correspond (`:87-88`), donc `make_channel` est appelé (`:90`). Dans le processus du lecteur, `module_hub().hub` reste nul, puisque `install_module_log_sinks` n'est appelé que par les modules GPU (`src/gpu/module_logs.h:38`). `make_channel` arrive donc à `spdlog::default_logger()->clone(...)` (`:77`) sur un pointeur nul : segfault. A18-01 le démontre (code 139) et en donne la correction : un journal nul (`null_sink`) quand `default_logger()` est nul, plus un vidage puis un `ChannelCache` vidé dans `close_log`. Rien à ajouter. |
| 46 | `src/core/model_index.cpp:356-410`, `src/core/model_index.h:62-70` | **Doublon d'A06-11** pour le code mort ; **confirmé** pour le commentaire d'en-tête (A20-14). | Lu. `grep` dans `src/` : `parse_models_lock`, `models_for_profile`, `core::find_model` (`model_index.cpp:390`), `model_path` et `ModelEntry` n'ont aucun appelant hors de `model_index.cpp` et de `tests/unit/core/model_index_test.cpp`. Le `find_model` de `src/app/play/model_choice.cpp:44` est une autre fonction (`options.cpp:629`), qui cherche un fichier parmi des candidats. Aucun fichier `models.lock` n'existe dans le dépôt, ni dans `tools/release` ni dans `.github`. A06-11 (`SYNTHESE.md:124`) liste déjà ces symboles avec la même correction (les retirer avec leurs tests). |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A20-14 | Mineur | `src/core/model_index.h:1-6` | L'en-tête dit que « aucun binaire n'entre dans le dépôt » et que « chaque modèle […] se télécharge depuis l'URL de son index, puis se vérifie par sa taille et son SHA-256 ». C'est faux : aucun modèle ne se télécharge (le code de `models.lock` est mort, A06-11). Les modèles sont livrés dans le paquet, et `tools/upscale/models/2x_ArtCNN_C4F32_rgb_fp16.onnx` est versionné dans le dépôt (notes de `v0.10.2` : « le modèle ArtCNN est livré avec le lecteur »). Seuls les paquets de moteurs (`engines.lock`) suivent encore ce chemin. | Qui reprend le dépôt lit cet en-tête et conclut que les modèles sont vérifiés par SHA-256 à l'installation. Il ne cherche donc pas de contrôle d'intégrité ailleurs, alors que les modèles livrés n'en ont aucun au chargement. | Avec le retrait d'A06-11, réécrire l'en-tête pour décrire le seul index des moteurs : paquets téléchargés, taille et SHA-256 vérifiés. Dire que les modèles sont livrés avec le lecteur. | Lu (`model_index.h:1-6`, `ls tools/upscale/models`, `tools/release/notes/v0.10.2.md:4`). |

Aucune baisse de qualité d'image n'est en jeu dans ces deux renvois.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A18-01 | Toujours présent | `log.cpp:77` et `:163-165` inchangés au commit `0cfcdee`. |
| A06-11 | Toujours présent | Les symboles de `models.lock` n'ont toujours que leurs tests. |
