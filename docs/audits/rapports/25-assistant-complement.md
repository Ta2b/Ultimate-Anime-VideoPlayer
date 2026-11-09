# Complément de l'audit 25. Assistant de premier lancement

Branche : `claude/project-thread-ohwowc`. Commit audité : `b93b670`. Date : `2026-11-09`.

Renvois reçus : ceux de l'audit 43 (`/mnt/project-files/audits-docs/complements/25.md`). Chacun a été relu dans le
code au commit ci-dessus.

## Verdict

Sur deux renvois, un est un doublon de A43-07, l'autre est confirmé comme défaut mineur nouveau (A25-19). Aucun ne
touche la qualité d'image.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A25-19 | Mineur | `src/app/play/first_run.cpp:445` (assistant), `src/app/play/settings_system.cpp:118` (panneau), traductions `src/app/play/i18n_fr_wizard.cpp:77` et `src/app/play/i18n_fr_system.cpp:66` | Deux textes pour la même option de mise à jour. L'assistant décrit « Automatic » par « Downloaded, installed at next launch. », le panneau par « Downloaded, verified, installed at next launch. ». Les deux sont vrais, mais l'assistant omet la vérification, qui est faite : signature ed25519 du manifeste (`updater.cpp:422-430`), SHA-256 et taille de l'archive au téléchargement (`updater.cpp:470`) puis de nouveau avant l'installation (`updater.cpp:561-562`). C'est aussi deux clés de traduction à tenir pour une phrase. Les deux autres modes partagent déjà leur texte (« A notification; you start the update yourself. », « No checks. »). | Premier lancement, page Mises à jour : « Téléchargée, installée au prochain lancement. ». Plus tard, Général > Mises à jour, survol de « Automatique » : « Téléchargée, vérifiée, installée au prochain lancement. ». L'utilisateur prudent qui hésitait à l'assistant n'apprend la vérification qu'après coup. | Une seule table des modes (libellé et détail), par exemple sortie de `kModes` de `settings_system.cpp:116-119` dans un en-tête commun, lue par l'assistant (`first_run.cpp:441-449`) et par le panneau ; garder « Downloaded, verified, installed at next launch. » et retirer l'entrée « Downloaded, installed at next launch. » de `i18n_fr_wizard.cpp:77`. | Lu. |

## Renvois traités

| Renvoi | Où | Conclusion | Preuve |
|---|---|---|---|
| de 43 | `src/app/play/first_run.cpp:29-31` | **Doublon de A43-07** (`SYNTHESE.md`). Confirmé dans le code : `note()` appelle `ImGui::TextWrapped` sur le texte brut, alors que les notes du panneau passent par `french_spaces` (`panel_style.h:33`, `:101`). Toutes les notes de l'assistant en héritent, y compris les détails des boutons radio (`radio_note`, `first_run.cpp:393-401`). `i18n_fr_wizard.cpp` compte 41 lignes avec une espace ordinaire avant « : ; ? ! » ou autour de « » » (par exemple `:12`, `:14`, `:20`), qu'ImGui peut couper en début de ligne. La correction proposée par A43-07 (appliquer `french_spaces` dans `tr()` côté français) couvre ce cas sans toucher l'assistant ; aucun nouvel ID. | Lu, et `grep -c` sur `i18n_fr_wizard.cpp`. |
| de 43 | `src/app/play/first_run.cpp:445` | **Confirmé : A25-19.** Pas de doublon : A43-04 (cinq noms pour le test de la machine) et A43-05 (infobulle « Finish ») parlent d'autres textes. | Lu. |

## Qualité d'image

Rien à signaler : les deux renvois portent sur des textes.
