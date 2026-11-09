# Complément de l'audit 23. Réglages : le panneau

Branche : `claude/project-thread-q4styx`. Commit audité : `1af18a4`. Date : `2026-11-09`.

## Verdict

Les cinq renvois reçus sont tous déjà couverts par un défaut de `SYNTHESE.md` (A22-09, A38-01, A49-03, A49-04,
A49-05, A43-07). La lecture du champ du dossier des moteurs ajoute un défaut propre au panneau : « Annuler » et
« Rétablir » changent le dossier enregistré sans changer le texte affiché (A23-16).

## Cadrage retenu

Pas de question posée. Seuls les renvois de `audits-docs/complements/23.md` sont vérifiés dans le code. Une baisse de
qualité faite pour la vitesse compte comme défaut sauf mesure : les trois qui passent par ces renvois (FP8, anneau
RGB10A2, LUT composée) sont déjà notés par l'audit 49 avec leur gain connu ou son absence.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A23-16 | Mineur | `src/app/play/settings_engines.cpp:75-85`, `src/app/play/settings_panel.cpp:545-568`, `src/app/play/settings_panel.h:128-131` | Le champ « Dossier d'installation » ne recopie `engines_dir` dans son tampon qu'une fois par ouverture du panneau (`folder.read`). « Annuler » et « Rétablir » remplacent `draft_` en entier, `engines_dir` compris, sans remettre `folder.read` à faux : le brouillon (puis `play.toml`, `live_part` le porte, `settings.cpp:410`) reçoit l'ancienne valeur pendant que le champ montre toujours la nouvelle. Le tampon fait aussi 512 octets : un chemin plus long est coupé sans mot par `snprintf`, et la coupe est enregistrée à la première frappe. | Taper `D:\Moteurs`, quitter le champ (un pas d'annulation), puis cliquer « Annuler » pour défaire un autre réglage pris juste avant : `play.toml` reprend l'ancien dossier, le champ affiche `D:\Moteurs`. Au prochain lancement, TensorRT s'installe dans un dossier que le panneau ne montre pas. | Comparer à chaque image le tampon à `engines_dir` hors édition (`!ImGui::IsItemActive()`) et le recopier s'ils diffèrent, ou remettre `engines_folder_ = {}` à chaque Annuler, Rétablir et « Tout réinitialiser ». Agrandir le tampon (ou `InputText` sur `std::string` avec `ImGuiInputTextFlags_CallbackResize`). Corriger avec A22-09 (ne recopier qu'à la validation). | Lu. |

## Renvois vérifiés

| Renvoi | Où | Conclusion | Preuve |
|---|---|---|---|
| de 22 : dossier des moteurs recopié à chaque frappe | `src/app/play/settings_engines.cpp:81-84` | **Doublon d'A22-09.** Confirmé dans le code : `InputTextWithHint` rend vrai à chaque frappe et `engines_dir = folder.text.data()` suit ; `live_part` porte la valeur (`settings.cpp:410`), écrite au plus une fois par seconde. A22-09 décrit déjà le chemin à moitié tapé et la correction (recopier à la validation). Le défaut de synchronisation vu en relisant ce champ est noté à part (A23-16). | Lu. |
| de 24 : « Signaler un problème » copie la fin brute du journal | `src/app/play/settings_system.cpp:493-511` | **Doublon d'A38-01** (même cause qu'A39-03). Le code copie `ajp.log` tel quel depuis `size − 48 Kio` (`:498-503`) ; l'infobulle (`:507-509`) promet « sans votre nom d'utilisateur ni les dossiers de vos vidéos ». A38-01 nomme déjà les chemins complets journalisés et les racines de montage non nettoyées. La coupe au milieu d'une ligne reste A23-14, le presse-papiers vide sous Linux A24-03. | Lu. |
| de 49 : commentaire FP8 des préréglages | `src/app/play/auto_config.cpp:167-169` contre `:202` | **Doublon d'A49-03.** Le commentaire dit que les préréglages ne posent plus le FP8 ; la branche calibrée pose `PrecisionTier::Performance` dès que `cal.span_model` n'est pas vide. Qualité contre vitesse : gain mesuré de 6,5 % en lecture (`docs/span-fp8-cutlass.md`, cité par A49-03), sans que le b8f64 FP16 soit essayé. | Lu. |
| de 49 : commentaire RGB10A2 « derrière une option » | `src/app/play/player_run.cpp:171` contre `src/app/play/options.h:113-114` | **Doublon d'A49-04** (même cause qu'A21-03), qui cite déjà ce commentaire. Le commentaire dit « derrière --ring-rgb10a2, tant que la porte de banding n'est pas faite » ; `ring_rgb10a2 = true` par défaut, l'option est `--no-ring-rgb10a2`. Qualité contre mémoire : aucun gain de vitesse mesuré. | Lu. |
| de 49 : commentaire LUT « sans écart » | `src/render/cube_lut.h:38-42` | **Réfuté à la lettre, doublon d'A49-05 pour le fond.** Le commentaire ne promet l'absence d'écart que pour « une pile d'une LUT de 17 ou 33 points » refaite à 65 : c'est exact (64 est multiple de 16 et de 32, les nœuds fins tombent sur les cellules grossières, et l'interpolation trilinéaire d'une fonction multilinéaire par morceaux, mélange compris, la reproduit). Il se tait sur une pile de plusieurs LUT, où l'écart va jusqu'à 15,5 codes PQ (A49-05, A41-01) : le compléter en même temps que la correction d'A49-05. | Lu et raisonné ; pas de programme lancé. |
| de 38 : commentaire FP8 des préréglages | `src/app/play/auto_config.cpp:167-169` | **Doublon d'A49-03** (même renvoi que celui de 49 ci-dessus ; le choix du modèle RIFE de la même branche est A38-02). | Lu. |
| de 43 : séparateur « ; » en dur dans le bandeau « To apply » | `src/app/play/settings.cpp:493`, `:495`, affiché par `src/app/play/settings_panel.cpp:584-587` | **Doublon d'A43-07**, qui nomme déjà cette ligne. Confirmé : les raisons sont jointes par `" ; "` littéral (typographie française dans l'interface anglaise), la clé anglaise `" ; playback resumes at the same spot"` (`i18n_fr_settings.cpp:54`) porte la même espace, et le bandeau passe par `TextWrapped` sans `french_spaces` : en français le point-virgule peut ouvrir une ligne. | Lu. |

## Renvois

Aucun.

## Points positifs

- Annuler et Rétablir ne retiennent un état qu'au relâchement (`settings_panel.cpp:545-551`) : la saisie du dossier
  compte pour un seul pas, pas un par lettre.
- « Tout réinitialiser » garde le dossier des moteurs (`settings_panel.cpp:298`).

## Questions ouvertes

Aucune.
