# Complément de l'audit 24. Interface : rendu, contrôles et messages

Branche : `claude/project-thread-bvmd7h`. Commit audité : `4b37a38`. Date : `2026-11-09`.

Renvois traités : les deux de la liste `complements/24.md`, venus des audits 25 et 43.

## Verdict

Le renvoi de 25 est confirmé : à l'accueil, « Run the wizard again » n'ouvre que l'ancien écran des moteurs, parce
que l'accueil jette le drapeau `wizard` de la demande. Nouveau défaut A24-24. Le renvoi de 43 est un doublon
d'A43-01, qui cite déjà cette ligne et dont la correction la couvre. Aucun des deux ne touche à la qualité de
l'image.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A24-24 | Mineur | `src/app/play/home_screen.cpp:307-316` (bouton : `src/app/play/settings_panel.cpp:282`) | Le visiteur de l'accueil prend `const OpenEngineScreen&` sans le nommer et construit `EngineScreen` sans `restart_wizard` (faux par défaut, `engine_screen.h:35-36`). `run_engine_screen` est appelé avec `revisit = true`, donc `wizard_mode = !revisit \|\| s.restart_wizard` vaut faux (`engine_screen.cpp:210`) : `view.revisit` passe à vrai et seul l'écran des moteurs s'affiche, sans langues, mises à jour, intégration ni calibration. La lecture, elle, passe `.restart_wizard = open != nullptr && open->wizard` (`engine_screen.cpp:350-357`). Le bouton est le même dans les deux cas (page Général > À propos, `about_page`, montrée aussi à l'accueil puisque le panneau y est ouvert d'office, `home_screen.cpp:239`). | Lecteur lancé sans vidéo, panneau de l'accueil, Général > À propos > « Run the wizard again » : l'écran des moteurs s'ouvre ; « Commencer » enregistre les moteurs et revient à l'accueil. Les langues, le mode de mise à jour, le lecteur par défaut et la calibration de la machine ne sont jamais proposés. Le cas visé est justement celui de l'accueil : quelqu'un qui veut refaire l'assistant avant d'ouvrir un épisode (nouvelle carte, autre écran). | Nommer le paramètre et passer `.restart_wizard = open.wizard` comme `handle_engine_change`. Puis, l'assistant ayant écrit `play.toml`, relire ce fichier dans `st.saved` et `running` (et `from_file`, `started`) avant `refresh()` : sinon un réglage touché ensuite à l'accueil (`dirty`) fait réécrire par `save()` l'ancienne chaîne par-dessus les choix de l'assistant, par la fusion de `save_settings` (`settings_toml.cpp:609-616`), le même mécanisme qu'A25-01 en lecture. Test : un `HomeState` dont l'action `OpenEngineScreen{.wizard = true}` doit construire un `EngineScreen` avec `restart_wizard` vrai (extraire la construction dans une fonction pure, partagée avec `handle_engine_change`). | Lu. |

## Renvois traités

| Renvoi | De l'audit | Conclusion | Note |
|---|---|---|---|
| `src/app/play/home_screen.cpp:307-316` : « Run the wizard again » depuis l'accueil n'ouvre que l'écran des moteurs | 25 | Confirmé : A24-24 | Pas un doublon : A25-01, A25-06 et A25-09 traitent l'assistant relancé en lecture, qui s'ouvre bien en entier. A24-24 est le seul chemin où il ne s'ouvre pas. Si A24-24 est corrigé, A25-06 (calibration annulée rendue en Auto) et A25-09 (assistant relancé depuis les valeurs par défaut) s'y appliquent aussi, et l'infobulle « Finish » d'A43-05 s'y montre déjà. |
| `src/app/play/home_screen.cpp:178` : « EDID illisible : OLED inconnu » en dur dans Vidéo > Affichage | 43 | Doublon d'A43-01 | A43-01 cite cette ligne et sa jumelle `player_run.cpp:290` dans son scénario, et sa correction demande de faire passer `display_text` par `Translated`. Vérifié au commit audité : les deux littéraux sont toujours là, montrés derrière `i18n::tr("Display: ")` (`settings_panel.cpp:145`), donc « Écran : EDID illisible : OLED inconnu » en français et « Display: EDID illisible : OLED inconnu » en anglais. Aucune entrée de traduction n'existe pour ce texte. Il s'affiche chaque fois que `window_display` rend rien (`window_vrr.cpp:93-99`, `read_display` sans EDID lu). Précision utile à la correction d'A43-01 : le texte doit être une clé `tr("…")` avec son entrée française, et le même dans les deux fichiers (une fonction `display_text(const std::optional<DisplayFacts>&)` à côté de `describe`, `display_edid.cpp:269`). Pas de nouvel ID. |

## Points positifs

- La lecture fait déjà ce qu'il faut : `handle_engine_change` lit le drapeau `wizard` et le porte jusqu'à
  `run_engine_screen` ; la correction de l'accueil n'a qu'à la reprendre.
- `describe(const DisplayFacts&)` passe tous ses morceaux par `i18n::tr` : seul le cas « aucun écran lu » échappe
  à la traduction.
