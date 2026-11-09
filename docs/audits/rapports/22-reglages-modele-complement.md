# Complément de l'audit 22. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-r55xzk`. Commit audité : `1af18a4`. Date : `2026-11-09`.

## Verdict

Sur les trois renvois, deux donnent un défaut nouveau (A22-17 et A22-18, mineurs tous deux) et un est réfuté :
`--data-dir` partage les moteurs par choix écrit, et le dossier `ajp/ajp` est l'emplacement normal de SDL pour
toutes les lectures. Aucun des trois ne touche à la qualité d'image.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les trois renvois de `audits-docs/complements/22.md`, sans
question à Tatoub, sans toucher au code. Les ID suivent le rapport 22, qui s'arrête à A22-16. Tout est lu, rien
n'est lancé.

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 23 | `src/app/play/settings.h:304-307` | Confirmé : **A22-17** | Le commentaire du seuil HDR est au-dessus de `kSubtitleDelayMs`, pas de `kTrueHdrPeak`. |
| 43 | `src/app/play/settings.cpp:130-138` | Confirmé, autrement que dit : **A22-18** | Le menu ne peut pas afficher « GMFSS » : aucune valeur GMFSS n'entre dans `PlaySettings`. Les deux `case` sont du code mort, avec trois gardes et deux valeurs d'énumération qui ne servent qu'à eux. |
| 40 | `src/app/play/options.cpp:254-255`, `:491` | Réfuté | Voir « Renvoi réfuté ». |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A22-17 | Mineur | `src/app/play/settings.h:305-307` | Le commentaire « Le bas de la plage est le seuil du HDR (kMinHdrPeakNits, host_facts.h, P3.66) : un pic réglable fait toujours une sortie HDR » précède `kSubtitleDelayMs` (l. 307) et non `kTrueHdrPeak` (l. 308). Lu tel quel, il dit que −30 s est un seuil HDR. Le lien vrai est tenu par `static_assert(limits::kTrueHdrPeak.lo == kMinHdrPeakNits)` (`host_facts.cpp:277`), seul garde-fou. | Qui touche la plage des sous-titres croit toucher le seuil HDR, ou l'inverse ; qui cherche pourquoi `kTrueHdrPeak` commence à 350 ne trouve rien au-dessus. | Remonter `kTrueHdrPeak` juste sous le commentaire (ou le commentaire sous `kSubtitleDelayMs`). Aucun effet sur le binaire. | Lu |
| A22-18 | Mineur | `src/app/play/settings.cpp:118-121`, `:134-137`, `src/app/play/settings.h:53-56`, `src/app/play/player_run.cpp:104-107`, `src/app/play/export_job.cpp:49-50`, `src/app/play/free_engines.h:27-28` | Code mort de GMFSS (retiré en 0.10.0). `label(MotionEngine)` rend encore « GMFSS Fortuna » et « GMFSS Fortuna Union », mais aucune valeur `Gmfss` ou `GmfssUnion` n'atteint `PlaySettings` : `play.toml` les ramène à RIFE dès la lecture (`settings_toml.cpp:253`), le menu ne les propose pas (`offered`, `settings.h:65-68`), et aucune affectation de `motion_engine` dans `src/` ne les produit (relevé de `motion_engine =`). Les gardes `is_gmfss` de `run_chain` et `export_chain`, leur ligne de journal « débranchée de la lecture », et les deux `case` de `free_engines.h` ne se déclenchent donc jamais. Les deux valeurs d'énumération ne restent que pour que la boucle de `settings_toml.cpp:245-251` reconnaisse « gmfss » et « gmfss_union » par `name()`. | Aucun effet visible aujourd'hui. Le risque est à l'entretien : un nouveau moteur ajouté après `Apple` passe par une énumération de cinq valeurs dont deux mortes ; un futur `switch` doit traiter deux cas fantômes ; les docs et le journal laissent croire que GMFSS peut encore arriver jusqu'à la chaîne. | Traiter « gmfss » et « gmfss_union » comme « optical_flow » (`settings_toml.cpp:243-244` : nom reconnu, RIFE à sa place), puis retirer `Gmfss`, `GmfssUnion`, `is_gmfss`, les deux `case` de `name` et de `label`, les gardes de `player_run.cpp` et `export_job.cpp`, et ramener `kMotionEngineCount` à 3. Garder le test de relecture d'un ancien fichier (`settings_test.cpp:68-69`) ; adapter ceux qui construisent `MotionEngine::Gmfss` à la main (`settings_test.cpp:66`, `:71`, `:82-83`, `:90`, `auto_config_test.cpp:32`). Vérifier qu'aucun fichier n'enregistre l'énumération par sa valeur numérique (rien trouvé dans `src/`). | Lu |

## Renvoi réfuté

**De 40, `--data-dir` n'isole pas les moteurs (dossier `ajp` doublé).** Les deux constats sont exacts, mais ni
l'un ni l'autre n'est un défaut.

- Partage des moteurs : c'est le comportement voulu et écrit. L'aide dit « settings and log in this folder
  (engines and models shared) » (`options.cpp:491`) ; `main.cpp:411-412` dit « réglages, profils et journal
  ailleurs (les essais du benchmark) ; moteurs et modèles restent ». `--data-dir` ne change que
  `settings_dir` (`main.cpp:413-416`), et `PlayerPaths::engines` reste `local_data(data) / "engines"`
  (`main.cpp:478`, `:489`). C'est ce qu'il faut au banc : `auto_bench_run.cpp:73`, `:89` relance le lecteur avec
  `--data-dir` pour mesurer sans écrire `play.toml`, et des moteurs isolés obligeraient chaque essai à
  reconstruire ses moteurs TensorRT (plusieurs minutes chacun).
- Dossier `ajp/ajp` : c'est `SDL_GetPrefPath("ajp", "ajp")` (`main.cpp:337`), soit organisation puis
  application, pour toutes les lectures, avec ou sans `--data-dir`. `user_data` le sait et remonte de deux
  niveaux (`default_player.cpp:215-219`, `default_player.h:55-57`). Les docs le donnent comme emplacement normal
  (`docs/handoff-pc.md:54`, `docs/luts.md:23`). Le changer déplacerait les données de tous les utilisateurs
  pour un gain nul.

Remarque voisine, non auditée : avec `--data-dir`, `play.toml` est lu dans le dossier donné mais `--bake-looks`
lit les LUT sous `data / "luts"` (`main.cpp:453-454`), alors que `docs/luts.md:23` présente le dossier `luts`
comme celui de `play.toml`. Sans effet hors des essais ; à regarder par l'audit 19 si le banc s'en sert.

## Baisses de qualité pour la vitesse

Aucune dans ces trois renvois.
