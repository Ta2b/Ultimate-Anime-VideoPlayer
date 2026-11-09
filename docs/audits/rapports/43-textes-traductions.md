# Rapport de l'audit 43. Textes et traductions

Branche : `claude/project-thread-soifke`. Commit audité : `2f98b1e`. Date : `2026-11-09`.

## Verdict

Le mécanisme est sain (les 946 entrées ont les mêmes trous de format dans les deux langues, aucune clé utilisée ne
manque), mais ses garde-fous sont trop étroits : 192 entrées mortes, des textes français qui passent dans
l'interface anglaise sans que le test le voie, des nombres au mauvais séparateur dans les deux langues, et une
même notion (le test de la machine) sous cinq noms.

## Cadrage retenu

Pas de questions posées à Tatoub (consigne du coordinateur) : tout a été audité au maximum. L'anglais et le
français comptent autant ; l'anglais est la langue par défaut (`i18n.h:11`), donc tout texte français en dur est
vu par défaut. La règle de Tatoub sur la qualité a été appliquée aux textes : un texte qui cache le coût en
qualité d'un choix rapide est un défaut. Le VRR est hors sujet.

Méthode : un script lit toutes les chaînes littérales de `src/` (littéraux adjacents joints, commentaires
retirés), les 946 entrées des tables, et compare. Un petit programme lié à `i18n.cpp` et aux tables
(`g++ -std=c++23`, sans autre dépendance) a servi de preuve à l'exécution. Le projet complet n'a pas été construit
(pas de dépendances dans ce conteneur) : les tests proposés sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A43-01 | Majeur | `tests/unit/app/i18n_test.cpp:83`, `:106-156`, `:157`, `:51-53` | Le test i18n laisse passer ce qu'il doit arrêter. (a) Son motif `tr(?:_format)?\(\s*"` ne voit pas `ui_tr(("…"))` : 6 appels de `chain.cpp:105`, `:119`, `:135` (×2), `:176-177` ne sont pas vérifiés. (b) Les clés passées par variable (`tr_view(kPresets[…].label)`, `tr(choice.label)`, `tr(mode.label)`, `tr(page.name)`, `label(PrecisionTier)`, `track_text("audio")`) ne sont jamais vérifiées : « Performance » (`precision.h:29`), « Image » (`auto_config.cpp:36`) et « audio » (`pipeline.cpp:365`) n'ont pas d'entrée aujourd'hui. (c) Le contrôle des phrases françaises exclut 50 fichiers, dont des fichiers qui affichent (`ui.cpp`, `player.cpp`, `player_run.cpp`, `home_screen.cpp`, `chain.cpp`, `chain_hdr.cpp`, `session.cpp`), et ne voit qu'un littéral accentué : « chaleur », « actif », « Go », « sortie SDR » passent. (d) `placeholders` ne compte que les `{` : un `{:.1f}` contre `{}` passerait (aujourd'hui aucun écart, vérifié par script). (e) Aucun test ne cherche les entrées mortes (A43-02). | Tous les textes français en dur trouvés dans cet audit et dans 11, 24, 25, 41 sont passés à travers ce test : `chain_hdr.cpp:367` (GPU-14), `placebo_screen.cpp:65`, `home_screen.cpp:178` et `player_run.cpp:290` (« EDID illisible : OLED inconnu », montré dans Vidéo > Affichage par `settings_panel.cpp:145`), `gpu_telemetry.cpp:58`, `:73-75`, `first_run.cpp:203-205`. | Motif `\b(?:ui_)?tr(?:_format\|_view)?\(\(*\s*"` ; un test qui parcourt les tables de clés (`kPresets`, `kPriorities`, `kHeights`, `kLanguages`, `kPages`, `kModes`, `label(PrecisionTier)`) et exige une entrée ; réduire la liste `developer` aux fichiers qui n'affichent rien et marquer `i18n-ok` ligne par ligne ailleurs ; comparer la suite des `{…}` et non leur nombre ; faire passer tout texte montré par `Translated` (`i18n.h:38-53`), y compris `set_black_status`, `set_output`, `display_text`. | Lu, et script : 809 appels à littéral, 6 hors du motif du test. |
| A43-02 | Mineur | `src/app/play/i18n_fr_sound.cpp` (96), `i18n_fr_settings.cpp` (54), `i18n_fr_wizard.cpp` (25), `i18n_fr_system.cpp` (6), `i18n_fr.cpp` (6), autres (5) : liste en annexe | **UX-21, renvoi 38.** 186 entrées sur 946 n'ont plus aucun texte anglais dans `src/`, et 6 de plus ne servent qu'à `bench_page` (`settings_system.cpp:369-491`), sans appelant : 192 entrées mortes (20 %). Elles viennent d'anciennes fonctions : son « Anime » et profils audio (96), grade adaptatif et looks nommés (54), étapes HDR et audio de l'assistant (25). | Un traducteur relit 192 textes que personne ne voit ; une entrée morte garde un texte faux qu'un retour de code ressusciterait (« Finish », A43-05). Les fiches de chantier qui comptent les entrées sont faussées. | Retirer les 192 entrées, retirer `bench_page` et `BenchNote` (A23-15) ; ajouter le test (e) d'A43-01 : toute entrée doit avoir son littéral hors des tables. | Script (méthode ci-dessus) : 186 orphelines, 6 dans du code mort. |
| A43-03 | Mineur | `src/app/play/i18n.h:30-33`, `src/app/play/i18n.cpp:30-32` | **UX-10.** `tr_format` ne localise pas les nombres et `decimal_point()` ne sert qu'à deux endroits (`chain_stats.cpp:81-82`, `options.cpp:585`). Résultat dans les deux sens : 11 entrées françaises à `{:.Nf}` affichent un point en français ; huit fonctions remplacent le point par une virgule sans regarder la langue (`frame_stats.cpp:93`, `:99`, `session.cpp:732`, `chain_stats.cpp:105`, `:115`, `:118`, `settings_chain_tabs.cpp:275`, `gpu_telemetry.cpp:75`, plus `span_derive.cpp:93`, `:99`). | Interface anglaise (le défaut) : les statistiques montrent « 16,7 ms », « 23,976 fps », « 7,8 / 12,0 Go ». Interface française : l'infobulle d'une ligne du bench montre « 2.5 % d'images perdues » (`settings_bench_results.cpp:161`), le vrai noir « 0 sous 0.0010 nit », l'écran « noir 0.00050 nit » (`display_edid.cpp:274`). | Un seul chemin : `tr_format` passe ses arguments flottants par une fonction `number(double, precision)` qui met `decimal_point()`, ou formate avec `std::locale` choisi par langue et la spécification `L` ; retirer les `replace('.', ',')`. Test : en français, `tr_format("{:.1f} % lost", 2.5)` vaut « 2,5 % d'images perdues » ; en anglais, `format_ms(16700)` vaut « 16.7 ms ». | Démontré : programme lié à `i18n.cpp` et aux tables, langue française : `tr_format("{:.1f} % lost", 2.5)` rend « 2.5 % d'images perdues ». |
| A43-04 | Mineur | `src/app/play/i18n_fr_wizard.cpp:11`, `:15`, `:37`, `i18n_fr_settings.cpp:30`, `:15`, `i18n_fr_system.cpp:112`, `i18n_fr_settings.cpp:254`, `:42-43` | Un seul nom par notion n'est pas tenu. Le test de la machine qui règle Auto s'appelle « Machine calibration », « Calibrating this machine », « Optimise now », « optimization », « Optimize for this machine » (code mort) et « bench » (« Run the bench », « Bench results ») ; l'anglais mêle l'orthographe britannique (« Optimise », 4 clés) et américaine (« Optimize », « optimization », « Color », 6 clés) ; « calibration » désigne aussi les images de préparation du FP8 (`i18n_fr_visible.cpp:56`, `:73`) ; « Setup wizard » et « assistant » coexistent en anglais ; « Downscale » et « Upscale » restent en anglais dans l'interface française (en-têtes du tableau du bench) alors que « Réduction » et « Agrandissement » servent partout ailleurs. | Une nouvelle carte : la notification dit « optimise for this PC again (Video > Post-processing) » (`player.cpp:487`). Sur cette page, aucun bouton « Optimise » : le bouton s'appelle « Run the bench » et n'apparaît qu'en mode avancé (`settings_panel.cpp:485-491`). L'utilisateur ne trouve pas ce qu'on lui demande. | Choisir un nom (proposition : « Optimize for this machine » / « Optimiser pour cette machine », orthographe américaine partout) et l'employer dans l'assistant, la notification, le bouton et ses textes ; dire « mode avancé » dans la notification ou montrer le bouton hors du mode avancé ; garder « calibration » pour le FP8 seul ; traduire « Downscale »/« Upscale ». | Lu ; liste des clés par script. |
| A43-05 | Mineur | `src/app/play/settings_panel.cpp:283`, `src/app/play/i18n_fr_settings.cpp:198` | **Renvoi 25.** L'infobulle de « Run the wizard again » dit « Nothing changes until "Finish". » ; l'assistant n'a pas de bouton « Finish » : son dernier bouton est « Resume playback » quand on le relance, « Start » sinon (`first_run.cpp:540`, `:567`). La clé « Finish » (`i18n_fr_wizard.cpp:83`) est morte. | Général > À propos, survol de « Relancer l'assistant » : « Rien ne change avant « Terminer » ». À la dernière page, il n'y a que « Reprendre la lecture ». | « Nothing changes until "Resume playback". » (le libellé tiré de la même clé que le bouton, par `tr_format` comme `first_run.cpp:537-540`). | Lu. |
| A43-06 | Mineur | `src/app/play/i18n_fr_image.cpp:88`, `i18n_fr_settings.cpp:32` | Le coût en qualité du FP8 n'est jamais dit, et un même mot nomme deux choses. L'infobulle des modèles SPAN dit « FP8: prepared once on this PC, then faster » sans dire que le calcul est moins précis ; le bench écrit « SPAN: Balanced Sharp in FP8 » : « Balanced » est ici le modèle, alors que le cran de précision « Balanced » veut dire FP16 et que le FP8 s'appelle « Performance » (`precision.h:26-36`, `settings_chain_tabs.cpp:33-36`). | Un utilisateur qui veut la meilleure image lit « SPAN : Balanced Sharp en FP8 » et croit avoir le cran Équilibre ; il ne sait pas que l'Auto a choisi une précision réduite. Contraire à la règle de Tatoub : une baisse de qualité pour la vitesse doit se voir. | Dire le prix : « FP8: prepared once on this PC, then faster, with slightly less precision (checked against FP16) » ; nommer le cran par sa précision partout où un nom de modèle le côtoie (« SPAN : Balanced Sharp, précision FP8 »). Le choix lui-même relève de 02 et 49. | Lu. À confirmer : la mesure de la perte (dB) du FP8 que l'Auto retient. |
| A43-07 | Mineur | `src/app/play/french_text.h:12-30`, `src/app/play/i18n_fr_settings.cpp:54`, `i18n_fr_messages.cpp:147` | Typographie appliquée à moitié. `french_spaces` (espaces insécables avant « : ; ? ! % ») ne sert qu'aux notes et infobulles du panneau (`panel_style.h:33`, `:101`) ; l'assistant a sa propre `note` sans elle (`first_run.cpp:29-31`), comme les notifications et les boutons. Elle tourne aussi en anglais. À l'inverse, des clés anglaises portent la typographie française : « " ; playback resumes at the same spot" », joint par « " ; " » en dur (`settings.cpp:493`), et « registry: HKCU\{} : {} ». | Assistant en français, fenêtre étroite : « Calibration de la machine » puis, en début de ligne, « : avec « Optimiser… ». En anglais, le bandeau « To apply: SPAN reloads … ; playback resumes … » montre un point-virgule détaché. | Appliquer `french_spaces` dans `tr()` côté français (les tables sont des littéraux : une table précalculée au premier appel suffit) et nulle part en anglais ; corriger les deux clés anglaises et le séparateur (`tr("; ")`). | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| UX-09 | Toujours présent | `first_run.cpp:203-205` (« actif », « inactif », « Go », « Mo ») : A25-12. Même défaut non relevé dans l'OSD : `gpu_telemetry.cpp:73-76` écrit « Go » et une virgule en anglais (renvoi 11). |
| UX-10 | Toujours présent | A43-03 (cause dans `i18n.h`), déjà vu par 13 et 23. |
| UX-20 | Toujours présent | `chain.cpp:113` change la langue en direct, mais l'en-tête (`:131`, seulement si le facteur, la réduction ou le HDR changent), les pistes (`:81`, `pipeline.cpp:361-367`), la ligne de sortie (`:152`, seulement si le grade change), `display_text` (`player_run.cpp:290`) et l'état du vrai noir gardent l'ancienne langue jusqu'à la vidéo suivante. Code en zone 08 (déjà confirmé par 08). |
| UX-21 | Toujours présent | A43-02 : 186 orphelines + 6 dans `bench_page`, sans appelant. |
| GPU-14 | Toujours présent | `chain_hdr.cpp:367` : « Vrai noir : mesure impossible (voir le journal). » en dur, montré dans Couleurs par `settings_panel.cpp:150`. Zone 41. |
| Renvoi 25 (`settings_panel.cpp:283`) | Toujours présent | A43-05. |
| Renvoi 39 (`gpu_telemetry.cpp:58`) | Toujours présent, doublon d'A11-10 | « chaleur » sans `tr()` ; à corriger avec la ligne VRAM (UX-09 ci-dessus). |
| Renvoi 38 (`i18n_fr_*.cpp`) | Toujours présent | A43-02 : 192, pas 180. |
| Renvoi 41 (`placebo_screen.cpp:64-66`) | Toujours présent | `output_description` rend « sortie HDR10 (PQ BT.2020) » ou « sortie SDR » sans `tr()` ; `output_text` (`chain_hdr.cpp:225-235`) y ajoute « · grade » traduit. Interface anglaise, statistiques : « sortie SDR · grade ». Le test ne le voit pas (`placebo_screen.cpp` est dans la liste `developer`, A43-01). |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 11 | `src/app/play/gpu_telemetry.cpp:73-76` | « Go » et virgule décimale en dur dans la ligne VRAM de l'OSD anglais (UX-09), à côté d'A11-10. |
| 24 | `src/app/play/home_screen.cpp:178`, `src/app/play/player_run.cpp:290` | « EDID illisible : OLED inconnu » en dur, montré dans Vidéo > Affichage (`settings_panel.cpp:145`) en anglais. |
| 41 | `src/app/play/placebo_screen.cpp:64-66` | Ligne de sortie en français en dur (voir ci-dessus). |
| 41 | `src/app/play/chain_hdr.cpp:367` | GPU-14 toujours présent. |
| 25 | `src/app/play/first_run.cpp:29-31` | La `note` de l'assistant n'applique pas `french_spaces` (A43-07). |
| 25 | `src/app/play/first_run.cpp:445` | « Downloaded, installed at next launch. » ; le panneau dit « Downloaded, verified, installed at next launch. » (`settings_system.cpp:118`) : deux textes pour la même option. |
| 08 | `src/app/play/chain.cpp:113-131` | UX-20 : rien ne reformate en-tête, pistes, sortie, écran et vrai noir quand la langue change en direct. |
| 22 | `src/app/play/settings.cpp:130-138` | `label(MotionEngine)` nomme encore « GMFSS Fortuna » et « GMFSS Fortuna Union », retirés des sources (ADR 0012). |
| 23 | `src/app/play/settings.cpp:493` | Séparateur « " ; " » en dur dans le bandeau anglais « To apply » (A43-07). |
| 07 | `src/app/play/player.cpp:486-487`, `src/app/play/settings_panel.cpp:485-491` | La notification « New graphics card » renvoie à un bouton qui n'existe qu'en mode avancé, sous un autre nom (A43-04). |

## Points positifs

- Les 946 entrées ont exactement la même suite de `{…}` (type et précision compris) en anglais et en français :
  aucun `std::format_error` possible à l'exécution (script).
- Toutes les clés littérales utilisées ont leur entrée (809 appels), y compris les 6 que le test ne voit pas.
- Les tables sont des littéraux `constexpr` : `tr()` rend un pointeur stable, sans allocation ; la langue est un
  atomique, lu sans verrou par les threads qui composent des messages.
- `Translated` (`i18n.h:38-53`) empêche par le type un détail de refus ou une phrase en dur d'atteindre les
  messages brefs et l'écran de préparation ; l'étendre (A43-01) fermerait les fuites restantes.
- Le français est soigné : espaces avant « : ; ? ! », guillemets « », aucune entrée laissée en anglais sauf noms
  propres et mots identiques (et « Downscale »/« Upscale », A43-04).
- L'infobulle « Restore lines » dit honnêtement ce que coûte la réduction (« Off: such a source keeps all its
  detail ») : le modèle à suivre pour le FP8 (A43-06).

## Questions ouvertes

- Quel nom unique pour le test de la machine (A43-04) ? Proposition : « Optimiser pour cette machine ».
- Orthographe anglaise : américaine partout (proposition), ou britannique ?

## Annexe : les 192 entrées mortes (numéros de ligne de la clé)

**`src/app/play/i18n_fr.cpp`** (6) : 49, 76, 77, 78, 85, 86

**`src/app/play/i18n_fr_image.cpp`** (1) : 20

**`src/app/play/i18n_fr_messages.cpp`** (2) : 26, 62

**`src/app/play/i18n_fr_settings.cpp`** (54) : 29, 50, 51, 55, 57, 62, 74, 77, 81, 83, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 107, 108, 109, 110, 111, 112, 119, 120, 121, 122, 123, 130, 136, 180, 185, 186, 187, 191, 196, 210, 211, 212, 215, 216, 218, 229, 231, 233, 237, 285, 286, 288, 289, 309

**`src/app/play/i18n_fr_sound.cpp`** (96) : 10, 14, 15, 16, 19, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 46, 49, 50, 51, 56, 57, 58, 59, 62, 63, 64, 66, 67, 68, 69, 70, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84, 86, 88, 90, 92, 94, 95, 100, 101, 102, 103, 104, 105, 106, 107, 108, 114, 116, 120, 121, 123, 124, 125, 126, 127, 128, 129, 131, 134, 135, 138, 139, 140, 145, 147, 149, 150, 151, 152, 153, 154, 158, 162, 164, 166, 169, 172, 174, 176, 177

**`src/app/play/i18n_fr_stats.cpp`** (1) : 35

**`src/app/play/i18n_fr_system.cpp`** (6) : 45 (code mort), 111 (code mort), 112 (code mort), 116 (code mort), 131 (code mort), 146 (code mort)

**`src/app/play/i18n_fr_visible.cpp`** (1) : 76

**`src/app/play/i18n_fr_wizard.cpp`** (25) : 44, 48, 49, 50, 51, 57, 62, 64, 66, 67, 83, 84, 86, 88, 99, 101, 102, 107, 108, 110, 125, 127, 129, 130, 136
