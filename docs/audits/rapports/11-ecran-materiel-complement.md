# Complément de l'audit 11. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-f2fod4`. Commit audité : `e5a7e72`. Date : `2026-11-09`.

## Verdict

Des deux renvois, celui de l'audit 34 est un doublon d'A34-05, qui décrit déjà le maximum de toutes les sorties ;
celui de l'audit 43 est pour moitié un doublon d'A43-03 (la virgule) et donne pour l'autre moitié un défaut
nouveau, **A11-14** (l'unité « Go » en dur). Aucune baisse de qualité d'image faite pour la vitesse n'est en
jeu dans ces renvois.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les deux renvois de `audits-docs/complements/11.md`, sans
question à Tatoub, sans toucher au code. Les ID suivent le rapport 11, qui s'arrête à A11-13. Le conteneur n'a
ni Windows ni GPU : tout est lu, rien n'est lancé.

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 34 | `src/app/play/system_peak.cpp:150-151` (aujourd'hui `:144-147`) | Doublon d'A34-05 | Voir « Doublons ». Les lignes citées ont glissé : le filtre sur le moniteur est à `:144`, le maximum à `:145-147`. |
| 43 | `src/app/play/gpu_telemetry.cpp:73-76` | Virgule : doublon d'A43-03. « Go » : confirmé, **A11-14** | A43-03 cite déjà `gpu_telemetry.cpp:75` parmi les huit `replace('.', ',')` faits sans regarder la langue. L'unité n'a d'ID nulle part : A25-12 ne couvre que `first_run.cpp`, et le rapport 43 la note dans UX-09 sans ID. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A11-14 | Mineur | `src/app/play/gpu_telemetry.cpp:73-74`, `tests/unit/app/gpu_telemetry_test.cpp:95` | **UX-09.** L'unité de la ligne VRAM de l'OSD de télémétrie est écrite en dur, « Go », sans `i18n::tr`. Le test d'i18n ne la voit pas (littéral sans `tr()`, ni accent : A43-01 c). Le seul test de la ligne (`gpu_telemetry_test.cpp:95`) tourne en français et fige « 16,0 Go ». La valeur est en Gio (Mio / 1024), ce qui ne change rien à l'affichage. | Interface anglaise, OSD de télémétrie ouvert sur une 4070 Ti : « … · VRAM 7,2 / 12,0 Go ». Avec la virgule d'A43-03, toute la ligne VRAM est en français. | `std::format("{} / {} {}", …, i18n::tr("GB"))` avec l'entrée « Go » dans `i18n_fr_*.cpp`, à corriger avec « chaleur » (A11-10) et la virgule (A43-03, par `decimal_point()`). Ajouter un cas en anglais au test : « VRAM 7.2 / 16.0 GB ». | Lu. |

## Doublons

**Renvoi 34 → A34-05.** A34-05 dit déjà, mot pour mot : « Quand `MonitorFromWindow` échoue (`monitor ==
nullptr`), il garde la sortie la plus lumineuse de toutes, pas celle de la fenêtre », avec la même correction
(ne rien rendre sans moniteur). Le comportement est même voulu par l'en-tête (`system_peak.h:32` : « Sans
lui, la plus lumineuse des sorties en HDR »). Deux précisions pour qui corrigera A34-05, sans nouvel ID :

- Le cas est rare. `monitor` n'est nul que si `window_monitor` échoue (`host_facts.cpp:160-168`) : pas de
  HWND, ou `GetMonitorInfoW` en échec. `MonitorFromWindow(…, MONITOR_DEFAULTTONEAREST)` rend toujours un
  écran, y compris pour une fenêtre à cheval sur deux écrans : le scénario « passe entre deux écrans »
  d'A34-05 ne mène pas à `nullptr`. Les deux appelants (`home_screen.cpp:173`, `player_run.cpp:133` et `:551`)
  passent une fenêtre déjà créée. La gravité Mineur d'A34-05 est juste.
- Le même repli touche les autres requêtes du même relevé. Avec `monitor` nul, `device` l'est aussi
  (`host_facts.cpp:243`), et `path_on` accepte alors tout chemin (`host_facts.cpp:171-172`) :
  `windows_sdr_white_nits(nullptr)` et `windows_hdr_active(nullptr)` lisent le premier chemin de
  `QueryDisplayConfig`, pas l'écran de la fenêtre. La correction d'A34-05 doit couvrir les trois : sans
  moniteur, ne rien rendre et laisser la voie EDID et le repli de 600 nits prendre la suite. Le paramètre par
  défaut `monitor = nullptr` de `system_peak_nits` (`system_peak.h:34`) peut alors devenir obligatoire sous
  Windows.

**Renvoi 43, virgule → A43-03.** `std::ranges::replace(vram, '.', ',')` (`gpu_telemetry.cpp:75`) met la
virgule quelle que soit la langue ; A43-03 le cite déjà, avec la correction commune (`decimal_point()` ou
`tr_format` qui localise les nombres).

## Baisses de qualité pour la vitesse

Aucune : les deux renvois portent sur le pic annoncé à la chaîne et sur un texte de l'OSD.

## Renvois

Aucun.
