# Audit 24. Interface : rendu, contrôles et messages

**Audit par zone. Priorité : haute.** 31 fichiers, 4 092 lignes ; tests rattachés : 4 fichiers, 574 lignes.

## Mission

Ce que l'utilisateur voit et touche hors du panneau de réglages : rendu ImGui par libplacebo, barre de contrôle, clavier et souris, accueil, messages à l'écran, erreurs, progression des constructions.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Coût du rendu : interface rejouée à chaque présentation, atlas des polices.
- Messages : un seul emplacement de 1,6 s (UX-04), messages répétés (UX-17).
- Clavier : AZERTY, raccourcis en dur, une faute qui remet tout par défaut (UX-11).
- Échap et focus (UX-08), touches de l'assistant (UX-23).
- Textes qui débordent (UX-15, UX-22).
- Erreurs : jamais de détail interne affiché.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (24 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (7 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (6 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (5 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (5 inclusions)
- [17. Sous-titres](17-sous-titres.md) (5 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (4 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (3 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (3 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (3 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (3 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (2 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (2 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [16. Audio](16-audio.md) (1 inclusion)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (15 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (13 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (8 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (6 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (4 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (4 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [28. CI, publication et signature](28-ci-publication.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [20. Socle commun (core)](20-socle-core.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| UX-04 | Majeur | Un seul emplacement de message, affiché 1,6 s. Les messages s'écrasent ou fusionnent en 30 à 40 mots illisibles. |
| QUA-16 | Mineur | Restes : statiques mutables, `strtoul` sans contrôle, 172 traductions orphelines, CHANGELOG sans 0.10.1 ni 0.10.2, `moving_area` sans effet. |
| UX-08 | Mineur | Échap ferme tout le panneau et jette les changements, même quand on veut fermer une liste. |
| UX-11 | Mineur | En AZERTY, « . » maintenu ne répète pas, et « ; » ou « - » ne sont jamais traduits. |
| UX-15 | Mineur | Le libellé de préparation passe sous « Cancel and watch » et peut sortir de la fenêtre à 1 280 px. |
| UX-17 | Mineur | Le message d'erreur se répète (« ne se lit pas : le fichier ne se lit pas »). La boîte fatale renvoie deux fois au journal. |
| UX-22 | Mineur | Les pastilles débordent (« Recommenc… » coupé sur `menu.png`). |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/build_progress.cpp` | 165 |
| `src/app/play/build_progress.h` | 61 |
| `src/app/play/controls.cpp` | 184 |
| `src/app/play/controls.h` | 141 |
| `src/app/play/controls_draw.cpp` | 398 |
| `src/app/play/controls_draw.h` | 48 |
| `src/app/play/controls_icons.cpp` | 159 |
| `src/app/play/controls_icons.h` | 36 |
| `src/app/play/home_screen.cpp` | 350 |
| `src/app/play/home_screen.h` | 64 |
| `src/app/play/home_view.cpp` | 60 |
| `src/app/play/home_view.h` | 20 |
| `src/app/play/input.cpp` | 138 |
| `src/app/play/input.h` | 101 |
| `src/app/play/osd_text.cpp` | 95 |
| `src/app/play/osd_text.h` | 53 |
| `src/app/play/panel_state.h` | 14 |
| `src/app/play/panel_style.h` | 132 |
| `src/app/play/placebo_imgui.cpp` | 213 |
| `src/app/play/placebo_imgui.h` | 58 |
| `src/app/play/placebo_overlays.cpp` | 114 |
| `src/app/play/scripted_input.cpp` | 124 |
| `src/app/play/scripted_input.h` | 30 |
| `src/app/play/text_fit.h` | 27 |
| `src/app/play/ui.cpp` | 623 |
| `src/app/play/ui.h` | 145 |
| `src/app/play/user_keys.cpp` | 43 |
| `src/app/play/user_keys.h` | 19 |
| `src/app/play/visible_error.h` | 49 |
| `src/core/keys.cpp` | 265 |
| `src/core/keys.h` | 163 |

## Tests rattachés

- `tests/unit/app/build_progress_test.cpp`
- `tests/unit/app/controls_test.cpp`
- `tests/unit/app/osd_text_test.cpp`
- `tests/unit/core/keys_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
