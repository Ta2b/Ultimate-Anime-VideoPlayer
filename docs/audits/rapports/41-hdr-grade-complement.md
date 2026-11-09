# Complément de l'audit 41. Renvois reçus après la fusion du rapport

Branche : `claude/project-thread-5cymlr`. Commit audité : `1559e8b`. Date : `2026-11-09`.

## Verdict

Sur les trois renvois, un donne un défaut nouveau (A41-12, mineur), un est le défaut connu GPU-14, toujours présent,
et le troisième est un doublon d'A40-06 : le grade SDR n'explique pas l'image assombrie, il laisse ce vert pâle
inchangé. Aucune baisse de qualité faite pour la vitesse n'apparaît dans ces lignes.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les trois renvois de `audits-docs/complements/41.md`, sans question à
Tatoub et sans toucher au code. Les ID suivent le rapport 41, qui s'arrête à A41-11. Le conteneur n'a ni GPU ni
`~/ajp-deps` : tout est lu, et le calcul du grade sur le pixel du test est fait à la main sur les shaders.

## Conclusion par renvoi

| De | Où | Conclusion | Détail |
|---|---|---|---|
| 43 | `src/app/play/placebo_screen.cpp:64-66` | Confirmé : **A41-12** | La ligne n'est pas traduite. Le rapport 43 l'a relevée sans lui donner d'ID ; que le test ne la voie pas relève d'A43-01. |
| 43 | `src/app/play/chain_hdr.cpp:367` | Doublon de GPU-14, toujours présent | Voir « Défauts connus revérifiés ». Le texte en dur est le même défaut, et A43-01 couvre le test qui le laisse passer. |
| 40 | `tests/unit/app/e2e_play_test.cpp:241-243` | Doublon d'A40-06. Le grade SDR est mis hors de cause | Calcul ci-dessous : sur ce pixel, le grade SDR ne fait rien. Le facteur 0,62 vient d'ailleurs, et le commentaire du test l'attribue à tort au grade. |

### Le grade SDR sur le pixel du test e2e

Le centre du clip vaut RVB 155, 238, 151 (luma 214, d'après A40-06). La chaîne suivie sur une sortie SDR :

1. `decide_grade` (`src/render/grade_params.cpp:98-104`) donne `GradeMode::Sdr`. Mais `grade_for`
   (`src/app/play/placebo_grade.cpp:58-65`) prend `choice.sdr_anime` dès qu'il existe : c'est le grade PQ passé
   par `sdr_tonemap` (`src/app/play/chain_hdr.cpp:100-104`).
2. La première passe (`kSdrToPq`, `src/render/grade.cpp:74-89`) porte l'image en PQ, avec le blanc à `sdr_white`
   (203 nits). Le canal le plus fort, le vert (238/255), y vaut 0,859 × 203 = 174 nits avec le gamma 2,2 d'une
   cible sRGB (0,847 en 2,4).
3. La passe anime (`3-anime.hook:80-129`) le multiplie par `an_lift`. Le plan est uni, donc ni émetteur ni
   étincelle : `iso` et `spark` valent 0, et le gain est `an_lift`.
4. La passe finale (`anime-hdr-black.hook:232-241`) divise par `tm_white`, soit `sdr_white × an_lift`
   (`grade_params.cpp:171`). Le relèvement s'annule. Le maximum RVB vaut 0,859, sous le genou de 0,90 de
   `ajt_curve`, où la courbe est l'identité. Le vert ressort à 238.
5. Les passes 1 et 2 ne touchent que les ombres (`shadows`) et le peps (`an_punch`, nul à `anime_contrast` 100) :
   rien sur un aplat clair.

Le grade rend donc ce pixel à environ 214 de luma, pas à 125. C'est cohérent avec la mesure d'A40-06 : 132 avec le
grade, 134 sans. Le facteur 0,62 est en amont ou à côté du grade : étiquette de couleur de l'anneau, espace de la
swapchain de lavapipe, ou instantané (A10-18). Le commentaire du test (« Le grade SDR la rend vers 125 sur 255 »,
`e2e_play_test.cpp:236-238`) est faux et a caché le défaut. La correction d'A40-06 doit aussi le réécrire.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A41-12 | Mineur | `src/app/play/placebo_screen.cpp:64-66`, `src/app/play/chain_hdr.cpp:225-235`, `src/app/play/session.cpp:720` | `output_description` rend « sortie HDR10 (PQ BT.2020) » ou « sortie SDR » en dur, sans `tr()`. `output_text` y ajoute « · HDR », « · grade » ou « · grade off: HDR source », qui sont traduits. La ligne s'affiche dans les statistiques (`osd_lines`, « Statistiques » > sortie, ou toutes) et part au journal (`player_run.cpp:227`). Elle ne dit pas non plus si le grade est vraiment posé : « · grade » suit `s.grade`, même quand le hook est refusé ou que la pile manque (A41-04). | Interface en anglais, statistiques affichées : « sortie SDR · grade ». Interface en français, HDR coupé dans le système : « sortie SDR · étalonnage » s'affiche, et rien ne dit que c'est la méthode HDR anime ramenée au SDR. | Rendre `i18n::tr("HDR10 output (PQ BT.2020)")` et `i18n::tr("SDR output")`, avec leurs entrées dans `i18n_fr_*.cpp`. Pour le grade, ne dire « · grade » que si `graded(target_color)` est vrai, sinon « · grade off ». | Lu. |

## Doublons

| Renvoi | Défaut existant | Ce que le renvoi ajoute |
|---|---|---|
| 43 → 41, `chain_hdr.cpp:367` | GPU-14 | Rien de neuf sur le texte. Le cas est réel : `grade_ramp` refuse quand le grade n'est pas posé sur la cible (`placebo_grade.cpp:255-256`), que `set_looks` échoue (`:279`) ou que le rendu de la rampe échoue (`:340`). `black_dirty` passe alors à `false` (`chain_hdr.cpp:346`) et la mesure n'est plus retentée tant que rien ne bouge : le vrai noir reste coupé, ce que le message dit. |
| 40 → 41, `e2e_play_test.cpp:241-243` | A40-06 | Le grade SDR est mis hors de cause par le calcul : c'est l'identité sur ce pixel. Le commentaire du test est faux. |

## Baisses de qualité pour la vitesse

Aucune dans ces trois renvois.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-14 | Toujours présent | `chain_hdr.cpp:367` : « Vrai noir : mesure impossible (voir le journal). » est en dur, et s'affiche dans Couleurs (`settings_panel.cpp:150`). Correction : `i18n::tr("True black: cannot measure (see the log).")` avec son entrée française. Les quatre autres états de `refresh_black` sont déjà traduits. |
| A40-06 | Toujours présent, grade hors de cause | `e2e_play_test.cpp:236-243`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 40 | `tests/unit/app/e2e_play_test.cpp:236-238` | Le commentaire attribue la luma de 125 au grade SDR, alors que le grade est l'identité sur ce pixel (calcul ci-dessus). |

## Points positifs

- La route SDR de la méthode anime annule le relèvement des surfaces avant sa courbe (`tm_white = sdr_white ×
  an_lift`) : une image SDR claire ressort fidèle jusqu'à 90 % du blanc.
- Les quatre autres messages du vrai noir passent par `tr()`.

## Questions ouvertes

Aucune.
