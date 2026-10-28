# Mesures du mode Auto et des moteurs d'interpolation, RTX 4070 Ti (28 octobre 2026)

Pour régler le contrôleur à coût mesuré (audit, H4). Machine : RTX 4070 Ti, écran MAG 272U E16 (QD-OLED, HDR10, **165 Hz, VRR actif**),
plein écran, KDE Wayland ; bureau actif (Brave tient ~35 % du GPU). Binaire : branche `claude/project-thread-v15ttw` (PR #38), build CUDA
RelWithDebInfo sm_89, TensorRT 11.3. Extrait : *Re:Zero 4e saison, épisode 18*, à 16:00 (`--start=960 --seconds=60`), 1080p 23,976 i/s,
mouvement fort. Chaîne **réelle du PC** (celle de `play.toml`) : réduction Mitchell 1080p → 540p, SPAN Sharp1 Performance ×2, RTX HDR
(Maxine), grade ; son coupé. Un run par case, 60 s ; moteurs : RIFE 4.25 (variante mouvement), GMFSS base Performance (demi-flot),
GMFSS Union Performance, DistilDRBA v2 lite. `--rife k` (×1 = aucune interpolation), `--stats` en TSV, données isolées par `--data-dir`.

## 1. Multiplicateur fixe (tête be07da9)

Cases : paires en retard / instants sautés pour les rattraper (journal « RIFE : N paires en retard… ») · images manquées (FrameStats
`missing`) · retard de présentation p95 (ms) · intervalle entre images neuves p50 / p90 (ms, TSV).

| moteur | ×1 | ×2 | ×3 | ×4 | ×5 | ×6 |
|---|---|---|---|---|---|---|
| RIFE 4.25 | 0/0 · 6 · 0 · 41/49 | 1/1 · 15 · 0 · 20,7/29 | 3/5 · 16 · 0 · 13,7/22 | 1/3 · 29 · 2,7 · 10,3/18 | **13/13 · 104 · 9,0** · 7,8/13 | 205/271 · 843 · 17,1 · 7,8/9,1 |
| GMFSS base perf | 0/0 · 7 · 0 | 1/1 · 14 · 0,5 · 20,3/30 | **469/470 · 486 · 19** · 16,5/27 | 1131/1661 · 1629 · 23 | 931/2788 · 2751 · 43 | 1189/4281 · 4143 · 56 |
| GMFSS Union perf | 0/0 · 6 · 0 | 1/1 · 14 · 0,7 · 20,5/29 | **507/508 · 518 · 20,5** · 16,6/27 | 1119/1681 · 1662 · 24,8 | 961/2828 · 2792 · 43 | 1214/4308 · 4118 · 60 |
| DistilDRBA lite | 0/0 · 7 · 0 | 1/1 · 20 · 0 | 0/0 · 18 · 0 · 13,6/22 | 1/1 · 46 · 0 · 11,2/14 | 1/1 · 59 · 0 · 7,9/13 | 0/0 · **962** · 3,6 · 7,9/8,8 |

Ce que la chaîne tient en lecture directe : **RIFE ×4** (×5 limite : 13 paires en retard et 104 images manquées contre 29 à ×4), **DistilDRBA ×5**
(0,75 ms par passe contre 4,3 ms pour RIFE), **GMFSS base et Union ×2 seulement** (dès ×3 le tiers des paires arrive en retard).
À ×6 les images manquées viennent de la chaîne entière (SPAN + RTX HDR par image créée), pas de l'interpolation : DistilDRBA, pourtant à 0,75 ms
par passe, en perd 962.

### Temps GPU d'un appel (médiane / p95, ms)
| étape | coût |
|---|---|
| RIFE 4.25, 11×576×960 (une passe) | 4,2–4,5 / 5,0–5,2 |
| DistilDRBA lite, 7×576×960 | 0,73–0,77 / 0,77 |
| GMFSS base, fusion 3×320×512 | 10,1–10,3 / 10,7–11,8 |
| GMFSS Union, fusion 3×320×512 | 10,5–12,1 / 12,2–12,5 |
| GMFSS (paire) 6×640×1024 | 6,8–7,5 / 8,3–8,5 |
| GMFSS, splatting | 3,75–3,8 / 3,9–4,05 |
| SPAN 3×540×960 | 1,8 / 3,5 |
| entrée de RIFE | 0,4–0,8 / 1,3–1,6 |

Le journal ne donne pas de p90 par appel : médiane et p95 seulement.

## 2. Mode Auto, `--rife 6`, `interpolation.auto = true`

### Tête be07da9 (avant le recalage du contrôleur)
| moteur | changements de palier | comportement | fin : paires en retard / sautés / manquées / retard p95 |
|---|---|---|---|
| RIFE | 14 | monte ×2 → ×6 en 14 s, puis ×6 ↔ ×5 ↔ ×4 ; plafond appris 0 | 7 / 8 / 217 / 8,6 ms |
| GMFSS base | 9 | ×2 ↔ ×3 sans fin ; plafond appris 0 | 5 / 5 / 28 / 3,6 ms |
| GMFSS Union | 9 | idem | 5 / 5 / 28 / 3,8 ms |
| DistilDRBA | 5 | monte à ×6 en 14 s et y reste ; plafond appris 0 | 1 / 1 / 804 / 3,6 ms |

### Tête 0f90469 (contrôleur recalé : fenêtre de 64 paires, plafond appris après 2 échecs)
Palier tenu : secondes passées à chaque palier. Journal en fin de run : « auto : ×k (plafond N, plafond appris M) ».

| moteur | changements | secondes par palier | dernière ligne du journal | paires en retard / sautés | sautées / manquées | retard p95 |
|---|---|---|---|---|---|---|
| RIFE | 12 | ×2 4 · ×3 4 · ×4 15 · ×5 18 · ×6 21 | `auto : ×5 (plafond 6, plafond appris 0)` | 13 / 14 | 294 / **354** | 9,6 ms |
| GMFSS base | 5 | ×2 59 · ×3 3 | `auto : ×2 (plafond 6, plafond appris 2)` | 5 / 5 | 4 / 23 | 1,4 ms |
| GMFSS Union | 5 | ×2 59 · ×3 3 | `auto : ×2 (plafond 6, plafond appris 2)` | 4 / 4 | 5 / 26 | 1,5 ms |
| DistilDRBA | 5 | ×2 4 · ×3 3 · ×4 4 · ×5 3 · ×6 48 | `auto : ×6 (plafond 6, plafond appris 0)` | 1 / 1 | 675 / **707** | 3,8 ms |

Lecture :
- **GMFSS** : le plafond appris pose ×2 (« plafond appris 2 ») après deux échecs à ×3 : 5 changements au lieu de 9, le palier tenu 59 s sur 62.
  Résultat : 23–26 images manquées, retard p95 1,4–1,5 ms (contre 28 et 3,6 avant).
- **RIFE** : l'Auto reste à ×5 sans en faire un plafond appris. **×5 laisse trop d'images manquées : 354 sur 60 s** (contre 29 à ×4 en fixe, 104 à ×5 en
  fixe), retard p95 9,6 ms : à ×5 la chaîne ne tient pas sans à-coups, le contrôleur ne le voit que par les paires en retard (13).
- **DistilDRBA** : l'Auto monte à ×6 et y reste **alors que 707 images sont manquées** (retard p95 3,8 ms) : le contrôleur est aveugle à ces manques,
  parce que les paires ne sont pas en retard (1 seule) ; ce sont les images créées en aval (SPAN et RTX HDR, ×6) qui manquent à la présentation.

### Tête 00f00ee (le contrôleur écoute aussi les images manquées à l'écran : fenêtres de 4 s)
| moteur | changements | secondes par palier | dernière ligne du journal | paires en retard / sautés | sautées / manquées | retard p95 |
|---|---|---|---|---|---|---|
| RIFE | 13 | ×2 4 · ×3 3 · ×4 25 · ×5 24 · ×6 6 | `auto : ×4 (plafond 6, plafond appris 5)` | 7 / 9 | 104 / **163** | 0,5 ms |
| GMFSS base | 5 | ×2 60 · ×3 2 | `auto : ×2 (plafond 6, plafond appris 2)` | 6 / 6 | 3 / 23 | 0,0 ms |
| GMFSS Union | 5 | ×2 60 · ×3 2 | `auto : ×2 (plafond 6, plafond appris 2)` | 5 / 5 | 5 / 24 | 2,6 ms |
| DistilDRBA | 8 | ×2 4 · ×3 4 · ×4 3 · ×5 43 · ×6 8 | `auto : ×5 (plafond 6, plafond appris 5)` | 0 / 0 | 185 / **229** | 0,0 ms |

Attendu (RIFE ×4, DistilDRBA ×5, GMFSS ×2) : **atteint** pour les trois en fin de run. Reste : le plafond appris n'arrive qu'après ~35 s pour RIFE (le run
monte d'abord à ×6, redescend à ×5, retente ×6 puis ×5), et le palier tenu ×5 (RIFE 24 s, DistilDRBA 43 s) laisse encore 163 et 229 images manquées sur 60 s,
contre 29 pour ×4 en multiplicateur fixe : ×5 est tenu avec des à-coups. Images manquées par rapport à la tête 0f90469 : RIFE 354 → 163, DistilDRBA 707 → 229.

## 3. VRR
- **Plein écran, 26 lancements** : `VRR : capable · armé` à chaque fois (jamais « non capable » ni « coupé »).
- **Fenêtré** (1280×720) : **aucune ligne « VRR : … »** n'est écrite (elle l'est à l'ouverture du plein écran) ; seule la ligne d'écran
  « MAG 272U E16 · HDR10 … · VRR actif » apparaît, qui dit la capacité de l'écran, pas l'armement par la fenêtre : le journal ne dit pas si le VRR est armé en fenêtré.

## 4. Profil de la fusion GMFSS Union
`tools/trt/layer_report.py` sur l'engine de la fusion (`--onnx gmfss_fortuna_union_s05_fusion.onnx`) : **217 couches, 0 reformatage dont 0 hors des E/S,
0 couche FP32 hors îlots déclarés**. Au trtexec (720p de travail) : 12,1 ms, 9,0 ms de convolutions (voir `gmfss-distil-mesures-2026-10-28.md`).
