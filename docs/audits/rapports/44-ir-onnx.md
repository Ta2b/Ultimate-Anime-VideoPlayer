# Rapport de l'audit 44. Représentation intermédiaire et import ONNX

Branche : `claude/project-thread-83suyt` (la session ne peut pousser que sur sa branche désignée). Commit audité :
`135f06c`. Date : `2026-11-09`.

## Verdict

L'import ONNX et les passes de l'IR sont justes et ne coûtent rien de mesurable en qualité au-dessus du F16 (écart
de ±0,01 dB sur des images réelles avec ou sans passes), mais le chemin Vulkan laisse en F16 la chaîne flot → grille
de DistilDRBA, ce qui coûte jusqu'à 2 dB sur une scène, et l'IR n'a aucune fusion élément par élément, ce qui rend
DistilDRBA lié au nombre de dispatchs (230 contre 134 pour RIFE 4.22 lite) alors qu'il déplace 4,7 fois moins
d'octets.

## Cadrage retenu

Pas de questions à Tatoub, sur consigne du coordinateur. Cadrage appliqué :

- priorité réduite : l'IR ne sert qu'au moteur Vulkan (repli AMD, Intel, et NVIDIA sans TensorRT) ; TensorRT ne
  la traverse pas, Core ML n'en garde que `onnx_io_type` et `onnx_unshuffle` (hors cible) ;
- toute perte de qualité d'image faite pour la vitesse est un défaut sauf mesure qui la justifie, rapportée avec
  son gain (consigne de Tatoub : « entacher la qualité pour gagner des performances ne doit presque jamais se
  faire ») ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA pèsent autant ; question posée : qu'est-ce qui empêche DistilDRBA d'être
  nettement plus rapide que RIFE à qualité voisine ;
- VRR hors champ.

Méthode : lecture de tout `src/gpu/ir` et de `src/gpu/vulkan/net_build.cpp` ; construction de `libir.a`,
`ajp_onnx_info` et des tests IR hors CMake (13 972 assertions, 122 cas, tous verts) ; un pilote qui rejoue
`prepare_graph` (`optimize` puis `fold_float_casts`, sans `hoist_downscales`) sur les modèles livrés ; un
interpréteur à stockage F16 émulé (chaque tenseur F16 arrondi en RTE à l'écriture) pour mesurer la qualité ; trois
plans réels tirés de `20261103_181020.mp4` (1024 × 576, mpdecimate, notés f3, g1, g4.5). Le GPU n'est pas mesuré :
les temps restent à confirmer sur matériel.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A44-01 | Majeur (repli Vulkan) | `src/gpu/vulkan/net_build.cpp` (choix F16), `src/gpu/ir/pass_cast.cpp` | Sous Vulkan en F16, la chaîne qui construit la grille de `GridSample` de DistilDRBA (flot, mise à l'échelle, addition de la grille de base, normalisation) reste en F16. Une coordonnée de grille en F16 n'a que 11 bits de mantisse : sur 1024 px, le pas utile vaut environ 0,5 px près des bords. C'est la cause côté IR de A03-18 (le modèle exporté est F16 de bout en bout). | Paliers Qualité et Performance de DistilDRBA (graphe IR entier), et Équilibré quand la voie fusionnée est coupée ; mouvement ample (plan f3, panoramique). | Une passe IR qui promeut en F32 les 114 tenseurs de la chaîne de grille (2 canaux, coût négligeable face aux convolutions) ; `fold_float_casts` sait déjà retirer les Cast autour. | Mesuré contre la référence FP32 entière, interpréteur F16 : f3 u=0,625 52,15 → 54,22 dB (écart max 16,3 → 9,0/255) ; f3 u=0,25 52,01 → 54,02 dB (15,3 → 8,0/255) ; g1 et g4.5 : +0,01 à +0,04 dB, max 19,0 → 17,7/255 sur g1. Gain nul ou positif partout, coût en octets < 1 % du trafic. |
| A44-02 | Mineur, à confirmer sur GPU | `src/gpu/ir/pass_*.cpp` (aucune passe de fusion élément par élément) | L'IR ne fond que biais, résiduels, activations et portes dans les Conv. DistilDRBA après passes, à 576 × 1024 F16 avec unités matricielles : 230 dispatchs (Conv 63, Slice 38, Mul 34, Resize 26, Add 23, Concat 19, GridSample 10, DepthToSpace 9), dont 103 touchent ≤ 50 k éléments et 6 sont scalaires ; RIFE 4.22 lite : 134. Trafic par image : DistilDRBA 373 Mo, RIFE 4.22 lite 1 767 Mo, RIFE 4.22 2 301 Mo, RIFE 4.25 2 104 Mo. C'est la réponse côté IR à « pourquoi DistilDRBA n'est pas nettement plus rapide » : il est lié aux dispatchs et barrières, pas à la bande passante. | Tout palier DistilDRBA hors voie fusionnée. | Fondre la construction de grille (Mul/Add/Slice/Concat sur le flot) dans `GridSample`, les chaînes Slice → Concat en vues, et les Mul/Add scalaires dans l'épilogue de la Conv voisine. Mesurer le temps par image avant et après sur `linux-release`. | Lu et compté (pilote `drv`) ; temps à confirmer : `ajp --bench` Vulkan sur DistilDRBA Qualité avant et après. |
| A44-03 | Mineur | `src/gpu/ir/import.cpp:141` (`sort_nodes`) | Tri topologique quadratique : chaque tour parcourt tous les nœuds restants. Un ONNX hostile de 0,9 Mo (40 000 Relu en ordre inverse) prend 8,5 s à charger (1,76 s pour 20 000), sans que le jeton d'arrêt de la construction Vulkan soit consulté pendant `prepare_graph`. Les passes recalculent aussi `consumers()` à chaque appel. | Modèle utilisateur malveillant ou mal ordonné. | Kahn avec compteur d'entrées par nœud (linéaire) ; consulter le jeton d'arrêt entre passes. | Mesuré (`genrev.py` + `onnx_info`). |
| A44-04 | Mineur | `src/gpu/ir/onnx_reader.cpp` (`int64_data` empaqueté, `string_data`) | Amplification mémoire bornée seulement par la taille du fichier (2 Gio) : un int64 empaqueté sur 1 octet en coûte 8, une chaîne vide de 2 octets devient une `std::string` de 32 octets (×16). | ONNX hostile proche de 2 Gio : `bad_alloc` au lieu d'un refus propre. | Borner le nombre d'éléments de ces champs (par exemple à la taille du fichier / 8) et refuser au-delà. | Lu. |
| A44-05 | Mineur (dette) | `src/gpu/ir/import.cpp:238`, `:242`, `:249` | `path.string()` dans les messages d'erreur de `load_onnx_file`, contre la règle `core::to_utf8` ; sous Windows un chemin non ASCII peut lever à l'endroit même où l'on signale une erreur. | Modèle dans un dossier au nom accentué, fichier illisible. | `core::to_utf8(path)`. | Lu. |
| A44-06 | Mineur, sans perte mesurée | `src/gpu/ir/pass_fuse.cpp:103` (`fold_into_conv`) | Le facteur par canal (beta des ResConv, jusqu'à 3e-7) est multiplié dans des poids F16 puis arrondi : DistilDRBA block0 convblock.7 a 4 618 poids fondus à zéro, block1 convblock.6 en a 4 945, et 25 à 60 % de sous-normaux selon le canal. | Tout modèle F16 à ResConv (RIFE, DistilDRBA). | Garder tel quel tant que la mesure ne bouge pas ; sinon appliquer le facteur dans un épilogue FP32 au lieu des poids. | Mesuré : écart max 0,0489 (F16 sans passes) contre 0,0499 (F16 avec toutes les passes) sur entrée aléatoire ; sur images réelles ±0,01 dB. Le test ajouté hors dépôt (c = 3e-6) montre 2,7 % d'erreur relative sur le canal seul. |
| A44-07 | Mineur | `src/gpu/ir/onnx_reader.cpp:208`, `:236` | Les valeurs INT32 lues dans `int32_data` (varints int64) ne sont pas vérifiées dans [INT32_MIN, INT32_MAX]. | ONNX malformé : constante entière tronquée silencieusement plus loin. | Refuser hors plage. | Lu. |
| A44-08 | Dette | `src/gpu/ir/onnx_io_type.cpp:648` (`with_float32`) | Code mort : seul `onnx_io_type_test.cpp` l'appelle depuis le retrait de VitisAI ; `copy_tensor_f32` n'y vérifie pas le débordement du compte. | Aucun en production. | Retirer la fonction et ses tests. | Lu (grep). |
| A44-09 | Dette | `src/gpu/ir/describe.cpp:117` (`describe_stats`) | N'affiche pas `stats.quants` (FP8 replié). | Diagnostic incomplet dans les journaux. | Ajouter le champ. | Lu. |
| A44-10 | Mineur (outil) | `src/gpu/ir/onnx_info_main.cpp` | `ajp_onnx_info` ne rejoue pas les passes propres à Vulkan (`transposes_as_convs` avec phases, `fold_float_casts`, `hoist_downscales`) : le graphe qu'il décrit n'est pas celui qui tourne. | Diagnostic d'un modèle Vulkan. | Option `--vulkan` qui appelle le même `prepare_graph`. | Lu. |
| A44-11 | Tests | `tests/unit/gpu_ir/` | Aucun test sur les modèles livrés en CI ; la branche de `fold_grid_transposes` qui traverse des Cast (`pass_pixel.cpp:90`) n'est pas testée ; le fuzz couvre `load_onnx`, `infer_shapes`, `optimize`, `plan_memory` mais pas `fold_float_casts` ni `transposes_as_convs`. | Régression silencieuse d'une passe sur un vrai modèle. | Test d'équivalence interpréteur avant/après passes sur RIFE et DistilDRBA (seuil en dB sur une image fixe) ; étendre le fuzz. | Lu ; la branche à travers Cast, testée ici, est juste. |
| A44-12 | Mineur (perf) | `src/gpu/ir/pass_transpose.cpp` (mode phases) | En mode phases, le DepthToSpace qui suit la ConvTranspose réécrite reste en place : deux passes DepthToSpace là où une suffit. | Vulkan avec unités matricielles. | Fondre les deux réarrangements en un. | Lu ; calcul des phases vérifié juste. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| SEC-21 | Corrigé pour l'essentiel | `load_onnx` est fuzzé (`tests/unit/gpu_ir`), lectures bornées dans `protobuf.cpp` ; la graine de 16 octets signalée en A28-13 est toujours la seule. |
| TRT-07 (partie Vulkan/IR) | Toujours présent, sans perte | Aucune conversion FP32 → F16 dans l'IR : un modèle FP32 tourne en FP32 sous Vulkan, et `!request.fp16` promeut tout en F32 (`net_build.cpp`). |
| A03-18 | Confirmé et mesuré côté IR | Voir A44-01 : chaîne de grille en F16, jusqu'à 2 dB de perte sur un panoramique. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 03 / 46 | voie fusionnée DistilDRBA | A44-01 mesure A03-18 sur le chemin IR ; la voie fusionnée doit être vérifiée de même (grille en F32). |
| 49 | qualité DistilDRBA F16 | DistilDRBA F16 contre FP32, émulation CPU : 51 à 52 dB, écart max 13 à 19/255 sur trois plans réels (pour A49-14). |
| 45 | shaders Vulkan | À vérifier : arrondi f32 → f16 du GPU (RTE contre RTZ), mise à zéro des sous-normaux F16 (A44-06 en dépend), accumulation des coopmat en F16 ou F32. |

## Points positifs

- Lecture protobuf bornée partout (varints, longueurs, profondeur), refus typés et messages clairs ; le nombre
  d'éléments d'un tenseur est borné par son tampon.
- Les passes conservent la qualité : sur trois plans réels, « sans passes » et « passes livrées » sont égaux à
  ±0,01 dB.
- Les réécritures délicates sont justes : ConvTranspose en Conv par phases, repli des transpositions de grille
  (y compris à travers les Cast), `fold_float_casts` avec vérification des types dans `net_alias`.
- Un interpréteur FP32 de référence et des tests de propriétés existent : ils ont permis toutes les mesures de ce
  rapport sans GPU.
- `-ffp-contract=off` tient la référence reproductible.

## Questions ouvertes

- Gain réel en temps de A44-02 et coût de A44-01 sur GPU (à mesurer sur `linux-release`, AMD et Intel).
- Comportement F16 du matériel (renvoi 45) : si le GPU met les sous-normaux à zéro, A44-06 peut devenir mesurable.
