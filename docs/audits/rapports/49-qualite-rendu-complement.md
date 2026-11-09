# Complément de l'audit 49. Qualité du rendu

Branche : `claude/project-thread-06u5c3`. Commit vérifié : `10e9c7d`. Date : `2026-11-09`.

Renvois reçus après le rapport `49-qualite-rendu.md`, vérifiés dans le code. Les IDs suivent ceux du rapport
(dernier : A49-14).

## Verdict

Les deux renvois sont exacts sur les faits mais n'ouvrent pas de défaut sur leur point principal : le FP8 de SPAN
construit par défaut sous Linux est déjà A49-03 et A34-01, et la mesure de DistilDRBA en FP16 nourrit A49-14 et
A03-18. Deux défauts mineurs en sortent : une page de documentation qui annonce encore le FP8 dans les préréglages
(A49-15), et une perte du FP16 de DistilDRBA que la grille n'explique pas, sans porte ni variante FP32 (A49-16).

## Renvois traités

| De | Où | Conclusion |
|---|---|---|
| 27 | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | Doublon de A49-03 (ce qui joue le FP8) et de A34-01 (le défaut de construction, mêmes lignes). Une partie du constat est réfutée : les préréglages non calibrés ne posent pas le FP8. Reste un défaut de documentation : A49-15. |
| 44 | qualité DistilDRBA F16 (mesure pour A49-14) | Doublon de A49-14 pour l'absence de mesure écrite, et de A03-18, A44-01 et A46-04 pour la part due à la grille. La part qui reste après correction de la grille est nouvelle : A49-16. |

### Renvoi de 27 : noyaux FP8 de SPAN construits par défaut sous Linux

Lu, et exact : `CMakeLists.txt:27-32` pose `AJP_WITH_SPAN_FP8` à ON sous Linux dès que les en-têtes de CUTLASS sont
là, à OFF ailleurs. Mais construire les noyaux ne change aucune image : ils ne servent qu'au cran Performance de SPAN
(`src/app/play/chain_setup.cpp:292-297`, `span_variant_order`). Ce cran n'est posé que par trois endroits :

- l'Auto calibré (`src/app/play/auto_config.cpp:202`, d'après `src/app/play/machine_calibration.cpp:142` et
  `:252`) : c'est A49-03 ;
- la calibration elle-même, pour ses essais (`machine_calibration.cpp:142`) : A49-03 encore ;
- le choix à la main « … · FP8 » dans la liste des modèles (`src/app/play/settings_chain_tabs.cpp:93-108`) : un
  choix de l'utilisateur, pas un défaut.

Le constat du renvoi « pris par défaut dans les préréglages sur RTX 40 et 50 » (`docs/nvidia-linux.md:45`) est
**réfuté pour les préréglages non calibrés** : `auto_config.cpp:169` pose `PrecisionTier::Balanced` (FP16) à tous
les rangs, et le réglage par défaut vaut `Balanced` (`src/app/play/settings.h:215`). C'est la page de
documentation qui est fausse (A49-15).

La mesure citée est confirmée dans `docs/span-fp8-cutlass.md:19-35` : 9,09 à 9,34 ms contre 14,25 ms au banc
(−34 à −36 %), mais −6,5 % seulement en lecture (14,5 contre 15,53 ms, `:21`), pour 49,8 à 52,0 dB contre le
SPAN d'origine en FP32 et 49,15 dB contre le moteur FP16. Ces chiffres sont déjà ceux de A49-03 et du compromis C3.
L'écart entre Windows et Linux est A34-01.

### Renvoi de 44 : DistilDRBA F16 contre FP32

La mesure est celle de l'audit 44 (interpréteur CPU à stockage F16 émulé, trois plans réels de 1024 × 576) : 51 à
52 dB, écart maximal 13 à 19/255. Elle n'a pas été rejouée ici (ni GPU ni `onnx` dans ce fil) et ne porte pas sur
TensorRT : elle reste à confirmer sur la carte. Ce que le code confirme :

- le DistilDRBA livré est en FP16 de bout en bout : `tools/drba/export_distil_drba.py:197` le convertit après un
  contrôle fait en FP32 seulement, contre PyTorch, en 256 × 192 (`:155-167`, `:179-180`, `:194-195`) ;
- la chaîne de la grille reste en FP16 parce que `GRID_OPS` ne contient pas `Concat`
  (`tools/rife/convert_rife_fp16.py:35`) : A03-18, toujours présent ;
- aucun `distil_drba_v2_lite_fp32.onnx` n'est livré ni publié (le nom n'existe que dans les rapports d'audit) : au
  cran Qualité, `prepare_distil_drba` écrit « manque, Équilibre à la place » et joue le FP16
  (`src/app/play/player_engines.cpp:219-228`) ;
- les contrôles de qualité existants comparent le fusionné au graphe FP16 entier, jamais au FP32
  (`tests/unit/backend_tensorrt/drba_fused_test.cpp:369`, `:396`, seuil 50 dB ; `docs/drba-fusionne.md:58-61`).

A49-14 reste donc ouvert tel quel : la mesure existe maintenant dans un rapport d'audit, pas dans le script de
conversion ni dans `docs/interpolation.md`. La mesure de 44 dit aussi d'où vient la perte (A44-01) : corriger la
grille rend 2 dB sur le panoramique f3 (52,15 → 54,22 dB, écart maximal 16,3 → 9,0/255), mais presque rien sur les
plans g1 et g4.5 (+0,01 à +0,04 dB, écart maximal 19,0 → 17,7/255 sur g1). Cette part-là n'est couverte par aucun
défaut : A49-16.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A49-15 | Mineur | `docs/nvidia-linux.md:45` | La page, classée « code actuel » par `docs/README.md:18`, annonce « SPAN en basse précision dans les préréglages (ADR 0007) : FP8 CUTLASS sur RTX 40 et RTX 50 … désactivation : préréglage Personnalisé ». Le code fait l'inverse hors calibration (`src/app/play/auto_config.cpp:167-169` : FP16, le FP8 se choisit à la main) et l'ADR 0007 est suspendue (`docs/adr/0007-span-basse-precision-par-defaut.md:37-42`). La page ne dit pas non plus le seul cas où le FP8 est posé d'office, l'Auto calibré (A49-03). | Un lecteur de la page (ou un audit, comme le renvoi de 27) croit que tout préréglage joue le FP8 sur RTX 40 ; un utilisateur qui veut l'éviter passe en Personnalisé pour rien, alors que l'Auto calibré, lui, le pose sans le dire. | Récrire la ligne d'après le code : FP16 dans les préréglages, FP8 au choix dans la liste des modèles ; et l'accorder à ce qui sera décidé pour A49-03. | Lu. |
| A49-16 | Mineur | `tools/drba/export_distil_drba.py:155-198`, `src/app/play/player_engines.cpp:219-228`, `tests/unit/backend_tensorrt/drba_fused_test.cpp:369` | DistilDRBA n'existe qu'en FP16, et ce FP16 perd autant face au FP32 que le FP8 de SPAN (51 à 52 dB, écart maximal 13 à 19/255, mesure de 44) sans aucune porte : SPAN FP8 a un reçu et des seuils, DistilDRBA FP16 rien. La grille (A03-18) n'explique qu'une partie : sur le plan g1 l'écart maximal reste à 17,7/255 une fois la grille en FP32. Aucun graphe `_fp32` n'est livré, donc le cran Qualité retombe en FP16 ; le gain de vitesse du FP16 sur le FP32 n'est mesuré nulle part pour ce réseau (seul le FP16 l'est : 2,10 ms entier, 0,92 ms fusionné en 1088 × 1920, `docs/drba-fusionne.md:52-55`). | Niveau « Très rapide » (DistilDRBA, choisi à la main ; hors de l'Auto tant que `kDistilDrbaInAuto` est faux, `src/app/play/auto_config.h:22`) : quelques pixels par image créée s'écartent de 13 à 19 codes 8 bits du modèle d'origine, y compris sur des plans sans grand mouvement, et le cran Qualité ne peut rien y changer. | Après A03-18, chercher sur g1 quelles couches portent l'écart (comme le script de RIFE l'a fait pour Resize et GridSample) et les garder en FP32 ; ajouter au script d'export la mesure FP16 contre FP32 à la taille de lecture (c'est A49-14) avec un seuil ; mesurer le temps du graphe FP32 sur la carte et le livrer comme cran Qualité s'il tient. En faire la « porte de qualité » attendue avant de passer `kDistilDrbaInAuto` à vrai. | Lu pour le code ; la mesure est celle de 44, non rejouée. À confirmer : sur la 4070 Ti, `linux-release`, sortie du moteur TensorRT FP16 contre ONNX Runtime FP32 du modèle d'origine, à 720p, sur f3, g1 et g4.5 (PSNR, écart maximal, p99,9) ; puis temps par image du graphe FP32. |

## Doublons

- Renvoi de 27 : A49-03 (`SYNTHESE.md:413`, l'Auto calibré en FP8) et A34-01 (`SYNTHESE.md:494`, mêmes lignes du
  CMake, écart Windows et Linux). Le renvoi 02 → 34 (`SYNTHESE.md:731`) portait déjà sur ces lignes.
- Renvoi de 44 : A49-14 (`SYNTHESE.md:424`, aucune mesure écrite), A03-18 (`SYNTHESE.md:104`), A44-01
  (`SYNTHESE.md:618`) et A46-04 (`SYNTHESE.md:633`) pour la grille. A31-06 (`SYNTHESE.md:437`) couvre déjà la
  lenteur du cran Qualité retombé en FP16 ; A49-16 n'en reprend que le côté qualité.

## Baisse de qualité pour la vitesse

| Baisse | Ce qu'on perd | Gain mesuré | Justifiée ? |
|---|---|---|---|
| SPAN FP8 CUTLASS, construit par défaut sous Linux (renvoi de 27) | 49,8 à 52,0 dB contre le FP32 d'origine | −34 à −36 % au banc, −6,5 % en lecture, 72 W au lieu de 149 W (`docs/span-fp8-cutlass.md:19-25`) | La construction seule ne coûte rien à l'image. Posé d'office par l'Auto calibré, le gain en lecture ne justifie pas la perte : déjà A49-03 et compromis C3, rien à ajouter. |
| DistilDRBA en FP16 seulement (renvoi de 44) | 51 à 52 dB, écart maximal 13 à 19/255 (émulation CPU de 44) | Aucun : le temps du FP32 n'a jamais été mesuré | Non, faute de mesure : A49-16. À ajouter au compromis C13 du rapport. |
