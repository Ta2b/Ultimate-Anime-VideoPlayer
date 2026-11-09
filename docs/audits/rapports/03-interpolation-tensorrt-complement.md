# Complément de l'audit 03. Pourquoi DistilDRBA n'est pas beaucoup plus rapide que RIFE

Branche : `claude/project-thread-gnywvn`. Commit audité : `1e3d71f`. Date : `2026-11-09`.

## Verdict

Par passe, DistilDRBA fusionné tient sa promesse (0,37 ms contre 1,21 ms pour RIFE 4.25 fusionné à 960 × 576, soit
3,3 fois moins) ; il la perd par paire et par image, pour trois raisons qui s'additionnent : il refait une passe
à chaque image créée là où RIFE en fait une par paire, son entrée coûte autant ou plus que son réseau, et le
réseau d'interpolation ne pèse de toute façon que quelques pour cent du temps de service d'une image.

## Cadrage retenu

Consigne de Tatoub (09/11), relayée par le coordinateur : aucune question, auditer au maximum ; RIFE 4.25, RIFE
4.22 lite et DistilDRBA comptent autant, avec un intérêt particulier pour le potentiel de DistilDRBA ; le VRR
est hors sujet. Ce complément suit une question (la vitesse de DistilDRBA face à RIFE sous TensorRT) à travers
les zones 03, 04, 05, 08 et 09 : les défauts gardent un ID `A03-xx` à la suite du rapport 03, la colonne « Où »
dit la zone réelle. Les pistes connues A03-03, A09-01, A09-02, A09-07 et A08-06 sont reliées plus bas.

Le fil tourne sans carte NVIDIA : rien n'a été mesuré ici. Les chiffres viennent des relevés du dépôt, cités
avec leur fichier ; ils datent de la fin octobre (RTX 4070 Ti, TensorRT 11.3) et la chaîne a changé depuis.
Ce qui est calculé à partir d'eux est marqué « estimé », et chaque défaut donne la manipulation qui le confirme.

## Ce que disent les mesures déjà au dépôt

| Mesure | RIFE 4.25 fusionné | DistilDRBA fusionné | Source |
|---|---|---|---|
| Une passe, 960 × 576 | 1,21 ms (1,32 en lecture) | 0,37 ms | `docs/plan-v0.6.md:121`, `docs/rife-fusionne.md:56` |
| Une passe, 1280 × 768 | 2,20 ms | 0,4 à 0,5 ms (estimé : blocs 0,064 + 0,075 + 0,129 ms mesurés, plus encodeur et noyaux) | `docs/rife-fusionne.md:10`, `docs/fp8-interpolation.md:105` |
| Une passe, 1920 × 1088 | non mesuré | 0,917 ms (moteurs 0,695, noyaux 0,152) | `tools/bench/fp8/drba_fusion/resultats.txt:15-25` |
| Lecture en Rapide, scène lourde, 720p vers 1440p | 60 à 61 i/s, écart de service 16,2 ms | 67,5 i/s, écart de service 13,1 ms | `docs/auto-libre.md:157-172` |

Trois lectures :

1. **Le réseau n'est pas le budget.** Une image créée coûtait 13 à 16 ms de service ; la passe de DistilDRBA en
   prend 0,4 à 0,5. Passer de RIFE à DistilDRBA ne peut donc rendre que la part de RIFE dans ces 16 ms, et le
   relevé le montre : + 11 % d'images, pas « beaucoup plus ». Le reste (SPAN, warp à la taille de sortie,
   écriture de l'anneau, présentation) est commun aux deux moteurs et relève de l'audit 32.
2. **RIFE amortit, DistilDRBA non.** Avec les poses gardées (le défaut), RIFE fait une passe au milieu de la
   paire et la réutilise pour tous les instants ; DistilDRBA fait une passe par instant (A03-13). À 1280 × 768 :
   RIFE 4.25 coûte 2,2 ms par paire calme quel que soit l'écran ; DistilDRBA coûte environ 0,45 ms × le nombre
   d'instants, soit 0,7 ms à 60 Hz (1,5 instant par paire en 24 i/s), 1,8 ms à 120 Hz, 2,7 ms à 165 Hz, l'écran
   de Tatoub. Le croisement est vers × 5 : au-delà, DistilDRBA fait **plus** de travail de réseau que RIFE 4.25
   sur une paire calme. Il ne garde l'avantage que sur les paires en mouvement fort, où RIFE repasse à chaque
   instant.
3. **L'avantage par passe est en partie mangé avant le réseau.** Chaque côté de la paire réécrit deux entrées
   de 7 plans pleine taille (A03-14), et l'encodeur retravaille des images déjà encodées (A03-15).

RIFE 4.22 lite fusionné n'a aucune mesure par passe dans les documents : la comparaison qui intéresse Tatoub
(4.22 lite contre DistilDRBA) ne peut pas se faire sur pièces (A03-17).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A03-13 | Majeur | `src/app/play/upscale_drba.cpp:247-257`, `src/app/play/upscale_drba.cpp:179-216` (zone 09) ; à comparer à `src/app/play/upscale_motion.cpp:111-126`, `:185-189` | DistilDRBA lance une passe complète du réseau **par image créée**. RIFE, lui, a deux économies que DistilDRBA n'a pas : une passe au milieu sert toute la paire quand les poses sont gardées ou que le mouvement reste sous `kSingleRunMotion720` (`kernels/spec/warp.h:51`), et le warp met ce mouvement à l'instant voulu. C'est la quatrième économie dont DistilDRBA est privé, la plus lourde, absente d'A09-07. | Écran 165 Hz, source 24 i/s, plan calme, 5 à 6 instants par paire : RIFE 4.25 fait 1 passe (2,2 ms), DistilDRBA 5 à 6 passes (2,2 à 2,7 ms, estimé). Le « Très rapide » du menu fait autant ou plus de réseau que l'« Équilibré ». | Deux voies, la première sans risque pour l'image. **(a) Passer les instants d'une paire en un seul lot.** Les blocs sont de petites convolutions limitées par la latence, pas par le débit : le bloc 0 ne prend que 33 % de plus pour 2,1 fois plus de pixels (0,064 ms à 1280 × 768, 0,085 ms à 1920 × 1088). Un lot de N instants (mêmes images, seul le plan du temps change ; les deux côtés peuvent partager le lot, le côté ne sert qu'à la colle) coûterait bien moins que N passes. Les instants sont connus à l'arrivée de la paire (`upscale_stage.cpp:473-489`). Demande des moteurs à lot (1 à 8) dans `derive_drba_fused` et une colle par élément du lot. **(b) Une passe par côté, puis le warp aux autres instants** comme RIFE : au plus 2 passes par paire. Touche à l'image (le réseau règle l'instant pixel par pixel) : à juger à l'œil. | Lu. À confirmer : même extrait, `--rife=6`, `[interpolation] engine = "distil_drba"` puis RIFE 4.25 ; comparer au journal de fin « DistilDRBA : N passes » et « RIFE mouvement : N passes ; N paires à une passe », et les lignes « moteur … fusionné » du profil GPU multipliées par leur nombre d'appels. Pour (a), avant tout code : `trtexec --shapes` en lot 1, 2, 4 sur `bloc0`, `bloc1`, `bloc2` (`tools/bench/fp8/drba_fusion/moteurs.sh`) ; la voie vaut si le lot 4 coûte moins de 2 fois le lot 1. |
| A03-14 | Majeur | `src/app/play/upscale_drba.cpp:193-208` (zone 09), `src/gpu/cuda/backend/drba_fused_kernels.cu:104-114`, `:154-164`, `:170-179`, `src/gpu/cuda/backend/tensorrt_backend.cpp:419`, `src/gpu/cuda/backend/tensorrt_rife.cpp:118-124` | L'entrée de DistilDRBA n'a pas reçu l'« entrée par pointeurs » de RIFE (étape 5.2). Chaque changement de côté réécrit `pair01` et `pair12` par deux `rife_input` de 7 plans remplis : 4 appels par paire, 55 Mo écrits à 1280 × 768, contre un `rife_pair` pour RIFE. Or le moteur fusionné n'en lit presque rien : 4 pixels sur 16 des plans d'images (`pre_kernel`, moyenne 2 × 2 tous les 4 pixels), jamais les plans 0 à 2 ni le plan 6 de `pair12`, et, en variante mouvement (le chemin par défaut), jamais la pleine résolution. `rife_time` écrit en plus deux plans entiers par instant suivant, puisque `note_rife_reader` ne tient pour « compact » que RIFE fusionné. Ces écritures se font sur la voie d'interpolation, au moment de produire, alors que RIFE prépare son entrée en avance sur la voie des mesures (`upscale_ahead.cpp:105-114`, `upscale_stage.cpp:522-524`). | En Rapide à 60 ou 72 Hz, une paire a un ou deux instants, presque toujours un de chaque côté : chaque image créée paie deux `rife_input`. Le relevé du 28/10 donne 0,4 à 0,8 ms médian à l'étape « entrée de RIFE » (11 plans, 960 × 576, en lecture) : si l'ordre de grandeur tient pour 7 plans, l'entrée coûte plus que la passe (0,37 ms). Au mieux (borne mémoire du banc, 500 Go/s), 4 appels font 0,2 ms par paire, la moitié d'une passe. | Donner au moteur fusionné les trois images de travail et l'instant : `pre_kernel` lit `before`, `a`, `b` (ou `a`, `b`, `after`) à leur adresse, décalées de `left`, `top`, le noir hors fenêtre, et lit l'instant à l'élément 0 d'un petit tampon (les paramètres d'un noyau sont figés dans le graphe CUDA capturé, d'où la lecture sur le périphérique). `pair01` et `pair12` ne restent que pour le moteur entier et la variante image (dont la fin lit la pleine résolution). | Lu. À confirmer : ligne « entrée de RIFE » du profil GPU en lecture DistilDRBA (`--one-lane` pour un coût propre), nombre d'appels et médiane, face à « moteur DistilDRBA fusionné ». |
| A03-15 | Mineur | `src/gpu/cuda/backend/drba_fused_engine.cpp:225-242`, `:311-313`, `src/app/play/upscale_drba.cpp:188-191`, `:210` | L'encodeur (lot de 3) et les réductions sont refaits à chaque changement de côté : 6 encodages par paire pour une seule image nouvelle (`after`), puisque le côté a lit (avant, a, b) et le côté b (a, b, après). La reprise D6 ne sert que les instants suivants du même côté ; avec un ou deux instants par paire elle ne joue jamais. Sur une carte de moins de 7,5 Gio (une seule voie, `tensorrt_lanes.cpp:25`, `:75`), SPAN écrit dans la même zone entre deux instants et l'annule (`last_user() != this`). C'est le pendant d'A03-08 pour RIFE. | 60 Hz : 1,5 passe par paire, 3 images encodées à chaque passe, soit 4,5 encodages par paire au lieu de 1. L'encodeur vaut 0,106 ms sur 0,917 à 1920 × 1088 (12 % de la passe). | Garder par image source ses réductions et son encodage (quatre emplacements qui tournent comme `before`, `rgb_`, `after`), hors de la zone partagée ; `pre` et `post` rassemblent X_0, X_1, X_2 depuis trois adresses. Se fait avec A03-14 (même changement d'interface). | Lu. À confirmer : `nsys`, part de l'encodeur dans une passe à 1280 × 768. |
| A03-16 | Mineur | `src/gpu/cuda/backend/drba_fused_kernels.cu:277-290`, `src/app/play/upscale_drba.cpp:273-279` | En variante mouvement, la fin agrandit le flot et le masque × t jusqu'à la taille remplie (5 plans), que le warp rééchantillonne aussitôt à la taille de sortie : deux interpolations bilinéaires de suite, et la fin est le noyau le plus cher de la chaîne (0,085 ms sur 0,152 à 1920 × 1088, à sa borne mémoire ; mesuré en variante image, la variante mouvement ne l'est pas). | Chaque image créée en « warp après SPAN », le chemin par défaut. | Sortir le flot et le masque à 1/t (le DepthToSpace de Y_2, sans agrandissement) et laisser le warp faire la seule mise à l'échelle (`Motion::area` la prévoit). Demande une fenêtre dont le coin n'est pas multiple de t pour certaines hauteurs, et change le sous-pixel du flot : à valider par PSNR contre la sortie actuelle. | Lu. À confirmer : banc `drba_fusion_bench`, fin en variante mouvement, avec et sans agrandissement. |
| A03-17 | Mineur | `tests/unit/backend_tensorrt/drba_fused_test.cpp`, `docs/drba-fusionne.md:51-79`, `docs/rife-fusionne.md:48-63` | Rien ne mesure ce que la question demande. Les bancs et les documents donnent le temps **par passe**, chaque moteur contre son réseau entier ; aucun ne donne le coût **par paire** ou **par image créée** (entrée, passes, warps) de DistilDRBA contre RIFE à chaîne égale, et RIFE 4.22 lite fusionné n'a aucun chiffre. La décision « Très rapide = DistilDRBA » du menu (`settings.h:82`) repose sur le temps par passe. | Une régression qui double l'entrée de DistilDRBA ne se voit dans aucun chiffre suivi. | Un banc caché `[.bench]` : pour 1 à 6 instants par paire, coût GPU total d'une paire (entrée, passes, warps à 1440p) pour RIFE 4.25, RIFE 4.22 lite et DistilDRBA fusionnés, à 1280 × 768 ; le tableau va dans `docs/interpolation.md`. | Lu. |

## Qualité : où le modèle est dégradé (ajout du 09/11, à la demande de Tatoub)

Question posée : tirer mieux parti de la vitesse sans négliger la qualité, et vérifier que le modèle n'est
dégradé nulle part. Le trajet relu : export, conversion FP16, ingestion, entrée, réseau, colle, sortie, warp.

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A03-18 | Majeur | `tools/rife/convert_rife_fp16.py:35`, `:56-65`, `tools/drba/export_distil_drba.py:65-73`, `src/gpu/cuda/backend/drba_fused_kernels.cu:64-69` | La grille des dix GridSample de DistilDRBA est calculée en **FP16**, ce que le convertisseur interdit pour RIFE (« perd 1 à 2 pixels de précision », `:8-9`). Le convertisseur remonte de chaque GridSample par `Transpose`, `Add`, `Mul` ; dans l'export de DistilDRBA un `Concat` s'intercale (`torch.cat([x, y])` puis `permute`) et arrête la remontée : `Add(grille, Mul(flot, pas))` reste en FP16. Un pas de FP16 près de 1 vaut 0,31 pixel à 1280 de large et 0,47 à 1920, surtout dans la moitié droite et basse de l'image. Les noyaux fusionnés reproduisent cet arrondi exprès (`grid`, deux arrondis), et tous les contrôles de qualité comparent au graphe FP16 lui-même : l'écart au modèle d'origine en FP32 n'a jamais été mesuré à la taille de lecture (`check` compare le FP32 avant conversion, en 256 × 192). | Panoramique lent : les warps internes placent les traits sur une grille d'un tiers à un demi pixel au lieu de les suivre ; tremblement fin des contours, plus fort à droite de l'image. En variante image, le warp final du graphe a le même défaut à pleine résolution. | Ajouter `Concat` à `GRID_OPS` pour que la chaîne de la grille reste en FP32 dans le graphe entier, et retirer les arrondis de `grid` dans la colle (calcul en FP32, sans coût). Le fusionné devient alors plus fidèle au modèle d'origine que le graphe FP16 actuel ; ses tests se comparent au graphe corrigé. | Lu, et vérifié sur le modèle livré (`distil_drba_v2_lite.onnx`, parcours du graphe avec `onnx`) : devant le dernier GridSample, `/Sub_41` et `/Sub_43` (la base) passent par un Cast vers FP16, `/Add_20` et `/Add_21` sont en FP16, puis `/Cast_65` (FP16) et un Cast vers FP32. Reste à mesurer : sortie du graphe FP16 actuel et du graphe corrigé contre le FP32 d'origine (ONNX Runtime), 1280 × 768, paires d'anime avec 6 et 20 px de décalage ; l'écart attendu est le « plancher de 60 à 67 dB » de `docs/drba-fusionne.md:60-61`. |
| A03-19 | Mineur | `src/app/play/chain_setup.cpp:359-363`, `:378-381`, `src/app/play/upscale_drba.cpp:273-279` | La vitesse de DistilDRBA n'est pas réinvestie dans l'image. Au-dessus de 720 lignes de travail il estime, comme RIFE, sur une copie réduite à 720p (D4), alors qu'une passe à 1920 × 1088 ne coûte que 0,92 ms, moins qu'une passe de RIFE 4.25 à 960 × 576. | Source 1080p agrandie en 4K : le flot est estimé à 720p puis étiré 3 fois au warp ; les petits mouvements (mèches, bouches) sont lissés. | Pour DistilDRBA seul, estimer à la taille de travail jusqu'à 1080 lignes ; garder le demi-flot au-delà, comme l'auteur le conseille. À juger à l'œil et au temps. | Lu. À confirmer : même plan, estimé en 720p puis en 1080p, contours des petits objets en mouvement. |

Les autres points du trajet, relus :

- **Déjà au tableau des audits, et qui touchent ce que le modèle reçoit ou rend** : chroma rééchantillonnée en
  bilinéaire avant le modèle (A03-01) ; variante mouvement jamais comparée au graphe (A03-03) ; fantôme d'une
  image avant certaines coupes (A09-01) ; pas de débandage avant SPAN avec DistilDRBA (A09-07) ; cran
  Performance en FP8, 49,6 dB au pire, et deux fois plus lent (A09-02).
- **Conformes au modèle d'origine** : l'instant (`drba_call`, tau = 1 + t avant le milieu, t après), le côté
  répété au début, à une coupe et en fin de flux, le remplissage noir, l'ordre des canaux, les réductions
  (bit à bit contre l'hôte), le sigmoïde et le mélange.
- **Choix assumés, à connaître** : le warp après SPAN remplace l'image du réseau par un warp des images
  agrandies avec son flot (plus net, un SPAN par source, mais SPAN ne nettoie plus le warp) ; un côté « tenu »
  rend l'image sans passe (seuil des dessins tenus) ; seul le modèle v2 lite est reconnu par la découpe, le v1
  à cinq échelles de l'auteur, plus lourd, n'a jamais été essayé alors que la marge de temps existe.
- **Les corrections de vitesse proposées plus haut** : A03-13 (a), A03-14 et A03-15 rendent la même image, au
  bit près pour les deux dernières. A03-16 change le sous-pixel du flot et A03-13 (b) change l'image : avec la
  consigne « sans négliger la qualité », (b) est à écarter et A03-16 à ne faire que si le PSNR tient.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A03-03 | Toujours présent, et préalable | La branche mouvement du fusionné (`drba_fused_kernels.cu:284-290`) reste sans test ni essai au chargement, et c'est le chemin par défaut. A03-13 (a), A03-14 et A03-16 la modifient toutes : écrire d'abord le test contre le graphe mouvement entier, sinon chaque gain de vitesse se fait à l'aveugle. |
| A09-07 | Toujours présent, complété | Les trois économies manquantes (avance, saut des instants passés, débandage) sont confirmées (`upscale_stage.h:221`, `:319`, `upscale_stage.cpp:510-511`). A03-13 en ajoute une quatrième, la passe unique par paire, et A03-14 une cinquième, l'entrée préparée en avance. Le saut des instants passés compte double pour DistilDRBA : une passe jetée n'est pas amortie sur d'autres instants. |
| A09-02 | Toujours présent | `drba_fused_choice.cpp:9` : le cran Performance coupe le fusionné et charge le graphe FP8 entier, environ 1,9 ms par passe au lieu de 0,92 (`docs/fp8-interpolation.md:102-104`). Qui cherche la vitesse et choisit « Performance » obtient une passe deux fois plus lente. À corriger avec le reste : c'est le seul réglage qui ralentit DistilDRBA d'un facteur 2 d'un clic. |
| A09-01 | Toujours présent, lien faible avec la vitesse | Les coupes jugées sur le score brut (`upscale_drba.cpp:104`, `:155`) coûtent une passe inutile et un fantôme par coupe manquée ; rare, sans effet sur le débit moyen. Reste un défaut d'image. |
| A08-06 | Toujours présent, hors du chemin fusionné | `prepare_distil_drba` rend le fusionné avant de toucher à la variante mouvement (`player_engines.cpp:247-267`) : le moteur construit puis jeté n'arrive que si le fusionné est refusé **et** que la dérivation mouvement échoue. Coût à l'ouverture seulement, aucun en lecture. |
| A03-08 | Toujours présent | Même cause qu'A03-15 côté RIFE (les deux images réencodées à chaque paire). Un cache d'encodages par image source servirait les deux moteurs. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 32 | `docs/auto-libre.md:157-172`, `src/app/play/live_settings.h:416-422` | Le temps de service d'une image (13 à 16 ms au relevé de `docs/auto-libre.md`) est dix à trente fois celui d'une passe d'interpolation : c'est lui qui borne la cadence en Rapide, quel que soit le moteur. Refaire ce relevé sur la chaîne actuelle (sans RTX HDR, warp après SPAN) et le décomposer par étape. |
| 07 | `src/app/play/machine_calibration.h:14`, `src/app/play/auto_config.cpp:203`, `src/app/play/auto_config.h:22` | La calibration n'essaie pas DistilDRBA et l'Auto calibré impose RIFE 4.22 lite ; `kDistilDrbaInAuto` est faux. DistilDRBA ne tourne donc que choisi à la main, et aucune mesure de la machine ne dit s'il y gagne. |
| 23 | `src/app/play/settings_chain_tabs.cpp:375-383` | La liste « Warp and SPAN » est grisée dès que le moteur n'est pas RIFE, alors que l'ordre s'applique à DistilDRBA (`chain_setup.cpp:376-377`) et décide de sa variante. |
| 09 | `src/app/play/upscale_drba.cpp:144`, `:150` | DistilDRBA mesure la paire (b, après) à la taille de travail (`rgb_`, `drba_.after`) quand RIFE la mesure sur la copie réduite (`rife_source`, `upscale_ahead.cpp:92-93`) : plus de pixels lus au-dessus de 720 lignes de travail, et un score pris sur une autre image que celle de RIFE. |
| 05 | `src/prepare/trtexec_command.cpp:105-106`, `src/gpu/cuda/backend/tensorrt_drba_fused.cpp:47-66` | Les formats d'entrée et de sortie des quatre morceaux sont laissés à TensorRT ; chaque frontière (8 par passe) peut porter un reformatage. À lire dans les `.layers.json` des morceaux (`layer_counts`) : si des « Reformat » y sont, la colle pourrait écrire au format que le moteur attend. À confirmer, sans certitude. |

## Points positifs

- Par passe, le fusionné est bien construit : 57 des 109 canaux de X_2 ne sont jamais copiés, toute la chaîne
  tient dans un graphe CUDA, la reprise D6 saute l'encodeur pour les instants d'un même côté
  (`drba_fused_engine.cpp:225`), et un côté tenu ne coûte aucune passe (`upscale_drba.cpp:220-222`).
- Les morceaux se construisent au niveau d'optimisation 5 (`model_source.cpp:123`,
  `trtexec_command.cpp:76`) ; le FP8 a été mesuré puis écarté pour de bonnes raisons (blocs trop petits,
  + 8 % et + 46 % sur les blocs 1 et 2). Il n'y a rien à gagner du côté de la précision.
- Avec trois voies, l'interpolation a sa propre zone d'activation (`tensorrt_lanes.cpp:75`) : SPAN n'annule
  pas la reprise D6 sur une carte de 8 Go et plus.
- Le warp après SPAN vaut pour DistilDRBA comme pour RIFE (un SPAN par source, `upscale_drba.cpp:258-262`),
  avec l'écriture directe dans l'anneau.

## Questions ouvertes

Rien à demander à Tatoub avant d'agir ; l'ordre proposé, chaque étape se décidant sur une mesure de sa machine :

1. Le banc d'A03-17 et le relevé du renvoi 32 d'abord : ils disent si l'interpolation pèse encore assez dans
   une image pour mériter le reste. Si le service d'une image reste au-dessus de 10 ms, le gain le plus gros
   pour la fluidité n'est pas dans DistilDRBA mais dans ce que les deux moteurs partagent.
2. Le test d'A03-03, puis A03-14 et A03-15 ensemble (entrée par pointeurs, cache par image) : sans effet sur
   l'image, gain estimé de 0,2 à 1 ms par paire selon ce que coûte vraiment l'entrée.
3. L'essai `trtexec` par lots d'A03-13 (a), une heure de mesure sans code. S'il passe, DistilDRBA devient le
   seul moteur dont le coût par paire ne croît presque pas avec la fréquence de l'écran, ce qui est la
   promesse attendue à 144 et 165 Hz.
4. A03-13 (b) seulement si (a) échoue, et après un jugement à l'œil.
5. Si, après 2 et 3, DistilDRBA bat RIFE 4.22 lite par paire à qualité jugée égale : ouvrir la porte de
   qualité de `kDistilDrbaInAuto` et le faire essayer par la calibration (renvoi 07).
