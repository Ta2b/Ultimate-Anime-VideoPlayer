# FP8 pour RIFE, DistilDRBA et GMFSS : ce qui marche, ce qui ne marche pas

> **Mise à jour du 30/10 (0.7.0-rc.1) : le FP8 de GMFSS est retiré.** Sur la 0.7.0-rc.1, Tatoub a vu « une très mauvaise image dans les gros
> mouvements » avec la paire GMFSS en FP8 : les 46,9 dB mesurés l'étaient sur des paires de test qui ne couvraient pas assez de mouvement fort. Le
> cran Performance n'est plus proposé pour GMFSS (menu) ni choisi par les préréglages ; GMFSS reste en FP16 (Équilibre) ou FP32 (Qualité). Les deux
> graphes FP8 restent au catalogue (`models-interp-1`), sans être offerts ; un `play.toml` qui demande Performance lit Équilibre. Rouvrir ce chantier
> demande une mesure sur des mouvements forts (voir le plan v0.6, point 25). Le texte ci-dessous est l'état d'avant ce retour.

Demande de Tatoub (29/10) : « ajoute les modèles FP8 au lecteur pour RIFE, DistilDRBA et GMFSS ». Règle du projet : un
graphe FP8 n'est activé par défaut que si le gain mesuré est réel (**au moins 8 % en ABAB** sur le moteur concerné) **et**
que la qualité tient (PSNR moyen ≥ 42 dB, pire cas ≥ 38 dB, SSIM moyen ≥ 0,985 contre le FP16). Ce document dit ce que
les mesures ont donné, sans arrondir en faveur de la demande.

## Résumé

| Réseau | Gain FP8 (ABAB, médiane GPU) | Qualité (pire cas) | Verdict |
|---|---|---|---|
| **GMFSS** paire s05, base et union, FP8 hors du squelette de GMFlow | **−15 %** (9,9 → 8,4 ms) | 46,9 dB base, 48,6 dB union ; SSIM ≥ 0,998 | **livré, par défaut** (cran Performance de GMFSS) |
| GMFSS paire, toutes les convolutions en FP8 | −10 à −22 % (bruité) | 33,2 dB base, 36,4 dB union | écarté : pire cas sous 38 dB |
| GMFSS fusion s05 en FP8 | union −8,6 à −9,2 % (14,8 → 13,4 ms) | union 36,9 dB (cas de mouvement fort), base 39,1 dB | écarté : gardes non tenues |
| **DistilDRBA** v2 lite entier | −11 à −12 % (1,28 → 1,13 ms à 768×1280) | 49,6 dB ; SSIM 0,9994 | livré **facultatif**, éteint : il coupe DistilDRBA fusionné (2× plus rapide) |
| DistilDRBA, sous-moteurs du fusionné (blocs 0 à 2) | −1,5 % / +8 % / +46 % | — | aucun gain : rien à dériver |
| **RIFE** 4.25 entier | −5 à −6 % (8,3 → 7,9 ms) | 31,3 dB ; SSIM 0,972 | écarté : sous 8 %, gardes non tenues |
| RIFE 4.22 lite entier | **+27 % (plus lent)** | 30,8 dB | écarté |
| RIFE 4.25, sous-moteurs du fusionné (blocs 0 à 4) | −2 % / −3 % / +30 % / +34 % / +14 % (somme +16 %) | — | écarté : plus lent |

Par défaut, donc : **GMFSS passe au FP8 sur la paire** (gain réel, qualité tenue), sur une RTX 40 et après avec TensorRT, et
retombe seul sur le FP16 si le graphe manque ou si le moteur ne calcule pas le FP8. **RIFE et DistilDRBA restent en FP16**,
et le cran Performance de ces deux réseaux reste visible dans Réglages avancés > Fluidité avec une note honnête (voir
plus bas) : aucun graphe FP8 de RIFE n'est publié, celui de DistilDRBA l'est comme fichier facultatif.

## Méthode

Machine : RTX 4070 Ti, TensorRT 11.3.0.99 (réseau fortement typé : la précision vient des types du graphe), 720p de travail
(plans remplis 768×1280, 384×640 pour la fusion GMFSS).

- **Graphes** : `tools/fp8/interp_fp8.py quantize` insère des QuantizeLinear/DequantizeLinear E4M3 (opset 19) devant chaque
  convolution retenue : poids en E4M3 par tenseur, activations par tenseur, échelles statiques calibrées sur le maximum de
  l'activation (sur la paire GMFSS toute en FP8, le maximum donne 34,6 dB au pire cas contre 33,2 dB au percentile 99,99 % et 33,1 au
  99,999 %). Les convolutions
  de moins de 32 canaux (entrée ou sortie) restent en FP16. ModelOpt n'est pas utilisé : il demande PyTorch, et ses
  réglages par défaut n'apportent rien de plus que ce placement pour TensorRT (Q/DQ devant chaque convolution).
  `build_interp_fp8.sh` refait tous les graphes et les gardes de qualité depuis une vidéo.
- **Images** : de vraies images d'anime (Re:Zero 4e saison, épisode 18), neuf instants de 95 s à 1330 s, 1280×720 puis
  plan rempli 768×1280. Par instant, les deux couples d'images consécutives qui bougent le plus (écart moyen) : l'anime se
  joue souvent « sur deux », et un couple d'images identiques donne un flot nul et des PSNR sans signification. **Piège
  rencontré** : une première série sur de simples couples consécutifs donnait 75 dB de moyenne pour RIFE FP8, fausse ; avec
  les couples qui bougent, le pire cas tombe à 31 dB. Entrées bien formées : RIFE 11 plans (images, instant, grille X/Y,
  pas X/Y), DistilDRBA `pair01`/`pair12` à 7 plans, GMFSS 6 plans.
- **Calibration** sur 4 instants, **qualité sur 7 cas hors calibration** (autres images ou autres couples), image finale
  contre le FP16 (ONNX Runtime CPU, Q/DQ émulés exactement ; TensorRT diffère à l'ordre d'accumulation près). GMFSS : la
  chaîne complète paire → splats → fusion, comme le lecteur.
- **Durées** : `tools/fp8/abab_trt.sh`, `trtexec --useCudaGraph --noDataTransfers`, 500 itérations d'échauffement, 300
  mesurées, niveau de construction 5, moteurs construits une fois, **A B A B A B (3 tours)**, médiane GPU. Les médianes
  varient avec les horloges d'une séance à l'autre (la paire FP16 va de 9,7 à 11,2 ms) : seules les comparaisons de la même
  séance comptent. Les sous-moteurs des chemins fusionnés se mesurent un par un (`abab_pieces.sh`) ; la découpe vient de
  `derive_rife_fused` / `derive_drba_fused` du lecteur.

## Mesures

### GMFSS (s05, 768×1280)

Paire FP8 hors du squelette de GMFlow (`/flownet/backbone/` en FP16 ; FeatureNet, upsampler et MetricNet en FP8, 15
convolutions sur 37), médiane GPU en ms (FP16 / FP8) :

| Tour | base | union |
|---|---|---|
| 1 | 9,88 / 8,40 (−15,0 %) | 9,87 / 8,39 (−15,0 %) |
| 2 | 9,86 / 8,37 (−15,1 %) | 9,86 / 8,43 (−14,5 %) |
| 3 | 9,89 / 8,41 (−15,0 %) | 9,82 / 8,41 (−14,3 %) |

Minimum : 9,28 → 8,04 ms (−13,4 %). Avec **toutes** les convolutions en FP8 (32 sur 37) le gain était du même ordre, mais le
squelette de GMFlow (la régression du flot) perd 12 dB au pire cas : la recherche par groupes (FeatureNet seul 49,5 dB,
upsampler 47,8, MetricNet 60,8, squelette 34,7 sur les deux cas les plus durs) l'a désigné. Qualité du graphe livré,
7 cas hors calibration :

| | PSNR moyen | PSNR pire | SSIM moyen | SSIM pire |
|---|---|---|---|---|
| base | 56,4 dB | 46,9 dB | 0,9992 | 0,9982 |
| union | 60,0 dB | 48,6 dB | 0,9996 | 0,9986 |

Contrôle sur TensorRT lui-même (cas le plus dur, t960_0, 768×1280) : les 14 sorties de la paire FP8 contre la paire FP16 du
même moteur donnent 41 à 57 dB (mouvement 41,4 à 50,3 dB, caractéristiques 47,2 à 57,3 dB, images 99 dB), dans l'ordre de
ce qu'émule ONNX Runtime (46 à 57 dB) : les PSNR de la chaîne complète ci-dessus sont ceux de l'émulation, pas de
TensorRT. Lecture réelle (`ajp play`, TensorRT, `--data-dir` jetable) : le cran Performance par défaut charge
`*_s05fp8_pair.onnx` (moteur 1280×768 construit en 46 s, 579 couches, 0 reformatage) avec la fusion FP16 d'Équilibre, sans
repli ni avertissement ; en 40 s de lecture à 720p la chaîne n'a terminé aucune paire (GMFSS ×3 ne tient pas à 720p, voir
le plan), le contrôle des sorties a donc été fait sur le moteur seul.

La fusion reste en FP16 : en FP8 (39 convolutions sur 41 pour base, 75 sur 81 pour union) elle gagne 8,6 à 9,2 % sur union
mais tombe à 36,9 dB au pire cas (mouvement fort), sous la garde ; la fusion de base n'a pas été chronométrée (même
cause, 39,1 dB sur une série moins dure). Le cran Performance de GMFSS utilise donc la paire FP8 avec la fusion FP16
(`gmfss_files`, déjà ainsi dans le code).

### DistilDRBA v2 lite

Entier FP16 / FP8, médiane (ms) : 1,275 / 1,131 (−11,3 %) ; 1,284 / 1,138 (−11,4 %) ; 1,286 / 1,126 (−12,4 %). Qualité,
7 cas : 62,0 dB moyen, 49,6 dB au pire, SSIM 0,9994 au pire (version `_s05` : 60,9 / 47,4 dB, SSIM 0,9996). **Mais** le
lecteur lit DistilDRBA en version fusionnée par défaut (quatre sous-moteurs FP16 et des noyaux CUDA : 0,92 ms contre
2,10 ms pour le réseau entier à 1088×1920), et ce chemin n'existe qu'au cran Équilibre. Choisir Performance remplace donc
0,92 ms par ≈ 1,9 ms (2,10 ms − 11 %) : un FP8 « 11 % plus rapide » qui rend la lecture deux fois plus lente. Les
sous-moteurs eux-mêmes ne gagnent rien en FP8 (bloc 0 −1,5 %, bloc 1 +8 %, bloc 2 +46 %, médianes 0,064 / 0,075 / 0,129
ms : trop petits, limités par la latence et les passes Q/DQ). Le graphe `distil_drba_v2_lite_fp8.onnx` (et `_s05`) est
publié comme fichier **facultatif** (`SHA256SUMS.facultatif`, jamais téléchargé par `fetch-models.sh`), pour un poste où
le fusionné ne se construit pas.

### RIFE

4.25 entier, médiane (ms) FP16 / FP8 : 8,33 / 7,86 (−5,7 %) ; 8,36 / 7,85 (−6,1 %) ; 8,31 / 7,91 (−4,8 %). 4.22 lite : 7,85
/ 9,98 (+27,1 %), +27,9 %, +27,6 %. Qualité : 4.25 57,8 dB moyen mais 31,3 dB au pire (SSIM 0,972) ; 4.22 lite 56,9 / 30,8 dB
(SSIM 0,975) : la garde du pire cas n'est pas tenue (avec la calibration par percentile : 27,7 et 29,9 dB). Sous-moteurs du
fusionné de 4.25 (médiane, ms, FP16 / FP8) : bloc 0 0,097 / 0,095 ; bloc 1 0,070 / 0,068 ; bloc 2 0,108 / 0,141 (+31 %) ;
bloc 3 0,179 / 0,240 (+34 %) ; bloc 4 0,778 / 0,885 (+14 %) ; somme 1,23 → 1,43 ms (+16 %). En ne gardant en FP8 que les
convolutions d'au moins 64 canaux : blocs 0 et 1 inchangés ou −1 %, blocs 2 et 3 +28 % et +26 %. Les convolutions des
blocs de RIFE sont petites (basse résolution : 96×160 pour les trois premiers) : la latence et les passes
Quantize/Dequantize les dominent, le FP8 n'y a rien à gagner. Aucun graphe FP8 de RIFE n'est publié. Au cran
Performance de RIFE le lecteur couperait de plus RIFE fusionné (1,3 ms contre 4,5 ms pour le RIFE entier à 960×576, voir
`rife-fusionne.md`).

## Dans le lecteur

- `precision.h` : les trois crans ne changent que le fichier ONNX chargé. Performance : le graphe `_fp8` à côté du modèle
  (RIFE, DistilDRBA : `tier_variant` / `sibling_tiers`) ou la paire `*_s05fp8_pair.onnx` (GMFSS : `gmfss_files`) ; absent, ou
  moteur sans FP8, Équilibre (journal : « Équilibre à la place » avec la raison). Un graphe que le moteur refuse à la
  construction ou au chargement fait reprendre toute la chaîne en Équilibre (`prepare_chain`, journal « chaîne refusée
  aux crans de précision choisis »).
- **Par défaut** : `gmfss_precision` vaut Performance (réglage par défaut et préréglage Auto de la tranche Ultra hors priorité
  Image). Les `play.toml` déjà écrits gardent leur choix ; un ancien `gmfss_quality = false` reste Équilibre. `span_precision`
  et `motion_precision` (RIFE, DistilDRBA) restent Équilibre, et l'automatique ne les passe jamais en FP8.
  Conséquence : GMFSS fusionné (`gmfss_fused`, éteint par défaut, `gmfss-fusionne.md`) ne se découpe qu'au cran Équilibre ;
  pour l'essayer il faut passer GMFSS en Équilibre (le journal dit « cran de précision hors Équilibre » sinon).
- **Réglages avancés > Fluidité** : le sélecteur de précision de RIFE et de DistilDRBA propose Performance quand le graphe
  `_fp8` est là, avec une infobulle qui donne les chiffres de ce document (RIFE : −5 % à +27 %, qualité hors garde, coupe
  RIFE fusionné ; DistilDRBA : −11 %, qualité tenue, coupe le fusionné).
- Pas de dérivation native C++ à la façon du `.span8` : SPAN a un paquet FP8 parce que ses convolutions en
  pleine résolution y gagnent 34 à 37 % avec des noyaux CUTLASS dédiés (`span-fp8-cutlass.md`). Ici le gain de GMFSS vient d'un graphe ONNX que
  TensorRT fond déjà ; le reste n'a pas de gain à dériver. Le graphe FP8 de GMFSS est calibré hors ligne sur des images
  d'anime et publié (`tools/release/publier-modeles-fp8.sh`), comme les autres graphes d'interpolation.

## Refaire

```
tools/fp8/build_interp_fp8.sh VIDEO.mkv SORTIE         # graphes FP8 + gardes de qualité (SORTIE/quality.txt)
flock ~/ajp-reports/gpu-bench.lock tools/fp8/abab_trt.sh A.onnx B.onnx "frames:1x6x768x1280" 3   # ABAB 3 tours
tools/release/publier-modeles-fp8.sh SORTIE            # essai à blanc ; --publier pour téléverser (Tatoub)
```

Fichiers publiés (release `models-interp-1` de `Ta2b/ajp-releases`, SHA-256 dans le rapport de livraison et dans
`SHA256SUMS`) : `gmfss_fortuna_base_s05fp8_pair.onnx`, `gmfss_fortuna_union_s05fp8_pair.onnx` (par défaut) ;
`distil_drba_v2_lite_fp8.onnx`, `distil_drba_v2_lite_s05_fp8.onnx` (facultatifs).
