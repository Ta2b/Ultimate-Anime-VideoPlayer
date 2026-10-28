# GMFSS Union et DistilDRBA : où va le temps (RTX 4070 Ti, TensorRT 11.3, 28 octobre 2026)

Mesures au `trtexec` (graphes ONNX FP16 de `~/.local/share/AnimeJaNai-HDR-linux/animejanai/rife/`, `--noDataTransfers`, sous le
verrou GPU ; le bureau garde ~35 % du GPU, donc minimums et médianes). But : décider si les optimisations prévues au plan (« calculs
indépendants du temps déplacés dans le graphe paire » pour GMFSS Union, « warps hors du graphe » pour DistilDRBA) valent une PR.

## DistilDRBA v2 lite
| Entrée (pair01, pair12) | Passe | Où va le temps |
|---|---|---|
| 768×1280 (720p de travail) | **0,86 ms** (min 0,82) | 131 couches, chacune < 0,05 ms ; convolutions, redimensionnements, conversions |
| 2176×3840 (2160p) | 11,7 ms (min 11,0) | les warps réécrits par Myelin (« MoveGrid… ») ≈ 4 ms (≈ 30 %), conversions et reformatages ≈ 5 ms, convolutions ≈ 3 ms |

**Décision : pas de PR.** À la résolution de travail par défaut (720p), une passe coûte moins d'une milliseconde : sortir les
warps du graphe gagnerait au mieux 0,2 à 0,3 ms par image créée, contre 4 ms de SPAN et 1 ms de RTX HDR. Le gain n'existe qu'à
2160p de travail, une configuration que les préréglages n'emploient pas. Le ré-export PyTorch et les noyaux CUDA de warp seraient un
risque pour la qualité (le script d'export compare au réseau d'origine) sans bénéfice mesurable à la tranche 3.

## GMFSS Union (demi-flot `_s05`), graphe « fusion »
Entrées 720p : `half` 384×640, `warped0_1/1_1` 67×384×640, `warped*_2` 128×192×320, `warped*_3` 192×96×160, `time` 1×1×1×1.

| Mesure | Résultat |
|---|---|
| Passe complète, médiane (min) | **12,1 ms** (11,5) |
| Convolutions (GridNet et IFNet) | **9,0 ms**, soit ≈ 80 % |
| Conversions, pointwise Myelin | 2,1 ms |
| GridSample et redimensionnements | 0,16 ms |
| Les plus lourdes couches | `residual_model_tail/upsample` 0,91 ms, un reformatage avant `conv_last` 0,58 ms, `upsample.0/Conv` 0,53 ms |

**Calculs indépendants du temps dans la fusion : aucun.** Le parcours du graphe (1115 nœuds) depuis les entrées `time` et `warped*`
laisse hors dépendance 0 convolution, 0 GridSample, 0 redimensionnement : seulement 340 constantes et 33 opérations de forme. Le
premier bloc de l'IFNet concatène déjà l'instant `time` à ses entrées, et GridNet lit des sources splattées à cet instant. Il n'y a
donc rien à déplacer vers le graphe « paire » (qui ne tourne qu'une fois par paire) : la proposition ne s'applique pas à ce graphe tel
qu'il est exporté.

### Options de construction de TensorRT (même graphe, même entrée)
| Option de `trtexec` | Médiane (min) |
|---|---|
| par défaut | 12,09 ms (11,48) |
| `--builderOptimizationLevel=5` | 12,24 ms (11,46) |
| `--builderOptimizationLevel=5 --tilingOptimizationLevel=3` | 12,72 ms (11,94) |
| `--builderOptimizationLevel=2` | 13,62 ms (12,47) |

Le réglage par défaut est le bon : un niveau plus haut ne gagne rien (dans le bruit), le tuilage coûte, un niveau plus bas perd 12 %.

## Ce qui resterait à essayer (non fait, à décider)
- Remplacer le `upsample` final (ConvTranspose ≈ 0,9 ms + reformatage ≈ 0,6 ms + `conv_last` ≈ 0,4 ms, ≈ 16 % de la fusion) : c'est un
  changement du réseau, donc de la qualité, à valider sur images de référence, pas une optimisation d'export.
- GMFSS Union reste le choix du niveau Ultra : à 12 ms la fusion tient à ×2 ou ×3 sur un écran 165 Hz, pas au-delà.
