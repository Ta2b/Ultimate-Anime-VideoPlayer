# TemporalFix natif : relevés

Relevés du chantier TemporalFix en noyaux CUDA (`--tfix-native`). L'état actuel du code est décrit dans
[`docs/tfix-natif.md`](../../tfix-natif.md) ; ce dossier garde les chiffres bruts et les notes datées.

| Fichier | Contenu |
|---|---|
| [`plafonds.md`](plafonds.md) | plafond théorique de chaque morceau sur la RTX 4070 Ti, contre le mesuré (19 novembre 2026) |
| [`fiche-chantier.md`](fiche-chantier.md) | fiche de départ du chantier, historique |
| `kern-natif-0bd175e4.csv` | nsys `cuda_gpu_kern_sum`, lecture 30 s en demi-taille, natif avant la colle en 16 octets |
| `kern-trt-0bd175e4.csv` | le même profil avec TensorRT |
| `kern-tete-fondue-4warps.csv` | le même profil avec la tête fondue (4 warps) |

Les plafonds se recalculent avec [`tools/bench/tfix_native/plafonds.py`](../../../tools/bench/tfix_native/plafonds.py).
