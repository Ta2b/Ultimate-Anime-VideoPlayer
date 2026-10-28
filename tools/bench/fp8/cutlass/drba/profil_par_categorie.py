#!/usr/bin/env python3
"""Regroupe le profil par couche de trtexec (--exportProfile) de DistilDRBA par catégorie.

    python profil_par_categorie.py prof.json
"""

import collections
import json
import re
import sys

rows = [r for r in json.load(open(sys.argv[1])) if "name" in r and "averageMs" in r]
total = sum(r["averageMs"] for r in rows)
cat = collections.defaultdict(float)
for r in rows:
    n = r["name"]
    if "Grid" in n:
        k = "warp (GridSample)"
    elif re.search(r"convblock\.\d/conv/Conv", n):
        k = "ResConv fusionnées (conv + beta + saut + LeakyRelu)"
    elif re.search(r"/Conv|ConvTranspose", n) and "Reformatting" not in n:
        k = "autres convs (tête, conv0, lastconv, encodeur)"
    elif "DepthToSpace" in n:
        k = "DepthToSpace"
    elif "Resize" in n and "cast" not in n:
        k = "Resize"
    elif "Reformatting" in n:
        k = "reformatage"
    elif "cast" in n.lower():
        k = "casts"
    elif "myl" in n:
        k = "élémentaires fusionnés (myl)"
    else:
        k = "autres (Slice, Mul...)"
    cat[k] += r["averageMs"]
print(f"total du profil {total:.3f} ms (la latence réelle en graphe CUDA est 2,10 ms)")
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"{k:58s} {v:.3f} ms {100 * v / total:5.1f} %")
