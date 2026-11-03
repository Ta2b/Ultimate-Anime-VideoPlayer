#!/usr/bin/env python3
"""Mesures de référence (étape 1) : 18 plans de calibration.mkv x configurations, contre « sansgrade ».
Écrit mesures-reference.tsv ; cadres de visage (x0,y0,x1,y1, cadre 2560x1440) posés à la main sur les plans
2, 3, 8, 14. À lancer dans le dossier qui contient frames/ ; configurations en arguments (par défaut les
quatre de l'étape 1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gref  # noqa: E402

FACES = {2: (1540, 380, 1920, 740), 3: (1040, 520, 1520, 800), 8: (880, 600, 1280, 940), 14: (1040, 540, 1600, 1000)}
CONFIGS = sys.argv[1:] or ["sansgrade", "actuel", "sanspile", "ancien-sanspile"]
COLS = [
    "plan", "config", "moyenne_nits", "moyenne/ref", "p99_9_nits", "p99_9/ref", "part_>400", "part_>700",
    "visage_mediane", "visage_p95", "visage_mediane/ref", "visage_grain_dE",
]  # fmt: skip

with open("mesures-reference.tsv", "w") as out:
    out.write("\t".join(COLS) + "\n")
    for k in range(1, 19):
        ref = f"frames/plan{k:02d}-sansgrade.pq.ppm"
        for c in CONFIGS:
            p = f"frames/plan{k:02d}-{c}.pq.ppm"
            if not os.path.exists(p):
                continue
            m = gref.mesure(p, None if c == "sansgrade" else ref, FACES.get(k))
            row = [str(k), c] + [("%.3f" % m[x]) if x in m else "" for x in COLS[2:]]
            out.write("\t".join(row) + "\n")
            out.flush()
for k, f in FACES.items():
    for c in CONFIGS:
        p = f"frames/plan{k:02d}-{c}.pq.ppm"
        if os.path.exists(p):
            gref.fausses(p, f"fausses-plan{k:02d}-{c}.png", [f])
print("ok")
