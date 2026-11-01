#!/usr/bin/env python3
"""Résume adaptive-log.tsv (écrit par le lecteur avec « Record scene statistics ») série par série : la luminosité
moyenne (à comparer à kSeriesRef = 0,30 de chain_hdr.cpp), la part du temps passée à chaque position t du grade
adaptatif, et les bornes des statistiques de scène. Aide à choisir les seuils (adaptive_lo/hi, kSeriesRef, bs).

    tools/adaptive/summarize_log.py ~/.local/share/ajp/ajp/adaptive-log.tsv [...]
"""
import csv
import statistics
import sys
from collections import defaultdict


def quantile(values, q):
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def main(paths):
    series = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    series[row["fichier"]].append({k: float(v) for k, v in row.items() if k != "fichier"})
                except (TypeError, ValueError):
                    continue  # ligne coupée ou d'un autre schéma
    for name, rows in series.items():
        mean = [r["moyenne"] for r in rows]
        t = [r["t"] for r in rows]
        print(f"{name[:70]}")
        print(f"  {len(rows)} s ; luminosité moyenne {statistics.fmean(mean):.3f} "
              f"(p10 {quantile(mean, .1):.3f}, p50 {quantile(mean, .5):.3f}, p90 {quantile(mean, .9):.3f}) ; "
              f"référence du lecteur 0,30")
        print(f"  t : sombre (<0,1) {sum(x < .1 for x in t) / len(t):.0%}, milieu {sum(.1 <= x <= .9 for x in t) / len(t):.0%}, "
              f"clair (>0,9) {sum(x > .9 for x in t) / len(t):.0%}")
        black = [r["part_noir"] for r in rows]
        bright = [r["part_clair"] for r in rows]
        print(f"  part de noir : moyenne {statistics.fmean(black):.4f} ; part claire : moyenne {statistics.fmean(bright):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
