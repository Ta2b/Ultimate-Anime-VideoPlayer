#!/usr/bin/env python3
"""Simulation du grade du contrat v2 sur une liste de plans (chantier 8.11), sans GPU ni lecteur.

Chaque plan passe par la courbe (curve.py, le portage de src/render/curve.cpp) comme le planificateur
(src/app/play/grade_planner.cpp) la prépare : APL du plan contre la médiane de la série, limiteur ABL à la
surface claire du plan, budget 0,9 × MaxFALL. Rendu par plan : E, T, pic et APL affichées prédites, et les
critères de 8.11 que la courbe seule décide (pic ≤ C_pk + 0,5 %, APL ≤ 0,9 × MaxFALL). Code retour 1 si un
plan les manque. Les critères sur les images (teinte, peau, ΔE_ITP, coupes) : tools/grade/check_hdr.py.

  simulate.py PLANS.tsv --peak 800 --max-fall 250 [--abl 0.02:800,0.10:620,0.25:420,0.50:300,1:250] [--e 0]
  simulate.py --synthetic 30 --peak 800 --max-fall 250

PLANS.tsv : nom, APL (nits), maximum (nits), surface claire (0 à 1), puis l'histogramme de luminance en 32
parts séparées par des virgules (ShotLight::lum_hist) ; « # » commente. Tout en nits du contenu (sous K).
"""
import argparse
import math
import random
import statistics
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # curve.py, à côté (python -I ne l'ajoute pas)
import curve  # noqa: E402

BUDGET_SHARE = 0.9
PEAK_SLACK = 1.005


def parse_abl(text, peak):
    if not text:
        return [(0.02, peak), (1.0, peak)]
    points = []
    for item in text.split(","):
        area, nits = item.split(":")
        points.append((float(area), float(nits)))
    return sorted(points)


def abl_at(points, area):
    if area <= points[0][0]:
        return points[0][1]
    for (a0, n0), (a1, n1) in zip(points, points[1:]):
        if area <= a1:
            return n0 + (area - a0) / (a1 - a0) * (n1 - n0)
    return points[-1][1]


def read_shots(path):
    shots = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, apl, top, bright, hist = line.split("\t")
            parts = [float(x) for x in hist.split(",")]
            if len(parts) != curve.HIST_BINS:
                raise ValueError(f"{name} : {curve.HIST_BINS} parts d'histogramme attendues")
            shots.append({"name": name, "apl": float(apl), "max": float(top), "bright": float(bright),
                          "hist": parts})
    return shots


def synthetic(count, seed=8):
    """Des plans d'anime plausibles : la plupart vers 20 à 60 nits, quelques sombres et quelques éclats."""
    rng = random.Random(seed)
    shots = []
    for i in range(count):
        center = rng.choice([0.35, 0.45, 0.5, 0.55, 0.6, 0.7])  # code PQ du gros de l'image
        width = rng.uniform(0.04, 0.12)
        hist = [math.exp(-(((b + 0.5) / curve.HIST_BINS - center) / width) ** 2) for b in range(curve.HIST_BINS)]
        total = sum(hist)
        hist = [h / total for h in hist]
        if rng.random() < 0.3:  # un éclat
            hist[-6] += 0.01
            hist = [h / 1.01 for h in hist]
        apl = sum(h * curve.nits((b + 0.5) / curve.HIST_BINS) for b, h in enumerate(hist))
        top = max(curve.nits(max(b for b, h in enumerate(hist) if h > 1e-4) / curve.HIST_BINS + 0.03), 203.0)
        bright = sum(h for b, h in enumerate(hist) if curve.nits((b + 0.5) / curve.HIST_BINS) > 203.0)
        shots.append({"name": f"plan-{i + 1:02d}", "apl": apl, "max": top, "bright": bright, "hist": hist})
    return shots


def simulate(shots, peak, max_fall, abl, user_e):
    series = statistics.median(s["apl"] for s in shots)
    budget = BUDGET_SHARE * max_fall
    rows = []
    for s in shots:
        c = curve.plan_curve(curve.CurveInput(
            shot_apl_nits=s["apl"], series_apl_nits=series, user_e=user_e, content_max_nits=s["max"],
            panel_peak_nits=peak, abl_limit_nits=abl_at(abl, max(s["bright"], abl[0][0])),
            apl_budget_nits=budget, lum_hist=s["hist"]))
        shown_peak = curve.map_luminance(c, s["max"])
        shown_apl = curve.predicted_apl(c, s["hist"])
        floor = c.knee_out * 1.05  # T ne descend jamais sous le genou : le budget peut alors manquer
        ok_peak = shown_peak <= peak * PEAK_SLACK
        ok_apl = shown_apl <= budget * 1.001 or c.top_out <= floor + 1e-6
        rows.append((s["name"], c.e, c.top_out, shown_peak, shown_apl, ok_peak, ok_apl))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shots", nargs="?")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--peak", type=float, required=True, help="C_pk, nits du contenu")
    ap.add_argument("--max-fall", type=float, required=True, help="MaxFALL, nits du contenu")
    ap.add_argument("--abl", default="", help="surface:nits,… (nits du contenu)")
    ap.add_argument("--e", type=float, default=0.0, help="décalage E de l'utilisateur")
    args = ap.parse_args()
    if bool(args.shots) == bool(args.synthetic):
        ap.error("un fichier de plans ou --synthetic N")
    shots = read_shots(args.shots) if args.shots else synthetic(args.synthetic)
    rows = simulate(shots, args.peak, args.max_fall, parse_abl(args.abl, args.peak), args.e)
    print("plan\tE\tT\tpic\tAPL\tverdict")
    failed = 0
    for name, e, top, shown_peak, shown_apl, ok_peak, ok_apl in rows:
        verdict = "ok" if ok_peak and ok_apl else ("pic" if not ok_peak else "APL")
        failed += verdict != "ok"
        print(f"{name}\t{e:+.3f}\t{top:.0f}\t{shown_peak:.1f}\t{shown_apl:.1f}\t{verdict}")
    print(f"# {len(rows) - failed}/{len(rows)} plans tiennent pic ≤ C_pk + 0,5 % et APL ≤ 0,9 × MaxFALL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
