#!/usr/bin/env python3
"""Écart de couleur ΔE_ITP entre deux instantanés 16 bits PQ BT.2020 (étape 2.4 du plan NVIDIA).

ΔE_ITP = 720 · ‖(ΔI, ΔT, ΔP)‖ (ITU-R BT.2124), avec T = Ct / 2 et P = Cp ; 1 est à peu près le seuil
visible. Rendu : moyenne, p95, p99, p99,9 et maximum sur tous les pixels.

  de_itp.py A.ppm B.ppm [--max-p95 0.2] [--max-p999 3] [--heatmap carte.ppm] [--json]
      code retour 1 si p95 dépasse --max-p95 ou p99,9 --max-p999 ; --heatmap écrit une carte 8 bits (noir 0,
      blanc >= 2) ; --json écrit le résumé en JSON sur une ligne (tools/grade/compare_shots.sh).

Les PPM viennent de `ajp play … --snapshot=FICHIER.pq.ppm --snapshot-at=S` sur une sortie HDR10
(placebo_snapshot.cpp : P6, maximum 65535, poids fort d'abord). A : `--legacy-grade=split`, B : par défaut.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from hdr import linear_to_pq, pq_to_linear, read_ppm16  # noqa: E402,F401  (check_hdr.py les reprend d'ici)


RGB_TO_LMS = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]]) / 4096.0
LMS_TO_ICTCP = np.array([[2048, 2048, 0], [6610, -13613, 7003], [17933, -17390, -543]]) / 4096.0


def itp(codes):
    lms = pq_to_linear(codes) @ RGB_TO_LMS.T
    ictcp = linear_to_pq(lms) @ LMS_TO_ICTCP.T
    return ictcp * np.array([1.0, 0.5, 1.0])  # T = Ct / 2


def delta_e(a, b):
    return 720.0 * np.linalg.norm(itp(a) - itp(b), axis=-1)


def summary(de):
    flat = de.ravel()
    return {
        "moyenne": float(flat.mean()),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "p99.9": float(np.percentile(flat, 99.9)),
        "max": float(flat.max()),
    }


def write_heatmap(path, de):
    v = np.clip(de / 2.0, 0.0, 1.0)
    img = (v * 255.0 + 0.5).astype(np.uint8)
    h, w = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.repeat(img[:, :, None], 3, axis=2).tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--max-p95", type=float, default=None)
    parser.add_argument("--max-p999", type=float, default=None)
    parser.add_argument("--heatmap", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    a = read_ppm16(args.a)
    b = read_ppm16(args.b)
    if a.shape != b.shape:
        print(f"tailles différentes : {a.shape[1]}x{a.shape[0]} et {b.shape[1]}x{b.shape[0]}", file=sys.stderr)
        return 2
    de = delta_e(a, b)
    s = summary(de)
    if args.json:
        print(json.dumps(s))
    else:
        print("ΔE_ITP : " + ", ".join(f"{k} {v:.3f}" for k, v in s.items()))
    if args.heatmap:
        write_heatmap(args.heatmap, de)
    if args.max_p95 is not None and s["p95"] > args.max_p95:
        print(f"p95 {s['p95']:.3f} au-dessus de {args.max_p95}", file=sys.stderr)
        return 1
    if args.max_p999 is not None and s["p99.9"] > args.max_p999:
        print(f"p99,9 {s['p99.9']:.3f} au-dessus de {args.max_p999}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
