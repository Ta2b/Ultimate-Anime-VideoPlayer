#!/usr/bin/env python3
"""Écart de couleur ΔE_ITP entre deux instantanés 16 bits PQ BT.2020 (étape 2.4 du plan NVIDIA).

ΔE_ITP = 720 · ‖(ΔI, ΔT, ΔP)‖ (ITU-R BT.2124), avec T = Ct / 2 et P = Cp ; 1 est à peu près le seuil
visible. Rendu : moyenne, p95, p99, p99,9 et maximum sur tous les pixels.

  de_itp.py A.ppm B.ppm [--max-p95 0.2] [--heatmap carte.ppm]
      code retour 1 si p95 dépasse --max-p95 ; --heatmap écrit une carte 8 bits (noir 0, blanc >= 2).

Les PPM viennent de `ajp play … --snapshot=FICHIER.pq.ppm --snapshot-at=S` sur une sortie HDR10
(placebo_snapshot.cpp : P6, maximum 65535, poids fort d'abord). A : `--legacy-grade=split`, B : par défaut.
"""
import argparse
import sys

import numpy as np


def read_ppm16(path):
    with open(path, "rb") as f:
        data = f.read()
    fields = []
    i = 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    i += 1
    if fields[0] != b"P6" or int(fields[3]) != 65535:
        raise ValueError(f"{path} : PPM 16 bits (P6, 65535) attendu")
    w, h = int(fields[1]), int(fields[2])
    pixels = np.frombuffer(data, dtype=">u2", count=w * h * 3, offset=i)
    return pixels.reshape(h, w, 3).astype(np.float64) / 65535.0


def pq_to_linear(e):
    """Codes PQ (0 à 1) → lumière / 10 000 nits."""
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    p = np.power(np.clip(e, 0.0, 1.0), 1.0 / m2)
    return np.power(np.maximum(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1)


def linear_to_pq(y):
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    q = np.power(np.clip(y, 0.0, 1.0), m1)
    return np.power((c1 + c2 * q) / (1.0 + c3 * q), m2)


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
    parser.add_argument("--heatmap", default=None)
    args = parser.parse_args(argv)
    a = read_ppm16(args.a)
    b = read_ppm16(args.b)
    if a.shape != b.shape:
        print(f"tailles différentes : {a.shape[1]}x{a.shape[0]} et {b.shape[1]}x{b.shape[0]}", file=sys.stderr)
        return 2
    de = delta_e(a, b)
    s = summary(de)
    print("ΔE_ITP : " + ", ".join(f"{k} {v:.3f}" for k, v in s.items()))
    if args.heatmap:
        write_heatmap(args.heatmap, de)
    if args.max_p95 is not None and s["p95"] > args.max_p95:
        print(f"p95 {s['p95']:.3f} au-dessus de {args.max_p95}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
