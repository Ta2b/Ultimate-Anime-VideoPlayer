#!/usr/bin/env python3
"""PSNR / SSIM de sorties trtexec (--exportOutput) contre la sortie FP16.
Usage : quality.py ref.json autre.json [autre2.json ...]   (images sorties clippées dans [0, 1]).
SSIM : fenêtre uniforme 7x7 (intégrale), moyenne sur les 3 canaux ; PSNR sur les 3 canaux, pic 1."""
import json, sys
import numpy as np

def load(p):
    d = json.load(open(p))
    o = d[0] if isinstance(d, list) else d
    v = np.asarray(o["values"], np.float32)
    dims = [int(x) for x in o["dimensions"].split("x")]
    return np.clip(v.reshape(dims)[0], 0, 1)

def box(x, k=7):
    c = np.cumsum(np.cumsum(np.pad(x, ((0, 0), (1, 0), (1, 0))), 1), 2, dtype=np.float64)
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / (k * k)

def ssim(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = box(a), box(b)
    va, vb, cab = box(a * a) - ma * ma, box(b * b) - mb * mb, box(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())

ref = load(sys.argv[1])
for p in sys.argv[2:]:
    x = load(p)
    mse = float(((ref - x) ** 2).mean())
    print(f"{p}: PSNR {10 * np.log10(1 / max(mse, 1e-12)):.2f} dB  SSIM {ssim(ref, x):.5f}  erreur max {np.abs(ref - x).max():.4f}")
