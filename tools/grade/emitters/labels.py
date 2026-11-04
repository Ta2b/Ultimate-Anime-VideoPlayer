#!/usr/bin/env python3
"""Les étiquettes du réseau des émetteurs (méga plan A6) : la carte des émetteurs v2 de
src/render/emitters.cpp (A5), portée en numpy.

Même calcul que le C++ : moyenne des blocs 8 × 8 (luminance, saturation), couronne des blocs à une distance
de Tchebychev de 2 à 4 (bords répétés), centre-couronne en diaphragmes, porte blanche ou colorée,
étincelle. `labels.py --self-test` compare à des valeurs relevées sur la version C++.
"""
import sys

import numpy as np

BLOCK = 8
RING_IN = 2
RING_OUT = 4


def smooth(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def block_means(values, bh, bw):
    """La moyenne de chaque bloc 8 × 8 (les blocs du bord peuvent être incomplets)."""
    h, w = values.shape
    sums = np.zeros((bh, bw), dtype=np.float64)
    counts = np.zeros((bh, bw), dtype=np.float64)
    ys = np.arange(h) // BLOCK
    xs = np.arange(w) // BLOCK
    np.add.at(sums, (ys[:, None], xs[None, :]), values)
    np.add.at(counts, (ys[:, None], xs[None, :]), 1.0)
    return (sums / counts).astype(np.float32)


def ring_means(level):
    bh, bw = level.shape
    pad = np.pad(level, RING_OUT, mode="edge").astype(np.float64)
    total = np.zeros_like(level, dtype=np.float64)
    count = 0
    for j in range(-RING_OUT, RING_OUT + 1):
        for i in range(-RING_OUT, RING_OUT + 1):
            if max(abs(i), abs(j)) < RING_IN:
                continue
            total += pad[RING_OUT + j:RING_OUT + j + bh, RING_OUT + i:RING_OUT + i + bw]
            count += 1
    return (total / count).astype(np.float32)


def emitter_map_v2(nits, chroma, white_nits=203.0):
    """nits, chroma : tableaux h × w (luminance linéaire en nits, saturation (max − min) / max). Rend h × w."""
    nits = np.maximum(np.asarray(nits, dtype=np.float32), 0.0)
    chroma = np.clip(np.asarray(chroma, dtype=np.float32), 0.0, 1.0)
    h, w = nits.shape
    bh = (h + BLOCK - 1) // BLOCK
    bw = (w + BLOCK - 1) // BLOCK
    level = block_means(nits, bh, bw)
    sat = block_means(chroma, bh, bw)
    ring = ring_means(level)
    white = max(float(white_nits), 1.0)
    block = np.maximum(level, 1e-3)
    stops = np.log2(block / np.maximum(ring, 1e-3))
    iso = smooth(0.5, 1.5, stops)
    gate = np.maximum(smooth(0.55 * white, 0.95 * white, block),
                      smooth(0.30 * white, 0.60 * white, block) * smooth(0.45, 0.75, sat))
    per_block = iso * gate
    up = lambda b: np.repeat(np.repeat(b, BLOCK, axis=0), BLOCK, axis=1)[:h, :w]
    block_px = up(block)
    spark = smooth(1.0, 1.5, np.log2(np.maximum(nits, 1e-3) / block_px)) * smooth(0.6 * white, white, nits)
    return np.clip(np.maximum(up(per_block), spark), 0.0, 1.0).astype(np.float32)


def linear_to_inputs(rgb_nits):
    """Une image RGB linéaire en nits (h × w × 3, BT.2020) : la luminance et la saturation du lecteur."""
    rgb = np.maximum(np.asarray(rgb_nits, dtype=np.float32), 0.0)
    y = rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    chroma = np.where(hi > 1e-4, (hi - lo) / np.maximum(hi, 1e-4), 0.0)
    return y, chroma


# Relevées sur la version C++ (src/render/emitters.cpp) pour l'image de self_test_image().
CPP_REFERENCE = [
    ((28, 20), 1.000000),
    ((21, 13), 0.972793),
    ((35, 27), 0.972793),
    ((19, 20), 0.000000),
    ((37, 20), 0.000000),
    ((48, 12), 0.901919),
    ((41, 9), 0.901919),
    ((50, 40), 1.000000),
    ((51, 40), 0.000000),
    ((5, 5), 0.000000),
    ((60, 44), 0.000000),
    ((30, 30), 0.000000),
    ((44, 20), 0.000000),
    ((16, 16), 0.000000),
]


def self_test_image():
    nits = np.zeros((48, 64), dtype=np.float32)
    chroma = np.zeros((48, 64), dtype=np.float32)
    for y in range(48):
        for x in range(64):
            if x == 50 and y == 40:
                n = 210.0
            elif 20 <= x < 36 and 12 <= y < 28:
                n = 195.0
            elif 40 <= x < 56 and 8 <= y < 16:
                n = 110.0
            else:
                n = 5.0 + 2.0 * (x % 7)
            nits[y, x] = n
            chroma[y, x] = 0.9 if (40 <= x < 56 and 8 <= y < 16) else 0.1 + 0.01 * (y % 5)
    return nits, chroma


def self_test():
    m = emitter_map_v2(*self_test_image())
    worst = max(abs(float(m[y, x]) - v) for (x, y), v in CPP_REFERENCE)
    print(f"labels.py : écart maximal au C++ {worst:.2e} sur {len(CPP_REFERENCE)} points")
    return 0 if worst <= 1e-4 else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    print(__doc__)
