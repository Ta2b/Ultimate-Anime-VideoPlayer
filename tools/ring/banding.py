#!/usr/bin/env python3
"""Banding d'instantanés 16 bits (étape 5.5c du plan NVIDIA).

Compte les paliers d'un dégradé sombre : sur chaque ligne, une plage d'au moins --run pixels de même code
RGB, dans la zone sombre (luminance BT.2020 sous --dark), bordée des deux côtés d'un pas d'un seul code
10 bits (au plus 64 sur 65 535). Rendu : paliers par mégapixel.

  banding.py A.ppm [B.ppm …]                 paliers de chaque image
  banding.py --compare A.ppm B.ppm [--max-ratio 1.10]
      rapport B / A (A : anneau RGBA16, B : RGB10A2) ; code retour 1 au-delà de --max-ratio.

Les PPM viennent de `ajp play … --snapshot=FICHIER --snapshot-at=S`, sur les plans sombres de
data/calibration/calibration.mkv, une fois avec l'anneau par défaut et une fois avec --ring-rgb10a2.
"""
import argparse
import sys

import numpy as np

STEP_10BIT = 65535.0 / 1023.0  # un code 10 bits, en codes 16 bits


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
    if fields[0] != b"P6":
        raise ValueError(f"{path} : PPM binaire (P6) attendu")
    width, height, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    dtype = ">u2" if maxval > 255 else "u1"
    pixels = np.frombuffer(data, dtype=dtype, count=width * height * 3, offset=i).reshape(height, width, 3)
    scale = 65535.0 / maxval
    return pixels.astype(np.float64) * scale


def bands(image, run=16, dark=0.10):
    height, width, _ = image.shape
    luma = (0.2627 * image[..., 0] + 0.6780 * image[..., 1] + 0.0593 * image[..., 2]) / 65535.0
    count = 0
    for y in range(height):
        row = image[y]
        # Plages de codes identiques sur la ligne.
        change = np.any(row[1:] != row[:-1], axis=1)
        starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
        ends = np.concatenate((starts[1:], [width]))
        for k in range(1, len(starts) - 1):
            a, b = starts[k], ends[k]
            if b - a < run or luma[y, a:b].max() >= dark:
                continue
            here = row[a]
            left = row[a - 1]
            right = row[b]
            dl = np.abs(here - left).max()
            dr = np.abs(here - right).max()
            if 0 < dl <= STEP_10BIT + 0.5 and 0 < dr <= STEP_10BIT + 0.5:
                count += 1
    return count * 1e6 / (width * height)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--compare", action="store_true", help="deux images : rapport B / A")
    parser.add_argument("--max-ratio", type=float, default=1.10)
    parser.add_argument("--run", type=int, default=16, help="longueur minimale d'un palier (pixels)")
    parser.add_argument("--dark", type=float, default=0.10, help="luminance maximale de la zone sombre")
    args = parser.parse_args()
    values = [bands(read_ppm16(path), args.run, args.dark) for path in args.images]
    for path, value in zip(args.images, values):
        print(f"{path}\t{value:.1f} paliers par mégapixel")
    if args.compare:
        if len(values) != 2:
            parser.error("--compare demande deux images")
        ratio = values[1] / values[0] if values[0] > 0 else (1.0 if values[1] == 0 else float("inf"))
        print(f"rapport B / A : {ratio:.3f} (porte : {args.max_ratio})")
        return 1 if ratio > args.max_ratio else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
