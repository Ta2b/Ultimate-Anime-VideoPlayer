#!/usr/bin/env python3
"""Banding d'instantanés 16 bits (étape 5.5c du plan NVIDIA).

Compte les paliers d'un dégradé sombre : sur chaque ligne, une plage d'au moins --run pixels de même code
RGB, dans la zone sombre (luminance BT.2020 des codes PQ sous le seuil --dark, en nits), bordée des deux côtés
d'un pas d'un seul code 10 bits (au plus 64 sur 65 535). Rendu : paliers par mégapixel.

Le seuil est en nits (B46-04) : les dégradés sombres se voient entre 0,5 et 20 nits, d'où 20 nits par défaut
(code PQ 0,36 environ). --dark-pq prend un code PQ à la place (l'ancien --dark 0,10 valait 0,33 nit : il
excluait presque toute la plage utile).

  banding.py A.ppm [B.ppm …]                 paliers de chaque image
  banding.py --compare A.ppm B.ppm [--max-ratio 1.10]
      rapport B / A (A : anneau RGBA16, B : RGB10A2) ; code retour 1 au-delà de --max-ratio.

Les PPM viennent de `ajp play … --snapshot=FICHIER --snapshot-at=S`, sur les plans sombres de
data/calibration/calibration.mkv, une fois avec l'anneau par défaut et une fois avec --ring-rgb10a2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import hdr  # noqa: E402

DARK_NITS = 20.0  # la zone sombre : sous ce niveau, un palier d'un code 10 bits se voit

STEP_10BIT = 65535.0 / 1023.0  # un code 10 bits, en codes 16 bits


def read_ppm16(path):
    """Codes 16 bits (flottants 0 à 65535) d'un PPM binaire de tout maximum."""
    return hdr.read_ppm16(path, strict16=False) * 65535.0


def dark_code(nits=DARK_NITS):
    """Le code PQ (0 à 1) d'un niveau en nits : le seuil de bands()."""
    return float(hdr.nits_to_pq(nits))


def bands(image, run=16, dark=None):
    """Paliers par mégapixel ; dark : seuil en code PQ (0 à 1), DARK_NITS par défaut."""
    dark = dark_code() if dark is None else dark
    height, width, _ = image.shape
    # La luminance BT.2020 se pondère en lumière linéaire (nits), puis se recode en PQ pour se comparer au seuil :
    # pondérer les codes PQ eux-mêmes écartait la sélection de la zone visée sur les pixels colorés (C46-10).
    luma = hdr.linear_to_pq(hdr.pq_to_linear(image / 65535.0) @ hdr.LUMA_2020)
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
    parser.add_argument("--dark", type=float, default=DARK_NITS, help="niveau maximal de la zone sombre, en nits (20)")
    parser.add_argument("--dark-pq", type=float, default=None, help="le même seuil en code PQ (0 à 1), prioritaire")
    args = parser.parse_args()
    dark = args.dark_pq if args.dark_pq is not None else dark_code(args.dark)
    values = [bands(read_ppm16(path), args.run, dark) for path in args.images]
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
