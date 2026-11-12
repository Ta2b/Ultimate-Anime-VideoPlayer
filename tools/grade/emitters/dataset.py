#!/usr/bin/env python3
"""Le jeu de données du réseau des émetteurs (méga plan A6).

Des images tirées des vidéos (ffmpeg, une toutes les N secondes, converties en RGB linéaire BT.2020 sur
l'échelle du lecteur : le blanc SDR à 203 nits), découpées en tuiles de 256 × 256, chacune avec son
étiquette : la carte des émetteurs v2 (labels.py, la même que src/render/emitters.cpp) au 1/8. Les tuiles
vont dans DOSSIER/tiles-XXXXX.npz (rgb : 3 × 256 × 256 en float16, label : 32 × 32 en float16).

    dataset.py --every 4 --out DOSSIER video1.mkv video2.mkv …
    dataset.py --self-test        (sans vidéo : une image synthétique, la forme des tuiles)

Une correction à la main (un émetteur manqué, une peau prise pour une lumière) s'écrit dans
DOSSIER/corrections.tsv : « tuile, x8, y8, valeur » ; elle remplace l'étiquette du bloc à l'entraînement.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import hdr  # noqa: E402
import labels  # noqa: E402

TILE = 256
SDR_WHITE = 203.0
# BT.709 → BT.2020, pour les sources SDR (le lecteur fait la même conversion avant le grade).
BT709_TO_2020 = hdr.BT709_TO_2020.astype(np.float32)


def srgb_to_nits(code):
    """Un code SDR 0 à 1 vers des nits, gamma 2,4 (BT.1886), le blanc à 203."""
    return SDR_WHITE * np.power(np.clip(code, 0.0, 1.0), 2.4)


def frames(path, every, width=1920, height=1080):
    """Les images de la vidéo, une toutes les `every` secondes, en RGB 8 bits (ffmpeg)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vf",
           f"fps=1/{every},scale={width}:{height}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        size = width * height * 3
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def tiles_of(rgb8):
    """Les tuiles d'une image 8 bits : (rgb linéaire 3 × 256 × 256 en nits, étiquette 32 × 32)."""
    lin = srgb_to_nits(rgb8.astype(np.float32) / 255.0) @ BT709_TO_2020.T
    y, chroma = labels.linear_to_inputs(lin)
    full = labels.emitter_map_v2(y, chroma)
    h, w = y.shape
    for ty in range(0, h - TILE + 1, TILE):
        for tx in range(0, w - TILE + 1, TILE):
            block = full[ty:ty + TILE, tx:tx + TILE]
            label = block.reshape(TILE // labels.BLOCK, labels.BLOCK, TILE // labels.BLOCK,
                                  labels.BLOCK).max(axis=(1, 3))
            rgb = lin[ty:ty + TILE, tx:tx + TILE].transpose(2, 0, 1)
            yield rgb.astype(np.float16), label.astype(np.float16)


def write(out, index, rgb, label):
    np.savez_compressed(os.path.join(out, f"tiles-{index:05d}.npz"), rgb=rgb, label=label)


def self_test():
    img = np.full((512, 512, 3), 20, dtype=np.uint8)
    img[200:240, 300:340] = 250  # un halo
    found = list(tiles_of(img))
    ok = len(found) == 4 and all(r.shape == (3, TILE, TILE) and l.shape == (32, 32) for r, l in found)
    lit = float(found[1][1].max())  # la tuile en haut à droite contient le halo
    ok = ok and lit > 0.9 and float(found[2][1].max()) < 0.05
    print(f"dataset.py : {len(found)} tuiles, halo {lit:.2f} : {'ok' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*")
    parser.add_argument("--every", type=float, default=4.0)
    parser.add_argument("--out", default="emitters-data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.videos:
        parser.error("au moins une vidéo")
    os.makedirs(args.out, exist_ok=True)
    index = 0
    for video in args.videos:
        for rgb8 in frames(video, args.every):
            for rgb, label in tiles_of(rgb8):
                write(args.out, index, rgb, label)
                index += 1
        print(f"{video} : {index} tuiles en tout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
