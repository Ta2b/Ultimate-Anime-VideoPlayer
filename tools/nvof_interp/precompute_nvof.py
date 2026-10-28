#!/usr/bin/env python3
"""Calcule une fois les flux NVOF de chaque triplet (frame1 → frame3 et retour), écrits en
nvof.npy à côté des images : 4 × H/4 × W/4 en float16, (F13x, F13y, F31x, F31y) en pixels.

NVOF lit une luma 8 bits ; celle-ci est la luma BT.709 en plage limitée (16-235), celle qu'un
fichier vidéo donne au lecteur. Le calcul passe par nvof_dump (le pilote NVIDIA), construit depuis
nvof_dump/ :

    python -I tools/nvof_interp/precompute_nvof.py atd12k/train_10k --tool build/nvof_dump/nvof_dump
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

GRID = 4


def frame(folder: Path, name: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        if (folder / f"{name}{ext}").exists():
            return folder / f"{name}{ext}"
    return None


def luma(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip(np.rint(16.0 + 219.0 * y), 0, 255).astype(np.uint8)


def cells(n: int) -> int:
    return (n + GRID - 1) // GRID


def to_planes(raw: bytes, w: int, h: int) -> np.ndarray:
    """Les deux flux S10.5 du fichier de nvof_dump, en 4 plans de pixels."""
    count = cells(w) * cells(h) * 2
    vectors = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32.0
    if vectors.size != 2 * count:
        raise ValueError(f"{vectors.size} valeurs, {2 * count} attendues")
    forward = vectors[:count].reshape(cells(h), cells(w), 2)
    backward = vectors[count:].reshape(cells(h), cells(w), 2)
    return np.concatenate((forward, backward), axis=2).transpose(2, 0, 1)


def run_chunk(todo: list[Path], tool: Path, offset: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        tasks, sizes = [], {}
        for i, folder in enumerate(todo, offset):
            first, last = frame(folder, "frame1"), frame(folder, "frame3")
            if first is None or last is None:
                continue
            a, b = luma(first), luma(last)
            if a.shape != b.shape:
                print(f"{folder} : tailles différentes, ignoré")
                continue
            h, w = a.shape
            a.tofile(work / f"{i}a.raw")
            b.tofile(work / f"{i}b.raw")
            tasks.append(f"{w} {h} {work / f'{i}a.raw'} {work / f'{i}b.raw'} {work / f'{i}.flow'}\n")
            sizes[str(work / f"{i}.flow")] = (folder, w, h)
        # Tâches groupées par taille : une session NVOF par taille.
        tasks.sort(key=lambda t: tuple(int(v) for v in t.split()[:2]))
        done = subprocess.run([str(tool)], input="".join(tasks), text=True, capture_output=True, check=False)
        if done.returncode != 0:
            raise SystemExit(done.stderr.strip())
        for line in done.stdout.splitlines():
            out = line.removeprefix("ok ").strip()
            folder, w, h = sizes[out]
            planes = to_planes(Path(out).read_bytes(), w, h)
            np.save(folder / "nvof.npy", planes.astype(np.float16))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="dossier des triplets (frame1, frame2, frame3)")
    parser.add_argument("--tool", type=Path, required=True, help="l'exécutable nvof_dump")
    parser.add_argument("--chunk", type=int, default=256, help="triplets par lot")
    parser.add_argument("--force", action="store_true", help="recalcule les nvof.npy présents")
    args = parser.parse_args()

    folders = sorted(p.parent for p in args.root.rglob("frame1.*"))
    todo = [f for f in folders if args.force or not (f / "nvof.npy").exists()]
    print(f"{len(folders)} triplets, {len(todo)} à calculer")
    # Par lots : les lumas brutes d'un lot seulement sur disque à la fois.
    for begin in range(0, len(todo), args.chunk):
        run_chunk(todo[begin : begin + args.chunk], args.tool, begin)
        print(f"{min(begin + args.chunk, len(todo))}/{len(todo)}", flush=True)
    print("flux écrits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
