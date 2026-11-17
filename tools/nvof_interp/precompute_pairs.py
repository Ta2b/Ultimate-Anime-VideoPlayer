#!/usr/bin/env python3
"""Calcule une fois les flux NVOF de toutes les paires d'une suite (écrite par sequences.py).

Pour chaque suite de N images et chaque écart g de --ecarts (2 à 8 par défaut, g < N), les paires (i, i + g) :
flows.npy (P × 4 × H/4 × W/4, float16 : F_ij x, y puis F_ji x, y, en pixels) et pairs.npy (P × 2, les indices i, j)
et mag.npy (P, float32 : le 90e centile de |F_ij| en pixels, qui sert à tirer plus souvent les grands mouvements).
NVOF tourne comme dans le lecteur : grille 4 sur une luma réduite à --hauteur-flux (720p, la configuration retenue à
l'étape 1 bis, la meilleure et la moins chère), puis les vecteurs sont ramenés en pixels des images (×1,5 pour des
images 1080p). Une case de flux couvre alors H / (hauteur du flux) × 4 pixels de l'image, 6 en 1080p.
Une suite qui a déjà ses flux est sautée.

    python -I tools/nvof_interp/precompute_pairs.py /ssd/ajp-dataset/suites --tool build/nvof_dump/nvof_dump
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precompute_nvof import to_planes  # noqa: E402


def luma(rgb: np.ndarray) -> np.ndarray:
    """La luma BT.709 en plage limitée (16-235) que NVOF lit dans le lecteur."""
    y = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) / 255.0
    return np.clip(np.rint(16.0 + 219.0 * y), 0, 255).astype(np.uint8)


def at_height(rgb: np.ndarray, height: int | None) -> np.ndarray:
    """Réduction par moyenne de surface (le filtre du banc nvof_warp) à la hauteur du flux."""
    h, w, _ = rgb.shape
    if height is None or height == h:
        return rgb
    width = round(w * height / h / 2) * 2
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((width, height), Image.BOX))


def pairs_of(n: int, gaps: list[int]) -> list[tuple[int, int]]:
    return [(i, i + g) for g in gaps if g < n for i in range(n - g)]


def run_chunk(runs: list[Path], tool: Path, gaps: list[int], tmp: Path, flow_height: int | None) -> int:
    tasks, owners = [], {}
    for r, folder in enumerate(runs):
        frames = np.load(folder / "frames.npy", mmap_mode="r")
        n = frames.shape[0]
        for i in range(n):
            small = luma(at_height(frames[i], flow_height))
            small.tofile(tmp / f"{r}_{i}.raw")
        h, w = small.shape
        for p, (i, j) in enumerate(pairs_of(n, gaps)):
            out = tmp / f"{r}_{p}.flow"
            # Champs séparés par des tabulations : le dossier peut contenir des espaces.
            tasks.append("\t".join((str(w), str(h), str(tmp / f"{r}_{i}.raw"), str(tmp / f"{r}_{j}.raw"), str(out))))
            owners[str(out)] = (r, p, w, h)
    tasks.sort(key=lambda t: tuple(int(v) for v in t.split("\t")[:2]))  # une session NVOF par taille
    done = subprocess.run([str(tool), "--grid", "4"], input="\n".join(tasks) + "\n", text=True, capture_output=True,
                          check=False)
    if done.returncode != 0:
        raise SystemExit(done.stderr.strip())
    flows: dict[int, dict[int, np.ndarray]] = {}
    for line in done.stdout.splitlines():
        out = line.removeprefix("ok ").strip()
        r, p, w, h = owners.pop(out)
        # Vecteurs en pixels du flux, ramenés en pixels des images.
        scale = np.load(runs[r] / "frames.npy", mmap_mode="r").shape[1] / h
        flows.setdefault(r, {})[p] = (to_planes(Path(out).read_bytes(), w, h) * scale).astype(np.float16)
    holes = 0
    for r, folder in enumerate(runs):
        n = np.load(folder / "frames.npy", mmap_mode="r").shape[0]
        pairs = pairs_of(n, gaps)
        got = flows.get(r, {})
        if len(got) != len(pairs):  # une paire sans flux : la suite reste sans flows.npy, à relancer
            holes += 1
            continue
        np.save(folder / "pairs.npy", np.array(pairs, np.int16))
        mags = [np.percentile(np.hypot(got[p][0].astype(np.float32), got[p][1].astype(np.float32)), 90)
                for p in range(len(pairs))]
        np.save(folder / "mag.npy", np.array(mags, np.float32))
        np.save(folder / "flows.npy", np.stack([got[p] for p in range(len(pairs))]))  # en dernier : complet
    for f in tmp.iterdir():
        f.unlink()
    return holes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="dossier de sequences.py")
    parser.add_argument("--tool", type=Path, required=True, help="l'exécutable nvof_dump")
    parser.add_argument("--ecarts", default="2,3,4,5,6,7,8")
    parser.add_argument("--chunk", type=int, default=64, help="suites par appel de nvof_dump")
    parser.add_argument("--hauteur-flux", type=int, default=720,
                        help="hauteur de la luma donnée à NVOF (0 : celle des images)")
    args = parser.parse_args()
    gaps = sorted({int(g) for g in args.ecarts.split(",")})

    runs = sorted(m.parent for m in args.root.rglob("meta.json"))
    todo = [r for r in runs if not (r / "flows.npy").exists()]
    print(f"{len(runs)} suites, {len(todo)} à calculer", flush=True)
    holes = 0
    # Les lumas brutes passent par la mémoire (/dev/shm) quand elle existe : rien n'use le disque.
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(dir=shm) as tmp:
        for begin in range(0, len(todo), args.chunk):
            holes += run_chunk(todo[begin : begin + args.chunk], args.tool, gaps, Path(tmp), args.hauteur_flux or None)
            print(f"{min(begin + args.chunk, len(todo))}/{len(todo)}", flush=True)
    if holes:
        print(f"{holes} suite(s) sans flux : relancer la commande")
        return 1
    print("flux écrits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
