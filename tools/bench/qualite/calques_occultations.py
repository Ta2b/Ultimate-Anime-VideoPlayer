#!/usr/bin/env python3
"""Rejouer les zones de désocclusion dans les deux sens du flux, hors GPU.

Le même extrait fournit les tâches de carte monodirectionnelles, les dumps
aller-retour et la source de l'image RGB. Le banc lit la caméra et la grille
des calques réelles de carte_rejeu --grille-calques. Les détections sont des
candidats pour protéger le bord des calques, sans décision de rendu.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calques_rejeu import read_tasks  # noqa: E402
from nvof_mesures import sample  # noqa: E402
from rife_budget import cell_centers, inside, read_dump, video_positions  # noqa: E402

CLOSURE_720 = 1.25
PHOTO_CODES = 12.0
MIN_PHOTO_SAMPLES = 2
MIN_TILE_CELLS = 2
GRID = 4
SAMPLE_OFFSETS = ((0, 0), (-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))


@dataclass(frozen=True)
class Layers:
    decor: np.ndarray
    tile_px: int
    camera: np.ndarray


def read_layers(replay, path, divisor, count):
    with tempfile.TemporaryDirectory() as temp:
        lines, _ = read_tasks(path, temp)
        run = subprocess.run([replay, "--grille-calques", "--diviseur", str(divisor)],
                             input="".join(lines[:count]), capture_output=True, text=True, check=True)
    layers = {}
    for line in run.stdout.splitlines():
        if not line.startswith("grille "):
            continue
        _, index, cols, rows, side, *fields = line.split()
        bits = np.frombuffer(fields[-1].encode("ascii"), dtype=np.uint8) == ord("1")
        tx, a, b, ty, c, d = map(float, fields[:-1])
        camera = np.array(((a, b, tx), (c, d, ty), (0., 0., 1.)))
        layers[int(index)] = Layers(bits.reshape(int(rows), int(cols)), int(side), camera)
    return layers


def camera_coords(camera, x, y):
    return (camera[0, 0] * x + camera[0, 1] * y + camera[0, 2],
            camera[1, 0] * x + camera[1, 1] * y + camera[1, 2])


def rgb_difference(source, other, x, y, qx, qy):
    h, w = source.shape[1:]
    error = np.maximum.reduce([np.abs(sample(a, x, y) - sample(b, qx, qy))
                               for a, b in zip(source, other)])
    return inside(x, y, w, h) & inside(qx, qy, w, h) & (error > PHOTO_CODES / 255.0)


def direction_evidence(source, other, forward, reverse, size, camera):
    w, h = size
    fx, fy = forward.flow
    cx, cy = cell_centers(w, GRID), cell_centers(h, GRID)
    qx, qy = cx[None, :] + fx, cy[:, None] + fy
    rx, ry = np.interp(qx, cx, np.arange(len(cx))), np.interp(qy, cy, np.arange(len(cy)))
    bx, by = (sample(p, rx, ry) for p in reverse.flow)
    closure = inside(qx, qy, w, h) & (np.hypot(fx + bx, fy + by) * 720 / h > CLOSURE_720)
    sy, sx = np.asarray(source.shape[1:]) / (h, w)
    photo = np.zeros(fx.shape, dtype=np.uint8)
    both = np.zeros_like(photo)
    for ox, oy in SAMPLE_OFFSETS:
        x = (cx[None, :] + ox + 0.5) * sx - 0.5
        y = (cy[:, None] + oy + 0.5) * sy - 0.5
        disagreement = rgb_difference(source, other, x, y, x + fx * sx, y + fy * sy)
        tx, ty = camera_coords(camera, x, y)
        camera_disagreement = rgb_difference(source, other, x, y, tx, ty)
        photo += disagreement
        both += disagreement & camera_disagreement
    return closure, closure & (photo >= MIN_PHOTO_SAMPLES), closure & (both >= MIN_PHOTO_SAMPLES)


def adjacent_protection(decor):
    protected = np.pad(~decor, 1)
    return decor & (protected[1:-1, :-2] | protected[1:-1, 2:] |
                    protected[:-2, 1:-1] | protected[2:, 1:-1])


def extend_protection(evidence, decor, side):
    count = np.add.reduceat(np.add.reduceat(evidence.astype(np.uint16), np.arange(0, evidence.shape[0], side),
                                            axis=0), np.arange(0, evidence.shape[1], side), axis=1)
    if count.shape != decor.shape:
        raise ValueError("les cellules du flux ne couvrent pas la même grille que les calques")
    return adjacent_protection(decor) & (count >= MIN_TILE_CELLS)


def read_pairs(paths):
    rows = []
    for line in paths.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        width, height, a, b, flow, *_ = line.split("\t")
        rows.append({"paire": [a, b], "taille_luma": [int(width), int(height)], "flux": flow})
    if not rows:
        raise ValueError("aucune paire aller-retour")
    return rows


def run(args):
    rows = read_pairs(args.aller_retour)
    base = read_pairs(args.carte)[:len(rows)]
    if [(r["paire"], r["taille_luma"]) for r in base] != [(r["paire"], r["taille_luma"]) for r in rows]:
        raise ValueError("les tâches des calques et du flux aller-retour doivent nommer les mêmes images")
    frames = video_positions(args.aller_retour, args.video, rows, args.debut)
    layers = read_layers(args.rejeu, args.carte, args.diviseur, len(rows))
    report = []
    for i, row in enumerate(rows):
        size = row["taille_luma"]
        forward, reverse = read_dump(args.aller_retour.parent / row["flux"], *size, GRID)
        if reverse is None:
            raise ValueError(f"paire {i} sans retour")
        if i not in layers:
            report.append({"paire": i, "avec_calques": False})
            continue
        a, b = [frames[name] for name in row["paire"]]
        layer = layers[i]
        full_size = np.asarray(a.shape[1:][::-1])
        work_size = np.asarray(size) * args.diviseur
        resize = np.diag([*(full_size / work_size), 1.0])
        resize[:2, 2] = (full_size / work_size - 1.0) * 0.5  # indices aux centres des pixels
        camera = resize @ layer.camera @ np.linalg.inv(resize)
        # Un veto pour la plaque en b vient de b→a. Un veto pour les calques
        # peut venir de l'un ou l'autre côté de la paire.
        f = direction_evidence(a, b, forward, reverse, size, camera)
        r = direction_evidence(b, a, reverse, forward, size, np.linalg.inv(camera))
        side = layer.tile_px // (GRID * args.diviseur)
        if side < 1 or layer.tile_px % (GRID * args.diviseur):
            raise ValueError("la tuile de scène doit couvrir un nombre entier de cellules du flux")
        plaque = extend_protection(r[2], layer.decor, side)
        pair = extend_protection(f[2] | r[2], layer.decor, side)
        report.append({"paire": i, "avec_calques": True, "tuiles_decor": int(layer.decor.sum()),
                       "bord_decor": int(adjacent_protection(layer.decor).sum()),
                       "cellules": int(f[0].size),
                       "indices_a": [int(x.sum()) for x in f], "indices_b": [int(x.sum()) for x in r],
                       "tuiles_a_proteger_plaque": int(plaque.sum()), "tuiles_a_proteger_calques": int(pair.sum())})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejeu", type=Path, required=True)
    parser.add_argument("--carte", type=Path, required=True)
    parser.add_argument("--aller-retour", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--debut")
    parser.add_argument("--diviseur", type=int, default=2)
    args = parser.parse_args()
    if args.diviseur < 1:
        parser.error("diviseur positif attendu")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=1))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
