#!/usr/bin/env python3
"""Cuit une pile de LUT du lecteur en une seule LUT 3D.

Le lecteur applique la pile comme render/grade.cpp (look_source) : pour chaque emplacement d'intensité non nulle,
dans l'ordre, c = mix(c, clamp(lut(c)), intensité), la LUT lue par interpolation trilinéaire entre ses points
(texture 3D filtrée, position (c·(N−1)+0,5)/N). Une LUT dont le nom ou le titre contient « hdr », « pq » ou « 2020 »
lit des codes PQ BT.2020 : toute la chaîne est alors en PQ (render/looks.cpp, look_is_pq). Ici les mêmes lectures,
les entrées noires d'un gamut sans usage remplacées par leurs voisins comme à la lecture (render/cube_lut.cpp,
fill_unreachable), puis la pile est évaluée en chaque point d'une grille N³ et écrite en .cube.

  bake_stack.py --luts DOSSIER --out FICHIER.cube [--size 65] [--check] "LUT:intensité" ["LUT:intensité" …]
  bake_stack.py --luts DOSSIER --out FICHIER.cube --play-toml play.toml     (la pile du play.toml : look, looks)

Les intensités à 0 sont écartées. --check compare, sur des points tirés au hasard, la pile d'origine et la LUT
cuite (écarts en codes PQ ; un code 10 bits vaut 1/1023).
"""
import argparse
import os
import sys

import numpy as np


def read_cube(path):
    size = None
    title = ""
    dmin = np.zeros(3)
    dmax = np.ones(3)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("TITLE"):
                title = s[5:].strip().strip('"')
            elif s.startswith("LUT_3D_SIZE"):
                size = int(s.split()[1])
            elif s.startswith("DOMAIN_MIN"):
                dmin = np.array([float(x) for x in s.split()[1:4]])
            elif s.startswith("DOMAIN_MAX"):
                dmax = np.array([float(x) for x in s.split()[1:4]])
            elif s.startswith("LUT_") or s.startswith("DOMAIN_"):
                raise SystemExit(f"{path} : mot-clé non géré : {s}")
            else:
                rows.append([float(x) for x in s.split()[:3]])
    if size is None or len(rows) != size ** 3:
        raise SystemExit(f"{path} : {len(rows)} lignes pour LUT_3D_SIZE {size}")
    data = np.clip((np.array(rows, dtype=np.float32) - dmin) / (dmax - dmin), 0.0, 1.0)
    # index = b·n² + g·n + r : le rouge varie le plus vite ; le tableau est rangé [b, g, r, canal].
    return title, size, data.reshape(size, size, size, 3)


def fill_unreachable(lut):
    """render/cube_lut.cpp : fill_unreachable (une couche de voisins valides par tour)."""
    n = lut.shape[0]
    idx = np.indices((n, n, n))  # b, g, r
    bsum = idx[0] + idx[1] + idx[2]
    valid = ~((bsum > 3) & np.all(lut == 0.0, axis=3))
    lut = lut.copy()
    while not valid.all():
        sums = np.zeros_like(lut)
        counts = np.zeros(lut.shape[:3])
        for axis in range(3):
            for step in (-1, 1):
                shifted_v = np.roll(valid, step, axis=axis)
                shifted_l = np.roll(lut, step, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = slice(0, 1) if step == 1 else slice(n - 1, n)
                shifted_v[tuple(edge)] = False  # pas de voisin hors de la grille
                sums += np.where(shifted_v[..., None], shifted_l, 0.0)
                counts += shifted_v
        fill = (~valid) & (counts > 0)
        if not fill.any():
            break
        lut[fill] = sums[fill] / counts[fill][:, None]
        valid = valid | fill
    return lut


def sample(lut, c):
    """Lecture trilinéaire d'une LUT [b, g, r, 3] en des couleurs c (N, 3) dans [0, 1]."""
    n = lut.shape[0]
    p = np.clip(c, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(p).astype(int), n - 2)
    f = p - i0
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    out = np.zeros((c.shape[0], 3))
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                out += w * lut[b0 + db, g0 + dg, r0 + dr]
    return out


def apply_stack(layers, c):
    for lut, mix in layers:
        c = c + mix * (np.clip(sample(lut, c), 0.0, 1.0) - c)
    return c


def stack_from_play_toml(path):
    import tomllib
    with open(path, "rb") as f:
        grade = tomllib.load(f).get("grade", {})
    out = [(grade.get("look", ""), float(grade.get("look_mix", 1.0)))]
    mixes = grade.get("looks_mix", [])
    for i, name in enumerate(grade.get("looks", [])):
        out.append((name, float(mixes[i]) if i < len(mixes) else 1.0))
    return [(n, m) for n, m in out if n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--luts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=65)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--play-toml")
    ap.add_argument("slots", nargs="*", help='"nom.cube:intensité"')
    a = ap.parse_args()
    slots = stack_from_play_toml(a.play_toml) if a.play_toml else []
    for s in a.slots:
        name, _, mix = s.rpartition(":")
        slots.append((name, float(mix)))
    slots = [(n, m) for n, m in slots if m > 0.0]
    if not slots:
        raise SystemExit("pile vide")
    layers = []
    pq = False
    for name, mix in slots:
        title, size, lut = read_cube(os.path.join(a.luts, name))
        layers.append((fill_unreachable(lut), mix))
        pq = pq or any(w in (name + " " + title).lower() for w in ("pq", "hdr", "2020"))
    if not pq:
        raise SystemExit("aucune LUT d'HDR dans la pile : la chaîne SDR (BT.709, gamma) n'est pas cuite ici")
    n = a.size
    axis = np.linspace(0.0, 1.0, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)  # le rouge varie le plus vite
    baked = np.clip(apply_stack(layers, grid), 0.0, 1.0)
    name = os.path.basename(a.out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f'# Pile de LUT cuite par tools/look/bake_stack.py : {len(slots)} emplacements, PQ BT.2020.\n')
        f.write("# " + " ; ".join(f"{n}:{m:g}" for n, m in slots) + "\n")
        f.write(f'TITLE "{os.path.splitext(name)[0]}"\nLUT_3D_SIZE {n}\n')
        for row in baked:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    print(f"{a.out} : {n}³ points, {len(slots)} emplacements")
    if a.check:
        rng = np.random.default_rng(1)
        pts = rng.random((200000, 3))
        # plus de points près des gris et des noirs, où l'œil est le plus sensible
        gray = rng.random((100000, 1)) ** 2.0
        pts = np.vstack([pts, np.repeat(gray, 3, axis=1) + 0.02 * rng.standard_normal((100000, 3))])
        pts = np.clip(pts, 0.0, 1.0)
        want = apply_stack(layers, pts)
        got = sample(np.clip(baked, 0.0, 1.0).reshape(n, n, n, 3)[::1], pts) if False else None
        lut_baked = baked.reshape(n, n, n, 3)
        got = sample(lut_baked, pts)
        err = np.abs(got - want).max(axis=1) * 1023.0
        print("écart en codes 10 bits : moyen %.3f, p99 %.3f, p99,9 %.3f, max %.3f" %
              (err.mean(), np.percentile(err, 99), np.percentile(err, 99.9), err.max()))


if __name__ == "__main__":
    sys.exit(main())
