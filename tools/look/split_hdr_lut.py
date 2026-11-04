#!/usr/bin/env python3
"""Sépare une LUT d'HDR (PQ BT.2020 → PQ BT.2020, « HDR grading base ver.9 ») en courbe neutre et couleur.

Chantier 8.3. La courbe neutre est la LUT lue sur l'axe des gris (4 096 points, en intensité I d'ICtCp, rendue
monotone) : au contrat d'écran v2 (8.4), c'est la courbe par plan (render/curve) qui la remplace. La couleur est
ce qui reste : pour chaque couleur, la sortie de la LUT en ICtCp, son intensité ramenée par l'inverse de la
courbe neutre, ses Ct/Cp gardés. Elle est réexprimée dans le domaine SDR du lecteur (BT.709, gamma 2,4,
blanc = 203 nits, docs/luts.md), écrêtée en douceur au-dessus de 0,9, et écrite en « <nom> (couleur).cube »
avec « # ajp-domain: sdr ». --compose enchaîne d'abord les LUT données, dans l'ordre (la pile du lecteur, comme
des nœuds de Resolve), en une seule LUT PQ, puis sépare celle-ci : sa couleur est le look par défaut du contrat v2
(ver9-couleur.cube), sa courbe neutre celle que la courbe par plan remplace.

  split_hdr_lut.py LUT.cube [LUT2.cube …] [--out DOSSIER] [--size 33] [--compose ver9-couleur.cube]
  split_hdr_lut.py --check

Rapport : la courbe neutre aux points 1/5/20/50/100/203 nits et le décalage de teinte de la couleur (moyenne
et p95 en degrés, couleurs de chroma ≥ 0,02). --check : l'identité rend l'identité, une LUT de pure courbe
(en I d'ICtCp) rend une couleur identité, ΔE_ITP ≤ 0,1 (BT.2124).
"""
import argparse
import os
import sys

import numpy as np

CURVE_POINTS = 4096
REPORT_NITS = (1.0, 5.0, 20.0, 50.0, 100.0, 203.0)
REFERENCE_NITS = 203.0
SDR_GAMMA = 2.4
SOFT_KNEE = 0.9
CHECK_MAX_DE = 0.1

M1, M2 = 0.1593017578125, 78.84375
C1, C2, C3 = 0.8359375, 18.8515625, 18.6875

# BT.2100 : RGB BT.2020 linéaire → LMS, puis L'M'S' (PQ) → ICtCp.
RGB_TO_LMS = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]], dtype=np.float64) / 4096.0
LMS_TO_ICTCP = np.array([[2048, 2048, 0], [6610, -13613, 7003], [17933, -17390, -543]],
                        dtype=np.float64) / 4096.0
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)
ICTCP_TO_LMS = np.linalg.inv(LMS_TO_ICTCP)
# BT.709 → BT.2020 linéaire (BT.2087).
BT709_TO_2020 = np.array([[0.6274039, 0.3292830, 0.0433131], [0.0690973, 0.9195404, 0.0113623],
                          [0.0163914, 0.0880133, 0.8955953]])
BT2020_TO_709 = np.linalg.inv(BT709_TO_2020)


def pq_to_linear(e):
    """Codes PQ (0 à 1) → lumière / 10 000 nits."""
    p = np.power(np.clip(e, 0.0, 1.0), 1.0 / M2)
    return np.power(np.maximum(p - C1, 0.0) / (C2 - C3 * p), 1.0 / M1)


def linear_to_pq(y):
    q = np.power(np.clip(y, 0.0, 1.0), M1)
    return np.power((C1 + C2 * q) / (1.0 + C3 * q), M2)


def ictcp_of_pq(rgb_pq):
    lms = pq_to_linear(rgb_pq) @ RGB_TO_LMS.T
    return linear_to_pq(lms) @ LMS_TO_ICTCP.T


def pq_of_ictcp(ictcp):
    lms = pq_to_linear(ictcp @ ICTCP_TO_LMS.T) @ LMS_TO_RGB.T
    return linear_to_pq(np.maximum(lms, 0.0))


def delta_e_itp(a_pq, b_pq):
    """ΔE_ITP (BT.2124) entre deux tableaux de couleurs PQ BT.2020 : T = Ct / 2."""
    d = ictcp_of_pq(a_pq) - ictcp_of_pq(b_pq)
    return 720.0 * np.sqrt(d[..., 0] ** 2 + (0.5 * d[..., 1]) ** 2 + d[..., 2] ** 2)


# --- Fichiers .cube : rouge le plus rapide, table rangée [b, g, r, canal]. ---

def read_cube(path):
    size = None
    lo, hi = np.zeros(3), np.ones(3)
    values = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            key = words[0].upper()
            if key == "LUT_3D_SIZE":
                size = int(words[1])
            elif key == "DOMAIN_MIN":
                lo = np.array([float(w) for w in words[1:4]])
            elif key == "DOMAIN_MAX":
                hi = np.array([float(w) for w in words[1:4]])
            elif key in ("TITLE", "LUT_1D_SIZE", "LUT_3D_INPUT_RANGE"):
                if key == "LUT_1D_SIZE":
                    raise ValueError(f"{path} : LUT 1D refusée")
                continue
            else:
                values.append([float(w) for w in words[:3]])
    if size is None or len(values) != size ** 3:
        raise ValueError(f"{path} : LUT_3D_SIZE absent ou {len(values)} entrées")
    if np.any(lo != 0.0) or np.any(hi != 1.0):
        raise ValueError(f"{path} : domaine autre que 0..1 non pris en charge")
    return np.array(values).reshape(size, size, size, 3)


def write_cube(path, table, title, domain):
    size = table.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"# ajp-domain: {domain}\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        for rgb in table.reshape(-1, 3):
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")


def identity_table(size):
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def fill_dark_zone(table):
    """Les entrées noires hors de la zone utile (sortie nulle pour une entrée non noire) reprennent la moyenne
    de leurs voisines valides, de proche en proche, comme au chargement dans le lecteur (docs/luts.md)."""
    size = table.shape[0]
    valid = table.max(axis=-1) > 1e-6
    valid[0, 0, 0] = True
    filled = int((~valid).sum())
    table = table.copy()
    while not valid.all():
        total = np.zeros_like(table)
        count = np.zeros(table.shape[:3])
        for axis in range(3):
            for step in (-1, 1):
                shifted_valid = np.roll(valid, step, axis=axis)
                shifted = np.roll(table, step, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = 0 if step == 1 else size - 1
                shifted_valid[tuple(edge)] = False
                total += np.where(shifted_valid[..., None], shifted, 0.0)
                count += shifted_valid
        grow = (~valid) & (count > 0)
        table[grow] = total[grow] / count[grow][:, None]
        valid |= grow
    return table, filled


def pchip(x, y, at):
    """Interpolation cubique monotone de Fritsch-Carlson des points (x, y) en `at`."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slope = np.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    inner = np.where(same_sign, (w1 + w2) / (w1 / np.where(same_sign, delta[:-1], 1.0)
                                                 + w2 / np.where(same_sign, delta[1:], 1.0)), 0.0)
    slope[1:-1] = inner
    slope[0], slope[-1] = delta[0], delta[-1]
    i = np.clip(np.searchsorted(x, at, side="right") - 1, 0, len(x) - 2)
    t = (at - x[i]) / h[i]
    h00, h10 = (1 + 2 * t) * (1 - t) ** 2, t * (1 - t) ** 2
    h01, h11 = t * t * (3 - 2 * t), t * t * (t - 1)
    return h00 * y[i] + h10 * h[i] * slope[i] + h01 * y[i + 1] + h11 * h[i] * slope[i + 1]


def sample(table, rgb):
    """Interpolation trilinéaire de la table en des couleurs rgb (… × 3) dans 0..1."""
    size = table.shape[0]
    x = np.clip(rgb, 0.0, 1.0) * (size - 1)
    i0 = np.minimum(np.floor(x).astype(int), size - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0.0
    for db, wb in ((0, 1.0 - fb), (1, fb)):
        for dg, wg in ((0, 1.0 - fg), (1, fg)):
            for dr, wr in ((0, 1.0 - fr), (1, fr)):
                out = out + wb * wg * wr * table[b0 + db, g0 + dg, r0 + dr]
    return out


class Split:
    """La courbe neutre d'une LUT PQ et sa composante couleur."""

    def __init__(self, table):
        self.table = table
        # Les nœuds gris de la table seulement : entre deux, l'interpolation trilinéaire mêle des nœuds colorés
        # et l'axe des gris n'y est plus gris. Les intensités des nœuds, rendues monotones, puis Fritsch-Carlson.
        size = table.shape[0]
        nodes = np.linspace(0.0, 1.0, size)
        diagonal = table[np.arange(size), np.arange(size), np.arange(size)]
        intensity = np.maximum.accumulate(ictcp_of_pq(diagonal)[:, 0])
        self.grey = np.linspace(0.0, 1.0, CURVE_POINTS)
        curve = np.maximum.accumulate(pchip(nodes, intensity, self.grey))
        # Strictement croissante : un plat de la LUT s'inverse sans ambiguïté (pente minimale 1e-6).
        self.curve = curve + 1e-6 * self.grey

    def neutral_nits(self, nits):
        code = linear_to_pq(np.asarray(nits) / 10000.0)
        return pq_to_linear(np.interp(code, self.grey, self.curve)) * 10000.0

    def color(self, rgb_pq):
        """La composante couleur sur des codes PQ : intensité ramenée par l'inverse de la courbe, Ct/Cp de la
        LUT."""
        ictcp = ictcp_of_pq(sample(self.table, rgb_pq))
        ictcp[..., 0] = np.interp(ictcp[..., 0], self.curve, self.grey)
        return np.clip(pq_of_ictcp(ictcp), 0.0, 1.0)


def sdr_to_pq(sdr):
    linear709 = np.power(np.clip(sdr, 0.0, 1.0), SDR_GAMMA) * (REFERENCE_NITS / 10000.0)
    return linear_to_pq(linear709 @ BT709_TO_2020.T)


def soft_clip(x):
    """Identité jusqu'à SOFT_KNEE, puis une approche douce de 1 (tanh, pente 1 au genou)."""
    span = 1.0 - SOFT_KNEE
    return np.where(x <= SOFT_KNEE, x, SOFT_KNEE + span * np.tanh((x - SOFT_KNEE) / span))


def pq_to_sdr(pq):
    linear709 = (pq_to_linear(pq) @ BT2020_TO_709.T) * (10000.0 / REFERENCE_NITS)
    return soft_clip(np.power(np.maximum(linear709, 0.0), 1.0 / SDR_GAMMA))


def sdr_color_table(split, size):
    """La couleur d'une LUT sur la grille SDR du lecteur."""
    return pq_to_sdr(split.color(sdr_to_pq(identity_table(size))))


def chain(tables):
    """Les LUT enchaînées dans l'ordre, aux nœuds de la plus fine."""
    out = identity_table(max(t.shape[0] for t in tables))
    for table in tables:
        out = sample(table, out)
    return out


def hue_shift(split, count=20000, seed=7):
    """Le décalage de teinte de la couleur (degrés) sur des couleurs SDR tirées au hasard, de chroma ≥ 0,02."""
    rng = np.random.default_rng(seed)
    pq = sdr_to_pq(rng.random((count, 3)))
    before = ictcp_of_pq(pq)
    after = ictcp_of_pq(split.color(pq))
    chroma = np.hypot(before[:, 1], before[:, 2])
    keep = (chroma >= 0.02) & (np.hypot(after[:, 1], after[:, 2]) >= 1e-4)
    if not keep.any():
        return 0.0, 0.0
    h0 = np.arctan2(before[keep, 2], before[keep, 1])
    h1 = np.arctan2(after[keep, 2], after[keep, 1])
    dh = np.abs(np.degrees(np.angle(np.exp(1j * (h1 - h0)))))
    return float(dh.mean()), float(np.percentile(dh, 95))


def report(name, split, filled):
    print(f"{name} :")
    if filled:
        print(f"  zone noire : {filled} entrées reprises de leurs voisines")
    points = ", ".join(f"{n:g} → {o:.1f}" for n, o in zip(REPORT_NITS, split.neutral_nits(REPORT_NITS)))
    print(f"  courbe neutre (nits) : {points}")
    mean, p95 = hue_shift(split)
    print(f"  décalage de teinte de la couleur : moyenne {mean:.2f}°, p95 {p95:.2f}°")


def color_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem} (couleur).cube"


def check():
    """L'identité rend l'identité ; une LUT de pure courbe en I rend une couleur identité."""
    grid = identity_table(33)
    failures = 0
    # Points de contrôle : les nœuds de la grille jusqu'au blanc de référence (le domaine SDR du lecteur) ; entre
    # les nœuds, l'écart est celui de l'interpolation de la table, le même dans le lecteur.
    nodes = grid.reshape(-1, 3)
    probe = nodes[nodes.max(axis=1) <= linear_to_pq(REFERENCE_NITS / 10000.0) + 1e-9]

    identity = Split(grid)
    de = float(delta_e_itp(identity.color(probe), probe).max())
    print(f"identité : ΔE_ITP max {de:.4f}")
    failures += de > CHECK_MAX_DE

    # Pure courbe : I' = I (1,2 − 0,2 I) (l'expansion douce d'une LUT de base), Ct/Cp gardés.
    ictcp = ictcp_of_pq(grid)
    ictcp[..., 0] *= 1.2 - 0.2 * ictcp[..., 0]
    curve_only = Split(pq_of_ictcp(ictcp))
    de = float(delta_e_itp(curve_only.color(probe), probe).max())
    nits = curve_only.neutral_nits(REFERENCE_NITS)
    print(f"pure courbe : ΔE_ITP max {de:.4f} (203 nits → {nits:.1f})")
    failures += de > CHECK_MAX_DE

    # En SDR, l'identité jusqu'au genou de l'écrêtage doux.
    sdr, ref = sdr_color_table(identity, 17), identity_table(17)
    below = ref.max(axis=-1) <= SOFT_KNEE
    err = float(np.abs(sdr - ref)[below].max())
    print(f"identité SDR (sous le genou {SOFT_KNEE}) : écart max {err:.5f}")
    failures += err > 2e-3
    print("--check :", "réussi" if failures == 0 else "ÉCHEC")
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("luts", nargs="*")
    parser.add_argument("--out", default=".")
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--compose")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    if not args.luts:
        parser.error("au moins une LUT, ou --check")
    if not 17 <= args.size <= 65:
        parser.error("--size de 17 à 65")
    os.makedirs(args.out, exist_ok=True)
    tables = []
    for path in args.luts:
        table, filled = fill_dark_zone(read_cube(path))
        split = Split(table)
        report(os.path.basename(path), split, filled)
        out = os.path.join(args.out, color_name(path))
        write_cube(out, sdr_color_table(split, args.size), color_name(path)[:-5], "sdr")
        print(f"  écrit : {out}")
        tables.append(table)
    if args.compose:
        split = Split(chain(tables))
        report(f"pile de {len(tables)} LUT", split, 0)
        out = os.path.join(args.out, args.compose)
        write_cube(out, sdr_color_table(split, args.size), os.path.splitext(args.compose)[0], "sdr")
        print(f"  écrit : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
