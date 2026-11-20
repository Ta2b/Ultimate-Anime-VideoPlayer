#!/usr/bin/env python3
"""Sonde des lueurs fixes sur une image source SDR BT.709, sans rendu ni GPU.

Le contraste bloc-zone approxime les deux cartes PQ avant la passe anime. La
sonde de couronne compte les directions où un blanc est entouré d'un dégradé
coloré. Les rayons sont en pixels de la source. Le résultat sert à comparer
des régions annotées. --directionnel accepte une suite de PPM à PTS source
distincts afin de confirmer une couronne avec la teinte de l'image précédente.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from hdr import LUMA_709, nits_to_pq, read_ppm16  # noqa: E402


AREA_WEIGHTS = np.array([0.0088, 0.0182, 0.0336, 0.0553, 0.0816, 0.1078, 0.1273, 0.1346,
                         0.1273, 0.1078, 0.0816, 0.0553, 0.0336, 0.0182, 0.0088])
DIRECTIONS = np.stack((np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)), axis=-1)


def block_means(values, size):
    h, w = values.shape
    padded = np.pad(values, ((0, -h % size), (0, -w % size)), mode="edge")
    return padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size).mean(axis=(1, 3))


def blurred_area(pq, size):
    area = block_means(pq, size)
    margin = len(AREA_WEIGHTS) // 2
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (margin, margin)
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(area, pad, mode="edge"),
                                                         len(AREA_WEIGHTS), axis=axis)
        area = windows @ AREA_WEIGHTS
    return area


def at_bilinear(values, x, y, block):
    u = np.clip((x + 0.5) / block - 0.5, 0, values.shape[1] - 1)
    v = np.clip((y + 0.5) / block - 0.5, 0, values.shape[0] - 1)
    c = u.astype(int)
    r = v.astype(int)
    c1, r1 = np.minimum(c + 1, values.shape[1] - 1), np.minimum(r + 1, values.shape[0] - 1)
    fu, fv = u - c, v - r
    top = values[r, c] * (1 - fu) + values[r, c1] * fu
    bottom = values[r1, c] * (1 - fu) + values[r1, c1] * fu
    return top * (1 - fv) + bottom * fv


def annular_support(pq, chroma, x, y, radius):
    h, w = pq.shape
    support = np.zeros_like(x)
    directions = np.zeros_like(x)
    centre = pq[y, x]
    for ux, uy in DIRECTIONS:
        nx, ny = x + round(ux * radius), y + round(uy * radius)
        fx, fy = x + round(ux * 2 * radius), y + round(uy * 2 * radius)
        inside = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
        nx, ny = np.clip(nx, 0, w - 1), np.clip(ny, 0, h - 1)
        near = pq[ny, nx]
        far = pq[np.clip(fy, 0, h - 1), np.clip(fx, 0, w - 1)]
        # Un palier blanc seul, un bord brutal ou un dégradé neutre ne comptent pas.
        falls_from_core = (centre - near >= 0.008) & (centre - near <= 0.08)
        colored_halo = (near - far >= 0.008) & (chroma[ny, nx] >= 0.1)
        support += inside & falls_from_core & colored_halo
        directions += inside
    return np.divide(support, directions, out=np.zeros_like(centre), where=directions > 0)


def measure(rgb, regions, upscale, radii):
    white = 203
    pq = nits_to_pq((rgb ** 2.4 @ LUMA_709) * white)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    local_size, area_size = 8 // upscale, 32 // upscale
    local = block_means(pq, local_size)
    area = blurred_area(pq, area_size)
    h, w = pq.shape
    result = {}
    for name, (left, top, width, height) in regions:
        if not (0 <= left < left + width <= w and 0 <= top < top + height <= h):
            raise ValueError(f"{name} : région hors de l'image {w} × {h}")
        white_core = rgb[top:top + height, left:left + width].min(axis=2) >= 0.9
        y, x = np.nonzero(white_core)
        row = {"blancs": int(x.size)}
        result[name] = row
        if x.size == 0:
            continue
        x, y = x + left, y + top
        contrast = at_bilinear(local, x, y, local_size) - at_bilinear(area, x, y, area_size)
        row["bloc_zone_pq_p95"] = float(np.percentile(contrast, 95))
        row["blancs_au_seuil_pq_0_20"] = int(np.count_nonzero(contrast >= 0.20))
        row["couronnes"] = {}
        for radius in radii:
            weights = annular_support(pq, chroma, x, y, radius)
            row["couronnes"][radius] = float(np.percentile(weights, 95))
    return result


@dataclass(frozen=True)
class GlowMap:
    pq: np.ndarray
    scores: np.ndarray
    hues: np.ndarray
    x: np.ndarray
    y: np.ndarray


def white_centres(rgb, pq, block):
    y, x = np.nonzero(rgb.min(axis=2) >= 0.90)
    order = np.argsort(-pq[y, x], kind="stable")
    cols = (pq.shape[1] + block - 1) // block
    _, first = np.unique((y[order] // block) * cols + x[order] // block, return_index=True)
    chosen = order[first]
    return x[chosen], y[chosen]


def directional_map(rgb, block, radial_profile=False):
    """Ancres et teinte d'arcs cohérents, une ancre au plus par tuile de sortie 32 × 32."""
    h, w = rgb.shape[:2]
    pq = nits_to_pq((rgb ** 2.4 @ LUMA_709) * 203)
    x, y = white_centres(rgb, pq, block)
    pq_map = block_means(pq, block)
    scores = np.zeros(pq_map.shape, dtype=np.float32)
    hues = np.zeros((*pq_map.shape, 2), dtype=np.float32)
    if x.size == 0:
        return GlowMap(pq_map, scores, hues, x, y)

    color_maps = [block_means(rgb[..., c], block) for c in range(3)]
    directions = DIRECTIONS[::2]
    rings = {}
    for distance in (block, 2 * block, 3 * block, 4 * block, 6 * block):
        offset = np.rint(directions * distance).astype(int)
        xx, yy = x + offset[:, 0, None], y + offset[:, 1, None]
        inside = (0 <= xx) & (xx < w) & (0 <= yy) & (yy < h)
        colors = np.stack([at_bilinear(c, xx, yy, block) for c in color_maps], axis=-1)
        hue = colors[..., (0, 2)] - colors[..., 1, None]
        dominant = colors.max(axis=-1)
        gap = 2 * dominant + colors.min(axis=-1) - colors.sum(axis=-1) if radial_profile else None
        rings[distance] = inside, at_bilinear(pq_map, xx, yy, block), hue, dominant, gap

    cy, cx = y // block, x // block
    for radius in (block, 2 * block):
        va, la, a, max_a, gap_a = rings[radius]
        vb, lb, b, max_b, _ = rings[2 * radius]
        vc, lc, c, _, gap_c = rings[3 * radius]
        inside = va & vb & vc
        visible = inside.sum(axis=0)
        dy = pq[y, x] - la
        gradual = ((dy >= 0.005) & (la - lb >= 0.007) & (lb - lc >= 0.003) &
                   (dy <= 0.67 * np.maximum(pq[y, x] - lc, 0.02)))
        mag_a = np.linalg.norm(a, axis=-1)
        mag_b = np.linalg.norm(b, axis=-1)
        da, db = a - c, b - c
        mag_da = np.linalg.norm(da, axis=-1)
        mag_db = np.linalg.norm(db, axis=-1)
        support = (inside & gradual & (mag_a >= 0.14) & (mag_b >= 0.12) &
                   ((a * b).sum(axis=-1) >= 0.94 * mag_a * mag_b) &
                   (mag_da >= 0.035) & (mag_da - mag_db >= 0.006) &
                   ((da * db).sum(axis=-1) >= 0.91 * mag_da * mag_db) &
                   (max_a >= 0.52) & (max_b >= 0.34))
        # Quand l'image coupe une partie du halo, deux rayons peuvent suffire.
        # Le canal dominant doit alors rester lumineux pendant la baisse de luma.
        support &= (visible == 8) | (max_b >= 0.93 * max_a)
        if radial_profile:
            # Un aplat peut n'atténuer que le canal faible. Sur un arc tronqué,
            # demander aussi que les deux canaux dominants divergent vers le fond.
            support &= (visible == 8) | (gap_c - gap_a >= 0.010)
        next_a = np.roll(a, 1, axis=0)
        adjacent = ((a * next_a).sum(axis=-1) >= 0.96 * mag_a * np.roll(mag_a, 1, axis=0))
        arc = support & np.roll(support, 1, axis=0) & adjacent
        rays = arc | np.roll(arc, -1, axis=0)
        share = rays.sum(axis=0) / np.maximum(visible, 1)
        accepted = ((visible == 8) & (share >= 0.625)) | ((visible >= 3) & (visible < 8) & (share >= 0.4))

        keep = accepted & (share > scores[cy, cx])
        cells = (cy[keep], cx[keep])
        summed = (a * rays[..., None]).sum(axis=0)[keep]
        hues[cells] = summed / np.maximum(np.linalg.norm(summed, axis=-1, keepdims=True), 1e-6)
        scores[cells] = share[keep]
    return GlowMap(pq_map, scores, hues, x, y)


def confirm_directional(current, previous):
    """N'autorise que des ancres actuelles, confirmées à une tuile près sur l'image source précédente."""
    if previous is None:
        return np.zeros_like(current.scores), 0.0
    if current.pq.shape != previous.pq.shape:
        raise ValueError("les images de la séquence doivent avoir la même taille")
    delta = float(np.abs(current.pq - previous.pq).mean())
    if delta >= 0.075:
        return np.zeros_like(current.scores), delta

    h, w = current.scores.shape
    scores = np.pad(previous.scores, 1)
    hues = np.pad(previous.hues, ((1, 1), (1, 1), (0, 0)))
    agreed = np.zeros((h, w), dtype=bool)
    for oy in range(3):
        for ox in range(3):
            same_hue = (current.hues * hues[oy:oy + h, ox:ox + w]).sum(axis=-1) >= 0.90
            agreed |= (scores[oy:oy + h, ox:ox + w] > 0) & same_hue
    return current.scores * agreed, delta


def directional_maps(sources, block, radial_profile, jobs):
    def extract(path):
        rgb = read_ppm16(path, strict16=False)
        return rgb.shape[:2], directional_map(rgb, block, radial_profile)

    if jobs == 1:
        yield from map(extract, sources)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for start in range(0, len(sources), jobs):
            # Au plus jobs images en mémoire, remises dans l'ordre des PTS.
            yield from pool.map(extract, sources[start:start + jobs])


def measure_directional(sources, regions, upscale, radial_profile=False, jobs=1):
    block = 32 // upscale
    previous = None
    expected = None
    result = []
    for path, (shape, current) in zip(sources, directional_maps(sources, block, radial_profile, jobs)):
        if expected is not None and shape != expected:
            raise ValueError("les images de la séquence doivent avoir la même taille")
        expected = shape
        h, w = shape
        confirmed, delta = confirm_directional(current, previous)
        previous = current
        row = {"source": str(path), "delta_pq": delta,
               "ancres_brutes": int(np.count_nonzero(current.scores)),
               "ancres_confirmées": int(np.count_nonzero(confirmed)), "régions": {}}
        for name, (left, top, width, height) in regions:
            if not (0 <= left < left + width <= w and 0 <= top < top + height <= h):
                raise ValueError(f"{name} : région hors de l'image {w} × {h}")
            x, y = current.x, current.y
            inside = (left <= x) & (x < left + width) & (top <= y) & (y < top + height)
            cells = (y[inside] // block, x[inside] // block)
            row["régions"][name] = {"cœurs": int(inside.sum()),
                                   "ancres_brutes": int(np.count_nonzero(current.scores[cells])),
                                   "ancres_confirmées": int(np.count_nonzero(confirmed[cells]))}
        result.append(row)
    return result


def region(text):
    try:
        name, rect = text.split("=", 1)
        x, y, w, h = (int(v) for v in rect.split(","))
        if name and w > 0 and h > 0:
            return name, (x, y, w, h)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("région attendue : NOM=X,Y,LARGEUR,HAUTEUR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", help="PPM RGB SDR BT.709 déployé, dans l'ordre source")
    parser.add_argument("--directionnel", action="store_true",
                        help="ancres de couronne à 1/32, confirmées par l'image précédente si fournie")
    parser.add_argument("--profil-couronne", action="store_true",
                        help="écarte les arcs au bord dont les deux canaux dominants évoluent ensemble")
    parser.add_argument("--region", required=True, type=region, action="append")
    parser.add_argument("--jobs", type=int, default=1, help="nombre de cartes source calculées en parallèle")
    parser.add_argument("--upscale", choices=(1, 2), type=int, default=2)
    parser.add_argument("--rayon", type=int, action="append")
    args = parser.parse_args()
    radii = args.rayon or [8, 16, 32]
    if any(r <= 0 for r in radii):
        parser.error("les rayons doivent être positifs")
    if not args.directionnel and len(args.source) != 1:
        parser.error("plusieurs images demandent --directionnel")
    if args.jobs < 1:
        parser.error("--jobs doit être positif")
    if args.jobs != 1 and not args.directionnel:
        parser.error("--jobs demande --directionnel")
    if args.directionnel and args.rayon:
        parser.error("--rayon ne s'applique pas au mode directionnel")
    if args.profil_couronne and not args.directionnel:
        parser.error("--profil-couronne demande --directionnel")
    try:
        if args.directionnel:
            stats = measure_directional(args.source, args.region, args.upscale, args.profil_couronne, args.jobs)
        else:
            stats = measure(read_ppm16(args.source[0], strict16=False), args.region, args.upscale, radii)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
