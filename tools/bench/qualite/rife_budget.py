#!/usr/bin/env python3
"""Recherche hors ligne des paires où le milieu RIFE pourrait suffire.

  python3 -I tools/bench/qualite/rife_budget.py DOSSIER/taches.tsv [--limite 24] [--json rapport.json]

Les lignes TSV de carte_rejeu donnent largeur, hauteur, lumas A et B, dump NVOF, sortie de carte
(facultative). Les chemins sont relatifs au TSV. Pour de nouveaux dumps, nvof_dump --grid 4
--preset medium --cost écrit les deux sens. Les anciens dumps aller seul restent lisibles, mais
ne peuvent proposer aucune suppression de passe. Avec --reseau et --video, rejoue les candidats
contre RIFE à chaque instant par ONNX Runtime. Le début vient d'extrait.txt ou de --debut. Les lumas
de la vidéo sont comparées aux captures avant la mesure pour retrouver exactement les mêmes images.
--temoin LIGNE mesure une paire refusée, par exemple pour comprendre ce qui fait perdre le milieu.
--diagnostic-luma mesure sans la vidéo, à partir des lumas recopiées en RGB.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvof_mesures import at_cells, sample, texture  # noqa: E402

COST_HIGH = 24  # kSceneCostHigh, carte de la scène
CLOSURE_MAX_720 = 1.0  # hypothèses de départ pour la mesure, à confronter à RIFE
TRANSLATION_MAX_720 = 1.0
PHOTO_MAX = 8.0  # codes de luma
MIN_CONTOURS = 16  # cellules de traits dans la partie commune aux images
MIN_OVERLAP = 0.80

MID = "milieu à comparer"
ALL = "paires à garder en référence complète"
ONE_WAY = "flux aller seul"
HELD = "lumas identiques"


@dataclass(frozen=True)
class Direction:
    flow: np.ndarray
    cost: np.ndarray


def read_dump(path, width, height, grid):
    """Un ou deux ensembles vecteurs S10.5 puis carte de coût, tels qu'écrits par nvof_dump."""
    rows = (height + grid - 1) // grid
    cols = (width + grid - 1) // grid
    cells = rows * cols
    segment = 5 * cells
    data = path.read_bytes()
    if len(data) not in (segment, 2 * segment):
        raise ValueError(f"{path} : {len(data)} octets, {segment} ou {2 * segment} attendus (avec les coûts)")

    def direction(offset):
        flow = np.frombuffer(data, "<i2", cells * 2, offset).reshape(rows, cols, 2)
        cost = np.frombuffer(data, np.uint8, cells, offset + 4 * cells).reshape(rows, cols)
        return Direction(flow.transpose(2, 0, 1).astype(np.float32) / 32, cost)

    return direction(0), direction(segment) if len(data) == 2 * segment else None


def read_luma(path, width, height):
    pixels = np.fromfile(path, np.uint8)
    if pixels.size != width * height:
        raise ValueError(f"{path} : luma de {pixels.size} octets, {width * height} attendus")
    return pixels.reshape(height, width)


def coordinates(width, height):
    return np.arange(width, dtype=np.float32)[None, :], np.arange(height, dtype=np.float32)[:, None]


def inside(x, y, width, height):
    return (0 <= x) & (x <= width - 1) & (0 <= y) & (y <= height - 1)


def photo_at_cells(a, b, flow, grid):
    """Pire écart de luma d'une cellule recalée. Un trait fin peut manquer au centre de la cellule."""
    h, w = a.shape
    x, y = coordinates(w, h)
    fx, fy = np.repeat(np.repeat(flow, grid, axis=1), grid, axis=2)[:, :h, :w]
    qx, qy = x + fx, y + fy
    error = np.abs(a.astype(np.float32) - sample(b.astype(np.float32), qx, qy))
    error = np.where(inside(qx, qy, w, h), error, 0)
    return np.maximum.reduceat(np.maximum.reduceat(error, np.arange(0, w, grid), axis=1),
                               np.arange(0, h, grid), axis=0)


def cell_centers(size, grid):
    """Le dernier bloc peut être tronqué par le bord, comme dans nvof_mesures.at_cells."""
    return np.minimum(np.arange(0, size, grid, dtype=np.float32) + grid // 2, size - 1)


def direction_report(a, b, direction, reverse, grid):
    h, w = a.shape
    fx, fy = direction.flow
    rows, cols = fx.shape
    cx, cy = cell_centers(w, grid), cell_centers(h, grid)
    qx, qy = cx[None, :] + fx, cy[:, None] + fy
    overlap = inside(qx, qy, w, h)
    contours = (at_cells(texture(a), grid, rows, cols) == 1) & overlap
    supported = contours & (direction.cost < COST_HIGH)
    scale = 720.0 / h

    photo = photo_at_cells(a, b, direction.flow, grid)
    if supported.any():
        dx, dy = np.median(fx[supported]), np.median(fy[supported])
        spread = np.hypot(fx - dx, fy - dy) * scale
        translation = [float(dx * scale), float(dy * scale)]
    else:
        spread = np.full(fx.shape, np.inf, dtype=np.float32)
        translation = None
    reasons = {
        "coût": direction.cost >= COST_HIGH,
        "luma recalée": photo > PHOTO_MAX,
        "translation": spread > TRANSLATION_MAX_720,
    }
    closure_p95 = None
    if reverse is not None:
        rx = np.interp(qx, cx, np.arange(cols))
        ry = np.interp(qy, cy, np.arange(rows))
        bx, by = [sample(plane, rx, ry) for plane in reverse.flow]
        # Une interpolation des coûts pourrait masquer une cellule coûteuse. Garder le voisin le plus proche.
        return_cost = reverse.cost[np.rint(ry).astype(int), np.rint(rx).astype(int)]
        closure = np.hypot(fx + bx, fy + by) * scale
        reasons["retour coûteux"] = return_cost >= COST_HIGH
        reasons["aller-retour"] = closure > CLOSURE_MAX_720
        if contours.any():
            closure_p95 = float(np.percentile(closure[contours], 95))
    return {
        "recouvrement": float(overlap.mean()),
        "contours": int(contours.sum()),
        "contours_douteux": int((contours & np.logical_or.reduce(list(reasons.values()))).sum()),
        "refus_par_cause": {name: int((contours & mask).sum()) for name, mask in reasons.items()},
        "translation_720": translation,
        "fermeture_p95_720": closure_p95,
        "photo_max_contours": float(photo[contours].max()) if contours.any() else None,
    }


def examine(a, b, forward, reverse, grid):
    adv = direction_report(a, b, forward, reverse, grid)
    ret = direction_report(b, a, reverse, forward, grid) if reverse is not None else None
    if np.array_equal(a, b):
        choice = HELD
    elif ret is None:
        choice = ONE_WAY
    elif all(d["recouvrement"] >= MIN_OVERLAP and d["contours"] >= MIN_CONTOURS
             and d["contours_douteux"] == 0 for d in (adv, ret)):
        choice = MID
    else:
        choice = ALL
    return {"proposition": choice, "aller": adv, "retour": ret}


def run(tasks, grid, limit):
    rows = []
    with tasks.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            try:
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) not in (5, 6):
                    raise ValueError("cinq ou six colonnes attendues")
                w, h = map(int, fields[:2])
                if w <= grid or h <= grid:
                    raise ValueError("il faut au moins deux cellules par axe")
                first, second, dump = [tasks.parent / name for name in fields[2:5]]
                forward, reverse = read_dump(dump, w, h, grid)
                a, b = read_luma(first, w, h), read_luma(second, w, h)
                rows.append({"ligne": line, "paire": [fields[2], fields[3]], "taille_luma": [w, h],
                             **examine(a, b, forward, reverse, grid)})
            except (ValueError, OSError) as error:
                raise ValueError(f"{tasks}:{line} : {error}") from error
            if len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"{tasks} : aucune paire")
    return rows


def video_positions(tasks, video, rows, start):
    """Image RGB à la position de chaque luma du dump, vérifiée avant le rejeu coloré."""
    from nvof_mesures import decode, probe

    if start is None:
        note = tasks.parent / "extrait.txt"
        start = next((line.split("\t", 1)[1] for line in note.read_text(encoding="utf-8").splitlines()
                      if line.startswith("debut\t")), None)
        if start is None:
            raise ValueError(f"{note} : début manquant, passer --debut")

    paths = {name for row in rows for name in row["paire"]}
    indices = {name: int(Path(name).stem) for name in paths}
    if min(indices.values()) < 0:
        raise ValueError("les lumas doivent porter le numéro positif ou nul de l'image extraite")
    sizes = {tuple(row["taille_luma"]) for row in rows}
    if len(sizes) != 1:
        raise ValueError("une seule taille de luma par extrait")
    width, height = sizes.pop()
    positions = sorted(set(indices.values()))
    lumas = decode(video, start, positions[-1] + 1, height)
    for name, index in indices.items():
        if index >= len(lumas) or not np.array_equal(lumas[index], read_luma(tasks.parent / name, width, height)):
            raise ValueError(f"{name} : luma différente après extraction, vérifier la vidéo et --debut")
    sw, sh, _ = probe(video)
    pick = "+".join(f"eq(n,{i})" for i in positions)
    command = ["ffmpeg", "-v", "error", "-ss", start, "-i", video, "-an", "-sn", "-vf",
               f"select='{pick}',format=rgb24", "-frames:v", str(len(positions)), "-fps_mode",
               "passthrough", "-f", "rawvideo", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    if len(raw) != len(positions) * sw * sh * 3:
        raise ValueError("l'extraction RGB ne contient pas toutes les images choisies")
    rgb = np.frombuffer(raw, np.uint8).reshape(len(positions), sh, sw, 3)
    by_index = dict(zip(positions, rgb))
    return {name: by_index[i].transpose(2, 0, 1).astype(np.float32) / 255.0 for name, i in indices.items()}


def gray_positions(tasks, rows):
    result = {}
    for row in rows:
        w, h = row["taille_luma"]
        for name in row["paire"]:
            if name not in result:
                luma = read_luma(tasks.parent / name, w, h)
                result[name] = np.repeat(luma[None], 3, axis=0).astype(np.float32) / 255.0
    return result


def edge_error(ref, approx):
    """Écart sur les contours des deux images, y compris un bord chromatique à luma égale."""
    mask = np.zeros(ref.shape[1:], dtype=bool)
    for image in (ref, approx):
        horizontal = (np.abs(np.diff(image, axis=2)) > 16 / 255).any(axis=0)
        vertical = (np.abs(np.diff(image, axis=1)) > 16 / 255).any(axis=0)
        mask[:, :-1] |= horizontal
        mask[:, 1:] |= horizontal
        mask[:-1] |= vertical
        mask[1:] |= vertical
    return float(255 * np.percentile(np.abs(ref[:, mask] - approx[:, mask]), 99.9)) if mask.any() else None


def compare_instants(estimate, a, b, count):
    """Simule une passe centrale ou trois passes (milieu et deux instants extrêmes)."""
    import rife_instants
    from rife_taille import blend, worst_block

    flow, mask, _ = estimate(0.5)
    rows = []
    for i in range(1, count + 1):
        t = i / (count + 1)
        if t == 0.5:
            exact = derived = blend(a, b, flow, mask)
        else:
            direct_flow, direct_mask, _ = estimate(t)
            exact = blend(a, b, direct_flow, direct_mask)
            derived = rife_instants.blend_at(a, b, flow, mask, t)
        error = rife_instants.mesures.ecarts(exact, derived)
        rows.append({
            "t": t,
            "exact_en_trois_passes": i in (1, count) or t == 0.5,
            "psnr_milieu": rife_instants.mesures.psnr(exact, derived),
            "pire_bloc_milieu": worst_block(exact, derived, n=min(64, *a.shape[1:])),
            "p999_codes": error["p999"],
            "max_codes": error["pire"],
            "p999_contours_codes": edge_error(exact, derived),
        })
    return rows


def quality(tasks, rows, args):
    import rife_instants
    from rife_taille import motion_session

    chosen = [r for r in rows if r["proposition"] == MID][:args.limite_qualite]
    witnesses = {r["ligne"]: r for r in rows if r["proposition"] == ALL}
    for line in args.temoin:
        if line not in witnesses:
            raise ValueError(f"--temoin {line} : aucune paire aller-retour écartée à cette ligne")
    selected = chosen + [witnesses[line] for line in dict.fromkeys(args.temoin)]
    if not selected:
        return 0
    rgb = (video_positions(tasks, args.video, selected, args.debut) if args.video is not None
           else gray_positions(tasks, selected))
    sess, names = motion_session(args.reseau)
    align = 128 if args.reseau.name.startswith("rife_v4.25_lite") else rife_instants.modeles.ALIGN
    for row in selected:
        a, b = (rgb[name] for name in row["paire"])

        def estimate(t):
            return rife_instants.estimate_at(sess, names, a, b, t, align, args.hauteur_rife)

        instants = compare_instants(estimate, a, b, args.instants)
        remain = [i for i in instants if not i["exact_en_trois_passes"]]
        one_pass = min(i["psnr_milieu"] for i in instants)
        three_passes = min((i["psnr_milieu"] for i in remain), default=200.0)
        row["comparaison"] = {
            "entrée": "vidéo RGB (lumas vérifiées)" if args.video is not None else "diagnostic luma",
            "taille": [a.shape[2], a.shape[1]],
            "hauteur_rife": min(a.shape[1], args.hauteur_rife),
            "contrôle": row["proposition"] == ALL,
            "psnr_pire_une_passe": one_pass,
            "psnr_pire_trois_passes": three_passes,
            "instants": instants,
        }
        print(f"  ligne {row['ligne']} : {one_pass:.1f} dB au pire instant en une passe, "
              f"{three_passes:.1f} dB en trois passes, {max(i['max_codes'] for i in instants):.1f} codes au pire pixel")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("taches", type=Path)
    parser.add_argument("--grille", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--limite", type=int, default=0, help="nombre maximal de paires, 0 pour toutes")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--reseau", type=Path, help="fichier ONNX d'un RIFE livré (11 plans)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="source originale correspondant aux dumps")
    source.add_argument("--diagnostic-luma", action="store_true", help="rejouer la luma en trois canaux")
    parser.add_argument("--debut", help="début de l'extraction vidéo si différent d'extrait.txt")
    parser.add_argument("--instants", type=int, choices=range(3, 11), default=6)
    parser.add_argument("--hauteur-rife", type=int, default=720)
    parser.add_argument("--limite-qualite", type=int, default=4)
    parser.add_argument("--temoin", action="append", type=int, default=[])
    args = parser.parse_args()
    if args.limite < 0:
        parser.error("--limite doit être positive ou nulle")
    if args.hauteur_rife < 2 or args.limite_qualite < 0:
        parser.error("hauteur RIFE ou limite de qualité invalide")
    has_source = args.video is not None or args.diagnostic_luma
    if (args.reseau is not None) != has_source:
        parser.error("la comparaison demande --reseau et --video ou --diagnostic-luma")
    if args.temoin and args.reseau is None:
        parser.error("--temoin demande --reseau")
    try:
        rows = run(args.taches, args.grille, args.limite)
        if args.reseau is not None:
            measured = quality(args.taches, rows, args)
            print(f"{measured} paires comparées à RIFE à chaque instant")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")
    summary = Counter(row["proposition"] for row in rows)
    print(f"{len(rows)} paires, {sum(row['retour'] is not None for row in rows)} flux aller-retour")
    for label in (MID, ALL, ONE_WAY, HELD):
        print(f"  {label} : {summary[label]}")
    if args.json:
        args.json.write_text(json.dumps({"résumé": summary, "paires": rows}, ensure_ascii=False,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
