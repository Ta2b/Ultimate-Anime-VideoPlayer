#!/usr/bin/env python3
"""Ce que coûte à l'image un instant déduit du milieu de la paire plutôt qu'estimé par RIFE à cet instant, selon le
mouvement de la paire (processeur, ONNX Runtime).

  python3 -I tools/bench/qualite/rife_instants.py --video EXTRAIT_1080P.mkv --reseau rife_v4.22_lite.onnx

Le lecteur estime le milieu de chaque paire ; sous kSingleRunMotion720 (15 px à 720p), les autres instants s'en
déduisent (warp_blend_at : chaque point en ligne droite, masque déplacé par la loi de mélange), au-dessus RIFE
repasse à chaque instant. Ce banc mesure, sur un vrai épisode, l'écart entre les deux, paire par paire : dans une
suite de quatre images 1080p qui diffèrent toutes (k à k+3, l'anime animé « en uns »), k et k+3 forment la paire,
k+1 et k+2 sont les vérités à t = 1/3 et 2/3. Le mouvement est estimé à 720p (kRifeMotionHeight) puis remonté en
1080p, comme dans rife_taille.py. Pour chaque instant : PSNR de RIFE à l'instant et de l'instant déduit du milieu
contre la vraie image, et pire bloc 64 × 64. Le déplacement que la déduction prolonge vaut 2·|t − ½| fois le
mouvement de la paire (99,9e centile de |flot| au milieu, en pixels de 720p) : la colonne « prolongé ».

Il faut python3 avec numpy, onnx, onnxruntime, et ffmpeg. Résultats et lecture : docs/interpolation.md (Une seule
passe RIFE par paire calme).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import rife_taille as rt  # noqa: E402

BLEND_LAW_B = 2.591  # kBlendLawB (kernels/spec/warp.h) ; A = 1, C = 0
MASK_EPSILON = 1e-4
CUT = 0.12  # écart moyen au-delà duquel deux images sont de plans différents (sources.windows)


def estimate_at(sess, names, a, b, t, align):
    """Flot (4 plans) et masque de RIFE à l'instant t, estimés à 720p et remontés en 1080p."""
    width = int(round(rt.WIDTH * rt.REFERENCE / rt.HEIGHT / 2)) * 2
    sa, sb = rt.resize(a, width, rt.REFERENCE, rt._area_axis), rt.resize(b, width, rt.REFERENCE, rt._area_axis)
    (pa, (top, left)), (pb, _) = modeles.pad(sa, align), modeles.pad(sb, align)
    x = modeles.rife_input(pa, pb, t, 11).astype(np.float16)
    _, flow, mask = sess.run(names, {sess.get_inputs()[0].name: x})
    window = (slice(None), slice(top, top + rt.REFERENCE), slice(left, left + width))
    flow = flow[0].astype(np.float32)[window]
    mask = 1.0 / (1.0 + np.exp(-mask[0].astype(np.float32)[window]))
    up = rt.resize(flow, rt.WIDTH, rt.HEIGHT, rt._linear_axis)
    up[0::2] *= rt.WIDTH / width
    up[1::2] *= rt.HEIGHT / rt.REFERENCE
    biggest = float(np.percentile(np.abs(flow).max(0), 99.9))
    return up, rt.resize(mask, rt.WIDTH, rt.HEIGHT, rt._linear_axis), biggest


def blend_at(a, b, flow, mask, t):
    """warp_blend_at (kernels/spec/warp.h) : l'instant t depuis le mouvement du milieu."""
    mx = (1 - t) * flow[0] + t * flow[2]
    my = (1 - t) * flow[1] + t * flow[3]
    planes = rt.warp(np.concatenate([flow, mask]), -mx, -my)  # flot et masque lus où le point était au milieu
    dx, dy = planes[2] - planes[0], planes[3] - planes[1]
    mm = np.clip(planes[4:5], MASK_EPSILON, 1 - MASK_EPSILON)
    logit = np.log(mm / (1 - mm)) + BLEND_LAW_B * np.log((1 - t) / t)
    m = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(rt.warp(a, -t * dx, -t * dy) * m + rt.warp(b, (1 - t) * dx, (1 - t) * dy) * (1 - m), 0, 1)


def quadruples(frames, count):
    """count suites k..k+3 d'images toutes différentes, d'un même plan, réparties dans l'extrait."""
    from sources import differs
    cand = []
    for k in range(len(frames) - 3):
        win = frames[k:k + 4]
        if all(differs(win[i], win[i + 1]) for i in range(3)) and float(np.abs(win[0] - win[3]).mean()) < CUT:
            cand.append(k)
    return [(k, frames[k:k + 4]) for k in cand[::max(1, len(cand) // count)][:count]]


def measure(sess, names, align, win):
    a, b = win[0], win[3]
    mid_flow, mid_mask, biggest = estimate_at(sess, names, a, b, 0.5, align)
    row = {"mouvement_720": biggest}
    for i, t in ((1, 1 / 3), (2, 2 / 3)):
        flow, mask, _ = estimate_at(sess, names, a, b, t, align)
        exact = rt.blend(a, b, flow, mask)
        derived = blend_at(a, b, mid_flow, mid_mask, t)
        row[f"rife_{i}"] = mesures.psnr(win[i], exact)
        row[f"deduit_{i}"] = mesures.psnr(win[i], derived)
        row[f"pire_bloc_rife_{i}"] = rt.worst_block(win[i], exact)
        row[f"pire_bloc_deduit_{i}"] = rt.worst_block(win[i], derived)
    return row


def summary(rows):
    lines = ["| Mouvement (720p) | prolongé à t = ⅓ | Instants | RIFE (dB) | Déduit − RIFE : moyenne | pire | "
             "Pire bloc, écart moyen |", "|---|---:|---:|---:|---:|---:|---:|"]
    edges = (0, 15, 30, 45, 60, 90, 1e9)
    for lo, hi in zip(edges, edges[1:]):
        group = [r for r in rows if lo <= r["mouvement_720"] < hi]
        if not group:
            continue
        delta = [r[f"deduit_{i}"] - r[f"rife_{i}"] for r in group for i in (1, 2)]
        block = [r[f"pire_bloc_deduit_{i}"] - r[f"pire_bloc_rife_{i}"] for r in group for i in (1, 2)]
        rife = np.mean([r[f"rife_{i}"] for r in group for i in (1, 2)])
        label = f"{lo} à {hi} px" if hi < 1e9 else f"{lo} px et plus"
        span = f"{lo / 3:.0f} à {hi / 3:.0f} px" if hi < 1e9 else f"{lo / 3:.0f} px et plus"
        lines.append(f"| {label} | {span} | {len(delta)} | {rife:.2f} | {np.mean(delta):+.2f} | {min(delta):+.2f} "
                     f"| {np.mean(block):+.2f} |".replace(".", ","))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, help="un extrait 1080p (24 i/s)")
    p.add_argument("--reseau", action="append", required=True, help="un RIFE livré (11 plans), plusieurs possibles")
    p.add_argument("--suites", type=int, default=40)
    p.add_argument("--images", type=int, default=740, help="images lues au début de l'extrait")
    p.add_argument("--json")
    p.add_argument("--md")
    args = p.parse_args()
    wins = quadruples(rt.decode(args.video, args.images), args.suites)
    results, text = {}, ""
    for path in args.reseau:
        name = os.path.basename(path)
        sess, names = rt.motion_session(path)
        align = 128 if name.startswith("rife_v4.25_lite") else modeles.ALIGN
        rows = []
        for k, win in wins:
            rows.append({"image": k, **measure(sess, names, align, win)})
            print(name, json.dumps({key: round(v, 2) if isinstance(v, float) else v for key, v in rows[-1].items()}),
                  flush=True)
        results[name] = rows
        text += f"\n## {name}\n\n{len(rows)} paires (k, k + 3), instants 1/3 et 2/3\n\n" + summary(rows)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


if __name__ == "__main__":
    main()
