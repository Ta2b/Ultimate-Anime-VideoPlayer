#!/usr/bin/env python3
"""Vidéo côte à côte des dessins tenus lissés (« Lisser les dessins tenus », smooth_hold_u de src/app/play/rife_plan.h)
par RIFE 4.26 et par GMFSS Fortuna Union : l'usage pour lequel GMFSS est recommandé (entre deux dessins distincts d'un
anime en deux ou en trois), où aucune vraie image intermédiaire n'existe. Seul l'œil juge.

  python3 -I tools/bench/qualite/gmfss_tenus.py --video EXTRAIT_1080P.mkv --gmfss roue/vsgmfss_fortuna \\
      --reseau rife_v4.26.onnx --debut 160 --images 140 --sortie tenus.mp4

Un dessin tenu de n images (2 à 6, comme kSmoothMaxHold) suivi d'un autre dessin du même plan : l'image de rang r de
la tenue devient l'instant r/n entre les deux dessins. Une image qui change à chaque image, une pose plus longue ou une
coupe reste telle quelle. Tout se fait à la taille de travail (720p) ; la sortie met RIFE à gauche, GMFSS à droite
(960 × 540 chacun), à vitesse réelle, puis la même chose ralentie de moitié.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmfss_instants as gi  # noqa: E402
import rife_taille as rt  # noqa: E402
from sources import differs  # noqa: E402

MAX_HOLD = 6  # kSmoothMaxHold
CUT = 0.12


def decode(video, first, count, height):
    width = int(round(1920 * height / 1080 / 2)) * 2
    chain = f"trim=start_frame={first},setpts=PTS-STARTPTS,scale={width}:{height}:flags=area"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", chain, "-frames:v", str(count),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    return [f.transpose(2, 0, 1).astype(np.float32) / 255.0 for f in frames]


def plan(frames):
    """Pour chaque image : None (telle quelle) ou (début du dessin, début du suivant, u)."""
    starts = [0] + [i for i in range(1, len(frames)) if differs(frames[i - 1], frames[i])]
    out = [None] * len(frames)
    for d, nxt in zip(starts, starts[1:]):
        hold = nxt - d
        if not 2 <= hold <= MAX_HOLD or float(np.abs(frames[d] - frames[nxt]).mean()) > CUT:
            continue
        for r in range(1, hold):
            out[d + r] = (d, nxt, r / hold)
    return out


def slow_plan(frames, factor):
    """Ralenti ×factor entre dessins distincts (les tenues retirées, comme une déduplication) : chaque dessin, puis
    factor − 1 instants vers le suivant ; une coupe garde le dessin seul. Liste de (indice d'image, None) ou
    (dessin, suivant, u)."""
    starts = [0] + [i for i in range(1, len(frames)) if differs(frames[i - 1], frames[i])]
    out = []
    for d, nxt in zip(starts, starts[1:]):
        out.append((d, None))
        if float(np.abs(frames[d] - frames[nxt]).mean()) <= CUT:
            out += [(None, (d, nxt, k / factor)) for k in range(1, factor)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--gmfss", required=True)
    p.add_argument("--reseau", required=True)
    p.add_argument("--debut", type=int, default=0)
    p.add_argument("--images", type=int, default=120)
    p.add_argument("--hauteur", type=int, default=720)
    p.add_argument("--device", default="cpu")
    p.add_argument("--union-neuf", help="le train_log/ du « new union model » : ancien union à gauche, nouveau à droite")
    p.add_argument("--ralenti", type=int, default=0,
                   help="ralenti ×N entre dessins distincts : RIFE 4.26 à gauche, à droite une fusion union au "
                        "milieu puis RIFE 4.26 sur chaque demi-paire (N pair)")
    p.add_argument("--droite", choices=("melange", "union"), default="melange",
                   help="avec --ralenti : à droite le mélange (union au milieu + RIFE) ou union à chaque instant")
    p.add_argument("--sortie", required=True)
    args = p.parse_args()
    import torch

    torch.set_grad_enabled(False)
    frames = decode(args.video, args.debut, args.images, args.hauteur)
    if args.ralenti:
        steps = slow_plan(frames, args.ralenti)
    else:
        steps = [(i, step) for i, step in enumerate(plan(frames))]
    print(f"{len(frames)} images, {sum(s is not None for _, s in steps)} images créées", flush=True)
    nets = gi.load(args.gmfss, args.device, 1.0)
    right_nets = gi.load(args.gmfss, args.device, 1.0, args.union_neuf) if args.union_neuf else None
    titles = ("GMFSS union", "GMFSS new union") if right_nets else ("RIFE 4.26", "GMFSS union")
    if args.ralenti:
        titles = ("RIFE 4.26", "union au milieu + RIFE 4.26" if args.droite == "melange" else "GMFSS union")
    rife = rt.motion_session(args.reseau)
    pairs = {}
    w, h = 960, 540
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}",
                            "-r", "24000/1001", "-i", "-", "-vf",
                            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{titles[0]}':"
                            "x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5,"
                            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{titles[1]}':"
                            f"x={w + 20}:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5",
                            "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p",
                            args.sortie + ".tmp.mp4"], stdin=subprocess.PIPE)
    for n, (i, step) in enumerate(steps):
        if step is None:
            left = right = frames[i]
        else:
            d, nxt, u = step
            a, b = frames[d], frames[nxt]
            key = (d, nxt)
            if key not in pairs:
                (pa, window), (pb, _) = gi.padded(a), gi.padded(b)
                pairs = {key: (gi.Gmfss(nets, pa, pb), gi.Gmfss(right_nets, pa, pb) if right_nets else None, window)}
            g, g_right, window = pairs[key]
            if args.ralenti and args.droite == "melange":
                mid = g.middle().cpu().numpy()[window]
                half = (a, mid, 2 * u) if u < 0.5 else (mid, b, 2 * u - 1)
                right = mid if u == 0.5 else gi.rife_at(*rife, *half)[0]
            else:
                right = g.at(u).cpu().numpy()[window]
            if g_right is not None:
                left, right = right, g_right.at(u).cpu().numpy()[window]
            else:
                left = gi.rife_at(*rife, a, b, u)[0]
        row = np.concatenate([rt.resize(np.clip(x, 0, 1), w, h, rt._area_axis) for x in (left, right)], 2)
        enc.stdin.write((row.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8).tobytes())
        print(f"{n + 1}/{len(steps)}", flush=True)
    enc.stdin.close()
    enc.wait()
    if args.ralenti:
        os.replace(args.sortie + ".tmp.mp4", args.sortie)
        print(args.sortie)
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.sortie + ".tmp.mp4", "-filter_complex",
                    "[0:v]split[a][b];[b]setpts=2*PTS[s];[a][s]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "libx264",
                    "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", args.sortie], check=True)
    os.remove(args.sortie + ".tmp.mp4")
    print(args.sortie)


if __name__ == "__main__":
    main()
