#!/usr/bin/env python3
"""Ce que coûte à l'image RIFE estimé plus petit que l'image qu'il déforme, sans carte graphique (ONNX Runtime).

  python3 -I tools/bench/qualite/rife_taille.py --video EXTRAIT_1080P.mkv --reseau rife_v4.26.onnx [--json r.json]

Le lecteur estime le mouvement à 720p dès que la chaîne travaille plus haut (kRifeMotionHeight), et l'échelle par
paire (pair_scale) l'estime à la moitié de cette taille sur les paires qui bougent fort. Ce banc mesure ces choix
sur un vrai épisode. Pour chaque fenêtre k, k+1, k+2 d'une vidéo 1080p où les trois images diffèrent :

  - k et k+2 sont réduites à la hauteur H (moyenne de surface, comme le filtre Area du lecteur) ;
  - RIFE en donne le flot et le masque (les sorties internes du graphe livré, avant son propre warp) ;
  - le flot remonte en bilinéaire à 1080p, ses valeurs mises à l'échelle, le masque aussi ;
  - k et k+2 en 1080p sont déformées et mélangées comme l'étage de warp (bilinéaire, bords répétés) ;
  - la sortie se compare à la vraie k+1, et à celle de l'estimation en 720p (le lecteur aujourd'hui).

Le mouvement de la paire (le 99,9e centile de |flot| en pixels de l'estimation en 720p, comme k_flow_max) dit si
elle passe le seuil de l'échelle par paire (single_run_motion : 15 px à 720p). « recompose » vérifie le banc : à
pleine taille, le warp d'ici contre la sortie du graphe entier (attendu au-dessus de 50 dB).

Il faut python3 avec numpy, onnx, onnxruntime, et ffmpeg. Résultats et lecture : docs/interpolation.md (Échelle par
paire, Mesure sur processeur).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import sources  # noqa: E402

HEIGHT, WIDTH = 1080, 1920
REFERENCE = 720  # kRifeMotionHeight
SINGLE_RUN_720 = 15.0  # kSingleRunMotion720


def motion_session(path):
    """Le graphe livré, avec le flot (pixels de l'entrée) et le logit du masque en sorties de plus."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    graph = model.graph
    mask = [n for n in graph.node if n.op_type == "Sigmoid"][-1].input[0]
    flow = [n for n in graph.node if n.op_type == "Reshape"][-1].input[0]
    for name in (flow, mask):
        graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, None))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess, [graph.output[0].name, flow, mask]


def _area_axis(n_in, n_out):
    """Poids de la moyenne de surface d'un axe (n_out ≤ n_in) : la part de chaque pixel d'entrée sous chaque sortie."""
    edges = np.arange(n_out + 1) * (n_in / n_out)
    w = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        lo, hi = edges[o], edges[o + 1]
        for i in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            w[o, i] = min(hi, i + 1) - max(lo, i)
    return w / w.sum(1, keepdims=True)


def _linear_axis(n_in, n_out):
    """Poids du bilinéaire d'un axe, centres demi-pixel, bords répétés."""
    centre = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    lo = np.floor(centre).astype(int)
    f = (centre - lo).astype(np.float32)
    w = np.zeros((n_out, n_in), np.float32)
    rows = np.arange(n_out)
    np.add.at(w, (rows, np.clip(lo, 0, n_in - 1)), 1.0 - f)
    np.add.at(w, (rows, np.clip(lo + 1, 0, n_in - 1)), f)
    return w


def resize(img, width, height, axis):
    """img (C × H × W) à width × height, par des poids séparables (_area_axis ou _linear_axis)."""
    _, h, w = img.shape
    return np.einsum("yh,chw,xw->cyx", axis(h, height), img, axis(w, width), optimize=True).astype(np.float32)


def warp(img, fx, fy):
    """Lecture bilinéaire de img en (x + fx, y + fy), bords répétés : le warp de RIFE (grille align_corners)."""
    _, h, w = img.shape
    x = np.clip(np.arange(w, dtype=np.float32)[None, :] + fx, 0, w - 1)
    y = np.clip(np.arange(h, dtype=np.float32)[:, None] + fy, 0, h - 1)
    x0, y0 = np.minimum(np.floor(x).astype(int), w - 2), np.minimum(np.floor(y).astype(int), h - 2)
    ax, ay = x - x0, y - y0
    top = img[:, y0, x0] * (1 - ax) + img[:, y0, x0 + 1] * ax
    bottom = img[:, y0 + 1, x0] * (1 - ax) + img[:, y0 + 1, x0 + 1] * ax
    return top * (1 - ay) + bottom * ay


def estimate(sess, names, a, b, align):
    """Sortie du graphe, flot (4 plans, pixels de a) et masque (après sigmoïde) de la paire, dans la fenêtre."""
    (pa, (top, left)), (pb, _) = modeles.pad(a, align), modeles.pad(b, align)
    x = modeles.rife_input(pa, pb, 0.5, 11).astype(np.float16)
    out, flow, mask = sess.run(names, {sess.get_inputs()[0].name: x})
    _, h, w = a.shape
    window = (slice(None), slice(top, top + h), slice(left, left + w))
    mask = mask[0].astype(np.float32)[window]
    return (np.clip(out[0].astype(np.float32), 0, 1)[window], flow[0].astype(np.float32)[window],
            1.0 / (1.0 + np.exp(-mask)))


def blend(a, b, flow, mask):
    return np.clip(warp(a, flow[0], flow[1]) * mask + warp(b, flow[2], flow[3]) * (1.0 - mask), 0, 1)


def at_height(sess, names, a, b, height, align):
    """(sortie du graphe entier ou None, image du milieu rendue en 1080p, flot estimé)."""
    if height >= HEIGHT:
        out, flow, mask = estimate(sess, names, a, b, align)
        return out, blend(a, b, flow, mask), flow
    width = int(round(WIDTH * height / HEIGHT / 2)) * 2
    sa, sb = resize(a, width, height, _area_axis), resize(b, width, height, _area_axis)
    _, flow, mask = estimate(sess, names, sa, sb, align)
    up = resize(flow, WIDTH, HEIGHT, _linear_axis)
    up[0::2] *= WIDTH / width
    up[1::2] *= HEIGHT / height
    return None, blend(a, b, up, resize(mask, WIDTH, HEIGHT, _linear_axis)), flow


def worst_block(ref, test, n=64):
    """PSNR du pire bloc n × n : là où un défaut se verrait."""
    _, h, w = ref.shape
    return min(mesures.psnr(ref[:, y:y + n, x:x + n], test[:, y:y + n, x:x + n])
               for y in range(0, h - n + 1, n) for x in range(0, w - n + 1, n))


def decode(video, count):
    """Les count premières images de la vidéo, en octets RGB (une image flottante par fenêtre seulement)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={WIDTH}:{HEIGHT}", "-frames:v",
                          str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True)
    frames = np.frombuffer(raw.stdout, np.uint8).reshape(-1, HEIGHT, WIDTH, 3)

    class Frames:
        def __len__(self):
            return len(frames)

        def __getitem__(self, k):
            if isinstance(k, slice):
                return [self[i] for i in range(*k.indices(len(frames)))]
            return frames[k].transpose(2, 0, 1).astype(np.float32) / 255.0

    return Frames()


def measure(sess, names, align, win, heights):
    a, truth, b = win[0], win[1], win[2]
    row, outs = {}, {}
    for height in heights:
        direct, out, flow = at_height(sess, names, a, b, height, align)
        outs[height] = out
        if direct is not None:
            row["recompose"] = mesures.psnr(direct, out)
        if height == REFERENCE:
            row["mouvement_720"] = float(np.percentile(np.abs(flow).max(0), 99.9))
        row[f"verite_{height}"] = mesures.psnr(truth, out)
        row[f"pire_bloc_{height}"] = worst_block(truth, out)
    for height in heights:
        if height != REFERENCE:
            row[f"contre_720_{height}"] = mesures.psnr(outs[REFERENCE], outs[height])
    return row


def summary(rows, heights):
    lines = ["| Estimé à | Paires | Vérité (dB) | Écart à 720p : moyenne | pire | Pire bloc, écart moyen |",
             "|---|---:|---:|---:|---:|---:|"]
    groups = (("toutes", rows), ("calmes (< 15 px)", [r for r in rows if r["mouvement_720"] < SINGLE_RUN_720]),
              ("fortes (≥ 15 px)", [r for r in rows if r["mouvement_720"] >= SINGLE_RUN_720]))
    for label, group in groups:
        if not group:
            continue
        for height in heights:
            delta = [r[f"verite_{height}"] - r[f"verite_{REFERENCE}"] for r in group]
            block = [r[f"pire_bloc_{height}"] - r[f"pire_bloc_{REFERENCE}"] for r in group]
            lines.append(f"| {height}p, {label} | {len(group)} | {np.mean([r[f'verite_{height}'] for r in group]):.2f} "
                         f"| {np.mean(delta):+.2f} | {min(delta):+.2f} | {np.mean(block):+.2f} |".replace(".", ","))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, help="un extrait 1080p (24 i/s)")
    p.add_argument("--reseau", action="append", required=True, help="un RIFE livré (11 plans), plusieurs possibles")
    p.add_argument("--hauteurs", default="1080,720,540,360")
    p.add_argument("--fenetres", type=int, default=30)
    p.add_argument("--images", type=int, default=740, help="images lues au début de l'extrait")
    p.add_argument("--json")
    p.add_argument("--md")
    args = p.parse_args()
    heights = sorted({int(v) for v in args.hauteurs.split(",")} | {REFERENCE}, reverse=True)
    wins = sources.windows(decode(args.video, args.images), args.fenetres)
    results, text = {}, ""
    for path in args.reseau:
        name = os.path.basename(path)
        sess, names = motion_session(path)
        align = 128 if name.startswith("rife_v4.25_lite") else modeles.ALIGN
        rows = []
        for label, win in wins:
            rows.append({"cas": label, **measure(sess, names, align, win, heights)})
            print(name, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in rows[-1].items()}),
                  flush=True)
        results[name] = rows
        checks = [r["recompose"] for r in rows if "recompose" in r]
        text += f"\n## {name}\n\n{len(rows)} fenêtres" + (f", recompose {np.mean(checks):.1f} dB" if checks else "")
        text += "\n\n"
        text += summary(rows, heights)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


if __name__ == "__main__":
    main()
