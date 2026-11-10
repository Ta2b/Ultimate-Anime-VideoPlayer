"""Chroma d'une source 4:4:4 : ce que perd sa réduction en 4:2:0 avant SPAN (audits A15-07 et A49-13).

Comme travail.py, tout se joue à l'échelle ½ : un plan 1080p de calibration.mkv est la vérité, sa moitié (moyenne
2 × 2) tient lieu de source 4:4:4, en Y'CbCr BT.709 10 bits. Trois chemins jusqu'à SPAN :

  « 4:2:0 d'avant »   swscale réduit la chroma en bilinéaire et la pose au centre en largeur, l'ingestion la
                      remonte en Spline36 en la croyant à gauche (le lecteur avant la correction) ;
  « 4:2:0 spline »    swscale la réduit en spline, arrondi exact, posée à gauche (ce que garde TensorRT) ;
  « 4:4:4 »           la chroma entière arrive à l'ingestion (Vulkan après la correction).

Chaque chemin se compare à la source (ce que l'ingestion rend), puis, agrandi ×2 par SPAN, au plan.

  python3 -I tools/bench/qualite/chroma444.py --modeles ~/.cache/ajp-release-models/modeles --md chroma444.md

La réduction est celle de swscale lui-même (ffmpeg en ligne de commande) ; la remontée Spline36 lit la chroma
à gauche en largeur et au centre en hauteur, comme l'ingestion du lecteur.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import sources  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SPAN = "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"
KR, KB = 0.2126, 0.0722  # BT.709
# Le filtre de swscale et la place qu'on lui donne pour la chroma (rien : la sienne, au centre en largeur).
SWS = {"4:2:0 d'avant": "flags=bilinear",
       "4:2:0 spline": "flags=spline+accurate_rnd:out_h_chr_pos=0:out_v_chr_pos=128"}


def to_yuv10(rgb):
    """R'G'B' [0, 1] (3, h, w) en Y'CbCr BT.709 plage limitée, codes 10 bits entiers."""
    r, g, b = rgb
    y = KR * r + (1 - KR - KB) * g + KB * b
    cb = (b - y) / (2 * (1 - KB))
    cr = (r - y) / (2 * (1 - KR))
    return np.stack([np.rint(64 + 876 * y), np.rint(512 + 896 * cb), np.rint(512 + 896 * cr)]).astype(np.uint16)


def to_rgb(y, cb, cr):
    y = (y - 64.0) / 876.0
    cb = (cb - 512.0) / 896.0
    cr = (cr - 512.0) / 896.0
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return np.clip(np.stack([r, g, b]), 0.0, 1.0).astype(np.float32)


def swscale_420(yuv, scale):
    """La chroma de yuv (4:4:4 10 bits) réduite par swscale : les plans Cb et Cr en (h/2, w/2)."""
    _, h, w = yuv.shape
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p10le", "-s", f"{w}x{h}", "-i", "-",
           "-vf", f"scale={scale}", "-pix_fmt", "yuv420p10le", "-f",
           "rawvideo", "-"]
    out = subprocess.run(cmd, input=yuv.astype("<u2").tobytes(), stdout=subprocess.PIPE, check=True).stdout
    data = np.frombuffer(out, dtype="<u2")
    chroma = data[h * w:].reshape(2, h // 2, w // 2)
    return chroma[0].astype(np.float32), chroma[1].astype(np.float32)


def spline36(x):
    x = np.abs(x)
    near = ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    mid = ((-6.0 / 11.0 * (x - 1) + 270.0 / 209.0) * (x - 1) - 156.0 / 209.0) * (x - 1)
    far = ((1.0 / 11.0 * (x - 2) - 45.0 / 209.0) * (x - 2) + 26.0 / 209.0) * (x - 2)
    return np.where(x < 1, near, np.where(x < 2, mid, np.where(x < 3, far, 0.0)))


def up_axis(plane, axis, shift):
    """plane doublé le long d'axis en Spline36 ; shift : la position, en points de sortie, du premier point
    de chroma (0 : à gauche, 0,5 : au centre). Bords répétés."""
    n = plane.shape[axis]
    at = (np.arange(2 * n) - shift) / 2.0
    base = np.floor(at).astype(int)
    out = np.zeros(np.moveaxis(plane, axis, 0).shape[1:] + (2 * n,), dtype=np.float64)
    moved = np.moveaxis(plane, axis, -1)
    total = np.zeros(2 * n)
    for tap in range(-2, 4):
        idx = base + tap
        weight = spline36(at - idx)
        out += moved[..., np.clip(idx, 0, n - 1)] * weight
        total += weight
    return np.moveaxis(out / total, -1, axis)


def ingest_420(y, cb, cr):
    """L'ingestion du lecteur : chroma 4:2:0 remontée en Spline36, à gauche en largeur, au centre en hauteur."""
    full = [up_axis(up_axis(c, 1, 0.0), 0, 0.5) for c in (cb, cr)]
    return to_rgb(y.astype(np.float32), full[0], full[1])


def run(args):
    plans = sources.stills(args.calibration, args.plans)
    if not plans:
        raise SystemExit(f"{args.calibration} : aucun plan lu")
    net = modeles.Network(os.path.join(args.modeles, "onnx", SPAN))
    rows = {}
    for plan in plans:
        yuv = to_yuv10(sources.halve(plan))  # la source 4:4:4, en 960 × 540
        whole = to_rgb(*yuv.astype(np.float32))
        paths = {name: ingest_420(yuv[0], *swscale_420(yuv, scale)) for name, scale in SWS.items()}
        paths["4:4:4"] = whole
        for name, rgb in paths.items():
            rows.setdefault(("ingestion, contre la source", name), []).append(mesures.compare(whole, rgb))
            rows.setdefault(("après SPAN ×2, contre le plan", name), []).append(
                mesures.compare(plan, net.upscale(rgb)))
    lines = ["| Mesure | Chemin | PSNR (dB) | Pire plan (dB) | SSIM | ΔE_ITP p99,9 |", "|---|---|---:|---:|---:|---:|"]
    for (stage, name), got in rows.items():
        psnr = [g["psnr"] for g in got]
        shown = "identique" if not np.isfinite(np.mean(psnr)) else f"{np.mean(psnr):.2f}"
        worst = "identique" if not np.isfinite(min(psnr)) else f"{min(psnr):.2f}"
        lines.append(f"| {stage} | {name} | {shown} | {worst} | {np.mean([g['ssim'] for g in got]):.4f} | "
                     f"{np.mean([g['itp_p999'] for g in got]):.2f} |".replace(".", ","))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modeles", required=True, help="dossier des modèles (onnx/)")
    parser.add_argument("--calibration", default=os.path.join(ROOT, "data", "calibration", "calibration.mkv"))
    parser.add_argument("--plans", type=int, default=6)
    parser.add_argument("--md", help="écrire le tableau dans ce fichier")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
