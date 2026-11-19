#!/usr/bin/env python3
"""Tuiles de SPAN sous un panoramique lent (docs/tuiles-span.md) : la caméra est-elle réelle, et décaler la
sortie d'avant d'une fraction de pixel resterait-il sous la porte des tuiles ?

  python3 -I tools/bench/qualite/tuiles_panoramique.py --video spy13.mkv --debut 1026 --images 48 \\
      [--modele ~/modeles/span.onnx] [--fournisseur cuda]

camera     sans la carte de la scène ni NVOF : la translation de chaque paire d'images voisines par corrélation
           de phase (affinée au dixième de pixel), sa médiane signée, et la cohérence de sa direction,
           |Σ d| / Σ |d| : près de 1 sous un panoramique, près de 0 si la carte voit une caméra fantôme. Puis la
           translation directe de la première image à la dernière, qui doit valoir la somme des paires.
souspixel  avec --modele (un SPAN 2× du lecteur, ou tout réseau 2× à trois plans) : la première image, une
           fenêtre au milieu, décalée de δ en Lanczos (le compositeur d'un panoramique) ; son calcul complet
           contre la sortie de l'image d'avant décalée de 2δ (bicubique, Lanczos) ou reprise telle quelle.
           Écarts comme la porte des tuiles : pire bloc 16 × 16 (écart moyen des trois plans, niveaux 8 bits),
           99e centile des blocs, écart moyen ; porte : pire bloc sous 1, écart moyen sous 0,05. Un décalage
           entier d'entrée sert de témoin : 2 pixels est exact pour SPAN (pas du dégroupage), 1 ne l'est pas.

Il faut numpy, ffmpeg, et onnxruntime pour souspixel. Mesure du 19 novembre, ArtCNN C4F32 sur trois images de
Spy×Family (4:53, 720p) : un décalage sous-pixel de la sortie s'écarte de 1,3 à 2,9 niveaux au pire bloc quel
que soit le filtre (0,1 à 0,3 d'écart moyen), et la reprise sans décalage de 1,4 à 1,6 dès δ = 0,05.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DELTAS = (0.02, 0.05, 0.1, 0.25, 0.4125, 0.5, 1.0, 2.0)
BLOCK = 16


def decode(video, start, count, size):
    """count images RGB (H × W × 3, float64 en niveaux 0 à 255) dès start secondes, à la taille donnée."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", video, "-vf",
           f"scale={w}:{h}:flags=bicubic:in_color_matrix=bt709:in_range=tv:out_range=pc", "-frames:v",
           str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float64)


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def phase_shift(a, b):
    """La translation (dx, dy) qui mène a sur b (b(x) ≈ a(x − d)) au pixel près par le pic de la corrélation de
    phase, puis affinée par Lucas-Kanade (refine), et la hauteur du pic (1 : un pur décalage)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    cross = np.conj(fa) * fb
    r = np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
    h, w = r.shape
    y, x = np.unravel_index(np.argmax(r), r.shape)

    def vertex(m, c, p):
        den = m - 2.0 * c + p
        return 0.0 if den == 0.0 else 0.5 * (m - p) / den

    fy = vertex(r[(y - 1) % h, x], r[y, x], r[(y + 1) % h, x])
    fx = vertex(r[y, (x - 1) % w], r[y, x], r[y, (x + 1) % w])
    y = y - h if y > h // 2 else y
    x = x - w if x > w // 2 else x
    dx, dy = refine(a, b, x + fx, y + fy)
    return dx, dy, float(r.max())


def refine(a, b, dx, dy, steps=4, margin=24):
    """Lucas-Kanade sur une translation : b ramenée de (dx, dy) doit retomber sur a ; le pic seul est biaisé
    d'environ un dixième de pixel (essai : 0,89 lu pour 0,8)."""
    for _ in range(steps):
        back = shift(b[:, :, None], -dx, -dy, "bicubique")[:, :, 0]
        mean = 0.5 * (a + back)
        gy, gx = np.gradient(mean)
        sl = (slice(margin, -margin), slice(margin, -margin))
        gx, gy, e = gx[sl], gy[sl], (back - a)[sl]
        m = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
        if abs(np.linalg.det(m)) < 1e-9:
            break
        ex, ey = np.linalg.solve(m, [np.sum(gx * e), np.sum(gy * e)])
        dx, dy = dx - ex, dy - ey
        if abs(ex) + abs(ey) < 1e-3:
            break
    return dx, dy


def camera(frames):
    """Les translations des paires voisines et le bilan : médiane signée, cohérence, directe contre somme."""
    lum = [luma(f) for f in frames]
    pairs = [phase_shift(lum[i], lum[i + 1]) for i in range(len(lum) - 1)]
    d = np.array([(dx, dy) for dx, dy, _ in pairs])
    norms = np.hypot(d[:, 0], d[:, 1])
    travelled = norms.sum()
    total = d.sum(axis=0)
    direct = phase_shift(lum[0], lum[-1])
    return {"paires": pairs, "mediane": tuple(np.median(d, axis=0)), "norme_mediane": float(np.median(norms)),
            "coherence": float(np.hypot(*total) / travelled) if travelled > 0 else 0.0,
            "somme": tuple(total), "directe": direct[:2]}


def kernel(name, t):
    t = np.abs(t)
    if name == "bicubique":  # Keys a = −0,5 (Catmull-Rom)
        a = -0.5
        return np.where(t <= 1, (a + 2) * t**3 - (a + 3) * t**2 + 1,
                        np.where(t < 2, a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a, 0.0))
    if name == "lanczos3":
        return np.where(t < 3, np.sinc(t) * np.sinc(t / 3), 0.0)
    raise ValueError(name)


def shift_axis(img, d, axis, name):
    """img décalée de d pixels le long de axis (sortie(x) = img(x − d)), bords répétés."""
    n = int(np.floor(d))
    f = d - n
    taps = np.arange(-3, 5)
    weights = kernel(name, taps - f) if f else (taps == 0).astype(np.float64)
    weights = weights / weights.sum()
    idx = np.arange(img.shape[axis])
    out = np.zeros_like(img)
    for t, w in zip(taps, weights):
        if w:
            out += w * np.take(img, np.clip(idx - n - t, 0, img.shape[axis] - 1), axis=axis)
    return out


def shift(img, dx, dy, name):
    """img (H × W × C) décalée de (dx, dy)."""
    return shift_axis(shift_axis(img, dx, 1, name), dy, 0, name)


def shift_x(img, d, name):
    return shift_axis(img, d, 1, name)


def gate(ref, test, margin):
    """Pire bloc, 99e centile des blocs et écart moyen, en niveaux, hors d'une marge (les bords répétés)."""
    err = np.abs(ref - test).mean(axis=2)[margin:-margin, margin:-margin]
    h, w = err.shape[0] // BLOCK * BLOCK, err.shape[1] // BLOCK * BLOCK
    blocks = err[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))
    return float(blocks.max()), float(np.percentile(blocks, 99)), float(err.mean())


def subpixel(net, frame, window, deltas):
    """Les écarts à la porte de chaque reprise, pour chaque δ (pixels d'entrée)."""
    h, w, _ = frame.shape
    top, left = (h - window) // 2 & ~1, (w - window) // 2 & ~1
    a = frame[top:top + window, left:left + window]

    def upscale(img):
        return 255.0 * net.upscale((img / 255.0).transpose(2, 0, 1).astype(np.float32)).transpose(1, 2, 0)

    out_a = upscale(a)
    margin = 64
    rows = []
    for delta in deltas:
        full = upscale(shift_x(a, delta, "lanczos3"))
        for name in ("bicubique", "lanczos3"):
            moved = shift_x(out_a, 2 * delta, name)
            rows.append((delta, f"sortie décalée de 2δ ({name})", *gate(full, moved, margin)))
        rows.append((delta, "sortie reprise sans décalage", *gate(full, out_a, margin)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--debut", type=float, default=0.0, help="secondes")
    p.add_argument("--images", type=int, default=48)
    p.add_argument("--taille", default="1920x1080", help="taille de travail du lecteur")
    p.add_argument("--modele", help="un réseau 2× (SPAN) : ajoute la mesure du décalage sous-pixel")
    p.add_argument("--fenetre", type=int, default=512, help="côté de la fenêtre de la mesure sous-pixel")
    p.add_argument("--fournisseur", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--detail", action="store_true", help="la translation de chaque paire")
    args = p.parse_args()
    size = tuple(int(v) for v in args.taille.split("x"))
    frames = decode(args.video, args.debut, args.images, size)
    if len(frames) < 2:
        raise SystemExit("moins de deux images décodées")
    cam = camera(frames)
    print(f"## Caméra par corrélation de phase ({len(frames)} images dès {args.debut} s, {size[0]} × {size[1]})")
    if args.detail:
        for i, (dx, dy, peak) in enumerate(cam["paires"]):
            print(f"- paire {i} : {dx:+.3f} {dy:+.3f} px (pic {peak:.2f})")
    print(f"- médiane signée : {cam['mediane'][0]:+.3f} {cam['mediane'][1]:+.3f} px par paire ; "
          f"norme médiane {cam['norme_mediane']:.3f} px")
    print(f"- cohérence de direction : {100 * cam['coherence']:.0f} % (près de 100 : un panoramique ; "
          "près de 0 : rien de réel)")
    print(f"- somme des paires {cam['somme'][0]:+.2f} {cam['somme'][1]:+.2f} px, "
          f"translation directe {cam['directe'][0]:+.2f} {cam['directe'][1]:+.2f} px")
    if not args.modele:
        return
    import modeles

    net = modeles.Network(args.modele, provider=args.fournisseur)
    print(f"\n## Sortie décalée contre calcul complet ({net.name}, fenêtre {args.fenetre} px)")
    print("| δ entrée (px) | reprise | pire bloc | 99e centile | écart moyen |")
    print("|---|---|---|---|---|")
    for delta, what, worst, p99, mean in subpixel(net, frames[0], args.fenetre, DELTAS):
        print(f"| {delta:g} | {what} | {worst:.2f} | {p99:.2f} | {mean:.3f} |")
    print("\nPorte des tuiles : pire bloc sous 1 niveau, écart moyen sous 0,05.")


if __name__ == "__main__":
    main()
