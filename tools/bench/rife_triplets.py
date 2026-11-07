#!/usr/bin/env python3
"""Banc de qualité des RIFE (P6.19) : chaque modèle interpole le milieu de triplets tirés d'une vidéo, comparé à la
vraie image du milieu.

  rife_triplets.py VIDEO MODELE.onnx [MODELE.onnx...] [--triplets 40] [--size 640x360] [--provider cuda]

Les triplets (k, k+1, k+2) sont pris à intervalles réguliers parmi ceux où les trois images diffèrent (l'anime se
joue « sur deux » : un dessin tenu au milieu ne teste rien). Chaque modèle reçoit k et k+2 à t = 0,5 (entrée v1 à
11 plans de vs-mlrt, remplie au multiple de 128 par répétition du bord) ; son image se compare à k+1 par PSNR et
SSIM (luminance, fenêtres 8 × 8). Le temps par paire est celui d'ONNX Runtime, sur le fournisseur choisi : il
classe les modèles entre eux, il ne vaut pas le temps du lecteur (TensorRT, Vulkan).

Sortie : une ligne par modèle (PSNR moyen, pire, SSIM moyen, ms par paire) ; --csv écrit le détail par triplet.
Ce que P6.19 garde : 4.26 seulement s'il bat 4.25 en PSNR moyen à coût égal, ou 4.22 lite à qualité égale."""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np


def frames(video, size, limit):
    """Les images de la vidéo à la taille donnée, en RGB float32 3 × H × W, au plus limit."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h}:flags=bicubic", "-frames:v", str(limit),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return [f.transpose(2, 0, 1).astype(np.float32) / 255.0 for f in arr]


def differs(a, b, level=0.7 / 255.0):
    """Deux images différentes : écart moyen de luminance au-dessus de 0,7 niveau 8 bits (le seuil des tenus)."""
    return float(np.abs(a.mean(0) - b.mean(0)).mean()) > level


def triplets(images, count):
    cand = [k for k in range(len(images) - 2)
            if differs(images[k], images[k + 1]) and differs(images[k + 1], images[k + 2])]
    if not cand:
        return []
    step = max(1, len(cand) // count)
    return cand[::step][:count]


def pad(img, align=128):
    _, h, w = img.shape
    ph, pw = (h + align - 1) // align * align, (w + align - 1) // align * align
    return np.pad(img, ((0, 0), (0, ph - h), (0, pw - w)), mode="edge")


def rife_input(a, b, t):
    _, h, w = a.shape
    x = np.empty((1, 11, h, w), np.float32)
    x[0, 0:3], x[0, 3:6], x[0, 6] = a, b, t
    xs = 2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0
    ys = 2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0
    x[0, 7], x[0, 8] = xs[None, :], ys[:, None]
    x[0, 9], x[0, 10] = 2.0 / (w - 1), 2.0 / (h - 1)
    return x


def psnr(ref, test):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 200.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b):
    def lum(x):
        x = x.astype(np.float64)
        return x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722

    a, b = lum(a), lum(b)
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    A = a[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    B = b[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    ma, mb, va, vb = A.mean(1), B.mean(1), A.var(1), B.var(1)
    cov = ((A - ma[:, None]) * (B - mb[:, None])).mean(1)
    c1, c2 = 0.01**2, 0.03**2
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))))


def session(path, provider):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=providers)


def run_model(path, images, picked, provider):
    sess = session(path, provider)
    inp = sess.get_inputs()[0]
    if inp.shape[1] != 11:
        raise SystemExit(f"{path} : entrée à {inp.shape[1]} plans, ce banc prend l'entrée v1 à 11 plans")
    dtype = np.float16 if "float16" in inp.type else np.float32
    _, h, w = images[0].shape
    rows, spent = [], 0.0
    for k in picked:
        x = rife_input(pad(images[k]), pad(images[k + 2]), 0.5).astype(dtype)
        start = time.perf_counter()
        out = sess.run(None, {inp.name: x})[0]
        spent += time.perf_counter() - start
        mid = np.clip(out[0, :3, :h, :w].astype(np.float32), 0.0, 1.0)
        rows.append((k, psnr(images[k + 1], mid), ssim(images[k + 1], mid)))
    return rows, 1000.0 * spent / max(1, len(picked))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("models", nargs="+")
    p.add_argument("--triplets", type=int, default=40)
    p.add_argument("--size", default="640x360")
    p.add_argument("--frames", type=int, default=2000, help="images lues au plus depuis le début")
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--csv")
    a = p.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    images = frames(a.video, size, a.frames)
    picked = triplets(images, a.triplets)
    if not picked:
        raise SystemExit("aucun triplet de trois images différentes")
    print(f"{len(picked)} triplets sur {len(images)} images, {size[0]}x{size[1]}")
    detail = []
    for model in a.models:
        rows, ms = run_model(model, images, picked, a.provider)
        ps = [r[1] for r in rows]
        print(f"{os.path.basename(model):32s} PSNR moyen {np.mean(ps):6.2f} dB  pire {min(ps):6.2f}  "
              f"SSIM {np.mean([r[2] for r in rows]):.4f}  {ms:8.1f} ms/paire")
        detail += [(os.path.basename(model), *r) for r in rows]
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            csv.writer(f).writerows([("modele", "image", "psnr", "ssim"), *detail])
    return 0


if __name__ == "__main__":
    sys.exit(main())
