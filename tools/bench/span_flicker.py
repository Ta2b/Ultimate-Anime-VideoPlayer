#!/usr/bin/env python3
"""Scintillement de SPAN sur plans fixes (P6.21) : avant d'envisager un modèle fusionné ou spatio-temporel, mesurer
si l'agrandissement image par image fait vibrer ce qui ne bouge pas.

  span_flicker.py VIDEO MODELE.onnx [MODELE.onnx...] [--pairs 40] [--size 960x540] [--provider cuda]

Les paires retenues sont des dessins tenus (écart moyen de luminance sous 0,7 niveau 8 bits) qui ne sont pas des
doublons exacts : seul le bruit de compression les sépare. Pour chaque paire, l'écart moyen entre les deux images
agrandies par le modèle se compare à celui de l'agrandissement bicubique ×2 des mêmes images (ffmpeg n'y est pour
rien : numpy, noyau de Keys a = -0,5) : rapport 1, SPAN ne fait pas plus vibrer que le bicubique ; au-dessus, il
amplifie le bruit d'une image à l'autre. Les modèles sont ceux du lecteur : RGB 1 × 3 × H × W en entrée, ×2 en
sortie (fp16 ou fp32).

Sortie : par modèle, le rapport moyen et le 95e centile, et l'écart moyen agrandi en niveaux 8 bits.
Ce que P6.21 garde : un rapport moyen au-dessus de 1,5 sur des épisodes réels rouvre la piste d'un SPAN temporel ;
sinon elle reste écartée."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualite"))
from modeles import session  # noqa: E402
from sources import bicubic2, frames  # noqa: E402


def held_pairs(images, count, level=0.7 / 255.0):
    cand = []
    for k in range(len(images) - 1):
        d = float(np.abs(images[k].mean(0) - images[k + 1].mean(0)).mean())
        if 0.0 < d <= level:
            cand.append(k)
    step = max(1, len(cand) // count) if cand else 1
    return cand[::step][:count]


def upscale(sess, img):
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    out = sess.run(None, {inp.name: img[None].astype(dtype)})[0]
    return np.clip(out[0].astype(np.float32), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("models", nargs="+")
    p.add_argument("--pairs", type=int, default=40)
    p.add_argument("--size", default="960x540")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    images = frames(a.video, size, a.frames)
    picked = held_pairs(images, a.pairs)
    if not picked:
        raise SystemExit("aucune paire de dessins tenus (écart non nul sous 0,7 niveau)")
    print(f"{len(picked)} paires tenues sur {len(images)} images, {size[0]}x{size[1]}")
    base = [float(np.abs(bicubic2(images[k + 1]) - bicubic2(images[k])).mean()) for k in picked]
    for model in a.models:
        sess = session(model, a.provider)
        ratios, levels = [], []
        for k, b in zip(picked, base):
            d = float(np.abs(upscale(sess, images[k + 1]) - upscale(sess, images[k])).mean())
            ratios.append(d / max(b, 1e-9))
            levels.append(255.0 * d)
        print(f"{os.path.basename(model):64s} rapport moyen {np.mean(ratios):5.2f}  "
              f"95e centile {np.percentile(ratios, 95):5.2f}  écart {np.mean(levels):.3f} niveau")
    return 0


if __name__ == "__main__":
    sys.exit(main())
