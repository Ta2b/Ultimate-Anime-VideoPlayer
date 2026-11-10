"""Hauteur de travail et modèle de SPAN : ce que perd une source réduite avant SPAN (audits A49-01 et A49-02).

La vraie image au-dessus de 1080p n'existe pas pour un anime 1080p. Le banc joue donc tout à l'échelle ½, comme
la mesure « vérité, agrandissement » de banc.py : un plan 1080p de calibration.mkv est la vérité, sa moitié
(moyenne 2 × 2) tient lieu de source. À cette échelle :

  source « 1080p »   la moitié du plan, 960 × 540 ;
  écran « 4K »       le plan lui-même, 1920 × 1080 ;
  écran « 1440p »    le plan réduit à 1280 × 720 ;
  écran « 1080p »    la source.

Chaque chaîne réduit la source à une hauteur de travail (Hermite, le filtre que l'Auto pose), l'agrandit ×2 par
SPAN, puis la ramène à l'écran comme libplacebo par défaut (Lanczos vers le haut, Hermite vers le bas). Le
résultat se compare à la vérité de l'écran. Les hauteurs de travail sont dites à l'échelle du lecteur : « 720p »
est ici 360 lignes.

  python3 -I tools/bench/qualite/travail.py --modeles ~/.cache/ajp-release-models/modeles --md travail.md

Limite : les traits sont deux fois plus fins que ceux que SPAN a appris, et SPAN renforce les traits au lieu de
rendre l'original (voir docs/banc-qualite.md). Les chiffres classent des chaînes entre elles, ils ne disent pas
la qualité absolue d'une chaîne.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import sources  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SPAN = {
    "b8f64": "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
    "b5f48": "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
}
# Hauteurs de travail du lecteur pour une source 1080p, et les écrans. Ici tout est à la moitié.
WORK = (1080, 720, 540)
SCREENS = (2160, 1440, 1080)


def hermite(x):
    x = np.abs(x)
    return np.where(x < 1, 2 * x**3 - 3 * x**2 + 1, 0.0)


def lanczos3(x):
    x = np.abs(x)
    return np.where(x < 3, np.sinc(x) * np.sinc(x / 3), 0.0)


def _axis(n_in, n_out, kernel, radius):
    """Poids du rééchantillonnage d'un axe : centres demi-pixel, noyau élargi quand on réduit, bords répétés."""
    scale = n_out / n_in
    stretch = max(1.0, 1.0 / scale)
    centre = (np.arange(n_out) + 0.5) / scale - 0.5
    taps = int(np.ceil(radius * stretch)) * 2 + 1
    base = np.floor(centre).astype(int) - taps // 2
    idx = base[:, None] + np.arange(taps)[None, :]
    w = kernel((idx - centre[:, None]) / stretch)
    w /= w.sum(1, keepdims=True)
    return np.clip(idx, 0, n_in - 1), w.astype(np.float32)


def resize(img, size, kernel, radius):
    """img (3 × H × W) à size (largeur, hauteur) par un filtre séparable ; inchangée si la taille est la même."""
    w_out, h_out = size
    _, h, w = img.shape
    if (w, h) == (w_out, h_out):
        return img
    iy, wy = _axis(h, h_out, kernel, radius)
    rows = np.einsum("cktw,kt->ckw", img[:, iy, :], wy)
    ix, wx = _axis(w, w_out, kernel, radius)
    return np.clip(np.einsum("chkt,kt->chk", rows[:, :, ix], wx), 0.0, 1.0).astype(np.float32)


def to_screen(img, size):
    """La mise à l'échelle finale de libplacebo par défaut : Lanczos vers le haut, Hermite vers le bas."""
    return resize(img, size, lanczos3, 3) if size[1] > img.shape[1] else resize(img, size, hermite, 1)


def width_of(height):
    return height * 16 // 9 // 2 * 2


def run(args):
    plans = sources.stills(args.calibration, args.plans)
    if not plans:
        raise SystemExit(f"{args.calibration} : aucun plan lu")
    nets = {name: modeles.Network(os.path.join(args.modeles, "onnx", file)) for name, file in SPAN.items()}
    rows = {}
    for plan in plans:
        source = sources.halve(plan)  # la source « 1080p », en 960 × 540
        truth = {2160: plan, 1440: resize(plan, (1280, 720), hermite, 1), 1080: source}
        for name, net in nets.items():
            for work in WORK:
                reduced = resize(source, (width_of(work // 2), work // 2), hermite, 1)
                up = net.upscale(reduced)
                for screen in SCREENS:
                    shown = to_screen(up, (truth[screen].shape[2], truth[screen].shape[1]))
                    rows.setdefault((screen, work, name), []).append(mesures.compare(truth[screen], shown))
        for screen in SCREENS:  # sans SPAN : la source portée à l'écran, le repère
            shown = to_screen(source, (truth[screen].shape[2], truth[screen].shape[1]))
            rows.setdefault((screen, 1080, "sans SPAN"), []).append(mesures.compare(truth[screen], shown))
    lines = ["| Écran | Travail | SPAN | PSNR (dB) | Pire plan (dB) | SSIM | ΔE_ITP p99,9 |", "|---|---|---|---:|---:|---:|---:|"]
    for (screen, work, name), got in sorted(rows.items(), key=lambda kv: (-kv[0][0], -kv[0][1], kv[0][2])):
        psnr = [g["psnr"] for g in got]
        lines.append(f"| {screen}p | {work}p | {name} | {np.mean(psnr):.2f} | {min(psnr):.2f} | "
                     f"{np.mean([g['ssim'] for g in got]):.4f} | {np.mean([g['itp_p999'] for g in got]):.2f} |"
                     .replace(".", ","))
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
