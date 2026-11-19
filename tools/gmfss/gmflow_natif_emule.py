#!/usr/bin/env python3
"""Émulation du GMFlow natif (docs/gmflow-natif.md) contre ONNX Runtime sur le graphe « paire » de GMFSS union.

Le graphe paire est exporté (export_gmfss.py) deux fois à la taille du test : en FP32 pur (la vérité) et en FP16
tel que le lecteur le donne à TensorRT (l'actuel). Les paires de test sont de vraies images d'anime
(data/calibration/calibration.mkv) déplacées par un mouvement connu (translation, zoom, rotation, objet qui
traverse le décor). Pour chaque recette de gmflow_natif_reseau.py, le flot de GMFlow (motion{k}_1, les deux sens)
est comparé à celui du graphe FP32 : EPE moyen, 99,9e centile et pire pixel, et PSNR du flot (crête = plus grand
|flot| de la vérité). Puis l'image du milieu (t = 0,5, toute la suite de GMFSS en FP32 PyTorch, seul le flot
change) contre celle du flot FP32 : PSNR de l'image, l'unité des gardes du projet (docs/fp8-interpolation.md).

    python -I tools/gmfss/gmflow_natif_emule.py ROUE/vsgmfss_fortuna --work DOSSIER [--width 640 --height 384]

Les échelles FP8 (recettes fp8-*) se calibrent sur trois paires qui ne sont pas des paires de test.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
import gmflow_natif_reseau as R  # noqa: E402
from export_gmfss import export, prepare, run_onnx  # noqa: E402
from gmfss_graph import Fusion, Pair, load_gmfss, pair_axes, pair_names, splat_all  # noqa: E402

DEPOT = Path(__file__).resolve().parents[2]
VIDEO = DEPOT / "data/calibration/calibration.mkv"

# (instant dans calibration.mkv, mouvement) ; un mouvement : (dx, dy, zoom, degrés, objet) en pixels de la paire.
TESTS = [
    (69.0, "translation", (6, -3, 1.0, 0.0, False)),
    (69.0, "panoramique", (60, 0, 1.0, 0.0, False)),
    (39.0, "zoom", (24, 10, 1.03, 0.0, False)),
    (39.0, "objet", (10, 5, 1.0, 0.0, True)),
    (99.0, "rotation", (0, 0, 0.97, 4.0, False)),
    (99.0, "fort", (40, -25, 1.0, 2.0, True)),
]
CALIBRATION = [
    (15.0, "c-zoom", (24, 10, 1.03, 0.0, False)),
    (51.0, "c-objet", (10, 5, 1.0, 0.0, True)),
    (81.0, "c-fort", (40, -25, 1.0, 2.0, True)),
]
ORDRE = ["fp32", "fp16", "fp16-acc16", "fp16-pv16", "fp8-gemm", "fp8-gemm-acc16", "fp8-tout"]


def image(instant: float, w: int, h: int) -> torch.Tensor:
    """Une image de la vidéo à w × (h utile), RGB [0, 1], [1, 3, h, w]."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(instant), "-i", str(VIDEO), "-frames:v", "1", "-vf",
                          f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    a = np.frombuffer(out, np.uint8).reshape(h, w, 3).astype(np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def deplace(img: torch.Tensor, dx: float, dy: float, zoom: float, deg: float) -> torch.Tensor:
    """img vue après un mouvement de caméra (affine autour du centre), bords recopiés."""
    _, _, h, w = img.shape
    a = math.radians(deg)
    m = torch.tensor([[math.cos(a) / zoom, -math.sin(a) / zoom, -2 * dx / w],
                      [math.sin(a) / zoom, math.cos(a) / zoom, -2 * dy / h]], dtype=torch.float32)
    g = F.affine_grid(m.unsqueeze(0), list(img.shape), align_corners=False)
    return F.grid_sample(img, g, mode="bicubic", padding_mode="border", align_corners=False).clamp(0, 1)


def paire(instant: float, mvt, w: int, h: int) -> torch.Tensor:
    """frames [1, 6, h, w] : l'image et sa voisine, la hauteur utile (16:9) complétée en recopiant le bord comme
    gmfss_frames ; un objet (une découpe d'une autre image) traverse le décor si demandé."""
    hu = w * 9 // 16
    img0 = image(instant, w, hu)
    dx, dy, zoom, deg, objet = mvt
    img1 = deplace(img0, dx, dy, zoom, deg)
    if objet:
        autre = image(instant + 3.0, w, hu)
        oh, ow = hu // 3, w // 4
        y0, x0 = hu // 3, w // 3
        img0[:, :, y0 : y0 + oh, x0 : x0 + ow] = autre[:, :, :oh, :ow]
        img1[:, :, y0 + 18 : y0 + 18 + oh, x0 - 30 : x0 - 30 + ow] = autre[:, :, :oh, :ow]
    frames = torch.cat([img0, img1], 1)
    return F.pad(frames, (0, 0, 0, h - hu), mode="replicate")


def graphes(paquet: Path, travail: Path, w: int, h: int, scale: float) -> tuple[Path, Path]:
    """Le graphe paire FP32 pur et le FP16 du lecteur, exportés une fois dans travail."""
    GMFSS = load_gmfss(paquet)
    modeles = str(paquet / "models")
    suffixe = "" if scale == 1.0 else "_s05"
    p32, p16 = travail / f"paire{suffixe}_fp32.onnx", travail / f"paire{suffixe}_fp16.onnx"
    frames = paire(*TESTS[0][::2], w, h)
    if not p32.exists():
        g = GMFSS(modeles, "union", scale, False).eval()
        export(Pair(g), (frames,), p32, ["frames"], pair_names(), pair_axes())
    if not p16.exists():
        g = prepare(GMFSS(modeles, "union", scale, False).eval(), "fp16")
        export(Pair(g), (frames.half(),), p16, ["frames"], pair_names(), pair_axes())
    return p32, p16


def flots_onnx(chemin: Path, frames: torch.Tensor, dtype) -> torch.Tensor:
    """Les deux flots de la paire (motion0_1, motion1_1 : 0 -> 1 et 1 -> 0 à la demi-taille), en f32."""
    sorties = dict(zip(pair_names(), run_onnx(chemin, {"frames": frames.to(dtype)})))
    return torch.cat([torch.from_numpy(sorties[f"motion{k}_1"][:, :2]).float() for k in (0, 1)], 0)


def flots_emules(frames: torch.Tensor, sd: dict, r: R.Recette, scale: float) -> torch.Tensor:
    """Le flot de la paire par la recette : GMFlow sur les demi-images (et leur moitié au demi-flot), ramené à la
    demi-taille comme Pair (bilinéaire, / scale), arrondi en FP16 comme la sortie du graphe."""
    demi = F.interpolate(frames, scale_factor=0.5, mode="bilinear")
    if scale != 1.0:
        demi = F.interpolate(demi, scale_factor=scale, mode="bilinear")
    demi = R.stocke(demi, r)
    flot = R.gmflow(demi[:, :3], demi[:, 3:], sd, r)
    if scale != 1.0:
        flot = F.interpolate(flot, scale_factor=1 / scale, mode="bilinear") / scale
    return R.stocke(flot, r)


def ecarts(flot: torch.Tensor, verite: torch.Tensor) -> dict:
    epe = (flot - verite).norm(dim=1).flatten()
    crete = max(float(verite.norm(dim=1).max()), 1.0)
    mse = float(((flot - verite) ** 2).mean())
    psnr = 10 * math.log10(crete**2 / mse) if mse > 0 else float("inf")
    return {"epe": float(epe.mean()), "p999": float(torch.quantile(epe, 0.999)), "pire": float(epe.max()),
            "psnr": psnr}


class FlotDonne(torch.nn.Module):
    """Remplace GMFlow dans GMFSS : rend le flot donné (deux sens en lot, à la taille de GMFlow)."""

    def __init__(self):
        super().__init__()
        self.flot = None

    def forward(self, *args, **kwargs):
        return self.flot


@torch.inference_mode()
def image_milieu(gmfss, frames: torch.Tensor, flot_gmflow: torch.Tensor | None) -> torch.Tensor:
    """L'image t = 0,5 de GMFSS en FP32 PyTorch (splats de gmfss_graph), avec ce flot à la place de GMFlow."""
    original = gmfss.flownet
    if flot_gmflow is not None:
        gmfss.flownet = FlotDonne()
        gmfss.flownet.flot = flot_gmflow
    try:
        sorties = Pair(gmfss)(frames)
    finally:
        gmfss.flownet = original
    half0, half1, warped = splat_all(sorties, 0.5)
    return Fusion(gmfss)(half0, half1, torch.tensor([[[[0.5]]]]), *warped)


def psnr_image(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1 / mse) if mse > 0 else float("inf")


def flot_gmflow(frames: torch.Tensor, sd: dict, r: R.Recette, scale: float) -> torch.Tensor:
    """Le flot à la taille de GMFlow (ce que FlotDonne doit rendre), en f32."""
    demi = F.interpolate(frames, scale_factor=0.5, mode="bilinear")
    if scale != 1.0:
        demi = F.interpolate(demi, scale_factor=scale, mode="bilinear")
    demi = R.stocke(demi, r)
    return R.stocke(R.gmflow(demi[:, :3], demi[:, 3:], sd, r), r)


def calibre(sd: dict, w: int, h: int, scale: float) -> dict:
    r = R.recette("fp16")
    r.releve = {}
    for instant, _, mvt in CALIBRATION:
        flots_emules(paire(instant, mvt, w, h), sd, r, scale)
    return r.releve


def ligne(nom: str, e: dict, img: float | None, extra: str = "") -> str:
    img_txt = f"{img:7.2f}" if img is not None else "      -"
    return (f"| {nom:<16} | {e['epe']:.4f} | {e['p999']:.3f} | {e['pire']:.3f} | {e['psnr']:6.2f} | {img_txt} |"
            f"{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="dossier vsgmfss_fortuna extrait de la roue")
    parser.add_argument("--work", type=Path, required=True, help="dossier des graphes exportés")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0)
    parser.add_argument("--recettes", default=",".join(ORDRE))
    parser.add_argument("--images", action="store_true", help="aussi l'image du milieu (plus long)")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    paquet = args.package.resolve()
    sd = torch.load(paquet / "models/flownet.pkl", map_location="cpu")
    p32, p16 = graphes(paquet, args.work, args.width, args.height, args.scale)
    echelles = calibre(sd, args.width, args.height, args.scale)
    recettes = args.recettes.split(",")
    GMFSS = load_gmfss(paquet)
    gmfss = GMFSS(str(paquet / "models"), "union", args.scale, False).eval() if args.images else None
    cumul: dict[str, list] = {}
    print(f"GMFlow, paire {args.width}x{args.height}, échelle {args.scale} ; contre ONNX Runtime FP32")
    for instant, nom, mvt in TESTS:
        frames = paire(instant, mvt, args.width, args.height)
        verite = flots_onnx(p32, frames, torch.float32)
        actuel = flots_onnx(p16, frames, torch.float16)
        ref_img = image_milieu(gmfss, frames, None) if gmfss is not None else None
        amp = float(verite.norm(dim=1).max())
        print(f"\n{nom} (t = {instant} s, |flot| max {amp:.1f} px à la demi-taille)")
        print("| recette          | EPE    | p99,9 | pire  | PSNR flot | PSNR image |")
        lignes = [("onnx-fp16", actuel, None)]
        for nom_r in recettes:
            r = R.recette(nom_r, echelles)
            t0 = time.time()
            g = flot_gmflow(frames, sd, r, args.scale)
            flot = F.interpolate(g, scale_factor=1 / args.scale, mode="bilinear") / args.scale if args.scale != 1.0 else g
            lignes.append((nom_r, R.stocke(flot, r), g, r, time.time() - t0))
        for item in lignes:
            nom_r, flot = item[0], item[1]
            e = ecarts(flot, verite)
            img = None
            if gmfss is not None and nom_r != "onnx-fp16":
                img = psnr_image(image_milieu(gmfss, frames, item[2]), ref_img)
            elif gmfss is not None and args.scale == 1.0:
                img = psnr_image(image_milieu(gmfss, frames, flot), ref_img)
            extra = f" débordements f16 {item[3].debordements}" if len(item) > 3 and item[3].debordements else ""
            print(ligne(nom_r, e, img, extra), flush=True)
            cumul.setdefault(nom_r, []).append((e, img))
    print("\nSynthèse (moyenne des EPE, pire des p99,9 et des PSNR) :")
    print("| recette          | EPE moyen | p99,9 pire | PSNR flot moyen / pire | PSNR image moyen / pire |")
    for nom_r, rows in cumul.items():
        epe = sum(e["epe"] for e, _ in rows) / len(rows)
        p999 = max(e["p999"] for e, _ in rows)
        ps = [e["psnr"] for e, _ in rows]
        im = [i for _, i in rows if i is not None]
        im_txt = f"{sum(im) / len(im):6.2f} / {min(im):6.2f}" if im else "      -"
        print(f"| {nom_r:<16} | {epe:.4f} | {p999:.3f} | {sum(ps) / len(ps):6.2f} / {min(ps):6.2f} | {im_txt} |")


if __name__ == "__main__":
    main()
