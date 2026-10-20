#!/usr/bin/env python3
"""Exporte DistilDRBA (routineLife1/DistilDRBA, MIT) en un graphe ONNX FP16 pour le lecteur
(kernels/spec/distil_drba.h).

DistilDRBA est un RIFE à trois images entraîné à imiter DRBA : l'image du milieu img1 est le
centre, et l'instant tau de [0,5 ; 1,5] (1 : img1 elle-même) se lit entre le milieu de la paire
(img0, img1) et celui de la paire (img1, img2). Le réseau règle lui-même l'instant de chaque pixel
d'après le mouvement des deux côtés : un personnage animé en deux tient sa pose.

Entrées du graphe, chacune la sortie de rife_input du lecteur en 7 plans (RIFE v2) :
  pair01 : img0 (0-2), img1 (3-5), u (6)
  pair12 : img1 (0-2), img2 (3-5), u (6)
avec u = tau / 2, dans ]0, 1[ comme l'instant que rife_input accepte. Sortie : out, 1 × 3 × H × W.
H et W sont libres, multiples de 64 (128 avec --scale 0.5).

forward de DistilDRBA choisit le côté (img0 ou img2) par un if Python sur tau : il est réécrit
ici en sélection sur tenseurs, les blocs et l'encodeur du dépôt restant tels quels. Le graphe est
comparé au forward d'origine pour tau des deux côtés, puis converti en FP16 avec les règles de
convert_rife_fp16.py (grille, GridSample et Resize en FP32).

    git clone https://github.com/routineLife1/DistilDRBA
    python -I tools/drba/export_distil_drba.py DistilDRBA --weights DistilDRBA/weights/v2_lite.pkl \\
        --out modeles/distil_drba_v2_lite.onnx
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rife"))
import convert_rife_fp16  # noqa: E402

SCALES = {"v1": (16, 8, 4, 2, 1), "v2_lite": (16, 8, 4)}


def load_ifnet(repo: Path, weights: Path):
    """L'IFNet du dépôt dont les poids correspondent (v1 ou v2 lite), en eval, et son nom."""
    sys.path.insert(0, str(repo))
    if importlib.util.find_spec("tqdm") is None:  # importé par les modèles pour leur banc d'essai
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=None)
    for arch, module in (("v1", "models.drba.distilDRBA"), ("v2_lite", "models.drba.distilDRBA_v2_lite")):
        net = importlib.import_module(module).IFNet()
        try:
            net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        except RuntimeError:
            continue
        return net.eval(), arch
    raise SystemExit(f"{weights} : ni v1 ni v2 lite")


def warp(image, flow):
    """warplayer.warp du dépôt, la grille calculée depuis la forme (dimensions libres)."""
    _, _, h, w = flow.shape
    gx = torch.arange(w, dtype=flow.dtype).view(1, 1, 1, w) * (2.0 / (w - 1)) - 1.0
    gy = torch.arange(h, dtype=flow.dtype).view(1, 1, h, 1) * (2.0 / (h - 1)) - 1.0
    x = gx + flow[:, 0:1] * (2.0 / (image.shape[3] - 1))
    y = gy + flow[:, 1:2] * (2.0 / (image.shape[2] - 1))
    grid = torch.cat([x, y], 1).permute(0, 2, 3, 1)
    return F.grid_sample(image, grid.to(image.dtype), mode="bilinear", padding_mode="border", align_corners=True)


def resize(x, factor):
    return x if factor == 1 else F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


class Exported(nn.Module):
    """forward de DistilDRBA (v1 ou v2 lite), tau lu dans le plan 6 et le côté choisi sur tenseur."""

    def __init__(self, net, arch: str, scale: float):
        super().__init__()
        self.net = net
        self.arch = arch
        self.scales = [s / scale for s in SCALES[arch]]

    def forward(self, pair01, pair12):
        img0, img1, img2 = pair01[:, 0:3], pair01[:, 3:6], pair12[:, 3:6]
        tau = pair01[:, 6:7, 0:1, 0:1] * 2.0
        scales = self.scales
        encode_scale = min(1 / scales[-1], 1)
        f0, f1, f2 = (self.net.encode(resize(x, encode_scale)) for x in (img0, img1, img2))
        # 0,5 ≤ tau < 1 : vers img0 ; 1 < tau ≤ 1,5 : vers img2 (le if de forward).
        before = (tau < 1.0).to(img0.dtype)
        inp0, inp1 = img1, before * img0 + (1 - before) * img2
        h0, h1 = f1, before * f0 + (1 - before) * f2
        blocks = [getattr(self.net, f"block{k}") for k in range(len(scales))]
        flow = mask = feat = tmap = None
        for k, s in enumerate(scales):
            current = 1 / s
            if flow is None:
                f0s, f1s, f2s = (resize(x, current / encode_scale) for x in (f0, f1, f2))
                i0s, i1s, i2s = (resize(x, current) for x in (img0, img1, img2))
                plane = torch.zeros_like(i0s[:, :1]) + tau
                out = blocks[k](torch.cat((i0s, i1s, i2s, f0s, f1s, f2s, plane), 1))
            else:
                h0s, h1s = (resize(x, current / encode_scale) for x in (h0, h1))
                p0s, p1s = (resize(x, current) for x in (inp0, inp1))
                wf0, wi0 = warp(h0s, flow[:, :2]), warp(p0s, flow[:, :2])
                wf1, wi1 = warp(h1s, flow[:, 2:4]), warp(p1s, flow[:, 2:4])
                if self.arch == "v1":
                    out = blocks[k](torch.cat((wi0, wi1, wf0, wf1, tmap, mask, feat), 1), flow)
                else:
                    i0s, i1s, i2s = (resize(x, current) for x in (img0, img1, img2))
                    f0s, f1s, f2s = (resize(x, current / encode_scale) for x in (f0, f1, f2))
                    plane = torch.zeros_like(i0s[:, :1]) + tau
                    out = blocks[k](torch.cat((i0s, i1s, i2s, f0s, f1s, f2s, wi0, wi1, wf0, wf1, plane, mask, feat), 1),
                                    flow)
            if self.arch == "v1":
                fd, mask, feat, tmap = out
                flow = fd if flow is None else flow + fd
            else:
                flow, mask, feat = out
            if k == len(scales) - 1:
                step = scales[k]
            elif scales[k + 1] >= 1:
                step = scales[k] / scales[k + 1]
            else:
                step = 1
            if step != 1:
                flow, mask, feat = resize(flow, step) * step, resize(mask, step), resize(feat, step)
                if tmap is not None:
                    tmap = resize(tmap, step)
        mask = torch.sigmoid(mask)
        return warp(inp0, flow[:, :2]) * mask + warp(inp1, flow[:, 2:4]) * (1 - mask)


def frames(h: int, w: int, seed: int):
    """Trois images lisses qui glissent, dans [0, 1]."""
    rng = np.random.default_rng(seed)
    base = torch.from_numpy(rng.random((1, 3, h // 8, w // 8), dtype=np.float32))
    base = F.interpolate(base, size=(h + 16, w + 16), mode="bicubic", align_corners=False).clamp(0, 1)
    return [base[:, :, 4 * k : 4 * k + h, 3 * k : 3 * k + w].contiguous() for k in range(3)]


def packed(images, tau: float):
    u = torch.full_like(images[0][:, :1], tau / 2)
    return torch.cat([images[0], images[1], u], 1), torch.cat([images[1], images[2], u], 1)


def check(net, arch: str, scale: float, onnx_path: Path, h: int, w: int) -> None:
    """Le graphe contre forward du dépôt, en fp32, des deux côtés de img1."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    images = frames(h, w, 7)
    for tau in (0.625, 0.875, 1.25, 1.5):
        a, b = packed(images, tau)
        with torch.no_grad():
            want = net(*images, timestep=tau, scale_list=[s / scale for s in SCALES[arch]])[0].numpy()
        got = session.run(None, {"pair01": a.numpy(), "pair12": b.numpy()})[0]
        error = float(np.abs(got - want).max())
        print(f"  tau {tau} : écart max {error:.2e}")
        if error > 2e-3:
            raise SystemExit(f"tau {tau} : le graphe s'écarte de forward ({error})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="clone de routineLife1/DistilDRBA (models/drba)")
    parser.add_argument("--weights", type=Path, required=True, help="v1.pkl ou v2_lite.pkl")
    parser.add_argument("--out", type=Path, required=True, help="graphe ONNX FP16 écrit")
    parser.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0,
                        help="échelle du flot : 0,5 pour une entrée 4K (multiples de 128)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    args = parser.parse_args()
    align = int(64 / args.scale)
    args.width, args.height = (-(-n // align) * align for n in (args.width, args.height))
    net, arch = load_ifnet(args.repo, args.weights)
    print(f"DistilDRBA {arch}, flot à l'échelle {args.scale}")
    model = Exported(net, arch, args.scale).eval()
    a, b = packed(frames(args.height, args.width, 3), 0.75)
    with tempfile.TemporaryDirectory() as scratch:
        fp32 = Path(scratch) / "fp32.onnx"
        axes = {2: "H", 3: "W"}
        torch.onnx.export(model, (a, b), str(fp32), input_names=["pair01", "pair12"], output_names=["out"],
                          dynamic_axes={"pair01": axes, "pair12": axes, "out": axes}, opset_version=17,
                          do_constant_folding=True, dynamo=False)
        check(net, arch, args.scale, fp32, args.height, args.width)
        check(net, arch, args.scale, fp32, args.height + align, args.width + 2 * align)
        shape = (1, 7, 768, 1280)
        half = convert_rife_fp16.convert_model(onnx.load(str(fp32)), {"pair01": shape, "pair12": shape})
    onnx.checker.check_model(half)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(half, str(args.out))
    print(f"{args.out} : {args.out.stat().st_size / 1e6:.1f} Mo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
