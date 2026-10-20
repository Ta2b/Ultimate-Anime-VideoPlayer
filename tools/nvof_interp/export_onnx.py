#!/usr/bin/env python3
"""Exporte FlowJaNai (model.py) en ONNX FP16 pour le lecteur, sur le contrat de la variante
« mouvement » de RIFE (kernels/spec/warp.h).

Entrées du graphe :
  frames : 1 × 7 × H × W, la sortie de rife_input en 7 plans (img0 0-2, img1 3-5, t 6) ;
  nvof   : 1 × 4 × H/4 × W/4, (F01x, F01y, F10x, F10y) en pixels, la grille 4 de NVOF.
Sorties : flow (1 × 4 × H × W, f0x f0y f1x f1y) et mask (1 × 1 × H × W, poids de a).
H et W sont libres, multiples de 64. Le graphe est comparé au modèle PyTorch à deux tailles, puis
converti en FP16 avec les règles de convert_rife_fp16.py (grille et GridSample en FP32).

    python -I tools/nvof_interp/export_onnx.py runs/flowjanai/best.pt --out modeles/flowjanai.onnx
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rife"))
import convert_rife_fp16  # noqa: E402
from model import FlowJaNai  # noqa: E402


class Graph(nn.Module):
    """Le réseau sur les tenseurs du lecteur."""

    def __init__(self, net: FlowJaNai) -> None:
        super().__init__()
        self.net = net

    def forward(self, frames: torch.Tensor, nvof: torch.Tensor):
        return self.net(frames[:, 0:3], frames[:, 3:6], frames[:, 6:7], nvof)


def sample(h: int, w: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(1, 7, h, w, generator=g)
    frames[:, 6:7] = 0.5
    nvof = (torch.rand(1, 4, h // 4, w // 4, generator=g) - 0.5) * 40.0
    return frames, nvof


def check(graph: Graph, path: Path, h: int, w: int) -> None:
    import onnxruntime as ort

    frames, nvof = sample(h, w, h * w)
    with torch.no_grad():
        flow, mask = graph(frames, nvof)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got_flow, got_mask = session.run(None, {"frames": frames.numpy(), "nvof": nvof.numpy()})
    for name, want, got in (("flow", flow.numpy(), got_flow), ("mask", mask.numpy(), got_mask)):
        err = float(np.abs(want - got).max())
        print(f"{h}x{w} {name} : écart max {err:.2e}")
        if err > 1e-3 * max(1.0, float(np.abs(want).max())):
            raise SystemExit(f"{name} : l'ONNX s'écarte du modèle PyTorch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="best.pt ou last.pt de train.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    net = FlowJaNai()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["net"])
    graph = Graph(net.eval())
    axes = {2: "h", 3: "w"}
    flow_axes = {2: "h4", 3: "w4"}
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "flowjanai_fp32.onnx"
        torch.onnx.export(graph, sample(256, 384, 0), str(fp32), input_names=["frames", "nvof"],
                          output_names=["flow", "mask"],
                          dynamic_axes={"frames": axes, "nvof": flow_axes, "flow": axes, "mask": axes},
                          opset_version=17, do_constant_folding=True, dynamo=False)
        check(graph, fp32, 256, 384)
        check(graph, fp32, 320, 576)
        half = convert_rife_fp16.convert_model(onnx.load(str(fp32)),
                                               {"frames": (1, 7, 768, 1280), "nvof": (1, 4, 192, 320)})
    onnx.checker.check_model(half)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(half, str(args.out))
    print(f"{args.out} : {args.out.stat().st_size / 1e6:.1f} Mo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
