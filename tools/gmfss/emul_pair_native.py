#!/usr/bin/env python3
"""Émulation du FeatureNet et du MetricNet natifs de GMFSS (étape 4 de docs/gmfss-natif.md) contre ONNX Runtime.

Refait en PyTorch ce que font prepare/gmfss_pair_native.cpp et gpu/cuda/backend/gmfss_pair_native_net.cu, avec
les arrondis FP16 aux mêmes endroits que les noyaux de rife_native_kernels.cuh : PReLU de l'entrée arrondi au
chargement de la tuile, somme et biais arrondis (stage), résiduel puis PReLU de sortie arrondis (finish8), et
pour la queue du MetricNet tanh arrondi puis × 10 arrondi, la moitié (moyenne 2 × 2) et le quart (moyenne des
quatre pixels du centre de chaque 4 × 4) du Resize linéaire « half_pixel ». Les sommes sont en f32 (acc32).

Vérifie aussi la coupe de la paire : le morceau que garde TensorRT (frames → entrée du MetricNet, flots des trois
niveaux, half0, half1), extrait par nom comme le fait prepare/gmfss_pair_split.cpp, rend ce que rend la paire.
Enfin la paire native passe par les splats et la fusion (ONNX) : PSNR de l'image finale contre la paire entière.

    python -I tools/gmfss/emul_pair_native.py modeles/gmfss/gmfss_fortuna_union_pair.onnx \
        modeles/gmfss/gmfss_fortuna_union_fusion.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
from gmfss_graph import fusion_names, splat_all  # noqa: E402


def h16(t):
    return t.half().float()


def prelu(x, a):
    return torch.where(x < 0, x * a, x)


class Weights:
    """Poids et pentes lus par leurs noms, comme gmfss_pair_native.cpp."""

    def __init__(self, model):
        g = model.graph
        self.w = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in g.initializer}
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.w[n.output[0]] = numpy_helper.to_array(a.t).astype(np.float32)
        self.nodes = {n.name: n for n in g.node}

    def conv(self, name):
        return torch.from_numpy(self.w[f"gmfss.{name}.weight"]), torch.from_numpy(self.w[f"gmfss.{name}.bias"])

    def slope(self, node):
        for i in self.nodes[node].input:
            v = self.w.get(i)
            if v is not None and v.size == 1:
                return float(v.reshape(-1)[0])
        raise KeyError(node)


def conv_step(x, wb, stride=1, in_slope=None, out_slope=None, skip=None):
    """Une convolution des noyaux : PReLU à la lecture, somme f32 + biais arrondie, résiduel et PReLU arrondis."""
    if in_slope is not None:
        x = h16(prelu(x, in_slope))
    y = h16(F.conv2d(x, wb[0], wb[1], stride=stride, padding=1))
    if skip is None and out_slope is None:
        return y
    if skip is not None:
        y = y + skip
    if out_slope is not None:
        y = prelu(y, out_slope)
    return h16(y)


def features(W, image):
    out = []
    x = image
    for b in (1, 2, 3):
        s0 = W.slope(f"/feat_ext/block{b}/block{b}.0/Mul")
        s2 = W.slope(f"/feat_ext/block{b}/block{b}.2/Mul")
        t = conv_step(x, W.conv(f"feat_ext.block{b}.1"), stride=2, in_slope=s0, out_slope=s2)
        x = conv_step(t, W.conv(f"feat_ext.block{b}.3"))
        out.append(x)
    return out


def metric(W, inp):
    m = conv_step(inp, W.conv("metricnet.metric_in"))
    for i in (1, 2, 3):
        s = W.slope(f"/metricnet/metric_net{i}/metric_net{i}.0/Mul")
        m = conv_step(m, W.conv(f"metricnet.metric_net{i}.1"), in_slope=s, skip=m)
    s = W.slope("/metricnet/metric_out/metric_out.0/Mul")
    out = conv_step(m, W.conv("metricnet.metric_out.1"), in_slope=s)
    met = h16(h16(torch.tanh(out)) * 10)
    half = h16(F.avg_pool2d(met, 2))
    quarter = h16(F.avg_pool2d(met[:, :, 1:, 1:], 2, stride=4))
    return met, half, quarter


def psnr(got, ref, peak):
    mse = float(((got - ref) ** 2).mean())
    return 10 * np.log10(peak * peak / mse) if mse > 0 else float("inf")


def tensor_names(model):
    """Les tenseurs de la coupe, relevés comme gmfss_pair_split.cpp : l'entrée de la convolution metric_in et
    les flots, premières entrées des Concat qui écrivent motion{k}_{l}."""
    metric_in = next(n for n in model.graph.node if n.op_type == "Conv" and "gmfss.metricnet.metric_in.weight" in n.input)
    flows = {}
    for n in model.graph.node:
        if n.op_type == "Concat" and n.output[0].startswith("motion"):
            flows[n.output[0]] = n.input[0]
    return metric_in.input[0], [flows[f"motion{k}_{lv}"] for k in (0, 1) for lv in (1, 2, 3)]


def frames_for(width, height, seed, low, high):
    g = torch.Generator().manual_seed(seed)
    base = F.interpolate(torch.rand(1, 3, height // 16 + 2, width // 16 + 2, generator=g), (height + 16, width + 16),
                         mode="bicubic", align_corners=False)
    base = low + (high - low) * base.clamp(0, 1)
    img0 = base[:, :, 8 : 8 + height, 8 : 8 + width]
    img1 = base[:, :, 5 : 5 + height, 11 : 11 + width]
    return torch.cat([img0, img1], 1).half()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair", type=Path)
    parser.add_argument("fusion", type=Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    args = parser.parse_args()

    model = onnx.load(str(args.pair))
    W = Weights(model)
    metric_input, flows = tensor_names(model)
    extended = onnx.load(str(args.pair))
    extended.graph.output.append(onnx.helper.make_tensor_value_info(metric_input, onnx.TensorProto.FLOAT16, None))
    session = ort.InferenceSession(extended.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]

    # Le morceau de TensorRT : extrait par noms, il doit rendre les mêmes tenseurs que la paire.
    cut = Path(args.pair).with_name("emul_pair_flow.onnx")
    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(args.pair)))
    piece = onnx.utils.Extractor(inferred).extract_model(["frames"], [metric_input, *flows, "half0", "half1"])
    onnx.save(piece, str(cut))
    flow_session = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])
    cut_nodes = {n.name for n in onnx.load(str(cut)).graph.node}
    leftovers = [n for n in cut_nodes if n.startswith("/feat_ext") or "/metricnet/metric_" in n]
    print(f"morceau TensorRT : {len(cut_nodes)} nœuds, FeatureNet ou convolutions du MetricNet restés : {len(leftovers)}")

    fusion = ort.InferenceSession(str(args.fusion), providers=["CPUExecutionProvider"])
    worst = {}
    for low, high, seed in ((0.0, 1.0, 1), (-0.06, 1.06, 2)):
        frames = frames_for(args.width, args.height, seed, low, high)
        ref = dict(zip(names, session.run(None, {"frames": frames.numpy()})))
        part = flow_session.run(None, {"frames": frames.numpy()})
        split_gap = max(float(np.abs(a.astype(np.float32) - ref[n].astype(np.float32)).max())
                        for n, a in zip([metric_input], part[:1]))
        split_gap = max(split_gap, float(np.abs(part[-2].astype(np.float32) - ref["half0"].astype(np.float32)).max()))
        print(f"images dans [{low}, {high}] : écart de la coupe {split_gap:.2e}")

        native = {}
        for side in (0, 1):
            image = frames[:, 3 * side : 3 * side + 3].float()
            f1, f2, f3 = features(W, image)
            native[f"feature{side}_1"], native[f"source{side}_2"], native[f"source{side}_3"] = f1, f2, f3
        met, half, quarter = metric(W, torch.from_numpy(part[0].astype(np.float32)))
        for side in (0, 1):
            for level, m in ((1, met), (2, half), (3, quarter)):
                flow = torch.from_numpy(part[1 + 3 * side + level - 1].astype(np.float32))
                native[f"motion{side}_{level}"] = torch.cat([flow, m[:, side : side + 1]], 1)
        for name, value in native.items():
            r = torch.from_numpy(ref[name].astype(np.float32))
            if name.startswith("motion"):
                value, r = value[:, 2:], r[:, 2:]
            peak = float(r.abs().max())
            p = psnr(value, r, peak)
            gap = float((value - r).abs().max())
            print(f"  {name:12s} {tuple(r.shape)} crête {peak:7.3f} PSNR {p:6.1f} dB, écart max {gap:.2e}")
            worst[name] = min(worst.get(name, float("inf")), p)

        # De bout en bout : splats et fusion (ONNX) sur la paire native, contre la paire entière.
        for t in (0.25, 0.5):
            images = []
            for outputs in (ref, {**ref, **{k: v.half().numpy() for k, v in native.items()}}):
                pair = [torch.from_numpy(outputs[n]) for n in names[:-1]]
                half0, half1, warped = splat_all(pair, t)
                time = torch.tensor([[[[t]]]], dtype=torch.float16)
                feeds = dict(zip(fusion_names("union"), [half0, half1, time, *warped]))
                images.append(fusion.run(None, {k: v.numpy() for k, v in feeds.items()})[0].astype(np.float32))
            p = psnr(images[1], images[0], 1.0)
            print(f"  image finale t={t} : PSNR {p:.1f} dB, écart max {np.abs(images[1] - images[0]).max():.2e}")
            worst["image"] = min(worst.get("image", float("inf")), p)
    cut.unlink()
    print("pire PSNR :", {k: round(v, 1) for k, v in worst.items()})


if __name__ == "__main__":
    main()
