#!/usr/bin/env python3
"""Émulation des sommes du GMFSS natif, morceau par morceau, contre ONNX Runtime (docs/gmfss-natif.md, « Essai par
morceaux et réglage »).

Refait en PyTorch le GridNet de la fusion (prepare/gmfss_native.cpp), le FeatureNet et le MetricNet de la paire
(prepare/gmfss_pair_native.cpp), avec les arrondis FP16 des noyaux de rife_native_kernels.cuh : poids en FP16,
PReLU de l'entrée arrondi au chargement de la tuile, somme et biais arrondis (stage), résiduel puis PReLU de
sortie arrondis (finish8). Les sommes suivent l'accumulateur des noyaux, dans leur ordre (tranche de Kc canaux,
prise, pas de 16 canaux) :
  f16    l'accumulateur en FP16, arrondi après chaque mma (16 produits, sommés en f32) ;
  mixtes chaque prise sommée en FP16 (Kc produits), les prises entre elles en f32 ;
  f32    tout en f32.
La tranche Kc est celle de l'ancien choix (64 si les canaux le permettent hors pas 2, sinon 32 ou 16) : le
réglage au banc peut en prendre une autre, ce qui ne change que l'ordre des arrondis f16. K découpé n'est pas
émulé (ses parts se somment en f32 : un peu plus juste).

Les morceaux et leurs crans sont ceux de gpu/cuda/backend/gmfss_mix.h : lignes 0, 1, 2 et queue du GridNet
(la tête : conv_last), FeatureNet (la tête : les trois convolutions de sortie) et MetricNet (la tête :
metric_out) ; f16, f16 et tête f32 (morceau à tête seulement), mixtes, f32. Chaque morceau est d'abord jugé
seul (les autres en f32), puis les mélanges (les 16 de la paire ; pour la fusion, un cran pour tous les
morceaux, ou ses 108 avec --all, deux heures à 1024 × 576) : la pire des sorties de l'unité contre ONNX
Runtime, l'écart rapporté à max(1, crête de la référence) comme l'essai au chargement (gmfss_trial.cpp).

    python -I tools/gmfss/emul_mix_native.py modeles/gmfss/gmfss_fortuna_union_pair.onnx \\
        modeles/gmfss/gmfss_fortuna_union_fusion.onnx --width 1024 --height 576

La taille est celle des images de la paire (la fusion à sa moitié) ; ses côtés sont des multiples de 64, comme
le veut la découpe de GMFlow dans la paire exportée.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
from emul_pair_native import Weights, frames_for, h16, prelu, tensor_names  # noqa: E402
from gmfss_graph import fusion_names, splat_all  # noqa: E402

LADDER = {
    "f16": ("f16", "f16"),
    "f16 et tête f32": ("f16", "f32"),
    "mixtes": ("mixed", "mixed"),
    "f32": ("f32", "f32"),
}
F16_MAX = 65504.0


def default_kc(cin, stride):
    c = (cin + 15) // 16 * 16
    return 64 if c % 64 == 0 and stride == 1 else 32 if c % 32 == 0 else 16


def accumulate(views, weights, sums, kc):
    """Les sommes d'une convolution dans l'ordre des noyaux : views[t] [C, P], weights[t] [Cout, C]."""
    channels = views[0].shape[0]
    acc = torch.zeros(weights[0].shape[0], views[0].shape[1])
    for q0 in range(0, (channels + 15) // 16 * 16, kc):
        for view, weight in zip(views, weights):
            part = torch.zeros_like(acc) if sums == "mixed" else None
            for s0 in range(q0, min(q0 + kc, channels), 16):
                d = weight[:, s0 : s0 + 16] @ view[s0 : s0 + 16]
                if sums == "f16":
                    acc = h16(acc + d)
                elif sums == "mixed":
                    part = h16(part + d)
                else:
                    acc = acc + d
            if part is not None:
                acc = acc + part
    return acc


def stage(acc, bias):
    return h16((acc + bias[:, None]).clamp(-F16_MAX, F16_MAX))


def finish(y, skip, slope):
    if skip is None and slope is None:
        return y
    if skip is not None:
        y = y + skip
    if slope is not None:
        y = prelu(y, slope)
    return h16(y)


def conv(x, wb, sums, stride=1, in_slope=None, out_slope=None, skip=None):
    """Une convolution 3 × 3 de bord 1 des noyaux sur x [C, H, W]."""
    w, b = wb
    if in_slope is not None:
        x = h16(prelu(x, in_slope))
    c, height, width = x.shape
    oh, ow = (height - 1) // stride + 1, (width - 1) // stride + 1
    xp = F.pad(x, (1, 1, 1, 1))
    views, weights = [], []
    for dy in range(3):
        for dx in range(3):
            views.append(xp[:, dy : dy + stride * (oh - 1) + 1 : stride, dx : dx + stride * (ow - 1) + 1 : stride]
                         .reshape(c, -1))
            weights.append(h16(w[:, :, dy, dx]))
    y = stage(accumulate(views, weights, sums, default_kc(c, stride)), b).reshape(-1, oh, ow)
    return finish(y, skip, out_slope)


def conv_transposed(x, wb, sums, in_slope=None):
    """La transposée 4 × 4 de pas 2 en quatre phases 2 × 2 (poids ONNX [Cin][Cout][4][4])."""
    w, b = wb
    if in_slope is not None:
        x = h16(prelu(x, in_slope))
    c, height, width = x.shape
    xp = F.pad(x, (1, 1, 1, 1))
    out = torch.zeros(w.shape[1], 2 * height, 2 * width)
    for py in range(2):
        for px in range(2):
            views, weights = [], []
            for ty in range(2):
                for tx in range(2):
                    views.append(xp[:, py + ty : py + ty + height, px + tx : px + tx + width].reshape(c, -1))
                    weights.append(h16(w[:, :, 3 - py - 2 * ty, 3 - px - 2 * tx].t()))
            y = stage(accumulate(views, weights, sums, default_kc(c, 1)), b)
            out[:, py::2, px::2] = y.reshape(-1, height, width)
    return out


# Le GridNet de FusionNet_u, colonne par colonne (kBlocks de prepare/gmfss_native.cpp) : module, genre, entrée,
# résiduel, sortie. Le niveau d'un tenseur : la ligne du GridNet, le morceau des convolutions qui l'écrivent.
BLOCKS = [
    ("residual_model_head1", "same", "X1", None, "P"), ("residual_model_head0", "same", "X0", "P", "A"),
    ("residual_model_01", "same", "A", "A", "B"), ("residual_model_head2", "same", "X2", None, "Q"),
    ("downsample_model_10", "down", "A", "Q", "D"), ("residual_model_head3", "same", "X3", None, "R"),
    ("downsample_model_20", "down", "D", "R", "E"), ("residual_model_11", "same", "D", "D", "F"),
    ("downsample_model_11", "down", "B", "F", "F2"), ("residual_model_21", "same", "E", "E", "G"),
    ("downsample_model_21", "down", "F2", "G", "G2"), ("residual_model_24", "same", "G2", "G2", "H8"),
    ("residual_model_25", "same", "H8", "H8", "H9"), ("residual_model_14", "same", "F2", "F2", "I"),
    ("upsample_model_14", "up", "H8", "I", "I2"), ("residual_model_04", "same", "B", "B", "J"),
    ("upsample_model_04", "up", "I2", "J", "J2"), ("residual_model_15", "same", "I2", "I2", "K"),
    ("upsample_model_15", "up", "H9", "K", "K2"), ("residual_model_05", "same", "J2", "J2", "L"),
    ("upsample_model_05", "up", "K2", "L", "L2"),
]
LINE = {
    **{t: 0 for t in ("P", "A", "B", "J", "J2", "L", "L2")},
    **{t: 1 for t in ("Q", "D", "F", "F2", "I", "I2", "K", "K2")},
    **{t: 2 for t in ("R", "E", "G", "G2", "H8", "H9")},
}
FUSION_PIECES = ["ligne 0", "ligne 1", "ligne 2", "queue"]
PAIR_PIECES = ["FeatureNet", "MetricNet"]


class Fusion:
    def __init__(self, model):
        self.W = Weights(model)
        # Les entrées des têtes du GridNet, relevées par la structure : l'entrée du PReLU de chaque tête.
        nodes = {n.name: n for n in model.graph.node}
        self.inputs = {f"X{i}": nodes[f"/fusionnet/residual_model_head{i}/residual_model_head{i}.0/Relu"].input[0]
                       for i in range(4)}

    def slope(self, module, i):
        return self.W.slope(f"/fusionnet/{module}/{module}.{i}/Mul")

    def wb(self, name):
        return self.W.conv(f"fusionnet.{name}")

    def run(self, x, precisions):
        """Le GridNet sur les entrées x (X0 à X3, [C, H, W]) ; precisions : (corps, tête) par morceau."""
        t = dict(x)
        for module, kind, src, skip, dst in BLOCKS:
            sums = precisions[LINE[dst]][0]
            s1, s2 = self.slope(module, 0), self.slope(module, 2)
            first, second = self.wb(f"{module}.1"), self.wb(f"{module}.3")
            if kind == "up":
                mid = conv_transposed(t[src], first, sums, in_slope=s1)
                t[dst] = conv(mid, second, sums, in_slope=s2, skip=t.get(skip))
            else:
                mid = conv(t[src], first, sums, stride=2 if kind == "down" else 1, in_slope=s1, out_slope=s2)
                t[dst] = conv(mid, second, sums, skip=t.get(skip))
        body, head = precisions[3]
        tail = "residual_model_tail"
        slope = self.W.slope(f"/fusionnet/{tail}/conv_before_upsample/conv_before_upsample.1/Mul")
        y = conv(t["L2"], self.wb(f"{tail}.conv_before_upsample.0"), body, out_slope=slope)
        y = F.pixel_shuffle(conv(y, self.wb(f"{tail}.upsample.0"), body)[None], 2)[0]
        y = conv(y, self.wb(f"{tail}.conv_last"), head)
        return [y[:3].clamp(0, 1)]


class Pair:
    def __init__(self, model):
        self.W = Weights(model)

    def features(self, image, precision):
        body, head = precision
        out, x = [], image
        for b in (1, 2, 3):
            s0 = self.W.slope(f"/feat_ext/block{b}/block{b}.0/Mul")
            s2 = self.W.slope(f"/feat_ext/block{b}/block{b}.2/Mul")
            mid = conv(x, self.W.conv(f"feat_ext.block{b}.1"), body, stride=2, in_slope=s0, out_slope=s2)
            x = conv(mid, self.W.conv(f"feat_ext.block{b}.3"), head)
            out.append(x)
        return out

    def metric(self, inp, precision):
        body, head = precision
        m = conv(inp, self.W.conv("metricnet.metric_in"), body)
        for i in (1, 2, 3):
            s = self.W.slope(f"/metricnet/metric_net{i}/metric_net{i}.0/Mul")
            m = conv(m, self.W.conv(f"metricnet.metric_net{i}.1"), body, in_slope=s, skip=m)
        s = self.W.slope("/metricnet/metric_out/metric_out.0/Mul")
        out = conv(m, self.W.conv("metricnet.metric_out.1"), head, in_slope=s)[None]
        met = h16(h16(torch.tanh(out)) * 10)
        return met, h16(F.avg_pool2d(met, 2)), h16(F.avg_pool2d(met[:, :, 1:, 1:], 2, stride=4))

    def run(self, frames, metric_in, precisions):
        """Les sorties natives de la paire : caractéristiques des deux images, puis métriques (3e plan des
        mouvements)."""
        out = []
        for side in (0, 1):
            out += self.features(frames[0, 3 * side : 3 * side + 3].float(), precisions[0])
        met = self.metric(metric_in, precisions[1])
        for side in (0, 1):
            for level in range(3):
                out.append(met[level][0, side])
        return out


def worst_db(got, ref, groups=None):
    """La pire des sorties (groups : leurs indices, toutes par défaut)."""
    worst = float("inf")
    for i, (g, r) in enumerate(zip(got, ref)):
        if groups is not None and i not in groups:
            continue
        g, r = g.double(), r.double()
        mse = float(((g - r) ** 2).mean())
        if not np.isfinite(mse):
            return float("-inf")
        peak = max(1.0, float(r.abs().max()))
        worst = min(worst, 200.0 if mse == 0 else 10 * np.log10(peak * peak / mse))
    return worst


def heads_of(unit):
    return [False, False, False, True] if unit == "fusion" else [True, True]


def report(unit, pieces, run, reference, groups=None, every=False):
    """Chaque morceau seul (les autres en f32), puis les mélanges, en dB contre ORT et contre le natif f32 :
    tous s'ils sont au plus 32 (ou every), sinon un cran pour tous les morceaux (« f16 et tête f32 » : f16 sans
    tête) ; groups : des sorties jugées aussi à part, par nom."""
    names = list(LADDER)
    base = run([LADDER["f32"]] * len(pieces))
    print(f"\n{unit} : natif f32 partout contre ORT {worst_db(base, reference):.1f} dB")
    for name, group in (groups or {}).items():
        print(f"  dont {name} : {worst_db(base, reference, group):.1f} dB")
    for p, piece in enumerate(pieces):
        cells = []
        for name in names:
            if name == "f16 et tête f32" and not heads_of(unit)[p]:
                continue
            precisions = [LADDER["f32"]] * len(pieces)
            precisions[p] = LADDER[name]
            got = run(precisions)
            parts = "".join(f", {g} {worst_db(got, reference, i):.1f}" for g, i in (groups or {}).items())
            cells.append(f"{name} {worst_db(got, reference):.1f} dB (contre f32 {worst_db(got, base):.1f}{parts})")
        print(f"  {piece} seul : " + " ; ".join(cells))
    options = [[n for n in names if n != "f16 et tête f32" or heads_of(unit)[p]] for p in range(len(pieces))]
    mixes = list(itertools.product(*options))
    if len(mixes) > 32 and not every:
        mixes = [tuple(n if n in o else "f16" for o in options) for n in names]
    rows = []
    for mix in mixes:
        got = run([LADDER[n] for n in mix])
        rows.append((worst_db(got, reference), worst_db(got, base), mix))
    print(f"  {len(rows)} mélanges, du plus juste au moins juste (contre ORT, contre le natif f32) :")
    for db, db_base, mix in sorted(rows, reverse=True):
        print(f"    {db:6.1f} {db_base:6.1f}  " + ", ".join(f"{p} {n}" for p, n in zip(pieces, mix)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair", type=Path)
    parser.add_argument("fusion", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--only", choices=("pair", "fusion"))
    parser.add_argument("--all", action="store_true", help="tous les mélanges (108 pour la fusion)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    pair_model = onnx.load(str(args.pair))
    metric_input, flows = tensor_names(pair_model)
    pair_model.graph.output.append(onnx.helper.make_tensor_value_info(metric_input, onnx.TensorProto.FLOAT16, None))
    pair_session = ort.InferenceSession(pair_model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in pair_session.get_outputs()]
    frames = frames_for(args.width, args.height, 1, 0.0, 1.0)
    ref = dict(zip(names, pair_session.run(None, {"frames": frames.numpy()})))
    half = f"{args.width // 2}×{args.height // 2}"
    print(f"images {args.width}×{args.height} (la fusion à {half}), instant {args.time}")

    if args.only != "fusion":
        pair = Pair(pair_model)
        judged = [f"feature{s}_1" if lv == 1 else f"source{s}_{lv}" for s in (0, 1) for lv in (1, 2, 3)]
        reference = [torch.from_numpy(ref[n].astype(np.float32))[0] for n in judged]
        motions = [f"motion{s}_{lv}" for s in (0, 1) for lv in (1, 2, 3)]
        reference += [torch.from_numpy(ref[n].astype(np.float32))[0, 2] for n in motions]
        metric_in = torch.from_numpy(ref[metric_input].astype(np.float32))[0]
        groups = {"caractéristiques": range(0, 6), "métriques": range(6, 12)}
        report("paire", PAIR_PIECES, lambda p: pair.run(frames, metric_in, p), reference, groups, args.all)

    if args.only != "pair":
        fusion_model = onnx.load(str(args.fusion))
        fusion = Fusion(fusion_model)
        for x in fusion.inputs.values():
            fusion_model.graph.output.append(onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT16, None))
        session = ort.InferenceSession(fusion_model.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = [o.name for o in session.get_outputs()]
        pair_out = [torch.from_numpy(ref[n]) for n in names[:-1]]
        half0, half1, warped = splat_all(pair_out, args.time)
        time = torch.tensor([[[[args.time]]]], dtype=torch.float16)
        feeds = dict(zip(fusion_names("union"), [half0, half1, time, *warped]))
        got = dict(zip(outs, session.run(None, {k: v.numpy() for k, v in feeds.items()})))
        x = {k: torch.from_numpy(got[v].astype(np.float32))[0] for k, v in fusion.inputs.items()}
        reference = [torch.from_numpy(got["out"].astype(np.float32))[0]]
        report("fusion", FUSION_PIECES, lambda p: fusion.run(x, p), reference, every=args.all)


if __name__ == "__main__":
    main()
