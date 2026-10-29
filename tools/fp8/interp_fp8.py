#!/usr/bin/env python3
"""Graphes FP8 (E4M3, nœuds Q/DQ) des réseaux d'interpolation : RIFE, DistilDRBA, GMFSS (docs/fp8-interpolation.md).

Reproductible, sans PyTorch ni ModelOpt (ModelOpt exige torch ; ses réglages par défaut ne font pas mieux que ce
placement pour TensorRT : Q/DQ devant chaque convolution retenue, échelles statiques calibrées sur de vraies images).

  interp_fp8.py inputs  rife|drba|gmfss-pair|gmfss-fusion MODELE.onnx IMAGES_DIR SORTIE_DIR [--size HxW]
        fabrique les jeux d'entrées (.npz) bien formés depuis des PNG 1280x720 (couples d'images consécutives) :
        RIFE : 11 plans (a, b, t, grille X/Y, pas X/Y) ; DRBA : pair01 / pair12 à 7 plans ; GMFSS : 6 plans.
  interp_fp8.py capture MODELE.onnx PIECE.onnx... --inputs SORTIE_DIR/*.npz --out DIR
        exécute le MODELE entier (ONNX Runtime CPU) et enregistre pour chaque morceau d'un réseau fusionné les
        tenseurs qui entrent dans le morceau (mêmes noms de tenseurs que dans le graphe d'origine).
  interp_fp8.py quantize SRC.onnx DST.onnx --inputs a.npz b.npz ... [--min-channels 32] [--acts all|single]
        [--calib max|p99.99] [--keep-first] [--keep-last]
        écrit le graphe FP8 : poids par tenseur en E4M3 + DQ, activations Q/DQ devant chaque convolution retenue
  interp_fp8.py quality rife|drba|gmfss --ref FP16 --test FP8 --inputs *.npz
        PSNR / SSIM de l'image finale contre le FP16 sur chaque cas ; sortie 3 si les gardes (PSNR moyen >= 42 dB,
        pire cas >= 38 dB, SSIM moyen >= 0,985) ne sont pas tenues. GMFSS : --ref paire.onnx,fusion.onnx

Les convolutions de moins de --min-channels canaux d'entrée ou de sortie restent en FP16 (le FP8 n'aide pas sous
64 canaux, microbancs du 28/10, et la première et la dernière convolution sont les plus sensibles)."""
import argparse
import sys

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper, numpy_helper

FP8 = ml_dtypes.float8_e4m3fn


# ---------------------------------------------------------------- entrées bien formées
def load_png(path, height, width):
    from PIL import Image

    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0  # H W 3
    im = im.transpose(2, 0, 1)
    ph, pw = height - im.shape[1], width - im.shape[2]
    if ph < 0 or pw < 0:
        raise SystemExit(f"{path} est plus grande que {height}x{width}")
    return np.pad(im, ((0, 0), (0, ph), (0, pw)), mode="edge")


def pairs(images_dir):
    """Par instant, les deux couples d'images consécutives (k, k+1) qui BOUGENT le plus (écart moyen, avec une
    image k+2 pour DistilDRBA). L'anime se joue souvent « sur deux » : un couple d'images identiques ne teste
    rien (flot nul, PSNR de plusieurs dizaines de dB sans signification)."""
    import glob
    import os
    import re

    from PIL import Image

    groups = {}
    for p in sorted(glob.glob(os.path.join(images_dir, "t*_*.png"))):
        m = re.match(r"t(\d+)_(\d+)\.png", os.path.basename(p))
        groups.setdefault(m.group(1), {})[int(m.group(2))] = p
    out = []
    for t, g in groups.items():
        ks = sorted(g)
        small = {k: np.asarray(Image.open(g[k]).convert("L").resize((320, 180)), np.float32) for k in ks}
        diffs = sorted(((float(np.abs(small[k + 1] - small[k]).mean()), k) for k in ks[:-2]), reverse=True)
        chosen = []
        for _, k in diffs:
            if all(abs(k - c) >= 2 for c in chosen):
                chosen.append(k)
            if len(chosen) == 2:
                break
        for j, k in enumerate(chosen):
            out.append((f"t{t}_{j}", g[k], g[k + 1]))
    return out


def rife_input(a, b, t):
    _, h, w = a.shape
    x = np.empty((1, 11, h, w), np.float32)
    x[0, 0:3], x[0, 3:6], x[0, 6] = a, b, t
    xs = 2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0
    ys = 2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0
    x[0, 7], x[0, 8] = xs[None, :], ys[:, None]
    x[0, 9], x[0, 10] = 2.0 / (w - 1), 2.0 / (h - 1)
    return x.astype(np.float16)


def cmd_inputs(args):
    import os

    h, w = (int(v) for v in args.size.split("x"))
    os.makedirs(args.out, exist_ok=True)
    ps = pairs(args.images)
    prev = None
    for name, pa, pb in ps:
        a, b = load_png(pa, h, w), load_png(pb, h, w)
        if args.kind == "rife":
            d = {"input": rife_input(a, b, 0.5)}
        elif args.kind == "gmfss-pair":
            d = {"frames": np.concatenate([a, b])[None].astype(np.float16)}
        elif args.kind == "drba":
            # pair01 / pair12 : deux images + un plan d'instant et deux plans de flot nul (7 plans, 1 × 7 × H × W)
            def pair(x, y):
                z = np.zeros((1, 7, h, w), np.float32)
                z[0, 0:3], z[0, 3:6], z[0, 6] = x, y, 0.5
                return z.astype(np.float16)

            import re

            third = re.sub(r"_(\d+)\.png$", lambda m_: f"_{int(m_.group(1)) + 1}.png", pb)
            if not os.path.exists(third):
                continue
            d = {"pair01": pair(a, b), "pair12": pair(b, load_png(third, h, w))}
        else:
            raise SystemExit("gmfss-fusion : utiliser capture depuis la paire")
        np.savez(os.path.join(args.out, f"{name}.npz"), **d)
        print("écrit", name, {k: v.shape for k, v in d.items()})


# ---------------------------------------------------------------- capture
def cmd_capture(args):
    import os

    full = onnx.load(args.model)
    want = []
    for pth in args.pieces:
        pm = onnx.load(pth)
        want += [i.name for i in pm.graph.input if i.name not in {x.name for x in pm.graph.initializer}]
    want = list(dict.fromkeys(want))
    et = full.graph.input[0].type.tensor_type.elem_type
    have = {o.name for o in full.graph.output}
    for nm in want:
        if nm not in have:
            full.graph.output.append(helper.make_tensor_value_info(nm, et, None))
    sess = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
    os.makedirs(args.out, exist_ok=True)
    for npz in args.inputs:
        feed = dict(np.load(npz))
        outs = sess.run(want, feed)
        np.savez(os.path.join(args.out, os.path.basename(npz)), **dict(zip(want, outs)))
        print("capturé", os.path.basename(npz), want)


# ---------------------------------------------------------------- quantification
REDUCES = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum",
           "ReduceLogSumExp", "ReduceSumSquare"}


def reduce_axes_to_input(g):
    """Opset 18 : l'attribut axes des réductions devient une entrée (ReduceSum l'avait déjà depuis 13)."""
    for n in g.node:
        if n.op_type in REDUCES:
            for a in list(n.attribute):
                if a.name == "axes":
                    name = n.name + "_axes"
                    g.initializer.append(numpy_helper.from_array(np.array(a.ints, np.int64), name))
                    n.input.append(name)
                    n.attribute.remove(a)


def quantize(src, dst, input_files, min_channels=32, acts="all", calib="max", keep_first=False, keep_last=False,
             conv_range=None, skip=()):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    et = g.input[0].type.tensor_type.elem_type
    npdt = np.float16 if et == T.FLOAT16 else np.float32
    convs = [n for n in g.node if n.op_type == "Conv" and n.input[1] in inits]

    def chans(n):
        w = inits[n.input[1]]
        return w.dims[1] * int(next((a.i for a in n.attribute if a.name == "group"), 1)), w.dims[0]

    chosen = [n for i, n in enumerate(convs)
              if min(chans(n)) >= min_channels and (conv_range is None or any(a <= i < b for a, b in conv_range))
              and not n.name.startswith(tuple(skip))]
    if keep_first and chosen and chosen[0] is convs[0]:
        chosen = chosen[1:]
    if keep_last and chosen and chosen[-1] is convs[-1]:
        chosen = chosen[:-1]
    if not chosen:
        raise SystemExit("aucune convolution retenue")

    cal = onnx.ModelProto()
    cal.CopyFrom(m)
    acts_in = list(dict.fromkeys(n.input[0] for n in chosen))
    gout = {o.name for o in g.output}
    for nm in acts_in:
        if nm not in gout:
            cal.graph.output.append(helper.make_tensor_value_info(nm, et, None))
    sess = ort.InferenceSession(cal.SerializeToString(), providers=["CPUExecutionProvider"])
    amax = {}
    pct = None if calib == "max" else float(calib[1:])
    for f in input_files:
        z = np.load(f)
        feed = {i.name: z[i.name] for i in sess.get_inputs()}
        for o, v in zip(sess.get_outputs(), sess.run(None, feed)):
            if o.name not in acts_in:
                continue
            a = np.abs(v.astype(np.float32)).ravel()
            if pct is not None and a.size > 4096:
                x = float(np.percentile(a[:: max(1, a.size // 2_000_000)], pct))
            else:
                x = float(a.max())
            amax[o.name] = max(amax.get(o.name, 0.0), x)

    new, done = [], {}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers[i] = consumers.get(i, 0) + 1

    def const(name, arr):
        g.initializer.append(numpy_helper.from_array(arr, name))

    nq = 0
    wdone = set()
    for n in chosen:
        w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        s = max(float(np.abs(w).max()), 1e-8) / 448.0
        wq = np.clip(w / s, -448, 448).astype(FP8)
        wn = n.input[1]
        if wn not in wdone:  # un poids partagé par plusieurs convolutions n'est quantifié qu'une fois
            g.initializer.remove(inits[wn])
            t = onnx.TensorProto()
            t.name, t.data_type = wn + "_q8", T.FLOAT8E4M3FN
            t.dims.extend(wq.shape)
            t.raw_data = wq.view(np.uint8).tobytes()
            g.initializer.append(t)
            const(wn + "_sc", np.array(s, npdt))
            new.append(helper.make_node("DequantizeLinear", [wn + "_q8", wn + "_sc"], [wn + "_dq"]))
            wdone.add(wn)
        n.input[1] = wn + "_dq"
        a = n.input[0]
        if acts == "single" and consumers.get(a, 0) > 1:
            continue
        if a not in done:
            const(a + "_asc", np.array(max(amax[a], 1e-6) / 448.0, npdt))
            z = onnx.TensorProto()
            z.name, z.data_type, z.raw_data = a + "_azp", T.FLOAT8E4M3FN, b"\x00"
            g.initializer.append(z)
            new += [
                helper.make_node("QuantizeLinear", [a, a + "_asc", a + "_azp"], [a + "_aq"], saturate=1),
                helper.make_node("DequantizeLinear", [a + "_aq", a + "_asc", a + "_azp"], [a + "_adq"]),
            ]
            done[a] = a + "_adq"
        n.input[0] = done[a]
        nq += 1
    allnodes = list(g.node) + new
    del g.node[:]
    avail = {i.name for i in g.input} | {i.name for i in g.initializer}
    pend = allnodes
    while pend:  # tri topologique simple
        nxt = []
        for n in pend:
            if all(i == "" or i in avail for i in n.input):
                g.node.append(n)
                avail.update(n.output)
            else:
                nxt.append(n)
        assert len(nxt) < len(pend), "graphe cyclique ou entrée orpheline"
        pend = nxt
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            if op.version < 18:
                reduce_axes_to_input(g)
            op.version = max(op.version, 19)  # QuantizeLinear en float8 : opset 19
    try:
        onnx.checker.check_model(m, full_check=False)
    except onnx.checker.ValidationError as e:  # sorties de morceaux sans forme : TensorRT les accepte
        if "shape" not in str(e):
            raise
    onnx.save(m, dst)
    return nq, len(convs)


def cmd_quantize(args):
    nq, nc = quantize(args.src, args.dst, args.inputs, args.min_channels, args.acts, args.calib, args.keep_first, args.keep_last,
                       conv_range=[tuple(int(v) for v in r.split(":")) for r in args.conv_range.split(",")] if args.conv_range else None,
                       skip=args.skip)
    print(f"écrit {args.dst} : {nq} convolutions quantifiées sur {nc}")


# ---------------------------------------------------------------- métriques
def psnr(ref, test, peak=None):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0:
        return 200.0
    peak = peak if peak is not None else max(float(np.abs(ref).max()), 1e-6)
    return 10 * np.log10(peak * peak / mse)


def ssim(a, b, peak=1.0):
    # SSIM globale par fenêtres 8x8 sans recouvrement, sur la luminance ; suffisant pour une garde.
    def lum(x):
        x = x.astype(np.float64)
        return x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722 if x.shape[0] == 3 else x[0]

    a, b = lum(a), lum(b)
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    a, b = a[:h, :w], b[:h, :w]

    def blocks(x):
        return x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)

    A, B = blocks(a), blocks(b)
    ma, mb = A.mean(1), B.mean(1)
    va, vb = A.var(1), B.var(1)
    cov = ((A - ma[:, None]) * (B - mb[:, None])).mean(1)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))))


GATES = {"psnr_mean": 42.0, "psnr_worst": 38.0, "ssim_mean": 0.985}  # gardes de qualité du lecteur


def plain_session(path):
    """Sans optimisations : celles d'ONNX Runtime déplacent les Q/DQ FP8 à travers les Transpose (type refusé)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def gmfss_image(models, z, t=0.5):
    import pathlib

    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gmfss"))
    from gmfss_graph import fusion_names, soft_splat

    pair, fusion = plain_session(models[0]), plain_session(models[1])
    union = "time" in {i.name for i in fusion.get_inputs()}
    names = [o.name for o in pair.get_outputs()]
    named = {k: torch.from_numpy(v) for k, v in zip(names, pair.run(None, {"frames": z["frames"]}))}
    warped = []
    for side in (0, 1):
        ts = t if side == 0 else 1.0 - t
        for level in (1, 2, 3):
            motion, source = named[f"motion{side}_{level}"], named[f"source{side}_{level}"]
            if level == 1 and source.shape[1] == 64:
                source = torch.cat([named[f"half{side}"], source], 1)
            warped.append(soft_splat(source, motion[:, 0:2] * ts, motion[:, 2:3] * ts))
    feed = dict(zip(fusion_names("union" if union else "base"),
                    [named["half0"], named["half1"]] + ([torch.full((1, 1, 1, 1), t)] if union else []) + warped))
    out = fusion.run(None, {k: v.numpy().astype(np.float16) for k, v in feed.items()})[0]
    return np.clip(out[0].astype(np.float32), 0, 1)


def run_images(kind, models, feed_file):
    """L'image qu'un réseau (ou la paire de réseaux GMFSS) rend pour un jeu d'entrées (ONNX Runtime CPU : les
    Q/DQ FP8 y sont émulés exactement, ce que TensorRT calcule à l'ordre d'accumulation près)."""
    z = np.load(feed_file)
    if kind == "gmfss":
        return gmfss_image(models, z)
    sess = plain_session(models[0])
    out = sess.run(None, {i.name: z[i.name] for i in sess.get_inputs()})[0]
    return np.clip(out[0].astype(np.float32), 0, 1)


def cmd_quality(args):
    ref_models, test_models = args.ref.split(","), args.test.split(",")
    rows = []
    for f in args.inputs:
        a = run_images(args.kind, ref_models, f)
        b = run_images(args.kind, test_models, f)
        rows.append((f.split("/")[-1], psnr(a, b, 1.0), ssim(a, b)))
        print(f"{rows[-1][0]:16s} PSNR {rows[-1][1]:6.2f} dB  SSIM {rows[-1][2]:.4f}", flush=True)
    p = np.array([r[1] for r in rows])
    q = np.array([r[2] for r in rows])
    ok = p.mean() >= GATES["psnr_mean"] and p.min() >= GATES["psnr_worst"] and q.mean() >= GATES["ssim_mean"]
    print(f"PSNR moyen {p.mean():.2f} pire {p.min():.2f} | SSIM moyen {q.mean():.4f} pire {q.min():.4f} | "
          f"gardes (42 / 38 / 0,985) : {'TENUES' if ok else 'NON TENUES'}")
    return 0 if ok else 3


def cmd_psnr(args):
    A, B = np.load(args.a), np.load(args.b)
    for k in A.files:
        print(k, f"PSNR {psnr(A[k], B[k]):.2f} dB")


def cmd_gmfss_fusion(args):
    """Entrées du graphe « fusion » de GMFSS : la paire (ONNX Runtime CPU) puis les six splats à l'instant t,
    comme le lecteur (tools/gmfss/gmfss_graph.py ; demande PyTorch CPU, seulement ici)."""
    import os
    import pathlib

    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gmfss"))
    from gmfss_graph import fusion_names, soft_splat

    os.makedirs(args.out, exist_ok=True)
    sess = ort.InferenceSession(args.pair, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    for npz in args.inputs:
        frames = np.load(npz)["frames"]
        named = {k: torch.from_numpy(v) for k, v in zip(names, sess.run(None, {"frames": frames}))}
        warped = []
        for side in (0, 1):  # les six splats à l'instant t, comme le lecteur (gmfss_side_time)
            ts = args.t if side == 0 else 1.0 - args.t
            for level in (1, 2, 3):
                motion = named[f"motion{side}_{level}"]
                source = named[f"source{side}_{level}"]
                if level == 1 and source.shape[1] == 64:  # export v2 : l'image se joint aux caractéristiques
                    source = torch.cat([named[f"half{side}"], source], 1)
                warped.append(soft_splat(source, motion[:, 0:2] * ts, motion[:, 2:3] * ts))
        feeds = dict(zip(fusion_names(args.model_type), [named["half0"], named["half1"]]
                         + ([torch.full((1, 1, 1, 1), args.t)] if args.model_type == "union" else []) + warped))
        np.savez(os.path.join(args.out, os.path.basename(npz)),
                 **{k: v.numpy().astype(np.float16) for k, v in feeds.items()})
        print("fusion", os.path.basename(npz))


def cmd_shapes(args):
    m = onnx.load(args.model)
    init = {i.name for i in m.graph.initializer}
    print(";".join(
        i.name + ":" + "x".join(str(d.dim_value) for d in i.type.tensor_type.shape.dim)
        for i in m.graph.input if i.name not in init))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("kind", choices=("rife", "drba", "gmfss-pair"))
    p.add_argument("model")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--size", default="768x1280")
    p.set_defaults(fn=cmd_inputs)
    p = sub.add_parser("capture")
    p.add_argument("model")
    p.add_argument("pieces", nargs="+")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("quantize")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--min-channels", type=int, default=32)
    p.add_argument("--acts", choices=("all", "single"), default="all")
    p.add_argument("--calib", default="max")
    p.add_argument("--keep-first", action="store_true")
    p.add_argument("--keep-last", action="store_true")
    p.add_argument("--skip", action="append", default=[], help="préfixe du nom de nœud à laisser en FP16 (répétable) ; GMFSS : /flownet/backbone/")
    p.add_argument("--conv-range", help="A:B[,C:D...] : seules les convolutions d'indice A à B-1 (ordre du graphe), pour écarter les couches sensibles")
    p.set_defaults(fn=cmd_quantize)
    p = sub.add_parser("gmfss-fusion")
    p.add_argument("pair")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--model-type", choices=("base", "union"), default="base")
    p.set_defaults(fn=cmd_gmfss_fusion)
    p = sub.add_parser("shapes")
    p.add_argument("model")
    p.set_defaults(fn=cmd_shapes)
    p = sub.add_parser("quality")
    p.add_argument("kind", choices=("rife", "drba", "gmfss"))
    p.add_argument("--ref", required=True, help="FP16 ; GMFSS : paire,fusion")
    p.add_argument("--test", required=True, help="FP8 ; GMFSS : paire,fusion")
    p.add_argument("--inputs", nargs="+", required=True)
    p.set_defaults(fn=cmd_quality)
    p = sub.add_parser("psnr")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_psnr)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
