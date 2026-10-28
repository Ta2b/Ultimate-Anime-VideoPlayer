#!/usr/bin/env python3
"""Borne haute du FP8 sur TensorRT, sans SPAN : une chaîne de N convolutions 3x3 (C canaux, H x W), en FP16 et
en FP8 E4M3 Q/DQ. Motif « Conv, Relu, Q, DQ, Conv » : le Q se fond dans le noyau qui produit l'activation, le
DQ dans la convolution suivante, poids en constantes FP8 (une échelle par tenseur).
Si cette chaîne n'est pas plus rapide en FP8 qu'en FP16 sous trtexec, aucun SPAN ne le sera à ces canaux.

  python3 conv_microbench.py --out dossier [--channels 48,64,128,256] [--size 1088x1920] [--layers 8]
puis, pour chaque paire écrite (conv_c64_fp16.onnx, conv_c64_fp8.onnx) :
  trtexec --onnx=FICHIER --warmUp=500 --iterations=200 --useCudaGraph --noDataTransfers
et comparer « GPU Compute Time … median »."""
import argparse
from pathlib import Path
import numpy as np, onnx, ml_dtypes
from onnx import helper, numpy_helper, TensorProto as T

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--channels", default="48,64,128,256")
ap.add_argument("--size", default="1088x1920")
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--weights", choices=("per-tensor", "per-channel"), default="per-tensor")
args = ap.parse_args()
h, w = (int(v) for v in args.size.lower().split("x"))
args.out.mkdir(parents=True, exist_ok=True)
FP8 = ml_dtypes.float8_e4m3fn


def build(c: int, fp8: bool) -> onnx.ModelProto:
    rng = np.random.default_rng(c)
    nodes, inits = [], []
    cur = "input"
    for i in range(args.layers):
        wt = (rng.standard_normal((c, c, 3, 3)) * (1.0 / np.sqrt(9 * c))).astype(np.float32)
        wname = f"w{i}"
        if fp8:
            if args.weights == "per-tensor":
                s = np.array(max(float(np.abs(wt).max()), 1e-8) / 448, np.float16)
                q = (wt / float(s)).astype(FP8)
                axis = {}
            else:
                s = (np.abs(wt).reshape(c, -1).max(1) / 448).astype(np.float16)
                q = (wt / s.astype(np.float32).reshape(-1, 1, 1, 1)).astype(FP8)
                axis = {"axis": 0}
            t = onnx.TensorProto(); t.name = wname + "_q"; t.data_type = T.FLOAT8E4M3FN
            t.dims.extend(q.shape); t.raw_data = q.view(np.uint8).tobytes(); inits.append(t)
            inits.append(numpy_helper.from_array(s, wname + "_s"))
            nodes.append(helper.make_node("DequantizeLinear", [wname + "_q", wname + "_s"], [wname], **axis))
            a_scale = numpy_helper.from_array(np.array(1.0 / 448 * 4, np.float16), f"a{i}_s")
            zp = onnx.TensorProto(); zp.name = f"a{i}_z"; zp.data_type = T.FLOAT8E4M3FN; zp.raw_data = b"\x00"
            inits += [a_scale, zp]
            nodes.append(helper.make_node("QuantizeLinear", [cur, f"a{i}_s", f"a{i}_z"], [f"a{i}_q"], saturate=1))
            nodes.append(helper.make_node("DequantizeLinear", [f"a{i}_q", f"a{i}_s", f"a{i}_z"], [f"a{i}_dq"]))
            conv_in = f"a{i}_dq"
        else:
            inits.append(numpy_helper.from_array(wt.astype(np.float16), wname))
            conv_in = cur
        bias = numpy_helper.from_array(np.zeros(c, np.float16), f"b{i}")
        inits.append(bias)
        nodes.append(helper.make_node("Conv", [conv_in, wname, f"b{i}"], [f"c{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"]))
        cur = f"r{i}"
    graph = helper.make_graph(
        nodes, "convs",
        [helper.make_tensor_value_info("input", T.FLOAT16, [1, c, h, w])],
        [helper.make_tensor_value_info(cur, T.FLOAT16, [1, c, h, w])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


for c in (int(v) for v in args.channels.split(",")):
    for fp8 in (False, True):
        path = args.out / f"conv_c{c}_{'fp8' if fp8 else 'fp16'}.onnx"
        onnx.save(build(c, fp8), path)
        print("écrit", path)
