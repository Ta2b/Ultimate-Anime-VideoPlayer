#!/usr/bin/env python3
"""ONNX de comparaison pour le prototype CUTLASS : Conv3x3 64->64 + Relu sur 1x64x544x960, en FP16 et en FP8 E4M3 Q/DQ,
et (option --tail) la fin de bloc SPAN : c3 = Conv3x3(c2), s = c3 + x, att = Tanh(Conv1x1(c3)), out = s * att.
Mêmes motifs que tools/fp8/conv_microbench.py (poids en constantes FP8, une échelle par tenseur), plus un Q/DQ en sortie
pour que le noyau TensorRT produise bien de l'E4M3 comme le noyau CUTLASS.

  python3 gen_onnx.py --out dossier
  trtexec --onnx=conv_fp16.onnx --fp16 --useCudaGraph --warmUp=800 --iterations=300 --noDataTransfers
  trtexec --onnx=conv_fp8.onnx  --fp16 --fp8 --useCudaGraph --warmUp=800 --iterations=300 --noDataTransfers"""
import argparse
from pathlib import Path
import numpy as np, onnx, ml_dtypes
from onnx import helper, numpy_helper, TensorProto as T

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--size", default="544x960")
ap.add_argument("--full", action="store_true", help="écrit span_fp16.onnx / span_fp8.onnx : bloc SPAN complet (c1, c2, c3 + attention)")
ap.add_argument("--tail", action="store_true", help="écrit block_fp16.onnx / block_fp8.onnx au lieu des convs seules")
args = ap.parse_args()
h, w = (int(v) for v in args.size.lower().split("x"))
args.out.mkdir(parents=True, exist_ok=True)
FP8 = ml_dtypes.float8_e4m3fn
C = 64


def fp8_tensor(name, q):
    t = onnx.TensorProto()
    t.name, t.data_type = name, T.FLOAT8E4M3FN
    t.dims.extend(q.shape)
    t.raw_data = q.view(np.uint8).tobytes()
    return t


def zero_fp8(name):
    t = onnx.TensorProto()
    t.name, t.data_type, t.raw_data = name, T.FLOAT8E4M3FN, b"\x00"
    return t


def build(fp8: bool) -> onnx.ModelProto:
    rng = np.random.default_rng(1)
    wt = (rng.standard_normal((C, C, 3, 3)) * np.sqrt(2.0 / (9 * C))).astype(np.float32)
    bias = numpy_helper.from_array((rng.uniform(-0.1, 0.1, C)).astype(np.float16), "b")
    nodes, inits = [], [bias]
    src = "input"
    if fp8:
        s = np.array(float(np.abs(wt).max()) / 448, np.float16)
        inits += [fp8_tensor("w_q", (wt / float(s)).astype(FP8)), numpy_helper.from_array(s, "w_s")]
        nodes.append(helper.make_node("DequantizeLinear", ["w_q", "w_s"], ["w"]))
        # entrée activations ~[0,1] -> échelle 1/448 ; sortie conv ~[0,~3] -> échelle 4/448
        for tag, sc in (("in", 1.0 / 448), ("out", 4.0 / 448)):
            inits += [numpy_helper.from_array(np.array(sc, np.float16), f"{tag}_s"), zero_fp8(f"{tag}_z")]
        nodes.append(helper.make_node("QuantizeLinear", ["input", "in_s", "in_z"], ["in_q"], saturate=1))
        nodes.append(helper.make_node("DequantizeLinear", ["in_q", "in_s", "in_z"], ["in_dq"]))
        src = "in_dq"
    else:
        inits.append(numpy_helper.from_array(wt.astype(np.float16), "w"))
    nodes.append(helper.make_node("Conv", [src, "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node("Relu", ["c"], ["r"]))
    out = "r"
    if fp8:
        nodes.append(helper.make_node("QuantizeLinear", ["r", "out_s", "out_z"], ["o_q"], saturate=1))
        nodes.append(helper.make_node("DequantizeLinear", ["o_q", "out_s", "out_z"], ["o"]))
        out = "o"
    graph = helper.make_graph(
        nodes, "conv1",
        [helper.make_tensor_value_info("input", T.FLOAT16, [1, C, h, w])],
        [helper.make_tensor_value_info(out, T.FLOAT16, [1, C, h, w])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if not args.tail and not args.full:
    for fp8 in (False, True):
        p = args.out / f"conv_{'fp8' if fp8 else 'fp16'}.onnx"
        onnx.save(build(fp8), p)
        print("écrit", p)


def qdq(nodes, inits, src, tag, scale):
    inits += [numpy_helper.from_array(np.array(scale, np.float16), f"{tag}_s"), zero_fp8(f"{tag}_z")]
    nodes.append(helper.make_node("QuantizeLinear", [src, f"{tag}_s", f"{tag}_z"], [f"{tag}_q"], saturate=1))
    nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_q", f"{tag}_s", f"{tag}_z"], [f"{tag}_dq"]))
    return f"{tag}_dq"


def weight(nodes, inits, name, wt, fp8):
    if not fp8:
        inits.append(numpy_helper.from_array(wt.astype(np.float16), name))
        return name
    s = np.array(float(np.abs(wt).max()) / 448, np.float16)
    inits += [fp8_tensor(name + "_q", (wt / float(s)).astype(FP8)), numpy_helper.from_array(s, name + "_s")]
    nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s"], [name]))
    return name


def build_block(fp8: bool) -> onnx.ModelProto:
    rng = np.random.default_rng(1)
    w3 = (rng.standard_normal((C, C, 3, 3)) * np.sqrt(2.0 / (9 * C))).astype(np.float32)
    w1 = (rng.standard_normal((C, C, 1, 1)) * np.sqrt(1.0 / C)).astype(np.float32)
    inits = [numpy_helper.from_array(rng.uniform(-0.1, 0.1, C).astype(np.float16), "b3"),
             numpy_helper.from_array(rng.uniform(-0.1, 0.1, C).astype(np.float16), "b1")]
    nodes = []
    w3n, w1n = weight(nodes, inits, "w3", w3, fp8), weight(nodes, inits, "w1", w1, fp8)
    c2, x = "c2", "x"
    if fp8:
        c2 = qdq(nodes, inits, "c2", "c2q", 1.0 / 448)
        x = qdq(nodes, inits, "x", "xq", 1.0 / 448)
    nodes.append(helper.make_node("Conv", [c2, w3n, "b3"], ["c3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    c3 = qdq(nodes, inits, "c3", "c3q", 4.0 / 448) if fp8 else "c3"
    nodes.append(helper.make_node("Conv", [c3, w1n, "b1"], ["a"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("Tanh", ["a"], ["att"]))
    nodes.append(helper.make_node("Add", ["c3", x], ["s"]))
    nodes.append(helper.make_node("Mul", ["s", "att"], ["m"]))
    out = qdq(nodes, inits, "m", "oq", 8.0 / 448) if fp8 else "m"
    vi = lambda n: helper.make_tensor_value_info(n, T.FLOAT16, [1, C, h, w])
    graph = helper.make_graph(nodes, "block", [vi("c2"), vi("x")], [vi(out)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if args.tail:
    for fp8 in (False, True):
        p = args.out / f"block_{'fp8' if fp8 else 'fp16'}.onnx"
        onnx.save(build_block(fp8), p)
        print("écrit", p)


def build_full(fp8: bool) -> onnx.ModelProto:
    """Bloc SPAN complet : c1 Conv+Relu, c2 Conv+Relu, c3 Conv ; s = c3 + x ; att = Tanh(Conv1x1(c3)) ; out = s * att."""
    rng = np.random.default_rng(1)
    inits, nodes = [], []
    for nm in ("1", "2", "3"):
        wt = (rng.standard_normal((C, C, 3, 3)) * np.sqrt(2.0 / (9 * C))).astype(np.float32)
        weight(nodes, inits, f"w{nm}", wt, fp8)
        inits.append(numpy_helper.from_array(rng.uniform(-0.1, 0.1, C).astype(np.float16), f"b{nm}"))
    wa = (rng.standard_normal((C, C, 1, 1)) * np.sqrt(1.0 / C)).astype(np.float32)
    weight(nodes, inits, "wa", wa, fp8)
    inits.append(numpy_helper.from_array(rng.uniform(-0.1, 0.1, C).astype(np.float16), "ba"))
    x = qdq(nodes, inits, "x", "xq", 1.0 / 448) if fp8 else "x"
    a = qdq(nodes, inits, x, "i1", 1.0 / 448) if fp8 else x
    nodes.append(helper.make_node("Conv", [a, "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node("Relu", ["c1"], ["r1"]))
    a = qdq(nodes, inits, "r1", "i2", 4.0 / 448) if fp8 else "r1"
    nodes.append(helper.make_node("Conv", [a, "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node("Relu", ["c2"], ["r2"]))
    a = qdq(nodes, inits, "r2", "i3", 4.0 / 448) if fp8 else "r2"
    nodes.append(helper.make_node("Conv", [a, "w3", "b3"], ["c3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    c3 = qdq(nodes, inits, "c3", "c3q", 4.0 / 448) if fp8 else "c3"
    nodes.append(helper.make_node("Conv", [c3, "wa", "ba"], ["att_pre"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("Tanh", ["att_pre"], ["att"]))
    nodes.append(helper.make_node("Add", ["c3", x], ["s"]))
    nodes.append(helper.make_node("Mul", ["s", "att"], ["m"]))
    out = qdq(nodes, inits, "m", "oq", 8.0 / 448) if fp8 else "m"
    vi = lambda n: helper.make_tensor_value_info(n, T.FLOAT16, [1, C, h, w])
    graph = helper.make_graph(nodes, "span_block", [vi("x")], [vi(out)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if args.full:
    for fp8 in (False, True):
        p = args.out / f"span_{'fp8' if fp8 else 'fp16'}.onnx"
        onnx.save(build_full(fp8), p)
        print("écrit", p)
