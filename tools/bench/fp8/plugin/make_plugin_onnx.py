#!/usr/bin/env python3
"""SPAN FP16 -> (1) ONNX FP8 Q/DQ de base avec VRAIE calibration (maximum observé sur des images
d'anime réelles), (2) même modèle où Add(saut)+Tanh+Mul+Q des blocs choisis sont remplacés par le
nœud custom AjpSpanTailFp8 (plugin TensorRT, voir span_tail_plugin.cpp).

Usage : make_plugin_onnx.py fp16.onnx dossier_sortie --frames f1.png f2.png ... [--blocks 2,3,4,5,6,7]
Sorties : <dossier>/qdq.onnx (référence FP8 Q/DQ), <dossier>/plugin.onnx, <dossier>/scales.txt.
Les deux modèles partagent exactement les mêmes échelles : seule la fusion change."""
import argparse, os, sys
import numpy as np, onnx, onnxruntime as ort, ml_dtypes
from onnx import numpy_helper, helper, TensorProto as T
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("outdir")
ap.add_argument("--frames", nargs="+", required=True)
ap.add_argument("--crop", default="544x960", help="HxW du recadrage natif pour la calibration")
ap.add_argument("--blocks", default="2,3,4,5,6,7")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
FP8 = ml_dtypes.float8_e4m3fn
ch, cw = map(int, a.crop.split("x"))

m = onnx.load(a.src); g = m.graph
et = g.input[0].type.tensor_type.elem_type; npdt = np.float16 if et == T.FLOAT16 else np.float32
convs = [n for n in g.node if n.op_type == "Conv"]

# --- calibration : maximum absolu de chaque entrée de convolution sur les images fournies ---
def load_crop(p):
    im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
    y0, x0 = (im.shape[0] - ch) // 2, (im.shape[1] - cw) // 2
    return im[y0:y0 + ch, x0:x0 + cw].transpose(2, 0, 1)[None]
cal = onnx.ModelProto(); cal.CopyFrom(m)
for nm in dict.fromkeys(n.input[0] for n in convs): cal.graph.output.append(helper.make_tensor_value_info(nm, et, None))
sess = ort.InferenceSession(cal.SerializeToString(), providers=["CPUExecutionProvider"])
amax = {}
outs = {x.name for x in g.output}
for p in a.frames:
    for o, v in zip(sess.get_outputs(), sess.run(None, {"input": load_crop(p).astype(npdt)})):
        if o.name in outs: continue
        amax[o.name] = max(amax.get(o.name, 0), float(np.abs(v.astype(np.float32)).max()))
    print("calibré", os.path.basename(p), flush=True)
scale_of = {k: max(v, 1e-6) / 448 for k, v in amax.items()}
with open(os.path.join(a.outdir, "scales.txt"), "w") as f:
    for k, v in scale_of.items(): f.write(f"{k}\tamax={amax[k]:.4f}\tscale={v:.6g}\n")

# --- modèle Q/DQ de base (même construction que make_fp8_qdq.py, échelles calibrées ici) ---
inits = {i.name: i for i in g.initializer}; new = []; done = {}
def const(name, arr): g.initializer.append(numpy_helper.from_array(arr, name))
for n in convs:
    w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
    s = np.abs(w).reshape(w.shape[0], -1).max(1) / 448; s[s == 0] = 1
    wq = (w / s.reshape(-1, 1, 1, 1)).astype(FP8)
    wn = n.input[1]
    g.initializer.remove(inits[wn])
    t = onnx.TensorProto(); t.name = wn + "_q8"; t.data_type = T.FLOAT8E4M3FN; t.dims.extend(wq.shape); t.raw_data = wq.view(np.uint8).tobytes(); g.initializer.append(t)
    const(wn + "_sc", s.astype(npdt))
    new.append(helper.make_node("DequantizeLinear", [wn + "_q8", wn + "_sc"], [wn + "_dq"], axis=0))
    n.input[1] = wn + "_dq"
    x = n.input[0]
    if x not in done:
        const(x + "_asc", np.array(scale_of[x], npdt))
        z = onnx.TensorProto(); z.name = x + "_azp"; z.data_type = T.FLOAT8E4M3FN; z.raw_data = b"\x00"; g.initializer.append(z)
        new += [helper.make_node("QuantizeLinear", [x, x + "_asc", x + "_azp"], [x + "_aq"], saturate=1),
                helper.make_node("DequantizeLinear", [x + "_aq", x + "_asc", x + "_azp"], [x + "_adq"])]
        done[x] = x + "_adq"
    n.input[0] = done[x]

def toposort(nodes):
    avail = {i.name for i in g.input} | {i.name for i in g.initializer}; pend = list(nodes); res = []
    while pend:
        nxt = []
        for n in pend:
            if all(i == "" or i in avail for i in n.input): res.append(n); avail.update(n.output)
            else: nxt.append(n)
        assert len(nxt) < len(pend), "cycle"; pend = nxt
    return res

for op in m.opset_import:
    if op.domain in ("", "ai.onnx"): op.version = max(op.version, 19)
base_nodes = list(g.node) + new
def save(nodes, path, check=True):
    del g.node[:]; g.node.extend(toposort(nodes))
    if check: onnx.checker.check_model(m)
    onnx.save(m, path); print("écrit", path)
save(base_nodes, os.path.join(a.outdir, "qdq.onnx"))

# --- remplacement par le plugin ---
byname = {n.name: n for n in base_nodes}
nodes = list(base_nodes); fused = []
for b in [int(x) for x in a.blocks.split(",") if x]:
    add, tanh, mul = byname[f"/block_{b}/Add"], byname[f"/block_{b}/Tanh"], byname[f"/block_{b}/Mul"]
    skip, c3 = add.input[0], add.input[1]
    att = tanh.input[0]
    assert mul.input[0] == add.output[0] and mul.input[1] == tanh.output[0]
    mo = mul.output[0]
    q = next(n for n in nodes if n.op_type == "QuantizeLinear" and n.input[0] == mo)
    plug = helper.make_node("AjpSpanTailFp8", [c3 + "_aq", att, skip + "_aq"], [q.output[0]], name=f"/block_{b}/AjpSpanTail",
                            domain="trt.plugins", plugin_version="1", plugin_namespace="",
                            s_c3=float(scale_of[c3]), s_skip=float(scale_of[skip]), s_out=float(scale_of[mo]))
    drop = {id(add), id(tanh), id(mul), id(q)}
    nodes = [n for n in nodes if id(n) not in drop] + [plug]
    fused.append((mo, q.output[0], plug))
    print(f"bloc {b} : s_c3={scale_of[c3]:.5g} s_skip={scale_of[skip]:.5g} s_out={scale_of[mo]:.5g}")
# Si la sortie FP16 d'origine a encore des consommateurs (Concat final), on la rend par un DQ du FP8.
for mo, qo, plug in fused:
    if any(mo in n.input for n in nodes if n is not plug):
        nodes.append(helper.make_node("DequantizeLinear", [qo, mo + "_asc", mo + "_azp"], [mo]))
m.opset_import.append(helper.make_opsetid("trt.plugins", 1))
save(nodes, os.path.join(a.outdir, "plugin.onnx"), check=False)
