#!/usr/bin/env python3
"""ONNX FP16 -> ONNX FP8 E4M3 Q/DQ (format attendu par TensorRT >= 10 : QuantizeLinear/DequantizeLinear
float8e4m3fn, opset 19). Poids : échelle par canal de sortie (absmax/448), constante FP8 + DQ.
Activations : échelle par tenseur (maximum calibré / 448) devant chaque Conv. Usage :
  make_fp8_qdq.py entree.onnx sortie.onnx [--weights per-tensor|per-channel] [--acts all|single-consumer|none]

Variantes à chronométrer (TensorRT ne fuse que certaines formes de Q/DQ) :
  --weights per-tensor     une échelle par poids, la forme que TensorRT fuse en FP8 (défaut) ;
  --weights per-channel    une échelle par canal de sortie (la première version, mesurée 15 % plus lente) ;
  --acts all               Q/DQ devant chaque Conv (défaut) ;
  --acts single-consumer   seulement quand l'activation ne sert qu'à cette Conv : le Q peut alors se fondre
                           dans le noyau qui la produit au lieu d'un noyau Quantize à part ;
  --acts none              poids seuls (aucune quantification des activations)."""
import sys, numpy as np, onnx, onnxruntime as ort, ml_dtypes
from onnx import numpy_helper, helper, TensorProto as T
from PIL import Image, ImageDraw
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--weights", choices=("per-tensor", "per-channel"), default="per-tensor")
ap.add_argument("--acts", choices=("all", "single-consumer", "none"), default="all")
args = ap.parse_args(); src, dst = args.src, args.dst
FP8 = ml_dtypes.float8_e4m3fn
def synth(seed, h=256, w=384):
    r = np.random.default_rng(seed); im = Image.new("RGB", (w, h), tuple(int(x) for x in r.integers(40, 230, 3))); d = ImageDraw.Draw(im)
    for _ in range(25):
        c = tuple(int(x) for x in r.integers(0, 255, 3)); x0, y0 = int(r.integers(0, w-40)), int(r.integers(0, h-40))
        (d.ellipse if r.random() < .5 else d.rectangle)([x0, y0, x0+int(r.integers(20, 150)), y0+int(r.integers(20, 120))], fill=c, outline=(10, 10, 10), width=int(r.integers(1, 4)))
    a = np.asarray(im, np.float32)/255
    return np.clip(a+np.linspace(0, 1, w)[None, :, None]*.25-.12, 0, 1).transpose(2, 0, 1)[None]
m = onnx.load(src); g = m.graph
et = g.input[0].type.tensor_type.elem_type; npdt = np.float16 if et == T.FLOAT16 else np.float32
convs = [n for n in g.node if n.op_type == "Conv"]
cal = onnx.ModelProto(); cal.CopyFrom(m)
for nm in dict.fromkeys(n.input[0] for n in convs): cal.graph.output.append(helper.make_tensor_value_info(nm, et, None))
sess = ort.InferenceSession(cal.SerializeToString(), providers=["CPUExecutionProvider"])
amax = {}
for k in range(3):
    for o, v in zip(sess.get_outputs(), sess.run(None, {"input": synth(k).astype(npdt)})):
        if o.name in [x.name for x in g.output]: continue
        amax[o.name] = max(amax.get(o.name, 0), float(np.abs(v.astype(np.float32)).max()))
inits = {i.name: i for i in g.initializer}; new = []; done = {}
def const(name, arr): g.initializer.append(numpy_helper.from_array(arr, name))
for n in convs:
    w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
    if args.weights == "per-tensor":
        s = np.full(w.shape[0], max(float(np.abs(w).max()), 1e-8)/448, np.float32)
    else:
        s = np.abs(w).reshape(w.shape[0], -1).max(1)/448; s[s == 0] = 1
    wq = (w/s.reshape(-1, 1, 1, 1)).astype(FP8)
    wn = n.input[1]
    g.initializer.remove(inits[wn])
    t = onnx.TensorProto(); t.name = wn+"_q8"; t.data_type = T.FLOAT8E4M3FN; t.dims.extend(wq.shape); t.raw_data = wq.view(np.uint8).tobytes(); g.initializer.append(t)
    if args.weights == "per-tensor":
        const(wn+"_sc", np.array(s[0], npdt))
        new.append(helper.make_node("DequantizeLinear", [wn+"_q8", wn+"_sc"], [wn+"_dq"]))
    else:
        const(wn+"_sc", s.astype(npdt))
        new.append(helper.make_node("DequantizeLinear", [wn+"_q8", wn+"_sc"], [wn+"_dq"], axis=0))
    n.input[1] = wn+"_dq"
    a = n.input[0]
    consumers = sum(a in other.input for other in g.node)
    skip = args.acts == "none" or (args.acts == "single-consumer" and consumers > 1)
    if skip:
        continue
    if a not in done:
        sa = np.array(max(amax[a], 1e-6)/448, npdt); const(a+"_asc", sa)
        z = onnx.TensorProto(); z.name = a+"_azp"; z.data_type = T.FLOAT8E4M3FN; z.raw_data = b"\x00"; g.initializer.append(z)
        new += [helper.make_node("QuantizeLinear", [a, a+"_asc", a+"_azp"], [a+"_aq"], saturate=1),
                helper.make_node("DequantizeLinear", [a+"_aq", a+"_asc", a+"_azp"], [a+"_adq"])]
        done[a] = a+"_adq"
    n.input[0] = done[a]
allnodes = list(g.node)+new; del g.node[:]
avail = {i.name for i in g.input}|{i.name for i in g.initializer}; pend = allnodes
while pend:
    nxt = []
    for n in pend:
        if all(i == "" or i in avail for i in n.input): g.node.append(n); avail.update(n.output)
        else: nxt.append(n)
    assert len(nxt) < len(pend); pend = nxt
for op in m.opset_import:
    if op.domain in ("", "ai.onnx"): op.version = max(op.version, 19)
onnx.checker.check_model(m); onnx.save(m, dst); print("écrit", dst, len(convs), "convolutions quantifiées")
