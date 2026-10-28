#!/usr/bin/env python3
"""Qualité du SPAN ENTIER en FP8 (8 blocs, noyaux CUTLASS de span_net) contre ONNX Runtime CPU (poids FP16 évalués
en fp32), sur des recadrages 1088x1920 d'images d'anime réelles.

  1. Calibration : les échelles E4M3 par tenseur (sortie de conv_1, puis relu1, relu2 et sortie de chacun des 8 blocs)
     sont mesurées sur D'AUTRES images que celles du test, de deux façons : maximum observé et percentile 99,99
     de |valeur| (maximum, sur les images de calibration, de la statistique de chaque image).
  2. Test : pour chaque image de test, l'image finale 2x de span_net (sous le verrou GPU) est comparée à la sortie
     ORT : PSNR (crête 1, image coupée à [0,1]) et SSIM (fenêtre uniforme 7x7, float64, moyenne sur les canaux).

  python span_net.py --model MODELE.onnx --span-net ./span_net --calib a.ppm b.ppm ... --test s5.ppm s9.ppm s14.ppm
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np, onnx
from onnx import numpy_helper, TensorProto as T
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--span-net", type=Path, required=True)
ap.add_argument("--calib", type=Path, nargs="+", required=True)
ap.add_argument("--test", type=Path, nargs="+", required=True)
ap.add_argument("--workdir", type=Path, required=True)
ap.add_argument("--gpu-prefix", default=str(Path.home() / "ajp-tools/bin/ajp-gpu"))
args = ap.parse_args()
CH, CW = 1088, 1920
OFFS = [(320, 176), (0, 0), (640, 352), (320, 0), (0, 352)]  # positions (x, y) des recadrages successifs
work = args.workdir
work.mkdir(parents=True, exist_ok=True)


def read_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        assert int(f.readline()) == 255
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)


def to_fp32(model):
    g = model.graph
    for i in g.initializer:
        if i.data_type == T.FLOAT16:
            i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float32), i.name))
    for n in g.node:
        for a in n.attribute:
            if a.t.data_type == T.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == T.FLOAT16:
                    a.i = T.FLOAT
    for v in list(g.input) + list(g.output):
        if v.type.tensor_type.elem_type == T.FLOAT16:
            v.type.tensor_type.elem_type = T.FLOAT
    del g.value_info[:]
    return model


def psnr(ref, got, peak=1.0):
    mse = float(np.mean((ref.astype(np.float64) - got) ** 2))
    return 200.0 if mse == 0 else 10 * np.log10(peak * peak / mse)


def box(a, w=7):
    c = np.cumsum(np.cumsum(a, axis=-2, dtype=np.float64), axis=-1)
    c = np.pad(c, [(0, 0)] * (a.ndim - 2) + [(1, 0), (1, 0)])
    return (c[..., w:, w:] - c[..., :-w, w:] - c[..., w:, :-w] + c[..., :-w, :-w]) / (w * w)


def ssim(ref, got, data_range=1.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    out = []
    for c in range(ref.shape[0]):
        x, y = ref[c].astype(np.float64), got[c].astype(np.float64)
        mx, my = box(x), box(y)
        vx, vy, cxy = box(x * x) - mx * mx, box(y * y) - my * my, box(x * y) - mx * my
        out.append((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())
    return float(np.mean(out))


model = to_fp32(onnx.load(str(args.model)))
init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

# tenseurs à calibrer : conv_1, puis (relu1, relu2, sortie) de chaque bloc
CAL = ["/conv_1/conv_1.1/eval_conv/Conv_output_0"]
for k in range(1, 9):
    CAL += [f"/block_{k}/act1/Relu_output_0", f"/block_{k}/act1_1/Relu_output_0", f"/block_{k}/Mul_output_0"]
UNSH = "/conv_1/conv_1.0/Reshape_1_output_0"
for nm in CAL + [UNSH]:
    model.graph.output.append(onnx.helper.make_tensor_value_info(nm, T.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def crop(path, i):
    ox, oy = OFFS[i % len(OFFS)]
    img = read_ppm(path)[oy:oy + CH, ox:ox + CW].astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1))[None]


# ---------------------------------------------------------------- calibration
stats_file = work / "calib_stats.json"
if stats_file.exists():
    stats = json.loads(stats_file.read_text())
else:
    stats = {"max": [0.0] * len(CAL), "p9999": [0.0] * len(CAL)}
    for i, p in enumerate(args.calib):
        inp = crop(p, i)
        res = sess.run(CAL + [UNSH], {"input": inp})
        # le dégroupage supposé (canal = c*4 + a*2 + b) doit retrouver le tenseur du modèle
        if i == 0:
            mine = inp[0].reshape(3, CH // 2, 2, CW // 2, 2).transpose(0, 2, 4, 1, 3).reshape(12, CH // 2, CW // 2)
            print("dégroupage : écart max avec le modèle", float(np.abs(mine - res[-1][0]).max()), flush=True)
        for j, a in enumerate(res[:-1]):
            ab = np.abs(a).ravel()
            stats["max"][j] = max(stats["max"][j], float(ab.max()))
            k = int(0.9999 * ab.size)
            stats["p9999"][j] = max(stats["p9999"][j], float(np.partition(ab, k)[k]))
        print(f"calibration {p.parent.name}/{p.name} #{i} faite", flush=True)
    stats_file.write_text(json.dumps(stats))

# ---------------------------------------------------------------- poids
def kcrs_to_krsc(w):
    return np.ascontiguousarray(w.transpose(0, 2, 3, 1)).astype(np.float32)


w1 = kcrs_to_krsc(init["conv_1.1.eval_conv.weight"])  # (64,3,3,12)
np.pad(w1, [(0, 0), (0, 0), (0, 0), (0, 4)]).tofile(work / "w_c1.f32")
init["conv_1.1.eval_conv.bias"].astype(np.float32).tofile(work / "b_c1.f32")
for k in range(1, 9):
    for i, n in enumerate(("c1_r", "c2_r", "c3_r"), 1):
        kcrs_to_krsc(init[f"block_{k}.{n}.eval_conv.weight"]).tofile(work / f"blk{k}_w{i}.f32")
        init[f"block_{k}.{n}.eval_conv.bias"].astype(np.float32).tofile(work / f"blk{k}_b{i}.f32")
    init[f"block_{k}.att_conv.weight"].reshape(64, 64).astype(np.float32).tofile(work / f"blk{k}_wa.f32")
    init[f"block_{k}.att_conv.bias"].astype(np.float32).tofile(work / f"blk{k}_ba.f32")
init["conv_cat.weight"].reshape(64, 256).astype(np.float32).tofile(work / "w_cat.f32")
init["conv_cat.bias"].astype(np.float32).tofile(work / "b_cat.f32")
kcrs_to_krsc(init["to_pixel.eval_conv.weight"]).tofile(work / "w_pix.f32")
init["to_pixel.eval_conv.bias"].astype(np.float32).tofile(work / "b_pix.f32")

# ---------------------------------------------------------------- test
refs = []
for i, p in enumerate(args.test):
    rf = work / f"ref{i}.npy"
    inp = crop(p, i)
    if rf.exists():
        ref = np.load(rf)
    else:
        ref = np.clip(sess.run(["output"], {"input": inp})[0][0], 0, 1).astype(np.float32)
        np.save(rf, ref)
        print(f"référence ORT {p.name} faite", flush=True)
    refs.append((inp, ref))

print(f"\n{'calibration':<14} {'image':<8} {'PSNR dB':>8} {'SSIM':>8}")
summary = {}
for mode in ("max", "p9999"):
    (work / "scales.f32").write_bytes((np.array(stats[mode], np.float32) / 448.0).tobytes())
    ps_, ss_ = [], []
    for i, (p, (inp, ref)) in enumerate(zip(args.test, refs)):
        inp.astype(np.float32).tofile(work / "input.f32")
        r = subprocess.run([args.gpu_prefix, str(args.span_net), str(work)], capture_output=True, text=True)
        if i == 0 and mode == "max":
            print(r.stdout)
        got = np.clip(np.fromfile(work / "out.f32", np.float32).reshape(3, 4 * (CH // 2), 4 * (CW // 2)), 0, 1)
        ps, ss = psnr(ref, got), ssim(ref, got)
        ps_.append(ps), ss_.append(ss)
        print(f"{mode:<14} {p.name:<8} {ps:8.2f} {ss:8.5f}", flush=True)
    summary[mode] = (float(np.mean(ps_)), float(np.mean(ss_)), float(np.min(ps_)))
for m, (a, b, c) in summary.items():
    print(f"moyenne {m:<8} PSNR {a:.2f} dB (min {c:.2f})  SSIM {b:.5f}")
