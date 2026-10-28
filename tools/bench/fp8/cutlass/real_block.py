#!/usr/bin/env python3
"""Bloc SPAN réel (poids d'AnimeJaNai SPANF3 b8f64) : référence ONNX Runtime CPU contre le bloc FP8 CUTLASS (span_block).

Pour chaque recadrage 1088x1920 d'une image d'anime (le modèle « unshuffle » en tire des tenseurs 64x544x960), on
  1. exécute le modèle complet sur ONNX Runtime CPU (poids FP16 évalués en fp32 : la référence FP16 sans arrondi de calcul)
     et on récupère l'entrée x du bloc testé, ses activations internes et la sortie du bloc ;
  2. calibre des échelles E4M3 par tenseur (max sur tous les recadrages, méthode « max » de quantize_span.py) ;
  3. écrit x, poids (KRSC), biais, échelles dans un dossier et lance span_block (3 noyaux CUTLASS) sous le verrou GPU ;
  4. compare la sortie du bloc (PSNR, SSIM) ; pour le dernier bloc, réinjecte la sortie dans la fin du réseau
     (concat, conv_cat, to_pixel, DepthToSpace) et compare l'IMAGE finale (PSNR, SSIM).

  python real_block.py --model MODELE.onnx --span-block ./span_block --images s5.ppm s9.ppm s14.ppm --block 8
"""
import argparse, os, subprocess, tempfile
from pathlib import Path
import numpy as np, onnx
from onnx import numpy_helper, TensorProto as T

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--span-block", type=Path, required=True)
ap.add_argument("--images", type=Path, nargs="+", required=True)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--gpu-prefix", default=str(Path.home() / "ajp-tools/bin/ajp-gpu"))
ap.add_argument("--workdir", type=Path, default=None)
args = ap.parse_args()
K = args.block
CH, CW = 1088, 1920


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
    """Le modèle FP16 en fp32 : les poids FP16 sont exactement représentables, seul le calcul change."""
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


def psnr(ref, got, peak=None):
    peak = float(np.abs(ref).max()) if peak is None else peak
    mse = float(np.mean((ref.astype(np.float64) - got) ** 2))
    return 200.0 if mse == 0 else 10 * np.log10(peak * peak / mse)


def box(a, w=7):
    """Moyenne glissante w x w par image intégrale (valide : sans bords)."""
    c = np.cumsum(np.cumsum(a, axis=-2, dtype=np.float64), axis=-1)
    c = np.pad(c, [(0, 0)] * (a.ndim - 2) + [(1, 0), (1, 0)])
    s = c[..., w:, w:] - c[..., :-w, w:] - c[..., w:, :-w] + c[..., :-w, :-w]
    return s / (w * w)


def ssim(ref, got, data_range):
    """SSIM (fenêtre uniforme 7x7, comme skimage par défaut) moyen sur les canaux ; ref/got de forme (C,H,W)."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    out = []
    for c in range(ref.shape[0]):
        x, y = ref[c].astype(np.float64), got[c].astype(np.float64)
        mx, my = box(x), box(y)
        vx, vy, cxy = box(x * x) - mx * mx, box(y * y) - my * my, box(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        out.append(s.mean())
    return float(np.mean(out))


import onnxruntime as ort

model = to_fp32(onnx.load(str(args.model)))
init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
X = "/conv_1/conv_1.1/eval_conv/Conv_output_0" if K == 1 else f"/block_{K-1}/Mul_output_0"
NAMES = {"x": X, "r1": f"/block_{K}/act1/Relu_output_0", "r2": f"/block_{K}/act1_1/Relu_output_0",
         "mul": f"/block_{K}/Mul_output_0"}
extra = {}
last = K == 8
if last:
    extra = {"conv1": "/conv_1/conv_1.1/eval_conv/Conv_output_0", "b1mul": "/block_1/Mul_output_0", "image": "output"}
want = {**NAMES, **extra}
for nm in set(want.values()):
    if nm != "output":
        model.graph.output.append(onnx.helper.make_tensor_value_info(nm, T.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

offsets = [(320, 176), (0, 0), (640, 352)]  # (x, y) de chaque recadrage : centre, coin haut-gauche, coin bas-droit
acts = []
for img_path, (ox, oy) in zip(args.images, offsets):
    img = read_ppm(img_path)[oy:oy + CH, ox:ox + CW].astype(np.float32) / 255.0
    inp = np.ascontiguousarray(img.transpose(2, 0, 1))[None]
    names = sorted(set(want.values()))
    res = dict(zip(names, sess.run(names, {"input": inp})))
    acts.append({**{k: res[v] for k, v in want.items()}, "inp": inp})
    print(f"{img_path.name} @({ox},{oy}) : référence ORT CPU faite", flush=True)

amax = lambda key: max(float(np.abs(a[key]).max()) for a in acts)
scales = np.array([amax("x"), amax("r1"), amax("r2"), amax("mul")], np.float32) / 448.0
print("échelles E4M3 (x, relu1, relu2, sortie) :", scales)

pre = f"block_{K}."
wk = lambda n: np.ascontiguousarray(init[pre + n + ".eval_conv.weight"].transpose(0, 2, 3, 1)).astype(np.float32)  # KCRS -> KRSC
bk = lambda n: init[pre + n + ".eval_conv.bias"].astype(np.float32)
work = args.workdir or Path(tempfile.mkdtemp(prefix="span_block_"))
work.mkdir(parents=True, exist_ok=True)

rem_sess = None
if last:
    ex = onnx.utils.Extractor(model)
    rem = ex.extract_model(["input", extra["conv1"], extra["b1mul"], NAMES["mul"], NAMES["r1"]], ["output"])
    # l'ordre des entrées du Concat est (conv_1, block_8 Mul, block_1 Mul, block_8 relu1) : on nomme par tenseur
    rem_sess = ort.InferenceSession(rem.SerializeToString(), providers=["CPUExecutionProvider"])

print(f"\n{'recadrage':<10} {'tenseur':<28} {'PSNR dB':>8} {'SSIM':>7}")
for idx, (img_path, a) in enumerate(zip(args.images, acts)):
    d = work / f"crop{idx}"
    d.mkdir(exist_ok=True)
    nhwc = lambda t: np.ascontiguousarray(t[0].transpose(1, 2, 0)).astype(np.float32)
    nhwc(a["x"]).tofile(d / "x.f32")
    for i, n in enumerate(("c1_r", "c2_r", "c3_r"), 1):
        wk(n).tofile(d / f"w{i}.f32")
        bk(n).tofile(d / f"b{i}.f32")
    init[pre + "att_conv.weight"].reshape(64, 64).astype(np.float32).tofile(d / "wa.f32")
    init[pre + "att_conv.bias"].astype(np.float32).tofile(d / "ba.f32")
    np.array([a["x"].shape[2], a["x"].shape[3]], np.float32).tofile(d / "dims.f32")
    scales.tofile(d / "scales.f32")
    r = subprocess.run([args.gpu_prefix, str(args.span_block), str(d)], capture_output=True, text=True)
    print(r.stdout.strip() if idx == 0 else "", r.stderr.strip())
    H, W = a["x"].shape[2:]
    got = np.fromfile(d / "out.f32", np.float32).reshape(H, W, 64).transpose(2, 0, 1)
    got_r1 = np.fromfile(d / "r1_out.f32", np.float32).reshape(H, W, 64).transpose(2, 0, 1)
    ref, ref_r1 = a["mul"][0], a["r1"][0]
    print(f"{img_path.name:<10} {'sortie du bloc (64x544x960)':<28} {psnr(ref, got):8.2f} {ssim(ref, got, float(np.abs(ref).max())):7.4f}")
    print(f"{img_path.name:<10} {'sortie de c1 (après Relu)':<28} {psnr(ref_r1, got_r1):8.2f} {ssim(ref_r1, got_r1, float(np.abs(ref_r1).max())):7.4f}")
    if last:
        feeds = {"input": a["inp"], extra["conv1"]: a["conv1"], extra["b1mul"]: a["b1mul"], NAMES["mul"]: got[None], NAMES["r1"]: got_r1[None]}
        img_got = rem_sess.run(None, feeds)[0][0]
        img_ref = a["image"][0]
        ref8, got8 = np.clip(img_ref, 0, 1), np.clip(img_got, 0, 1)
        print(f"{img_path.name:<10} {'IMAGE finale 2x (3 canaux)':<28} {psnr(ref8, got8, 1.0):8.2f} {ssim(ref8, got8, 1.0):7.4f}")
