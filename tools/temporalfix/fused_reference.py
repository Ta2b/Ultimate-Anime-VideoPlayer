#!/usr/bin/env python3
# Référence numpy de TemporalFix fusionné (docs/temporalfix.md, « Fusionné ») : la même décomposition que les
# noyaux de src/gpu/cuda/backend/tfix_*_kernels.cu, les convolutions passées par onnxruntime sur les morceaux
# que coupe prepare/tfix_split.cpp, comparée au graphe entier. Usage : fused_reference.py s2.onnx 64 96
# Attendu : environ 78 dB (s1, s2, s3), le pixel (0, 0) mis à part (terme de coin laissé de côté).
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def piece(m, ins, outs, in_dims):
    init = {i.name: i for i in m.graph.initializer}
    prod = {}
    for n in m.graph.node:
        for o in n.output: prod[o] = n
    keep = []; seen = set(); stack = list(outs); used_init = set()
    while stack:
        t = stack.pop()
        if t in seen or t in ins or t == '': continue
        seen.add(t)
        if t in init: used_init.add(t); continue
        n = prod[t]
        if id(n) not in [id(k) for k in keep]: keep.append(n)
        stack.extend(n.input)
    order = [n for n in m.graph.node if any(n is k for k in keep)]
    g = helper.make_graph(order, 'piece',
        [helper.make_tensor_value_info(i, TensorProto.FLOAT16, d) for i, d in zip(ins, in_dims)],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT16, None) for o in outs],
        [init[i] for i in used_init])
    pm = helper.make_model(g, opset_imports=m.opset_import)
    pm.ir_version = m.ir_version
    return pm

def names():
    return {
      'enc': (['/Reshape_1_output_0'], ['/encode/Resize_output_0']),
      'b0': (['/aligner/Concat_57_output_0'], ['/aligner/blocks.0/out_conv/Conv_output_0']),
      'b1': (['/aligner/blocks.1/Concat_output_0'], ['/aligner/blocks.1/out_conv/Conv_output_0']),
      'b2': (['/aligner/blocks.2/Concat_output_0'], ['/aligner/blocks.2/out_conv/Conv_output_0']),
      'conf': (['/conf_head/Concat_output_0'], ['/conf_head/net/net.2/Conv_output_0']),
    }

T = np.array([-1.0, -0.66650390625, -0.333251953125, 0.333251953125, 0.66650390625, 1.0], np.float32)
J = [0, 1, 2, 4, 5, 6]
BW = np.array([1, 4, 6, 4, 1], np.float32) / 16

def ceil_div(a, b): return -(-a // b)

def taps(n_in, n_out):
    # Resize linéaire half_pixel par tailles (onnxruntime) : position bornée à [0, n_in - 1].
    i = np.arange(n_out, dtype=np.float32)
    x = (i + 0.5) * (np.float32(n_in) / np.float32(n_out)) - 0.5
    x = np.clip(x, 0, n_in - 1)
    x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, n_in - 1)
    return x0, x1, (x - x0).astype(np.float32)

def resize(a, h, w):
    y0, y1, wy = taps(a.shape[-2], h); x0, x1, wx = taps(a.shape[-1], w)
    r = a[..., y0, :] * (1 - wy)[:, None] + a[..., y1, :] * wy[:, None]
    return r[..., x0] * (1 - wx) + r[..., x1] * wx

def blur(a):
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(2, 2), (2, 2)])
    h, w = a.shape[-2:]
    r = sum(BW[i] * p[..., i:i + h, :] for i in range(5))
    return sum(BW[i] * r[..., :, i:i + w] for i in range(5))

def pyr(a, s):
    h, w = a.shape[-2:]
    return resize(blur(a), ceil_div(h, s), ceil_div(w, s))

def warp(img, fx, fy):
    # GridSample bilinéaire, zéros dehors, align_corners 0 : point (x + fx, y + fy) en pixels.
    c, h, w = img.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    px = xx + fx; py = yy + fy
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    ax = px - x0; ay = py - y0
    out = np.zeros_like(img); valid = np.zeros((h, w), np.float32)
    for dy, wy in ((0, 1 - ay), (1, ay)):
        for dx, wx in ((0, 1 - ax), (1, ax)):
            yi = y0 + dy; xi = x0 + dx
            ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            wgt = np.where(ok, wy * wx, 0)
            out += img[:, np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)] * wgt
            valid += wgt
    return out, valid

def sig(x): return 1 / (1 + np.exp(-x))
def f16(a): return a.astype(np.float16).astype(np.float32)

class Net:
    def __init__(self, path):
        self.m = onnx.load(path); self.s = {}
    def run(self, key, x):
        if (key, x.shape) not in self.s:
            ins, outs = names()[key]
            pm = piece(self.m, ins, outs, [list(x.shape)])
            self.s[(key, x.shape)] = ort.InferenceSession(pm.SerializeToString(), providers=['CPUExecutionProvider'])
        sess = self.s[(key, x.shape)]
        return sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float16)})[0].astype(np.float32)

def pack(net, image):
    feat = net.run('enc', image[None])[0]
    both = np.concatenate([image, feat])
    return {'img': image, 'feat': feat, 8: f16(pyr(both, 8)), 4: f16(pyr(both, 4)), 2: f16(pyr(both, 2))}

def window(net, packs):
    c = packs[3]; img = c['img']; h, w = img.shape[1:]
    # Niveau 8 : X0.
    X0 = np.stack([np.concatenate([c[8][:3], packs[j][8][:3], c[8][3:], packs[j][8][3:],
                                   np.full((1,) + c[8].shape[1:], T[k])]) for k, j in enumerate(J)])
    Y = net.run('b0', f16(X0)); flow_prev = None; X = X0
    for lvl, (src, dst) in enumerate(((8, 4), (4, 2))):
        Ls = packs[3][src].shape[1:]; Ld = packs[3][dst].shape[1:]
        Z = resize(Y, *Ls)  # le Resize du bloc
        flow = Z[:, 0:2] if flow_prev is None else flow_prev + Z[:, 0:2]
        fin = 2 * resize(flow, *Ld)
        m = resize(Z[:, 2:3], *Ld); hid = resize(Z[:, 3:11], *Ld)
        Xn = []
        for k, j in enumerate(J):
            wv, _ = warp(packs[j][dst], fin[k, 0], fin[k, 1])
            Xn.append(np.concatenate([c[dst][:3], wv[:3], c[dst][3:], wv[3:], np.full((1,) + Ld, T[k]), m[k], hid[k], fin[k]]))
        X = f16(np.stack(Xn)); flow_prev = X[:, 24:26]
        Y = net.run('b1' if lvl == 0 else 'b2', X)
    L2 = packs[3][2].shape[1:]
    Z = resize(Y, *L2)
    flow2 = flow_prev + Z[:, 0:2]
    flowF = f16(2 * resize(flow2, h, w)); mF = resize(Z[:, 2:3], h, w)
    Wk = []; Vk = []
    for k, j in enumerate(J):
        wv, v = warp(packs[j]['img'], flowF[k, 0], flowF[k, 1]); Wk.append(wv); Vk.append(v)
    Wk = np.stack(Wk); Vk = np.stack(Vk)
    # Tête de confiance, à demi-résolution.
    pW = pyr(f16(Wk), 2); fh = 0.5 * pyr(flowF, 2); mh = pyr(mF, 2)
    Xc = []
    for k, j in enumerate(J):
        wf, _ = warp(packs[j][2][3:], fh[k, 0], fh[k, 1])
        cimg = c[2][:3]; cf = c[2][3:]
        Xc.append(np.concatenate([cimg, pW[k], np.abs(pW[k] - cimg), cf, wf, np.abs(wf - cf), fh[k],
                                  np.sqrt((fh[k] ** 2).sum(0, keepdims=True)), np.full((1,) + L2, T[k])]))
    conf = net.run('conf', f16(np.stack(Xc))) + mh
    ck = sig(resize(conf, h, w))[:, 0]
    wk = ck * Vk
    sk = sig((wk - 0.60009765625) * 12); ak = wk * sk
    num = img + (ak[:, None] * Wk).sum(0); den = np.maximum(ak.sum(0) + 1, 1.0132789611816406e-06)
    g = sig((sk.sum(0) - 0.5) * 4)
    return img + g * (num / den - img)

def psnr(a, b): return 10 * np.log10(1 / np.mean((a.astype(np.float64) - b) ** 2))

if __name__ == '__main__':
    path, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(1)
    # Images lisses déplacées : un vrai mouvement plutôt que du bruit.
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    frames = []
    for t in range(7):
        base = np.stack([np.sin((xx + 1.5 * t) / 7 + c) * np.cos((yy - t) / 5 + 2 * c) for c in range(3)]) * 0.4 + 0.5
        frames.append(f16(base + rng.normal(0, 0.03, base.shape)).astype(np.float32))
    full = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    x = np.concatenate(frames)[None].astype(np.float16)
    want = full.run(None, {'input': x})[0][0].astype(np.float32)
    net = Net(path)
    got = window(net, [pack(net, f) for f in frames])
    mask = np.ones_like(want, bool); mask[:, 0, 0] = False  # le terme de coin, laissé de côté
    print(f'{path} {H}x{W}: psnr {psnr(got[mask], want[mask]):.1f} dB, max {np.abs(got - want)[mask].max():.4f}, coin {np.abs(got - want)[:, 0, 0]}')
