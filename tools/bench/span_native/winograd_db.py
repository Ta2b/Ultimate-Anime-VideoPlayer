"""Ce que Winograd coûterait en justesse au SPAN natif en FP16, mesuré sur processeur avant d'écrire un noyau.

Refait le graphe ONNX d'un SPAN en numpy, couche par couche, avec les arrondis des noyaux natifs : activations en
f16 entre les couches, poids en f16, chaque pas de 16 produits sommé exactement puis ajouté à un accumulateur f16
(sommes f16) ou f32. Les convolutions 3×3 des blocs passent au choix par le calcul direct des noyaux d'aujourd'hui
(dans leur ordre : position du filtre, puis 16 canaux), par Winograd F(2×2, 3×3) ou par F(4×4, 3×3) : poids
transformés en f32 puis arrondis en f16, entrée transformée en f32 puis arrondie en f16 (ce que lirait le mma),
sommes sur les canaux par pas de 16, transformée de sortie en f32. Chaque mode est comparé au même graphe en FP32
par ONNX Runtime (tools/bench/qualite/fp32.py) : PSNR (crête 1), plus grand écart. Le mode « f32 » de ce script
doit rendre la référence à plus de 100 dB : c'est le contrôle de l'interprète.

    python3 -I tools/bench/span_native/winograd_db.py MODELE.onnx [--entree DOSSIER_EXPORT | --taille H W]
        [--image IMAGE] [--modes f32,direct,wino23,wino43] [--sommes f16|f32] [--coin H W [--decalage Y X]]

--entree : un dossier de banc.sh EXPORT (entree.f16, dims.txt) ; --coin découpe un coin H × W de l'entrée (le
calcul numpy est lent : 270 × 480 prend quelques minutes par mode). Sans --entree, une mire.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qualite"))
import fp32  # noqa: E402  (tools/bench/qualite)

F16 = np.float16

# Winograd (Lavin et Gray) : Y = Aᵀ [(G g Gᵀ) ⊙ (Bᵀ d B)] A.
BT23 = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], np.float64)
G23 = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64)
AT23 = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], np.float64)
BT43 = np.array([[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0],
                 [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]], np.float64)
G43 = np.array([[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6], [1 / 24, 1 / 12, 1 / 6],
                [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]], np.float64)
AT43 = np.array([[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]], np.float64)
WINO = {"wino23": (BT23, G23, AT23, 2), "wino43": (BT43, G43, AT43, 4)}


def h(x):
    """Arrondi f16, rendu en f32 pour la suite du calcul."""
    return np.asarray(x, np.float32).astype(F16).astype(np.float32)


def chunked_sum(a, b, sums):
    """Σ_k a[..., k] b[k, n] par pas de 16 (k dans l'ordre), chaque pas exact puis ajouté à l'accumulateur."""
    k = a.shape[-1]
    acc = np.zeros(a.shape[:-1] + (b.shape[-1],), np.float32)
    for k0 in range(0, k, 16):
        step = np.matmul(a[..., k0:k0 + 16].astype(np.float64), b[k0:k0 + 16].astype(np.float64))
        acc = acc + step.astype(np.float32)
        if sums == "f16":
            acc = h(acc)
    return acc


def conv_direct(x, w, pad, sums):
    """x (C, H, W) f16, w (O, C, kh, kw) f16 : l'ordre des noyaux, K = position du filtre puis canal."""
    c, hh, ww = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cols = np.stack([xp[:, dy:dy + hh, dx:dx + ww] for dy in range(kh) for dx in range(kw)])  # (T, C, H, W)
    a = cols.transpose(2, 3, 0, 1).reshape(hh * ww, kh * kw * c)
    b = w.transpose(2, 3, 1, 0).reshape(kh * kw * c, o)
    return chunked_sum(a, b, sums).T.reshape(o, hh, ww)


def conv_winograd(x, w, mode, sums):
    bt, g, at, m = WINO[mode]
    a = m + 2
    c, hh, ww = x.shape
    o = w.shape[0]
    th, tw = -(-hh // m), -(-ww // m)
    xp = np.pad(x, ((0, 0), (1, th * m - hh + 1), (1, tw * m - ww + 1)))
    # Tuiles d'entrée a × a, pas m : (C, th, tw, a, a).
    tiles = np.stack([np.stack([xp[:, i:i + th * m:m, j:j + tw * m:m] for j in range(a)], -1) for i in range(a)], -2)
    v = h(np.einsum("pi,ctwij,qj->pqtwc", bt, tiles.astype(np.float64), bt))  # (a, a, th, tw, C)
    u = h(np.einsum("pi,ocij,qj->pqco", g, w.astype(np.float64), g))  # (a, a, C, O)
    mm = np.empty((a, a, th, tw, o), np.float32)
    for p in range(a):
        for q in range(a):
            mm[p, q] = chunked_sum(v[p, q], u[p, q], sums)
    y = np.einsum("ip,pqtwo,jq->otiwj", at, mm.astype(np.float64), at)  # (O, th, m, tw, m)
    return y.reshape(o, th * m, tw * m)[:, :hh, :ww].astype(np.float32)


class Interp:
    """Interprète numpy des opérateurs d'un SPAN ; un opérateur inconnu arrête tout et se nomme."""

    def __init__(self, model, mode, sums):
        from onnx import numpy_helper
        self.g = model.graph
        self.mode, self.sums = mode, sums
        self.vals = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
        self.np = numpy_helper
        self.convs = 0

    def run(self, inp):
        exact = self.mode == "f32"
        self.vals[self.g.input[0].name] = inp.astype(np.float32) if exact else h(inp)
        for node in self.g.node:
            outs = getattr(self, "op_" + node.op_type, None)
            if outs is None:
                raise SystemExit(f"opérateur {node.op_type} non pris en charge ({node.name})")
            res = outs(node, [self.vals.get(n) if n else None for n in node.input],
                       {a.name: self.np_attr(a) for a in node.attribute})
            res = res if isinstance(res, tuple) else (res,)
            for name, val in zip(node.output, res):
                if not exact and isinstance(val, np.ndarray) and val.dtype.kind == "f":
                    val = h(val)
                self.vals[name] = val
        return self.vals[self.g.output[0].name]

    def np_attr(self, a):
        from onnx import helper
        v = helper.get_attribute_value(a)
        return self.np.to_array(v) if hasattr(v, "dims") else v

    def op_Conv(self, n, i, at):
        x, w = i[0], i[1].astype(np.float32)
        b = i[2] if len(i) > 2 and i[2] is not None else None
        assert at.get("group", 1) == 1 and set(at.get("strides", [1, 1])) == {1}, n.name
        pads = at.get("pads", [0, 0, 0, 0])
        assert len(set(pads)) == 1, n.name
        k = w.shape[2]
        outs = []
        for img in x:
            if self.mode == "f32":
                y = conv_direct(img.astype(np.float64), w.astype(np.float64), pads[0], "f64")
            else:
                wq = h(w)
                three = k == 3 and w.shape[0] == w.shape[1]  # les 3×3 carrées des blocs
                if self.mode in WINO and three:
                    y = conv_winograd(img, wq, self.mode, self.sums)
                else:
                    y = conv_direct(img, wq, pads[0], self.sums)
            outs.append(y)
        self.convs += 1
        y = np.stack(outs).astype(np.float32)
        return y + b.reshape(1, -1, 1, 1).astype(np.float32) if b is not None else y

    def op_Relu(self, n, i, at):
        return np.maximum(i[0], 0)

    def op_LeakyRelu(self, n, i, at):
        return np.where(i[0] >= 0, i[0], i[0] * at.get("alpha", 0.01))

    def op_Sigmoid(self, n, i, at):
        return 1 / (1 + np.exp(-i[0]))

    def op_Tanh(self, n, i, at):
        return np.tanh(i[0])

    def op_Add(self, n, i, at):
        return i[0] + i[1]

    def op_Sub(self, n, i, at):
        return i[0] - i[1]

    def op_Mul(self, n, i, at):
        return i[0] * i[1]

    def op_Div(self, n, i, at):
        if i[0].dtype.kind in "iu" and i[1].dtype.kind in "iu":
            return np.trunc(i[0] / i[1]).astype(i[0].dtype)
        return i[0] / i[1]

    def op_Mod(self, n, i, at):
        return np.fmod(i[0], i[1]) if at.get("fmod", 0) else np.mod(i[0], i[1])

    def op_Slice(self, n, i, at):
        x = i[0]
        axes = i[3] if len(i) > 3 and i[3] is not None else range(len(i[1]))
        steps = i[4] if len(i) > 4 and i[4] is not None else [1] * len(i[1])
        sl = [slice(None)] * x.ndim
        for s0, e0, ax, st in zip(i[1], i[2], axes, steps):
            sl[int(ax)] = slice(int(s0), int(max(min(e0, 2**62), -(2**62))), int(st))
        return x[tuple(sl)]

    def op_Pad(self, n, i, at):
        pads = [int(v) for v in i[1]]
        mode = at.get("mode", b"constant")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        r = len(pads) // 2
        width = [(pads[k], pads[k + r]) for k in range(r)]
        if mode == "constant":
            value = float(i[2]) if len(i) > 2 and i[2] is not None and i[2].size else 0.0
            return np.pad(i[0], width, constant_values=value)
        return np.pad(i[0], width, mode={"reflect": "reflect", "edge": "edge"}[mode])

    def op_ConstantOfShape(self, n, i, at):
        v = at.get("value", np.zeros(1, np.float32))
        return np.full([int(d) for d in i[0]], v.reshape(-1)[0], v.dtype)

    def op_Clip(self, n, i, at):
        lo = i[1] if len(i) > 1 and i[1] is not None else at.get("min", -np.inf)
        hi = i[2] if len(i) > 2 and i[2] is not None else at.get("max", np.inf)
        return np.clip(i[0], lo, hi)

    def op_Identity(self, n, i, at):
        return i[0]

    def op_Cast(self, n, i, at):
        return i[0].astype(np.float32) if i[0].dtype.kind == "f" else i[0]

    def op_Constant(self, n, i, at):
        return np.asarray(at["value"])

    def op_Concat(self, n, i, at):
        return np.concatenate(i, axis=at["axis"])

    def op_Shape(self, n, i, at):
        return np.array(i[0].shape, np.int64)

    def op_Gather(self, n, i, at):
        return np.take(i[0], i[1], axis=at.get("axis", 0))

    def op_Unsqueeze(self, n, i, at):
        axes = at["axes"] if "axes" in at else i[1]
        return np.expand_dims(i[0], tuple(int(a) for a in axes))

    def op_Reshape(self, n, i, at):
        shape = [int(s) for s in i[1]]
        shape = [i[0].shape[k] if s == 0 else s for k, s in enumerate(shape)]
        return i[0].reshape(shape)

    def op_Transpose(self, n, i, at):
        return i[0].transpose(at["perm"])

    def op_DepthToSpace(self, n, i, at):
        b, c, hh, ww = i[0].shape
        s = at["blocksize"]
        if at.get("mode", b"DCR") in (b"CRD", "CRD"):
            x = i[0].reshape(b, c // s // s, s, s, hh, ww).transpose(0, 1, 4, 2, 5, 3)
        else:
            x = i[0].reshape(b, s, s, c // s // s, hh, ww).transpose(0, 3, 4, 1, 5, 2)
        return x.reshape(b, c // s // s, hh * s, ww * s)

    def op_SpaceToDepth(self, n, i, at):
        b, c, hh, ww = i[0].shape
        s = at["blocksize"]
        x = i[0].reshape(b, c, hh // s, s, ww // s, s).transpose(0, 3, 5, 1, 2, 4)
        return x.reshape(b, c * s * s, hh // s, ww // s)


def psnr(a, ref):
    d = a.astype(np.float64) - ref.astype(np.float64)
    finite = np.isfinite(d)
    mse = float(np.mean(d[finite] ** 2)) if finite.any() else float("inf")
    worst = float(np.max(np.abs(d[finite]))) if finite.any() else float("nan")
    return (10 * np.log10(1 / mse) if mse > 0 else float("inf")), worst, int(d.size - finite.sum())


def entree(args):
    if args.image:
        from PIL import Image
        x = np.asarray(Image.open(args.image).convert("RGB"), np.float32).transpose(2, 0, 1)[None] / 255
    elif args.entree:
        d = Path(args.entree)
        hh, ww = (int(v) for v in (d / "dims.txt").read_text().split())
        x = np.fromfile(d / "entree.f16", dtype="<f2").astype(np.float32).reshape(1, 3, hh, ww)
    else:
        hh, ww = args.taille
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        x = np.stack([0.5 + 0.45 * np.sin(0.05 * xx * (c + 1) + 0.03 * yy) * np.where((xx // 24 + yy // 24) % 2, 1, 0.5)
                      for c in range(3)])[None]
    if args.coin:
        y0, x0 = args.decalage
        x = x[:, :, y0:y0 + args.coin[0], x0:x0 + args.coin[1]]
    return np.ascontiguousarray(x, dtype=np.float32)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("modele")
    p.add_argument("--entree")
    p.add_argument("--taille", type=int, nargs=2, default=[96, 160])
    p.add_argument("--image", help="une image (PNG, JPEG) au lieu d'un dossier d'EXPORT")
    p.add_argument("--coin", type=int, nargs=2)
    p.add_argument("--decalage", type=int, nargs=2, default=[0, 0], help="origine du coin (ligne, colonne)")
    p.add_argument("--modes", default="f32,direct,wino23,wino43")
    p.add_argument("--sommes", choices=["f16", "f32"], default="f16")
    args = p.parse_args(argv)
    import onnx
    import onnxruntime as ort

    data = fp32.promoted_bytes(args.modele)
    model = onnx.load_from_string(data)
    x = entree(args)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    ref = session.run(None, {session.get_inputs()[0].name: x})[0]
    print(f"{Path(args.modele).name}, entrée {x.shape[3]}×{x.shape[2]}, sommes {args.sommes}, "
          "contre ONNX Runtime FP32")
    for mode in args.modes.split(","):
        start = time.time()
        it = Interp(model, mode, args.sommes)
        out = it.run(x)
        db, worst, bad = psnr(out, ref)
        print(f"{mode:7s} : {db:6.2f} dB, plus grand écart {worst:.5f}, non finies {bad}, {it.convs} convolutions, "
              f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
