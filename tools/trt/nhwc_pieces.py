#!/usr/bin/env python3
"""Les blocs d'un RIFE fusionné (morceaux « -blocN.onnx » de prepare/rife_split.h) réécrits pour des E/S NHWC
alignées sur 8 canaux : l'essai des E/S hwc8 (docs/tests-pc.md, E/S NHWC) sans --inputIOFormats, que
trtexec 11 n'a plus (prepare/trtexec_command.h, sanitize_trt11).

  python3 -I tools/trt/nhwc_pieces.py DOSSIER_DU_MANIFESTE [--verifier]

Pour chaque bloc, écrit « <bloc>-nhwc.onnx » à côté :
  - l'entrée devient 1 × H × W × Cp (Cp : C arrondi au multiple de 8 supérieur, 15 → 16, 28 → 32), suivie
    d'une Transpose vers NCHW ; la première convolution prend Cp canaux, ses poids complétés de zéros ;
  - la ConvTranspose de sortie rend 56 canaux au lieu de 52 (poids et biais complétés de zéros), suivie d'une
    Transpose vers 1 × h × w × 56.
En mémoire, un tenseur NHWC de Cp canaux est rangé comme le format HWC8 de TensorRT : si le constructeur
choisit HWC8 pour la première convolution et la dernière, les deux Transpose ne coûtent plus rien et les
« Reformatting CopyNode » d'E/S disparaissent du profil. C'est ce que la mesure dit (trtexec --dumpProfile,
GPU Compute Time contre le morceau d'origine) ; les noyaux de colle n'écriront ce format qu'après.

--verifier : passe une entrée aléatoire par l'original et par la réécriture sous ONNX Runtime (processeur) et
dit l'écart maximal (attendu : celui des arrondis FP16 d'un ordre de calcul différent, au plus quelques
millièmes). Il faut onnx, numpy et, pour --verifier, onnxruntime.
"""
import argparse
import glob
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

OUT_CHANNELS = 52


def padded(c):
    return (c + 7) // 8 * 8


def initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    raise SystemExit(f"{name} : poids introuvable")


def replace(graph, name, array):
    init = initializer(graph, name)
    init.CopyFrom(numpy_helper.from_array(array, name))


def rewrite(model):
    g = model.graph
    if len(g.input) != 1 or len(g.output) != 1:
        raise SystemExit("un bloc a une entrée et une sortie")
    x, y = g.input[0], g.output[0]
    dims = [d.dim_value for d in x.type.tensor_type.shape.dim]
    _, c, h, w = dims
    cp = padded(c)
    first = [n for n in g.node if x.name in n.input]
    last = [n for n in g.node if y.name in n.output]
    if len(first) != 1 or first[0].op_type != "Conv" or len(last) != 1 or last[0].op_type != "ConvTranspose":
        raise SystemExit("bloc inattendu : une Conv en entrée, une ConvTranspose en sortie")
    conv, deconv = first[0], last[0]
    weight = numpy_helper.to_array(initializer(g, conv.input[1]))
    replace(g, conv.input[1], np.concatenate(
        [weight, np.zeros((weight.shape[0], cp - c) + weight.shape[2:], weight.dtype)], axis=1))
    # ConvTranspose : poids [entrées, sorties / groupe, kh, kw]
    weight = numpy_helper.to_array(initializer(g, deconv.input[1]))
    if weight.shape[1] != OUT_CHANNELS:
        raise SystemExit(f"ConvTranspose à {weight.shape[1]} sorties, {OUT_CHANNELS} attendues")
    out = padded(OUT_CHANNELS)
    replace(g, deconv.input[1], np.concatenate(
        [weight, np.zeros((weight.shape[0], out - OUT_CHANNELS) + weight.shape[2:], weight.dtype)], axis=1))
    bias = numpy_helper.to_array(initializer(g, deconv.input[2]))
    replace(g, deconv.input[2], np.concatenate([bias, np.zeros(out - OUT_CHANNELS, bias.dtype)]))

    elem = x.type.tensor_type.elem_type
    nhwc_in = helper.make_tensor_value_info("x_nhwc", elem, [1, h, w, cp])
    nchw = x.name + "_nchw"
    conv.input[0] = nchw
    to_nchw = helper.make_node("Transpose", ["x_nhwc"], [nchw], name="entree_nhwc", perm=[0, 3, 1, 2])
    inner = y.name + "_nchw"
    deconv.output[0] = inner
    to_nhwc = helper.make_node("Transpose", [inner], ["y_nhwc"], name="sortie_nhwc", perm=[0, 2, 3, 1])
    nodes = [to_nchw] + list(g.node) + [to_nhwc]
    del g.node[:]
    g.node.extend(nodes)
    out_dims = [1, h // 2, w // 2, out] if h else []
    nhwc_out = helper.make_tensor_value_info("y_nhwc", elem, out_dims)
    del g.input[:]
    g.input.append(nhwc_in)
    del g.output[:]
    g.output.append(nhwc_out)
    return c, cp


def verify(original, rewritten, c, cp):
    import onnxruntime as ort
    a = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(rewritten, providers=["CPUExecutionProvider"])
    shape = a.get_inputs()[0].shape
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=shape).astype(np.float16)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    xn = np.zeros((1, shape[2], shape[3], cp), np.float16)
    xn[..., :c] = x[0].transpose(1, 2, 0)
    yb = b.run(None, {"x_nhwc": xn})[0]
    real = yb[..., :OUT_CHANNELS].transpose(0, 3, 1, 2).astype(np.float32)
    worst = float(np.abs(real - ya.astype(np.float32)).max())
    spill = float(np.abs(yb[..., OUT_CHANNELS:]).max())
    return worst, spill


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dossier", help="le dossier du manifeste .rifefused")
    p.add_argument("--verifier", action="store_true")
    args = p.parse_args()
    pieces = sorted(f for f in glob.glob(os.path.join(args.dossier, "*-bloc[0-9].onnx")))
    if not pieces:
        raise SystemExit("aucun « -blocN.onnx » dans ce dossier")
    for path in pieces:
        model = onnx.load(path)
        c, cp = rewrite(model)
        target = path[:-len(".onnx")] + "-nhwc.onnx"
        onnx.save(model, target)
        line = f"{os.path.basename(target)} : entrée {c} → {cp} canaux, sortie 52 → 56"
        if args.verifier:
            worst, spill = verify(path, target, c, cp)
            line += f", écart maximal {worst:.2e}, canaux ajoutés {spill:.0e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
