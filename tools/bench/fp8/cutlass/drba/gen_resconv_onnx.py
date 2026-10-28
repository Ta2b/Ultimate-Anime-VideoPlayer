#!/usr/bin/env python3
"""Génère les ONNX FP16 de référence TensorRT : la suite de 8 ResConv d'un bloc DistilDRBA
(y = LeakyRelu(0,2)(Conv3x3(x) * beta + x)) à la forme réelle de chaque bloc, plus la conv de tête.

    python gen_resconv_onnx.py SORTIE_DIR
Écrit resconv_C{192,128,96}.onnx. Les poids sont aléatoires (seul le temps compte ici).
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# (canaux, hauteur, largeur) des trois blocs à 1920x1088 : conv0 réduit l'image de 1/64, 1/32, 1/16.
BLOCS = {192: (17, 30), 128: (34, 60), 96: (68, 120)}


def chaine(c: int, h: int, w: int, n: int = 8) -> onnx.ModelProto:
    rng = np.random.default_rng(1)
    nodes, inits = [], []
    prev = "x"
    for i in range(n):
        wt = rng.normal(0, np.sqrt(2 / (9 * c)) * 0.5, (c, c, 3, 3)).astype(np.float16)
        b = rng.uniform(-0.05, 0.05, (c,)).astype(np.float16)
        beta = rng.uniform(0.8, 1.2, (1, c, 1, 1)).astype(np.float16)
        inits += [numpy_helper.from_array(wt, f"w{i}"), numpy_helper.from_array(b, f"b{i}"),
                  numpy_helper.from_array(beta, f"beta{i}")]
        nodes += [
            helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Mul", [f"c{i}", f"beta{i}"], [f"m{i}"]),
            helper.make_node("Add", [f"m{i}", prev], [f"a{i}"]),
            helper.make_node("LeakyRelu", [f"a{i}"], [f"y{i}"], alpha=0.2),
        ]
        prev = f"y{i}"
    g = helper.make_graph(
        nodes, f"resconv{c}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, c, h, w])],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT16, [1, c, h, w])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for c, (h, w) in BLOCS.items():
        onnx.save(chaine(c, h, w), str(out / f"resconv_C{c}.onnx"))
