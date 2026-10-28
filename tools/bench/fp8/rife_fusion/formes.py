#!/usr/bin/env python3
"""Affiche formes des tenseurs aux frontieres et constantes (Split, Reshape, Mul, Resize) de RIFE v4.25."""
import sys

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

m = onnx.load(sys.argv[1])
m = shape_inference.infer_shapes(m)
g = m.graph
shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
for n in g.node:
    if n.op_type == "Constant":
        inits[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
keep = ("Split", "Reshape", "Concat", "Resize", "DepthToSpace", "ConvTranspose", "Transpose", "GridSample", "Sigmoid", "Sub")
for n in g.node:
    if n.op_type in keep or (n.op_type in ("Mul", "Add") and any(i in inits and inits[i].size < 8 for i in n.input) and "convblock" not in n.name):
        ins = []
        for i in n.input:
            if i in inits and inits[i].size < 12:
                ins.append(f"{i}={inits[i].tolist()}")
            else:
                ins.append(f"{i}{shapes.get(i, '')}")
        print(n.op_type, n.name, "|", "; ".join(ins), "->", [(o, shapes.get(o)) for o in n.output])
