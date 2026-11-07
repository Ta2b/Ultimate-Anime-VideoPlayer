#!/usr/bin/env python3
"""Un modèle ×2 qui n'agrandit que la luma (ArtCNN, P6.7) en un modèle RGB que le lecteur joue comme SPAN.

  python3 -I tools/upscale/wrap_luma_rgb.py ArtCNN_C4F32.onnx 2x_ArtCNN_C4F32_rgb_fp16.onnx

Entrée et sortie comme les SPAN d'AnimeJaNai : [1, 3, H, W] en FP16, R'G'B' entre 0 et 1. Dedans, en FP32 :
la luma Y' (BT.709) passe par le réseau ; la différence R'G'B' − Y' (la couleur, sans détail fin) est agrandie
en bilinéaire, puis ajoutée à la luma agrandie. Le réseau d'origine est recopié tel quel, nœuds et poids
préfixés « luma/ ». Demande le paquet onnx.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PREFIX = "luma/"
LUMA_709 = (0.2126, 0.7152, 0.0722)


def wrap(model: onnx.ModelProto) -> onnx.ModelProto:
    inner = model.graph
    if len(inner.input) != 1 or len(inner.output) != 1:
        raise SystemExit("le réseau doit avoir une entrée et une sortie")
    dims = inner.input[0].type.tensor_type.shape.dim
    if len(dims) != 4 or dims[1].dim_value != 1:
        raise SystemExit("le réseau doit prendre une seule composante, [1, 1, H, W]")
    rename = {inner.input[0].name: "y"}

    def name(n: str) -> str:
        return rename.get(n, PREFIX + n) if n else n

    nodes = [
        helper.make_node("Cast", ["input"], ["rgb"], to=TensorProto.FLOAT),
        helper.make_node("Conv", ["rgb", "luma_weights"], ["y"]),
        helper.make_node("Concat", ["y", "y", "y"], ["y3"], axis=1),
        helper.make_node("Sub", ["rgb", "y3"], ["chroma"]),
        helper.make_node("Resize", ["chroma", "", "scales"], ["chroma_up"], mode="linear",
                         coordinate_transformation_mode="half_pixel"),
    ]
    for node in inner.node:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        copy.name = PREFIX + node.name if node.name else ""
        copy.input[:] = [name(n) for n in node.input]
        copy.output[:] = [name(n) for n in node.output]
        nodes.append(copy)
    y_up = name(inner.output[0].name)
    nodes += [
        helper.make_node("Concat", [y_up, y_up, y_up], ["y_up3"], axis=1),
        helper.make_node("Add", ["chroma_up", "y_up3"], ["rgb_up"]),
        helper.make_node("Clip", ["rgb_up", "zero", "one"], ["rgb_clipped"]),
        helper.make_node("Cast", ["rgb_clipped"], ["output"], to=TensorProto.FLOAT16),
    ]
    initializers = [
        numpy_helper.from_array(np.array(LUMA_709, dtype=np.float32).reshape(1, 3, 1, 1), "luma_weights"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
        numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(1, dtype=np.float32), "one"),
    ]
    for init in inner.initializer:
        copy = onnx.TensorProto()
        copy.CopyFrom(init)
        copy.name = name(init.name)
        initializers.append(copy)
    graph = helper.make_graph(
        nodes, "luma_rgb",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 3, "height", "width"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, "height2", "width2"])],
        initializers)
    opset = max(13, next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")))
    out = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                            producer_name="ajp wrap_luma_rgb")
    out.ir_version = model.ir_version
    onnx.checker.check_model(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    onnx.save(wrap(onnx.load(sys.argv[1])), sys.argv[2])
    print(f"écrit : {sys.argv[2]}")
