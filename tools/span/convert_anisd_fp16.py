#!/usr/bin/env python3
"""Convertit un SPAN d'AniSD (Sirosky/Upscale-Hub, release AniSD) du FP32 vers le FP16 que le lecteur attend.

Le lecteur refuse une entrée fp32, et TensorRT 11, fortement typé, suit les types de l'ONNX à la lettre : le
fichier publié (« 2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx », 1,6 Mo, entrée dynamique batch_size×3×width×height)
passerait donc en FP32 sur toute la chaîne. Ce script ne touche pas au graphe (SPAN à 48 canaux, 6 blocs,
DepthToSpace ×2, rien que Conv, Sigmoid, Mul, Add, Sub, Concat) : il passe en FP16 les poids, les constantes,
l'entrée et la sortie, et nomme les axes spatiaux « height » et « width » comme les modèles d'AnimeJaNai (ceux de
la release les ont intervertis).

    python -I tools/span/convert_anisd_fp16.py 2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx sortie.onnx

Pas de PyTorch : onnx et numpy suffisent. Refuse un poids qui dépasse la plage du FP16.
"""

from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

FP16_MAX = 65504.0


def to_fp16(tensor: TensorProto) -> TensorProto:
    values = numpy_helper.to_array(tensor)
    if not np.all(np.abs(values) <= FP16_MAX):
        sys.exit(f"{tensor.name} : valeur hors de la plage du FP16")
    return numpy_helper.from_array(values.astype(np.float16), tensor.name)


def set_io(value: onnx.ValueInfoProto) -> None:
    """FP16, 1×3×height×width."""
    value.type.tensor_type.elem_type = TensorProto.FLOAT16
    shape = value.type.tensor_type.shape
    del shape.dim[:]
    shape.dim.add().dim_value = 1
    shape.dim.add().dim_value = 3
    shape.dim.add().dim_param = "height"
    shape.dim.add().dim_param = "width"


def convert(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    if len(g.input) != 1 or len(g.output) != 1:
        sys.exit("un SPAN a une entrée et une sortie")
    inits = [to_fp16(t) if t.data_type == TensorProto.FLOAT else t for t in g.initializer]
    del g.initializer[:]
    g.initializer.extend(inits)
    for node in g.node:
        if node.op_type == "Cast":
            sys.exit("Cast inattendu : graphe différent de celui d'AniSD")
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                attr.t.CopyFrom(to_fp16(attr.t))
    del g.value_info[:]  # types intermédiaires FP32 : recalculés
    set_io(g.input[0])
    set_io(g.output[0])
    model.producer_name = "convert_anisd_fp16"
    return model


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model = convert(onnx.load(sys.argv[1]))
    onnx.checker.check_model(model)
    onnx.save(model, sys.argv[2])


main()
