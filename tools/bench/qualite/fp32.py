"""La référence FP32 d'un graphe FP16 livré : le même graphe, où tout tenseur float16 devient float32.

Les poids gardent leur valeur (un float16 s'écrit exactement en float32) : la référence a donc les poids du fichier
livré et tout le calcul en FP32. Elle mesure ce que coûte le calcul en FP16 (activations, sommes, grilles), pas
l'arrondi des poids fait à la conversion ; pour celui-ci, donner le graphe d'origine au banc (--origine).
"""
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def _tensor(t):
    if t.data_type == F16:
        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))


def _types(values):
    for v in values:
        if v.type.tensor_type.elem_type == F16:
            v.type.tensor_type.elem_type = F32


def _graph(g):
    for t in g.initializer:
        _tensor(t)
    _types(g.input)
    _types(g.output)
    _types(g.value_info)
    for node in g.node:
        for a in node.attribute:
            if node.op_type == "Cast" and a.name == "to" and a.i == F16:
                a.i = F32
            if a.type == onnx.AttributeProto.TENSOR:
                _tensor(a.t)
            elif a.type == onnx.AttributeProto.TENSORS:
                for t in a.tensors:
                    _tensor(t)
            elif a.type == onnx.AttributeProto.GRAPH:
                _graph(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sub in a.graphs:
                    _graph(sub)


def promote(model):
    """Une copie du modèle, entièrement en float32."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    _graph(out.graph)
    return out


def has_fp16(model):
    g = model.graph
    return any(t.data_type == F16 for t in g.initializer) or any(
        v.type.tensor_type.elem_type == F16 for v in list(g.input) + list(g.output))


def promoted_bytes(path):
    """Le graphe FP32 de ce fichier, sérialisé pour ONNX Runtime."""
    return promote(onnx.load(path)).SerializeToString()
