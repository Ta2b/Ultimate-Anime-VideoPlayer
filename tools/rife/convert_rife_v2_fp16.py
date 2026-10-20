#!/usr/bin/env python3
"""Convertit les RIFE v2 de vs-mlrt (dossier rife_v2, entrée à 7 plans) en FP16 pour le lecteur.

Le lecteur refuse une entrée fp32, et TensorRT 11, fortement typé, suit les types de l'ONNX à la
lettre. Le v2 calcule lui-même la grille d'échantillonnage à partir de la forme de l'image
(Range, Cast, Reciprocal, Mul et Sub sur les coordonnées), puis l'ajoute au flot avant chaque
GridSample. En fp16, la grille, normalisée dans [-1, 1], perd 1 à 2 pixels de précision : vs-mlrt
force ces couches en fp32 sur TensorRT. Ici, restent en fp32 :
  - tout ce qui ne dépend pas des valeurs de l'image (le calcul des coordonnées, Shape coupe la
    dépendance) ;
  - la chaîne Transpose(Add(grille, Mul(flot, pas))) qui mène à chaque GridSample, et GridSample.
Le reste passe en fp16, l'entrée et la sortie comprises ; un Cast est posé à chaque frontière.
(convert_rife_fp16.py d'animejanai-inference-rtx, écrit pour le v1, laisse les Cast vers fp32 du
v2 et la constante de /Mul_1 de part et d'autre : le graphe produit est mal typé.)

    python -I tools/rife/convert_rife_v2_fp16.py rife_v2/ sortie/ [rife_v4.25.onnx ...]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

FP16_MAX = 65504.0
GRID_OPS = ("Transpose", "Add", "Mul")  # flot + grille, de GridSample vers l'amont
SHAPE_ONLY = ("Shape", "Size")  # leur sortie ne dépend que de la forme
FIXED_FLOAT_INPUTS = {"Resize": {1, 2}}  # roi et scales : float32 quel que soit T
F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16


def data_dependent(graph) -> set[str]:
    """Les tenseurs qui dépendent des valeurs de l'entrée (Shape et Size coupent la chaîne)."""
    tainted = {i.name for i in graph.input}
    for node in graph.node:  # l'ONNX exporté est trié dans l'ordre topologique
        if node.op_type not in SHAPE_ONLY and any(i in tainted for i in node.input):
            tainted.update(node.output)
    return tainted


def fp32_nodes(graph, tainted: set[str]) -> set[str]:
    """Le calcul des coordonnées, et la chaîne qui ajoute le flot à la grille avant GridSample."""
    prod = {o: n for n in graph.node for o in n.output}
    keep = {n.name for n in graph.node if not any(o in tainted for o in n.output)}
    for node in graph.node:
        if node.op_type != "GridSample":
            continue
        keep.add(node.name)
        stack = [node.input[1]]
        while stack:
            p = prod.get(stack.pop())
            if p is None or p.name in keep or p.op_type not in GRID_OPS:
                continue
            keep.add(p.name)
            stack.extend(p.input)
    return keep


def to_fp16(t: TensorProto) -> TensorProto:
    arr = np.clip(numpy_helper.to_array(t).astype(np.float32), -FP16_MAX, FP16_MAX)
    return numpy_helper.from_array(arr.astype(np.float16), t.name)


def convert_model(model: onnx.ModelProto) -> onnx.ModelProto:
    types = {v.name: v.type.tensor_type.elem_type for v in shape_inference.infer_shapes(model).graph.value_info}
    g = model.graph
    del g.value_info[:]
    if not any(n.op_type == "GridSample" for n in g.node):
        raise RuntimeError("aucun GridSample : pas un graphe RIFE")
    if len(g.input) != 1 or g.input[0].type.tensor_type.shape.dim[1].dim_value != 7:
        raise RuntimeError("entrée attendue : [N, 7, H, W] (RIFE v2)")
    tainted = data_dependent(g)
    keep = fp32_nodes(g, tainted)
    for t in g.initializer:
        types[t.name] = t.data_type
    for i in list(g.input) + list(g.output):
        types[i.name] = i.type.tensor_type.elem_type

    consumers = defaultdict(list)
    for n in g.node:
        for k, t in enumerate(n.input):
            consumers[t].append((n, k))

    def wants_fp32(node, k) -> bool:
        return node.name in keep or k in FIXED_FLOAT_INPUTS.get(node.op_type, ())

    # Le monde de chaque tenseur flottant : fp32 s'il sort d'un nœud gardé, ou s'il s'agit d'une
    # constante qu'un nœud gardé lit ; fp16 sinon. Les Cast vers float du monde fp16 visent fp16.
    world: dict[str, int] = {g.input[0].name: F16}
    for t in g.initializer:
        if t.data_type == F32:
            world[t.name] = F32 if any(wants_fp32(n, k) for n, k in consumers[t.name]) else F16
            if world[t.name] == F16:
                t.CopyFrom(to_fp16(t))
    for n in g.node:
        for o in n.output:
            if types.get(o) != F32:
                continue
            if n.op_type == "Constant":
                world[o] = F32 if any(wants_fp32(c, k) for c, k in consumers[o]) else F16
                value = next(a for a in n.attribute if a.name == "value")
                if world[o] == F16:
                    value.t.CopyFrom(to_fp16(value.t))
            else:
                world[o] = F32 if n.name in keep else F16
            if n.op_type == "Cast" and world[o] == F16:
                next(a for a in n.attribute if a.name == "to").i = F16

    casts: dict[tuple[str, int], str] = {}
    new_nodes = []

    def cast(tensor: str, to: int) -> str:
        if (tensor, to) not in casts:
            name = f"{tensor}/cast{16 if to == F16 else 32}"
            new_nodes.append(helper.make_node("Cast", [tensor], [name], name=name, to=to))
            casts[(tensor, to)] = name
        return casts[(tensor, to)]

    for n in g.node:
        if n.op_type in ("Constant", "Cast"):
            continue
        for k, t in enumerate(n.input):
            if t not in world:
                continue
            want = F32 if wants_fp32(n, k) else F16
            if world[t] != want:
                n.input[k] = cast(t, want)

    out = g.output[0]
    if world.get(out.name) == F32:
        for n in g.node:
            for k, v in enumerate(n.output):
                if v == out.name:
                    n.output[k] = out.name + "_f32"
        new_nodes.append(helper.make_node("Cast", [out.name + "_f32"], [out.name], name="ajp/cast_output", to=F16))
    g.input[0].type.tensor_type.elem_type = F16
    out.type.tensor_type.elem_type = F16
    g.node.extend(new_nodes)
    order = topo_sort(g)
    del g.node[:]
    g.node.extend(order)
    return model


def topo_sort(g):
    prod = {o: n.name for n in g.node for o in n.output}
    nodes = {n.name: n for n in g.node}
    state: dict[str, int] = {}
    out = []

    def visit(name):
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise RuntimeError(f"cycle à {name}")
        state[name] = 1
        for t in nodes[name].input:
            if t in prod:
                visit(prod[t])
        state[name] = 2
        out.append(nodes[name])

    for n in list(g.node):
        visit(n.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="dossier des RIFE v2 en fp32")
    parser.add_argument("destination", type=Path, help="dossier des RIFE v2 en FP16")
    parser.add_argument("models", nargs="*", help="noms des modèles (défaut : tous les .onnx)")
    args = parser.parse_args()
    names = args.models or sorted(p.name for p in args.source.glob("*.onnx"))
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        model = convert_model(onnx.load(str(args.source / name)))
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, str(args.destination / name))
        before, after = (args.source / name).stat().st_size, (args.destination / name).stat().st_size
        print(f"{name} : {before / 1e6:.1f} Mo → {after / 1e6:.1f} Mo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
