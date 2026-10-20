#!/usr/bin/env python3
"""Convertit les RIFE de vs-mlrt en FP16 pour le lecteur : v1 (dossier rife, 11 plans) et v2
(dossier rife_v2, 7 plans).

Le lecteur refuse une entrée fp32, et TensorRT 11, fortement typé, suit les types de l'ONNX à la
lettre. Restent en fp32 :
  - tout ce qui ne dépend pas des valeurs de l'image : le calcul des coordonnées du v2 (Range,
    Cast, Reciprocal, Mul, Sub sur la forme ; Shape coupe la dépendance). En fp16, la grille
    normalisée dans [-1, 1] perd 1 à 2 pixels de précision ; vs-mlrt force ces couches en fp32 ;
  - la chaîne Transpose(Add(grille, Mul(flot, pas))) qui mène à chaque GridSample, et GridSample ;
  - les Resize des blocs. Sans eux, TensorRT 11.3 génère pour 4.25 lite (v1 et v2) des noyaux
    fusionnés FP16 qui sortent des NaN, carrés noirs à l'écran : 32 paires sur 50 en mouvement de
    Re:Zero, alors qu'aucune activation ne dépasse 85 ni aucune accumulation de Conv 56 (pas de
    débordement), et que TensorRT fp32 et ONNX Runtime FP16 sont propres. Les Resize en fp32 :
    0 paire sur 50.
Le reste passe en fp16, l'entrée et la sortie comprises ; un Cast est posé à chaque frontière.
(convert_rife_fp16.py d'animejanai-inference-rtx, écrit pour le v1, laisse le v2 mal typé à
/Mul_1 et les Resize en fp16.)

    python -I tools/rife/convert_rife_fp16.py rife_v2/ sortie/ [rife_v4.25.onnx ...]
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
FP32_OPS = ("GridSample", "Resize")  # voir plus haut : grille et NaN de TensorRT
F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16


def data_dependent(graph) -> set[str]:
    """Les tenseurs qui dépendent des valeurs de l'entrée (Shape et Size coupent la chaîne)."""
    tainted = {i.name for i in graph.input}
    for node in graph.node:  # l'ONNX exporté est trié dans l'ordre topologique
        if node.op_type not in SHAPE_ONLY and any(i in tainted for i in node.input):
            tainted.update(node.output)
    return tainted


def fp32_nodes(graph, tainted: set[str]) -> set[str]:
    """Le calcul des coordonnées, GridSample et Resize, et la chaîne qui ajoute le flot à la grille."""
    prod = {o: n for n in graph.node for o in n.output}
    keep = {n.name for n in graph.node if not any(o in tainted for o in n.output)}
    keep |= {n.name for n in graph.node if n.op_type in FP32_OPS}
    for node in graph.node:
        if node.op_type != "GridSample":
            continue
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


def tensor_types(model: onnx.ModelProto, planes: int) -> dict[str, int]:
    """Les types des tenseurs, inférés sur une entrée de taille fixe : avec des dimensions libres,
    l'inférence s'arrête avant les blocs du v1."""
    fixed = onnx.ModelProto()
    fixed.CopyFrom(model)
    for dim, value in zip(fixed.graph.input[0].type.tensor_type.shape.dim, (1, planes, 768, 1280)):
        dim.Clear()
        dim.dim_value = value
    inferred = shape_inference.infer_shapes(fixed, data_prop=True).graph.value_info
    return {v.name: v.type.tensor_type.elem_type for v in inferred}


def convert_model(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    if not any(n.op_type == "GridSample" for n in g.node):
        raise RuntimeError("aucun GridSample : pas un graphe RIFE")
    planes = g.input[0].type.tensor_type.shape.dim[1].dim_value if len(g.input) == 1 else 0
    if planes not in (7, 11):
        raise RuntimeError("entrée attendue : [N, 11, H, W] (RIFE v1) ou [N, 7, H, W] (RIFE v2)")
    types = tensor_types(model, planes)
    del g.value_info[:]
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
    parser.add_argument("source", type=Path, help="dossier des RIFE v1 ou v2 en fp32")
    parser.add_argument("destination", type=Path, help="dossier des RIFE en FP16")
    parser.add_argument("models", nargs="*", help="noms des modèles (défaut : tous les .onnx)")
    args = parser.parse_args()
    names = args.models or sorted(p.name for p in args.source.glob("*.onnx"))
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = onnx.load(str(args.source / name))
        # Vérification complète (types et formes) quand la source la passe : les exports v1 ont
        # un Concat que l'inférence de formes refuse déjà.
        try:
            onnx.checker.check_model(source, full_check=True)
            full = True
        except onnx.checker.ValidationError:
            full = False
        except shape_inference.InferenceError:
            full = False
        model = convert_model(source)
        onnx.checker.check_model(model, full_check=full)
        onnx.save(model, str(args.destination / name))
        before, after = (args.source / name).stat().st_size, (args.destination / name).stat().st_size
        print(f"{name} : {before / 1e6:.1f} Mo → {after / 1e6:.1f} Mo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
