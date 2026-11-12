#!/usr/bin/env python3
"""Un morceau de RIFE fusionné (« *-bloc*.onnx », forme fixe en lot de 1) réécrit en lot de N : le banc des instants
groupés (chantier 2 du fil RIFE) mesure un moteur de bloc qui prend N instants d'une paire d'un coup, contre N passes
seules.

  bloc_lot.py ENTREE.onnx SORTIE.onnx N

La première dimension des entrées, des sorties et des value_info passe de 1 à N. Une constante de forme lue par un
Reshape (ou un Expand, un Tile) qui commence par 1 passe aussi à N ; une autre forme constante, ou une forme
calculée, est nommée sur la sortie d'erreur et le script s'arrête (code 3) plutôt que d'écrire un graphe faux. Les
poids et les autres nœuds sont recopiés tels quels. Il faut python3 avec onnx et numpy."""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SHAPE_OPS = {"Reshape": 1, "Expand": 1, "Tile": 1}


def set_batch(value_info, n):
    dims = value_info.type.tensor_type.shape.dim
    if dims and dims[0].HasField("dim_value") and dims[0].dim_value == 1:
        dims[0].dim_value = n


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    source, target, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if n < 1:
        print("N doit valoir au moins 1", file=sys.stderr)
        return 2
    model = onnx.load(source)
    graph = model.graph
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        set_batch(vi, n)
    initializers = {init.name: init for init in graph.initializer}
    constants = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = attr
    problems = []
    for node in graph.node:
        if node.op_type not in SHAPE_OPS:
            continue
        name = node.input[SHAPE_OPS[node.op_type]]
        if name in initializers:
            array = numpy_helper.to_array(initializers[name]).copy()
            holder = ("init", initializers[name])
        elif name in constants:
            array = numpy_helper.to_array(constants[name].t).copy()
            holder = ("const", constants[name])
        else:
            problems.append(f"{node.op_type} {node.name} : forme calculée ({name})")
            continue
        if array.ndim != 1 or array.size == 0:
            problems.append(f"{node.op_type} {node.name} : forme {array.tolist()} inattendue")
            continue
        if array[0] == 1:
            array[0] = n
        elif array[0] not in (0, -1):
            problems.append(f"{node.op_type} {node.name} : lot {array[0]} dans la forme {array.tolist()}")
            continue
        tensor = numpy_helper.from_array(array.astype(np.int64), name)
        if holder[0] == "init":
            holder[1].CopyFrom(tensor)
        else:
            holder[1].t.CopyFrom(tensor)
    if problems:
        print("formes que le script ne sait pas mettre en lot :", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 3
    # Pas de onnx.checker : les morceaux de split_rife laissent la forme de leur sortie à TensorRT, ce que le
    # vérificateur refuse ; ce script ne touche pas aux sorties sans forme.
    onnx.save(model, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
