#!/usr/bin/env python3
"""Réécrit le graphe ONNX GMFSS (fusion) : le PReLU exporté en cinq noeuds
(Relu, Neg, Relu, Mul, Sub) redevient un noeud PRelu unique.

  PReLU(x) = relu(x) - a * relu(-x)        (a scalaire)

TensorRT fusionne alors l'activation dans l'épilogue de la convolution voisine
(ou, à défaut, la traite en un seul noyau élémentaire au lieu de plusieurs).

Option --depth2space : le PixelShuffle de la queue (Shape/Gather/Div/Cast/Unsqueeze/Concat/Reshape/
Transpose/Reshape, dont les formes se calculent à l'exécution) devient un noeud DepthToSpace(CRD) ;
sans ce calcul de formes, TensorRT n'a plus de raison d'envoyer tout le graphe à Myelin.
Option --sans-prelu : ne réécrit pas les PReLU (pour isoler l'effet du seul DepthToSpace).

Usage : reecrit_prelu.py entrée.onnx sortie.onnx [--depth2space] [--sans-prelu]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def constantes(m):
    """Table nom -> tableau numpy des initialiseurs et des noeuds Constant."""
    t = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            t[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return t


PS_PREFIXE = "/fusionnet/residual_model_tail/upsample/upsample.1/"


def depth2space(m):
    """Remplace le PixelShuffle exporté de la queue par DepthToSpace(CRD, bloc 2)."""
    entree = "/fusionnet/residual_model_tail/upsample/upsample.0/Conv_output_0"
    sortie = PS_PREFIXE + "Reshape_1_output_0"
    garde, premier = [], None
    for i, n in enumerate(m.graph.node):
        if n.name.startswith(PS_PREFIXE):
            premier = i if premier is None else premier
            continue
        garde.append(n)
    assert premier is not None, "PixelShuffle introuvable"
    garde.insert(premier, helper.make_node("DepthToSpace", [entree], [sortie], name=PS_PREFIXE + "DepthToSpace",
                                           blocksize=2, mode="CRD"))
    del m.graph.node[:]
    m.graph.node.extend(garde)
    # les valeurs intermédiaires typées de l'ancien sous-graphe ne servent plus
    vi = [v for v in m.graph.value_info if not v.name.startswith(PS_PREFIXE) or v.name == sortie]
    del m.graph.value_info[:]
    m.graph.value_info.extend(vi)


def main():
    m = onnx.load(sys.argv[1])
    if "--depth2space" in sys.argv:
        depth2space(m)
    if "--sans-prelu" in sys.argv:
        onnx.checker.check_model(m)
        onnx.save(m, sys.argv[2])
        print("DepthToSpace seul :", len(m.graph.node), "noeuds")
        return
    cst = constantes(m)
    prod = {o: n for n in m.graph.node for o in n.output}
    conso = {}
    for n in m.graph.node:
        for i in n.input:
            conso.setdefault(i, []).append(n)

    supprimes = set()
    ajouts = {}  # nom du Sub -> nouveau noeud PRelu
    for n in m.graph.node:
        if n.op_type != "Sub":
            continue
        a, b = (prod.get(i) for i in n.input)
        if not (a and b and a.op_type == "Relu" and b.op_type == "Mul"):
            continue
        # Mul(constante, Relu(Neg(x))) dans un ordre quelconque
        c = [i for i in b.input if i in cst]
        r = [i for i in b.input if i not in cst]
        if len(c) != 1 or len(r) != 1 or cst[c[0]].size != 1:
            continue
        r1 = prod[r[0]]
        if r1.op_type != "Relu":
            continue
        ng = prod[r1.input[0]]
        if ng.op_type != "Neg" or ng.input[0] != a.input[0]:
            continue
        x = a.input[0]
        pente = numpy_helper.from_array(
            np.full((1, 1, 1, 1), float(cst[c[0]].ravel()[0]), dtype=cst[c[0]].dtype),
            n.name + "/pente",
        )
        m.graph.initializer.append(pente)
        ajouts[n.name] = helper.make_node(
            "PRelu", [x, pente.name], [n.output[0]], name=n.name + "/PRelu"
        )
        supprimes.update({b.name})
        # Relu / Neg / Relu_1 ne sont supprimés que s'ils n'ont plus d'autre consommateur
        for t in (a, r1, ng):
            supprimes.add(t.name)

    # Recompte : un noeud partagé (Relu de x lu par deux activations) reste utile
    # tant qu'un Sub non réécrit le lit encore ; ici tout consommateur est un Sub réécrit.
    nouveaux = []
    for n in m.graph.node:
        if n.name in ajouts:
            nouveaux.append(ajouts[n.name])
        elif n.name in supprimes:
            continue
        else:
            nouveaux.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nouveaux)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    print("PReLU réécrits :", len(ajouts), "noeuds :", len(nouveaux))


if __name__ == "__main__":
    main()
