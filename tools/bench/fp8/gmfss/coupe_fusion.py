#!/usr/bin/env python3
"""Coupe le graphe « fusion » de GMFSS pour chercher pourquoi TensorRT l'envoie tout entier à Myelin.

Usage : coupe_fusion.py fusion.onnx sortie.onnx NOEUD_SORTIE [ENTREE ...]
  sans ENTREE : les huit entrées du graphe ; avec : les tenseurs nommés deviennent les entrées.
Exemple : coupe_fusion.py f.onnx a.onnx /fusionnet/Add_17_output_0   (sans la queue)
"""
import sys

import onnx
from onnx import shape_inference, utils


def main():
    modele, sortie, out = sys.argv[1], sys.argv[2], sys.argv[3]
    entrees = sys.argv[4:]
    m = shape_inference.infer_shapes(onnx.load(modele))
    if not entrees:
        entrees = [i.name for i in m.graph.input]
    tmp = sortie + ".formes.onnx"
    onnx.save(m, tmp)
    utils.extract_model(tmp, sortie, entrees, [out])
    print("coupe :", len(onnx.load(sortie).graph.node), "noeuds")


if __name__ == "__main__":
    main()
