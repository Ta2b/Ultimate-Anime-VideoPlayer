#!/usr/bin/env python3
"""Plafond théorique de la paire et de la fusion de GMFSS sur la RTX 4070 Ti (docs/gmfss-natif.md, « Plafonds
théoriques en 1080p »).

Fixe les dimensions nommées d'un graphe exporté par export_gmfss.py (« H », « W », puis « H/2 » à « W/8 »),
infère les formes, puis compte pour chaque convolution et chaque produit de matrices :
- ses FLOP (2 par multiplication-addition, sans compléter les canaux) ;
- ses octets minimaux en f16 (l'entrée lue une fois, la sortie écrite une fois, les poids).

Le plafond d'une couche est le plus lent du calcul et des octets ; celui d'un morceau (préfixe du nom des
nœuds : feat_ext, metricnet, flownet pour la paire ; ifnet, fusionnet pour la fusion) est la somme de ses
couches.

    python3 tools/gmfss/plafond_gmfss.py FUSION.onnx --height 1152 --width 1920 --parts ifnet,fusionnet --layers
    python3 tools/gmfss/plafond_gmfss.py PAIRE.onnx --height 1152 --width 1920 --parts feat_ext,metricnet
    python3 tools/gmfss/plafond_gmfss.py PAIRE.onnx --height 576 --width 960 --parts flownet

La dernière ligne donne GMFlow au demi-flot pour des plans de 1152 × 1920 : la paire le fait tourner sur la
moitié de ses images, que le demi-flot réduit encore de moitié. Demande onnx et onnxruntime (son inférence de
formes symbolique).
"""
import argparse
import collections

import numpy as np
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

# Débits de la RTX 4070 Ti (60 multiprocesseurs à 2,61 GHz, cœurs tensoriels sans creux) et sa bande passante.
PEAK = {"f16": 160.4e12, "f32": 80.2e12, "fp8": 320.8e12}  # sommes f16, sommes f32, FP8 à sommes f16
BANDWIDTH = 504e9


def inferred(path, height, width):
    model = onnx.load(path)
    dims = {"H": height, "W": width}
    for k in (2, 4, 8):
        dims[f"H/{k}"], dims[f"W/{k}"] = height // k, width // k
    for value in list(model.graph.input) + list(model.graph.output):
        for d in value.type.tensor_type.shape.dim:
            if d.dim_param in dims:
                d.dim_value = dims[d.dim_param]
    return SymbolicShapeInference.infer_shapes(model, auto_merge=True)


def layers(model, parts):
    """(morceau, nom, FLOP, octets) de chaque convolution et produit de matrices des morceaux demandés."""
    values = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in values}
    shapes.update({i.name: list(i.dims) for i in model.graph.initializer})
    rows = []
    for node in model.graph.node:
        part = node.name.strip("/").split("/")[0]
        if part not in parts:
            continue
        if node.op_type == "Conv":
            inp, weight, out = shapes[node.input[0]], shapes[node.input[1]], shapes[node.output[0]]
            flop = 2 * np.prod(out) * weight[1] * np.prod(weight[2:])
            moved = 2 * (np.prod(inp) + np.prod(out) + np.prod(weight))
        elif node.op_type in ("MatMul", "Gemm"):
            a, b, out = shapes[node.input[0]], shapes[node.input[1]], shapes[node.output[0]]
            flop = 2 * np.prod(out) * a[-1]
            moved = 2 * (np.prod(a) + np.prod(b) + np.prod(out))
        else:
            continue
        rows.append((part, node.name, float(flop), float(moved)))
    return rows


def ceiling(flop, moved, sums):
    return max(flop / PEAK[sums], moved / BANDWIDTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="graphe d'export_gmfss.py (la paire ou la fusion)")
    parser.add_argument("--height", type=int, required=True, help="H : hauteur des plans remplis")
    parser.add_argument("--width", type=int, required=True, help="W : largeur des plans remplis")
    parser.add_argument("--parts", required=True, help="préfixes des morceaux, séparés par des virgules")
    parser.add_argument("--layers", action="store_true", help="une ligne par couche en plus")
    args = parser.parse_args()
    rows = layers(inferred(args.model, args.height, args.width), set(args.parts.split(",")))
    totals = collections.defaultdict(lambda: np.zeros(6))
    for part, _, flop, moved in rows:
        t = totals[part]
        t += (flop, moved, ceiling(flop, moved, "f16"), ceiling(flop, moved, "f32"), ceiling(flop, moved, "fp8"), 1)
    print(f"{args.model} à {args.height} × {args.width}")
    for part, (flop, moved, f16, f32, fp8, count) in totals.items():
        print(f"  {part:10s} {int(count):4d} couches {flop / 1e9:8.1f} Gflop {moved / 1e6:8.0f} Mo  plafond f16 "
              f"{f16 * 1e3:6.2f} ms, f32 {f32 * 1e3:6.2f} ms, FP8 {fp8 * 1e3:6.2f} ms  (calcul f16 seul "
              f"{flop / PEAK['f16'] * 1e3:.2f} ms, octets seuls {moved / BANDWIDTH * 1e3:.2f} ms)")
    if args.layers:
        for part, name, flop, moved in rows:
            bound = "calcul" if flop / PEAK["f16"] >= moved / BANDWIDTH else "octets"
            print(f"    {name:75s} {flop / 1e9:7.1f} Gflop {moved / 1e6:7.1f} Mo  "
                  f"f16 {ceiling(flop, moved, 'f16') * 1e3:5.2f} ms ({bound})")


if __name__ == "__main__":
    main()
