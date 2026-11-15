#!/usr/bin/env python3
"""La fusion d'union avec un RIFE du lecteur à la place de son IFNet (essai du 15 novembre 2026).

L'IFNet d'union (un RIFE ancien, vers 4.6) coûte environ 3 ms de la fusion à 1280x768 sur une RTX 4070 Ti,
le double d'une passe RIFE 4.26 à la même taille (640x384, 1,5 ms) : ce script exporte la fusion sans lui,
l'image de l'IFNet devenant une entrée (FusionGiven), puis branche devant le graphe RIFE livré
(rife_v4.26.onnx, entrée de 11 plans du format v1 : a, b, l'instant, la grille et ses pas). Le graphe garde
les entrées et la sortie de la fusion d'origine (half0, half1, time, warped…, out) : le lecteur le charge
sans changement, sous le même nom de fichier ; la métadonnée « ajp.gmfss.ifnet » nomme le RIFE.

GridNet n'est pas réentraîné pour cette image. À l'œil (Re:Zero S4E18, ×2 puis ralenti ×4), Tatoub ne voit
pas de différence ; sur les instants qui diffèrent le plus, une lame rapide pâlit à certains et des mèches
restent plus nettes à d'autres.

RIFE travaille au 1/64 : la demi-taille doit être un multiple de 64, donc la taille de la chaîne un multiple
de 128 (kGmfssAlign).

    python3 -I tools/gmfss/ifnet_rife.py gmfss/roue/vsgmfss_fortuna --rife modeles/rife/rife_v4.26.onnx \\
        --out modeles/gmfss
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
from export_gmfss import export, frames_for, graph, prepare, run_onnx, settle_dims  # noqa: E402
from gmfss_graph import (  # noqa: E402
    FusionGiven,
    Pair,
    dims,
    fusion_axes,
    fusion_names,
    load_gmfss,
    splat_all,
    warped_names,
)

ALIGN = 128
IFNET_KEY = "ajp.gmfss.ifnet"


def export_given(gmfss, path, width, height):
    """La fusion FP16 dont l'image de l'IFNet est l'entrée « rife »."""
    img0, img1 = frames_for(width, height, seed=1)
    frames = torch.cat([img0, img1], 1).half()
    with torch.inference_mode():
        half0, half1, warped = splat_all(graph(gmfss, Pair(gmfss))(frames), 0.5)
    axes = {**fusion_axes(), "rife": dims(1)}
    export(graph(gmfss, FusionGiven(gmfss)), (half0, half1, half0.clone(), *warped), path,
           ["half0", "half1", "rife", *warped_names()], ["out"], axes)
    settle_dims(path, "union")


def rife_input_nodes(helper, tensor):
    """Les 11 plans de RIFE à partir de half0, half1 et time : a, b, l'instant étendu, la grille normalisée
    x puis y (2i / (n − 1) − 1) et ses pas 2 / (w − 1), 2 / (h − 1), calculés en FP32 puis en FP16 comme
    rife_input du lecteur. Les formes sont fixées par le lecteur (static_gmfss) : TensorRT les replie."""
    f32 = tensor.FLOAT
    c = {
        "i2": helper.make_tensor("ifr/i2", tensor.INT64, [1], [2]),
        "i3": helper.make_tensor("ifr/i3", tensor.INT64, [1], [3]),
        "zero": helper.make_tensor("ifr/zero", f32, [], [0.0]),
        "one": helper.make_tensor("ifr/one", f32, [], [1.0]),
        "two": helper.make_tensor("ifr/two", f32, [], [2.0]),
        "ones11": helper.make_tensor("ifr/ones11", tensor.INT64, [2], [1, 1]),
    }
    n = helper.make_node
    nodes = [
        n("Shape", ["half0"], ["ifr/shape"]),
        n("Gather", ["ifr/shape", "ifr/i2"], ["ifr/h"], axis=0),
        n("Gather", ["ifr/shape", "ifr/i3"], ["ifr/w"], axis=0),
        n("Concat", ["ifr/ones11", "ifr/h", "ifr/w"], ["ifr/plane"], axis=0),
        n("Cast", ["time"], ["ifr/t32"], to=f32),
        n("Expand", ["ifr/t32", "ifr/plane"], ["ifr/tplane"]),
    ]
    for axis, size, shape in (("x", "ifr/w", ["ifr/ones11", "ifr/ones11_1", "ifr/w"]),
                              ("y", "ifr/h", ["ifr/ones11", "ifr/h", "ifr/ones11_1"])):
        nodes += [
            n("Cast", [size], [f"ifr/{axis}n"], to=f32),
            n("Sub", [f"ifr/{axis}n", "ifr/one"], [f"ifr/{axis}n1"]),
            n("Div", ["ifr/two", f"ifr/{axis}n1"], [f"ifr/{axis}step"]),
            n("Squeeze", [size], [f"ifr/{axis}count"]),
            n("Cast", [f"ifr/{axis}count"], [f"ifr/{axis}countf"], to=f32),
            n("Range", ["ifr/zero", f"ifr/{axis}countf", "ifr/one"], [f"ifr/{axis}i"]),
            n("Mul", [f"ifr/{axis}i", f"ifr/{axis}step"], [f"ifr/{axis}2"]),
            n("Sub", [f"ifr/{axis}2", "ifr/one"], [f"ifr/{axis}g"]),
            n("Concat", shape, [f"ifr/{axis}shape"], axis=0),
            n("Reshape", [f"ifr/{axis}g", f"ifr/{axis}shape"], [f"ifr/{axis}line"]),
            n("Expand", [f"ifr/{axis}line", "ifr/plane"], [f"ifr/{axis}plane"]),
            n("Expand", [f"ifr/{axis}step", "ifr/plane"], [f"ifr/{axis}splane"]),
        ]
    c["ones11_1"] = helper.make_tensor("ifr/ones11_1", tensor.INT64, [1], [1])
    nodes += [
        n("Concat", ["ifr/tplane", "ifr/xplane", "ifr/yplane", "ifr/xsplane", "ifr/ysplane"], ["ifr/planes32"],
          axis=1),
        n("Cast", ["ifr/planes32"], ["ifr/planes"], to=tensor.FLOAT16),
        n("Concat", ["half0", "half1", "ifr/planes"], ["ifr/input"], axis=1),
    ]
    return nodes, list(c.values())


def compose(given_path, rife_path, out_path):
    """Le graphe RIFE (noms préfixés « ifr/ ») branché sur l'entrée « rife » de la fusion donnée."""
    import onnx
    from onnx import TensorProto, helper

    given = onnx.load(str(given_path))
    rife = onnx.load(str(rife_path))
    assert [i.name for i in rife.graph.input] == ["input"] and len(rife.graph.output) == 1, "RIFE v1 attendu"
    assert rife.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 11, "RIFE à 11 plans attendu"
    rename = {"input": "ifr/input", rife.graph.output[0].name: "rife"}

    def name(x):
        return rename.get(x, f"ifr/r/{x}" if x else x)

    for node in rife.graph.node:
        node.name = f"ifr/r/{node.name}"
        node.input[:] = [name(x) for x in node.input]
        node.output[:] = [name(x) for x in node.output]
        for attr in node.attribute:  # les sous-graphes (If, Loop) ne sont pas attendus dans RIFE
            assert attr.type != onnx.AttributeProto.GRAPH, node.op_type
    for init in rife.graph.initializer:
        init.name = name(init.name)
    prefix, constants = rife_input_nodes(helper, TensorProto)
    g = given.graph
    # GridNet ne lit que les splats et l'image de l'IFNet : half0 et half1 ne servent plus qu'à RIFE.
    halves = [helper.make_tensor_value_info(f"half{k}", TensorProto.FLOAT16, [1, 3, "H/2", "W/2"]) for k in (0, 1)]
    time = helper.make_tensor_value_info("time", TensorProto.FLOAT16, [1, 1, 1, 1])
    inputs = [*halves, time, *(i for i in g.input if i.name.startswith("warped"))]
    merged = helper.make_graph(list(prefix) + list(rife.graph.node) + list(g.node), g.name, inputs, g.output,
                               initializer=list(constants) + list(rife.graph.initializer) + list(g.initializer))
    opset = max(o.version for o in list(given.opset_import) + list(rife.opset_import) if o.domain in ("", "ai.onnx"))
    model = helper.make_model(merged, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = max(given.ir_version, rife.ir_version)
    for prop in given.metadata_props:
        model.metadata_props.add(key=prop.key, value=prop.value)
    model.metadata_props.add(key=IFNET_KEY, value=Path(rife_path).name)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    assert [i.name for i in model.graph.input] == fusion_names("union"), "les entrées de la fusion d'origine"


def rife_planes(half0, half1, t):
    """rife_input du lecteur, en numpy, pour la vérification."""
    _, _, h, w = half0.shape
    x = np.empty((1, 11, h, w), np.float32)
    x[0, 0:3], x[0, 3:6], x[0, 6] = half0[0], half1[0], t
    x[0, 7] = (2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0)[None, :]
    x[0, 8] = (2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0)[:, None]
    x[0, 9], x[0, 10] = 2.0 / (w - 1), 2.0 / (h - 1)
    return x.astype(np.float16)


@torch.inference_mode()
def check(reference, rife_path, given_path, merged_path, width, height, t):
    """Le graphe composé contre les deux graphes enchaînés à la main : la composition doit être exacte. Les
    entrées viennent de la paire en FP32 (en FP16 sur processeur, ses splats débordent à certaines tailles)."""
    img0, img1 = frames_for(width, height, seed=width + 3 * height)
    pair = [x.half() for x in Pair(reference)(torch.cat([img0, img1], 1))]
    half0, half1, warped = splat_all(pair, t)
    h0, h1 = half0.numpy(), half1.numpy()
    rife = run_onnx(rife_path, {"input": torch.from_numpy(rife_planes(h0, h1, t))})[0]
    feeds = dict(zip(["rife", *warped_names()], [torch.from_numpy(rife), *warped]))
    chained = run_onnx(given_path, feeds)[0].astype(np.float32)
    time = torch.tensor([[[[t]]]], dtype=torch.float16)
    merged = run_onnx(merged_path, dict(zip(fusion_names("union"), [half0, half1, time, *warped])))[0]
    gap = float(np.abs(merged.astype(np.float32) - chained).max())
    print(f"  {width}x{height} t={t} : composé contre enchaîné {gap:.2e} (NaN : RIFE {np.isnan(rife).any()}, "
          f"enchaîné {np.isnan(chained).any()}, composé {np.isnan(merged).any()})")
    return gap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="dossier vsgmfss_fortuna extrait de la roue")
    parser.add_argument("--rife", type=Path, required=True, help="RIFE du lecteur (rife_v4.26.onnx, 11 plans)")
    parser.add_argument("--out", type=Path, required=True, help="dossier de gmfss_fortuna_union_fusion.onnx")
    args = parser.parse_args()
    network = load_gmfss(args.package.resolve())
    reference = network(str(args.package / "models"), "union", 1.0, False).eval()
    gmfss = prepare(network(str(args.package / "models"), "union", 1.0, False).eval(), "fp16")
    args.out.mkdir(parents=True, exist_ok=True)
    merged = args.out / "gmfss_fortuna_union_fusion.onnx"
    with tempfile.TemporaryDirectory() as scratch:
        given = Path(scratch) / "given.onnx"
        export_given(gmfss, given, 256, 256)
        compose(given, args.rife, merged)
        gaps = [check(reference, args.rife, given, merged, w, h, t) for w, h, t in ((256, 256, 0.5), (384, 512, 0.25))]
    # Les deux chemins font les mêmes opérations FP16 ; seuls l'ordre des réductions et le repli des
    # constantes par ONNX Runtime diffèrent.
    if not all(np.isfinite(gaps)) or max(gaps) > 2e-3:
        sys.exit(f"composition : écarts {gaps} (≤ 2e-3) : le RIFE n'est pas branché comme il faut")
    print(merged)


if __name__ == "__main__":
    main()
