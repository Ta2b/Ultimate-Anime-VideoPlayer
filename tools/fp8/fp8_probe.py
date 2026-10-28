#!/usr/bin/env python3
"""Sonde FP8 pour un réseau quelconque (RIFE, DistilDRBA, GMFSS, SPAN) : NVIDIA Model Optimizer pose les Q/DQ
FP8 (les motifs que TensorRT fuse), trtexec chronomètre l'original et la variante, et le rapport des couches
dit combien de couches Reformat et Quantize isolées restent (un FP8 plus lent vient presque toujours de là).

Ne vérifie pas la qualité : la calibration est sur des entrées aléatoires, assez pour la vitesse. Un FP8 qui va
plus vite se calibre ensuite sur de vraies images (tools/span/quantize_span.py pour SPAN).

    pip install "nvidia-modelopt[onnx]"
    python tools/fp8/fp8_probe.py graphe.onnx --shape input=1x3x1088x1920 --out dossier
    python tools/fp8/fp8_probe.py gmfss_pair.onnx --shape frames=1x6x768x1280 --exclude Softmax,LayerNormalization
Les graphes à dimensions nommées (H, W…) se fixent avec --dim H=768 --dim W=1280. Écrit <nom>_fp8.onnx et
<nom>_fp8.json (temps, gain, couches). Aucune écriture dans le dossier des modèles du lecteur.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx


def fail(message: str) -> None:
    sys.exit(f"fp8_probe : {message}")


def parse_shapes(items: list[str]) -> dict[str, list[int]]:
    shapes = {}
    for item in items:
        name, _, dims = item.partition("=")
        if not dims:
            fail(f"--shape {item} : attendu nom=1x3x64x64")
        shapes[name] = [int(v) for v in dims.lower().split("x")]
    return shapes


def input_dtype(model: onnx.ModelProto, name: str) -> np.dtype:
    for value in model.graph.input:
        if value.name == name:
            return np.float16 if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 else np.float32
    fail(f"entrée {name} absente du graphe")
    raise AssertionError


def trtexec_run(trtexec: str, model: Path, shapes: dict[str, list[int]], layer_json: Path) -> tuple[float, dict]:
    shape_args = []
    for kind in ("minShapes", "optShapes", "maxShapes"):
        shape_args.append(f"--{kind}=" + ",".join(f"{n}:{'x'.join(map(str, d))}" for n, d in shapes.items()))
    command = [trtexec, f"--onnx={model}", *shape_args, "--warmUp=500", "--iterations=200", "--useCudaGraph",
               "--noDataTransfers", "--profilingVerbosity=detailed", f"--exportLayerInfo={layer_json}"]
    out = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    match = re.search(r"GPU Compute Time:.*?median = ([0-9.]+) ms", out.stdout)
    if out.returncode != 0 or not match:
        fail(f"trtexec a échoué sur {model.name} :\n{out.stdout[-1200:]}{out.stderr[-400:]}")
    counts = {"layers": 0, "reformat": 0, "quantize": 0, "fp8_conv": 0}
    try:
        text = layer_json.read_text()
        counts["layers"] = text.count('"LayerType"')
        counts["reformat"] = text.count('"Reformat"')
        counts["quantize"] = len(re.findall(r'"Name": "[^"]*(?:Quantize|_QuantizeLinear)[^"]*"', text))
        counts["fp8_conv"] = len(re.findall(r'(?:e4m3|FP8|fp8)', text))
    except OSError:
        pass
    return float(match.group(1)), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", action="append", default=[], help="nom=1x3x64x64 par entrée (répétable)")
    parser.add_argument("--dim", action="append", default=[], help="dimension nommée à fixer : H=768 (répétable)")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--method", choices=("max", "entropy"), default="max")
    parser.add_argument("--exclude", default="", help="types d'opérateurs laissés en FP16, séparés par des virgules")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    if args.dim:

        fixed = {k: int(v) for k, _, v in (d.partition("=") for d in args.dim)}
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param in fixed:
                    dim.dim_value = fixed[dim.dim_param]
    shapes = parse_shapes(args.shape)
    if not shapes:
        fail("au moins un --shape nom=dimensions")
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / (args.model.stem + "_fixed.onnx")
    onnx.save(model, str(base))
    target = args.out / (args.model.stem + "_fp8.onnx")

    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as error:
        fail(f"Model Optimizer introuvable ({error}) : pip install \"nvidia-modelopt[onnx]\"")
    rng = np.random.default_rng(0)
    data = {name: rng.random((args.samples, *dims[1:])).astype(input_dtype(model, name)) for name, dims in shapes.items()}
    wanted = {
        "onnx_path": str(base),
        "quantize_mode": "fp8",
        "calibration_data": data,
        "calibration_method": args.method,
        "calibration_shapes": ",".join(f"{n}:{'x'.join(map(str, d))}" for n, d in shapes.items()),
        "calibration_eps": ["cpu"],
        "output_path": str(target),
        "high_precision_dtype": "fp16",
        "op_types_to_exclude": [t for t in args.exclude.split(",") if t],
    }
    import inspect

    accepted = inspect.signature(quantize).parameters
    quantize(**{k: v for k, v in wanted.items() if k in accepted})
    result = onnx.load(str(target))
    qdq = sum(1 for n in result.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    print(f"{target.name} : {qdq} nœuds Q/DQ")

    half_ms, half_layers = trtexec_run(args.trtexec, base, shapes, args.out / "fp16_layers.json")
    fp8_ms, fp8_layers = trtexec_run(args.trtexec, target, shapes, args.out / "fp8_layers.json")
    report = {"model": args.model.name, "shapes": shapes, "qdq_nodes": qdq, "fp16_ms": half_ms, "fp8_ms": fp8_ms,
              "speedup": half_ms / fp8_ms, "fp16_layers": half_layers, "fp8_layers": fp8_layers}
    (args.out / (args.model.stem + "_fp8.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
