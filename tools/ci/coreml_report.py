#!/usr/bin/env python3
"""Rapport Core ML des modèles livrés, ouverts comme le lecteur les ouvre (gpu/coreml) : formes
d'entrée fixées, ONNX Runtime et son fournisseur CoreML en programme ML, mêmes options. Pour chaque
modèle, forme et répartition d'unités :
  - les nœuds qu'ONNX Runtime laisse au CPU hors de Core ML (chaque frontière est un aller-retour
    par la mémoire de l'hôte) ;
  - le plan de calcul de Core ML (ProfileComputePlan) : sur quelle unité chaque opération tourne,
    le CPU compris, que la répartition demandée soit le GPU ou non ;
  - le temps médian d'une exécution.
Sur une machine virtuelle (runner GitHub), les temps ne valent pas ceux d'un Mac ; les placements,
si.

    python -I tools/ci/coreml_report.py <dossier models/> [--runs 8]
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# (nom, fichier sous models/, formes des entrées, répartitions essayées)
CASES = [
    ("SPAN Performance 540p", "onnx/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
     [[1, 3, 540, 960]], ["CPUAndGPU", "ALL", "CPUOnly"]),
    ("SPAN Performance 1080p", "onnx/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
     [[1, 3, 1080, 1920]], ["CPUAndGPU", "ALL"]),
    ("SPAN Balanced 540p", "onnx/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
     [[1, 3, 540, 960]], ["CPUAndGPU", "ALL"]),
    ("RIFE 4.25 720p", "rife/rife_v4.25.onnx", [[1, 11, 768, 1280]], ["CPUAndGPU", "ALL"]),
    ("RIFE 4.22 lite 720p", "rife/rife_v4.22_lite.onnx", [[1, 11, 768, 1280]], ["CPUAndGPU", "ALL"]),
    ("DistilDRBA v2 lite 540p", "rife/distil_drba_v2_lite.onnx", [[1, 7, 576, 1024], [1, 7, 576, 1024]],
     ["CPUAndGPU"]),
]

PLACED = re.compile(r"(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)")
CAPABILITY = re.compile(r"number of partitions supported by CoreML: (\d+).*?nodes in the graph: (\d+).*?"
                        r"supported by CoreML: (\d+)")
UNSUPPORTED = re.compile(r"Operator type: \[(\w+)\] index: \[\d+\] name: \[([^\]]*)\] supported: \[0\]")
CPU_NODES = re.compile(r"placed on \[CPUExecutionProvider\]\. Number of nodes: \d+\n((?:[ \t]+\S+ \([^)\n]*\)\n)+)")
DEVICE = re.compile(r"Operation: ([^,\s]+),? Device Usage: (\w+)(?:, Estimated Cost: ([\d.eE+-]+))?",
                    re.IGNORECASE)


def fixed_model(source: Path, shapes: list[list[int]], target: Path) -> None:
    import onnx

    model = onnx.load(str(source))
    initializers = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    for value, dims in zip(inputs, shapes, strict=True):
        for dim, size in zip(value.type.tensor_type.shape.dim, dims, strict=True):
            dim.ClearField("dim_param")
            dim.dim_value = size
    onnx.save(model, str(target))


def run_case(model: Path, units: str, runs: int, cache: Path) -> None:
    """Dans un processus à part : le journal verbeux d'ONNX Runtime va sur stderr."""
    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(0)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 0
    coreml = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": units,
        "RequireStaticInputShapes": "0",
        "SpecializationStrategy": "FastPrediction",
        "AllowLowPrecisionAccumulationOnGPU": "1",
        "ProfileComputePlan": "1",
        "ModelCacheDirectory": str(cache),
    }
    started = time.perf_counter()
    session = ort.InferenceSession(str(model), options, providers=[("CoreMLExecutionProvider", coreml)])
    opened = time.perf_counter() - started
    rng = np.random.default_rng(0)
    feeds = {}
    for value in session.get_inputs():
        dtype = np.float16 if value.type == "tensor(float16)" else np.float32
        feeds[value.name] = rng.random(value.shape, dtype=np.float32).astype(dtype)
    for _ in range(2):
        session.run(None, feeds)
    took = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feeds)
        took.append((time.perf_counter() - t0) * 1e3)
    print(f"RESULT opened={opened:.1f}s median={statistics.median(took):.1f}ms min={min(took):.1f}ms",
          flush=True)


def summarize(log: str) -> list[str]:
    lines = []
    for m in CAPABILITY.finditer(log):
        lines.append(f"  CoreML : {m[1]} partition(s), {m[3]}/{m[2]} nœuds")
    for m in PLACED.finditer(log):
        lines.append(f"  {m[3]} nœud(s) sur {m[2]}")
    for m in CPU_NODES.finditer(log):
        ops = collections.Counter(line.split()[0] for line in m[1].splitlines() if line.strip())
        lines.append("  au CPU hors de Core ML : " + ", ".join(f"{op} ×{n}" for op, n in ops.most_common()))
    unsupported = collections.Counter(m[1] for m in UNSUPPORTED.finditer(log))
    if unsupported:
        lines.append("  refusés par CoreML : " + ", ".join(f"{op} ×{n}" for op, n in unsupported.most_common()))
    devices = collections.defaultdict(collections.Counter)
    costs = collections.Counter()
    for m in DEVICE.finditer(log):
        devices[m[2]][m[1]] += 1
        costs[m[2]] += float(m[3] or 0.0)
    total = sum(costs.values()) or 1.0
    for device, ops in sorted(devices.items()):
        top = ", ".join(f"{op} ×{n}" for op, n in ops.most_common(8))
        lines.append(f"  plan Core ML, {device} : {sum(ops.values())} op(s), {costs[device] / total:.0%} du coût"
                     f" ({top})")
    result = re.search(r"RESULT .*", log)
    lines.append("  " + (result[0] if result else "échec : " + log.strip().splitlines()[-1][:300]))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--child", nargs=2, metavar=("MODEL", "UNITS"))
    args = parser.parse_args()
    if args.child:
        with tempfile.TemporaryDirectory() as cache:
            run_case(Path(args.child[0]), args.child[1], args.runs, Path(cache))
        return 0
    import onnxruntime as ort

    print(f"ONNX Runtime {ort.__version__}, fournisseurs {ort.get_available_providers()}")
    with tempfile.TemporaryDirectory() as work:
        for name, file, shapes, unit_list in CASES:
            source = args.models / file
            if not source.exists():
                print(f"{name} : {source} absent")
                continue
            model = Path(work) / (source.stem + "_" + "x".join(map(str, shapes[0])) + ".onnx")
            fixed_model(source, shapes, model)
            for units in unit_list:
                child = subprocess.run(
                    [sys.executable, "-I", __file__, str(args.models), "--runs", str(args.runs),
                     "--child", str(model), units],
                    capture_output=True, text=True, timeout=1800, check=False)
                print(f"{name}, {units} :")
                print("\n".join(summarize(child.stdout + child.stderr)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
