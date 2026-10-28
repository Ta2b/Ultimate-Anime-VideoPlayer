#!/usr/bin/env python3
"""Génère la paire GMFSS FP8 que le lecteur cherche (gmfss_fortuna_{base,union}_s05fp8_pair.onnx).

Passe par tools/fp8/fp8_probe.py (NVIDIA Model Optimizer pose les Q/DQ FP8, trtexec chronomètre l'original et
la variante) et n'installe le fichier que si le FP8 va au moins 5 % plus vite que le FP16 : un FP8 plus lent
(Q/DQ non fusionnés) ne doit pas devenir le cran Performance. --force installe quand même.

    pip install "nvidia-modelopt[onnx]"
    python tools/gmfss/quantize_gmfss_fp8.py models/gmfss_fortuna_base_s05_pair.onnx \
        --shape frames=1x6x768x1280 --work /tmp/gmfss_fp8 --models models

Le graphe d'entrée est la paire _s05 (GMFlow en FP16) ; Softmax et LayerNormalization restent en FP16.
Ne vérifie pas la qualité : à juger à l'œil sur de vraies images avant de livrer le fichier.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MIN_SPEEDUP = 1.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="paire _s05 (gmfss_fortuna_base_s05_pair.onnx ou _union_)")
    parser.add_argument("--shape", action="append", required=True, help="nom=1x6x768x1280 par entrée")
    parser.add_argument("--work", type=Path, required=True, help="dossier de travail (rapports, graphes)")
    parser.add_argument("--models", type=Path, required=True, help="dossier des modèles du lecteur")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--exclude", default="Softmax,LayerNormalization")
    parser.add_argument("--force", action="store_true", help="installer même si le gain est sous 5 %%")
    args = parser.parse_args()

    if "_s05_" not in args.model.name:
        sys.exit(f"quantize_gmfss_fp8 : {args.model.name} n'est pas une paire _s05")
    target_name = args.model.name.replace("_s05_", "_s05fp8_")
    probe = Path(__file__).resolve().parent.parent / "fp8" / "fp8_probe.py"
    command = [sys.executable, str(probe), str(args.model), "--out", str(args.work), "--trtexec", args.trtexec,
               "--exclude", args.exclude]
    for shape in args.shape:
        command += ["--shape", shape]
    subprocess.run(command, check=True)

    report = json.loads((args.work / (args.model.stem + "_fp8.json")).read_text())
    speedup = float(report["speedup"])
    print(f"FP8 {report['fp8_ms']:.2f} ms contre FP16 {report['fp16_ms']:.2f} ms : x{speedup:.2f}")
    if speedup < MIN_SPEEDUP and not args.force:
        print(f"sous x{MIN_SPEEDUP:.2f} : {target_name} non installé (--force pour passer outre)")
        return 1
    args.models.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.work / (args.model.stem + "_fp8.onnx"), args.models / target_name)
    print(f"installé : {args.models / target_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
