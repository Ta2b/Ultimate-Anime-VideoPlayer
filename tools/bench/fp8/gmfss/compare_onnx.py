#!/usr/bin/env python3
"""Compare deux graphes ONNX de fusion GMFSS (ONNX Runtime CPU) sur des entrées .npy réelles.

Usage : compare_onnx.py dossier_entrees graphe.onnx   (compare à dossier/ref_out.npy)
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)


def main():
    d = Path(sys.argv[1])
    s = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
    feeds = {i.name: np.load(d / f"{i.name}.npy") for i in s.get_inputs()}
    out = s.run(None, feeds)[0].astype(np.float32)
    ref = np.load(d / "ref_out.npy").astype(np.float32)
    e = np.abs(out - ref)
    print(f"écart max {e.max():.3e}  moyen {e.mean():.3e}  p99,9 % {np.quantile(e, 0.999):.3e}")
    psnr = 10 * np.log10(1.0 / max((e**2).mean(), 1e-20))
    print(f"PSNR contre la référence : {psnr:.2f} dB")


if __name__ == "__main__":
    main()
