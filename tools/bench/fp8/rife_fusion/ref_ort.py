#!/usr/bin/env python3
"""Reference ONNX Runtime CPU du reseau entier sur les paires de test ; ecrit les sorties .npy."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from commun import MODELE, entree_rife, paire

sortie = sys.argv[1]
cas = [(n, d) for n in ("s5", "s9", "s14") for d in (6, 20)]
if len(sys.argv) > 2:
    cas = cas[: int(sys.argv[2])]
os.makedirs(sortie, exist_ok=True)
so = ort.SessionOptions()
so.intra_op_num_threads = 8
sess = ort.InferenceSession(MODELE, so, providers=["CPUExecutionProvider"])
for n, d in cas:
    a, b = paire(n, d)
    x = entree_rife(a, b)
    np.save(f"{sortie}/entree_{n}_{d}.npy", x)
    t = time.time()
    y = sess.run(None, {"input": x})[0]
    print(n, d, y.shape, y.dtype, f"{time.time() - t:.1f}s", float(y.min()), float(y.max()), flush=True)
    np.save(f"{sortie}/ref_{n}_{d}.npy", y)
