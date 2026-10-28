#!/usr/bin/env python3
"""Recadrages 544x960 (hors images de calibration) en binaires FP16 NCHW pour trtexec --loadInputs.
Usage : prep_inputs.py sortie_dir image1.png image2.png ..."""
import os, sys
import numpy as np
from PIL import Image

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for i, p in enumerate(sys.argv[2:], 1):
    im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
    y0, x0 = (im.shape[0] - 544) // 2, (im.shape[1] - 960) // 2
    c = im[y0:y0 + 544, x0:x0 + 960].transpose(2, 0, 1)[None].astype(np.float16)
    c.tofile(os.path.join(out, f"in{i}.bin"))
    print("écrit", os.path.join(out, f"in{i}.bin"), c.shape)
