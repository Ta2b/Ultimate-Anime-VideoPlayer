#!/usr/bin/env python3
"""PSNR / SSIM (fenetre uniforme 7x7, float64, pic 1.0) des sorties fp16 brutes contre la reference ORT CPU
(`complet.bin` de ref_bornes.py), pour la chaine fusionnee et le reseau entier TensorRT, sur les 6 cas.

Usage : qualite.py <dossier_bornes> <dossier_sorties>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from commun import H, W, psnr, ssim

bornes, sorties = sys.argv[1], sys.argv[2]
for cas in ("s5_6", "s5_20", "s9_6", "s9_20", "s14_6", "s14_20"):
    ref = np.fromfile(f"{bornes}/{cas}/complet.bin", np.float16).reshape(3, H, W).astype(np.float32)
    for genre in ("chaine", "plein"):
        fichier = f"{sorties}/{genre}_{cas}.bin"
        if not os.path.exists(fichier):
            continue
        s = np.fromfile(fichier, np.float16).reshape(3, H, W).astype(np.float32)
        print(f"{cas:7s} {genre:7s} PSNR {psnr(s, ref):6.2f} dB  SSIM {ssim(s, ref):.6f}  max|diff| {np.abs(s - ref).max():.4f}")
