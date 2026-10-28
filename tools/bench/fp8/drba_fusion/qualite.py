#!/usr/bin/env python3
"""PSNR / SSIM (fenêtre uniforme 7x7, float64, pic 1,0) des sorties fp16 brutes contre la référence ORT CPU
(`complet.bin` : graphe sans optimisation ; `complet_opt.bin` : ORT avec ses optimisations par défaut) pour la
chaîne fusionnée et le réseau entier TensorRT, plus chaîne contre réseau entier TensorRT.

Usage : qualite.py <dossier_refs> <dossier_sorties>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import CAS, H, W, nom_cas, psnr, ssim

refs, sorties = sys.argv[1], sys.argv[2]


def lire(f):
    return np.fromfile(f, np.float16).reshape(3, H, W).astype(np.float32)


for c in CAS:
    cas = nom_cas(c)
    chaine = lire(f"{sorties}/chaine_{cas}.bin")
    plein = lire(f"{sorties}/plein_{cas}.bin")
    for nom_ref in ("complet", "complet_opt"):
        ref = lire(f"{refs}/{cas}/{nom_ref}.bin")
        for genre, s in (("chaine", chaine), ("TRT entier", plein)):
            print(f"{cas:12s} contre {nom_ref:11s} {genre:10s} PSNR {psnr(s, ref):6.2f} dB  SSIM {ssim(s, ref):.6f}  "
                  f"max|diff| {np.abs(s - ref).max():.4f}")
    print(f"{cas:12s} chaine contre TRT entier      PSNR {psnr(chaine, plein):6.2f} dB  SSIM {ssim(chaine, plein):.6f}")
