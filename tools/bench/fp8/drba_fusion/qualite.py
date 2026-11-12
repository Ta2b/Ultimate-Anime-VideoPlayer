#!/usr/bin/env python3
"""PSNR / SSIM (fenêtre uniforme 7x7, float64, pic 1,0) des sorties fp16 brutes contre la référence ORT CPU
(`complet.bin` : graphe sans optimisation ; `complet_opt.bin` : ORT avec ses optimisations par défaut) pour la
chaîne fusionnée et le réseau entier TensorRT, plus chaîne contre réseau entier TensorRT.

Usage : qualite.py <dossier_refs> <dossier_sorties>
Code de sortie 1 si le PSNR de la chaîne fusionnée contre la référence « complet » passe sous 50 dB (le seuil de
la fiche, appliqué ici et plus lu à l'œil, C46-24).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import CAS, H, W, nom_cas, psnr, ssim

refs, sorties = sys.argv[1], sys.argv[2]


def lire(f):
    return np.fromfile(f, np.float16).reshape(3, H, W).astype(np.float32)


SEUIL_PSNR_DB = 50.0
pire = float("inf")
for c in CAS:
    cas = nom_cas(c)
    chaine = lire(f"{sorties}/chaine_{cas}.bin")
    plein = lire(f"{sorties}/plein_{cas}.bin")
    for nom_ref in ("complet", "complet_opt"):
        ref = lire(f"{refs}/{cas}/{nom_ref}.bin")
        if nom_ref == "complet":
            pire = min(pire, psnr(chaine, ref))
        for genre, s in (("chaine", chaine), ("TRT entier", plein)):
            print(f"{cas:12s} contre {nom_ref:11s} {genre:10s} PSNR {psnr(s, ref):6.2f} dB  SSIM {ssim(s, ref):.6f}  "
                  f"max|diff| {np.abs(s - ref).max():.4f}")
    print(f"{cas:12s} chaine contre TRT entier      PSNR {psnr(chaine, plein):6.2f} dB  SSIM {ssim(chaine, plein):.6f}")

tenu = pire >= SEUIL_PSNR_DB
print(f"pire PSNR de la chaîne contre complet : {pire:.2f} dB (seuil {SEUIL_PSNR_DB:.0f} dB) : {'TENU' if tenu else 'NON TENU'}")
sys.exit(0 if tenu else 1)
