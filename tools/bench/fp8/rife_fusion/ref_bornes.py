#!/usr/bin/env python3
"""Reference ONNX Runtime CPU : ecrit, pour chaque cas, l'entree, les tenseurs aux frontieres (encode,
entrees x0..x4 et sorties y0..y4 des blocs) et la sortie du reseau entier, en binaires fp16 bruts.

Usage : ref_bornes.py <dossier_morceaux> <dossier_sortie> [nombre_de_cas]
"""
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from commun import entree_rife, paire

morceaux, sortie = sys.argv[1], sys.argv[2]
cas = [(n, d) for n in ("s5", "s9", "s14") for d in (6, 20)]
if len(sys.argv) > 3:
    cas = cas[: int(sys.argv[3])]

ENC_IN = "/Reshape_output_0"
ENC_OUT = "/encode/cnn3/ConvTranspose_output_0"
BLOC_IN = ["/block0/Resize_output_0", "/block1/Resize_output_0", "/block2/Resize_output_0",
           "/block3/Resize_output_0", "/Concat_5_output_0"]
BLOC_OUT = [f"/block{b}/lastconv/lastconv.0/ConvTranspose_output_0" for b in range(5)]


def session(nom):
    return ort.InferenceSession(f"{morceaux}/{nom}.onnx", providers=["CPUExecutionProvider"])


S = {nom: session(nom) for nom in ["enc", "rife_fixe"] + [f"bloc{b}" for b in range(5)] + [f"colle{b}" for b in range(6)]}
for n, d in cas:
    dossier = f"{sortie}/{n}_{d}"
    os.makedirs(dossier, exist_ok=True)
    a, b = paire(n, d)
    x = entree_rife(a, b)
    x.tofile(f"{dossier}/in.bin")
    complet = S["rife_fixe"].run(None, {"input": x})[0]
    complet.tofile(f"{dossier}/complet.bin")
    enc = S["enc"].run(None, {ENC_IN: x[:, 0:6].reshape(2, 3, x.shape[2], x.shape[3])})[0]
    enc.tofile(f"{dossier}/enc.bin")
    ys = []
    for k in range(5):
        feeds = {"input": x, ENC_OUT: enc}
        for j in range(k):
            feeds[BLOC_OUT[j]] = ys[j]
        xb = S[f"colle{k}"].run(None, feeds)[0]
        xb.tofile(f"{dossier}/x{k}.bin")
        ys.append(S[f"bloc{k}"].run(None, {BLOC_IN[k]: xb})[0])
        ys[-1].tofile(f"{dossier}/y{k}.bin")
    feeds = {"input": x, ENC_OUT: enc}
    for k in range(5):
        feeds[BLOC_OUT[k]] = ys[k]
    S["colle5"].run(None, feeds)[0].tofile(f"{dossier}/out.bin")
    print(n, d, "ok", flush=True)
