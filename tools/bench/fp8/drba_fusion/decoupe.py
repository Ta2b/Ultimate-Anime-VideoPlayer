#!/usr/bin/env python3
"""Découpe DistilDRBA v2 lite en 4 moteurs basse résolution (encodeur, blocs 0 à 2) et morceaux de colle
(tout le reste : réductions d'images, mélange du côté, Resize, warps, Concat, mise en forme finale), puis
vérifie par ONNX Runtime CPU que l'enchaînement redonne la sortie du réseau entier.

Usage : decoupe.py <dossier_sortie> [paire01.npy paire12.npy]
Écrit : plie.onnx (graphe plié), enc1.onnx, enc.onnx (lot de 3 images), bloc0..2.onnx, colle_pre/0/1/2/3.onnx.
"""
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference
from onnx.utils import extract_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import BLOC_IN, BLOC_OUT, ENC_IN, ENC_OUT, MODELE

dossier = sys.argv[1]
os.makedirs(dossier, exist_ok=True)
plie = os.path.join(dossier, "plie.onnx")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plie.py"), MODELE, plie], check=True)

IN = ["pair01", "pair12"]
F3 = ENC_OUT
# Moteurs : un sous-ONNX par morceau basse résolution.
extract_model(plie, f"{dossier}/enc1.onnx", [ENC_IN[0]], [ENC_OUT[0]])
for b in range(3):
    extract_model(plie, f"{dossier}/bloc{b}.onnx", [BLOC_IN[b]], [BLOC_OUT[b]])
# Colle : bornée par les entrées du réseau, les sorties de l'encodeur et celles des blocs précédents.
extract_model(plie, f"{dossier}/colle_pre.onnx", IN, ENC_IN)
extract_model(plie, f"{dossier}/colle0.onnx", IN + F3, [BLOC_IN[0]])
for b in range(1, 3):
    extract_model(plie, f"{dossier}/colle{b}.onnx", IN + F3 + BLOC_OUT[:b], [BLOC_IN[b]])
extract_model(plie, f"{dossier}/colle3.onnx", IN + F3 + BLOC_OUT, ["out"])

# Encodeur en lot de 3 : l'entrée 1x3x272x480 devient 3x3x272x480 (poids partagés par les trois images).
m = onnx.load(f"{dossier}/enc1.onnx")
for vi in list(m.graph.input) + list(m.graph.output):
    vi.type.tensor_type.shape.dim[0].dim_value = 3
del m.graph.value_info[:]
m = shape_inference.infer_shapes(m)
onnx.save(m, f"{dossier}/enc.onnx")

noeuds = {}
for nom in ["enc1"] + [f"bloc{b}" for b in range(3)] + ["colle_pre"] + [f"colle{b}" for b in range(4)]:
    g = onnx.load(f"{dossier}/{nom}.onnx").graph
    noeuds[nom] = len([n for n in g.node if n.op_type != "Constant"])
    print(nom, "noeuds :", noeuds[nom])
total = len(onnx.load(plie).graph.node)
print("noeuds du graphe plié :", total, "; somme des morceaux (enc1 compté 3 fois) :",
      sum(noeuds.values()) + 2 * noeuds["enc1"])

SANS_OPT = ort.SessionOptions()
SANS_OPT.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # coupes vérifiées sans fusions ORT
if len(sys.argv) > 3:
    p01, p12 = np.load(sys.argv[2]), np.load(sys.argv[3])

    def session(nom):
        return ort.InferenceSession(f"{dossier}/{nom}.onnx", SANS_OPT, providers=["CPUExecutionProvider"])

    ref = ort.InferenceSession(plie, SANS_OPT, providers=["CPUExecutionProvider"]).run(None, {"pair01": p01, "pair12": p12})[0]
    feeds = {"pair01": p01, "pair12": p12}
    imgs = session("colle_pre").run(ENC_IN, feeds)
    enc = session("enc1")
    fs = [enc.run(None, {ENC_IN[0]: i})[0] for i in imgs]
    feeds.update({n: v for n, v in zip(F3, fs)})
    # Le lot de 3 doit redonner les trois sorties de l'encodeur seul.
    lot = ort.InferenceSession(f"{dossier}/enc.onnx", SANS_OPT, providers=["CPUExecutionProvider"]).run(
        None, {ENC_IN[0]: np.concatenate(imgs, 0)})[0]
    print("encodeur en lot de 3 : max |lot - 3 x seul| =",
          float(np.abs(lot.astype(np.float32) - np.concatenate(fs, 0).astype(np.float32)).max()))
    ys = []
    for b in range(3):
        xb = session(f"colle{b}").run(None, feeds)[0]
        ys.append(session(f"bloc{b}").run(None, {BLOC_IN[b]: xb})[0])
        feeds[BLOC_OUT[b]] = ys[-1]
        print("bloc", b, "entrée", xb.shape, "sortie", ys[-1].shape)
    sortie = session("colle3").run(None, feeds)[0]
    ecart = np.abs(sortie.astype(np.float32) - ref.astype(np.float32))
    print("éléments différents :", int((ecart > 0).sum()), "sur", ecart.size, "; max", float(ecart.max()),
          "; PSNR", float(10 * np.log10(1.0 / max(float((ecart ** 2).mean()), 1e-30))))
