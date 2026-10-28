#!/usr/bin/env python3
"""Decoupe RIFE v4.25 en 6 moteurs basse resolution (encode + blocs 0..4) et 6 morceaux de colle
(tout ce qui est pleine resolution), puis verifie par ONNX Runtime CPU que l'enchainement des
morceaux redonne exactement la sortie du reseau entier.

Usage : decoupe.py <dossier_sortie> [entree.npy]
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

sys.path.insert(0, os.path.dirname(__file__))
from commun import MODELE, modele_fixe

dossier = sys.argv[1]
os.makedirs(dossier, exist_ok=True)
fixe = os.path.join(dossier, "rife_fixe.onnx")
onnx.save(modele_fixe(), fixe)

ENC_IN = "/Reshape_output_0"
ENC_OUT = "/encode/cnn3/ConvTranspose_output_0"
BLOC_IN = ["/block0/Resize_output_0", "/block1/Resize_output_0", "/block2/Resize_output_0",
           "/block3/Resize_output_0", "/Concat_5_output_0"]
BLOC_OUT = [f"/block{b}/lastconv/lastconv.0/ConvTranspose_output_0" for b in range(5)]
COLLE_OUT = BLOC_IN  # la colle b produit l'entree du bloc b ; la derniere produit "output"

# Moteurs : un sous-ONNX par morceau basse resolution.
extract_model(fixe, f"{dossier}/enc.onnx", [ENC_IN], [ENC_OUT])
for b in range(5):
    extract_model(fixe, f"{dossier}/bloc{b}.onnx", [BLOC_IN[b]], [BLOC_OUT[b]])
# Colle : entrees = "input", sortie encode et sorties des blocs precedents.
extract_model(fixe, f"{dossier}/colle0.onnx", ["input", ENC_OUT], [BLOC_IN[0]])
for b in range(1, 5):
    extract_model(fixe, f"{dossier}/colle{b}.onnx", ["input", ENC_OUT] + BLOC_OUT[:b], [BLOC_IN[b]])
extract_model(fixe, f"{dossier}/colle5.onnx", ["input", ENC_OUT] + BLOC_OUT, ["output"])

# Decompte des noeuds : tout le graphe doit etre couvert exactement une fois.
total = len(onnx.load(fixe).graph.node)
noeuds = {}
for nom in ["enc"] + [f"bloc{b}" for b in range(5)] + [f"colle{b}" for b in range(6)]:
    g = onnx.load(f"{dossier}/{nom}.onnx").graph
    noeuds[nom] = len([n for n in g.node if n.op_type != "Constant"])
    print(nom, "noeuds hors Constant :", noeuds[nom],
          "entrees", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input if i.name not in {x.name for x in g.initializer}],
          "sorties", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
print("noeuds hors Constant, reseau entier :", len([n for n in onnx.load(fixe).graph.node if n.op_type != "Constant"]),
      "morceaux :", sum(noeuds.values()))

# Enchainement ORT CPU.
x = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
if x is not None:
    def session(nom):
        return ort.InferenceSession(f"{dossier}/{nom}.onnx", providers=["CPUExecutionProvider"])

    ref = ort.InferenceSession(fixe, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    img = x[:, 0:6].reshape(2, 3, x.shape[2], x.shape[3])
    enc = session("enc").run(None, {ENC_IN: img})[0]
    ys = []
    for b in range(5):
        feeds = {"input": x, ENC_OUT: enc}
        for k in range(b):
            feeds[BLOC_OUT[k]] = ys[k]
        xb = session(f"colle{b}").run(None, feeds)[0]
        ys.append(session(f"bloc{b}").run(None, {BLOC_IN[b]: xb})[0])
        print("bloc", b, "entree", xb.shape, "sortie", ys[-1].shape)
    feeds = {"input": x, ENC_OUT: enc}
    for k in range(5):
        feeds[BLOC_OUT[k]] = ys[k]
    sortie = session("colle5").run(None, feeds)[0]
    ecart = np.abs(sortie.astype(np.float32) - ref.astype(np.float32))
    print("elements differents :", int((ecart > 0).sum()), "sur", ecart.size, "; ecart moyen", float(ecart.mean()),
          "; PSNR", float(10 * np.log10(1.0 / max(float((ecart ** 2).mean()), 1e-30))))
    print("max |chaine - entier| =", float(np.abs(sortie.astype(np.float32) - ref.astype(np.float32)).max()),
          "identique bit a bit :", bool(np.array_equal(sortie, ref)))
