#!/usr/bin/env python3
"""Référence ONNX Runtime CPU (FP16) du réseau entier : pour chaque cas, écrit en binaires fp16 bruts les entrées
(in01, in12), les tenseurs aux frontières (enc_in, enc_out, x0..x2, y0..y2) et la sortie du réseau entier
(complet), tous issus de la MÊME exécution SANS optimisation ORT (le graphe tel qu'écrit : Resize et GridSample
en FP32 entre les Cast, comme TensorRT en typage strict). complet_opt.bin : la sortie de ORT avec ses
optimisations par défaut (suppressions de Cast qui font calculer certains Resize en FP16).

Usage : ref_bornes.py <dossier_plie.onnx> <dossier_sortie> [nombre_de_cas]
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import BLOC_IN, BLOC_OUT, CAS, ENC_IN, ENC_OUT, entrees_drba, nom_cas, trois_images

plie, sortie = sys.argv[1], sys.argv[2]
cas = CAS[: int(sys.argv[3])] if len(sys.argv) > 3 else CAS

m = onnx.load(plie)
extra = ENC_IN + ENC_OUT + BLOC_IN + BLOC_OUT
for n in extra:
    m.graph.output.append(onnx.helper.make_tensor_value_info(n, 10, None))
so = ort.SessionOptions()
so.intra_op_num_threads = 8
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so_opt = ort.SessionOptions()
so_opt.intra_op_num_threads = 8
sess_opt = ort.InferenceSession(plie, so_opt, providers=["CPUExecutionProvider"])
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
for c in cas:
    dossier = f"{sortie}/{nom_cas(c)}"
    os.makedirs(dossier, exist_ok=True)
    p01, p12 = entrees_drba(trois_images(c[0], c[1]), c[2])
    p01.tofile(f"{dossier}/in01.bin")
    p12.tofile(f"{dossier}/in12.bin")
    res = dict(zip(["out"] + extra, sess.run(None, {"pair01": p01, "pair12": p12})))
    res["out"].tofile(f"{dossier}/complet.bin")
    sess_opt.run(None, {"pair01": p01, "pair12": p12})[0].tofile(f"{dossier}/complet_opt.bin")
    np.concatenate([res[n] for n in ENC_IN], 0).tofile(f"{dossier}/enc_in.bin")
    np.concatenate([res[n] for n in ENC_OUT], 0).tofile(f"{dossier}/enc_out.bin")
    for b in range(3):
        res[BLOC_IN[b]].tofile(f"{dossier}/x{b}.bin")
        res[BLOC_OUT[b]].tofile(f"{dossier}/y{b}.bin")
    print(nom_cas(c), "ok", flush=True)
