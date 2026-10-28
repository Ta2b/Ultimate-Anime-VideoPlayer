#!/usr/bin/env python3
"""Prépare le bloc résiduel plein résolution du réseau « fusion » de GMFSS pour le prototype CUDA.

Bloc retenu : fusionnet/residual_model_05 (64 canaux, 384x640), le plus gros motif fusionnable :
    p   = PReLU(x)                          (produit par l'épilogue du bloc précédent)
    q   = PReLU(Conv3x3(p) + b1)            <- noyau 1 : conv + biais + PReLU
    out = Conv3x3(q) + b2 + x               <- noyau 2 : conv + biais + saut
À partir d'entrées réelles (dossier produit par gen_entrees.py), ONNX Runtime CPU exécute la fusion
entière avec p, x et out en sorties supplémentaires. Écrit dans le dossier de sortie :
  p.bin, x.bin, ref.bin (FP16 NHWC), w1.bin, w2.bin (FP16 KRSC), b1.bin, b2.bin (FP32),
  params.txt (pente du PReLU intermédiaire), bloc.onnx (sous-graphe p, x -> out, pour TensorRT)
  et bloc_prelu.onnx (le même avec le PReLU en un noeud PRelu).

Usage : prep_bloc.py dossier_entrees dossier_sortie
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, utils

ort.set_default_logger_severity(3)
MODELE = Path.home() / ".local/share/ajp/ajp/modeles/gmfss_fortuna_base_fusion-1280x768-d47e821b27a101c0.onnx"
P = "/fusionnet/residual_model_05/residual_model_05.0/Sub_output_0"
X = "/fusionnet/Add_13_output_0"
OUT = "/fusionnet/Add_16_output_0"
W1 = "gmfss.fusionnet.residual_model_05.1.weight"
B1 = "gmfss.fusionnet.residual_model_05.1.bias"
W2 = "gmfss.fusionnet.residual_model_05.3.weight"
B2 = "gmfss.fusionnet.residual_model_05.3.bias"
SLOPE = "onnx::Mul_793"  # Mul de residual_model_05.2


def nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def main():
    entrees, sortie = Path(sys.argv[1]), Path(sys.argv[2])
    sortie.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(MODELE))
    ini = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            ini[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    # sorties supplémentaires
    for nom in (P, X, OUT):
        m.graph.output.append(helper.make_tensor_value_info(nom, onnx.TensorProto.FLOAT16, None))
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {i.name: np.load(entrees / f"{i.name}.npy") for i in s.get_inputs()}
    noms = [o.name for o in s.get_outputs()]
    res = dict(zip(noms, s.run(None, feeds)))
    for nom, fic in ((P, "p"), (X, "x"), (OUT, "ref")):
        a = res[nom]
        print(fic, a.shape, a.dtype, float(a.min()), float(a.max()))
        nhwc(a).astype(np.float16).tofile(sortie / f"{fic}.bin")
    for nom, fic in ((W1, "w1"), (W2, "w2")):
        w = ini[nom]  # OIHW -> KRSC
        w.transpose(0, 2, 3, 1).astype(np.float16).copy().tofile(sortie / f"{fic}.bin")
    ini[B1].astype(np.float32).tofile(sortie / "b1.bin")
    ini[B2].astype(np.float32).tofile(sortie / "b2.bin")
    pente = float(ini[SLOPE].ravel()[0])
    (sortie / "params.txt").write_text(f"{pente!r}\n")
    print("pente PReLU intermédiaire :", pente)
    # sous-graphe pour TensorRT
    complet = onnx.shape_inference.infer_shapes(onnx.load(str(MODELE)))
    onnx.save(complet, str(sortie / "fusion_formes.onnx"))
    utils.extract_model(str(sortie / "fusion_formes.onnx"), str(sortie / "bloc.onnx"), [P, X], [OUT])
    subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("reecrit_prelu.py")),
                    str(sortie / "bloc.onnx"), str(sortie / "bloc_prelu.onnx")], check=True)


if __name__ == "__main__":
    main()
