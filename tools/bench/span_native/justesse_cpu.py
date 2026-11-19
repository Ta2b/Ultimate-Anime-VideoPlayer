"""La justesse du SPAN natif sur processeur, dans le cloud, à partir de ce que le PC a exporté (banc.sh, EXPORT).

Le PC ne fait que les temps et une sortie de chaque mode des noyaux ; ce script refait le modèle d'origine en
FP32 par ONNX Runtime sur processeur (tools/bench/qualite/fp32.py), sur la même entrée, et dit l'écart de chaque
sortie : PSNR (crête 1), plus grand écart, valeurs non finies. Il dit aussi si la sortie sans bandes est la même,
au bit près, que celle du défaut : les bandes ne changent que l'ordre des tuiles (docs/span-native16.md).

    python3 -I tools/bench/span_native/justesse_cpu.py DOSSIER [DOSSIER…]

DOSSIER, un par modèle (banc.sh l'écrit) : modele.onnx, dims.txt (« H W »), entree.f16 (3×H×W, l'entrée telle
que les noyaux l'ont lue), et pour chaque mode <mode>.p0.f16, .p1.f16, .p2.f16 (les trois plans de la sortie,
2H×2W chacun, moins de 20 Mo par fichier). Demi-flottants petit-boutistes, comme les noyaux les écrivent.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qualite"))
import fp32  # noqa: E402  (tools/bench/qualite)

MODES = (("f16", "natif f16, bandes du lecteur"), ("b0", "natif f16, sans bandes"), ("f32", "natif f32"))


def lire(dossier):
    h, w = (int(v) for v in (dossier / "dims.txt").read_text().split())
    entree = np.fromfile(dossier / "entree.f16", dtype="<f2")
    if entree.size != 3 * h * w:
        raise SystemExit(f"{dossier}/entree.f16 : {entree.size} valeurs, {3 * h * w} attendues")
    sorties = {}
    for mode, _ in MODES:
        plans = [dossier / f"{mode}.p{c}.f16" for c in range(3)]
        if not all(p.exists() for p in plans):
            continue
        valeurs = [np.fromfile(p, dtype="<f2") for p in plans]
        if any(v.size != 4 * h * w for v in valeurs):
            raise SystemExit(f"{dossier}/{mode} : un plan n'a pas {4 * h * w} valeurs")
        sorties[mode] = np.stack(valeurs).reshape(1, 3, 2 * h, 2 * w)
    return h, w, entree.reshape(1, 3, h, w), sorties


def reference(modele, entree):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(fp32.promoted_bytes(str(modele)), options, providers=["CPUExecutionProvider"])
    nom = session.get_inputs()[0].name
    return session.run(None, {nom: entree.astype(np.float32)})[0].astype(np.float64)


def ecart(sortie, ref):
    d = sortie.astype(np.float64) - ref
    finies = np.isfinite(d)
    mse = float(np.mean(d[finies] ** 2)) if finies.any() else float("inf")
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0.0 else float("inf")
    pire = float(np.max(np.abs(d[finies]))) if finies.any() else float("nan")
    return psnr, pire, int(d.size - finies.sum())


def juger(dossier):
    h, w, entree, sorties = lire(dossier)
    if not sorties:
        raise SystemExit(f"{dossier} : aucune sortie exportée")
    ref = reference(dossier / "modele.onnx", entree)
    if ref.shape != (1, 3, 2 * h, 2 * w):
        raise SystemExit(f"{dossier} : la référence sort {ref.shape}, (1, 3, {2 * h}, {2 * w}) attendu")
    print(f"## {dossier.name} ({w}×{h} → {2 * w}×{2 * h}), contre ONNX Runtime FP32 sur processeur")
    for mode, texte in MODES:
        if mode in sorties:
            psnr, pire, non_finies = ecart(sorties[mode], ref)
            print(f"{texte} : {psnr:.2f} dB, plus grand écart {pire:.5f}, non finies {non_finies}")
    if "f16" in sorties and "b0" in sorties:
        a, b = sorties["f16"].view(np.uint16), sorties["b0"].view(np.uint16)
        differentes = int(np.count_nonzero(a != b))
        print("bandes : sortie identique au bit près à celle sans bandes" if differentes == 0
              else f"bandes : {differentes} valeurs diffèrent de la sortie sans bandes")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for chemin in argv:
        juger(Path(chemin))


if __name__ == "__main__":
    main(sys.argv[1:])
