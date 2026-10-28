#!/usr/bin/env python3
"""Fabrique des entrées RÉELLES pour le graphe « fusion » de GMFSS (1280x768).

Deux recadrages 1280x768 d'une capture d'anime (PPM) passent par le graphe « paire »
(ONNX Runtime CPU) ; les six splats se font ici (tools/gmfss/gmfss_graph.py, lecture seule) ;
les tenseurs de la fusion sortent en .npy FP16, avec la référence ONNX Runtime CPU de la fusion.

Usage : gen_entrees.py capture.ppm dossier_sortie [dx dy] [t]
  dx dy : décalage en pixels du second recadrage (mouvement réel connu, défaut 8 4)
  t     : instant interpolé (défaut 0.5)
"""
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RACINE / "tools" / "gmfss"))


def lire_ppm(chemin):
    with open(chemin, "rb") as f:
        assert f.readline().strip() == b"P6"
        ligne = f.readline()
        while ligne.startswith(b"#"):
            ligne = f.readline()
        w, h = map(int, ligne.split())
        assert int(f.readline()) == 255
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)


def main():
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    import torch

    from gmfss_graph import fusion_names, splat_all  # noqa: E402

    ppm, sortie = sys.argv[1], Path(sys.argv[2])
    dx, dy = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (8, 4)
    t = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    sortie.mkdir(parents=True, exist_ok=True)
    img = lire_ppm(ppm)
    h, w, _ = img.shape
    x0, y0 = (w - 1280) // 2 - 40, (h - 768) // 2
    a = img[y0 : y0 + 768, x0 : x0 + 1280]
    b = img[y0 + dy : y0 + dy + 768, x0 + dx : x0 + dx + 1280]
    frames = np.concatenate([a, b], axis=2).transpose(2, 0, 1)[None].astype(np.float16) / np.float16(255.0)
    modeles = Path.home() / ".local/share/ajp/ajp/modeles"
    paire = modeles / "gmfss_fortuna_base_pair-1280x768-0288b863f9b3f816.onnx"
    fusion = modeles / "gmfss_fortuna_base_fusion-1280x768-d47e821b27a101c0.onnx"
    sp = ort.InferenceSession(str(paire), providers=["CPUExecutionProvider"])
    sorties = sp.run(None, {"frames": frames})
    sorties_t = [torch.from_numpy(o) for o in sorties]
    half0, half1, warped = splat_all(sorties_t, t)
    noms = fusion_names("base")
    feeds = dict(zip(noms, [half0, half1, *warped]))
    for k, v in feeds.items():
        np.save(sortie / f"{k}.npy", v.numpy().astype(np.float16))
    sf = ort.InferenceSession(str(fusion), providers=["CPUExecutionProvider"])
    ref = sf.run(None, {k: v.numpy().astype(np.float16) for k, v in feeds.items()})[0]
    np.save(sortie / "ref_out.npy", ref)
    print("entrées :", {k: tuple(v.shape) for k, v in feeds.items()})
    print("référence :", ref.shape, ref.dtype, float(ref.min()), float(ref.max()))


if __name__ == "__main__":
    main()
