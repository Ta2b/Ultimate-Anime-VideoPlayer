#!/usr/bin/env python3
"""Banc de qualité des réseaux du lecteur contre leur référence FP32, sans carte graphique (ONNX Runtime).

  python3 -I tools/bench/qualite/banc.py --modeles DOSSIER [--video EPISODE.mkv] [--md rapport.md] [--json r.json]

DOSSIER est celui de tools/release/fetch-models.sh (onnx/ et rife/). Pour chaque réseau, deux questions :

  fidélité   le graphe livré (FP16) rend-il l'image de sa référence FP32 (fp32.py : mêmes poids, tout le calcul en
             FP32) ? PSNR, pire écart en codes 8 bits, 99,9e centile, ΔE_ITP ;
  vérité     l'image rendue vaut-elle la vraie ? Interpolation : l'image du milieu, retirée des entrées (scènes de
             synthèse de sources.py, ou fenêtres d'un épisode avec --video). Agrandissement : un plan 1080p réduit
             de moitié puis agrandi, comparé au plan, avec le bicubique pour repère.

Les seuils de seuils.json s'appliquent à la configuration qu'ils nomment (taille, scènes, plans) : le banc sort
en erreur si un réseau fait moins bien. --ecrire-seuils les refait depuis la mesure, avec une marge ; à ne faire
que pour une amélioration voulue ou une baisse justifiée par sa mesure.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import rapport  # noqa: E402
import sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPAN = ("2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
        "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
        "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
        "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
        # ArtCNN (P6.7), les modèles légers des petites cartes : leur image contre celle de SPAN et du bicubique.
        "2x_ArtCNN_C4F32_rgb_fp16.onnx",
        "2x_ArtCNN_C4F16_rgb_fp16.onnx")
MOTION = ("rife_v4.26.onnx", "rife_v4.25.onnx", "rife_v4.22_lite.onnx", "rife_v4.25_lite.onnx", "distil_drba_v2_lite.onnx")


def measure(path, cases, origin, provider):
    """Un réseau sur ses cas (nom, entrée, vérité) : le livré et sa référence, image par image."""
    shipped = modeles.Network(path, provider=provider)
    ref = modeles.Network(origin, provider=provider) if origin else modeles.Network(path, True, provider)
    run = (lambda n, x: n.upscale(x)) if shipped.kind == "span" else (lambda n, x: n.middle(x))
    rows = []
    for name, given, truth in cases:
        out, out32 = run(shipped, given), run(ref, given)
        rows.append({"cas": name, "fidelite": mesures.compare(out32, out), "verite": mesures.compare(truth, out),
                     "verite_fp32": mesures.compare(truth, out32)})
        print(f"  {shipped.name[:44]:44s} {name:16s} fidélité {rows[-1]['fidelite']['psnr']:6.2f} dB  "
              f"vérité {rows[-1]['verite']['psnr']:6.2f} dB", file=sys.stderr, flush=True)
    return {"modele": shipped.name, "genre": shipped.kind, "reference": "origine" if origin else "fp32 du livré",
            "ms": shipped.ms(), "ms_fp32": ref.ms(), "cas": rows, **summary(rows)}


def summary(rows):
    def col(group, key):
        return [r[group][key] for r in rows]

    return {"fidelite_psnr": float(np.mean(col("fidelite", "psnr"))),
            "fidelite_psnr_pire": min(col("fidelite", "psnr")),
            "fidelite_ecart_pire": max(col("fidelite", "ecart_pire")),
            "fidelite_ecart_p999": max(col("fidelite", "ecart_p999")),
            "fidelite_itp_p999": max(col("fidelite", "itp_p999")),
            "verite_psnr": float(np.mean(col("verite", "psnr"))),
            "verite_psnr_pire": min(col("verite", "psnr")),
            "verite_ssim": float(np.mean(col("verite", "ssim"))),
            "verite_itp": float(np.mean(col("verite", "itp_moyen"))),
            "verite_psnr_fp32": float(np.mean(col("verite_fp32", "psnr")))}


def baseline(cases):
    """Le repère de l'agrandissement : le bicubique ×2, mesuré contre la vraie image."""
    rows = [mesures.compare(truth, np.clip(sources.bicubic2(given), 0.0, 1.0)) for _, given, truth in cases]
    return {"psnr": float(np.mean([r["psnr"] for r in rows])), "ssim": float(np.mean([r["ssim"] for r in rows])),
            "itp": float(np.mean([r["itp_moyen"] for r in rows]))}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modeles", required=True, help="dossier de fetch-models.sh (onnx/ et rife/)")
    p.add_argument("--calibration", default=os.path.join(ROOT, "data", "calibration", "calibration.mkv"))
    p.add_argument("--video", help="un épisode : le mouvement réel remplace les scènes de synthèse")
    p.add_argument("--taille", default="1280x720", help="taille des images d'interpolation")
    p.add_argument("--scenes", type=int, default=8, help="scènes de synthèse, ou fenêtres de --video")
    p.add_argument("--plans", type=int, default=6, help="plans fixes agrandis par SPAN (18 au plus)")
    p.add_argument("--images", type=int, default=2000, help="images lues au plus dans --video")
    p.add_argument("--seulement", choices=("span", "mouvement"), help="ne mesurer qu'une famille")
    p.add_argument("--reseau", action="append", default=[], help="un fichier ONNX de plus (répétable)")
    p.add_argument("--origine", action="append", default=[], metavar="NOM=FICHIER",
                   help="le graphe FP32 d'origine d'un réseau, à la place du FP32 tiré du livré")
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seuils", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "seuils.json"))
    p.add_argument("--ecrire-seuils", action="store_true")
    p.add_argument("--json")
    p.add_argument("--md")
    return p.parse_args(argv)


def build_cases(a, size):
    """Les cas de chaque famille : (nom, entrée, vérité)."""
    plans = sources.stills(a.calibration, 18)
    if not plans:
        raise SystemExit(f"{a.calibration} : aucun plan lu")
    span = [(f"plan-{i + 1}", sources.halve(p), p) for i, p in enumerate(plans[:a.plans])]
    if a.video:
        wins = sources.windows(sources.frames(a.video, size, a.images), a.scenes)
        if not wins:
            raise SystemExit("aucune fenêtre de trois images différentes dans la vidéo")
    else:
        wins = sources.scenes(plans, size, a.scenes)
    return span, [(name, win, win[1]) for name, win in wins]


def main(argv=None):
    a = parse(argv)
    size = tuple(int(v) for v in a.taille.split("x"))
    origins = dict(o.split("=", 1) for o in a.origine)
    span_cases, motion_cases = build_cases(a, size)
    config = {"taille": a.taille, "scenes": len(motion_cases), "plans": len(span_cases),
              "source": "video" if a.video else "synthese"}
    paths = [] if a.seulement == "mouvement" else [os.path.join(a.modeles, "onnx", n) for n in SPAN]
    paths += [] if a.seulement == "span" else [os.path.join(a.modeles, "rife", n) for n in MOTION]
    paths += a.reseau
    results = []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"{path} : absent (tools/release/fetch-models.sh le télécharge)")
        kind = modeles.Network(path, provider=a.provider).kind
        cases = span_cases if kind == "span" else motion_cases
        results.append(measure(path, cases, origins.get(os.path.basename(path)), a.provider))
    report = {"config": config, "bicubique": baseline(span_cases) if a.seulement != "mouvement" else None,
              "reseaux": results}
    print(rapport.table(report))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(rapport.markdown(report))
    if a.ecrire_seuils:
        rapport.write_gates(a.seuils, report)
        print(f"seuils écrits : {a.seuils}")
        return 0
    failures = rapport.check_gates(a.seuils, report)
    for line in failures:
        print(f"SEUIL : {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
