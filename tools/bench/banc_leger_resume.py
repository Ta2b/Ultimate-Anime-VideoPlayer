#!/usr/bin/env python3
"""Résumé du banc des petites cartes (tools/bench/banc_leger.sh) : par extrait et par variante, la moyenne des tours.

  python3 -I tools/bench/banc_leger_resume.py DOSSIER
  python3 -I tools/bench/banc_leger_resume.py --selftest

Colonnes : images neuves par seconde, intervalle p95 entre deux images, pertes (images sautées ou manquées), temps
GPU du réseau par passe (Vulkan : « de la première estampille à la dernière » de --profile-dispatches ; TensorRT :
la médiane de chaque étape du .gpu.tsv), et la précision que le journal dit avoir prise pour SPAN (INT8 ou non).
Le verdict de l'Auto se lit avec les captures (capture-N-variante.ppm) : une variante légère n'entre dans
l'Auto que si elle rend mieux qu'« econome » sans perdre d'images.
"""
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banc_resume import read_stats, summarize  # noqa: E402

PASS = re.compile(r"extrait(\d+)-tour(\d+)-([a-z0-9-]+)\.tsv$")
VULKAN = re.compile(r"de la première estampille à la dernière : ([0-9.]+) ms")
# La variante INT8 prise (src/app/play/chain_setup.cpp, usable_variant) : son fichier et ses mesures ; un refus
# écrit sa raison à la place.
INT8 = re.compile(r"SPAN INT8 : \S+-int8-[0-9a-f]{16}\.onnx \(")


def vulkan_ms(log_text):
    """Le temps GPU par passe de chaque réseau déchargé pendant le passage (médianes de --profile-dispatches)."""
    return [float(v) for v in VULKAN.findall(log_text)]


def tensorrt_ms(path):
    """{étape: médiane en ms} du .gpu.tsv, vide sans lui."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 5:
                out[fields[0]] = float(fields[3])
    return out


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def collect(directory):
    """{(extrait, variante): [passage]}, chaque passage un dict de mesures."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "extrait*-tour*-*.tsv"))):
        if path.endswith(".gpu.tsv"):
            continue
        match = PASS.search(os.path.basename(path))
        if not match:
            continue
        summary = summarize(read_stats(path))
        if not summary:
            continue
        base = path[:-len(".tsv")]
        log = read_text(base + ".ajp.log") + read_text(base + ".log")
        summary["vulkan_ms"] = vulkan_ms(log)
        summary["tensorrt_ms"] = tensorrt_ms(base + ".gpu.tsv")
        summary["int8"] = INT8.search(log) is not None
        found.setdefault((int(match[1]), match[3]), []).append(summary)
    return found


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def report(found):
    lines = []
    for extrait in sorted({key[0] for key in found}):
        lines.append(f"extrait {extrait}")
        lines.append(f"  {'variante':14} {'tours':>5} {'neuves/s':>9} {'p95 ms':>8} {'pertes':>7} {'GPU ms':>8}  remarque")
        for (e, label), runs in sorted(found.items()):
            if e != extrait:
                continue
            gpu = [v for r in runs for v in r["vulkan_ms"]]
            stages = {}
            for r in runs:
                for name, ms in r["tensorrt_ms"].items():
                    stages.setdefault(name, []).append(ms)
            notes = []
            if stages:
                notes.append("TensorRT : " + ", ".join(f"{n} {mean(v):.2f}" for n, v in sorted(stages.items())))
            if label == "span-int8":
                notes.append("INT8 pris" if all(r["int8"] for r in runs) else "INT8 absent du journal : FP16 ?")
            lines.append(f"  {label:14} {len(runs):5d} {mean([r['images_neuves_s'] for r in runs]):9.2f} "
                         f"{mean([r['p95_ms'] for r in runs]):8.2f} {mean([r['pertes'] for r in runs]):7.1f} "
                         f"{f'{mean(gpu):.2f}' if gpu else '-':>8}  {' ; '.join(notes)}")
    return "\n".join(lines) if lines else "aucun passage lisible"


def selftest():
    import tempfile

    from banc_resume import COLUMNS

    def table(interval):
        rows = ["\t".join(COLUMNS)]
        for i in range(50):
            rows.append("\t".join(map(str, [i + 1, 1_000_000 + i * interval, i * interval, 1, 200, 900, 0, interval,
                                           0, "i", 0, 0, 0, 0])))
        return "\n".join(rows) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        for tour in (1, 2):
            for label, ms in (("econome", None), ("artcnn-c4f16", 3.5), ("span-int8", None)):
                base = os.path.join(tmp, f"extrait1-tour{tour}-{label}")
                with open(base + ".tsv", "w", encoding="utf-8") as f:
                    f.write(table(41_667))
                with open(base + ".ajp.log", "w", encoding="utf-8") as f:
                    if ms is not None:
                        f.write(f"total des dispatchs : 3.0 ms ; de la première estampille à la dernière : {ms} ms\n")
                if label == "span-int8":
                    with open(base + ".ajp.log", "w", encoding="utf-8") as f:
                        if tour == 1:
                            f.write("SPAN INT8 : 2x_b5f48-int8-0123456789abcdef.onnx (PSNR 45.1 dB)\n")
                        else:
                            f.write("SPAN INT8 : refusé, PSNR moyen 30.2 dB sous 42.0\n")
                    with open(base + ".gpu.tsv", "w", encoding="utf-8") as f:
                        f.write("etape\tmesures\ttotal_ms\tmediane_ms\tp95_ms\nrôle upscale\t10\t50.0\t5.0\t6.0\n")
        text = report(collect(tmp))
        assert "artcnn-c4f16" in text and " 3.50 " in text, text
        assert "TensorRT : rôle upscale 5.00" in text, text
        assert "INT8 absent du journal" in text, text
        assert " 41.67 " in text, text  # 41,667 ms entre deux images neuves
    print("OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 2:
        print(report(collect(sys.argv[1])))
    else:
        print(__doc__)
        sys.exit(2)
