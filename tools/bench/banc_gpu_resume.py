#!/usr/bin/env python3
"""Résumé du profil GPU d'un banc A/B (tools/bench/banc.sh) : lit les extraitN-tourT-V-*.gpu.tsv qu'écrit
`ajp play --stats` à côté de ses relevés et donne, par étape (« RIFE · rife_v4.25 », SPAN…), la médiane de chaque
passage, A puis B, et l'écart des médianes de B contre A.

  python3 -I tools/bench/banc_gpu_resume.py DOSSIER

Avec --one-lane, chaque étape a sa file : la médiane d'une étape est alors le coût propre de son moteur
(docs/rife-fusionne.md, docs/rife-candidats.md). Le nom d'une étape porte le modèle chargé, donc A et B n'ont pas
le même nom pour RIFE : les étapes se rangent par famille, le texte avant « · ».
"""
import glob
import os
import re
import statistics
import sys

NAME = re.compile(r"extrait(\d+)-tour(\d+)-([AB])-\d+\.gpu\.tsv$")


def read(path):
    """Les étapes d'un .gpu.tsv : nom -> médiane en ms."""
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = header.index("mediane_ms")
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) > col:
                out[cells[0]] = float(cells[col])
    return out


def family(name):
    return name.split("·")[0].strip()


def collect(folder):
    """{extrait: {famille: {"A": [(tour, étape, médiane)], "B": [...]}}}"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.gpu.tsv"))):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue  # préchauffage
        clip, tour, label = int(m.group(1)), int(m.group(2)), m.group(3)
        for stage, median in read(path).items():
            runs.setdefault(clip, {}).setdefault(family(stage), {"A": [], "B": []})[label].append(
                (tour, stage, median))
    return runs


def report(runs):
    lines = []
    for clip in sorted(runs):
        for fam, sides in sorted(runs[clip].items()):
            a = [v for _, _, v in sides["A"]]
            b = [v for _, _, v in sides["B"]]
            if not a or not b:
                continue
            names = " / ".join(sorted({s for _, s, _ in sides["A"]} | {s for _, s, _ in sides["B"]}))
            ma, mb = statistics.median(a), statistics.median(b)
            lines.append(f"extrait {clip} · {fam} ({names})")
            lines.append("  A : " + " ".join(f"{v:.2f}" for v in a) + f"  médiane {ma:.2f} ms")
            lines.append("  B : " + " ".join(f"{v:.2f}" for v in b) + f"  médiane {mb:.2f} ms")
            lines.append(f"  B contre A : {(mb / ma - 1) * 100:+.1f} %" if ma > 0 else "  A nul")
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    text = report(collect(argv[1]))
    print(text or "aucun .gpu.tsv de passage compté dans ce dossier")
    return 0 if text else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
