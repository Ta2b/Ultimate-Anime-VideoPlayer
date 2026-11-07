#!/usr/bin/env python3
"""Parité Windows contre Linux (P7.10) : compare deux dossiers de banc.sh, l'un joué sous Linux, l'autre
sous Windows, sur le même PC (double démarrage), avec les mêmes extraits et les mêmes options.

  python3 -I tools/bench/parite.py DOSSIER-LINUX DOSSIER-WINDOWS
  python3 -I tools/bench/parite.py --selftest

Seuils (docs/parite-windows-linux.md) : par extrait, le 99e centile du temps par image sous Windows ne dépasse
pas celui de Linux de plus de 5 %, et Windows ne perd pas plus d'images ; le temps médian, le rendu et
l'acquisition restent à 10 % près. Sortie 1 dès qu'un seuil est franchi, pour bloquer une publication.
"""
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banc_resume import read_stats, summarize  # noqa: E402

# (mesure, marge relative admise pour Windows ; None : aucune de plus que Linux)
THRESHOLDS = [("p99_ms", 0.05), ("pertes", None), ("p50_ms", 0.10), ("render_ms", 0.10), ("acquire_ms", 0.10)]


def gather(directory):
    """Moyenne de chaque mesure par extrait, sur tous les tours et les deux variantes."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "extrait*-tour*-*.tsv"))):
        if path.endswith(".gpu.tsv"):
            continue
        match = re.match(r"extrait(\d+)-tour\d+-[AB]-", os.path.basename(path))
        if not match:
            continue
        summary = summarize(read_stats(path))
        if summary:
            found.setdefault(int(match[1]), []).append(summary)
    return {extrait: {name: statistics.fmean(s[name] for s in runs) for name, _ in THRESHOLDS}
            for extrait, runs in found.items()}


def compare(linux_dir, windows_dir):
    linux, windows = gather(linux_dir), gather(windows_dir)
    lines, failed = [], False
    common = sorted(set(linux) & set(windows))
    if not common:
        return "aucun extrait commun aux deux dossiers", True
    for extrait in sorted(set(linux) ^ set(windows)):
        lines.append(f"extrait {extrait} : joué sur un seul système, ignoré")
    for extrait in common:
        lines.append(f"extrait {extrait}")
        for name, margin in THRESHOLDS:
            lin, win = linux[extrait][name], windows[extrait][name]
            limit = lin if margin is None else lin * (1 + margin)
            ok = win <= limit + 1e-9
            failed |= not ok
            lines.append(f"  {name:11} Linux {lin:9.2f}  Windows {win:9.2f}  limite {limit:9.2f}  "
                         f"{'ok' if ok else 'ÉCART'}")
    lines.append("parité tenue" if not failed else "parité NON tenue")
    return "\n".join(lines), failed


def selftest():
    import tempfile
    from banc_resume import COLUMNS

    def write(directory, interval_extra, dropped):
        rows = ["\t".join(COLUMNS)]
        for i in range(200):
            interval = 41_667 + (interval_extra if i >= 195 else 0)
            rows.append("\t".join(map(str, [i + 1, 1_000_000 + i * 41_667, i * 41_667, 1, 200, 900, 0, interval,
                                           dropped if i == 199 else 0, "i", 0, 0, 0, 0])))
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "extrait1-tour1-A-1.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        write(os.path.join(tmp, "linux"), 0, 0)
        write(os.path.join(tmp, "win-ok"), 0, 0)
        write(os.path.join(tmp, "win-lent"), 400_000, 0)
        write(os.path.join(tmp, "win-pertes"), 0, 3)
        report, failed = compare(os.path.join(tmp, "linux"), os.path.join(tmp, "win-ok"))
        assert not failed and "parité tenue" in report, report
        report, failed = compare(os.path.join(tmp, "linux"), os.path.join(tmp, "win-lent"))
        assert failed and "p99_ms" in report, report
        report, failed = compare(os.path.join(tmp, "linux"), os.path.join(tmp, "win-pertes"))
        assert failed and "ÉCART" in report, report
        _, failed = compare(os.path.join(tmp, "linux"), os.path.join(tmp, "vide"))
        assert failed
    print("parite : auto-test réussi")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        text, bad = compare(sys.argv[1], sys.argv[2])
        print(text)
        sys.exit(1 if bad else 0)
    else:
        print(__doc__)
        sys.exit(2)
