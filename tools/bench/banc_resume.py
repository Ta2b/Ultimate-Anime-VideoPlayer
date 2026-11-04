#!/usr/bin/env python3
"""Résumé du banc A/B (chantier 0.8) : lit les fichiers extraitN-tourT-V-*.tsv écrits par `ajp play --stats`
(14 colonnes) et compare A et B par extrait.

  python3 -I tools/bench/banc_resume.py DOSSIER
  python3 -I tools/bench/banc_resume.py --selftest

Gain acquis si |moy(B) - moy(A)| > 2 × (max(A) - min(A)), avec A mesuré plusieurs fois ; sinon « bruit ».
"""
import glob
import math
import os
import re
import statistics
import sys

COLUMNS = ["seq", "host_us", "pts_us", "fresh", "acquire_us", "render_us", "queue_wait_us", "interval_us",
           "dropped", "why", "missing", "lag_us", "uninterpolated", "display_gpu_us"]
# Colonnes du retour de présentation (6.3), absentes des relevés d'avant.
FEEDBACK = ["present_id", "feedback_id", "target_us", "visible_us", "vsync_k", "level"]


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def read_stats(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header not in (COLUMNS, COLUMNS + FEEDBACK):
            raise ValueError(f"{path} : en-tête inattendu ({len(header)} colonnes, {len(COLUMNS)} attendues)")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            rows.append(dict(zip(header, fields)))
    return rows


def summarize(rows):
    fresh = [r for r in rows if r["fresh"] == "1"]
    if len(rows) < 2:
        return None
    span_s = (int(rows[-1]["host_us"]) - int(rows[0]["host_us"])) / 1e6
    intervals = [int(r["interval_us"]) for r in fresh if int(r["interval_us"]) > 0]
    # Écart visé − affiché, sur les présentations revenues avec une date visée.
    misses = [int(r["visible_us"]) - int(r["target_us"]) for r in rows
              if r.get("feedback_id", "0") != "0" and int(r.get("target_us", "0")) > 0]
    return {
        "sigma_vise_ms": (statistics.pstdev(misses) / 1000.0) if len(misses) > 1 else 0.0,
        "images_neuves_s": len(fresh) / span_s if span_s > 0 else 0.0,
        "p50_ms": percentile(intervals, 50) / 1000.0,
        "p95_ms": percentile(intervals, 95) / 1000.0,
        "p99_ms": percentile(intervals, 99) / 1000.0,
        "sigma_ms": (statistics.pstdev(intervals) / 1000.0) if len(intervals) > 1 else 0.0,
        "pertes": int(rows[-1]["dropped"]) + int(rows[-1]["missing"]),
        "acquire_ms": statistics.fmean(int(r["acquire_us"]) for r in rows) / 1000.0,
        "render_ms": statistics.fmean(int(r["render_us"]) for r in rows) / 1000.0,
    }


def verdict(a_values, b_values):
    if not a_values or not b_values:
        return "?"
    mean_a, mean_b = statistics.fmean(a_values), statistics.fmean(b_values)
    noise = max(a_values) - min(a_values)
    return "gain acquis" if abs(mean_b - mean_a) > 2 * noise else "bruit"


def compare(directory):
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "extrait*-tour*-*.tsv"))):
        match = re.match(r"extrait(\d+)-tour(\d+)-([AB])-", os.path.basename(path))
        if not match:
            continue
        summary = summarize(read_stats(path))
        if summary:
            found.setdefault((int(match[1]), match[3]), []).append(summary)
    lines = []
    for extrait in sorted({key[0] for key in found}):
        lines.append(f"extrait {extrait}")
        for name in ("images_neuves_s", "p50_ms", "p95_ms", "p99_ms", "sigma_ms", "pertes", "acquire_ms",
                     "render_ms", "sigma_vise_ms"):
            a = [s[name] for s in found.get((extrait, "A"), [])]
            b = [s[name] for s in found.get((extrait, "B"), [])]
            if not a or not b:
                continue
            lines.append(f"  {name:16} A {statistics.fmean(a):9.2f}  B {statistics.fmean(b):9.2f}  "
                         f"écart {statistics.fmean(b) - statistics.fmean(a):+9.2f}  "
                         f"(A min/max {min(a):.2f}/{max(a):.2f})  {verdict(a, b)}")
    return "\n".join(lines) if lines else "aucun passage lisible"


def selftest():
    def table(offset, count):
        rows = ["\t".join(COLUMNS)]
        for i in range(count):
            host = 1_000_000 + i * 41_667
            rows.append("\t".join(map(str, [i + 1, host, i * 41_667, 1, 200, 900 + offset, 0,
                                           41_667 + (i % 3) * offset, 0, "i", 0, 0, 0, 0])))
        return "\n".join(rows) + "\n"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for tour in (1, 2, 3):
            for label, offset in (("A", 0), ("B", 500)):
                with open(os.path.join(tmp, f"extrait1-tour{tour}-{label}-1.tsv"), "w", encoding="utf-8") as f:
                    f.write(table(offset, 100))
        report = compare(tmp)
        assert "extrait 1" in report and "p95_ms" in report, report
        assert "gain acquis" in report, report  # B est nettement plus lent et A n'a aucun bruit
        assert percentile([1, 2, 3, 4], 50) == 2.5
        bad = os.path.join(tmp, "mauvais.tsv")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("a\tb\n")
        try:
            read_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("un en-tête à deux colonnes doit être refusé")
    print("banc_resume : auto-test réussi")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 2:
        print(compare(sys.argv[1]))
    else:
        print(__doc__)
        sys.exit(2)
