#!/usr/bin/env python3
"""Résumé du banc A/B (chantier 0.8) : lit les fichiers extraitN-tourT-V-*.tsv écrits par `kirari play --stats`
(14 colonnes) et compare A et B par extrait.

  python3 -I tools/bench/banc_resume.py DOSSIER [PASSAGE.tsv ...]
  python3 -I tools/bench/banc_resume.py --visible DEBUT FIN PASSAGE.tsv [...]
  python3 -I tools/bench/banc_resume.py --selftest

La cadence visible vient des retours horodatés du compositeur Wayland. Les temps du lecteur sont des
diagnostics distincts. DEBUT et FIN bornent les PTS en secondes, avec FIN exclue.

Un écart est acquis si |moy(B) - moy(A)| dépasse deux fois la plus grande étendue (max - min) des deux
variantes, avec au moins MIN_TOURS passages de chaque côté. Son signe et le sens de la mesure distinguent
un gain d'une régression. Avec moins de passages, l'écart reste « à confirmer ».
"""
import csv
import glob
import math
import os
import re
import statistics
import sys
from collections import Counter

COLUMNS = ["seq", "host_us", "pts_us", "fresh", "acquire_us", "render_us", "queue_wait_us", "interval_us",
           "dropped", "why", "missing", "lag_us", "uninterpolated", "display_gpu_us"]
# Colonnes du retour de présentation (6.3), absentes des relevés d'avant.
MIN_TOURS = 5  # passages de chaque côté avant de conclure à un gain (B46-15)
FEEDBACK = ["present_id", "feedback_id", "target_us", "visible_us", "vsync_k", "level"]
SCREEN_COLUMNS = ["seq", "presented_us", "refresh_ns", "msc", "flags"]
VISIBLE = ("images_neuves_s", "visible_p95_ms", "visible_p99_ms")
RUN_NAME = re.compile(r"extrait(\d+)-tour(\d+)-([AB])-[^.]+\.tsv")


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


def visible_stats(path, rows):
    """Lie chaque soumission à son propre retour, y compris les retours arrivés hors ordre."""
    path = os.fspath(path)
    sidecar = path.removesuffix(".tsv") + ".screen.tsv"
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != SCREEN_COLUMNS:
            raise ValueError(f"{sidecar} : en-tête de présentation inattendu")
        stamps = {int(row["seq"]): int(row["presented_us"]) for row in reader}
    # Un couple (surface, instant) n'a qu'une image visible. Si l'instant est partagé, garder la soumission
    # la plus récente. Les retours à 0 sont les abandons explicites, distincts des retours encore absents.
    confirmed = {}
    discarded = unanswered = 0
    for row in sorted(rows, key=lambda row: int(row.get("present_id", 0))):
        seq = int(row.get("present_id", 0))
        if not seq:
            continue
        stamp = stamps.get(seq)
        if stamp is None:
            unanswered += 1
        elif stamp == 0:
            discarded += 1
        else:
            confirmed[stamp] = row
    fresh = sorted(stamp for stamp, row in confirmed.items() if row["fresh"] == "1")
    if len(fresh) < 2:
        return None
    intervals = [b - a for a, b in zip(fresh, fresh[1:])]
    return {
        "images_neuves_s": (len(fresh) - 1) * 1e6 / (fresh[-1] - fresh[0]),
        "visible_p50_ms": percentile(intervals, 50) / 1000.0,
        "visible_p95_ms": percentile(intervals, 95) / 1000.0,
        "visible_p99_ms": percentile(intervals, 99) / 1000.0,
        "images_neuves": len(fresh),
        "repetitions_confirmees": sum(
            row["fresh"] == "0" and fresh[0] <= stamp <= fresh[-1] for stamp, row in confirmed.items()
        ),
        "abandons_compositeur": discarded,
        "sans_retour": unanswered,
    }


def summarize(rows, *, visible=None):
    fresh = [r for r in rows if r["fresh"] == "1"]
    if len(rows) < 2:
        return None
    intervals = [int(r["interval_us"]) for r in fresh if int(r["interval_us"]) > 0]
    # Écart visé − affiché, sur les présentations revenues avec une date visée.
    misses = [int(r["visible_us"]) - int(r["target_us"]) for r in rows
              if r.get("feedback_id", "0") != "0" and int(r.get("target_us", "0")) > 0]
    return {
        "sigma_vise_ms": (statistics.pstdev(misses) / 1000.0) if len(misses) > 1 else 0.0,
        "visible": visible,
        "p50_ms": percentile(intervals, 50) / 1000.0,
        "p95_ms": percentile(intervals, 95) / 1000.0,
        "p99_ms": percentile(intervals, 99) / 1000.0,
        "sigma_ms": (statistics.pstdev(intervals) / 1000.0) if len(intervals) > 1 else 0.0,
        "pertes": int(rows[-1]["dropped"]) + int(rows[-1]["missing"]),
        "acquire_ms": statistics.fmean(int(r["acquire_us"]) for r in rows) / 1000.0,
        "render_ms": statistics.fmean(int(r["render_us"]) for r in rows) / 1000.0,
    }


def verdict(a_values, b_values, *, higher_is_better):
    delta = statistics.fmean(b_values) - statistics.fmean(a_values)
    noise = max(max(a_values) - min(a_values), max(b_values) - min(b_values))
    if abs(delta) <= 2 * noise:
        return "bruit"
    if min(len(a_values), len(b_values)) < MIN_TOURS:
        return f"à confirmer (< {MIN_TOURS} passages)"
    improved = delta > 0 if higher_is_better else delta < 0
    return "gain acquis" if improved else "régression"


def read_runs(directory, paths=None):
    """Lit les passages de lecture ; les relevés .screen.tsv et .gpu.tsv ont d'autres formats."""
    if paths is None:
        paths = sorted(glob.glob(os.path.join(directory, "extrait*-tour*-*.tsv")))
    runs = []
    for path in paths:
        match = RUN_NAME.fullmatch(os.path.basename(path))
        if not match:
            continue
        rows = read_stats(path)
        summary = summarize(rows, visible=visible_stats(path, rows))
        if summary is None:
            raise ValueError(f"{path} : passage incomplet")
        extrait, tour, label = match.groups()
        runs.append((int(extrait), int(tour), label, summary))
    if not runs:
        raise ValueError(f"{directory} : aucun passage lisible")
    return runs


def compare(directory, paths=None):
    runs = read_runs(directory, paths)
    counts = Counter((extrait, tour, label) for extrait, tour, label, _ in runs)
    for extrait, tour in sorted({(extrait, tour) for extrait, tour, _, _ in runs}):
        if counts[extrait, tour, "A"] != counts[extrait, tour, "B"]:
            raise ValueError(f"extrait {extrait}, tour {tour} : passages A et B incomplets")
    found = {}
    for extrait, _, label, summary in runs:
        found.setdefault((extrait, label), []).append(summary)
    lines = []
    for extrait in sorted({key[0] for key in found}):
        lines.append(f"extrait {extrait}")
        a_runs, b_runs = found[extrait, "A"], found[extrait, "B"]
        a_visible, b_visible = ([r["visible"] for r in runs] for runs in (a_runs, b_runs))
        if all(a_visible) and all(b_visible):
            lines.append("  présentation confirmée par Wayland")
            for name in VISIBLE:
                lines.append(compare_values(name, [s[name] for s in a_visible], [s[name] for s in b_visible]))
        else:
            lines.append("  cadence visible indisponible pour un ou plusieurs passages")
        lines.append("  diagnostics du lecteur")
        for name in ("p50_ms", "p95_ms", "p99_ms", "sigma_ms", "pertes", "acquire_ms", "render_ms", "sigma_vise_ms"):
            lines.append(compare_values(name, [s[name] for s in a_runs], [s[name] for s in b_runs]))
    return "\n".join(lines)


def compare_values(name, a, b):
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    decision = verdict(a, b, higher_is_better=name == "images_neuves_s")
    return (f"  {name:16} A {mean_a:9.2f}  B {mean_b:9.2f}  écart {mean_b - mean_a:+9.2f}  "
            f"(A min/max {min(a):.2f}/{max(a):.2f})  {decision}")


def visible_window(path, start, end):
    rows = [row for row in read_stats(path) if start <= int(row["pts_us"]) < end]
    stats = visible_stats(path, rows)
    if stats is None:
        return f"{path} — cadence visible indisponible"
    return (f"{path} — {stats['images_neuves_s']:.3f} i/s, {stats['images_neuves']} images nouvelles, "
            f"{stats['repetitions_confirmees']} répétitions soumises et confirmées, "
            f"{stats['abandons_compositeur']} abandons, {stats['sans_retour']} sans retour, "
            f"intervalles p50/p95 {stats['visible_p50_ms']:.3f}/{stats['visible_p95_ms']:.3f} ms")


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
        for tour in range(1, MIN_TOURS + 1):
            for label, offset in (("A", 0), ("B", 500)):
                with open(os.path.join(tmp, f"extrait1-tour{tour}-{label}-1.tsv"), "w", encoding="utf-8") as f:
                    f.write(table(offset, 100))
        with open(os.path.join(tmp, "extrait1-tour1-A-1.gpu.tsv"), "w", encoding="utf-8") as f:
            f.write("fichier de profil\n")
        report = compare(tmp)
        assert "extrait 1" in report and "p95_ms" in report, report
        assert "régression" in report and "gain acquis" not in report, report
        # B46-15 : un B bruité n'est pas gagnant, et trois passages ne suffisent pas pour conclure.
        assert verdict([10.0] * 5, [11.0, 15.0, 7.0, 13.0, 9.0], higher_is_better=False) == "bruit"
        assert verdict([10.0] * 3, [12.0] * 3, higher_is_better=False).startswith("à confirmer")
        assert verdict([120.0] * 5, [90.0] * 5, higher_is_better=True) == "régression"
        assert verdict([12.0] * 5, [10.0] * 5, higher_is_better=False) == "gain acquis"
        for paths in ([], [os.path.join(tmp, "extrait1-tour1-A-1.tsv")]):
            try:
                compare(tmp, paths)
            except ValueError:
                pass
            else:
                raise AssertionError("un relevé vide ou incomplet doit être refusé")
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
    elif len(sys.argv) >= 5 and sys.argv[1] == "--visible":
        try:
            start, end = (round(float(arg) * 1e6) for arg in sys.argv[2:4])
            if end <= start:
                raise ValueError("la fin de la fenêtre doit suivre son début")
            for path in sys.argv[4:]:
                print(visible_window(path, start, end))
        except (OSError, ValueError) as error:
            sys.exit(str(error))
    elif len(sys.argv) >= 2:
        try:
            print(compare(sys.argv[1], sys.argv[2:] or None))
        except (OSError, ValueError) as error:
            sys.exit(str(error))
    else:
        print(__doc__)
        sys.exit(2)
