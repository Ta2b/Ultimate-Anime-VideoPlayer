#!/usr/bin/env python3
"""Complexité des fonctions de src/ (P5.15) : sous CCN 25 et 120 lignes, mesurées par lizard.

Les fonctions qui dépassent déjà sont listées dans complexity-baseline.txt avec leur mesure : elles ne peuvent
que baisser. Une nouvelle fonction au-delà des bornes, ou une fonction listée qui grossit, est refusée. Une
fonction découpée sous les bornes se retire de la liste (--update la réécrit).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE = os.path.join(os.path.dirname(__file__), "complexity-baseline.txt")
MAX_CCN, MAX_LENGTH = 25, 120


def lizard_rows(root, targets):
    out = subprocess.run([sys.executable, "-m", "lizard", "--csv", "-l", "cpp", *targets], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        # NLOC, CCN, token, PARAM, length, location, file, function, long_name, start, end
        cells = next(iter(csv.reader([line])))
        if len(cells) >= 9:
            yield int(cells[1]), int(cells[4]), cells[6], cells[7]


def cuda_rows():
    """Les .cu et .cuh de src/ (les noyaux TensorRT), que lizard ne lit pas sous leur extension : copiés en .cpp
    dans un dossier temporaire qui garde les chemins, mesurés, puis renommés comme à l'origine (B28-06)."""
    files = subprocess.run(["git", "ls-files", "-z", "--", "src/*.cu", "src/*.cuh"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split("\0")
    files = [f for f in files if f]
    if not files:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            dest = os.path.join(tmp, f + ".cpp")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(os.path.join(ROOT, f), dest)
        for ccn, length, path, name in lizard_rows(tmp, ["src"]):
            yield ccn, length, path.removesuffix(".cpp"), name


def measure():
    found = {}
    for ccn, length, path, name in [*lizard_rows(ROOT, ["src"]), *cuda_rows()]:
        if ccn > MAX_CCN or length > MAX_LENGTH:
            key = f"{os.path.relpath(path, ROOT) if os.path.isabs(path) else path}\t{name}"
            prev = found.get(key)
            found[key] = (max(ccn, prev[0]), max(length, prev[1])) if prev else (ccn, length)
    return found


def main():
    found = measure()
    if "--update" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8") as f:
            f.write("# fichier\tfonction\tCCN\tlignes : fonctions au-delà de CCN 25 ou 120 lignes, qui ne peuvent que "
                    "baisser (check-complexity.py)\n")
            for key in sorted(found):
                f.write(f"{key}\t{found[key][0]}\t{found[key][1]}\n")
        print(f"complexité : {len(found)} fonctions au-delà des bornes, liste réécrite")
        return 0
    known = {}
    for line in open(BASELINE, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        path, name, ccn, length = line.rstrip("\n").split("\t")
        known[f"{path}\t{name}"] = (int(ccn), int(length))
    errors = []
    for key, (ccn, length) in sorted(found.items()):
        path, name = key.split("\t")
        if key not in known:
            errors.append(f"{path} : {name} : CCN {ccn}, {length} lignes (au plus {MAX_CCN} et {MAX_LENGTH})")
        elif ccn > known[key][0] or length > known[key][1]:
            errors.append(f"{path} : {name} grossit : CCN {known[key][0]} → {ccn}, {known[key][1]} → {length} lignes")
    gone = [k for k in known if k not in found]
    for line in errors:
        print(line)
    if gone:
        print(f"complexité : {len(gone)} fonctions passées sous les bornes, à retirer de la liste (--update)")
    if errors:
        print("::error::complexité : découper la fonction (tools/ci/check-complexity.py)")
        return 1
    print(f"complexité : {len(found)} fonctions au-delà des bornes, aucune nouvelle, aucune qui grossit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
