#!/usr/bin/env python3
"""Complexité des fonctions de src/ (P5.15) : sous CCN 25 et 120 lignes, mesurées par lizard.

Les fonctions qui dépassent déjà sont listées dans complexity-baseline.txt avec leur mesure : elles ne peuvent
que baisser. Une nouvelle fonction au-delà des bornes, ou une fonction listée qui grossit, est refusée. Une
fonction découpée sous les bornes se retire de la liste (--update la réécrit).
"""
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE = os.path.join(os.path.dirname(__file__), "complexity-baseline.txt")
MAX_CCN, MAX_LENGTH = 25, 120


def measure():
    out = subprocess.run([sys.executable, "-m", "lizard", "--csv", "-l", "cpp", "src"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        # NLOC, CCN, token, PARAM, length, location, file, function, long_name, start, end
        cells = next(iter(__import__("csv").reader([line])))
        if len(cells) < 9:
            continue
        ccn, length, path, name = int(cells[1]), int(cells[4]), cells[6], cells[7]
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
