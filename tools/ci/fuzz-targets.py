#!/usr/bin/env python3
"""Les cibles libFuzzer déclarées dans tests/fuzz/CMakeLists.txt, une par ligne (C28-01).

La liste tient sur plusieurs lignes et se compose de plusieurs `set(...)` : un `sed` ligne à ligne en perdait onze.
fuzz.yml s'en sert pour construire et lancer la campagne ; check-fuzz-targets.sh vérifie qu'aucune cible
n'échappe à la campagne.
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[2]
text = (root / "tests/fuzz/CMakeLists.txt").read_text(encoding="utf-8")
targets = []
for match in re.finditer(r"set\(ajp_fuzz_(?:parser_)?targets\b([^)]*)\)", text):
    targets += match.group(1).split()
if not targets or len(set(targets)) != len(targets):
    sys.exit("fuzz-targets : liste vide ou cible en double dans tests/fuzz/CMakeLists.txt")
print("\n".join(targets))
