#!/usr/bin/env python3
"""Les .cpp de src/ à analyser pour une liste de fichiers changés (clang-tidy-src.sh).

Entrée standard : les fichiers suivis de src/ (en-têtes et .cpp), séparés par des octets nuls. Arguments : la
racine du dépôt et le fichier qui liste, un par ligne, les chemins changés. Sortie : un .cpp par ligne, ceux qui
ont changé et ceux qui incluent un en-tête changé, directement ou par d'autres en-têtes. Un #include "…" se lit
relatif à src/ (« core/log.h ») ou au dossier de l'incluant (« ajp_vk_backend_spirv.h »).
"""
import os
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
HEADER_EXT = (".h", ".hpp", ".inc", ".cuh")


def main() -> int:
    root, changed_path = sys.argv[1], sys.argv[2]
    tracked = [f for f in sys.stdin.read().split("\0") if f]
    with open(changed_path, encoding="utf-8") as f:
        changed = {line.strip() for line in f if line.strip()}

    # Qui inclut quoi : pour chaque fichier, les fichiers suivis qu'il inclut.
    known = set(tracked)
    includers: dict[str, set[str]] = {}
    for path in tracked:
        try:
            with open(os.path.join(root, path), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        here = os.path.dirname(path)
        for name in INCLUDE.findall(text):
            for candidate in (os.path.normpath(os.path.join("src", name)), os.path.normpath(os.path.join(here, name))):
                if candidate in known:
                    includers.setdefault(candidate, set()).add(path)
                    break

    # Les en-têtes changés, puis tout ce qui les inclut, de proche en proche.
    reached = {f for f in changed if f in known}
    todo = [f for f in reached if f.endswith(HEADER_EXT)]
    while todo:
        header = todo.pop()
        for user in includers.get(header, ()):
            if user not in reached:
                reached.add(user)
                if user.endswith(HEADER_EXT):
                    todo.append(user)

    for path in sorted(reached):
        if path.endswith(".cpp"):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
