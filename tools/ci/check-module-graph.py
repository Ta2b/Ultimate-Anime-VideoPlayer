#!/usr/bin/env python3
"""Le graphe des modules (P5.24, P5.26) : relu des appels ajp_add_module de src/, sans configurer CMake.

Refusé :
- un cycle entre modules (DEPS et PRIVATE_DEPS) ;
- une dépendance déclarée dont aucun fichier du module n'inclut un en-tête (render → subs, par exemple) ;
- une inclusion d'un autre module qui n'est pas déclarée, directement ou par une dépendance publique.
Les fichiers d'un module : ses SOURCES et les en-têtes du même nom ; un en-tête seul se rattache au module de
son dossier qui n'a pas de sources (module d'interface), sinon au module qui porte le dossier.

gpu et prepare (P5.24) : ajp::gpu est l'interface des moteurs (backend.h, kernel_ops.h), sans source ; prepare
la lit, et les backends chargés à l'exécution (vulkan/backend, cuda) lient prepare. Le dossier gpu/ porte les
deux côtés, le graphe des modules n'a pas de cycle, et ce script le garde ainsi.
"""
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")


def strip_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def modules():
    found = {}
    for base, _, files in os.walk(SRC):
        if "CMakeLists.txt" not in files:
            continue
        text = strip_comments(open(os.path.join(base, "CMakeLists.txt"), encoding="utf-8").read())
        for match in re.finditer(r"ajp_add_module\(", text):
            depth, i = 1, match.end()
            while depth and i < len(text):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            words = re.findall(r'"[^"]*"|[^\s()]+', text[match.end():i - 1])
            name, section = words[0], None
            mod = {"dir": base, "sources": [], "deps": [], "public": []}
            for word in words[1:]:
                if word in ("SOURCES", "DEPS", "PRIVATE_DEPS"):
                    section = word
                elif section == "SOURCES" and not word.startswith(('"$', "$")):
                    mod["sources"].append(os.path.normpath(os.path.join(base, word.strip('"'))))
                elif section in ("DEPS", "PRIVATE_DEPS"):
                    dep = re.search(r"ajp::(\w+)", word)
                    if dep and not word.startswith("ajp::dep::"):
                        mod["deps"].append(dep.group(1))
                        if section == "DEPS":
                            mod["public"].append(dep.group(1))
            found[name] = mod
    return found


def owner_of(path, mods):
    """Le module d'un fichier de src/ : celui qui le liste, sinon celui de son dossier le plus proche."""
    stem = os.path.splitext(path)[0]
    for name, mod in mods.items():
        if any(os.path.splitext(s)[0] == stem for s in mod["sources"]):
            return name
    best = None
    for name, mod in mods.items():
        d = mod["dir"]
        if path.startswith(d + os.sep) and (best is None or len(d) > len(mods[best]["dir"]) or
                                            (d == mods[best]["dir"] and not mod["sources"])):
            best = name
    return best


def reachable(name, mods):
    """Ce que le module voit : lui, ses dépendances, et les dépendances publiques de celles-ci."""
    seen, todo = {name}, list(mods[name]["deps"])
    while todo:
        dep = todo.pop()
        if dep in seen or dep not in mods:
            continue
        seen.add(dep)
        todo.extend(mods[dep]["public"])
    return seen


def main():
    mods = modules()
    errors = []
    # Cycles.
    state = {}

    def visit(name, stack):
        state[name] = 1
        for dep in mods[name]["deps"]:
            if dep not in mods:
                continue
            if state.get(dep) == 1:
                errors.append("cycle : " + " → ".join(stack[stack.index(dep):] + [dep]))
            elif dep not in state:
                visit(dep, stack + [dep])
        state[name] = 2

    for name in sorted(mods):
        if name not in state:
            visit(name, [name])
    # Inclusions.
    for name, mod in sorted(mods.items()):
        files = set(mod["sources"])
        for s in mod["sources"]:
            header = os.path.splitext(s)[0] + ".h"
            if os.path.exists(header):
                files.add(header)
        if not mod["sources"]:
            files = {os.path.join(mod["dir"], f) for f in os.listdir(mod["dir"]) if f.endswith(".h")}
        # Les en-têtes internes du module sans source à eux (prepare/protobuf_fields.h) : leurs
        # inclusions comptent aussi, suivies d'en-tête en en-tête.
        pending = list(files)
        while pending:
            path = pending.pop()
            if not os.path.exists(path):
                continue
            for inc in re.findall(r'^#include "([^"]+)"', open(path, encoding="utf-8").read(), re.M):
                target = os.path.normpath(os.path.join(SRC, inc))
                if target not in files and os.path.exists(target) and owner_of(target, mods) == name:
                    files.add(target)
                    pending.append(target)
        used = set()
        visible = reachable(name, mods)
        for path in sorted(files):
            if not os.path.exists(path):
                continue
            for inc in re.findall(r'^#include "([^"]+)"', open(path, encoding="utf-8").read(), re.M):
                target = os.path.join(SRC, inc)
                if not os.path.exists(target):
                    continue
                dep = owner_of(os.path.normpath(target), mods)
                if dep is None or dep == name:
                    continue
                used.add(dep)
                same_dir = mods[dep]["dir"] == mod["dir"]
                if dep not in visible and not same_dir:
                    errors.append(f"{os.path.relpath(path, ROOT)} inclut {inc} : ajp::{dep} non déclaré")
        used_dirs = {mods[d]["dir"] for d in used}
        for dep in mod["deps"]:
            if dep in mods and dep not in used and mods[dep]["dir"] not in used_dirs:
                errors.append(f"ajp::{name} déclare ajp::{dep} sans en inclure un en-tête")
    for line in errors:
        print(line)
    if errors:
        print("::error::graphe des modules : corriger les DEPS de ajp_add_module (tools/ci/check-module-graph.py)")
        return 1
    print(f"graphe des modules : {len(mods)} modules, sans cycle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
