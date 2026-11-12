#!/usr/bin/env python3
"""Le retard des dépendances (P1.27) : la baseline de vcpkg.json face au master de vcpkg, et chaque
entrée DEPS de tools/deps/linux-from-source.sh face à la dernière étiquette de son dépôt.

  tools/ci/deps_lag.py [--bump]

Écrit un tableau Markdown sur la sortie ; code 0 si rien ne retarde, 1 sinon. --bump pose aussi la
baseline de vcpkg.json sur le master de vcpkg (la tâche planifiée en fait une pull request).
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VCPKG = "https://github.com/microsoft/vcpkg"
# Les ports que le lecteur tire de vcpkg sous Windows (vcpkg.json et leurs dépendances visibles).
PORTS = ["sdl3", "imgui", "freetype", "harfbuzz", "fribidi", "ffmpeg", "dav1d", "spdlog", "tomlplusplus",
         "catch2", "vulkan-headers", "shaderc", "zlib", "libunibreak"]


def ls_remote(url, *refs):
    out = subprocess.run(["git", "ls-remote", url, *refs], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines()]


def port_version(commit, port):
    url = f"https://raw.githubusercontent.com/microsoft/vcpkg/{commit}/ports/{port}/vcpkg.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except OSError:
        return None
    for key in ("version", "version-semver", "version-string", "version-date"):
        if key in data:
            return data[key]
    return None


def shipped_version(manifest, port, commit):
    """La version que Windows livre : l'« overrides » de vcpkg.json d'abord, puis le port à la baseline (C28-11).

    La pull request hebdomadaire avance la baseline tandis que les overrides gardent la version fixée : lire la
    baseline seule comparait le FFmpeg de master et non celui qui part chez l'utilisateur."""
    for override in manifest.get("overrides", []):
        if override["name"] == port:
            return override["version"]
    return port_version(commit, port)


def version_key(text):
    return [int(n) for n in re.findall(r"\d+", text)]


def latest_tag(url, current):
    # Les étiquettes du même motif que la référence actuelle : « v1.2.3 », « release-3.4.18 »…
    prefix = re.match(r"^\D*", current).group(0)
    if not re.search(r"\d", current):
        return None
    tags = [ref.removeprefix("refs/tags/") for _, ref in ls_remote(url, f"refs/tags/{prefix}*")
            if not ref.endswith("^{}")]
    pattern = re.compile(re.escape(prefix) + r"[\d._]+$")
    stable = [t for t in tags if pattern.match(t)]
    return max(stable, key=version_key) if stable else None


def main():
    bump = "--bump" in sys.argv
    manifest_path = ROOT / "vcpkg.json"
    manifest = json.loads(manifest_path.read_text())
    baseline = manifest["builtin-baseline"]
    master = ls_remote(VCPKG, "refs/heads/master")[0][0]
    rows = []
    for port in PORTS:
        have, latest = shipped_version(manifest, port, baseline), port_version(master, port)
        if have and latest and have != latest:
            rows.append(f"| {port} (vcpkg) | {have} | {latest} |")
    script = (ROOT / "tools/deps/linux-from-source.sh").read_text()
    for name, url, ref in re.findall(r'^  "(\S+) (https://\S+) (\S+) [0-9a-f]{40}"$', script, re.M):
        latest = latest_tag(url, ref)
        if latest and version_key(latest) > version_key(ref):
            rows.append(f"| {name} (linux-from-source.sh) | {ref} | {latest} |")
    if not rows:
        print("Aucune dépendance en retard.")
        return 0
    print("| Dépendance | Ici | Dernière |\n|---|---|---|")
    print("\n".join(rows))
    if bump and baseline != master:
        manifest_path.write_text(manifest_path.read_text().replace(baseline, master))
        print(f"\nBaseline de vcpkg : {baseline[:12]} → {master[:12]}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
