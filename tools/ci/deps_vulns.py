#!/usr/bin/env python3
"""Les failles connues des bibliothèques livrées qui lisent des octets venus du dehors (A35-01) : FFmpeg, libass,
dav1d, FreeType, HarfBuzz, fribidi, zlib, SDL, libplacebo. Chaque version livrée est ramenée à son commit, puis
demandée à OSV (https://osv.dev), qui range les avis de sécurité (CVE, OSS-Fuzz) par commits touchés.

  tools/ci/deps_vulns.py

Versions lues : tools/deps/linux-from-source.sh (Linux), les ports de la baseline de vcpkg.json et ports/libass
(Windows) ; libdav1d et libsvtav1enc du paquet Linux viennent d'Ubuntu 24.04 (APT_PLAYER de versions.env) :
leur version installée est demandée à dpkg, puis à OSV dans l'écosystème Ubuntu. Les avis écartés exprès, avec leur raison, sont dans
tools/ci/deps-vulns-ignore.txt. Écrit un tableau Markdown ; code 0 si rien n'est ouvert, 1 sinon.
"""
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deps_lag import ROOT, ls_remote, port_version  # noqa: E402

OSV = "https://api.osv.dev/v1/query"
# Les ports de vcpkg qui touchent des octets hostiles, leur dépôt et le nom de l'étiquette d'une version.
PORTS = {
    "ffmpeg": ("https://github.com/FFmpeg/FFmpeg", lambda v: f"n{v}"),
    "freetype": ("https://github.com/freetype/freetype", lambda v: "VER-" + v.replace(".", "-")),
    "harfbuzz": ("https://github.com/harfbuzz/harfbuzz", lambda v: v),
    "fribidi": ("https://github.com/fribidi/fribidi", lambda v: f"v{v}"),
    "dav1d": ("https://code.videolan.org/videolan/dav1d.git", lambda v: v),
    "zlib": ("https://github.com/madler/zlib", lambda v: f"v{v}"),
    "sdl3": ("https://github.com/libsdl-org/SDL", lambda v: f"release-{v}"),
}
# Ce que linux-from-source.sh construit et que le paquet livre.
LINUX = {"ffmpeg", "libass", "libplacebo", "sdl3", "libunibreak"}


def tag_commit(url, tag):
    refs = dict((ref, sha) for sha, ref in ls_remote(url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"))
    return refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")


# Les bibliothèques que le paquet Linux copie d'Ubuntu (paquet binaire, paquet source dans OSV).
# FriBidi et lcms2 viennent aussi de l'image de construction : le paquet les livre (package-linux.sh ne les range pas
# parmi les bibliothèques du système) ; FreeType, HarfBuzz et zlib, eux, sont celles du système de l'utilisateur.
UBUNTU = {"libdav1d7": "dav1d", "libsvtav1enc1d1": "svt-av1", "libfribidi0": "fribidi", "liblcms2-2": "lcms2"}
UBUNTU_ECOSYSTEM = "Ubuntu:24.04:LTS"


def vulns(query):
    found, token = [], None
    while True:
        body = dict(query, page_token=token) if token else query
        request = urllib.request.Request(OSV, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        found += data.get("vulns", [])
        token = data.get("next_page_token")
        if not token:
            return found


def ubuntu_version(package):
    if not shutil.which("dpkg-query"):
        return None
    done = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package], capture_output=True, text=True)
    return done.stdout.strip() or None


def shipped():
    """(composant, version, origine, requête OSV) de chaque bibliothèque livrée ; requête None : introuvable."""
    out = []
    script = (ROOT / "tools/deps/linux-from-source.sh").read_text()
    for name, url, ref, commit in re.findall(r'^  "(\S+) (https://\S+) (\S+) ([0-9a-f]{40})"$', script, re.M):
        if name in LINUX:
            out.append((f"{name} (Linux)", ref, url, {"commit": commit}))
    baseline = json.loads((ROOT / "vcpkg.json").read_text())["builtin-baseline"]
    for port, (url, tag_of) in PORTS.items():
        version = port_version(baseline, port)
        if not version:
            # Hors ligne, 404 ou GitHub lent : la bibliothèque n'a pas été relevée. Elle est dite introuvable (code 1)
            # et non passée sous silence, sans quoi « Aucune faille connue » refermerait l'issue ouverte (B28-08).
            out.append((f"{port} (Windows)", "?", "port de vcpkg illisible", None))
            continue
        version = re.sub(r"#\d+$", "", str(version))
        commit = tag_commit(url, tag_of(version))
        out.append((f"{port} (Windows)", version, url, {"commit": commit} if commit else None))
    portfile = (ROOT / "ports/libass/portfile.cmake").read_text()
    ref = re.search(r"REF ([0-9a-f]{40})", portfile)
    if ref:
        out.append(("libass (Windows)", ref.group(1)[:12], "ports/libass", {"commit": ref.group(1)}))
    for binary, source in UBUNTU.items():
        version = ubuntu_version(binary)
        query = {"package": {"name": source, "ecosystem": UBUNTU_ECOSYSTEM}, "version": version}
        out.append((f"{source} (Linux)", version or "?", UBUNTU_ECOSYSTEM, query if version else None))
    return out


def ignored():
    path = ROOT / "tools/ci/deps-vulns-ignore.txt"
    entries = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        line = line.strip()
        if line and not line.startswith("#"):
            vuln, _, reason = line.partition(" ")
            entries[vuln] = reason.strip()
    return entries


def main():
    skip = ignored()
    rows, unresolved = [], []
    for name, version, origin, query in shipped():
        if not query:
            unresolved.append(f"{name} {version} ({origin})")
            continue
        for vuln in vulns(query):
            ids = [vuln["id"], *vuln.get("aliases", [])]
            if any(i in skip for i in ids):
                continue
            summary = (vuln.get("summary") or vuln.get("details") or "").splitlines()[0][:120]
            rows.append(f"| {name} | {version} | [{vuln['id']}](https://osv.dev/vulnerability/{vuln['id']}) | {summary} |")
    if unresolved:
        print("Versions introuvables (étiquette absente ou port illisible) : " + ", ".join(unresolved) + "\n")
    if not rows:
        print("Aucune faille connue dans les bibliothèques livrées.")
        return 1 if unresolved else 0
    print("| Bibliothèque | Version | Avis | Résumé |\n|---|---|---|---|")
    print("\n".join(rows))
    return 1


if __name__ == "__main__":
    sys.exit(main())
