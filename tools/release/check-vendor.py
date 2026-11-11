#!/usr/bin/env python3
"""Surveille les fichiers de NVIDIA que le lecteur tire de pypi.nvidia.com (B51-16).

Les index nvidia-vendor/*.lock donnent, pour chaque [[vendor_file]], l'adresse d'une roue, la plage d'octets
(data_offset, packed_size) d'un fichier dans cette roue, sa taille et son SHA-256 une fois décompressé. Si NVIDIA retire
ou remplace une roue, toutes les releases passées perdent l'installation de TensorRT en même temps : ce script le voit
avant les utilisateurs.

  python3 tools/release/check-vendor.py tools/release/nvidia-vendor/*.lock

Chaque adresse est interrogée (HEAD) ; et, par index, le plus petit fichier est relu par plage, décompressé et comparé
à sa taille et à son SHA-256 (les plus gros pèsent des centaines de Mo : un seul par index suffit à voir qu'une roue
a changé). Code de sortie 1 au premier écart. --selftest essaie la lecture sans réseau.
"""
import hashlib
import sys
import urllib.request
import zlib


def read_entries(text):
    """Les tables [[vendor_file]] d'un index : une liste de dictionnaires clé -> valeur."""
    entries, current = [], None
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if line.startswith("["):
            current = {} if line == "[[vendor_file]]" else None
            if current is not None:
                entries.append(current)
        elif current is not None and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            current[key] = value.strip('"')
    return entries


def check_range(entry, read_range):
    """Lit la plage du fichier, la décompresse et la compare ; rend un message d'écart, ou None."""
    start = int(entry["data_offset"])
    packed = int(entry["packed_size"])
    data = read_range(entry["url"], start, start + packed - 1)
    if len(data) != packed:
        return f"{entry['name']} : {len(data)} octets lus au lieu de {packed}"
    if entry.get("method") == "deflate":
        data = zlib.decompressobj(-15).decompress(data)
    if len(data) != int(entry["size"]):
        return f"{entry['name']} : {len(data)} octets décompressés au lieu de {entry['size']}"
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        return f"{entry['name']} : SHA-256 différent de l'index"
    return None


def http_range(url, first, last):
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        if answer.status != 206:
            raise OSError(f"{url} : réponse {answer.status} à une demande de plage")
        return answer.read()


def http_head(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as answer:
        return answer.status


def check_index(name, entries, read_range, head):
    """Les écarts d'un index (liste de messages) : une adresse qui ne répond plus, ou la plus petite plage qui a changé."""
    problems = []
    for url in sorted({e["url"] for e in entries}):
        try:
            if head(url) != 200:
                problems.append(f"{name} : {url} ne répond pas 200")
        except OSError as error:
            problems.append(f"{name} : {url} : {error}")
    if entries and not problems:
        smallest = min(entries, key=lambda e: int(e["packed_size"]))
        try:
            problem = check_range(smallest, read_range)
        except OSError as error:
            problem = f"{smallest['name']} : {error}"
        if problem:
            problems.append(f"{name} : {problem}")
    return problems


def selftest():
    payload = b"contenu d'essai " * 100
    packer = zlib.compressobj(wbits=-15)
    packed = packer.compress(payload) + packer.flush()
    wheel = b"x" * 10 + packed + b"y" * 10
    entry = {
        "url": "https://example.invalid/w.whl", "name": "f", "data_offset": "10", "packed_size": str(len(packed)),
        "size": str(len(payload)), "method": "deflate", "sha256": hashlib.sha256(payload).hexdigest(),
    }

    def reader(_url, first, last):
        return wheel[first:last + 1]

    assert check_range(entry, reader) is None
    assert "SHA-256" in check_range({**entry, "sha256": "0" * 64}, reader)
    assert "octets" in check_range({**entry, "size": "5"}, reader)
    text = "# x\n\n[[vendor_file]]\nid = \"tensorrt\"\nname = \"f\"\n"
    assert read_entries(text) == [{"id": "tensorrt", "name": "f"}]
    assert check_index("i", [entry], reader, lambda _u: 200) == []
    assert check_index("i", [entry], reader, lambda _u: 404) != []
    print("check-vendor : essai passé")


def main(paths):
    if paths == ["--selftest"]:
        selftest()
        return 0
    if not paths:
        print(__doc__)
        return 2
    problems = []
    for path in paths:
        with open(path, encoding="utf-8") as lock:
            entries = read_entries(lock.read())
        if not entries:
            problems.append(f"{path} : aucun [[vendor_file]]")
            continue
        problems += check_index(path, entries, http_range, http_head)
        print(f"{path} : {len(entries)} fichier(s) NVIDIA examinés")
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
