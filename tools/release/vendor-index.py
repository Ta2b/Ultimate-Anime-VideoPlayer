#!/usr/bin/env python3
"""Écrit les entrées [[vendor_file]] d'engines.lock pour une roue (zip) que NVIDIA publie.

Le lecteur ne télécharge pas la roue entière : pour chaque fichier utile il demande la plage de
ses octets (curl --range) et le décompresse lui-même. Ce script lit le répertoire central du zip
par plages HTTP, retrouve pour chaque membre choisi où ses octets commencent et combien ils pèsent,
puis lit ces octets (une fois, en flux) pour en écrire le SHA-256 décompressé. Il n'écrit rien
d'autre que du TOML sur la sortie standard : à coller dans engines.lock, à côté de l'[[engine]] du
module.

  vendor-index.py --engine tensorrt --platform linux --url https://pypi.nvidia.com/…/x.whl \
      --include 'libnvinfer\\.so\\.\\d+$' --include 'libnvinfer_builder_resource_(?P<arch>sm\\d+)_.*\\.so'

--include est une expression régulière cherchée dans le nom de base du membre ; un groupe
« arch » nomme l'architecture du GPU qui en a besoin (sans groupe : toujours installé).
--local PATH lit un fichier local au lieu de l'URL (essais).
"""
import argparse
import hashlib
import re
import struct
import sys
import urllib.request
import zlib


class Source:
    def __init__(self, url, local):
        self.url, self.local = url, local
        if local:
            with open(local, "rb") as f:
                f.seek(0, 2)
                self.size = f.tell()
        else:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as r:
                self.size = int(r.headers["Content-Length"])

    def read(self, start, length):
        if self.local:
            with open(self.local, "rb") as f:
                f.seek(start)
                return f.read(length)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def stream(self, start, length, chunk=1 << 20):
        done = 0
        while done < length:
            n = min(chunk, length - done)
            yield self.read(start + done, n)
            done += n


def central_directory(src):
    tail = src.read(max(0, src.size - 66000), min(src.size, 66000))
    at = tail.rfind(b"PK\x05\x06")
    if at < 0:
        sys.exit("fin de zip introuvable")
    _, _, _, _, entries, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[at:at + 22])
    if cd_offset == 0xFFFFFFFF or entries == 0xFFFF:
        loc = tail.rfind(b"PK\x06\x07", 0, at)
        zip64_at = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        rec = src.read(zip64_at, 56)
        entries, cd_size, cd_offset = struct.unpack("<QQQ", rec[32:56])
    data = src.read(cd_offset, cd_size)
    pos, out = 0, []
    for _ in range(entries):
        (_, _, _, _, method, _, _, _, csize, usize, nlen, elen, clen, _, _, _, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", data[pos:pos + 46]
        )
        name = data[pos + 46:pos + 46 + nlen].decode()
        extra = data[pos + 46 + nlen:pos + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, lho):
            e = 0
            while e + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[e:e + 4])
                if tag == 1:
                    vals = list(struct.unpack("<" + "Q" * (sz // 8), extra[e + 4:e + 4 + sz]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if lho == 0xFFFFFFFF:
                        lho = vals.pop(0)
                e += 4 + sz
        out.append((name, method, csize, usize, lho))
        pos += 46 + nlen + elen + clen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--url", required=True, help="URL https de la roue (écrite dans l'index)")
    ap.add_argument("--local", help="lire ce fichier local au lieu de l'URL")
    ap.add_argument("--include", action="append", required=True)
    a = ap.parse_args()
    patterns = [re.compile(p) for p in a.include]
    src = Source(a.url, a.local)
    for name, method, csize, usize, lho in sorted(central_directory(src)):
        base = name.rsplit("/", 1)[-1]
        match = next((m for m in (p.search(base) for p in patterns) if m), None)
        if match is None or name.endswith("/"):
            continue
        if method not in (0, 8):
            sys.exit(f"{name} : méthode {method} non gérée")
        header = src.read(lho, 30)
        nlen, elen = struct.unpack("<HH", header[26:30])
        data_offset = lho + 30 + nlen + elen
        sha = hashlib.sha256()
        if method == 8:
            d = zlib.decompressobj(-15)
            total = 0
            for part in src.stream(data_offset, csize):
                out = d.decompress(part)
                sha.update(out)
                total += len(out)
            out = d.flush()
            sha.update(out)
            total += len(out)
            if total != usize:
                sys.exit(f"{name} : {total} octets décompressés, {usize} annoncés")
        else:
            for part in src.stream(data_offset, csize):
                sha.update(part)
        arch = match.groupdict().get("arch") or ""
        print("[[vendor_file]]")
        print(f'id = "{a.engine}"')
        print(f'platform = "{a.platform}"')
        print(f'url = "{a.url}"')
        print(f'name = "{base}"')
        print(f"data_offset = {data_offset}")
        print(f"packed_size = {csize}")
        print(f"size = {usize}")
        print(f'method = "{"deflate" if method == 8 else "stored"}"')
        print(f'sha256 = "{sha.hexdigest()}"')
        if arch:
            print(f'arch = "{arch}"')
        print()


main()
