#!/usr/bin/env python3
"""Étape 1 (doc « Protéger les visages ») : mesures sur des cadres x.pq.ppm (PQ BT.2020 16 bits, grade compris).

  gref.py mesure  <cadre.pq.ppm> [--ref cadre_sans_grade.pq.ppm] [--face x0,y0,x1,y1]   -> une ligne de chiffres
  gref.py fausses <cadre.pq.ppm> <sortie.png> [--face x0,y0,x1,y1 ...]                    -> carte en fausses couleurs (nits)
  gref.py vue     <cadre.pq.ppm> <sortie.png>                                             -> aperçu SDR pour regarder le cadre
Les bandes d'interface (titre et statistiques en haut, barre de lecture en bas) sont exclues des mesures.
"""
import sys

import numpy as np

TOP, BOTTOM = 230, 230  # lignes exclues (interface du lecteur), pour un cadre de 1440 lignes


def lire(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split())
        mv = int(f.readline())
        raw = f.read()
    return np.frombuffer(raw, dtype=">u2").reshape(h, w, 3).astype(np.float64) / mv


def pq_vers_nits(e):
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.power(np.clip(e, 0, 1), 1 / m2)
    return 10000.0 * np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1)


def luminance(nits_rgb):  # BT.2020
    return 0.2627 * nits_rgb[..., 0] + 0.6780 * nits_rgb[..., 1] + 0.0593 * nits_rgb[..., 2]


def ictcp(nits_rgb):
    r, g, b = nits_rgb[..., 0], nits_rgb[..., 1], nits_rgb[..., 2]
    lc = (1688 * r + 2146 * g + 262 * b) / 4096
    mc = (683 * r + 2951 * g + 462 * b) / 4096
    sc = (99 * r + 309 * g + 3688 * b) / 4096

    def pq(x):
        m1, m2 = 2610 / 16384, 2523 / 4096 * 128
        c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
        y = np.power(np.clip(x, 0, 10000) / 10000.0, m1)
        return np.power((c1 + c2 * y) / (1 + c3 * y), m2)

    lp, mp, sp = pq(lc), pq(mc), pq(sc)
    i = 0.5 * lp + 0.5 * mp
    ct = (6610 * lp - 13613 * mp + 7003 * sp) / 4096
    cp = (17933 * lp - 17390 * mp - 543 * sp) / 4096
    return i, ct, cp


def de_itp(a, b):
    i1, t1, p1 = ictcp(a)
    i2, t2, p2 = ictcp(b)
    return 720.0 * np.sqrt((i1 - i2) ** 2 + (0.5 * (t1 - t2)) ** 2 + (p1 - p2) ** 2)


def zone_utile(h):
    return slice(TOP * h // 1440, h - BOTTOM * h // 1440)


def mesure(path, ref=None, face=None):
    a = lire(path)
    h = a.shape[0]
    z = zone_utile(h)
    n = pq_vers_nits(a)
    y = luminance(n)[z]
    out = {
        "fichier": path.split("/")[-1],
        "moyenne_nits": float(y.mean()),
        "p99_9_nits": float(np.percentile(y, 99.9)),
        "part_>400": float((y > 400).mean()),
        "part_>700": float((y > 700).mean()),
    }
    if ref:
        yr = luminance(pq_vers_nits(lire(ref)))[z]
        out["moyenne/ref"] = float(y.mean() / max(yr.mean(), 1e-6))
        out["p99_9/ref"] = float(np.percentile(y, 99.9) / max(np.percentile(yr, 99.9), 1e-6))
    if face:
        x0, y0, x1, y1 = face
        f = luminance(n[y0:y1, x0:x1]).ravel()
        out["visage_mediane"] = float(np.median(f))
        out["visage_p95"] = float(np.percentile(f, 95))
        if ref:
            fr = luminance(pq_vers_nits(lire(ref))[y0:y1, x0:x1]).ravel()
            out["visage_mediane/ref"] = float(np.median(f) / max(np.median(fr), 1e-6))
        # grain : ΔE_ITP entre pixels voisins (horizontal) sur le visage, valeur médiane
        blk = n[y0:y1, x0:x1]
        out["visage_grain_dE"] = float(np.median(de_itp(blk[:, 1:], blk[:, :-1])))
    return out


BANDES = [
    (0, 5, (20, 20, 60)),
    (5, 20, (30, 60, 130)),
    (20, 60, (40, 140, 160)),
    (60, 120, (60, 170, 80)),
    (120, 203, (200, 200, 60)),
    (203, 400, (230, 140, 40)),
    (400, 700, (220, 50, 40)),
    (700, 1e9, (255, 255, 255)),
]


def fausses(path, sortie, faces=()):
    from PIL import Image, ImageDraw

    y = luminance(pq_vers_nits(lire(path)))
    h, w = y.shape
    img = np.zeros((h, w, 3), np.uint8)
    for lo, hi, c in BANDES:
        img[(y >= lo) & (y < hi)] = c
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in faces:
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=4)
    im.resize((w // 2, h // 2), Image.NEAREST).save(sortie)


def vue(path, sortie):
    from PIL import Image

    a = lire(path)
    n = pq_vers_nits(a) / 203.0  # blanc de référence 203 nits = 1,0
    c = np.clip(n, 0, 1) ** (1 / 2.2)
    Image.fromarray((c * 255).astype(np.uint8)).resize((a.shape[1] // 2, a.shape[0] // 2)).save(sortie)


if __name__ == "__main__":
    cmd = sys.argv[1]
    args = sys.argv[2:]

    def opt(nom):
        if nom in args:
            i = args.index(nom)
            v = tuple(int(x) for x in args[i + 1].split(","))
            del args[i : i + 2]
            return v
        return None

    if cmd == "mesure":
        ref = None
        if "--ref" in args:
            i = args.index("--ref")
            ref = args[i + 1]
            del args[i : i + 2]
        print(mesure(args[0], ref, opt("--face")))
    elif cmd == "fausses":
        faces = []
        while "--face" in args:
            faces.append(opt("--face"))
        fausses(args[0], args[1], faces)
    elif cmd == "vue":
        vue(args[0], args[1])
