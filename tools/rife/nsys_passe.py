#!/usr/bin/env python3
"""Ce que coûtent les petits noyaux d'une passe de RIFE, d'après une capture nsys exportée en SQLite.

Les blocs grossiers (0 à 2 de 4.25) tournent sur des grilles qui n'occupent pas la carte : chaque couche y
coûte surtout sa latence (docs/rife-plafonds.md). Avant de les fondre en un lancement par bloc, il faut savoir
où passe ce temps : dans les noyaux eux-mêmes (une grille trop petite, des poids relus), ou entre eux (le
lancement suivant, le temps de remplir la carte). Le script range les noyaux de la voie la plus chargée par
passe (une passe : une suite de noyaux sans trou de plus de --trou ms) et sépare, pour les noyaux à petite
grille (moins de --blocs blocs : deux par SM de la RTX 4070 Ti par défaut), le temps dans les noyaux et le
temps perdu juste avant chacun.

  nsys profile -o rife --delay=15 --duration=20 --cuda-graph-trace=node kirari --debug ... video.mkv
  nsys export --type sqlite -o rife.sqlite rife.nsys-rep
  python3 tools/rife/nsys_passe.py rife.sqlite

Lecture : si l'écart avant les petits noyaux pèse autant que les petits noyaux eux-mêmes, un lancement par
bloc grossier (noyau persistant, barrière de grille entre les couches) a de quoi gagner ; s'il est faible,
le temps est dans les noyaux et c'est leur forme qu'il faut reprendre.
"""

import argparse
import sqlite3
import statistics
import sys


def kernels(db):
    """Les noyaux de la capture : (début, fin, voie, blocs de la grille, nom), en ns, triés par début."""
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute(
        "SELECT start, end, streamId, gridX * gridY * gridZ, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        " ORDER BY start"
    )
    return [(s, e, stream, blocks, names.get(short, "?")) for s, e, stream, blocks, short in rows]


def busiest_stream(all_kernels):
    """La voie qui lance le plus de noyaux : celle de RIFE pendant une lecture avec interpolation (une
    passe native en lance plus de cent, les autres voies quelques-uns par image)."""
    count = {}
    for _, _, stream, _, _ in all_kernels:
        count[stream] = count.get(stream, 0) + 1
    return max(count, key=count.get)


def passes(stream_kernels, gap_ns):
    """Les passes : des suites de noyaux de la voie sans trou de plus de gap_ns."""
    out, current = [], []
    for k in stream_kernels:
        if current and k[0] - current[-1][1] > gap_ns:
            out.append(current)
            current = []
        current.append(k)
    if current:
        out.append(current)
    return out


def summary(one_pass, small_blocks):
    """Une passe : sa durée, le temps dans ses noyaux, et pour les petites grilles noyaux et écarts."""
    span = one_pass[-1][1] - one_pass[0][0]
    busy = sum(e - s for s, e, _, _, _ in one_pass)
    small_busy = small_gap = small_count = 0
    for i, (s, e, _, blocks, _) in enumerate(one_pass):
        if blocks >= small_blocks:
            continue
        small_count += 1
        small_busy += e - s
        if i > 0:
            small_gap += max(0, s - one_pass[i - 1][1])
    return span, busy, small_busy, small_gap, small_count


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite", help="capture nsys exportée (nsys export --type sqlite)")
    parser.add_argument("--trou", type=float, default=1.0, help="trou qui sépare deux passes, en ms")
    parser.add_argument("--blocs", type=int, default=120, help="sous ce nombre de blocs, une petite grille")
    parser.add_argument("--voie", type=int, help="la voie à lire (défaut : celle qui lance le plus)")
    parser.add_argument("--min", type=int, default=20, help="noyaux au moins dans une passe retenue")
    args = parser.parse_args()

    db = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    all_kernels = kernels(db)
    if not all_kernels:
        sys.exit("aucun noyau dans la capture")
    stream = args.voie if args.voie is not None else busiest_stream(all_kernels)
    on_stream = [k for k in all_kernels if k[2] == stream]
    kept = [p for p in passes(on_stream, args.trou * 1e6) if len(p) >= args.min]
    if not kept:
        sys.exit("aucune passe d'au moins %d noyaux sur la voie %d" % (args.min, stream))

    rows = [summary(p, args.blocs) for p in kept]

    def med(i):
        return statistics.median(r[i] for r in rows) / 1e6

    print("voie %d : %d passes, %d noyaux par passe en médiane" % (
        stream, len(kept), statistics.median(len(p) for p in kept)))
    print("passe            %.3f ms (médiane)" % med(0))
    print("dans les noyaux  %.3f ms" % med(1))
    print("petites grilles  %.3f ms dans %d noyaux, %.3f ms d'écart juste avant eux" % (
        med(2), statistics.median(r[4] for r in rows), med(3)))

    # Les noyaux les plus lourds de la passe, par nom : de quoi relier le reste au profil de l'essai.
    per_name = {}
    for p in kept:
        for s, e, _, blocks, name in p:
            per_name.setdefault(name, []).append((e - s, blocks))
    print("\nnoyau (nom court)                                   médiane ms   par passe   blocs")
    heavy = sorted(per_name.items(), key=lambda kv: -sum(d for d, _ in kv[1]))
    for name, seen in heavy[:15]:
        print("%-50s %10.3f %11.1f %7d" % (
            name[:50], statistics.median(d for d, _ in seen) / 1e6, len(seen) / len(kept),
            statistics.median(b for _, b in seen)))


if __name__ == "__main__":
    main()
