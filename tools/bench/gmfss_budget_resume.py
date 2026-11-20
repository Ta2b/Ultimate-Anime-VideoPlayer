#!/usr/bin/env python3
"""Estime les coûts séparés entre paire et clé GMFSS depuis un profil de lecture.

Le résultat est un point de départ pour --debug --gmfss-budget, à éprouver
sur les mêmes dimensions, chemins de rendu et voies de calcul. La somme
des centiles des étapes ne représente pas un centile de latence du lecteur.
"""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    count: int
    median_ms: float
    p95_ms: float


def profile(path):
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            measurements = int(row["mesures"])
            median = float(row["mediane_ms"])
            p95 = float(row["p95_ms"])
            if measurements > 0 and math.isfinite(median) and 0 <= median <= p95 < math.inf:
                yield Stage(row["etape"], measurements, median, p95)


def matching(stages, prefix):
    return [stage for stage in stages if stage.name.startswith(prefix)]


def slowest(stages, prefix):
    return max(matching(stages, prefix), key=lambda stage: stage.p95_ms, default=None)


def estimate(stages, include_span, extra_key_ms):
    pairs = matching(stages, "GMFSS paire ")
    cold = [stage for stage in pairs if ", A reprise " not in stage.name]
    if not cold:
        raise ValueError("profil sans paire GMFSS complète")
    pair = [max(cold, key=lambda stage: stage.p95_ms)]
    pair_input = slowest(stages, "entrée de GMFSS")
    if pair_input:
        pair.append(pair_input)

    fusion = slowest(stages, "GMFSS fusion ")
    splats = slowest(stages, "splatting de GMFSS")
    if fusion is None or splats is None:
        raise ValueError("profil incomplet : fusion ou splats GMFSS absents")
    key = [fusion, splats]
    still = slowest(stages, "décor fixe de GMFSS")
    composition = slowest(stages, "composition de GMFSS")
    if still:
        key.append(still)
    if composition:
        compose_splats = slowest(stages, "splats de composition GMFSS")
        if compose_splats is None:
            raise ValueError("profil ancien : les splats de composition et de fusion sont mélangés")
        key.extend([composition, compose_splats])
    if include_span:
        span = slowest(stages, "moteur SPAN ")
        if span is None:
            raise ValueError("--key-span demandé sans étape SPAN")
        key.append(span)

    for label, components, extra in (("paire", pair, 0.0), ("clé", key, extra_key_ms)):
        print(label)
        for stage in components:
            print(f"  {stage.name:45} n={stage.count:4} médiane {stage.median_ms:6.2f}"
                  f" ms  p95 {stage.p95_ms:6.2f} ms")
        if extra:
            print(f"  autre coût fourni {extra:.2f} ms")
    for reused in [stage for stage in pairs if ", A reprise " in stage.name]:
        print(f"réemploi A avant admission : {reused.count} paires, p95 {reused.p95_ms:.2f} ms")

    pair_ms = math.ceil(sum(stage.p95_ms for stage in pair))
    key_ms = math.ceil(sum(stage.p95_ms for stage in key) + extra_key_ms)
    print(f"essai : --debug --gmfss-budget={pair_ms},{key_ms}")
    if min(stage.count for stage in pair + key) < 20:
        print("échantillon de moins de 20 mesures sur au moins une étape")
    if not include_span:
        print("aucun SPAN réservé à la clé dans cette estimation")
    print("rejouer la cadence et les pertes avec les mêmes voies et le même rendu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_tsv", type=Path)
    parser.add_argument("--key-span", action="store_true",
                        help="ajouter le SPAN dédié à chaque clé quand le warp suit SPAN")
    parser.add_argument("--extra-key-ms", type=float, default=0.0,
                        help="ajouter par clé le grade, ses copies et le redimensionnement si utilisés")
    args = parser.parse_args()
    if not math.isfinite(args.extra_key_ms) or args.extra_key_ms < 0.0:
        parser.error("--extra-key-ms doit être un coût positif fini ou nul")
    try:
        estimate(list(profile(args.gpu_tsv)), args.key_span, args.extra_key_ms)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
