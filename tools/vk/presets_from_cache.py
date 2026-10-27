#!/usr/bin/env python3
"""Tables pré-réglées du moteur Vulkan depuis le cache de l'autotuner d'un PC de référence.

L'autotuner (src/gpu/vulkan/backend/net_autotune.cpp) écrit, à chaque construction,
« <cache>/vulkan/autotune.tsv » : une ligne par couche tranchée, « clé \\t variante \\t réglages
\\t version du pilote », la clé valant « carte|portée|signature|version des candidats ». Ce script
relit un ou plusieurs de ces caches, garde les lignes de la version des candidats en cours (lue
dans net_autotune.cpp : une ligne tranchée sous une autre liste de candidats serait remesurée), et
écrit src/gpu/vulkan/backend/net_presets.cpp : par famille, le choix de chaque signature de couche.
Une signature tranchée différemment selon le modèle ou la résolution garde le choix le plus
fréquent (le premier dans l'ordre, à égalité) ; elle est signalée.

Sur un appareil de la famille, un choix de la table n'est pas remesuré : sa sortie est vérifiée une
fois contre le noyau direct, comme un choix repris d'une autre version du pilote.

Les caches relevés sont gardés dans tools/vk/presets, nommés « <Famille>-<carte>-<n>.tsv » : sans
--family, le script les relit tous, et une nouvelle famille s'ajoute sans faire perdre les autres.
Un nom « <Famille>-<pilote>-<carte>-<n>.tsv » (pilote : radv, amd, nvidia) réserve la table à ce
pilote : RADV et le pilote AMD classent différemment les variantes matricielles.

  cp autotune.tsv tools/vk/presets/AmdRdna4-radv-rx9070xt-1.tsv
  python3 tools/vk/presets_from_cache.py
  python3 tools/vk/presets_from_cache.py --family NvidiaAda a.tsv --family AmdRdna4 b.tsv
  clang-format-20 -i src/gpu/vulkan/backend/net_presets.cpp
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "src" / "gpu" / "vulkan" / "backend"
OUT = BACKEND / "net_presets.cpp"
KEPT = ROOT / "tools" / "vk" / "presets"


def candidate_set() -> str:
    text = (BACKEND / "net_autotune.cpp").read_text(encoding="utf-8")
    found = re.search(r'kCandidateSet = "([^"]+)"', text)
    if not found:
        sys.exit("kCandidateSet introuvable dans net_autotune.cpp")
    return found.group(1)


def families() -> set[str]:
    text = (BACKEND / "family.h").read_text(encoding="utf-8")
    body = re.search(r"enum class GpuFamily[^{]*\{(.*?)\};", text, re.S)
    if not body:
        sys.exit("GpuFamily introuvable dans family.h")
    return set(re.findall(r"^\s*(\w+)\s*,", body.group(1), re.M))


def read_cache(path: pathlib.Path, wanted_set: str) -> tuple[dict[int, collections.Counter], set[str], int]:
    """Les choix par signature, les cartes vues, et le nombre de lignes d'une autre version."""
    choices: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    cards: set[str] = set()
    stale = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) < 3:
            print(f"{path}:{number} : ligne abîmée, ignorée", file=sys.stderr)
            continue
        key, variant, tuning = fields[0], fields[1], fields[2]
        parts = key.rsplit("|", 2)
        if len(parts) != 3 or "|" not in parts[0]:
            print(f"{path}:{number} : clé illisible, ignorée", file=sys.stderr)
            continue
        prefix, signature, cset = parts
        if cset != wanted_set:
            stale += 1
            continue
        if not re.fullmatch(r"[0-9a-f]{16}", signature) or not re.fullmatch(r"(\d+(,\d+)*)?", tuning):
            print(f"{path}:{number} : signature ou réglages illisibles, ignorés", file=sys.stderr)
            continue
        if not re.fullmatch(r"[\w-]+", variant):
            print(f"{path}:{number} : variante illisible, ignorée", file=sys.stderr)
            continue
        cards.add(prefix.split("|", 1)[0])
        choices[int(signature, 16)][(variant, tuning)] += 1
    return choices, cards, stale


DRIVERS = {"radv": "Radv", "amd": "AmdProprietary", "nvidia": "NvidiaProprietary"}


def split_name(name: str) -> tuple[str, str | None]:
    """La famille et le pilote (ou rien) d'un nom « <Famille>[-<pilote>]-<carte>-<n>.tsv »."""
    parts = name.split("-")
    return parts[0], DRIVERS.get(parts[1]) if len(parts) > 1 else None


def table_text(family: str, choices: dict[int, collections.Counter]) -> list[str]:
    lines = []
    for signature in sorted(choices):
        counter = choices[signature]
        best = max(sorted(counter), key=lambda choice: counter[choice])
        if len(counter) > 1:
            others = ", ".join(f"{v} {t} ×{n}" for (v, t), n in sorted(counter.items()))
            print(f"{family} {signature:016x} : choix différents ({others}), gardé {best[0]} {best[1]}",
                  file=sys.stderr)
        lines.append(f'    {{0x{signature:016x}, "{best[0]}", "{best[1]}"}},')
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", nargs=2, action="append", default=[], metavar=("FAMILLE", "CACHE"),
                        help="une famille de GpuFamily et le autotune.tsv d'un appareil de cette famille "
                             "(sans : tools/vk/presets/<Famille>-*.tsv)")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    wanted_set = candidate_set()
    known = families()
    merged: dict[str, dict[int, collections.Counter]] = {}
    sources: dict[str, set[str]] = collections.defaultdict(set)
    sources_given = args.family or [(p.name, str(p)) for p in sorted(KEPT.glob("*.tsv"))]
    for given, cache in sources_given:
        family, driver = split_name(given)
        if family not in known or family in ("Unknown", "Software"):
            sys.exit(f"{family} : famille inconnue (family.h : {', '.join(sorted(known))})")
        choices, cards, stale = read_cache(pathlib.Path(cache), wanted_set)
        if stale:
            print(f"{cache} : {stale} lignes d'une autre version des candidats, ignorées", file=sys.stderr)
        table = merged.setdefault((family, driver), collections.defaultdict(collections.Counter))
        for signature, counter in choices.items():
            table[signature].update(counter)
        sources[(family, driver)] |= cards

    out = [
        "// Tables pré-réglées du moteur Vulkan (net_autotune.h : PresetTable), générées par",
        "// tools/vk/presets_from_cache.py depuis le cache de l'autotuner d'un PC de référence : ne pas",
        "// modifier à la main. Version des candidats : " + wanted_set + ".",
    ]
    for key in sorted(merged, key=str):
        family, driver = key
        label = family + (f" ({driver})" if driver else "")
        out.append(f"// {label} : {', '.join(sorted(sources[key]))}, {len(merged[key])} couches.")
    out += [
        "#include <array>",
        "#include <span>",
        "",
        '#include "gpu/vulkan/backend/net_autotune.h"',
        "",
        "namespace ajp::gpu::vk::net {",
    ]
    if merged:
        out += ["namespace {", ""]
        for key in sorted(merged, key=str):
            family, driver = key
            out.append(f"constexpr std::array<PresetEntry, {len(merged[key])}> k{family}{driver or ''}{{{{")
            out += table_text(family, merged[key])
            out += ["}};", ""]
        out += ["}  // namespace", ""]
    else:
        out.append("")
    out += [
        "std::span<const PresetTable> preset_tables() {",
        f"  static constexpr std::array<PresetTable, {len(merged)}> kTables{{" + ("{" if merged else "}"),
    ]
    if merged:
        for key in sorted(merged, key=str):
            family, driver = key
            pinned = f".driver = DriverKind::{driver}, " if driver else ""
            out.append(f"      {{.family = GpuFamily::{family}, {pinned}.entries = k{family}{driver or ''}}},")
        out.append("  }};")
    else:
        out[-1] += ";"
    out += ["  return kTables;", "}", "", "}  // namespace ajp::gpu::vk::net", ""]
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"{args.out} : {sum(len(t) for t in merged.values())} couches, {len(merged)} familles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
