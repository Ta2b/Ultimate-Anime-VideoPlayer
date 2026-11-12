#!/usr/bin/env python3
"""Rapport des tests matériels d'un poste NVIDIA (C28-02) : ce que la CI ne peut pas juger sans carte.

Aucune tâche de la CI n'exécute TensorRT, les noyaux CUDA, NVDEC ni les noyaux FP8 : une régression y passe CI verte.
Avant une étiquette finale, les tests cachés (`ctest -L nvidia|gpu|models`) se lancent sur le poste de référence, et ce
script en écrit le rapport que release-guard.yml vérifie (`check` : commit identique, aucun échec, sauts sous le
plafond de tools/release/hardware-skip-ceiling.txt).

  Linux    : ctest --preset linux-prefix-cuda-hardware --output-junit hw.xml
  Windows  : ctest --preset windows-rtx --output-junit hw.xml   (tous les tests : le rendu et le Vulkan de repli aussi)
  puis     : python3 tools/release/hardware_report.py write --platform linux|windows --junit hw.xml \
               --out hardware-report-<plateforme>.json
             gh release upload v<VERSION> hardware-report-linux.json hardware-report-windows.json
  contrôle : python3 tools/release/hardware_report.py check --sha <commit> --platform linux hardware-report-linux.json
"""
import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CEILINGS = ROOT / "tools/release/hardware-skip-ceiling.txt"


def run(*command):
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_junit(path):
    """(total, échecs, sautés, noms des sautés) d'un rapport JUnit de ctest."""
    total = failed = 0
    skipped = []
    for case in ET.parse(path).getroot().iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None or case.get("status") == "fail":
            failed += 1
        if case.find("skipped") is not None or case.get("status") == "notrun":
            skipped.append(case.get("name", "?"))
    return total, failed, skipped


def write(args):
    total, failed, skipped = read_junit(args.junit)
    gpu = run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    report = {
        "platform": args.platform,
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        "gpu": gpu.splitlines()[0] if gpu else "",
        "total": total,
        "failed": failed,
        "skipped": len(skipped),
        "skipped_names": skipped,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out} : {total} cas, {failed} échecs, {len(skipped)} sautés, carte « {report['gpu'] or 'inconnue'} »")
    return 0


def ceiling(platform):
    for line in CEILINGS.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^{platform}\s+(\d+)\s*(#.*)?$", line)
        if match:
            return int(match.group(1))
    sys.exit(f"hardware_report : aucun plafond pour « {platform} » dans {CEILINGS}")


def check(args):
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    problems = []
    if report.get("platform") != args.platform:
        problems.append(f"rapport d'une autre plateforme ({report.get('platform')})")
    if report.get("commit") != args.sha:
        problems.append(f"rapport du commit {report.get('commit')}, pas de {args.sha}")
    if report.get("dirty"):
        problems.append("arbre modifié au moment du lancement")
    if not report.get("gpu"):
        problems.append("aucune carte NVIDIA relevée (nvidia-smi)")
    if report.get("total", 0) < 1:
        problems.append("aucun test joué")
    if report.get("failed", 1) != 0:
        problems.append(f"{report.get('failed')} tests échoués")
    limit = ceiling(args.platform)
    if report.get("skipped", 10**9) > limit:
        problems.append(f"{report.get('skipped')} tests sautés, plus que le plafond {limit} : "
                        + ", ".join(report.get("skipped_names", [])))
    for problem in problems:
        print(f"hardware_report : {args.platform} : {problem}", file=sys.stderr)
    if not problems:
        print(f"{args.platform} : {report['total']} cas sur « {report['gpu']} », {report['skipped']} sautés (plafond {limit})")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--platform", required=True, choices=["linux", "windows"])
    w.add_argument("--junit", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(run=write)
    c = sub.add_parser("check")
    c.add_argument("--sha", required=True)
    c.add_argument("--platform", required=True, choices=["linux", "windows"])
    c.add_argument("report")
    c.set_defaults(run=check)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
