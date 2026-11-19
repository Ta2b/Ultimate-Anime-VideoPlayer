#!/usr/bin/env python3
"""Les calques et la plaque de décor rejoués sur processeur, sur les flux NVOF exportés une fois par le PC.

  python3 -I tools/bench/qualite/calques_rejeu.py --rejeu build/carte_rejeu/carte_rejeu \\
      --extrait panoramique=flux/panoramique/taches.tsv --extrait dialogue=flux/dialogue/taches.tsv@2 \\
      --sortie ~/calques-banc

Chaque extrait est un fichier de tâches de carte_rejeu (LARGEUR HAUTEUR A.y B.y FLUX [SORTIE], chemins relatifs
au fichier ; SORTIE est remplacée), toutes les paires voisines du plan dans l'ordre, comme carte_lecteur.py les
écrit. @D : le diviseur du travail au flux ; sinon le fichier « diviseur » à côté des tâches, sinon 2 pour un
flux de 361 à 540 lignes (source 1080p), 1 au-delà ou en dessous.

carte_rejeu --calques rejoue la carte, puis ce que le lecteur déciderait de chaque paire (app/play/layer_plan.h,
compilé tel quel) et ce que la garde de la composition en refuserait, sur les lumas du flux (calques.h). Par
extrait :

- la part des paires avec calques (visé : au moins la moitié sur un panoramique), dont avec la caméra lk d'une
  paire voisine, et les raisons des autres ; --sans-affinage rejoue le pire du lecteur, l'affinage toujours en
  retard ;
- la part de tuiles décor des paires à calques ;
- la garde des calques à l'instant ½ : part du décor rendue à RIFE (médiane, 90e centile des paires) et écart
  médian entre a et b déplacées par la caméra, en codes (sous kLayerGuardLow, 4 codes, rien n'est refusé) ;
- la plaque : clés, reprises, et la même garde contre b (kPlateGuardLow, 6 codes) ;
- la vitesse médiane de la caméra, pixels à 720p.

Sur les lumas du flux, sans SPAN ni RIFE : le lecteur compare des images agrandies, en RVB, et sa plaque est un
SPAN de la clé. Les décisions sont celles du lecteur ; la garde et les écarts en sont une approximation, à
comparer d'un extrait ou d'un réglage à l'autre. Vues : calques.h. Sortie : calques.md, calques.json et les
vues (PNG si ffmpeg est là).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from statistics import median

FIELDS = ("etat", "part", "garde", "ecart", "vitesse", "plaque", "pgarde", "pecart")


def divisor_of(tasks, given, flow_height):
    if given:
        return int(given)
    beside = os.path.join(os.path.dirname(tasks), "diviseur")
    if os.path.exists(beside):
        with open(beside, encoding="utf-8") as f:
            return int(f.read().strip())
    return 2 if 360 < flow_height <= 540 else 1


def read_tasks(path, tmp):
    """Les lignes de tâches, chemins absolus, SORTIE dans tmp ; et la hauteur du flux."""
    base = os.path.dirname(os.path.abspath(path))
    lines, height = [], 0
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(row for row in f if row.strip()):
            w, h, a, b, flow = line.rstrip("\n").split("\t")[:5]
            full = [p if os.path.isabs(p) else os.path.join(base, p) for p in (a, b, flow)]
            lines.append("\t".join([w, h, *full, os.path.join(tmp, f"{k}.carte")]) + "\n")
            height = int(h)
    return lines, height


def replay(args, name, tasks, given):
    with tempfile.TemporaryDirectory() as tmp:
        lines, height = read_tasks(tasks, tmp)
        divisor = divisor_of(tasks, given, height)
        views = os.path.join(args.sortie, name)
        cmd = [args.rejeu, "--diviseur", str(divisor), "--calques"] + (["--sans-affinage"] if args.sans_affinage else [])
        if args.pas > 0:
            os.makedirs(views, exist_ok=True)
            cmd += ["--vues", views, "--pas", str(args.pas)]
        done = subprocess.run(cmd, input="".join(lines), text=True, capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or f"carte_rejeu : code {done.returncode}")
    pairs = []
    for line in done.stdout.splitlines():
        if line.startswith("calques "):
            parts = line.split(" ", len(FIELDS) + 1)
            pair = dict(zip(FIELDS, map(float, parts[1:len(FIELDS) + 1])))
            pair["raison"] = parts[-1]
            pairs.append(pair)
    if args.pas > 0 and shutil.which("ffmpeg"):
        for f in sorted(os.listdir(views)):
            if f.endswith(".ppm"):
                ppm = os.path.join(views, f)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ppm, ppm[:-4] + ".png"], check=True)
                os.remove(ppm)
    return pairs, height * divisor


def quantile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else float("nan")


def measure(pairs, work_height):
    layered = [p for p in pairs if p["etat"] >= 1]
    guarded = [p for p in layered if p["garde"] >= 0]
    used = [p for p in pairs if p["plaque"] == 2 and p["pgarde"] >= 0]
    unit = 720.0 / work_height
    return {"paires": len(pairs), "avec": len(layered) / max(1, len(pairs)),
            "voisine": sum(p["etat"] == 2 for p in pairs) / max(1, len(pairs)),
            "part": sum(p["part"] for p in layered) / len(layered) if layered else float("nan"),
            "garde": median([p["garde"] for p in guarded]) if guarded else float("nan"),
            "garde90": quantile([p["garde"] for p in guarded], 0.9),
            "ecart": median([p["ecart"] for p in guarded if p["ecart"] >= 0] or [float("nan")]),
            "cles": sum(p["plaque"] == 1 for p in pairs), "reprises": sum(p["plaque"] == 2 for p in pairs),
            "pgarde": median([p["pgarde"] for p in used]) if used else float("nan"),
            "pgarde90": quantile([p["pgarde"] for p in used], 0.9),
            "pecart": median([p["pecart"] for p in used if p["pecart"] >= 0] or [float("nan")]),
            "vitesse": median([p["vitesse"] * unit for p in layered]) if layered else float("nan"),
            "sans": dict(Counter(p["raison"] for p in pairs if p["etat"] == 0).most_common())}


def report(rows):
    pct = lambda x: "–" if x != x else f"{100 * x:.0f} %"  # noqa: E731
    num = lambda x: "–" if x != x else f"{x:.1f}"  # noqa: E731
    L = ["# Calques et plaque rejoués sur processeur", "",
         "Décisions du lecteur (layer_plan.h) sur la carte rejouée par carte_rejeu ; garde et écarts sur les lumas "
         "du flux, une approximation de ceux du lecteur (images agrandies, RVB, plaque passée par SPAN). Garde : part "
         "du décor rendue à RIFE, médiane et 90e centile des paires ; écart : médiane, en codes (garde des calques "
         "de 4 à 12, de la plaque de 6 à 18). Visé : au moins la moitié des paires avec calques sur un panoramique.",
         "", "| Extrait | Paires | Avec calques (caméra voisine) | Décor | Garde (p90) | Écart | Plaque : clés, "
         "reprises | Garde plaque (p90) | Écart plaque | Vitesse (px à 720p) |",
         "|---|---:|---:|---:|---|---:|---|---|---:|---:|"]
    for name, m in rows.items():
        L.append(f"| {name} | {m['paires']} | {pct(m['avec'])} ({pct(m['voisine'])}) | {pct(m['part'])} | {pct(m['garde'])} "
                 f"({pct(m['garde90'])}) | {num(m['ecart'])} | {m['cles']}, {m['reprises']} | {pct(m['pgarde'])} "
                 f"({pct(m['pgarde90'])}) | {num(m['pecart'])} | {num(m['vitesse'])} |")
    L += ["", "Sans calques :", ""]
    for name, m in rows.items():
        why = ", ".join(f"{reason} {count}" for reason, count in m["sans"].items()) or "aucune"
        L.append(f"- {name} : {why}")
    return re.sub(r"(\d)\.(\d)", r"\1,\2", "\n".join(L)) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rejeu", required=True, help="l'exécutable carte_rejeu")
    p.add_argument("--extrait", action="append", required=True, help="NOM=TÂCHES[@DIVISEUR], plusieurs fois")
    p.add_argument("--pas", type=int, default=40, help="une vue toutes les PAS paires (0 : aucune)")
    p.add_argument("--sans-affinage", action="store_true",
                   help="l'affinage toujours en retard : la caméra lk d'une paire voisine seulement")
    p.add_argument("--sortie", required=True)
    args = p.parse_args()
    os.makedirs(args.sortie, exist_ok=True)
    rows = {}
    for spec in args.extrait:
        name, _, rest = spec.partition("=")
        tasks, _, given = rest.rpartition("@") if "@" in rest else (rest, "", "")
        try:
            pairs, work_height = replay(args, name, tasks, given)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"{name} ignoré : {e}", file=sys.stderr)
            continue
        rows[name] = measure(pairs, work_height)
        print(f"{name} : {len(pairs)} paires", file=sys.stderr, flush=True)
    if not rows:
        raise SystemExit("aucun extrait rejoué")
    text = report(rows)
    print(text)
    with open(os.path.join(args.sortie, "calques.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.sortie, "calques.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
