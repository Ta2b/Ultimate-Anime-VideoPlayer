#!/usr/bin/env python3
"""Porte 2.3 : la carte de la scène du lecteur sur les extraits du banc 2.1.

  python3 -I tools/bench/qualite/carte_lecteur.py --outil build/nvof_dump/nvof_dump \\
      --rejeu build/carte_rejeu/carte_rejeu --extrait dialogue=dlg.mkv@3:10 --extrait panoramique=pan.mkv@12:30 \\
      --sortie ~/carte-banc

nvof_scene.py classe ses tuiles avec son propre classeur (blocs texturés, part de décor, sans lissage) : ses parts
« incertain » et ses bascules ne disent rien de la carte du lecteur. Ici, chaque extrait est lu comme le lecteur le
lit : lumas à la taille du flux du lecteur (travail ramené à 1080 lignes au plus, divisé par scene_flow_divisor),
toutes les paires voisines dans l'ordre, doublons compris (animation en deux ou en trois), NVOF en grille 4,
réglage moyen, indices temporels et carte de coût, puis la carte du lecteur rejouée par carte_rejeu
(kernels::ref::scene_motion_of, lissage compris ; outil à construire, voir son CMakeLists.txt). Par extrait :

- la part des tuiles décor, dessin, effet et incertain, et celle des tuiles sûres (décor ou dessin de confiance
  0,5 ou plus, le seuil des clients) ;
- la bascule : part des tuiles dont la classe change d'une paire à la suivante, sur toutes les paires et sur les
  paires où la caméra est tenue (plans fixes) ;
- la caméra : part des paires où elle est ajustée, où elle est tenue ; coupes et fondus probables ;
- la queue des coûts de NVOF (part des vecteurs au-dessus de 32 à 160, 99e et 99,9e centiles) : kSceneCostHigh
  (160) se règle là-dessus ;
- des vues PNG (luma ; décor en bleu, dessin en rouge, effet en jaune, incertain sans teinte ; plus pâle : pas
  sûre) pour juger à l'œil la classe des tuiles sûres.

Sortie : carte.md, carte.json et les vues. Il faut python3 avec numpy, ffmpeg et ffprobe.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvof_mesures import decode, probe  # noqa: E402

KINDS = ("incertain", "décor", "dessin", "effet")
TINTS = {1: (40, 110, 255), 2: (255, 50, 40), 3: (255, 220, 40)}
SURE, COST_HIGH, COST_STEPS = 0.5, 160, (32, 64, 96, 128, 160)
FLOW_MAX, WORK_MAX, GRID, TILE = 720, 1080, 4, 16  # lignes du flux, lignes de travail, grille, côté d'une tuile


def flow_size(path):
    """(largeur, hauteur, diviseur) du flux du lecteur pour cette source (scene_flow_divisor)."""
    sw, sh, _ = probe(path)
    work_h = min(sh, WORK_MAX)
    work_w = int(round(sw * work_h / sh / 2)) * 2
    divisor = max(1, (work_h + FLOW_MAX - 1) // FLOW_MAX)
    return work_w // divisor, work_h // divisor, divisor


def replay(args, frames, divisor):
    """[{kinds, changed, confidence (rows × cols), camera_fit, background_held, drawing_held, cut, fade}], coûts."""
    n, h, w = frames.shape
    ch, cw = (h + GRID - 1) // GRID, (w + GRID - 1) // GRID
    with tempfile.TemporaryDirectory() as tmp:
        name = lambda k, ext: os.path.join(tmp, f"{k}.{ext}")  # noqa: E731
        for k in range(n):
            frames[k].tofile(name(k, "y"))
        nvof = "".join(f"{w}\t{h}\t{name(k, 'y')}\t{name(k + 1, 'y')}\t{name(k, 'flux')}\n" for k in range(n - 1))
        done = subprocess.run([args.outil, "--grid", str(GRID), "--preset", "medium", "--one-way", "--cost",
                               "--hints"], input=nvof, text=True, capture_output=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or f"nvof_dump : code {done.returncode}")
        tasks = "".join(f"{w}\t{h}\t{name(k, 'y')}\t{name(k + 1, 'y')}\t{name(k, 'flux')}\t{name(k, 'carte')}\n"
                        for k in range(n - 1))
        done = subprocess.run([args.rejeu, "--diviseur", str(divisor)], input=tasks, text=True, capture_output=True,
                              check=False)
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or f"carte_rejeu : code {done.returncode}")
        costs = np.zeros(256, np.int64)
        pairs = []
        for k, line in enumerate(done.stdout.splitlines()):
            _, cols, rows, *flags = line.split()
            cols, rows = int(cols), int(rows)
            raw = np.fromfile(name(k, "carte"), np.uint8)
            t = cols * rows
            pairs.append({"kinds": raw[:t].reshape(rows, cols), "changed": raw[t:2 * t].reshape(rows, cols),
                          "confidence": raw[2 * t:].view(np.float32).reshape(rows, cols),
                          **dict(zip(("camera_fit", "background_held", "drawing_held", "cut", "fade"),
                                     (f == "1" for f in flags)))})
            costs += np.bincount(np.fromfile(name(k, "flux"), np.uint8)[ch * cw * 4:], minlength=256)
    return pairs, costs


def view(path, a, pair):
    """Luma teintée par classe ; une tuile pas sûre est teintée deux fois moins."""
    h, w = a.shape
    kinds, sure = pair["kinds"], pair["confidence"] >= SURE
    tint = np.zeros((*kinds.shape, 3), np.float32)
    alpha = np.zeros(kinds.shape, np.float32)
    for kind, rgb in TINTS.items():
        tint[kinds == kind] = rgb
        alpha[kinds == kind] = np.where(sure[kinds == kind], 0.45, 0.2)
    up = lambda m: np.repeat(np.repeat(m, TILE, 0), TILE, 1)[:h, :w]  # noqa: E731
    gray = np.repeat(a[..., None], 3, -1).astype(np.float32)
    img = gray * (1 - up(alpha)[..., None]) + up(tint) * up(alpha)[..., None]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                    path], input=img.astype(np.uint8).tobytes(), check=True)


def measure(pairs, frames):
    kinds = np.stack([p["kinds"] for p in pairs])
    conf = np.stack([p["confidence"] for p in pairs])
    sure = ((kinds == 1) | (kinds == 2)) & (conf >= SURE)
    flips = (kinds[1:] != kinds[:-1]).mean((1, 2))
    held = np.array([p["background_held"] for p in pairs[1:]], bool)
    doubles = [k for k in range(len(frames) - 1) if np.array_equal(frames[k], frames[k + 1])]
    return {"paires": len(pairs), "doublons": len(doubles),
            "classes": [float((kinds == j).mean()) for j in range(4)], "sures": float(sure.mean()),
            "bascule": float(flips.mean()) if len(flips) else math.nan,
            "bascule_tenue": float(flips[held].mean()) if held.any() else math.nan,
            "camera_ajustee": float(np.mean([p["camera_fit"] for p in pairs])),
            "camera_tenue": float(np.mean([p["background_held"] for p in pairs])),
            "coupes": int(sum(p["cut"] for p in pairs)), "fondus": int(sum(p["fade"] for p in pairs))}


def cost_tail(costs):
    total = max(1, int(costs.sum()))
    c = np.cumsum(costs)
    return {"au_dessus": {s: float(costs[s:].sum() / total) for s in COST_STEPS},
            "c99": int(np.searchsorted(c, 0.99 * total)), "c999": int(np.searchsorted(c, 0.999 * total))}


def report(rows, tail):
    pct = lambda x: "–" if x != x else f"{100 * x:.1f} %"  # noqa: E731
    L = ["# Carte de la scène du lecteur (porte 2.3)", "",
         "Toutes les paires voisines, doublons compris, à la taille du flux du lecteur ; NVOF grille 4, réglage "
         "moyen, indices temporels ; carte rejouée par carte_rejeu (lissage compris). Sûres : décor ou dessin de "
         f"confiance ≥ {SURE}. Bascule : tuiles qui changent de classe d'une paire à la suivante ; tenue : sur les "
         "paires où la caméra est tenue.", "",
         "| Extrait | Flux | Paires (doublons) | Décor / dessin / effet / incertain | Sûres | Bascule | Bascule tenue "
         "| Caméra ajustée / tenue | Coupes, fondus |",
         "|---|---|---:|---|---:|---:|---:|---|---|"]
    for name, (size, m) in rows.items():
        c = m["classes"]
        L.append(f"| {name} | {size[0]} × {size[1]} | {m['paires']} ({m['doublons']}) | "
                 f"{100 * c[1]:.0f} / {100 * c[2]:.0f} / {100 * c[3]:.0f} / {100 * c[0]:.0f} | {pct(m['sures'])} | "
                 f"{pct(m['bascule'])} | {pct(m['bascule_tenue'])} | {100 * m['camera_ajustee']:.0f} / "
                 f"{100 * m['camera_tenue']:.0f} % | {m['coupes']}, {m['fondus']} |")
    L += ["", f"Coûts de NVOF, tous extraits : 99e centile {tail['c99']}, 99,9e {tail['c999']} ; part des vecteurs "
              "au-dessus de " + ", ".join(f"{s} : {100 * v:.3f} %" for s, v in tail["au_dessus"].items())
          + f" (kSceneCostHigh = {COST_HIGH}).", "",
          "Vues : décor en bleu, dessin en rouge, effet en jaune, incertain sans teinte ; teinte pâle : pas sûre."]
    return re.sub(r"(\d)\.(\d)", r"\1,\2", "\n".join(L)).replace("porte 2,3", "porte 2.3") + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outil", required=True, help="l'exécutable nvof_dump")
    p.add_argument("--rejeu", required=True, help="l'exécutable carte_rejeu")
    p.add_argument("--extrait", action="append", required=True, help="TYPE=CHEMIN[@DÉBUT], plusieurs fois")
    p.add_argument("--images", type=int, default=160, help="images lues par extrait")
    p.add_argument("--vues", type=int, default=4, help="vues PNG par extrait")
    p.add_argument("--sortie", required=True)
    args = p.parse_args()
    os.makedirs(args.sortie, exist_ok=True)
    rows, costs, seen = {}, np.zeros(256, np.int64), {}
    for spec in args.extrait:
        typ, _, rest = spec.partition("=")
        path, _, start = rest.partition("@") if not os.path.exists(rest) else (rest, "", "")
        seen[typ] = seen.get(typ, 0) + 1
        name = typ if seen[typ] == 1 else f"{typ}{seen[typ]}"
        w, h, divisor = flow_size(path)
        frames = decode(path, start, args.images, h)
        try:
            pairs, c = replay(args, frames, divisor)
        except RuntimeError as e:
            print(f"{name} ignoré : {e}", file=sys.stderr)
            continue
        costs += c
        rows[name] = ((frames.shape[2], h), measure(pairs, frames))
        every = max(1, len(pairs) // max(1, args.vues))
        for k in range(0, len(pairs), every)[: args.vues]:
            view(os.path.join(args.sortie, f"carte_{name}_{k:04d}.png"), frames[k], pairs[k])
        print(f"{name} : {len(pairs)} paires", file=sys.stderr, flush=True)
    if not rows:
        raise SystemExit("aucun extrait rejoué")
    tail = cost_tail(costs)
    text = report(rows, tail)
    print(text)
    with open(os.path.join(args.sortie, "carte.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.sortie, "carte.json"), "w", encoding="utf-8") as f:
        json.dump({"extraits": {n: m for n, (_, m) in rows.items()}, "couts": tail}, f, indent=1)


if __name__ == "__main__":
    main()
