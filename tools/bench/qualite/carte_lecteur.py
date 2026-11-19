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
- la caméra affinée par Lucas-Kanade comme dans le lecteur (porte 2.1) : part des paires affinées, écart du décor
  mesuré sur l'image (médiane, part sous 0,5 pixel à 720p), écart des vecteurs, tuiles gardées ;
- l'écart après la caméra, qui sépare le dessin du décor : part de pixels qui bougent sur le décor (le bruit) et
  sur les autres tuiles, tuiles qui bougent, recalées, effets par l'image ;
- les lumières (effets lumineux) : parmi les tuiles qui bougent, celles sans bord net, celles qui passent les
  critères d'une lumière tuile par tuile, et l'accord des pixels qui bougent (médiane, 90e centile) ;
- l'ajustement de la caméra : tuiles candidates contre le minimum demandé, paires où elles manquent, paires où
  le rejet des aberrants échoue, tuiles écartées (texture trop faible, coût, vecteurs épars), quartiles de la
  texture ;
- la queue des coûts de NVOF (part des vecteurs au-dessus de 8 à 64, 99e et 99,9e centiles) : kSceneCostHigh
  (24) se règle là-dessus ;
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
SURE, COST_HIGH, COST_STEPS = 0.5, 24, (8, 16, 24, 32, 48, 64)  # COST_HIGH : kSceneCostHigh
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
        for line in done.stdout.splitlines():
            head, *fields = line.split()
            if head in ("diag", "camera", "image"):  # les lignes qui suivent la carte de la même paire
                pairs[-1][head] = [float(f) for f in fields]
                continue
            k = len(pairs)
            cols, rows, flags = int(fields[0]), int(fields[1]), fields[2:]
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
            "coupes": int(sum(p["cut"] for p in pairs)), "fondus": int(sum(p["fade"] for p in pairs)),
            "ajustement": fit_diagnosis(pairs), "affinage": camera_diagnosis(pairs, frames.shape[1]),
            "image": image_diagnosis(pairs)}


def image_diagnosis(pairs):
    """L'écart après la caméra (ligne image de carte_rejeu), sur les paires où il est mesuré : part de ces paires,
    centiles médians de la part de pixels qui bougent sur le décor (50, 90, 99) et sur les autres tuiles (50, 90),
    écart moyen |e| médian du décor (codes), part des tuiles mesurées qui bougent par leurs pixels, par le seul
    recalage, et part des effets par l'image ; parmi les tuiles qui bougent, part sans bord net, part des
    lumières, accord médian et 90e centile (moyennes des paires)."""
    d = np.array([p["image"] for p in pairs if "image" in p], np.float64)
    if not len(d):
        return {}
    measured = d[d[:, 0] > 0]
    if not len(measured):
        return {"mesuree": 0.0}
    tiles = measured[:, 0]
    med = lambda j: float(np.median(measured[:, j]))  # noqa: E731
    out = {"mesuree": float(len(measured) / len(d)), "decor": [med(2), med(3), med(4)], "decor_ecart": med(5),
           "autres": [med(7), med(8)], "bougent": float(np.mean(measured[:, 9] / tiles)),
           "decalees": float(np.mean(measured[:, 10] / tiles)), "effets": float(np.mean(measured[:, 11] / tiles))}
    if measured.shape[1] >= 16:  # les colonnes des lumières (carte_rejeu d'après le 19 novembre)
        some = measured[measured[:, 9] > 0]  # les paires où des tuiles bougent
        out["lumieres"] = {"douces": float(np.mean(some[:, 12] / some[:, 9])) if len(some) else math.nan,
                           "lumieres": float(np.mean(measured[:, 13] / tiles)),
                           "accord": [float(np.mean(some[:, 14])) if len(some) else math.nan,
                                      float(np.mean(some[:, 15])) if len(some) else math.nan]}
    return out


def camera_diagnosis(pairs, flow_height):
    """La caméra comme le lecteur la tire (ligne camera de carte_rejeu), en pixels à 720p : part des paires
    ajustées qui sont affinées, écart médian mesuré sur l'image et part des paires sous 0,5 pixel (la porte 2.1),
    écart médian des vecteurs, tuiles gardées par l'ajustement."""
    c = np.array([p["camera"] for p in pairs if "camera" in p and p["camera"][0] == 1], np.float64)
    if not len(c):
        return {}
    unit = 720.0 / flow_height
    measured = c[:, 3] >= 0
    image = c[measured, 3] * unit
    return {"affinee": float(c[:, 1].mean()), "mesuree": float(measured.mean()),
            "ecart_image": float(np.median(image)) if len(image) else math.nan,
            "sous_demi": float((image < 0.5).mean()) if len(image) else math.nan,
            "ecart_vecteurs": float(np.median(c[:, 5]) * unit), "gardees": float(np.median(c[:, 6]))}


def fit_diagnosis(pairs):
    """Pourquoi la caméra s'ajuste ou non (ligne diag de carte_rejeu) : candidates et minimum médians, paires
    sans assez de candidates, paires qui en ont assez mais que le rejet des aberrants fait échouer, part des
    tuiles écartées (texture trop faible, coût, vecteurs épars) et quartiles médians de la texture."""
    d = np.array([p["diag"] for p in pairs if "diag" in p], np.float64)
    if not len(d):
        return {}
    fit = np.array([p["camera_fit"] for p in pairs if "diag" in p], bool)
    tiles = d[:, [0, 2, 3, 4]].sum(1)
    short = d[:, 0] < d[:, 1]
    return {"candidates": float(np.median(d[:, 0])), "minimum": float(np.median(d[:, 1])),
            "trop_peu": float(short.mean()), "rejet": float((~short & ~fit).mean()),
            "ecartees": [float(np.mean(d[:, j] / np.maximum(tiles, 1))) for j in (2, 3, 4)],
            "texture": [float(np.median(d[:, j])) for j in (5, 6, 7)]}


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
    L += ["", "## Ajustement de la caméra", "",
          "Candidates : tuiles qui peuvent ajuster la caméra (médiane par paire) / minimum demandé. Trop peu : "
          "paires sans assez de candidates ; rejet : paires qui en ont assez mais que le rejet des aberrants fait "
          "échouer. Écartées : part des tuiles écartées, texture sous kFitTexture, sans correspondance, vecteurs "
          "épars. "
          "Texture : quartiles médians, codes par pixel (kFitTexture : 1,5 pour ajuster ; kTextureFloor : 3, aplat "
          "en dessous).", "",
          "| Extrait | Candidates / minimum | Trop peu | Rejet | Écartées texture / coût / épars "
          "| Texture q1 / q2 / q3 |",
          "|---|---|---:|---:|---|---|"]
    for name, (_, m) in rows.items():
        f = m.get("ajustement")
        if not f:
            continue
        e, t = f["ecartees"], f["texture"]
        L.append(f"| {name} | {f['candidates']:.0f} / {f['minimum']:.0f} | {pct(f['trop_peu'])} | {pct(f['rejet'])} | "
                 f"{100 * e[0]:.0f} / {100 * e[1]:.0f} / {100 * e[2]:.0f} % | {t[0]:.1f} / {t[1]:.1f} / {t[2]:.1f} |")
    L += ["", "## Caméra affinée (porte 2.1)", "",
          "Sur les paires où la caméra est ajustée : part affinée par Lucas-Kanade (spec/scene_camera.h) ; écart "
          "du décor mesuré sur l'image après affinage (médiane, pixels à 720p ; part des paires mesurées, part sous "
          "0,5 pixel) ; écart des vecteurs de NVOF au modèle (médiane, pixels à 720p) ; tuiles gardées par "
          "l'ajustement (médiane).", "",
          "| Extrait | Affinée | Mesurée | Écart image | Sous 0,5 px | Écart vecteurs | Tuiles gardées |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, (_, m) in rows.items():
        f = m.get("affinage")
        if not f:
            continue
        L.append(f"| {name} | {pct(f['affinee'])} | {pct(f['mesuree'])} | {f['ecart_image']:.2f} | "
                 f"{pct(f['sous_demi'])} | {f['ecart_vecteurs']:.2f} | {f['gardees']:.0f} |")
    L += ["", "## Écart après la caméra (séparation dessin / décor)", "",
          "Sur les paires où il est mesuré (caméra affinée) : part de pixels qui bougent après la caméra, "
          "centiles médians 50 / 90 / 99 sur les tuiles du décor (le bruit : kMovingShare doit rester au-dessus) "
          "et 50 / 90 sur les autres ; écart moyen |e| médian du décor, codes ; part des tuiles mesurées qui "
          "bougent par leurs pixels (kMovingShare, 0.1), par le seul recalage (kShiftMoving, 1 pixel à 720p), et "
          "effets par l'image (kEffectShare 0.3, kEffectCost 12).", "",
          "| Extrait | Mesurée | Décor 50 / 90 / 99 | Écart décor | Autres 50 / 90 | Bougent | Recalées | Effets |",
          "|---|---:|---|---:|---|---:|---:|---:|"]
    for name, (_, m) in rows.items():
        f = m.get("image")
        if not f or "decor" not in f:
            continue
        dd, a = f["decor"], f["autres"]
        L.append(f"| {name} | {pct(f['mesuree'])} | {100 * dd[0]:.1f} / {100 * dd[1]:.1f} / {100 * dd[2]:.1f} % | "
                 f"{f['decor_ecart']:.2f} | {100 * a[0]:.1f} / {100 * a[1]:.1f} % | {pct(f['bougent'])} | "
                 f"{pct(f['decalees'])} | {pct(f['effets'])} |")
    L += ["", "## Lumières (effets lumineux)", "",
          "Sur les paires mesurées, parmi les tuiles qui bougent : part sans bord net (kLightSharp 0.25, kSharpEdge "
          "8 codes par pixel), part des tuiles mesurées qui sont des lumières tuile par tuile (kLightAgree 0.6, "
          "kLightKept 0.5 ; la carte demande aussi des voisines sans bord net et un vecteur qui suit la caméra), "
          "accord |éclaircis − assombris| / qui bougent, médiane et 90e centile. La part d'effet de la carte est "
          "dans le premier tableau.", "",
          "| Extrait | Sans bord net | Lumières | Accord 50 / 90 |", "|---|---:|---:|---|"]
    for name, (_, m) in rows.items():
        f = (m.get("image") or {}).get("lumieres")
        if f:
            L.append(f"| {name} | {pct(f['douces'])} | {pct(f['lumieres'])} | "
                     f"{f['accord'][0]:.2f} / {f['accord'][1]:.2f} |")
    L += ["", f"Coûts de NVOF, tous extraits : 99e centile {tail['c99']}, 99,9e {tail['c999']} ; part des vecteurs "
              "au-dessus de " + ", ".join(f"{s} : {100 * v:.3f} %" for s, v in tail["au_dessus"].items())
          + f" (kSceneCostHigh = {COST_HIGH}).", "",
          "Vues : décor en bleu, dessin en rouge, effet en jaune, incertain sans teinte ; teinte pâle : pas sûre."]
    text = re.sub(r"(\d)\.(\d)", r"\1,\2", "\n".join(L))
    return text.replace("porte 2,3", "porte 2.3").replace("porte 2,1", "porte 2.1") + "\n"


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
