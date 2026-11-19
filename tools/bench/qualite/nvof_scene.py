#!/usr/bin/env python3
"""Porte 2.1 : le flux optique matériel de NVIDIA (NVOF) sépare-t-il le décor des personnages ?

  python3 -I tools/bench/qualite/nvof_scene.py --outil build/nvof_dump/nvof_dump \\
      --extrait panoramique=pan.mkv@12:30 --extrait dialogue=dlg.mkv@3:10 --extrait sakuga=s.mkv@0:41 \\
      --sortie ~/nvof-banc

Un extrait est TYPE=CHEMIN[@DÉBUT] ; les types attendus par le guide : dialogue, panoramique, sakuga, fondu,
flash, zoom, parallaxe, pluie (seul « panoramique » entre dans le verdict). Pour chaque grille (--grilles) et
chaque hauteur où NVOF calcule (--hauteurs), sur des paires d'images voisines qui bougent (sens avant, préréglage
slow, carte de coût), le banc mesure :

- le temps GPU d'un appel (événements posés autour de nvOFExecute, sur les flux CUDA propres de la session) ;
- la qualité du flux : PSNR de b recalé sur a par le flux, moins celui de b tel quel (gain en dB) ;
- la carte de coût sur les aplats, les traits et le grain (classes tirées de la luma de a) ;
- le mouvement de caméra : affine à 6 paramètres, moindres carrés pondérés par la confiance (coût bas), avec
  rejet des aberrants, sur les blocs texturés seulement ; le décor est l'ensemble des blocs que le modèle explique ;
- l'écart du décor au modèle, mesuré sur l'image (Lucas-Kanade sur les tuiles de décor, nvof_mesures.photo_residual)
  et, pour information, sur les vecteurs NVOF (bruités d'environ 0,3 px, et pris par un personnage lent) ;
- des tuiles de 32 px (en 1080p) classées décor, dessin ou incertain, et leur stabilité d'une paire à l'autre ;
- sur les panoramiques, le mode pyramide (--pyramide 360:720) : NVOF en 360p, caméra ajustée, puis ce mouvement
  passé en indices externes (grille 8) à la passe 720p ; porte, résidu et part du décor avec et sans, par vitesse.
  Au-delà de 45 px, NVOF seul dépasse sa portée de recherche (écart moyen de 45 px à RIFE, 17 novembre).

Porte : sur les paires des panoramiques où la caméra bouge (0,5 px ou plus à 720p, vu par une configuration au
moins ; les mêmes paires pour toutes), l'écart médian du décor au modèle, mesuré sur l'image, sous 0,5 px à 720p,
avec au moins 40 % des blocs texturés gardés par l'ajustement, pour 90 % des paires. Sinon, on
garde les usages par paire (coupes, tri, tuiles) et on abandonne les calques.

Sortie : rapport.md, mesures.json et, par extrait, des vues PNG (luma ; tuiles décor en bleu, dessin en rouge ;
écart au modèle de caméra, du noir à 0 au jaune à 4 px). Il faut python3 avec numpy, ffmpeg et ffprobe.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvof_mesures import (at_cells, camera, decode, fit_pair, hint_field, moving_pairs, run_tool,  # noqa: E402
                          texture, tiles, warp_gain)

GATE_PX, GATE_SHARE, GATE_PAIRS, MOVING_PX = 0.5, 0.4, 0.9, 0.5
CLASSES = ("aplat", "trait", "grain")


def view(path, a, tile_map, t, grid, flow, model, height):
    """Trois panneaux côte à côte : luma, tuiles (décor bleu, dessin rouge), écart au modèle (0 à 4 px à 720p)."""
    h, w = a.shape
    gray = np.repeat(a[..., None], 3, -1).astype(np.float32)
    tint = np.zeros((*tile_map.shape, 3), np.float32)
    tint[tile_map == 0] = (40, 110, 255)
    tint[tile_map == 1] = (255, 50, 40)
    up = np.repeat(np.repeat(tint, t * grid, 0), t * grid, 1)[:h, :w]
    alpha = np.repeat(np.repeat((tile_map < 2).astype(np.float32), t * grid, 0), t * grid, 1)[:h, :w, None] * 0.45
    overlay = gray * (1 - alpha) + up * alpha
    err = np.clip(np.hypot(*(flow - model)) / (height / 720.0) / 4.0, 0, 1)
    err = np.repeat(np.repeat(err, grid, 0), grid, 1)[:h, :w]
    heat = np.stack([err * 255, err * 220, err * 40], -1)
    img = np.concatenate([gray, overlay, heat], 1).astype(np.uint8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * w}x{h}",
                    "-i", "-", path], input=img.tobytes(), check=True)


def pyramid(args, path, start, pairs, low, high, grid=4, hint_grid=8):
    """Passe en low (grille 4), caméra ajustée, puis passe en high avec et sans ce mouvement en indices externes.
    Rend [{image, camera_bas, sans: {...}, avec: {...}}], {"bas": [ms], "sans": [ms], "avec": [ms]}."""
    lo, hi = decode(path, start, args.images, low), decode(path, start, args.images, high)
    pairs = [k for k in pairs if k + 1 < min(len(lo), len(hi))]
    flows_lo, ms_lo, _ = run_tool(args.outil, grid, False, lo, pairs)
    hints, base = {}, {}
    _, h, w = hi.shape
    for k in pairs:
        flow, cost = flows_lo[k]
        cls = at_cells(texture(lo[k]), grid, *cost.shape)
        fit = camera(flow, cost, cls != 0, grid, low)
        base[k] = fit[2]["camera_px"] if fit else None
        hints[k] = hint_field(fit[2]["affine"] if fit else [0] * 6, high / low, w, h, hint_grid)
    plain, ms_plain, _ = run_tool(args.outil, grid, False, hi, pairs)
    guided, ms_guided, _ = run_tool(args.outil, grid, False, hi, pairs, (hint_grid, hints))
    rows = []
    for k in pairs:
        cls = at_cells(texture(hi[k]), grid, *plain[k][1].shape)
        row = {"image": k, "camera_bas": base[k]}
        for label, flows in (("sans", plain), ("avec", guided)):
            flow, cost = flows[k]
            fit = fit_pair(hi[k], hi[k + 1], flow, cost, cls != 0, grid, high)
            row[label] = {"gain_db": warp_gain(hi[k], hi[k + 1], flow, grid), **(fit[2] if fit else {})}
        rows.append(row)
    return rows, {"bas": ms_lo, "sans": ms_plain, "avec": ms_guided}


def pyramid_report(pyr, low, high):
    """Les panoramiques par vitesse de caméra (passe basse, px à 720p) : porte, résidu, part du décor, gain."""
    L = ["", f"## Pyramide : caméra en {low}p passée en indices externes à la passe {high}p", "",
         "Sur les extraits panoramiques. Par case : sans / avec les indices. Porte : part des paires où le résidu (image) "
         f"du décor est sous {GATE_PX} px avec au moins {100 * GATE_SHARE:.0f} % de décor.", "",
         "| Caméra (px à 720p) | Paires | Porte | Résidu médian | Part décor | Gain (dB) |", "|---|---:|---|---|---|---|"]
    rows = [r for clip in pyr.values() for r in clip["paires"] if r["camera_bas"] is not None]
    for lo, hi in ((MOVING_PX, 15), (15, 45), (45, 1e9), (MOVING_PX, 1e9)):
        group = [r for r in rows if lo <= r["camera_bas"] < hi]
        if not group:
            continue
        label = "toutes" if (lo, hi) == (MOVING_PX, 1e9) else (f"{lo:g} à {hi:g}" if hi < 1e9 else f"{lo:g} et plus")
        cells = []
        for fn in (lambda side: 100 * np.mean([r[side].get("residu_px", 99) < GATE_PX
                                               and r[side].get("part_decor", 0) >= GATE_SHARE for r in group]),
                   lambda side: np.median([r[side].get("residu_px", np.nan) for r in group]),
                   lambda side: 100 * np.median([r[side].get("part_decor", 0) for r in group]),
                   lambda side: np.mean([r[side]["gain_db"] for r in group])):
            cells.append(f"{fn('sans'):.1f} / {fn('avec'):.1f}")
        L.append(f"| {label} | {len(group)} | {cells[0]} % | {cells[1]} | {cells[2]} % | {cells[3]} |")
    ms = {k: [m for clip in pyr.values() for m in clip["temps"][k]] for k in ("bas", "sans", "avec")}
    if all(ms.values()):
        L += ["", f"Temps médian d'un appel : {low}p {np.median(ms['bas']):.2f} ms, {high}p sans indices "
                  f"{np.median(ms['sans']):.2f} ms, avec {np.median(ms['avec']):.2f} ms."]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# Rapport

def pct(hist, q):
    c = np.cumsum(hist)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] else float("nan")


def moving_set(results, names):
    """Les paires (extrait, image) où une configuration au moins voit bouger la caméra. Toutes les configurations sont
    jugées sur ces paires-là : une grille qui ne trouve plus le mouvement (portée de recherche dépassée) ne peut pas
    sortir les panoramiques rapides de son propre compte."""
    return {(name, r["image"]) for slot in results.values() for name in names
            for r in slot["paires"].get(name, []) if r.get("camera_px", 0) >= MOVING_PX}


def gate(rows, moving):
    """rows : [(extrait, ligne)] d'une configuration. Une paire mobile sans ajustement compte comme ratée."""
    judged = [r for name, r in rows if (name, r["image"]) in moving]
    good = [r for r in judged if r.get("residu_px", float("inf")) < GATE_PX and r.get("part_decor", 0) >= GATE_SHARE]
    return len(good), len(moving), sum(1 for r in judged if r.get("tuiles_mesurees", 0) > 0)


def report(results, times, info, clips, ref, pyr=None, levels=None):
    L = ["# Banc NVOF hors lecteur (porte 2.1)", "",
         f"Carte : {info.get('gpu', '?')}, pilote CUDA {info.get('cuda', '?')}, grilles acceptées : "
         f"{info.get('grilles', '?')}. Préréglage slow, sens avant, carte de coût allumée.", "",
         "## Temps GPU d'un appel (ms)", "", "| Grille | Hauteur | Médiane | 95e centile | Appels |",
         "|---|---|---:|---:|---:|"]
    for (g, hgt), ms in sorted(times.items()):
        L.append(f"| {g} × {g} | {hgt}p | {np.median(ms):.2f} | {np.percentile(ms, 95):.2f} | {len(ms)} |")
    L += ["", "## Qualité du flux et carte de coût", "",
          "Gain : PSNR de b recalé sur a, moins b tel quel (dB, plus haut : meilleur flux). Coût : médiane / "
          "90e centile par classe de pixels de a.", "",
          "| Grille | Hauteur | Gain moyen | " + " | ".join(f"Coût {c}" for c in CLASSES) + " |",
          "|---|---|---:|" + "---:|" * len(CLASSES)]
    for key in sorted(results):
        rows = [r for c in results[key]["paires"].values() for r in c]
        hists = results[key]["couts"]
        L.append(f"| {key[0]} × {key[0]} | {key[1]}p | {np.mean([r['gain_db'] for r in rows]):.2f} | "
                 + " | ".join(f"{pct(hists[c], .5)} / {pct(hists[c], .9)}" for c in CLASSES) + " |")
    L += ["", f"## Extraits ({ref[0]} × {ref[0]}, {ref[1]}p)", "",
          "Caméra : mouvement moyen du modèle ; résidu : écart médian du décor au modèle mesuré sur l'image (celui de la "
          "porte) ; résidu flux : le même sur les vecteurs NVOF ; px à 720p. Tuiles : "
          "part décor / dessin / incertain ; bascule : tuiles qui changent de classe entre deux paires voisines.", "",
          "| Extrait | Paires (doublons) | Gain | Coût moyen | Caméra | Part décor | Résidu | Résidu flux | Tuiles "
          "| Bascule |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name, _, _ in clips:
        rows = results.get(ref, {"paires": {}})["paires"].get(name, [])
        fit = [r for r in rows if "residu_px" in r]
        if not rows:
            continue
        med = lambda k: np.median([r[k] for r in fit]) if fit else float("nan")  # noqa: E731
        share = np.mean([r["tuiles"] for r in rows], 0)
        flips = [r["bascule"] for r in rows if r.get("bascule") is not None]
        L.append(f"| {name} | {len(rows)} ({results[ref]['doublons'][name]}) | "
                 f"{np.mean([r['gain_db'] for r in rows]):.2f} | {np.mean([r['cout_moyen'] for r in rows]):.1f} | "
                 f"{med('camera_px'):.2f} | {100 * med('part_decor'):.0f} % | {med('residu_px'):.2f} | "
                 f"{med('residu_flux_px'):.2f} | "
                 + " / ".join(f"{100 * s:.0f}" for s in share)
                 + f" | {100 * np.mean(flips) if flips else float('nan'):.1f} % |")
    L += ["", "## Porte deux-un", "",
          f"Paires des panoramiques où une configuration au moins voit bouger la caméra (≥ {MOVING_PX} px), les mêmes "
          f"pour toutes ; une paire sans ajustement est ratée. Résidu médian du décor sous {GATE_PX} px et "
          f"part du décor ≥ {100 * GATE_SHARE:.0f} %, sur {100 * GATE_PAIRS:.0f} % des paires.", "",
          "Mesurées sur l'image : paires dont 3 tuiles de décor au moins se prêtent à Lucas-Kanade ; les autres "
          "reprennent l'écart des vecteurs NVOF.", "",
          "| Grille | Hauteur | Paires bonnes / mobiles | Part | Mesurées sur l'image |", "|---|---|---:|---:|---:|"]
    verdicts = {}
    pans = [name for name, typ, _ in clips if typ == "panoramique"]
    moving_pairs_all = moving_set(results, pans)
    for key in sorted(results):
        rows = [(name, r) for name in pans for r in results[key]["paires"].get(name, [])]
        good, moving, photo = gate(rows, moving_pairs_all)
        verdicts[key] = (good / moving) if moving else None
        L.append(f"| {key[0]} × {key[0]} | {key[1]}p | {good} / {moving} | "
                 + (f"{100 * good / moving:.0f} %" if moving else "–") + f" | {photo} |")
    scored = {k: v for k, v in verdicts.items() if v is not None}
    if not scored:
        L += ["", "Verdict impossible : aucun panoramique où la caméra bouge (ajouter un extrait panoramique=…)."]
    else:
        best = max(scored, key=scored.get)
        word = "passe" if scored[best] >= GATE_PAIRS else "ne passe pas"
        L += ["", f"Verdict : la porte {word} (meilleure configuration {best[0]} × {best[0]} en {best[1]}p, "
                  f"{100 * scored[best]:.0f} % des paires)."]
    if pyr:
        L.append(pyramid_report(pyr, *levels))
    text = re.sub(r"(\d)\.(\d)", r"\1,\2", "\n".join(L)) + "\n"
    return text.replace("(porte 2,1)", "(porte 2.1)").replace("Porte deux-un", "Porte 2.1")


# ---------------------------------------------------------------------------------------------------------------

def parse_clip(spec):
    typ, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"--extrait {spec} : TYPE=CHEMIN[@DÉBUT] attendu")
    path, _, start = rest.partition("@") if not os.path.exists(rest) else (rest, "", "")
    return typ, path, start


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outil", required=True, help="l'exécutable nvof_dump")
    p.add_argument("--extrait", action="append", required=True, help="TYPE=CHEMIN[@DÉBUT], plusieurs fois")
    p.add_argument("--grilles", default="4,2,1")
    p.add_argument("--hauteurs", default="540,720,1080")
    p.add_argument("--images", type=int, default=160, help="images lues par extrait")
    p.add_argument("--paires", type=int, default=64, help="paires voisines gardées par extrait (sans les doublons)")
    p.add_argument("--indices", action="store_true", help="indices temporels de NVOF (paires qui se suivent)")
    p.add_argument("--reference", default="4:720", help="GRILLE:HAUTEUR des tableaux par extrait et des vues")
    p.add_argument("--pyramide", default="360:720",
                   help="BAS:HAUT, passe basse dont la caméra guide la passe haute (panoramiques) ; vide : sans")
    p.add_argument("--vues", type=int, default=4, help="vues PNG par extrait")
    p.add_argument("--sortie", required=True)
    args = p.parse_args()

    os.makedirs(args.sortie, exist_ok=True)
    grids = [int(g) for g in args.grilles.split(",")]
    heights = [int(h) for h in args.hauteurs.split(",")]
    ref = tuple(int(x) for x in args.reference.split(":"))
    clips, seen = [], {}
    for spec in args.extrait:
        typ, path, start = parse_clip(spec)
        seen[typ] = seen.get(typ, 0) + 1
        clips.append((typ if seen[typ] == 1 else f"{typ}{seen[typ]}", typ, (path, start)))
    results, times, info, pyr = {}, {}, {}, {}
    levels = tuple(int(x) for x in args.pyramide.split(":")) if args.pyramide else None
    for name, typ, (path, start) in clips:
        pairs = None
        for height in heights:
            frames = decode(path, start, args.images, height)
            if pairs is None:
                pairs, held = moving_pairs(frames, args.paires)
            pairs = [k for k in pairs if k + 1 < len(frames)]
            classes = {k: texture(frames[k]) for k in pairs}
            for grid in grids:
                key = (grid, height)
                try:
                    flows, ms, got = run_tool(args.outil, grid, args.indices, frames, pairs)
                except RuntimeError as e:
                    print(f"{name}, grille {grid}, {height}p ignorée : {e}", file=sys.stderr)
                    continue
                info.update(got)
                times.setdefault(key, []).extend(ms)
                slot = results.setdefault(key, {"paires": {}, "doublons": {}, "couts": {c: np.zeros(256, np.int64)
                                                                                     for c in CLASSES}})
                slot["doublons"][name] = held
                rows, last, shown = [], None, 0
                every = max(1, len(pairs) // max(1, args.vues))
                for i, k in enumerate(pairs):
                    flow, cost = flows[k]
                    ch, cw = cost.shape
                    cls = at_cells(classes[k], grid, ch, cw)
                    for c, label in enumerate(CLASSES):
                        slot["couts"][label] += np.bincount(cost[cls == c], minlength=256)
                    row = {"image": k, "gain_db": warp_gain(frames[k], frames[k + 1], flow, grid),
                           "cout_moyen": float(cost.mean()), "tuiles": [0.0, 0.0, 1.0], "bascule": None}
                    fit = fit_pair(frames[k], frames[k + 1], flow, cost, cls != 0, grid, height)
                    if fit is not None:
                        model, decor, stats = fit
                        row.update(stats)
                        tmap, t = tiles(decor, cls != 0, grid, height)
                        row["tuiles"] = [float((tmap == j).mean()) for j in range(3)]
                        if last is not None and k - last[0] <= 2:
                            row["bascule"] = float((tmap != last[1]).mean())
                        last = (k, tmap)
                        if key == ref and i % every == 0 and shown < args.vues:
                            view(os.path.join(args.sortie, f"{name}_{k:04d}.png"), frames[k], tmap, t, grid, flow,
                                 model, height)
                            shown += 1
                    else:
                        last = None
                    rows.append(row)
                slot["paires"][name] = rows
                print(f"{name} : grille {grid}, {height}p, {len(rows)} paires", file=sys.stderr, flush=True)
        if levels and typ == "panoramique" and pairs:
            try:
                rows, ms = pyramid(args, path, start, pairs, *levels)
                pyr[name] = {"paires": rows, "temps": ms}
            except RuntimeError as e:
                print(f"{name}, pyramide ignorée : {e}", file=sys.stderr)
    if not results:
        raise SystemExit("nvof_dump n'a rendu aucun flux")
    if ref not in results:
        ref = sorted(results)[0]
    text = report(results, times, info, clips, ref, pyr, levels)
    print(text)
    with open(os.path.join(args.sortie, "rapport.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.sortie, "mesures.json"), "w", encoding="utf-8") as f:
        json.dump({"info": info, "temps_ms": {f"{g}:{h}": v for (g, h), v in times.items()},
                   "paires": {f"{g}:{h}": r["paires"] for (g, h), r in results.items()}, "pyramide": pyr}, f,
                  indent=1)


if __name__ == "__main__":
    main()
