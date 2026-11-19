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
- des tuiles de 32 px (en 1080p) classées décor, dessin ou incertain, et leur stabilité d'une paire à l'autre.

Porte : sur les paires des panoramiques où la caméra bouge (0,5 px ou plus à 720p), l'écart moyen du décor au
modèle sous 0,5 px (à 720p) avec au moins 40 % des blocs texturés dans le décor, pour 90 % des paires. Sinon, on
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
import tempfile

import numpy as np

GATE_PX, GATE_SHARE, GATE_PAIRS, MOVING_PX = 0.5, 0.4, 0.9, 0.5
CLASSES = ("aplat", "trait", "grain")
TILES = ("décor", "dessin", "incertain")


# ---------------------------------------------------------------------------------------------------------------
# Images et outil

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


def decode(path, start, frames, height):
    """Les lumas 8 bits (n, h, w) telles qu'un décodeur les donne au lecteur : plan Y réduit par surface."""
    sw, sh = probe(path)
    w = int(round(sw * height / sh / 2)) * 2
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", start] if start else []) + [
        "-i", path, "-frames:v", str(frames), "-an", "-sn",
        "-vf", f"scale={w}:{height}:flags=area,format=yuv420p,extractplanes=y", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, w)


def moving_pairs(frames, limit):
    """Paires voisines (k, k + 1) qui bougent ; les images tenues (anime en 2 ou en 3) sont sautées."""
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1]).mean(axis=(1, 2))
    keep = [k for k, d in enumerate(diffs) if d > 0.5]
    return keep[:limit], len(diffs) - len(keep)


def run_tool(tool, grid, hints, frames, pairs):
    """{k: (flux 2 × ch × cw en px, coût ch × cw)}, [ms], {gpu, cuda, grilles}. Lève RuntimeError si refus."""
    n, h, w = frames.shape
    cw, ch = (w + grid - 1) // grid, (h + grid - 1) // grid
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for k in sorted({k for p in pairs for k in (p, p + 1)}):
            names[k] = os.path.join(tmp, f"{k}.y")
            frames[k].tofile(names[k])
        tasks = "".join(f"{w}\t{h}\t{names[k]}\t{names[k + 1]}\t{os.path.join(tmp, f'{k}.flow')}\n" for k in pairs)
        args = [tool, "--grid", str(grid), "--one-way", "--cost"] + (["--hints"] if hints else [])
        done = subprocess.run(args, input=tasks, text=True, capture_output=True, check=False)
        times, info = [], {}
        for line in done.stderr.splitlines():
            head, _, rest = line.partition(" ")
            if head == "temps_ms":
                times.append(float(rest.split()[1]))
            elif head in ("gpu", "cuda", "grilles"):
                info[head] = rest
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "code %d" % done.returncode)
        out = {}
        for k in pairs:
            raw = np.fromfile(os.path.join(tmp, f"{k}.flow"), np.uint8)
            flow = raw[: ch * cw * 4].view(np.int16).astype(np.float32).reshape(ch, cw, 2).transpose(2, 0, 1) / 32.0
            out[k] = (flow, raw[ch * cw * 4:].reshape(ch, cw))
        return out, times, info


# ---------------------------------------------------------------------------------------------------------------
# Mesures

def box(x, r):
    """Moyenne sur une fenêtre (2r + 1)², bords répétés."""
    p = np.pad(x.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: x.shape[0], : x.shape[1]]


def texture(a):
    """Classe par pixel : 0 aplat, 1 trait (près d'un bord net), 2 grain (texture sans bord net)."""
    f = a.astype(np.float32)
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2
    gy[1:-1] = (f[2:] - f[:-2]) / 2
    trait = box((np.hypot(gx, gy) > 16).astype(np.float32), 2) > 0
    std = np.sqrt(np.maximum(box(f * f, 2) - box(f, 2) ** 2, 0))
    cls = np.where(trait, 1, np.where(std < 1.0, 0, 2))
    return cls.astype(np.uint8)


def at_cells(img, grid, ch, cw):
    ys = np.minimum(np.arange(ch) * grid + grid // 2, img.shape[0] - 1)
    xs = np.minimum(np.arange(cw) * grid + grid // 2, img.shape[1] - 1)
    return img[np.ix_(ys, xs)]


def warp_gain(a, b, flow, grid):
    """PSNR de b recalé sur a par le flux a → b (a(x) ≈ b(x + F)), moins celui de b tel quel."""
    h, w = a.shape
    f = np.repeat(np.repeat(flow, grid, 1), grid, 2)[:, :h, :w]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = np.clip(x + f[0], 0, w - 1.001), np.clip(y + f[1], 0, h - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = sx - x0, sy - y0
    bf = b.astype(np.float32)
    top = bf[y0, x0] * (1 - fx) + bf[y0, x0 + 1] * fx
    bot = bf[y0 + 1, x0] * (1 - fx) + bf[y0 + 1, x0 + 1] * fx
    warped = top * (1 - fy) + bot * fy
    m = 16
    inner = (slice(m, h - m), slice(m, w - m))

    def psnr(u):
        mse = float(np.mean((u[inner] - a[inner].astype(np.float32)) ** 2))
        return 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))

    return psnr(warped) - psnr(bf)


def camera(flow, cost, textured, grid, height):
    """Affine pondérée et robuste. Rend (modèle 2 × ch × cw, masque décor, mesures) ou None si trop peu de blocs."""
    _, ch, cw = flow.shape
    if textured.sum() < 32:
        return None
    k = height / 720.0  # px natifs par px à 720p
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float64)
    xn, yn = (xx * grid - 0.5 * cw * grid) / (cw * grid), (yy * grid - 0.5 * ch * grid) / (ch * grid)
    X = np.stack([np.ones_like(xn), xn, yn], -1)[textured]
    u, v = flow[0][textured].astype(np.float64), flow[1][textured].astype(np.float64)
    c = cost[textured].astype(np.float64)
    w = 1.0 / (1.0 + c / max(1.0, float(np.median(c))))
    pu, pv = np.array([np.median(u), 0, 0]), np.array([np.median(v), 0, 0])  # départ : la translation majoritaire
    for _ in range(8):
        r = np.hypot(u - X @ pu, v - X @ pv)
        inl = r < np.clip(3 * 1.4826 * np.median(r[r < 2 * k]) if (r < 2 * k).any() else 2 * k, 0.75 * k, 2 * k)
        if inl.sum() < 16:
            return None
        sw = np.sqrt(w[inl])[:, None]
        pu = np.linalg.lstsq(X[inl] * sw, u[inl] * sw[:, 0], rcond=None)[0]
        pv = np.linalg.lstsq(X[inl] * sw, v[inl] * sw[:, 0], rcond=None)[0]
    Xall = np.stack([np.ones_like(xn), xn, yn], -1)
    model = np.stack([Xall @ pu, Xall @ pv])
    r = np.hypot(u - X @ pu, v - X @ pv)
    decor = np.zeros((ch, cw), bool)
    decor[textured] = inl
    return model, decor, {
        "camera_px": float(np.hypot(*model).mean() / k),
        "residu_px": float(r[inl].mean() / k),
        "part_decor": float(inl.mean()),
        "affine": [round(float(x), 4) for x in (*pu, *pv)],
    }


def tiles(decor, textured, grid, height):
    """Tuiles de 32 px en 1080p : 0 décor, 1 dessin, 2 incertain (peu de texture ou mélange)."""
    t = max(1, int(round(32 * height / 1080 / grid)))
    ch, cw = decor.shape
    th, tw = (ch + t - 1) // t, (cw + t - 1) // t
    pad = lambda m: np.pad(m, ((0, th * t - ch), (0, tw * t - cw))).reshape(th, t, tw, t).sum((1, 3))  # noqa: E731
    tex, dec = pad(textured.astype(np.int32)), pad(decor.astype(np.int32))
    frac = dec / np.maximum(tex, 1)
    out = np.full((th, tw), 2, np.uint8)
    solid = tex >= 0.25 * t * t
    out[solid & (frac >= 0.7)] = 0
    out[solid & (frac <= 0.3)] = 1
    return out, t


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


# ---------------------------------------------------------------------------------------------------------------
# Rapport

def pct(hist, q):
    c = np.cumsum(hist)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] else float("nan")


def gate(rows):
    moving = [r for r in rows if r.get("camera_px", 0) >= MOVING_PX]
    good = [r for r in moving if r["residu_px"] < GATE_PX and r["part_decor"] >= GATE_SHARE]
    return len(good), len(moving)


def report(results, times, info, clips, ref):
    L = [f"# Banc NVOF hors lecteur (porte 2.1)", "",
         f"Carte : {info.get('gpu', '?')}, pilote CUDA {info.get('cuda', '?')}, grilles acceptées : "
         f"{info.get('grilles', '?')}. Préréglage slow, sens avant, carte de coût allumée.", "",
         "## Temps GPU d'un appel (ms)", "", "| Grille | Hauteur | Médiane | 95e centile | Appels |",
         "|---|---|---:|---:|---:|"]
    for (g, hgt), ms in sorted(times.items()):
        ms = ms[2:] if len(ms) > 4 else ms  # les premiers appels d'une session paient l'initialisation
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
          "Caméra : mouvement moyen du modèle ; résidu : écart moyen du décor au modèle ; px à 720p. Tuiles : "
          "part décor / dessin / incertain ; bascule : tuiles qui changent de classe entre deux paires voisines.", "",
          "| Extrait | Paires (tenues) | Gain | Coût moyen | Caméra | Part décor | Résidu | Tuiles | Bascule |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name, _, _ in clips:
        rows = results.get(ref, {"paires": {}})["paires"].get(name, [])
        fit = [r for r in rows if "residu_px" in r]
        if not rows:
            continue
        med = lambda k: np.median([r[k] for r in fit]) if fit else float("nan")  # noqa: E731
        share = np.mean([r["tuiles"] for r in rows], 0)
        flips = [r["bascule"] for r in rows if r.get("bascule") is not None]
        L.append(f"| {name} | {len(rows)} ({results[ref]['tenues'][name]}) | "
                 f"{np.mean([r['gain_db'] for r in rows]):.2f} | {np.mean([r['cout_moyen'] for r in rows]):.1f} | "
                 f"{med('camera_px'):.2f} | {100 * med('part_decor'):.0f} % | {med('residu_px'):.2f} | "
                 + " / ".join(f"{100 * s:.0f}" for s in share)
                 + f" | {100 * np.mean(flips) if flips else float('nan'):.1f} % |")
    L += ["", "## Porte deux-un", "",
          f"Paires des panoramiques où la caméra bouge (≥ {MOVING_PX} px) : résidu du décor sous {GATE_PX} px et "
          f"part du décor ≥ {100 * GATE_SHARE:.0f} %, sur {100 * GATE_PAIRS:.0f} % des paires.", "",
          "| Grille | Hauteur | Paires bonnes / mobiles | Part |", "|---|---|---:|---:|"]
    verdicts = {}
    for key in sorted(results):
        rows = [r for name, typ, _ in clips if typ == "panoramique" for r in results[key]["paires"].get(name, [])]
        good, moving = gate(rows)
        verdicts[key] = (good / moving) if moving else None
        L.append(f"| {key[0]} × {key[0]} | {key[1]}p | {good} / {moving} | "
                 + (f"{100 * good / moving:.0f} %" if moving else "–") + " |")
    scored = {k: v for k, v in verdicts.items() if v is not None}
    if not scored:
        L += ["", "Verdict impossible : aucun panoramique où la caméra bouge (ajouter un extrait panoramique=…)."]
    else:
        best = max(scored, key=scored.get)
        word = "passe" if scored[best] >= GATE_PAIRS else "ne passe pas"
        L += ["", f"Verdict : la porte {word} (meilleure configuration {best[0]} × {best[0]} en {best[1]}p, "
                  f"{100 * scored[best]:.0f} % des paires)."]
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
    p.add_argument("--paires", type=int, default=64, help="paires qui bougent gardées par extrait")
    p.add_argument("--indices", action="store_true", help="indices temporels de NVOF (paires qui se suivent)")
    p.add_argument("--reference", default="4:720", help="GRILLE:HAUTEUR des tableaux par extrait et des vues")
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
    results, times, info = {}, {}, {}
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
                slot = results.setdefault(key, {"paires": {}, "tenues": {}, "couts": {c: np.zeros(256, np.int64)
                                                                                     for c in CLASSES}})
                slot["tenues"][name] = held
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
                    fit = camera(flow, cost, cls != 0, grid, height)
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
    if not results:
        raise SystemExit("nvof_dump n'a rendu aucun flux")
    if ref not in results:
        ref = sorted(results)[0]
    text = report(results, times, info, clips, ref)
    print(text)
    with open(os.path.join(args.sortie, "rapport.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.sortie, "mesures.json"), "w", encoding="utf-8") as f:
        json.dump({"info": info, "temps_ms": {f"{g}:{h}": v for (g, h), v in times.items()},
                   "paires": {f"{g}:{h}": r["paires"] for (g, h), r in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
