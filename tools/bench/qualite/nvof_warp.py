#!/usr/bin/env python3
"""Le flux optique matériel de NVIDIA (NVOF) vaut-il un guide sur l'anime ? Warp seul, sans réseau correcteur.

  python3 -I tools/bench/qualite/nvof_warp.py --video EXTRAIT_1080P.mkv --reseau rife_v4.26.onnx \\
      --outil build/nvof_dump/nvof_dump [--gmfss roue/vsgmfss_fortuna --device cuda] [--lpips] \\
      --json nvof.json --md nvof.md

Étape 1 de l'étude « interpolation guidée par le flux NVIDIA » (17 novembre 2026). Mêmes suites que
gmfss_instants.py : des images toutes différentes d'un même plan, k et k+N forment la paire, les images du milieu
sont les vérités à i/N. Pour chaque instant, l'image se fabrique en déformant a et b par un flux, puis en les
mêlant à poids fixes (1 − t, t), sans masque appris :

  nvof_g<G>_<H>p   flux NVOF (nvof_dump, grille G × G, calculé à la hauteur H puis ramené à la taille de travail),
                   porté à l'instant t par la loi de Super SloMo (F_t→0 = −(1−t)t·F01 + t²·F10,
                   F_t→1 = (1−t)²·F01 − t(1−t)·F10) ;
  gmflow_flot      le flux de GMFlow (la paire de GMFSS union), même loi : un estimateur fort, la cible à approcher ;
  rife_flot        le flux de RIFE 4.26 à l'instant, poids fixes (son masque retiré) ;
  rife             RIFE 4.26 tel que le lecteur le joue (flux et masque) : le niveau d'aujourd'hui.

Seuil de l'étude : sur les paires dont le mouvement dépasse 30 px à 720p, la meilleure configuration NVOF doit
rester à 1 dB ou moins de gmflow_flot (à défaut de GMFSS, de rife_flot). Sinon le flux NVOF ne guidera rien.
Le tableau des temps donne, par grille et par hauteur, la médiane d'un appel au matériel (un sens).

Il faut python3 avec numpy, onnx, onnxruntime et ffmpeg ; torch et la roue vsgmfss_fortuna pour gmflow_flot
(voir gmfss_instants.py) ; lpips pour --lpips. nvof_dump se construit depuis tools/nvof_interp/nvof_dump.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "gmfss"))
import gmfss_instants as gi  # noqa: E402
import mesures  # noqa: E402
import rife_taille as rt  # noqa: E402

LARGE = 30.0  # px à 720p : les paires où RIFE peine, celles que le flux NVOF doit aider
THRESHOLD_DB = 1.0
EDGES = (0, 15, 30, 45, 60, 90, 1e9)


# ---------------------------------------------------------------------------------------------------------------
# Flux NVOF (nvof_dump)

def luma8(img):
    """Luma BT.709 en plage limitée, 8 bits : ce qu'un fichier vidéo donne au lecteur (precompute_nvof.py)."""
    y = img[0] * 0.2126 + img[1] * 0.7152 + img[2] * 0.0722
    return np.clip(np.rint(16.0 + 219.0 * y), 0, 255).astype(np.uint8)


def at_height(img, height):
    if img.shape[1] == height:
        return img
    width = int(round(img.shape[2] * height / img.shape[1] / 2)) * 2
    return rt.resize(img, width, height, rt._area_axis)


def cells(n, grid):
    return (n + grid - 1) // grid


def to_planes(raw, w, h, grid):
    """Les deux flux S10.5 d'un fichier de nvof_dump : (F01, F10), chacun 2 × cellules en pixels de la hauteur H."""
    count = cells(w, grid) * cells(h, grid) * 2
    vectors = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32.0
    if vectors.size != 2 * count:
        raise ValueError(f"{vectors.size} valeurs, {2 * count} attendues (grille {grid}, {w}x{h})")
    shape = (cells(h, grid), cells(w, grid), 2)
    return (vectors[:count].reshape(shape).transpose(2, 0, 1), vectors[count:].reshape(shape).transpose(2, 0, 1))


def run_tool(tool, grid, preset, pairs):
    """pairs : [(clé, luma a, luma b)]. Rend ({clé: (F01, F10)}, {« LxH »: [ms par appel]})."""
    with tempfile.TemporaryDirectory() as tmp:
        tasks, meta = [], {}
        for n, (key, a, b) in enumerate(pairs):
            h, w = a.shape
            pa, pb, out = (os.path.join(tmp, f"{n}{s}") for s in ("a.raw", "b.raw", ".flow"))
            a.tofile(pa)
            b.tofile(pb)
            tasks.append((w, h, "\t".join((str(w), str(h), pa, pb, out)) + "\n"))
            meta[out] = (key, w, h)
        tasks.sort(key=lambda t: (t[0], t[1]))  # une session NVOF par taille
        done = subprocess.run([tool, "--grid", str(grid), "--preset", preset], input="".join(t[2] for t in tasks),
                              text=True, capture_output=True, check=False)
        times = {}
        for line in done.stderr.splitlines():
            if line.startswith("temps_ms "):
                _, size, ms = line.split()
                times.setdefault(size, []).append(float(ms))
        if done.returncode != 0:
            errors = [x for x in done.stderr.splitlines() if not x.startswith("temps_ms ")]
            raise RuntimeError("\n".join(errors).strip() or f"nvof_dump : code {done.returncode}")
        flows = {}
        for line in done.stdout.splitlines():
            out = line.removeprefix("ok ").strip()
            key, w, h = meta[out]
            with open(out, "rb") as f:
                flows[key] = to_planes(f.read(), w, h, grid)
        missing = [meta[o][0] for o in meta if meta[o][0] not in flows]
        if missing:
            raise RuntimeError(f"nvof_dump : {len(missing)} paire(s) sans flux")
        return flows, times


def to_work(flow, width, height, source_height):
    """Un flux de cellules (2 × h × w, pixels de la hauteur source) à la taille de travail, en pixels de travail."""
    up = rt.resize(flow, width, height, rt._linear_axis)
    return up * (height / source_height)


# ---------------------------------------------------------------------------------------------------------------
# Warp seul

def slomo(f01, f10, t):
    """Les flux de l'instant t vers a et vers b, en 4 plans (la variante « mouvement » du lecteur, warp.h)."""
    ft0 = -(1 - t) * t * f01 + t * t * f10
    ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
    return np.concatenate([ft0, ft1], 0).astype(np.float32)


def warp_fixed(a, b, flow4, t):
    return rt.blend(a, b, flow4, np.full(a.shape[1:], 1.0 - t, np.float32))


def epe(f, ref):
    """Erreur de point final de f contre ref : moyenne (px) et part des vecteurs faux de plus de 3 px."""
    d = np.sqrt(((f - ref) ** 2).sum(0))
    return float(d.mean()), float((d > 3.0).mean())


# ---------------------------------------------------------------------------------------------------------------

def configs(grids, heights):
    return [(g, h) for g in grids for h in heights]


def name(grid, height):
    return f"nvof_g{grid}_{height}p"


def measure_pair(win, height, rife, flows, gmfss_nets, lpips_model, keys):
    work = [gi.work(x, height) for x in win]
    n = len(work) - 1
    a, b = work[0], work[n]
    _, h, w = a.shape
    _, motion = gi.rife_at(*rife, a, b, 0.5)
    row = {"mouvement": motion, "ecart": n}
    pair_flows = {}
    for key in keys:
        grid, source_h = key
        f01, f10 = flows[key]
        pair_flows[name(*key)] = (to_work(f01, w, h, source_h), to_work(f10, w, h, source_h))
    if gmfss_nets is not None:
        (pa, window), (pb, _) = gi.padded(a), gi.padded(b)
        g = gi.Gmfss(gmfss_nets, pa, pb)
        crop = lambda x: x[0].float().cpu().numpy()[window]  # noqa: E731
        ref01, ref10 = crop(g.f01), crop(g.f10)
        pair_flows["gmflow_flot"] = (ref01, ref10)
        for label, (f01, _) in pair_flows.items():
            if label.startswith("nvof_"):
                row[f"epe_{label}"], row[f"faux3_{label}"] = epe(f01, ref01)
    for i in range(1, n):
        t = i / n
        rife_out, rflow, _ = gi.rife_flow(*rife, a, b, t)
        outs = {"rife": rife_out, "rife_flot": warp_fixed(a, b, rflow, t)}
        for label, (f01, f10) in pair_flows.items():
            outs[label] = warp_fixed(a, b, slomo(f01, f10, t), t)
        for label, out in outs.items():
            out = np.clip(out, 0, 1)
            row[f"{label}_{i}"] = mesures.psnr(work[i], out)
            row[f"pire_bloc_{label}_{i}"] = rt.worst_block(work[i], out)
            if lpips_model is not None:
                row[f"lpips_{label}_{i}"] = gi.perceptual(lpips_model, work[i], out)
    return row


def methods_of(rows):
    first = rows[0]
    return [m for m in dict.fromkeys(k[: -len("_1")] for k in first if k.endswith("_1")
                                     and not k.startswith(("pire_bloc_", "lpips_")))]


def mean_of(rows, method, prefix=""):
    vals = [r[f"{prefix}{method}_{i}"] for r in rows for i in gi.instants(r)]
    return float(np.mean(vals)) if vals else float("nan")


def summary(rows, height, times, preset):
    methods = methods_of(rows)
    gap = rows[0]["ecart"]
    lines = [f"Taille de travail {height}p, {len(rows)} paires (k, k + {gap}), "
             f"{sum(len(gi.instants(r)) for r in rows)} instants ; NVOF en préréglage {preset}.", "",
             "PSNR moyen (dB) par mouvement de la paire (px à 720p, mesuré par RIFE) :", "",
             "| Mouvement | Instants | " + " | ".join(methods) + " |", "|---|---:|" + "---:|" * len(methods)]
    for lo, hi in list(zip(EDGES, EDGES[1:])) + [(0, 1e9)]:
        group = [r for r in rows if lo <= r["mouvement"] < hi]
        if not group:
            continue
        label = "toutes" if (lo, hi) == (0, 1e9) else (f"{lo} à {hi}" if hi < 1e9 else f"{lo} et plus")
        lines.append(f"| {label} | {sum(len(gi.instants(r)) for r in group)} | "
                     + " | ".join(f"{mean_of(group, m):.2f}" for m in methods) + " |")
    if f"lpips_{methods[0]}_1" in rows[0]:
        lines += ["", "LPIPS moyen (plus bas : plus proche à l'œil) : "
                  + ", ".join(f"{m} {mean_of(rows, m, 'lpips_'):.4f}" for m in methods)]
    lines += ["", "Pire bloc 64 × 64, moyenne (dB) : "
              + ", ".join(f"{m} {mean_of(rows, m, 'pire_bloc_'):.2f}" for m in methods)]
    nvof = [m for m in methods if m.startswith("nvof_")]
    if f"epe_{nvof[0]}" in rows[0]:
        lines += ["", "Flux a → b contre GMFlow, à la taille de travail (toutes paires / plus de 30 px) :", "",
                  "| Configuration | Écart moyen (px) | Vecteurs à plus de 3 px | Écart moyen, > 30 px | > 3 px, > 30 px |",
                  "|---|---:|---:|---:|---:|"]
        large = [r for r in rows if r["mouvement"] >= LARGE] or rows
        for m in nvof:
            lines.append(f"| {m} | {np.mean([r[f'epe_{m}'] for r in rows]):.2f} | "
                         f"{100 * np.mean([r[f'faux3_{m}'] for r in rows]):.1f} % | "
                         f"{np.mean([r[f'epe_{m}'] for r in large]):.2f} | "
                         f"{100 * np.mean([r[f'faux3_{m}'] for r in large]):.1f} % |")
    lines += ["", "Temps d'un appel NVOF (un sens), médiane et 95e centile (ms) :", "",
              "| Grille | Taille | Médiane | 95e centile | Appels |", "|---|---|---:|---:|---:|"]
    for (grid, size), ms in sorted(times.items()):
        ms = ms[2:] if len(ms) > 4 else ms  # les premiers appels d'une session paient l'initialisation
        lines.append(f"| {grid} × {grid} | {size} | {np.median(ms):.2f} | {np.percentile(ms, 95):.2f} | {len(ms)} |")
    large = [r for r in rows if r["mouvement"] >= LARGE]
    target = "gmflow_flot" if "gmflow_flot" in methods else "rife_flot"
    if large and nvof:
        best = max(nvof, key=lambda m: mean_of(large, m))
        gapdb = mean_of(large, best) - mean_of(large, target)
        verdict = "passe" if gapdb >= -THRESHOLD_DB else "ne passe pas"
        lines += ["", f"Verdict ({len(large)} paires au-delà de {LARGE:.0f} px) : meilleure configuration {best}, "
                  f"{gapdb:+.2f} dB face à {target} : le seuil de −{THRESHOLD_DB:.0f} dB {verdict}."]
    else:
        lines += ["", f"Verdict impossible : aucune paire au-delà de {LARGE:.0f} px (prendre un extrait qui bouge plus, "
                  "ou --ecart 6)."]
    return re.sub(r"(\d)\.(\d)", r"\1,\2", "\n".join(lines)) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, help="un extrait 1080p (24 i/s)")
    p.add_argument("--reseau", required=True, help="rife_v4.26.onnx (11 plans)")
    p.add_argument("--outil", required=True, help="l'exécutable nvof_dump")
    p.add_argument("--grilles", default="4,2,1", help="grilles NVOF (1 et 2 : Ampere et après)")
    p.add_argument("--hauteurs-flux", default="720,1080", help="hauteurs où NVOF calcule le flux")
    p.add_argument("--preset", default="slow", choices=("slow", "medium", "fast"))
    p.add_argument("--gmfss", help="le dossier vsgmfss_fortuna extrait de la roue (ajoute gmflow_flot)")
    p.add_argument("--device", default="cpu", help="cpu ou cuda (GMFSS)")
    p.add_argument("--lpips", action="store_true", help="ajoute LPIPS (pip install lpips)")
    p.add_argument("--hauteur", type=int, default=720, help="taille de travail")
    p.add_argument("--ecart", type=int, default=3, help="images d'écart de la paire (3 : vérités à ⅓ et ⅔)")
    p.add_argument("--suites", type=int, default=16)
    p.add_argument("--images", type=int, default=740, help="images lues au début de l'extrait")
    p.add_argument("--json")
    p.add_argument("--md")
    args = p.parse_args()

    wins = gi.suites(rt.decode(args.video, args.images), args.suites, args.ecart)
    if not wins:
        raise SystemExit("aucune suite d'images toutes différentes dans l'extrait")
    grids = [int(g) for g in args.grilles.split(",")]
    heights = [int(h) for h in args.hauteurs_flux.split(",")]
    flows, times, keys = {}, {}, []
    for grid in grids:
        for height in heights:
            pairs = [((k, (grid, height)), luma8(at_height(win[0], height)), luma8(at_height(win[-1], height)))
                     for k, win in wins]
            try:
                got, spent = run_tool(args.outil, grid, args.preset, pairs)
            except RuntimeError as e:
                print(f"grille {grid}, {height}p ignorée : {e}", file=sys.stderr)
                continue
            keys.append((grid, height))
            flows.update(got)
            for size, ms in spent.items():
                times.setdefault((grid, size), []).extend(ms)
    if not keys:
        raise SystemExit("nvof_dump n'a rendu aucun flux")
    gmfss_nets = None
    if args.gmfss:
        import torch

        torch.set_grad_enabled(False)
        gmfss_nets = gi.load(args.gmfss, args.device, 1.0)
    lpips_model = None
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    rife = rt.motion_session(args.reseau)
    rows = []
    for k, win in wins:
        pair = {key: flows[(k, key)] for key in keys}
        rows.append({"image": k, **measure_pair(win, args.hauteur, rife, pair, gmfss_nets, lpips_model, keys)})
        print(json.dumps({key: round(v, 2) if isinstance(v, float) else v for key, v in rows[-1].items()}), flush=True)
    text = summary(rows, args.hauteur, times, args.preset)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"lignes": rows, "temps_ms": {f"{g}:{s}": v for (g, s), v in times.items()}}, out, indent=1)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


if __name__ == "__main__":
    main()
