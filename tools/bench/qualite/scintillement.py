#!/usr/bin/env python3
"""Mesure du scintillement : deux sorties du lecteur (ou plus) sur les mêmes images, laquelle tremble le moins.

  python3 -I tools/bench/qualite/scintillement.py SANS_TF.mkv AVEC_TF.mkv --source EXTRAIT.mkv \\
      --flux nvof --nvof-dump build/nvof_dump/nvof_dump [--porte 0.9] [--md r.md] [--json r.json]
  python3 -I tools/bench/qualite/scintillement.py A.mkv B.mkv --source EXTRAIT.mkv --flux rife --rife rife_v4.25.onnx

Les sorties sont des exports du lecteur (export-lossless, sans interpolation) d'un même extrait, et la source est
cet extrait (--source-debut s'il commence plus tôt que les exports). L'outil refuse des sorties d'une autre cadence
que la source ou décalées d'images, et signale des longueurs différentes. Pour chaque paire d'images N, N+1 :

  - le flux vient de la source, jamais d'une sortie (il serait juge et partie), et sert à toutes les sorties.
    NVOF (nvof_dump, le flux arrière N+1 → N et l'aller pour la cohérence) ou, sans carte NVIDIA, RIFE sur
    processeur. Les deux images sont déplacées vers le milieu de la paire, chacune de la moitié du chemin : le
    flou du rééchantillonnage est alors le même des deux côtés et ne compte pas comme un écart ;
  - une translation d'ensemble sous-pixel corrige ensuite le flux sur la source (Lucas-Kanade global) : RIFE ne voit
    pas un mouvement de moins d'un pixel par image, NVOF s'arrête au quart de pixel, et un panoramique lent mal
    recalé compterait comme du scintillement ;
  - le flux est « sûr » là où il recale la source elle-même (écart flouté sous --seuil niveaux), où il ne lit pas
    hors de l'image et, avec NVOF, où l'aller et le retour s'annulent. Le même masque vaut pour toutes les sorties ;
  - la paire est une coupe si moins de 30 % de l'image est sûre, un fondu ou un éclair si la luminance moyenne
    recalée de la source change de plus de --fondu niveau : les deux sont exclues, une coupe ouvre un nouveau plan ;
  - le score d'une sortie est l'écart moyen de luminance (niveaux 8 bits, lus en 16 bits : un export 10 bits garde
    ses quarts de niveau) entre ses deux images recalées, sur le masque sûr. Le grain et la compression de la source y entrent pour toutes les sorties pareil : seul l'écart
    entre sorties compte, jamais un score seul contre la source.

Le rapport est par plan (jamais une moyenne de la vidéo, où un plan raté se noie) : score de chaque sortie et la
meilleure. --porte F : sortie en erreur si la dernière sortie n'est pas devant la première sur au moins F des plans
(la porte 0.1 du guide : SPAN sans TemporalFix, puis avec, --porte 0.9).

Sans --source, le flux vient de la première sortie : à éviter, il la favorise.
Il faut python3 avec numpy, ffmpeg et ffprobe ; onnx et onnxruntime pour --flux rife. Le recalage est dans
recalage.py ; les paires d'un lot se mesurent en parallèle (WORKERS fils), les vidéos se décodent d'avance.
"""
import argparse
import concurrent.futures
import functools
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalage import Aligned, Sampler, box, grid, luma, scale  # noqa: E402,F401  (box, scale : test_banc.py)

CUT_SAFE = 0.30  # part sûre minimale d'une paire du même plan
GRID = 4  # NVOF : un vecteur par bloc de 4 × 4 (nvof_dump)
# Paires mesurées en parallèle, sur des fils : numpy lâche le GIL sur les grands tableaux. Au plus 8 (mémoire :
# une centaine de Mo par paire en 1080p).
WORKERS = max(1, min(8, os.cpu_count() or 1))


# Les fournisseurs de flux : refs (liste d'images H × W × 3 uint8 de la source) → par paire, une fonction sans
# argument qui rend son Aligned. Le gros du travail (les flux en pleine taille) se fait ainsi dans le fil qui mesure
# la paire, et la mémoire d'une paire repart avec elle.

def _nvof_aligned(v, w, h):
    """L'Aligned d'une paire depuis le fichier de nvof_dump (int16 S10.5, l'aller puis le retour)."""
    cw, ch = (w + GRID - 1) // GRID, (h + GRID - 1) // GRID
    v = v.astype(np.float32) / 32.0
    # nvof_dump : l'aller sur la grille de N, pointant dans N+1, puis le retour sur celle de N+1.
    fwd = scale(v[:ch * cw * 2].reshape(ch, cw, 2).transpose(2, 0, 1), w, h, GRID)
    bwd = scale(v[ch * cw * 2:].reshape(ch, cw, 2).transpose(2, 0, 1), w, h, GRID)
    gx, gy = grid(h, w)
    # Aller et retour : le vecteur arrière, suivi du vecteur aller là où il mène, revient au point.
    at = Sampler(gx + bwd[0], gy + bwd[1], h, w)
    err = np.hypot(bwd[0] + at(fwd[0]), bwd[1] + at(fwd[1]))
    # Chaque image fait la moitié du chemin vers le milieu de la paire (même flou des deux côtés).
    return Aligned(0.5 * bwd, -0.5 * bwd, err <= 1.0 + 0.05 * np.hypot(bwd[0], bwd[1]))


def flux_nvof(tool):
    def run(refs):
        h, w = refs[0].shape[:2]
        cw, ch = (w + GRID - 1) // GRID, (h + GRID - 1) // GRID
        with tempfile.TemporaryDirectory() as tmp:
            raws = []
            for i, ref in enumerate(refs):
                path = os.path.join(tmp, f"{i}.raw")
                np.clip(np.rint(16.0 + 219.0 / 255.0 * luma(ref)), 0, 255).astype(np.uint8).tofile(path)
                raws.append(path)
            outs = [os.path.join(tmp, f"{i}.flow") for i in range(len(refs) - 1)]
            tasks = "".join(f"{w}\t{h}\t{raws[i]}\t{raws[i + 1]}\t{o}\n" for i, o in enumerate(outs))
            done = subprocess.run([tool], input=tasks, text=True, capture_output=True, check=False)
            if done.returncode != 0:
                raise SystemExit(f"nvof_dump : {done.stderr.strip()}")
            result = []
            for o in outs:
                if not os.path.exists(o) or os.path.getsize(o) != ch * cw * 8:
                    raise SystemExit(f"nvof_dump : flux absent ou tronqué pour la paire {len(result)} du lot")
                result.append(functools.partial(_nvof_aligned, np.fromfile(o, np.int16), w, h))
            return result

    return run


def flux_rife(model, height):
    import modeles
    import rife_taille

    sess, names = rife_taille.motion_session(model)
    if sess.get_inputs()[0].shape[1] != 11:
        raise SystemExit(f"{model} : un RIFE à 11 plans (v1) est attendu, comme rife_taille.py")

    def run(refs):
        h, w = refs[0].shape[:2]
        fh = min(h, height)
        fw = int(round(w * fh / h / 2)) * 2
        # Réduction par moyenne de surface, comme le filtre Area du lecteur devant RIFE.
        imgs = [r.transpose(2, 0, 1).astype(np.float32) / 255.0 for r in refs]
        if (fw, fh) != (w, h):
            imgs = [rife_taille.resize(i, fw, fh, rife_taille._area_axis) for i in imgs]
        def aligned(flow):
            flow = scale(flow, w, h)
            flow[0::2] *= w / fw
            flow[1::2] *= h / fh
            return Aligned(flow[0:2], flow[2:4])

        # ONNX Runtime a ses propres fils : les paires passent dans RIFE l'une après l'autre.
        return [functools.partial(aligned, rife_taille.estimate(sess, names, a, b, 2 * modeles.ALIGN)[1])
                for a, b in zip(imgs, imgs[1:])]

    return run


@functools.lru_cache(maxsize=None)
def _pool():
    return concurrent.futures.ThreadPoolExecutor(WORKERS)


def measure_pairs(refs, outs, flux, seuil, fondu):
    """Les paires d'un lot, mesurées en parallèle : (« coupe » | « fondu » | « ok », {sortie : (moyen, 95e
    centile)}). flux rend par paire un Aligned, ou une fonction qui le rend."""
    pairs = flux(refs)
    ys = list(_pool().map(luma, refs))

    def one(k):
        al = pairs[k]() if callable(pairs[k]) else pairs[k]
        al.refine(ys[k], ys[k + 1])
        ra, rb = al.pair(ys[k], ys[k + 1])
        safe = al.inside & (box(np.abs(ra - rb)) < seuil)
        if safe.mean() < CUT_SAFE:
            return ("coupe", {})
        if abs(float(ra[safe].mean()) - float(rb[safe].mean())) > fondu:
            return ("fondu", {})
        scores = {}
        for name, frames in outs.items():
            oa, ob = al.pair(frames[k], frames[k + 1])
            d = np.abs(oa - ob)[safe]
            scores[name] = (float(d.mean()), float(np.percentile(d, 95)))
        return ("ok", scores)

    return list(_pool().map(one, range(len(pairs))))


def shots(rows, names, min_pairs):
    """Les paires regroupées en plans (une coupe ferme le plan), chacun avec le score moyen de chaque sortie."""
    out, cur = [], None

    def close():
        if cur and len(cur["ok"]) >= min_pairs:
            sc = {n: float(np.mean([s[n][0] for s in cur["ok"]])) for n in names}
            p95 = {n: float(np.mean([s[n][1] for s in cur["ok"]])) for n in names}
            out.append({"plan": len(out) + 1, "debut": cur["debut"], "fin": cur["fin"], "paires": len(cur["ok"]),
                        "exclues": cur["exclues"], "scores": sc, "p95": p95, "meilleure": min(sc, key=sc.get)})

    for k, (kind, scores) in enumerate(rows):
        if kind == "coupe":
            close()
            cur = None
            continue
        if cur is None:
            cur = {"debut": k, "fin": k + 1, "ok": [], "exclues": 0}
        cur["fin"] = k + 1
        if kind == "ok":
            cur["ok"].append(scores)
        else:
            cur["exclues"] += 1
    close()
    return out


def offset(refs, frames, reach=2):
    """Le décalage d'une sortie sur la source, en images, dans ±reach : celui où les images se ressemblent le plus
    (corrélation de la luminance, insensible au grade, qu'il soit SDR ou PQ). 0 si rien ne tranche (dessin tenu)."""
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).mean() / max(1e-6, np.sqrt((a * a).mean() * (b * b).mean())))

    n = min(len(refs), len(frames))
    ys = [luma(r[::4, ::4]) for r in refs[:n]]  # un pixel sur seize suffit à reconnaître l'image
    fs = [f[::4, ::4] for f in frames[:n]]
    score = {o: np.mean([corr(ys[i], fs[i + o]) for i in range(max(0, -o), min(n, n - o))])
             for o in range(-reach, reach + 1) if n - abs(o) >= 4}
    if not score:
        return 0
    best = max(score, key=score.get)
    return best if score[best] > score[0] + 0.02 else 0


def run(refs, outs, flux, seuil=10.0, fondu=0.75, chunk=32, min_pairs=8):
    """refs : itérable d'images de la source (H × W × 3 uint8) ; outs : {nom : itérable de luminances H × W}.
    Lit par lots de chunk images (le lot suivant reprend la dernière), rend (plans, décompte des paires). Une sortie
    décalée sur la source arrête tout (ses paires seraient recalées par le flux d'une autre paire) ; des longueurs
    différentes se lisent dans le décompte (« inegales »)."""
    names = list(outs)
    iters = [iter(refs)] + [iter(outs[n]) for n in names]
    buf = [[] for _ in iters]
    rows = []
    first = True
    while True:
        for b, it in zip(buf, iters):
            while len(b) < chunk:
                frame = next(it, None)
                if frame is None:
                    break
                b.append(frame)
        n = min(len(b) for b in buf)
        if n < 2:
            break
        if first:
            for i, nm in enumerate(names):
                shift = offset(buf[0][:n], buf[i + 1][:n])
                if shift:
                    # frames[i + shift] ressemble à la source i : l'image max(0, shift) de la sortie est la max(0, -shift)
                    # de la source.
                    raise SystemExit(f"l'image {max(0, shift)} de « {nm} » est l'image {max(0, -shift)} de la source : "
                                     "même extrait, même début (--source-debut), sans interpolation")
            first = False
        rows += measure_pairs(buf[0][:n], {nm: buf[i + 1][:n] for i, nm in enumerate(names)}, flux, seuil, fondu)
        if n < chunk:
            break
        buf = [b[n - 1:] for b in buf]
    count = {kind: sum(1 for r in rows if r[0] == kind) for kind in ("ok", "coupe", "fondu")}
    # Un flux plus long qu'un autre : il reste des images dans son lot ou derrière.
    count["inegales"] = any(len(b) > n for b in buf) or any(next(it, None) is not None for it in iters)
    return shots(rows, names, min_pairs), count


def verdict(plans, first, last):
    """Part des plans où last scintille moins que first."""
    if not plans:
        return 0.0
    return sum(1 for p in plans if p["scores"][last] < p["scores"][first]) / len(plans)


def markdown(plans, names, count, part):
    lines = [f"Paires : {count['ok']} mesurées, {count['coupe']} coupes, {count['fondu']} fondus ou éclairs. "
             f"Plans retenus : {len(plans)}." + (" Les vidéos n'ont pas la même longueur : seul le début commun est "
                                                  "mesuré." if count.get("inegales") else ""), "",
             f"« {names[-1]} » devant « {names[0]} » sur {part:.0%} des plans.", "",
             "| Plan | Images | Paires | " + " | ".join(names) + " | Meilleure |",
             "|---|---|---|" + "---|" * len(names) + "---|"]
    for p in plans:
        cells = " | ".join(f"{p['scores'][n]:.3f}" for n in names)
        lines.append(f"| {p['plan']} | {p['debut']}–{p['fin']} | {p['paires']} | {cells} | {p['meilleure']} |")
    return "\n".join(lines) + "\n"


def decode(video, size, rgb, start=None, limit=None):
    """Les images de la vidéo, une à une, à la taille donnée (moyenne de surface) : RGB uint8, ou luminance lue en
    16 bits (un export 10 bits garde ses quarts de niveau)."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-i", video]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-vf", f"scale={w}:{h}:flags=area"] + (["-frames:v", str(limit)] if limit else [])
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24" if rgb else "rgb48le", "-"]
    dtype = np.dtype(np.uint8) if rgb else np.dtype("<u2")
    size = w * h * 3 * dtype.itemsize
    # Les erreurs de ffmpeg à part : lues quand il finit seul, tues quand on l'arrête avant la fin (« Broken pipe »).
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                raw = proc.stdout.read(size)
                if len(raw) < size:
                    break
                frame = np.frombuffer(raw, dtype).reshape(h, w, 3)
                yield frame if rgb else luma(frame)
            if proc.wait() != 0:
                err.seek(0)
                raise SystemExit(f"ffmpeg, {os.path.basename(video)} : {err.read().decode(errors='replace').strip()}")
        finally:
            proc.kill()
            proc.wait()


def prefetch(frames, depth):
    """Les images d'un itérable, décodées d'avance dans un fil à part (au plus depth en attente) : ffmpeg et la
    conversion en luminance travaillent pendant que les paires se mesurent."""
    q, end = queue.Queue(depth), object()

    def work():
        try:
            for frame in frames:
                q.put(frame)
            q.put(end)
        except BaseException as e:  # noqa: B036 — l'erreur (ffmpeg, SystemExit) repart dans le fil principal
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not end:
        if isinstance(item, BaseException):
            raise item
        yield item


def rate(video):
    """La cadence de la vidéo en images par seconde (ffprobe : r_frame_rate, sinon avg_frame_rate), ou une sortie en
    erreur si elle est illisible. La valeur seule, sans clé : selon la version, la sortie csv de ffprobe finit par une
    virgule (« 24000/1001, »)."""
    err = ""
    for key in ("r_frame_rate", "avg_frame_rate"):
        done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={key}",
                               "-of", "default=noprint_wrappers=1:nokey=1", video],
                              capture_output=True, text=True, check=False)
        err = done.stderr.strip() or err
        num, _, den = (done.stdout.strip().splitlines() or [""])[0].strip().rstrip(",").partition("/")
        try:
            value = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue  # « 0/0 » : ffprobe ne sait pas
        if value > 0:
            return value
    raise SystemExit(f"ffprobe, {os.path.basename(video)} : {err or 'pas de cadence lisible'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sorties", nargs="+", help="les exports du lecteur, deux ou plus, sur le même passage")
    p.add_argument("--source", help="le passage source (le flux en vient ; sinon de la première sortie)")
    p.add_argument("--source-debut", type=float, help="secondes à sauter dans la source pour s'aligner")
    p.add_argument("--flux", choices=("nvof", "rife"), required=True)
    p.add_argument("--nvof-dump", help="l'exécutable nvof_dump (tools/nvof_interp/nvof_dump)")
    p.add_argument("--rife", help="un RIFE livré (11 plans), pour --flux rife")
    p.add_argument("--hauteur-rife", type=int, default=720, help="hauteur où RIFE estime le flux (720)")
    p.add_argument("--taille", default="1920x1080", help="taille de la mesure (1920x1080)")
    p.add_argument("--images", type=int, help="au plus ce nombre d'images")
    p.add_argument("--seuil", type=float, default=10.0, help="écart de la source recalée au-delà duquel le flux "
                   "n'est pas sûr, en niveaux (10)")
    p.add_argument("--fondu", type=float, default=0.75, help="variation de luminance moyenne d'un fondu (0,75)")
    p.add_argument("--plan-min", type=int, default=8, help="paires mesurées minimales d'un plan retenu (8)")
    p.add_argument("--porte", type=float, help="part des plans où la dernière sortie doit être devant la première")
    p.add_argument("--md")
    p.add_argument("--json")
    a = p.parse_args()
    if len(a.sorties) < 2:
        p.error("deux sorties au moins")
    if a.flux == "nvof" and not a.nvof_dump:
        p.error("--flux nvof demande --nvof-dump")
    if a.flux == "rife" and not a.rife:
        p.error("--flux rife demande --rife")
    try:
        size = tuple(int(v) for v in a.taille.split("x"))
    except ValueError:
        size = ()
    if len(size) != 2 or min(size) < 64 or size[0] % 2 or size[1] % 2:
        p.error("--taille : LARGEURxHAUTEUR, paires, 64 au moins (NVOF ne prend pas de taille impaire)")
    if a.source:
        source_rate = rate(a.source)
        for s in a.sorties:
            if abs(rate(s) - source_rate) > 0.01:
                p.error(f"{s} : {rate(s):.3f} i/s contre {source_rate:.3f} pour la source (interpolation ?)")
    names = []
    for s in a.sorties:
        base = os.path.splitext(os.path.basename(s))[0]
        names.append(base if base not in names else f"{base}-{len(names) + 1}")
    if a.source is None:
        print("sans --source : le flux vient de la première sortie, qu'il favorise", file=sys.stderr)
    flux = flux_nvof(a.nvof_dump) if a.flux == "nvof" else flux_rife(a.rife, a.hauteur_rife)
    depth = 40  # un lot et un peu plus
    refs = prefetch(decode(a.source or a.sorties[0], size, True, a.source_debut if a.source else None, a.images),
                    depth)
    outs = {n: prefetch(decode(s, size, False, None, a.images), depth) for n, s in zip(names, a.sorties)}
    plans, count = run(refs, outs, flux, a.seuil, a.fondu, min_pairs=a.plan_min)
    if count["ok"] == 0:
        raise SystemExit("aucune paire mesurée : vidéos illisibles, trop courtes, ou que des coupes")
    part = verdict(plans, names[0], names[-1])
    text = markdown(plans, names, count, part)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("# Scintillement\n\n" + text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"sorties": names, "flux": a.flux, "taille": a.taille, "paires": count, "plans": plans,
                       "devant": part, "porte": a.porte}, f, indent=1, ensure_ascii=False)
    if a.porte is not None and (not plans or part < a.porte):
        print(f"porte non passée : {part:.0%} des plans, {a.porte:.0%} demandés", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
