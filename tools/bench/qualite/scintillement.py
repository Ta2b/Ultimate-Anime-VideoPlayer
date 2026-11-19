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
Il faut python3 avec numpy, ffmpeg et ffprobe ; onnx et onnxruntime pour --flux rife.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CUT_SAFE = 0.30  # part sûre minimale d'une paire du même plan
GRID = 4  # NVOF : un vecteur par bloc de 4 × 4 (nvof_dump)


def luma(rgb):
    """Luminance BT.709 en niveaux 8 bits, d'une image H × W × 3 en uint8 ou uint16 ou 3 × H × W dans [0, 1]."""
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        y = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return y / 257.0 if rgb.dtype == np.uint16 else y
    return 255.0 * (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def box(img, r=1):
    """Moyenne (2r + 1) × (2r + 1), bords répétés."""
    p = np.pad(img, r, mode="edge").astype(np.float64)
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * r + 1
    h, w = img.shape
    return ((c[n:n + h, n:n + w] - c[:h, n:n + w] - c[n:n + h, :w] + c[:h, :w]) / (n * n)).astype(np.float32)


def sample(img, x, y):
    """Lecture bilinéaire de img (H × W) aux points (x, y), bords répétés."""
    h, w = img.shape
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.minimum(np.floor(x).astype(np.int32), w - 2), np.minimum(np.floor(y).astype(np.int32), h - 2)
    ax, ay = x - x0, y - y0
    top = img[y0, x0] * (1 - ax) + img[y0, x0 + 1] * ax
    return top * (1 - ay) + (img[y0 + 1, x0] * (1 - ax) + img[y0 + 1, x0 + 1] * ax) * ay


def grid(h, w):
    return np.mgrid[0:h, 0:w].astype(np.float32)[::-1]  # (x, y)


def scale(field, w, h, step=None):
    """Un champ C × h0 × w0 au bilinéaire en C × h × w (centres demi-pixel). Un flux garde ses unités : à multiplier.
    step : un point du champ pour step × step pixels (la grille NVOF, dont le dernier bloc d'une taille qui n'en est
    pas un multiple déborde de l'image) ; sinon le champ couvre l'image."""
    _, h0, w0 = field.shape
    x = (np.arange(w, dtype=np.float32) + 0.5) * (1.0 / step if step else w0 / w) - 0.5
    y = (np.arange(h, dtype=np.float32) + 0.5) * (1.0 / step if step else h0 / h) - 0.5
    xx, yy = np.broadcast_to(x[None, :], (h, w)), np.broadcast_to(y[:, None], (h, w))
    return np.stack([sample(c, xx, yy) for c in field])


class Aligned:
    """Une paire recalée : warp(N, fa) face à warp(N + 1, fb) ; fb absent vaut l'identité."""

    def __init__(self, fa, fb, coherent=None):
        _, h, w = fa.shape
        gx, gy = grid(h, w)
        self.a = (gx + fa[0], gy + fa[1])
        self.b = (gx + fb[0], gy + fb[1]) if fb is not None else None
        self.coherent = coherent
        self._bounds()

    def _bounds(self):
        h, w = self.a[0].shape
        inside = np.ones((h, w), bool)
        for x, y in (self.a, self.b) if self.b else (self.a,):
            inside &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        self.inside = inside if self.coherent is None else inside & self.coherent

    def pair(self, ya, yb):
        return sample(ya, *self.a), (sample(yb, *self.b) if self.b else yb)

    def refine(self, ya, yb, steps=4, step=2):
        """Corrige le flux d'une translation d'ensemble sous-pixel, par Lucas-Kanade global sur la source (un point sur
        step × step), chaque image prenant la moitié de la correction. RIFE ignore les mouvements de moins d'un pixel
        par image et NVOF est au quart de pixel : sur un panoramique lent, le reste du décalage compterait sinon comme
        du scintillement. Rend la translation ajoutée (x, y)."""
        total = np.zeros(2)
        for _ in range(steps):
            sub = (slice(None, None, step), slice(None, None, step))
            ra = sample(ya, self.a[0][sub], self.a[1][sub])
            rb = sample(yb, self.b[0][sub], self.b[1][sub]) if self.b else yb[sub]
            r = ra - rb
            gy, gx = np.gradient(0.5 * (ra + rb), step)
            m = self.inside[sub] & (np.abs(r) < 30.0)  # les occlusions et les coupes ne tirent pas la solution
            gx, gy, r = gx[m], gy[m], r[m]
            mat = np.array([[gx @ gx, gx @ gy], [gx @ gy, gy @ gy]], np.float64)
            if r.size < 64 or np.linalg.det(mat) < 1e-6 * max(1.0, np.trace(mat)) ** 2:
                break  # image unie ou trait dans un seul sens : la translation n'est pas déterminée
            d = np.clip(-np.linalg.solve(mat, np.array([gx @ r, gy @ r], np.float64)), -2.0, 2.0)
            self.a = (self.a[0] + 0.5 * d[0], self.a[1] + 0.5 * d[1])
            gxf, gyf = (self.b if self.b else grid(*ya.shape))
            self.b = (gxf - 0.5 * d[0], gyf - 0.5 * d[1])
            total += d
            if np.hypot(*d) < 0.01:
                break
        self._bounds()
        return tuple(total)


# Les fournisseurs de flux : refs (liste d'images H × W × 3 uint8 de la source) → un Aligned par paire.

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
                v = np.fromfile(o, np.int16).astype(np.float32) / 32.0  # S10.5
                # nvof_dump : l'aller sur la grille de N, pointant dans N+1, puis le retour sur celle de N+1.
                fwd = scale(v[:ch * cw * 2].reshape(ch, cw, 2).transpose(2, 0, 1), w, h, GRID)
                bwd = scale(v[ch * cw * 2:].reshape(ch, cw, 2).transpose(2, 0, 1), w, h, GRID)
                gx, gy = grid(h, w)
                # Aller et retour : le vecteur arrière, suivi du vecteur aller là où il mène, revient au point.
                back = np.stack([sample(c, gx + bwd[0], gy + bwd[1]) for c in fwd])
                err = np.hypot(bwd[0] + back[0], bwd[1] + back[1])
                # Chaque image fait la moitié du chemin vers le milieu de la paire (même flou des deux côtés).
                result.append(Aligned(0.5 * bwd, -0.5 * bwd, err <= 1.0 + 0.05 * np.hypot(bwd[0], bwd[1])))
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
        result = []
        for a, b in zip(imgs, imgs[1:]):
            _, flow, _ = rife_taille.estimate(sess, names, a, b, 2 * modeles.ALIGN)
            flow = scale(flow, w, h)
            flow[0::2] *= w / fw
            flow[1::2] *= h / fh
            result.append(Aligned(flow[0:2], flow[2:4]))
        return result

    return run


def measure_pairs(refs, outs, flux, seuil, fondu):
    """Les paires d'un lot : (« coupe » | « fondu » | « ok », {sortie : (moyen, 95e centile)})."""
    rows = []
    for k, al in enumerate(flux(refs)):
        ya, yb = luma(refs[k]), luma(refs[k + 1])
        al.refine(ya, yb)
        ra, rb = al.pair(ya, yb)
        safe = al.inside & (box(np.abs(ra - rb)) < seuil)
        if safe.mean() < CUT_SAFE:
            rows.append(("coupe", {}))
            continue
        if abs(float(ra[safe].mean()) - float(rb[safe].mean())) > fondu:
            rows.append(("fondu", {}))
            continue
        scores = {}
        for name, frames in outs.items():
            oa, ob = al.pair(frames[k], frames[k + 1])
            d = np.abs(oa - ob)[safe]
            scores[name] = (float(d.mean()), float(np.percentile(d, 95)))
        rows.append(("ok", scores))
    return rows


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
    ys = [luma(r) for r in refs[:n]]
    score = {o: np.mean([corr(ys[i], frames[i + o]) for i in range(max(0, -o), min(n, n - o))])
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


def rate(video):
    """La cadence de la vidéo en images par seconde (ffprobe), ou une sortie en erreur si elle est illisible."""
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", video], capture_output=True, text=True, check=False)
    num, _, den = done.stdout.strip().partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        raise SystemExit(f"ffprobe, {os.path.basename(video)} : {done.stderr.strip() or 'pas de piste vidéo'}") from None


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
    refs = decode(a.source or a.sorties[0], size, True, a.source_debut if a.source else None, a.images)
    outs = {n: decode(s, size, False, None, a.images) for n, s in zip(names, a.sorties)}
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
