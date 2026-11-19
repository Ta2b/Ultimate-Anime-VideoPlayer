"""Les mesures du banc NVOF (nvof_scene.py) : lecture des extraits, appel de nvof_dump, classes de texture, recalage,
caméra affine, écart du décor mesuré sur l'image, tuiles et indices de la pyramide."""
import os
import re
import subprocess
import tempfile

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# Images et outil

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,pix_fmt", "-of", "csv=p=0", path], capture_output=True, text=True,
                         check=True)
    w, h, fmt = out.stdout.strip().split(",")[:3]
    return int(w), int(h), fmt


def decode(path, start, frames, height):
    """Lumas 8 bits (n, h, w) comme le lecteur les donnera à NVOF : plan Y réduit à sa profondeur, puis ses 8 bits
    hauts (porte 2.2). Pas de yuv420p : swscale tramerait une source 10 bits et fausserait aplats et coût."""
    sw, sh, fmt = probe(path)
    w = int(round(sw * height / sh / 2)) * 2
    depth = re.search(r"p(\d+)(le|be)$", fmt)
    bits = int(depth.group(1)) if depth else 8
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", start] if start else []) + [
        "-i", path, "-frames:v", str(frames), "-an", "-sn",
        "-vf", f"scale={w}:{height}:flags=area,format={fmt},extractplanes=y", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if bits == 8:
        return np.frombuffer(raw, np.uint8).reshape(-1, height, w)
    wide = np.frombuffer(raw, "<u2" if depth.group(2) == "le" else ">u2").reshape(-1, height, w)
    return (wide >> (bits - 8)).astype(np.uint8)


def moving_pairs(frames, limit):
    """Paires voisines (k, k + 1), sauf les doublons exacts. Le mouvement ne se juge pas à la luminance (un panoramique
    lent sur un aplat la change à peine) : une image tenue reste, la caméra mesurée l'écarte de la porte."""
    keep = [k for k in range(len(frames) - 1) if not np.array_equal(frames[k], frames[k + 1])]
    return keep[:limit], len(frames) - 1 - len(keep)


def run_tool(tool, grid, hints, frames, pairs, external=None):
    """{k: (flux 2 × ch × cw en px, coût ch × cw)}, [ms], {gpu, cuda, grilles}. Lève RuntimeError si refus.
    external : (grille des indices, {k: int16 hh × hw × 2 en S10.5}), les indices externes de chaque paire."""
    _, h, w = frames.shape
    cw, ch = (w + grid - 1) // grid, (h + grid - 1) // grid
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for k in sorted({k for p in pairs for k in (p, p + 1)}):
            names[k] = os.path.join(tmp, f"{k}.y")
            frames[k].tofile(names[k])
        tasks = ""
        for k in pairs:
            task = f"{w}\t{h}\t{names[k]}\t{names[k + 1]}\t{os.path.join(tmp, f'{k}.flow')}"
            if external is not None:
                external[1][k].astype(np.int16).tofile(os.path.join(tmp, f"{k}.hint"))
                task += "\t" + os.path.join(tmp, f"{k}.hint")
            tasks += task + "\n"
        args = [tool, "--grid", str(grid), "--one-way", "--cost"] + (["--hints"] if hints else []) \
            + (["--hint-grid", str(external[0])] if external is not None else [])
        done = subprocess.run(args, input=tasks, text=True, capture_output=True, check=False)
        times, info = [], {}
        for line in done.stderr.splitlines():
            head, _, rest = line.partition(" ")
            if head == "temps_ms":
                times.append(float(rest.split()[1]))
            elif head in ("gpu", "cuda", "grilles"):
                info[head] = rest
        times = times[2:] if len(times) > 4 else times  # chaque lancement ouvre sa session : ses premiers appels chauffent
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


def sample(img, sx, sy):
    """Lecture bilinéaire de img (float32) aux positions sx, sy, en FP32, bornées à l'image."""
    h, w = img.shape
    sx, sy = np.clip(sx, 0, w - 1.001), np.clip(sy, 0, h - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = sx - x0, sy - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def warp_gain(a, b, flow, grid):
    """PSNR de b recalé sur a par le flux a → b (a(x) ≈ b(x + F)), moins celui de b tel quel."""
    h, w = a.shape
    f = np.repeat(np.repeat(flow, grid, 1), grid, 2)[:, :h, :w]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    bf = b.astype(np.float32)
    warped = sample(bf, x + f[0], y + f[1])
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
    # Écart des vecteurs NVOF au modèle, sur un décor large (3 px à 720p) : pour information. La porte prend l'écart
    # mesuré sur l'image (photo_residual), que ni le bruit des vecteurs ni un personnage lent ne faussent.
    wide = r < 3 * k
    return model, decor, {
        "camera_px": float(np.hypot(*model).mean() / k),
        "residu_flux_px": float(np.median(r[wide]) / k) if wide.any() else float("inf"),
        "part_decor": float(inl.mean()),
        "affine": [float(x) for x in (*pu, *pv)],
    }


def model_at_pixels(affine, grid, ch, cw, h, w):
    """Le modèle affine de camera() évalué en chaque pixel (2 × h × w, px natifs). camera() place un bloc à son coin :
    le pixel x correspond au bloc dont le centre est x."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (x - grid / 2 - 0.5 * cw * grid) / (cw * grid)
    yn = (y - grid / 2 - 0.5 * ch * grid) / (ch * grid)
    pu, pv = affine[:3], affine[3:]
    return np.stack([pu[0] + pu[1] * xn + pu[2] * yn, pv[0] + pv[1] * xn + pv[2] * yn]).astype(np.float32)


def photo_residual(a, b, affine, decor, grid, height):
    """Écart du décor au modèle de caméra, mesuré sur l'image : b recalé par le modèle est comparé à a sur chaque
    tuile de décor (32 px en 1080p, un quart au moins de ses pixels dans des blocs de décor, assez texturée dans
    les deux directions), et deux pas de Lucas-Kanade donnent le décalage qui reste. Rend (médiane en px à 720p,
    tuiles mesurées) ; (None, 0) sous 3 tuiles. Les vecteurs NVOF ne servent qu'à choisir le décor : leur bruit
    (environ 0,3 px sur un décor fixe, 4070 Ti) et un personnage lent ne comptent plus."""
    h, w = a.shape
    ch, cw = decor.shape
    t = max(8, int(round(32 * height / 1080)))
    th, tw = h // t, w // t
    fa, fb = box(a.astype(np.float32), 1).astype(np.float32), box(b.astype(np.float32), 1).astype(np.float32)
    model = model_at_pixels(affine, grid, ch, cw, h, w)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.repeat(np.repeat(decor, grid, 0), grid, 1)[:h, :w].astype(np.float32)
    per_tile = lambda m: m[: th * t, : tw * t].reshape(th, t, tw, t).sum((1, 3))  # noqa: E731
    up = lambda m: np.pad(np.repeat(np.repeat(m, t, 0), t, 1), ((0, h - th * t), (0, w - tw * t)))  # noqa: E731
    d = np.zeros((2, th, tw), np.float32)
    for _ in range(2):
        warped = sample(fb, x + model[0] + up(d[0]), y + model[1] + up(d[1]))
        gx, gy = np.zeros_like(warped), np.zeros_like(warped)
        gx[:, 1:-1] = (warped[:, 2:] - warped[:, :-2]) / 2
        gy[1:-1] = (warped[2:] - warped[:-2]) / 2
        e = (fa - warped) * mask
        sxx, sxy, syy = per_tile(gx * gx * mask), per_tile(gx * gy * mask), per_tile(gy * gy * mask)
        bx, by = per_tile(gx * e), per_tile(gy * e)
        det = sxx * syy - sxy * sxy
        safe = np.where(np.abs(det) > 1e-6, det, 1.0)
        d[0] += np.clip(np.where(np.abs(det) > 1e-6, (syy * bx - sxy * by) / safe, 0), -2, 2)
        d[1] += np.clip(np.where(np.abs(det) > 1e-6, (sxx * by - sxy * bx) / safe, 0), -2, 2)
    count = per_tile(mask)
    tr = sxx + syy
    lam_min = (tr - np.sqrt(np.maximum(tr * tr - 4 * det, 0))) / 2  # petite valeur propre du tenseur de structure
    # Un quart de la tuile au moins en blocs de décor ; la structure doit dépasser le grain (environ 2 niveaux² par
    # pixel après le flou 3 × 3) dans les deux sens, sinon Lucas-Kanade mesure du bruit ou un trait seul.
    valid = (count >= 0.25 * t * t) & (lam_min >= 8.0 * np.maximum(count, 1))
    if valid.sum() < 3:
        return None, 0
    return float(np.median(np.hypot(d[0], d[1])[valid]) / (height / 720.0)), int(valid.sum())


def fit_pair(a, b, flow, cost, textured, grid, height):
    """camera() puis l'écart mesuré sur l'image (residu_px, celui de la porte). Sans tuile mesurable, residu_px
    reprend l'écart des vecteurs (residu_flux_px) et tuiles_mesurees vaut 0."""
    fit = camera(flow, cost, textured, grid, height)
    if fit is None:
        return None
    model, decor, stats = fit
    photo, n = photo_residual(a, b, stats["affine"], decor, grid, height)
    stats["residu_px"] = photo if photo is not None else stats["residu_flux_px"]
    stats["tuiles_mesurees"] = n
    return model, decor, stats


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


def hint_field(affine, scale, w, h, hint_grid):
    """Le mouvement de caméra d'une passe basse, porté à la taille w × h : indices hh × hw × 2 en S10.5."""
    hw, hh = (w + hint_grid - 1) // hint_grid, (h + hint_grid - 1) // hint_grid
    yy, xx = np.mgrid[0:hh, 0:hw].astype(np.float64)
    xn, yn = (xx * hint_grid - 0.5 * hw * hint_grid) / (hw * hint_grid), (yy * hint_grid - 0.5 * hh * hint_grid) / (hh * hint_grid)
    pu, pv = np.array(affine[:3]), np.array(affine[3:])
    u = (pu[0] + pu[1] * xn + pu[2] * yn) * scale
    v = (pv[0] + pv[1] * xn + pv[2] * yn) * scale
    return np.clip(np.rint(np.stack([u, v], -1) * 32), -32768, 32767).astype(np.int16)
