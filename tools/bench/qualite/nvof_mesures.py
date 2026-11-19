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
    tuile de décor (32 px en 1080p, un quart au moins de ses pixels dans des blocs de décor), et deux pas de
    Lucas-Kanade donnent le décalage qui reste. Rend (médiane en px à 720p, tuiles mesurées, sorte) :
    - « tuiles » : 3 tuiles au moins texturées dans les deux sens ; le décalage entier se mesure ;
    - « bords » : sinon, 3 tuiles au moins texturées dans un sens (un bord droit, un décor rayé que la caméra longe) ;
      seul l'écart en travers du bord se mesure, et c'est le seul qui se voit ;
    - (None, 0, None) sinon.
    Les vecteurs NVOF ne servent qu'à choisir le décor : leur bruit (environ 0,3 px sur un décor fixe, 4070 Ti) et un
    personnage lent ne comptent plus."""
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
    count = per_tile(mask)
    # La structure doit dépasser le grain (environ 2 niveaux² par pixel après le flou 3 × 3), sinon Lucas-Kanade
    # mesure du bruit.
    floor = 8.0 * np.maximum(count, 1)
    d = np.zeros((2, th, tw), np.float32)
    for _ in range(2):
        warped = sample(fb, x + model[0] + up(d[0]), y + model[1] + up(d[1]))
        gx, gy = np.zeros_like(warped), np.zeros_like(warped)
        gx[:, 1:-1] = (warped[:, 2:] - warped[:, :-2]) / 2
        gy[1:-1] = (warped[2:] - warped[:-2]) / 2
        e = (fa - warped) * mask
        sxx, sxy, syy = per_tile(gx * gx * mask), per_tile(gx * gy * mask), per_tile(gy * gy * mask)
        bx, by = per_tile(gx * e), per_tile(gy * e)
        # Tenseur de structure : valeurs propres l1 ≥ l2, direction propre (c, s) de l1. Pseudo-inverse tronquée :
        # chaque direction assez texturée est résolue, l'autre laissée (sur un bord, l'écart le long du bord est
        # indéterminé). Sur une tuile texturée dans les deux sens, c'est la solution 2 × 2 entière.
        half, disc = (sxx + syy) / 2, np.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy)
        l1, l2 = half + disc, half - disc
        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        c, s = np.cos(theta), np.sin(theta)
        p1 = np.where(l1 >= floor, (c * bx + s * by) / np.maximum(l1, 1e-6), 0)
        p2 = np.where(l2 >= floor, (c * by - s * bx) / np.maximum(l2, 1e-6), 0)
        d[0] += np.clip(p1 * c - p2 * s, -2, 2)
        d[1] += np.clip(p1 * s + p2 * c, -2, 2)
    cover = count >= 0.25 * t * t
    full, edge = cover & (l2 >= floor), cover & (l1 >= floor)
    for valid, kind in ((full, "tuiles"), (edge, "bords")):
        if valid.sum() >= 3:
            return float(np.median(np.hypot(d[0], d[1])[valid]) / (height / 720.0)), int(valid.sum()), kind
    return None, 0, None


def flux_residual(flow, textured, affine, grid, height):
    """Écart médian des vecteurs NVOF des blocs texturés au modèle, sur ceux à moins de 3 px (px à 720p) : la mesure
    de secours quand l'image n'a pas 3 tuiles mesurables."""
    _, ch, cw = flow.shape
    k = height / 720.0
    m = model_at_pixels(affine, grid, ch, cw, ch * grid, cw * grid)[:, grid // 2::grid, grid // 2::grid][:, :ch, :cw]
    r = np.hypot(flow[0] - m[0], flow[1] - m[1])[textured]
    wide = r < 3 * k
    return float(np.median(r[wide]) / k) if wide.any() else float("inf")


def refine_affine(a, b, affine, decor, grid, height):
    """Affinage sous-pixel de l'affine NVOF par Lucas-Kanade global (Gauss-Newton sur les 6 paramètres) sur les pixels
    des blocs de décor : 3 pas sur les images floutées en 7 × 7 (portée), puis 3 en 3 × 3 (précision). Les pixels qui
    s'écartent (personnage dans un bloc de décor, occlusion) sont amortis à la Huber. Rend la nouvelle affine."""
    h, w = a.shape
    ch, cw = decor.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (x - grid / 2 - 0.5 * cw * grid) / (cw * grid)
    yn = (y - grid / 2 - 0.5 * ch * grid) / (ch * grid)
    mask = np.repeat(np.repeat(decor, grid, 0), grid, 1)[:h, :w]
    if mask.sum() < 1000:
        return list(affine)
    phi = [np.ones(int(mask.sum()), np.float32), xn[mask], yn[mask]]
    p = np.array(affine, np.float64)
    for radius, steps in ((3, 3), (1, 3)):
        fa, fb = box(a.astype(np.float32), radius).astype(np.float32), box(b.astype(np.float32), radius).astype(np.float32)
        for _ in range(steps):
            m = model_at_pixels(p, grid, ch, cw, h, w)
            warped = sample(fb, x + m[0], y + m[1])
            gx, gy = np.zeros_like(warped), np.zeros_like(warped)
            gx[:, 1:-1] = (warped[:, 2:] - warped[:, :-2]) / 2
            gy[1:-1] = (warped[2:] - warped[:-2]) / 2
            e = (fa - warped)[mask]
            J = np.stack([gx[mask] * f for f in phi] + [gy[mask] * f for f in phi], 1).astype(np.float64)
            c = 2.5 * 1.4826 * max(float(np.median(np.abs(e))), 0.25)
            wt = np.minimum(1.0, c / np.maximum(np.abs(e), 1e-6))
            H = (J * wt[:, None]).T @ J
            if np.linalg.cond(H) > 1e8:
                return list(p)
            p += np.linalg.solve(H, (J * wt[:, None]).T @ e)
    return list(p)


def block_edges(a, grid, ch, cw):
    """Blocs texturés dans un seul sens (un bord, des rayures) : le vecteur NVOF n'y vaut qu'en travers du bord, le
    long du bord il est indéterminé. Tenseur de structure de la luma de a (floutée en 3 × 3, comme photo_residual)
    sur une fenêtre de 3 × 3 blocs ; un sens se mesure au-delà de 8 niveaux² par pixel. Rend (masque, c, s), (c, s)
    la normale au bord."""
    h, w = a.shape
    f = box(a.astype(np.float32), 1).astype(np.float32)
    gx, gy = np.zeros_like(f), np.zeros_like(f)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2
    gy[1:-1] = (f[2:] - f[:-2]) / 2

    def per_block(m):
        p = np.zeros((ch * grid, cw * grid), np.float32)
        p[:h, :w] = m[: ch * grid, : cw * grid]
        return box(p.reshape(ch, grid, cw, grid).mean((1, 3)), 1)  # moyenne par pixel sur 3 × 3 blocs

    sxx, sxy, syy = per_block(gx * gx), per_block(gx * gy), per_block(gy * gy)
    half, disc = (sxx + syy) / 2, np.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy)
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    return (half + disc >= 8.0) & (half - disc < 8.0), np.cos(theta), np.sin(theta)


def decor_of(flow, textured, affine, grid, height, edges=None, frame=None):
    """Blocs de décor pour une affine donnée, au seuil de camera() (3 MAD, entre 0,75 et 2 px à 720p) : (masque, part).
    edges (block_edges) : un bloc texturé dans un seul sens est aussi gardé si son vecteur suit le modèle en travers
    de son bord (le seuil reste celui des vecteurs entiers : rien de ce qui était gardé ne sort). frame (h, w) : les
    blocs que le modèle envoie hors de l'image b ne sont pas jugés (ni décor ni dessin : ils n'ont rien en face)."""
    _, ch, cw = flow.shape
    k = height / 720.0
    m = model_at_pixels(affine, grid, ch, cw, ch * grid, cw * grid)[:, grid // 2::grid, grid // 2::grid][:, :ch, :cw]
    du, dv = flow[0] - m[0], flow[1] - m[1]
    r = np.hypot(du, dv)
    near = r[textured][r[textured] < 2 * k]
    thr = np.clip(3 * 1.4826 * np.median(near) if near.size else 2 * k, 0.75 * k, 2 * k)
    keep = r < thr
    if edges is not None:
        one, c, s = edges
        keep |= one & (np.abs(du * c + dv * s) < thr)
    judged_blocks = textured.copy()
    if frame is not None:
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32) * grid + grid // 2
        judged_blocks &= (xx + m[0] >= 0) & (xx + m[0] < frame[1]) & (yy + m[1] >= 0) & (yy + m[1] < frame[0])
    decor = judged_blocks & keep
    return decor, float(decor.sum() / judged_blocks.sum()) if judged_blocks.any() else 0.0


def judged(a, b, flow, textured, affine, decor, grid, height):
    """L'écart d'une affine comme la porte le juge : sur l'image (photo_residual), sinon sur les vecteurs NVOF des
    blocs texturés (flux_residual). Rend {residu_px, tuiles_mesurees, mesure}."""
    photo, n, kind = photo_residual(a, b, affine, decor, grid, height)
    if photo is not None:
        return {"residu_px": photo, "tuiles_mesurees": n, "mesure": kind}
    return {"residu_px": flux_residual(flow, textured, affine, grid, height), "tuiles_mesurees": 0,
            "mesure": "vecteurs"}


def variants(a, b, flow, cost, cls, grid, height, fit):
    """Les pistes de la porte 2.1 sur une paire, chacune jugée comme la porte (judged : résidu sur l'image, sinon
    sur les vecteurs ; part du décor) :
    - nvof : l'affine NVOF, ajustée sur les blocs texturés ;
    - traits : ajustée sur les blocs de traits seuls (sans le grain) ;
    - lk : l'affine NVOF affinée par Lucas-Kanade global sur son décor, décor repris avec elle sur les blocs
      texturés ;
    - traits_lk : l'affine des traits affinée de même, décor repris sur les blocs de traits ;
    - lk_bords : l'affine de lk ; décor repris en jugeant un bloc texturé dans un seul sens en travers de son bord
      (block_edges) et sans juger les blocs que la caméra sort de l'image.
    Rend {nom: {residu_px, part_decor, affine, tuiles_mesurees, mesure}}."""
    out = {}
    keep = ("residu_px", "part_decor", "affine", "tuiles_mesurees", "mesure")
    if fit is not None:
        out["nvof"] = {k: fit[2][k] for k in keep}
    lines = fit_pair(a, b, flow, cost, cls == 1, grid, height)
    if lines is not None:
        out["traits"] = {k: lines[2][k] for k in keep}
    for name, base, textured in (("lk", fit, cls != 0), ("traits_lk", lines, cls == 1)):
        if base is None:
            continue
        refined = [float(v) for v in refine_affine(a, b, base[2]["affine"], base[1], grid, height)]
        decor, share = decor_of(flow, textured, refined, grid, height)
        out[name] = {**judged(a, b, flow, textured, refined, decor, grid, height), "part_decor": share,
                     "affine": refined}
    if "lk" in out:
        refined = out["lk"]["affine"]
        decor, share = decor_of(flow, cls != 0, refined, grid, height, block_edges(a, grid, *cls.shape), a.shape)
        out["lk_bords"] = {**judged(a, b, flow, cls != 0, refined, decor, grid, height), "part_decor": share,
                           "affine": refined}
    return out


def fit_pair(a, b, flow, cost, textured, grid, height):
    """camera() puis l'écart comme la porte le juge (judged : residu_px, tuiles_mesurees, mesure)."""
    fit = camera(flow, cost, textured, grid, height)
    if fit is None:
        return None
    model, decor, stats = fit
    stats.update(judged(a, b, flow, textured, stats["affine"], decor, grid, height))
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
