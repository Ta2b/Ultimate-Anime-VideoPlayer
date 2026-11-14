#!/usr/bin/env python3
"""GMFSS Fortuna Union face à RIFE 4.26 sur de vrais instants, et ce que coûte à l'image chaque raccourci qui
ferait tenir GMFSS dans un niveau « High » (PyTorch pour GMFSS, ONNX Runtime pour RIFE).

  python3 -I tools/bench/qualite/gmfss_instants.py --video EXTRAIT_1080P.mkv \\
      --gmfss roue/vsgmfss_fortuna --reseau rife_v4.26.onnx --json r.json --md r.md

La roue vsgmfss_fortuna 1.0.0 (HolyWu, MIT ; réseau et poids de 98mxr/GMFSS_Fortuna, MIT) s'extrait sans
l'installer, comme pour tools/gmfss/export_gmfss.py :

  pip download --no-deps vsgmfss_fortuna==1.0.0 -d gmfss
  python -m zipfile -e gmfss/vsgmfss_fortuna-1.0.0-py3-none-any.whl gmfss/roue

Suites de quatre images 1080p toutes différentes d'un même plan (k à k+3, rife_instants.quadruples) : k et k+3
forment la paire, k+1 et k+2 sont les vérités à t = ⅓ et ⅔ (--ecart N : k et k+N, vérités à i/N, un mouvement de
source à faible cadence). Tout se fait à la taille de travail (720p par défaut,
réduction de surface ; remplie au multiple de 128 comme le faisait le lecteur). Pour chaque instant, le PSNR contre
la vraie image et le pire bloc 64 × 64 de :

  rife       RIFE 4.26 à l'instant (flot et masque du graphe livré, warp et mélange de l'étage du lecteur) ;
  gmfss      GMFSS Union (poids de la roue : l'ancien union) à l'instant : paire (une fois), six splats, IFNet et GridNet (une fusion par instant) ;
  splats     GMFSS sans fusion : les deux images splattées à l'instant, mélangées (ce qu'apporte GridNet) ;
  milieu     une seule fusion par paire, au milieu M ; l'instant t s'en déduit par le flot de GMFSS : M est splatté
             de (t − ½) fois le flot porté au milieu (le flot de chaque côté splatté à ½), mêlé à l'image de la
             paire la plus proche splattée à t, chacune pesant l'inverse de sa distance en temps ;
  milieu_rife  une seule fusion au milieu M, puis RIFE 4.26 sur la demi-paire (a, M) ou (M, b) : la moitié du
             mouvement, une passe de RIFE par instant.

Coût par paire à ×N (N − 1 instants), en fusions F (≈ 10,5 ms sur une 4070 Ti à 720p) et passes RIFE R (≈ 2,2 ms) :
gmfss (N − 1)·F ; milieu F (instants pairs : la fusion du milieu sert) ; milieu_rife F + (N − 2)·R.

Il faut python3 avec numpy, torch, onnx, onnxruntime, et ffmpeg ; une carte CUDA accélère GMFSS (--device cuda).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "gmfss"))
import mesures  # noqa: E402
import modeles  # noqa: E402
import rife_taille as rt  # noqa: E402
CUT = 0.12  # rife_instants.CUT : écart moyen au-delà duquel deux images sont de plans différents

GMFSS_ALIGN = 128  # plans de la paire à 128, comme le lecteur d'avant 0.10.0
METHODS = ("rife", "gmfss", "splats", "milieu", "milieu_rife")


# ---------------------------------------------------------------------------------------------------------------
# GMFSS (PyTorch)

# Les fichiers d'une archive de 98mxr (train_log/) sous les noms de la roue.
TRAIN_LOG_NAMES = {"flownet.pkl": "flownet.pkl", "metric.pkl": "metric_union.pkl", "feat.pkl": "feat_union.pkl",
                   "fusionnet.pkl": "fusionnet_union.pkl", "rife.pkl": "rife.pkl"}


def weights_dir(path):
    """Le dossier des poids sous les noms de la roue : models/ de la roue tel quel, ou un train_log/ de 98mxr
    (le « new union model ») relié sous ces noms dans un dossier temporaire."""
    import tempfile

    if os.path.exists(os.path.join(path, "metric_union.pkl")):
        return path
    linked = tempfile.mkdtemp(prefix="gmfss_poids_")
    for theirs, ours in TRAIN_LOG_NAMES.items():
        os.symlink(os.path.abspath(os.path.join(path, theirs)), os.path.join(linked, ours))
    return linked


def load(package, device, scale, weights=None):
    import torch
    from gmfss_graph import Fusion, Pair, load_gmfss

    gmfss_class = load_gmfss(os.path.abspath(package))
    models = weights_dir(weights or os.path.join(os.path.abspath(package), "models"))
    net = gmfss_class(models, "union", scale, False).eval().to(device)
    return Pair(net).eval(), Fusion(net).eval(), torch.device(device)


def splat_sums(source, flow, metric):
    """Le softmax splatting de soft_splat, sans la normalisation : (Σ source·e^m·w, Σ e^m·w), pour mêler
    plusieurs sources avant de diviser."""
    import torch

    scale = metric.float().exp()
    values = torch.cat([source.float() * scale, scale], 1)
    b, c, h, w = values.shape
    tx = torch.arange(w, dtype=torch.float32, device=flow.device).view(1, 1, w) + flow[:, 0].float()
    ty = torch.arange(h, dtype=torch.float32, device=flow.device).view(1, h, 1) + flow[:, 1].float()
    x0, y0 = torch.floor(tx), torch.floor(ty)
    out = torch.zeros(c, h * w, device=flow.device)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cx, cy = x0 + dx, y0 + dy
        wx = tx - x0 if dx else x0 + 1 - tx
        wy = ty - y0 if dy else y0 + 1 - ty
        keep = ((cx >= 0) & (cx < w) & (cy >= 0) & (cy < h))[0].reshape(-1)
        index = (cy * w + cx).long()[0].reshape(-1)[keep]
        out.index_add_(1, index, (values[0].reshape(c, -1) * (wx * wy)[0].reshape(1, -1))[:, keep])
    out = out.view(1, c, h, w)
    return out[:, :-1], out[:, -1:]


class Gmfss:
    """Une paire de GMFSS Union : ses sorties « paire » une fois, puis des fusions à la demande."""

    def __init__(self, nets, a, b):
        import torch
        import torch.nn.functional as F

        self.pair, self.fusion, self.device = nets
        self.a = torch.from_numpy(a)[None].to(self.device)
        self.b = torch.from_numpy(b)[None].to(self.device)
        with torch.inference_mode():
            self.out = self.pair(torch.cat([self.a, self.b], 1))
        named = dict(zip(__import__("gmfss_graph").pair_names(), self.out))
        up = lambda x: F.interpolate(x.float(), scale_factor=2.0, mode="bilinear")  # noqa: E731
        m0, m1 = named["motion0_1"], named["motion1_1"]
        self.f01, self.f10 = up(m0[:, 0:2]) * 2.0, up(m1[:, 0:2]) * 2.0  # flots en pixels de l'image entière
        self.z0, self.z1 = up(m0[:, 2:3]), up(m1[:, 2:3])
        self.fusions = 0
        self._middle = None

    def at(self, t):
        import torch
        from gmfss_graph import splat_all

        with torch.inference_mode():
            half0, half1, warped = splat_all(self.out, t)
            time_t = torch.full((1, 1, 1, 1), float(t), device=self.device)
            self.fusions += 1
            return self.fusion(half0, half1, time_t, *warped)[0].float()

    def middle(self):
        if self._middle is None:
            self._middle = self.at(0.5)
        return self._middle

    def side(self, t):
        """L'image de la paire la plus proche de t splattée à t (sommes), et sa distance en temps."""
        if t <= 0.5:
            return splat_sums(self.a, self.f01 * t, self.z0 * t), t
        return splat_sums(self.b, self.f10 * (1 - t), self.z1 * (1 - t)), 1 - t

    def splats(self, t):
        """GMFSS sans GridNet : les deux images splattées à t, chacune pesant 1 − sa distance."""
        (n0, d0), (n1, d1) = splat_sums(self.a, self.f01 * t, self.z0 * t), splat_sums(self.b, self.f10 * (1 - t),
                                                                                       self.z1 * (1 - t))
        w0, w1 = 1 - t, t
        return ((w0 * n0 + w1 * n1) / (w0 * d0 + w1 * d1 + 1e-7))[0]

    def from_middle(self, t):
        """L'instant t déduit de la seule fusion du milieu par le flot de GMFSS (voir l'en-tête)."""
        import torch

        if not hasattr(self, "_mid_flow"):
            # Le déplacement du milieu vers b, là où chaque point est au milieu : ½·F01 splatté depuis a, −½·F10
            # depuis b, mêlés par leurs poids ; la métrique du milieu de même, pour l'ordre des occlusions.
            na, da = splat_sums(torch.cat([0.5 * self.f01, 0.5 * self.z0], 1), 0.5 * self.f01, 0.5 * self.z0)
            nb, db = splat_sums(torch.cat([-0.5 * self.f10, 0.5 * self.z1], 1), 0.5 * self.f10, 0.5 * self.z1)
            mid = (na + nb) / (da + db + 1e-7)
            self._mid_flow, self._mid_z = mid[:, 0:2], mid[:, 2:3]
        s = (t - 0.5) / 0.5  # fraction du demi-mouvement, signée
        nm, dm = splat_sums(self.middle()[None], self._mid_flow * s, self._mid_z * abs(s))
        (nk, dk), dist = self.side(t)
        wm, wk = 1.0 / abs(t - 0.5), 1.0 / dist
        return ((wm * nm + wk * nk) / (wm * dm + wk * dk + 1e-7))[0].clamp(0, 1)


    def compose(self, t, fused, window, high_a, high_b):
        """La composition après SPAN du lecteur d'avant 0.10.0 (gmfss_compose, kernels/spec/gmfss.h) :
        out = H + up(F − L), H le mélange des sources agrandies splattées par le flot du niveau 1, L le même
        mélange à la taille de travail, F la fusion. Ici les vraies images 1080p jouent les sources agrandies
        (un SPAN parfait) : l'écart à la vérité ne vient que de la composition."""
        import torch
        import torch.nn.functional as F

        crop = lambda x: x[:, :, window[1], window[2]]  # noqa: E731
        f01, f10, z0, z1 = crop(self.f01), crop(self.f10), crop(self.z0), crop(self.z1)
        low_a, low_b = crop(self.a), crop(self.b)
        factor = high_a.shape[1] // low_a.shape[2]
        hi = lambda x: F.interpolate(x, scale_factor=float(factor), mode="bilinear")  # noqa: E731

        def mix(a, b, f01, f10, z0, z1):
            parts = []
            for src, flow, z, s in ((a, f01, z0, t), (b, f10, z1, 1 - t)):
                n, d = splat_sums(src, flow * s, z * s)
                parts.append((n / (d + 1e-7), d / (d + 1e-7)))
            (p0, c0), (p1, c1) = parts
            w0, w1 = (1 - t) * c0, t * c1
            return (w0 * p0 + w1 * p1) / (w0 + w1 + 1e-7)

        th = lambda x: torch.from_numpy(x)[None].to(self.device)  # noqa: E731
        high = mix(th(high_a), th(high_b), hi(f01) * factor, hi(f10) * factor, hi(z0), hi(z1))
        low = mix(low_a, low_b, f01, f10, z0, z1)
        fused = torch.from_numpy(fused)[None].to(self.device)
        return (high + hi(fused - low)).clamp(0, 1)[0].cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# RIFE (ONNX Runtime, comme rife_instants.py mais à la taille de travail)

def rife_flow(sess, names, a, b, t):
    (pa, (top, left)), (pb, _) = modeles.pad(a, modeles.ALIGN), modeles.pad(b, modeles.ALIGN)
    x = modeles.rife_input(pa, pb, t, 11).astype(np.float16)
    _, flow, mask = sess.run(names, {sess.get_inputs()[0].name: x})
    _, h, w = a.shape
    window = (slice(None), slice(top, top + h), slice(left, left + w))
    flow = flow[0].astype(np.float32)[window]
    mask = 1.0 / (1.0 + np.exp(-mask[0].astype(np.float32)[window]))
    return rt.blend(a, b, flow, mask), flow, mask


def rife_at(sess, names, a, b, t):
    (pa, (top, left)), (pb, _) = modeles.pad(a, modeles.ALIGN), modeles.pad(b, modeles.ALIGN)
    x = modeles.rife_input(pa, pb, t, 11).astype(np.float16)
    _, flow, mask = sess.run(names, {sess.get_inputs()[0].name: x})
    _, h, w = a.shape
    window = (slice(None), slice(top, top + h), slice(left, left + w))
    flow = flow[0].astype(np.float32)[window]
    mask = 1.0 / (1.0 + np.exp(-mask[0].astype(np.float32)[window]))
    return rt.blend(a, b, flow, mask), float(np.percentile(np.abs(flow).max(0), 99.9))


# ---------------------------------------------------------------------------------------------------------------

def work(img, height):
    width = int(round(img.shape[2] * height / img.shape[1] / 2)) * 2
    return img if height == img.shape[1] else rt.resize(img, width, height, rt._area_axis)


def padded(img):
    p, (top, left) = modeles.pad(img, GMFSS_ALIGN)
    _, h, w = img.shape
    return p, (slice(None), slice(top, top + h), slice(left, left + w))


def save_png(path, img):
    import subprocess

    _, h, w = img.shape
    data = (np.clip(img, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8).tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                    path], input=data, check=True)


def perceptual(model, ref, out):
    """LPIPS (AlexNet) : plus bas, plus proche de la vérité à l'œil ; le PSNR, lui, préfère les images douces."""
    import torch

    th = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32))[None] * 2 - 1  # noqa: E731
    with torch.inference_mode():
        return float(model(th(ref), th(out)).item())


def suites(frames, count, gap):
    """count suites k..k+gap d'images toutes différentes d'un même plan, réparties dans l'extrait (à gap = 3, les
    quadruples de rife_instants.py). Un grand écart imite une source à faible cadence : k et k+gap forment la paire,
    les images du milieu sont les vérités à i/gap."""
    from sources import differs

    cand = []
    for k in range(len(frames) - gap):
        win = frames[k:k + gap + 1]
        if all(differs(win[i], win[i + 1]) for i in range(gap)) and float(np.abs(win[0] - win[gap]).mean()) < CUT:
            cand.append(k)
    return [(k, frames[k:k + gap + 1]) for k in cand[::max(1, len(cand) // count)][:count]]


def instants(row):
    return range(1, row.get("ecart", 3))


def measure(nets, rife, win, height, high=False, lpips_model=None, images=None, key=0, variants=()):
    import torch

    full = win
    win = [work(x, height) for x in win]
    n = len(win) - 1
    a, b = win[0], win[n]
    (pa, window), (pb, _) = padded(a), padded(b)
    started = time.perf_counter()
    g = Gmfss(nets, pa, pb)
    crop = lambda x: x.cpu().numpy()[window] if isinstance(x, torch.Tensor) else x  # noqa: E731
    others = [(name, Gmfss(n, pa, pb)) for name, n in variants]
    _, motion = rife_at(*rife, a, b, 0.5)
    row = {"mouvement": motion, "ecart": n}
    for i in range(1, n):
        t = i / n
        mid = crop(g.middle())
        half = (a, mid, 2 * t) if t < 0.5 else (mid, b, 2 * t - 1)
        outs = {
            "rife": rife_at(*rife, a, b, t)[0],
            "gmfss": mid if t == 0.5 else crop(g.at(t)),
            "splats": crop(g.splats(t)),
            "milieu": mid if t == 0.5 else crop(g.from_middle(t)),
            "milieu_rife": mid if t == 0.5 else rife_at(*rife, *half)[0],
        }
        for name, other in others:
            outs[name] = crop(other.at(t))
        if images:
            save_png(os.path.join(images, f"{key:05d}_{i}_verite.png"), win[i])
        for name, out in outs.items():
            out = np.clip(out, 0, 1)
            row[f"{name}_{i}"] = mesures.psnr(win[i], out)
            row[f"pire_bloc_{name}_{i}"] = rt.worst_block(win[i], out)
            if lpips_model is not None:
                row[f"lpips_{name}_{i}"] = perceptual(lpips_model, win[i], out)
            if images:
                save_png(os.path.join(images, f"{key:05d}_{i}_{name}.png"), out)
        if high:
            # À la sortie (2 × la taille de travail) : GMFSS agrandi en bilinéaire, la composition d'avant,
            # et RIFE déformant les images 1080p par son flot remonté (le warp après SPAN de RIFE).
            factor = full[0].shape[1] // height
            hw, hh = full[0].shape[2], full[0].shape[1]
            fused = outs["gmfss"]
            rife_out, rflow, rmask = rife_flow(*rife, a, b, t)
            up = rt.resize(rflow, hw, hh, rt._linear_axis) * factor
            outs_hi = {
                "haut_gmfss_bilineaire": rt.resize(fused, hw, hh, rt._linear_axis),
                "haut_compose": g.compose(t, fused, window, full[0], full[n]),
                "haut_rife": rt.blend(full[0], full[n], up, rt.resize(rmask, hw, hh, rt._linear_axis)),
            }
            for name, out in outs_hi.items():
                out = np.clip(out, 0, 1)
                row[f"{name}_{i}"] = mesures.psnr(full[i], out)
                row[f"pire_bloc_{name}_{i}"] = rt.worst_block(full[i], out)
    row["secondes"] = time.perf_counter() - started
    return row


def summary(rows, height):
    global METHODS
    METHODS = tuple(m for m in dict.fromkeys(METHODS + ("gmfss_demi", "gmfss_neuf")) if f"{m}_1" in rows[0])
    gap = rows[0].get("ecart", 3)
    count = sum(len(instants(r)) for r in rows)
    lines = [f"Taille de travail {height}p, {len(rows)} paires (k, k + {gap}), instants i/{gap} : {count} instants.",
             "", "| Mouvement (px à la taille de travail) | Instants | " +
             " | ".join(f"{m} (dB)" for m in METHODS) + " | gmfss − rife | milieu − gmfss | milieu_rife − gmfss |",
             "|---|---:|" + "---:|" * (len(METHODS) + 3)]
    edges = (0, 15, 30, 45, 60, 90, 1e9)
    for lo, hi in list(zip(edges, edges[1:])) + [(0, 1e9)]:
        group = [r for r in rows if lo <= r["mouvement"] < hi]
        if not group:
            continue
        mean = {m: float(np.mean([r[f"{m}_{i}"] for r in group for i in instants(r)])) for m in METHODS}
        label = "toutes" if (lo, hi) == (0, 1e9) else (f"{lo} à {hi}" if hi < 1e9 else f"{lo} et plus")
        lines.append(f"| {label} | {sum(len(instants(r)) for r in group)} | " + " | ".join(f"{mean[m]:.2f}" for m in METHODS) +
                     f" | {mean['gmfss'] - mean['rife']:+.2f} | {mean['milieu'] - mean['gmfss']:+.2f} | "
                     f"{mean['milieu_rife'] - mean['gmfss']:+.2f} |")
    if "lpips_gmfss_1" in rows[0]:
        lines += ["", "LPIPS moyen (plus bas : plus proche à l'œil) : " + ", ".join(
            f"{m} {np.mean([r[f'lpips_{m}_{i}'] for r in rows for i in instants(r)]):.4f}" for m in METHODS)]
    lines += ["", "Pire bloc 64 × 64, moyenne (dB) : " + ", ".join(
        f"{m} {np.mean([r[f'pire_bloc_{m}_{i}'] for r in rows for i in instants(r)]):.2f}" for m in METHODS)]
    worse = {m: min(r[f"{m}_{i}"] - r[f"gmfss_{i}"] for r in rows for i in instants(r)) for m in METHODS if m != "gmfss"}
    lines.append("Pire écart d'un instant à GMFSS (dB) : " + ", ".join(f"{m} {v:+.2f}" for m, v in worse.items()))
    return "\n".join(lines).replace(".", ",") + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, help="un extrait 1080p (24 i/s)")
    p.add_argument("--gmfss", required=True, help="le dossier vsgmfss_fortuna extrait de la roue")
    p.add_argument("--reseau", required=True, help="rife_v4.26.onnx (11 plans)")
    p.add_argument("--device", default="cpu", help="cpu ou cuda (GMFSS)")
    p.add_argument("--echelle", type=float, default=1.0,
                   help="échelle du flot de GMFlow (1 : celle de GMFSS ; 0,5 : le demi-flot du lecteur d'avant)")
    p.add_argument("--hauteur", type=int, default=720, help="taille de travail")
    p.add_argument("--sortie", action="store_true",
                   help="mesure aussi à 1080p (taille de travail 540) : GMFSS agrandi, la composition après SPAN "
                        "d'avant 0.10.0, RIFE après SPAN")
    p.add_argument("--demi", action="store_true", help="ajoute GMFSS au demi-flot (gmfss_demi), celui du lecteur d'avant")
    p.add_argument("--union-neuf", help="le train_log/ du « new union model » de 98mxr (gmfss_neuf)")
    p.add_argument("--lpips", action="store_true", help="ajoute LPIPS (pip install lpips)")
    p.add_argument("--images-dir", help="écrit chaque instant de chaque méthode en PNG, et la vérité")
    p.add_argument("--ecart", type=int, default=3, help="images d'écart de la paire (3 : vérités à ⅓ et ⅔)")
    p.add_argument("--suites", type=int, default=40)
    p.add_argument("--images", type=int, default=740, help="images lues au début de l'extrait")
    p.add_argument("--json")
    p.add_argument("--md")
    args = p.parse_args()
    import torch

    torch.set_grad_enabled(False)
    wins = suites(rt.decode(args.video, args.images), args.suites, args.ecart)
    nets = load(args.gmfss, args.device, args.echelle)
    variants = []
    if args.demi:
        variants.append(("gmfss_demi", load(args.gmfss, args.device, 0.5)))
    if args.union_neuf:
        variants.append(("gmfss_neuf", load(args.gmfss, args.device, args.echelle, args.union_neuf)))
    lpips_model = None
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    if args.images_dir:
        os.makedirs(args.images_dir, exist_ok=True)
    rife = rt.motion_session(args.reseau)
    rows = []
    for k, win in wins:
        rows.append({"image": k, **measure(nets, rife, win, args.hauteur, args.sortie, lpips_model, args.images_dir, k, variants)})
        print(json.dumps({key: round(v, 2) if isinstance(v, float) else v for key, v in rows[-1].items()}), flush=True)
    text = summary(rows, args.hauteur)
    if args.sortie:
        hi = ("haut_gmfss_bilineaire", "haut_compose", "haut_rife")
        text += "\nÀ la sortie 1080p (vraies images 1080p comme sources agrandies), moyenne (dB) : " + ", ".join(
            f"{m} {np.mean([r[f'{m}_{i}'] for r in rows for i in instants(r)]):.2f}".replace(".", ",") for m in hi)
        text += " ; pire bloc : " + ", ".join(
            f"{m} {np.mean([r[f'pire_bloc_{m}_{i}'] for r in rows for i in instants(r)]):.2f}".replace(".", ",")
            for m in hi) + "\n"
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(rows, out, indent=1)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


if __name__ == "__main__":
    main()
