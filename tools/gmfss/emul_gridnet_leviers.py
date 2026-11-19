#!/usr/bin/env python3
"""Deux leviers du GridNet de la fusion GMFSS jugés sur processeur avant tout noyau : Winograd et le FP8 (E4M3)
(docs/gmfss-natif.md, « Winograd et FP8 du GridNet, émulés »).

Rejoue le GridNet comme emul_mix_native.py (mêmes arrondis FP16 des noyaux : poids en FP16, PReLU de l'entrée
arrondi au chargement, somme et biais arrondis, résiduel puis PReLU de sortie arrondis), morceau par morceau
(lignes 0, 1, 2 et queue), avec pour chaque morceau une recette :

  direct  le calcul des noyaux d'aujourd'hui, sommes f16, mixtes ou f32 dans leur ordre (emul_mix_native) ;
  wino23, wino43
          Winograd F(2×2, 3×3) ou F(4×4, 3×3) sur les convolutions 3×3 de pas 1 du morceau (méthode de
          tools/bench/span_native/winograd_db.py, PR #976) : poids transformés en f32 puis arrondis en f16
          (au chargement), entrée transformée en f32 puis arrondie en f16 (ce que lirait le mma), produits
          sommés sur les canaux par pas de 16, l'accumulateur en f16 (arrondi après chaque mma) ou en f32,
          transformée de sortie en f32, puis biais et arrondi comme le direct. Les descentes (pas 2), les
          transposées et conv_last (3 sorties) restent au direct ;
  fp8     E4M3 (float8_e4m3fn, saturé à 448) sur toutes les convolutions du morceau, transposées comprises,
          sauf conv_last : les poids seuls (déquantifiés, puis le calcul f16 des noyaux : un contrôle de
          l'erreur des poids, sans gain de calcul, aucun mma ne prend f16 × FP8), ou les poids et les
          activations (l'entrée de chaque convolution quantifiée au chargement, après son PReLU). Échelles des
          poids par tenseur ou par canal de sortie ; des activations par tenseur ou par canal d'entrée (alors
          repliée dans les poids : W[o, c] · s_c, puis l'échelle par canal de sortie), calibrées hors test
          (le maximum × 1,25 / 448 sur trois paires qui ne sont pas des paires de test, comme SPAN et GMFlow).
          Produits exacts, sommés par mma de 32 (k de m16n8k32) en f32 ou dans un accumulateur f16 en unités
          quantifiées (un dépassement de 65 504 y donne l'infini, comme le mma).

Les autres morceaux restent au direct en sommes f32 (« seul ») ou suivent le mélange demandé. Chaque variante est
mesurée sur l'image de la fusion contre ONNX Runtime FP32 (la fusion exportée, tout tenseur FP16 passé en FP32 :
tools/bench/qualite/fp32.py) et contre le natif f32 émulé, en dB au pire et en moyenne sur les paires de test.
Les entrées du GridNet sont celles de la fusion FP32 (ses tenseurs relevés par la structure), arrondies en FP16
comme les écrivent les splats et l'IFNet du lecteur.

Les paires de test sont à mouvement fort, puisque le FP8 de GMFSS a été retiré pour une mauvaise image dans les
gros mouvements (docs/fp8-interpolation.md) : de vraies paires d'images voisines de l'extrait de Re:Zero
(combat, effets, personnage en course), et deux mouvements de caméra connus sur une image de cet extrait
(panoramique de 96 pixels ; translation (64, −40), rotation de 2° et un objet qui traverse le décor).

    python -I tools/gmfss/emul_gridnet_leviers.py PAIRE.onnx FUSION.onnx --video EXTRAIT.mkv --work DOSSIER \\
        [--levier winograd|fp8|tout] [--width 1024 --height 576] [--paires N]

DOSSIER garde les entrées du GridNet de chaque paire et la calibration (une relance ne refait pas ONNX Runtime)
et y écrit resultats-<levier>.json, que lit plafond_leviers.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time as clock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "qualite"))
import fp32  # noqa: E402  (tools/bench/qualite)
import emul_mix_native as mix  # noqa: E402
from emul_mix_native import BLOCKS, LINE, Fusion, finish, h16, prelu, stage  # noqa: E402
from emul_pair_native import tensor_names  # noqa: E402
from gmflow_natif_emule import deplace  # noqa: E402
from gmfss_graph import fusion_names, splat_all  # noqa: E402

E4M3_MAX = 448.0
PIECES = ["ligne 0", "ligne 1", "ligne 2", "queue"]

# Winograd (Lavin et Gray) : Y = Aᵀ [(G g Gᵀ) ⊙ (Bᵀ d B)] A, les matrices de winograd_db.py.
BT23 = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
G23 = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
AT23 = [[1, 1, 1, 0], [0, 1, -1, -1]]
BT43 = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0],
        [0, 4, 0, -5, 0, 1]]
G43 = [[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6], [1 / 24, 1 / 12, 1 / 6],
       [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]]
AT43 = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]]
WINO = {name: tuple(torch.tensor(m, dtype=torch.float64) for m in mats) + (step,)
        for name, mats, step in (("wino23", (BT23, G23, AT23), 2), ("wino43", (BT43, G43, AT43), 4))}

# Les paires de test : (nom, image n de l'extrait, image n + 1 ou mouvement connu (dx, dy, degrés, objet) en
# pixels d'une paire de 1024 de large). Les paires de calibration du FP8 sont d'autres images.
TESTS = [
    ("combat", 453, None), ("course", 383, None), ("effets", 160, None), ("éclair", 178, None),
    ("panoramique", 600, (96, 0, 0.0, False)), ("fort", 300, (64, -40, 2.0, True)),
]
CALIBRATION = [("c-combat", 455, None), ("c-course", 382, None), ("c-fort", 650, (64, -40, 2.0, True))]


@dataclass(frozen=True)
class Recipe:
    """La recette d'un morceau : sommes (f16, mixed, f32), algo (direct, wino23, wino43) ; fp8 : None ou
    (échelle des poids « tensor » ou « channel », échelle des activations None, « tensor » ou « channel »)."""

    sums: str = "f32"
    algo: str = "direct"
    fp8: tuple | None = None

    def label(self):
        if self.fp8:
            w, a = self.fp8
            return f"fp8 poids/{w}" + (f" act/{a}" if a else " (poids seuls)") + f", sommes {self.sums}"
        return f"{self.algo}, sommes {self.sums}"


def q8(t):
    """E4M3 saturé (cvt.rn.satfinite), rendu en f32."""
    return t.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def weight_scale(w, mode):
    """Échelle E4M3 de poids [O, C, …] par tenseur ou par canal de sortie, de forme [O]."""
    m = w.abs().amax(dim=tuple(range(1, w.dim()))) if mode == "channel" else w.abs().max().expand(w.shape[0])
    return torch.where(m > 0, m / E4M3_MAX, torch.ones_like(m))


def chunked(views, weights, sums, kc, step):
    """Les sommes dans l'ordre des noyaux (tranche kc, prise, pas de step canaux), comme mix.accumulate ;
    step = 32 pour un mma FP8 (k de m16n8k32)."""
    channels = views[0].shape[0]
    acc = torch.zeros(weights[0].shape[0], views[0].shape[1])
    for q0 in range(0, (channels + 15) // 16 * 16, kc):
        for view, weight in zip(views, weights):
            part = torch.zeros_like(acc) if sums == "mixed" else None
            for s0 in range(q0, min(q0 + kc, channels), step):
                d = weight[:, s0 : s0 + step] @ view[s0 : s0 + step]
                if sums == "f16":
                    acc = h16(acc + d)
                elif sums == "mixed":
                    part = h16(part + d)
                else:
                    acc = acc + d
            if part is not None:
                acc = acc + part
    return acc


def taps(x, w, stride):
    """Les prises d'une convolution 3 × 3 de bord 1 : views [C, P], weights [O, C], la taille de sortie."""
    c, height, width = x.shape
    oh, ow = (height - 1) // stride + 1, (width - 1) // stride + 1
    xp = F.pad(x, (1, 1, 1, 1))
    views = [xp[:, dy : dy + stride * (oh - 1) + 1 : stride, dx : dx + stride * (ow - 1) + 1 : stride].reshape(c, -1)
             for dy in range(3) for dx in range(3)]
    return views, [w[:, :, dy, dx] for dy in range(3) for dx in range(3)], (oh, ow)


def phase_taps(x, w, py, px):
    """Les quatre prises d'une phase de la transposée 4 × 4 de pas 2 (poids ONNX [Cin][Cout][4][4])."""
    c, height, width = x.shape
    xp = F.pad(x, (1, 1, 1, 1))
    views, weights = [], []
    for ty in range(2):
        for tx in range(2):
            views.append(xp[:, py + ty : py + ty + height, px + tx : px + tx + width].reshape(c, -1))
            weights.append(w[:, :, 3 - py - 2 * ty, 3 - px - 2 * tx].t())
    return views, weights


def conv_wino(x, w, b, sums, mode):
    """Une convolution 3 × 3 de pas 1 par Winograd sur x [C, H, W] (déjà passé par son PReLU)."""
    bt, g, at, m = WINO[mode]
    a = m + 2
    c, height, width = x.shape
    th, tw = -(-height // m), -(-width // m)
    xp = F.pad(x, (1, tw * m - width + 1, 1, th * m - height + 1))
    tiles = xp.unfold(1, a, m).unfold(2, a, m).double()  # [C, th, tw, a, a]
    v = h16(torch.einsum("pi,ctwij,qj->pqtwc", bt, tiles, bt).float()).reshape(a * a, th * tw, c)
    u = h16(torch.einsum("pi,ocij,qj->pqco", g, h16(w).double(), g).float()).reshape(a * a, c, -1)
    if sums == "f32":
        mm = v @ u
    else:
        mm = torch.zeros(a * a, th * tw, u.shape[-1])
        for c0 in range(0, c, 16):
            mm = h16(mm + v[:, :, c0 : c0 + 16] @ u[:, c0 : c0 + 16])
    y = torch.einsum("ip,pqtwo,jq->otiwj", at, mm.double().reshape(a, a, th, tw, -1), at)
    y = y.reshape(-1, th * m, tw * m)[:, :height, :width].float()
    return stage(y.reshape(y.shape[0], -1), b).reshape(-1, height, width)


class Gridnet(Fusion):
    """Le GridNet de emul_mix_native.Fusion, chaque convolution selon la recette de son morceau."""

    def __init__(self, model):
        super().__init__(model)
        self.calib = {}  # clé → (max par tenseur, max par canal) de l'entrée, sur les paires de calibration
        self.record = False

    def note(self, key, t):
        """Calibration : le plus grand |t| du tenseur et de chaque canal (l'entrée, ou la sortie « key> »)."""
        if self.record:
            old = self.calib.get(key, (torch.tensor(0.0), torch.zeros(t.shape[0])))
            self.calib[key] = (torch.maximum(old[0], t.abs().max()), torch.maximum(old[1], t.abs().amax(dim=(1, 2))))

    def act_scale(self, key, x, mode):
        if mode == "tensor":
            return torch.full((x.shape[0],), max(float(self.calib[key][0]), 1e-6) * 1.25 / E4M3_MAX)
        m = self.calib[key][1]
        return torch.where(m > 0, m, torch.ones_like(m)) * 1.25 / E4M3_MAX

    def fp8_parts(self, key, x, w, r, kind):
        """Entrée et poids quantifiés (en unités E4M3) et l'échelle de chaque sortie ; poids seuls : déquantifiés."""
        wmode, amode = r.fp8
        wt = w.transpose(0, 1) if kind == "up" else w  # [O, C, …] (transposée : [Cin][Cout][4][4])
        if amode is None:
            s = weight_scale(wt, wmode)
            wq = h16(q8(wt / s.view(-1, *[1] * (wt.dim() - 1))) * s.view(-1, *[1] * (wt.dim() - 1)))
            return x, (wq.transpose(0, 1) if kind == "up" else wq), None
        sa = self.act_scale(key, x, amode)  # [C]
        wt = wt * sa.view(1, -1, *[1] * (wt.dim() - 2))
        s = weight_scale(wt, wmode)
        if r.sums == "f16":
            # L'accumulateur f16 tient Σ (x/sa)(W·sa/s) = y/s : l'échelle de chaque sortie est relevée jusqu'à
            # ce que le plus grand |y| de la calibration (× 1,25) y reste sous 16 384, le quart de la crête du
            # f16 (les sommes partielles dépassent la somme finale) ; les poids perdent alors le haut de l'E4M3.
            s = torch.maximum(s, self.calib[key + ">"][1] * 1.25 / 16384)
        wq = q8(wt / s.view(-1, *[1] * (wt.dim() - 1)))
        xq = q8(x / sa.view(-1, 1, 1))
        return xq, (wq.transpose(0, 1) if kind == "up" else wq), s

    def conv(self, key, x, wb, r, stride=1, in_slope=None, out_slope=None, skip=None, lever=True):
        w, b = wb
        if in_slope is not None:
            x = h16(prelu(x, in_slope))
        self.note(key, x)
        if lever and r.fp8:
            xq, wq, s = self.fp8_parts(key, x, w, r, "same")
            if s is None:  # poids seuls : le calcul f16 des noyaux sur les poids déquantifiés
                return mix.conv(xq, (wq, b), r.sums, stride=stride, out_slope=out_slope, skip=skip)
            views, weights, (oh, ow) = taps(xq, wq, stride)
            acc = chunked(views, weights, r.sums, 64 if r.sums == "mixed" else 1 << 30, 32 if x.shape[0] >= 32 else 16)
            y = stage(acc * s[:, None], b).reshape(-1, oh, ow)
        elif lever and r.algo != "direct" and stride == 1:
            y = conv_wino(x, w, b, r.sums, r.algo)
        elif r.sums == "f32":
            y = F.conv2d(x[None], h16(w), None, stride, 1)[0]
            self.note(key + ">", y)
            y = stage(y.flatten(1), b).reshape(y.shape)
        else:
            return mix.conv(x, wb, r.sums, stride=stride, out_slope=out_slope, skip=skip)
        return finish(y, skip, out_slope)

    def conv_t(self, key, x, wb, r, in_slope=None):
        w, b = wb
        if in_slope is not None:
            x = h16(prelu(x, in_slope))
        self.note(key, x)
        if r.fp8:
            xq, wq, s = self.fp8_parts(key, x, w, r, "up")
            if s is None:
                return mix.conv_transposed(xq, (wq, b), r.sums)
            c, height, width = x.shape
            out = torch.zeros(w.shape[1], 2 * height, 2 * width)
            for py in range(2):
                for px in range(2):
                    views, weights = phase_taps(xq, wq, py, px)
                    acc = chunked(views, weights, r.sums, 1 << 30, 32 if c >= 32 else 16)
                    out[:, py::2, px::2] = stage(acc * s[:, None], b).reshape(-1, height, width)
            return out
        if r.sums == "f32":
            y = F.conv_transpose2d(x[None], h16(w), None, stride=2, padding=1)[0]
            self.note(key + ">", y)
            return stage(y.flatten(1), b).reshape(y.shape)
        return mix.conv_transposed(x, wb, r.sums)

    def run(self, x, recipes):
        """Le GridNet sur x (X0 à X3, [C, H, W]) ; recipes : la recette de chaque morceau (PIECES)."""
        t = dict(x)
        for module, kind, src, skip, dst in BLOCKS:
            r = recipes[LINE[dst]]
            s1, s2 = self.slope(module, 0), self.slope(module, 2)
            first, second = self.wb(f"{module}.1"), self.wb(f"{module}.3")
            if kind == "up":
                mid = self.conv_t(f"{module}.1", t[src], first, r, in_slope=s1)
                t[dst] = self.conv(f"{module}.3", mid, second, r, in_slope=s2, skip=t.get(skip))
            else:
                stride = 2 if kind == "down" else 1
                mid = self.conv(f"{module}.1", t[src], first, r, stride=stride, in_slope=s1, out_slope=s2)
                t[dst] = self.conv(f"{module}.3", mid, second, r, skip=t.get(skip))
        r = recipes[3]
        tail = "residual_model_tail"
        slope = self.W.slope(f"/fusionnet/{tail}/conv_before_upsample/conv_before_upsample.1/Mul")
        y = self.conv("before", t["L2"], self.wb(f"{tail}.conv_before_upsample.0"), r, out_slope=slope)
        y = F.pixel_shuffle(self.conv("shuffle", y, self.wb(f"{tail}.upsample.0"), r)[None], 2)[0]
        y = self.conv("last", y, self.wb(f"{tail}.conv_last"), r, lever=False)
        return y[:3].clamp(0, 1)


def frame(video, n, width, height):
    """L'image n de l'extrait, RGB [0, 1], [1, 3, height, width]."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                          f"select=eq(n\\,{n}),scale={width}:{height}:flags=area", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    a = np.frombuffer(out, np.uint8).reshape(height, width, 3).astype(np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def frames_of(video, n, motion, width, height):
    """frames [1, 6, H, W] : deux images voisines de l'extrait, ou une image et sa vue après un mouvement connu
    (mis à l'échelle de la largeur), avec un objet découpé d'une autre image qui traverse le décor."""
    img0 = frame(video, n, width, height)
    if motion is None:
        return torch.cat([img0, frame(video, n + 1, width, height)], 1)
    k = width / 1024
    dx, dy, deg, objet = motion
    img1 = deplace(img0, dx * k, dy * k, 1.0, deg)
    if objet:
        autre = frame(video, n + 120, width, height)
        oh, ow, y0, x0 = height // 3, width // 4, height // 3, width // 3
        sy, sx = round(29 * k), round(-48 * k)
        img0[:, :, y0 : y0 + oh, x0 : x0 + ow] = autre[:, :, :oh, :ow]
        img1[:, :, y0 + sy : y0 + sy + oh, x0 + sx : x0 + sx + ow] = autre[:, :, :oh, :ow]
    return torch.cat([img0, img1], 1)


def gridnet_inputs(args, gridnet, pairs):
    """Pour chaque paire : les entrées du GridNet (arrondies en FP16) et l'image de la fusion FP32 d'ORT."""
    pair_model = fp32.promote(onnx.load(str(args.pair)))
    session = ort.InferenceSession(pair_model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    fusion_model = fp32.promote(onnx.load(str(args.fusion)))
    for x in gridnet.inputs.values():
        fusion_model.graph.output.append(onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None))
    fusion = ort.InferenceSession(fusion_model.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = [o.name for o in fusion.get_outputs()]
    data = []
    for name, n, motion in pairs:
        cache = args.work / f"entrees-{name}-{args.width}x{args.height}.pt"
        if not cache.exists():
            frames = frames_of(args.video, n, motion, args.width, args.height)
            pair_out = [torch.from_numpy(v) for v in session.run(None, {"frames": frames.numpy()})]
            flow = float(pair_out[names.index("motion0_1")][:, :2].norm(dim=1).amax()) * 2
            half0, half1, warped = splat_all(pair_out, args.time)
            time = torch.tensor([[[[args.time]]]], dtype=torch.float32)
            feeds = dict(zip(fusion_names("union"), [half0, half1, time, *warped]))
            got = dict(zip(outs, fusion.run(None, {k: v.numpy() for k, v in feeds.items()})))
            x = {k: h16(torch.from_numpy(got[v])[0]) for k, v in gridnet.inputs.items()}
            torch.save({"x": x, "ref": torch.from_numpy(got["out"])[0], "flow": flow}, cache)
        data.append((name, torch.load(cache)))
    return data


def psnr(got, ref):
    mse = float(((got.double() - ref.double()) ** 2).mean())
    return float("-inf") if not np.isfinite(mse) else 200.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def variants(levier):
    """(nom, recettes des quatre morceaux) : chaque morceau seul (les autres au direct f32), puis les mélanges."""
    f32, f16 = Recipe(), Recipe("f16")
    out = [("natif f32", [f32] * 4), ("natif f16 (retenu au PC)", [f16] * 4)]
    alone = [("direct", f16)]
    mixes = []
    if levier in ("winograd", "tout"):
        for mode in ("wino23", "wino43"):
            alone += [(mode, Recipe("f16", mode)), (mode, Recipe("f32", mode))]
            mixes += [(f"{mode} partout, sommes f16", [Recipe("f16", mode)] * 4),
                      (f"{mode} partout, sommes f32", [Recipe("f32", mode)] * 4),
                      (f"{mode} lignes 1, 2 et queue en f16, ligne 0 direct f16", [f16] + [Recipe("f16", mode)] * 3)]
    if levier in ("fp8", "tout"):
        for wa in (("tensor", None), ("channel", None), ("channel", "tensor"), ("channel", "channel")):
            alone.append(("fp8", Recipe("f32", fp8=wa)))
        alone.append(("fp8", Recipe("f16", fp8=("channel", "channel"))))
        for sums in ("f32", "f16"):
            r = Recipe(sums, fp8=("channel", "channel"))
            mixes += [(f"FP8 act/canal lignes 1 et 2, sommes {sums}, le reste direct f16", [f16, r, r, f16]),
                      (f"FP8 act/canal lignes 0, 1, 2, sommes {sums}, queue direct f16", [r, r, r, f16]),
                      (f"FP8 act/canal partout, sommes {sums}", [r] * 4)]
    for p, piece in enumerate(PIECES):
        for _, r in alone:
            recipes = [f32] * 4
            recipes[p] = r
            out.append((f"{piece} seul : {r.label()}", recipes))
    return out + mixes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair", type=Path)
    parser.add_argument("fusion", type=Path)
    parser.add_argument("--video", type=Path, required=True, help="l'extrait de Re:Zero (rezero-s4e18-1710.mkv)")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--levier", choices=("winograd", "fp8", "tout"), default="tout")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--paires", type=int, default=len(TESTS), help="les N premières paires de test")
    parser.add_argument("--seul", help="ne passer que les variantes dont le nom contient ce texte")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    args.work.mkdir(parents=True, exist_ok=True)

    gridnet = Gridnet(onnx.load(str(args.fusion)))
    data = gridnet_inputs(args, gridnet, TESTS[: args.paires])
    print(f"paires {args.width}×{args.height} (GridNet à la moitié), instant {args.time} ; flot le plus long : "
          + ", ".join(f"{n} {d['flow']:.0f} px" for n, d in data))
    rows = variants(args.levier)
    if args.levier in ("fp8", "tout"):
        calib = args.work / f"calibration-{args.width}x{args.height}.pt"
        if not calib.exists():
            gridnet.record = True
            for _, d in gridnet_inputs(args, gridnet, CALIBRATION):
                gridnet.run(d["x"], [Recipe()] * 4)
            gridnet.record = False
            torch.save(gridnet.calib, calib)
        gridnet.calib = torch.load(calib)
    if args.seul:
        rows = [r for r in rows if args.seul in r[0] or r[0].startswith("natif")]

    base, results = {}, []
    for name, recipes in rows:
        start = clock.time()
        ort_db, f32_db = [], []
        for pair, d in data:
            y = gridnet.run(d["x"], recipes)
            if name == "natif f32":
                base[pair] = y
            ort_db.append(psnr(y, d["ref"]))
            f32_db.append(psnr(y, base[pair]))
        row = {"variante": name, "recettes": [r.label() for r in recipes], "ort": ort_db, "f32": f32_db}
        results.append(row)
        print(f"{name:72s} ORT pire {min(ort_db):6.1f} moy {np.mean(ort_db):6.1f} | natif f32 pire {min(f32_db):6.1f} "
              f"moy {np.mean(f32_db):6.1f} | " + " ".join(f"{v:.1f}" for v in ort_db)
              + f"  ({clock.time() - start:.0f} s)", flush=True)
    out = args.work / f"resultats-{args.levier}{'-' + args.seul if args.seul else ''}.json"
    out.write_text(json.dumps({"paires": [n for n, _ in data], "lignes": results}, ensure_ascii=False, indent=1))
    print(f"écrit {out}")


if __name__ == "__main__":
    main()
