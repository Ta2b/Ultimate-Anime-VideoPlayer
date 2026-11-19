#!/usr/bin/env python3
"""Les couches d'attention de GMFlow natif rejouées telles que le C++ les lit, contre ONNX Runtime (docs/gmflow-natif.md,
PR 2).

Les couches (poids, fenêtres, décalages, masque, plan) viennent de gmflow_attention_vide (tools/gmfss/gmflow_attention),
qui appelle la lecture du lecteur (prepare/gmflow_attention_native.h) sur la paire ONNX. Ce script exécute une couche
avec les arrondis des noyaux de gmflow_attention_kernels.cuh :
  - GEMM QKV : jetons et poids en FP16, sommes en f32, q, k, v écrits en FP16 (bornés à ±65504) ; en cross-attention,
    k et v pris dans l'autre moitié du lot à l'entrée du bloc ;
  - attention par fenêtres, jetons rangés comme le noyau les lit (fenêtre (wy, wx), jeton t à ((wy · wh + t / ww + sh)
    mod H, (wx · ww + t mod ww + sw) mod W)), tuiles de 64 clés dans l'ordre de la fenêtre : S = q kᵀ en f32, facteur
    et masque en base 2 (s · log2(e)/√d + masque · log2(e) entre deux régions), maximum et somme courants en f32, P
    arrondi en FP16 avant P · V (sommes f32), O · (1/l) arrondi en FP16 ;
  - fusion en FP16 à sommes f32, LayerNorm en f32 (moyenne, puis variance des écarts), résiduel en f32, une écriture
    FP16.
Le contrôle f32 ne fait aucun arrondi : sur les couches lues dans la paire FP32, il doit rendre ONNX Runtime FP32 à
l'ordre des sommes près (poids rangés, fenêtres, décalage, masque, cible de la cross, résiduel). La référence « nœud
par nœud » arrondit en FP16 la sortie de chaque nœud du graphe (q, k, v, scores, scores / √d avec √d en FP16, masque,
softmax, P · V, fusion, LayerNorm, somme), ce que ferait un moteur FP16 qui ne fond rien.

Entrées et sorties des couches : la paire FP32 exécutée par ONNX Runtime (CPU) sur de vraies paires d'images de
data/calibration/calibration.mkv (deux images à 2/24 s d'écart, 1280 × 720 complétées à 768 comme la chaîne : GMFlow
sur 640 × 384, fenêtres de 40 × 24 = 960 jetons au huitième et de 20 × 12 = 240 au quart). Chaque couche part de son
entrée FP32 arrondie en FP16 (ce que le natif aurait stocké) et se compare à sa sortie FP32 : self, la sortie de la
couche (source + LN) ; cross, le message LN que lit le MLP.

    cmake -S tools/gmfss/gmflow_attention -B build/gmflow_attention && cmake --build build/gmflow_attention
    build/gmflow_attention/gmflow_attention_vide paire_fp32.onnx attn32
    build/gmflow_attention/gmflow_attention_vide gmfss_fortuna_union_pair.onnx attn16
    python -I tools/gmfss/gmflow_attention_emule.py --fp32 paire_fp32.onnx --plan32 attn32 --plan16 attn16 \\
        [--paires 4] [--dump DOSSIER]

paire_fp32.onnx : la paire union exportée en FP32 pur (gmflow_natif_emule.py --work DOSSIER l'écrit). --dump écrit,
pour le banc (gmflow_attention_banc verifier), l'entrée FP16, la sortie émulée FP16 et la sortie FP32 de chaque couche
mesurée sur la première paire.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

DEPOT = Path(__file__).resolve().parents[2]
VIDEO = DEPOT / "data/calibration/calibration.mkv"
INSTANTS = (15.0, 39.0, 69.0, 99.0, 51.0, 81.0)
LARGEUR, HAUTEUR = 1280, 768  # la paire : GMFlow sur 640 × 384
BLOCS = (0, 1, 5)  # sans décalage, décalé, le dernier
FP16_MAX = 65504.0
LOG2E = 1.4426950408889634
TUILE = 64


@dataclass
class Couche:
    cross: bool
    decale: bool
    qkv: torch.Tensor  # [384, 128]
    fusion: torch.Tensor  # [128, 128]
    gamma: torch.Tensor
    beta: torch.Tensor


@dataclass
class Plan:
    scale: float = 0.0
    masque: float = 0.0
    epsilon: float = 0.0
    splits: tuple = (0, 0)
    couches: list = field(default_factory=list)


def lit_plan(prefixe: str) -> Plan:
    """Le plan vidé par gmflow_attention_vide : PREFIXE.txt et les poids float32 de PREFIXE.bin."""
    poids = np.fromfile(prefixe + ".bin", dtype="<f4")
    p = Plan()
    for ligne in Path(prefixe + ".txt").read_text().splitlines():
        m = ligne.split()
        if m[0] in ("scale", "mask", "epsilon"):
            setattr(p, {"mask": "masque"}.get(m[0], m[0]), float(m[1]))
        elif m[0] == "splits":
            p.splits = (int(m[1]), int(m[2]))
        elif m[0] == "layer":
            a = int(m[4])
            t = [torch.from_numpy(poids[a + o : a + o + n].copy()) for o, n in ((0, 49152), (49152, 16384),
                                                                                  (65536, 128), (65664, 128))]
            p.couches.append(Couche(m[2] == "1", m[3] == "1", t[0].view(384, 128), t[1].view(128, 128), t[2], t[3]))
    return p


def f16(x: torch.Tensor) -> torch.Tensor:
    """Une valeur f32 telle qu'un noyau l'écrit en FP16 (bornée au plus grand fini, comme to_half2)."""
    return x.clamp(-FP16_MAX, FP16_MAX).half().float()


def fenetres(h: int, w: int, s: int, decale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Les jetons de chaque fenêtre dans l'ordre du noyau (token_of) : indices dans l'image [s², L] et régions."""
    wh, ww = h // s, w // s
    sh, sw = (wh // 2, ww // 2) if decale else (0, 0)
    fen, t = np.arange(s * s), np.arange(wh * ww)
    yr = (fen // s)[:, None] * wh + (t // ww)[None, :]
    xr = (fen % s)[:, None] * ww + (t % ww)[None, :]
    indices = ((yr + sh) % h) * w + (xr + sw) % w
    regions = np.zeros_like(indices)
    if decale:
        regions = ((yr >= h - wh).astype(int) + (yr >= h - sh)) * 3 + (xr >= w - ww) + (xr >= w - sw)
    return torch.from_numpy(indices), torch.from_numpy(regions)


def attention(q, k, v, regions, plan: Plan, arrondi: bool) -> torch.Tensor:
    """L'attention fondue sur des fenêtres [N, L, 128] : tuiles de 64 clés, softmax en ligne en base 2."""
    r = f16 if arrondi else (lambda x: x)
    echelle, masque = plan.scale * LOG2E, plan.masque * LOG2E
    n, longueur, _ = q.shape
    m = torch.full((n, longueur, 1), -math.inf)
    somme = torch.zeros(n, longueur, 1)
    o = torch.zeros(n, longueur, q.shape[-1])
    for j in range(0, longueur, TUILE):
        s = q @ k[:, j : j + TUILE].transpose(1, 2)
        diff = (regions[:, :, None] != regions[:, None, j : j + TUILE]).float()
        x = s * echelle + diff * masque
        haut = torch.maximum(m, x.amax(-1, keepdim=True))
        alpha = torch.exp2(m - haut)
        p = torch.exp2(x - haut)
        somme = somme * alpha + p.sum(-1, keepdim=True)
        o = o * alpha + r(p) @ v[:, j : j + TUILE]
        m = haut
    return r(o * (1.0 / somme))


def couche_native(c: Couche, plan: Plan, src: torch.Tensor, cles: torch.Tensor, h: int, w: int, s: int,
                  arrondi: bool) -> torch.Tensor:
    """Une couche comme les noyaux : src, cles [lot, h · w, 128] (déjà au format de stockage) ; rend la sortie."""
    r = f16 if arrondi else (lambda x: x)
    lot = src.shape[0]
    kv = torch.roll(cles, -(lot // 2), 0) if c.cross else cles  # le lot b lit (b + lot/2) mod lot
    poids = r(c.qkv)
    q, k, v = (r(t @ poids[i * 128 : (i + 1) * 128].t()) for i, t in enumerate((src, kv, kv)))
    indices, regions = fenetres(h, w, s, c.decale)
    pris = [t[:, indices.reshape(-1)].reshape(lot * s * s, -1, 128) for t in (q, k, v)]
    reg = regions.repeat(lot, 1)
    o = attention(*pris, reg, plan, arrondi)
    y = o @ r(c.fusion).t()
    moy = y.mean(-1, keepdim=True)
    var = ((y - moy) ** 2).mean(-1, keepdim=True)
    y = (y - moy) * torch.rsqrt(var + plan.epsilon) * c.gamma + c.beta
    sortie = torch.empty_like(src)
    sortie[:, indices.reshape(-1)] = y.reshape(lot, -1, 128)
    return r(sortie if c.cross else sortie + src)


def couche_noeuds(c: Couche, plan: Plan, src: torch.Tensor, cles: torch.Tensor, h: int, w: int,
                  s: int) -> torch.Tensor:
    """Le graphe FP16 nœud par nœud : chaque sortie de nœud arrondie en FP16 (softmax en f32 sur des scores FP16)."""
    lot = src.shape[0]
    kv = torch.roll(cles, -(lot // 2), 0) if c.cross else cles
    poids = f16(c.qkv)
    q, k, v = (f16(t @ poids[i * 128 : (i + 1) * 128].t()) for i, t in enumerate((src, kv, kv)))
    indices, regions = fenetres(h, w, s, c.decale)
    q, k, v = (t[:, indices.reshape(-1)].reshape(lot * s * s, -1, 128) for t in (q, k, v))
    reg = regions.repeat(lot, 1)
    scores = f16(f16(q @ k.transpose(1, 2)) / f16(torch.tensor(128.0**0.5)))
    if c.decale:
        scores = f16(scores + (reg[:, :, None] != reg[:, None, :]).float() * f16(torch.tensor(plan.masque)))
    o = f16(f16(torch.softmax(scores, -1)) @ v)
    y = f16(o @ f16(c.fusion).t())
    y = f16(torch.nn.functional.layer_norm(y, (128,), c.gamma, c.beta, plan.epsilon))
    sortie = torch.empty_like(src)
    sortie[:, indices.reshape(-1)] = y.reshape(lot, -1, 128)
    return sortie if c.cross else f16(sortie + src)


def noms(graphe: onnx.ModelProto, echelle: int, bloc: int) -> dict:
    """Les tenseurs du bloc à l'échelle : entrée x, sortie de la self, message de la cross."""
    sfx = "" if echelle == 0 else "_1"
    lecteurs: dict[str, list] = {}
    for n in graphe.graph.node:
        for i in n.input:
            lecteurs.setdefault(i, []).append(n)
    base = f"/flownet/transformer/layers.{bloc}/"
    ln_self = f"{base}self_attn/norm1{sfx}/LayerNormalization_output_0"
    ln_cross = f"{base}cross_attn_ffn/norm1{sfx}/LayerNormalization_output_0"
    add = next(n for n in lecteurs[ln_self] if n.op_type == "Add")
    return {"x": add.input[0], "s": add.output[0], "m": ln_cross}


def images(instant: float) -> np.ndarray:
    """frames [1, 6, 768, 1280] : deux images de la vidéo à 2/24 s d'écart, complétées en recopiant le bord."""
    hu = LARGEUR * 9 // 16
    out = []
    for t in (instant, instant + 2 / 24):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(VIDEO), "-frames:v", "1", "-vf",
                              f"scale={LARGEUR}:{hu}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(hu, LARGEUR, 3).astype(np.float32).transpose(2, 0, 1) / 255)
    x = np.concatenate(out, 0)[None]
    return np.pad(x, ((0, 0), (0, 0), (0, HAUTEUR - hu), (0, 0)), mode="edge")


def ecart(y: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (y - ref).double()
    eqm = float((d * d).mean())
    crete = float(ref.abs().max())
    return {"max": float(d.abs().max()), "moy": float(d.abs().mean()),
            "rel": math.sqrt(eqm) / max(float(ref.double().pow(2).mean().sqrt()), 1e-30),
            "psnr": 10 * math.log10(crete**2 / eqm) if eqm > 0 else float("inf")}


def ecrit(dossier: Path, nom: str, t: torch.Tensor, dtype) -> None:
    t.numpy().astype(dtype).tofile(dossier / f"{nom}.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, required=True, help="la paire union exportée en FP32 pur")
    parser.add_argument("--plan32", required=True, help="les couches vidées de la paire FP32 (préfixe)")
    parser.add_argument("--plan16", required=True, help="les couches vidées de la paire FP16 du lecteur (préfixe)")
    parser.add_argument("--paires", type=int, default=4, help="paires de calibration.mkv (6 au plus)")
    parser.add_argument("--dump", type=Path, help="dossier des entrées et sorties pour le banc (première paire)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    plan32, plan16 = lit_plan(args.plan32), lit_plan(args.plan16)
    graphe = onnx.load(str(args.fp32))
    tenseurs = {(e, b): noms(graphe, e, b) for e in (0, 1) for b in BLOCS}
    sorties = sorted({v for d in tenseurs.values() for v in d.values()})
    graphe.graph.output.extend(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None) for n in sorties)
    sess = ort.InferenceSession(graphe.SerializeToString(), providers=["CPUExecutionProvider"])
    print(f"Attention de GMFlow, paire {LARGEUR}x{HAUTEUR} (GMFlow sur 640x384) ; écarts à ONNX Runtime FP32")
    print("| paire | échelle | couche | recette | max | moyen | relatif | PSNR (dB) |")
    cumul: dict[tuple, list] = {}
    for numero, instant in enumerate(INSTANTS[: args.paires]):
        valeurs = dict(zip(sorties, sess.run(sorties, {"frames": images(instant)})))
        for (e, b), d in tenseurs.items():
            h, w = (48, 80) if e == 0 else (96, 160)
            s = plan16.splits[e]
            for cross in (False, True):
                i = 2 * b + int(cross)
                x, src = (torch.from_numpy(valeurs[d[k]]) for k in ("x", "s" if cross else "x"))
                ref = torch.from_numpy(valeurs[d["m" if cross else "s"]])
                lignes = [("contrôle f32", couche_native(plan32.couches[i], plan32, src, x, h, w, s, False)),
                          ("nœud par nœud FP16", couche_noeuds(plan16.couches[i], plan16, f16(src), f16(x), h, w, s)),
                          ("natif FP16", couche_native(plan16.couches[i], plan16, f16(src), f16(x), h, w, s, True))]
                genre = f"{i} {'cross' if cross else 'self'}{' décalée' if plan16.couches[i].decale else ''}"
                for nom, y in lignes:
                    r = ecart(y, ref)
                    cumul.setdefault((e, genre, nom), []).append(r)
                    print(f"| {instant:g} s | 1/{8 if e == 0 else 4} | {genre} | {nom} | {r['max']:.3e} | "
                          f"{r['moy']:.3e} | {r['rel']:.3e} | {r['psnr']:.1f} |", flush=True)
                if args.dump is not None and numero == 0:
                    args.dump.mkdir(parents=True, exist_ok=True)
                    stem = f"e{e}_c{i}"
                    ecrit(args.dump, stem + "_query", f16(src), np.float16)
                    ecrit(args.dump, stem + "_keys", f16(x), np.float16)
                    ecrit(args.dump, stem + "_natif", lignes[2][1], np.float16)
                    ecrit(args.dump, stem + "_ref", ref, np.float32)
    print("\nSynthèse (pire max, moyenne des écarts moyens et relatifs, PSNR moyen / pire) :")
    print("| échelle | couche | recette | max | moyen | relatif | PSNR moyen / pire |")
    for (e, genre, nom), rows in cumul.items():
        ps = [r["psnr"] for r in rows]
        print(f"| 1/{8 if e == 0 else 4} | {genre} | {nom} | {max(r['max'] for r in rows):.3e} | "
              f"{sum(r['moy'] for r in rows) / len(rows):.3e} | {sum(r['rel'] for r in rows) / len(rows):.3e} | "
              f"{sum(ps) / len(ps):.1f} / {min(ps):.1f} |")


if __name__ == "__main__":
    main()
