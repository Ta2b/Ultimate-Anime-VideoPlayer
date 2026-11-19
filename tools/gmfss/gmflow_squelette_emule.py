#!/usr/bin/env python3
"""Le squelette CNN de GMFlow natif rejoué tel que le C++ le lit, contre ONNX Runtime (docs/gmflow-natif.md, PR 1).

Le plan (tenseurs, opérations, statistiques) et les poids viennent de gmflow_squelette_vide (tools/gmfss/
gmflow_squelette), qui appelle la lecture du lecteur (prepare/gmflow_native.h) sur la paire ONNX. Ce script
exécute ce plan à la lettre avec les arrondis des noyaux de gmflow_backbone_kernels.cuh :
  - l'image FP16 telle quelle, sa normalisation rangée par la lecture dans les poids et le biais de la 7 × 7, les
    pixels hors de l'image à la moyenne de leur plan (pad), puis la 7 × 7 dépliée dans l'ordre
    k = (dy · 7 + dx) · 3 + plan des poids lus ([cout][49][3]) ;
  - poids et entrées en FP16, sommes en f32, biais ajouté en f32, sortie arrondie en FP16 (bornée à ±65504) ;
  - InstanceNorm : Σx et Σx² des valeurs FP16 écrites, moyenne = Σx/N, variance = Σx²/N - moyenne², puis
    (x - moyenne) · 1/√(variance + ε) en f32 à la lecture, Relu, arrondi FP16 (convolution) ou non (fin de bloc,
    qui arrondit une seule fois relu(raccourci + bloc)).
Le contrôle f32 ne fait aucun arrondi : sur le plan lu dans la paire FP32 (avec au bord la moyenne exacte du
graphe, voir pad_exact), il doit rendre ONNX Runtime FP32 à l'ordre des sommes près (indexation : poids rangés,
prises, ordre de la 7 × 7, normalisation de l'image rangée dans ses poids, lectures, statistiques, raccourcis,
trident aux poids partagés).

Les références : le sous-graphe du squelette (de half0, half1 aux deux sorties du trident) de la paire FP32 (la
vérité) et de la paire FP16 du lecteur exécutée par ONNX Runtime sur le CPU (« actuel » ; ce n'est pas TensorRT :
faute de noyaux FP16 sur x86, ONNX Runtime calcule en f32 entre des Cast qu'il retire, c'est-à-dire des poids FP16
et des activations f32, plus juste que TensorRT FP16 qui écrit chaque couche en FP16), sur de vraies paires
d'images d'anime (data/calibration/calibration.mkv, deux images à 2/24 s d'écart).

    cmake -S tools/gmfss/gmflow_squelette -B build/gmflow_squelette && cmake --build build/gmflow_squelette
    build/gmflow_squelette/gmflow_squelette_vide paire_fp32.onnx plan32
    build/gmflow_squelette/gmflow_squelette_vide gmfss_fortuna_union_pair.onnx plan16
    python -I tools/gmfss/gmflow_squelette_emule.py --fp32 paire_fp32.onnx --fp16 gmfss_fortuna_union_pair.onnx \\
        --plan32 plan32 --plan16 plan16 [--width 640 --height 384]

paire_fp32.onnx : la paire union exportée en FP32 pur (gmflow_natif_emule.py --work DOSSIER l'écrit).
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort
import torch
import torch.nn.functional as F

DEPOT = Path(__file__).resolve().parents[2]
VIDEO = DEPOT / "data/calibration/calibration.mkv"
INSTANTS = (15.0, 39.0, 69.0, 99.0)
QUART = "/flownet/backbone/trident_conv/Conv_output_0"
HUITIEME = "/flownet/backbone/trident_conv/Conv_1_output_0"
FP16_MAX = 65504.0


@dataclass
class Plan:
    image: int = 0
    quarter: int = -1
    eighth: int = -1
    stats: int = 0
    epsilon: float = 1e-5
    pad: list = field(default_factory=list)
    tensors: dict = field(default_factory=dict)  # id -> (niveau, canaux)
    convs: dict = field(default_factory=dict)  # id -> (noyau, cin, cout, poids [cout][k²][cin], biais)
    ops: list = field(default_factory=list)


def lit_plan(prefixe: str) -> Plan:
    """Le plan vidé par gmflow_squelette_vide : PREFIXE.txt et les poids float32 de PREFIXE.bin."""
    poids = np.fromfile(prefixe + ".bin", dtype="<f4")
    p = Plan()
    for ligne in Path(prefixe + ".txt").read_text().splitlines():
        m = ligne.split()
        if m[0] in ("image", "quarter", "eighth", "stats"):
            setattr(p, m[0], int(m[1]))
        elif m[0] == "epsilon":
            p.epsilon = float(m[1])
        elif m[0] == "pad":
            p.pad = [float(v) for v in m[1:]]
        elif m[0] == "tensor":
            p.tensors[int(m[1])] = (int(m[2]), int(m[3]))
        elif m[0] == "conv":
            k, cin, cout, a, b = (int(v) for v in m[2:])
            w = poids[a:b].reshape(cout, k * k, cin)
            p.convs[int(m[1])] = (k, cin, cout, torch.from_numpy(w.copy()), torch.from_numpy(poids[b : b + cout].copy()))
        elif m[0] == "op":
            p.ops.append((m[1], [int(v) for v in m[2:]]))
    return p


def f16(x: torch.Tensor) -> torch.Tensor:
    """Une valeur f32 telle qu'un noyau l'écrit en FP16 (bornée au plus grand fini, comme to_half2)."""
    return x.clamp(-FP16_MAX, FP16_MAX).half().float()


class Emulation:
    """Le plan exécuté opération par opération ; arrondi : les arrondis FP16 des noyaux, sinon tout en f32."""

    def __init__(self, plan: Plan, arrondi: bool):
        self.p = plan
        self.r = f16 if arrondi else (lambda x: x)
        self.t: dict[int, torch.Tensor] = {}
        self.s: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def norme(self, stats: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Moyenne et 1/√(variance + ε) par image et canal, en f32 comme norm_of."""
        somme, carres = self.s[stats]
        inv = torch.tensor(1.0 / n, dtype=torch.float32)
        m = somme * inv
        var = (carres * inv - m * m).clamp_min(0)
        return m.view(*m.shape, 1, 1), torch.rsqrt(var + self.p.epsilon).view(*m.shape, 1, 1)

    def lit(self, tenseur: int, stats: int, relu: int) -> torch.Tensor:
        x = self.t[tenseur]
        if stats >= 0:
            m, r = self.norme(stats, x.shape[2] * x.shape[3])
            x = (x - m) * r
        return x.clamp_min(0) if relu else x

    def ecrit(self, out: int, y: torch.Tensor, stats: int):
        self.t[out] = y
        if stats >= 0:
            yd = y.double()
            self.s[stats] = (yd.sum((2, 3)).float(), (yd * yd).sum((2, 3)).float())

    def tige(self, conv: tuple, img: torch.Tensor) -> torch.Tensor:
        """La 7 × 7 de pas 2 sur l'image telle quelle (la normalisation est dans ses poids et son biais), les
        pixels hors de l'image à pad ; dépliage k = (dy · 7 + dx) · 3 + plan, l'ordre des poids lus."""
        k, _, cout, w, b = conv
        n, _, h, wd = img.shape
        x = torch.stack([F.pad(img[:, c], (k // 2,) * 4, value=self.p.pad[c]) for c in range(3)], 1)
        cols = F.unfold(x, (k, k), stride=2)  # [N, 3·49, L], plan d'abord
        cols = cols.view(n, 3, k * k, -1).permute(0, 3, 2, 1).reshape(n, -1, k * k * 3)  # k = prise · 3 + plan
        y = cols.double() @ self.r(w.reshape(cout, k * k * 3)).double().t()
        y = (y.float() + b).permute(0, 2, 1).reshape(n, cout, h // 2, wd // 2)
        return self.r(y)

    def conv(self, conv: tuple, x: torch.Tensor, stride: int) -> torch.Tensor:
        k, cin, cout, w, b = conv
        poids = self.r(w).view(cout, k, k, cin).permute(0, 3, 1, 2)  # prise = ky · k + kx
        y = F.conv2d(self.r(x), poids.contiguous(), None, stride, k // 2)
        return self.r(y + b.view(1, -1, 1, 1))

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.t = {self.p.image: images}
        for genre, a in self.p.ops:
            if genre == "conv":
                c, stride, tin, sin, rin, out, stats = a
                conv = self.p.convs[c]
                if conv[0] == 7:
                    y = self.tige(conv, self.t[tin])
                else:
                    y = self.conv(conv, self.lit(tin, sin, rin), stride)
                self.ecrit(out, y, stats)
            else:
                tin, sin, rin, tsk, ssk, rsk, out = a
                y = (self.lit(tin, sin, rin) + self.lit(tsk, ssk, rsk)).clamp_min(0)
                self.ecrit(out, self.r(y), -1)
        return self.t[self.p.quarter], self.t[self.p.eighth]


def pad_exact(plan: Plan, paire: Path) -> Plan:
    """Le plan avec, au bord, la moyenne exacte du graphe (/flownet/Sub) au lieu de son arrondi FP16, et le biais
    de la 7 × 7 refait pour elle : ce qui reste d'écart au graphe FP32 n'est plus que l'ordre des sommes. Sur la
    paire FP16 du lecteur, la moyenne est déjà en FP16 et rien ne change."""
    graphe = onnx.load(str(paire))
    noeud = next(n for n in graphe.graph.node if n.name == "/flownet/Sub")
    valeurs = {n.output[0]: onnx.numpy_helper.to_array(n.attribute[0].t) for n in graphe.graph.node
               if n.op_type == "Constant" and n.attribute}
    valeurs.update({t.name: onnx.numpy_helper.to_array(t) for t in graphe.graph.initializer})
    moyenne = torch.from_numpy(valeurs[noeud.input[1]].astype(np.float64).reshape(3))
    k, cin, cout, w, b = plan.convs[0]
    b = b.double() - (w.double() * (moyenne - torch.tensor(plan.pad, dtype=torch.float64))).sum((1, 2))
    exact = Plan(**{**plan.__dict__, "convs": {**plan.convs, 0: (k, cin, cout, w, b.float())}})
    exact.pad = moyenne.tolist()
    return exact


def sous_graphe(chemin: Path, dossier: Path) -> Path:
    """Le squelette seul (de half0, half1 aux deux sorties du trident), extrait de la paire."""
    sortie = dossier / (chemin.stem + "_squelette.onnx")
    if not sortie.exists():
        onnx.utils.extract_model(str(chemin), str(sortie), ["half0", "half1"], [QUART, HUITIEME])
    return sortie


def images(instant: float, w: int, h: int) -> torch.Tensor:
    """Deux images de la vidéo à 2/24 s d'écart, w × (hauteur 16:9) complétée en recopiant le bord, [2, 3, h, w]."""
    hu = w * 9 // 16
    out = []
    for t in (instant, instant + 2 / 24):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(VIDEO), "-frames:v", "1", "-vf",
                              f"scale={w}:{hu}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(hu, w, 3).astype(np.float32) / 255
        out.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0))
    x = F.pad(torch.cat(out, 0), (0, 0, 0, h - hu), mode="replicate")
    return x.half().float()  # le lecteur donne des plans FP16


def onnx_squelette(chemin: Path, x: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    sess = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    feeds = {"half0": x[:1].numpy().astype(dtype), "half1": x[1:].numpy().astype(dtype)}
    q, e = sess.run(None, feeds)
    return torch.from_numpy(q.astype(np.float32)), torch.from_numpy(e.astype(np.float32))


def ecart(y: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (y - ref).double()
    eqm = float((d * d).mean())
    crete = float(ref.abs().max())
    return {"max": float(d.abs().max()), "moy": float(d.abs().mean()),
            "rel": math.sqrt(eqm) / max(float(ref.double().pow(2).mean().sqrt()), 1e-30),
            "psnr": 10 * math.log10(crete**2 / eqm) if eqm > 0 else float("inf")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, required=True, help="la paire union exportée en FP32 pur")
    parser.add_argument("--fp16", type=Path, required=True, help="la paire FP16 du lecteur")
    parser.add_argument("--plan32", required=True, help="le plan vidé de la paire FP32 (préfixe)")
    parser.add_argument("--plan16", required=True, help="le plan vidé de la paire FP16 (préfixe)")
    parser.add_argument("--width", type=int, default=640, help="largeur des images de GMFlow (multiple de 32)")
    parser.add_argument("--height", type=int, default=384, help="hauteur des images de GMFlow (multiple de 32)")
    parser.add_argument("--work", type=Path, default=Path("."), help="dossier des sous-graphes extraits")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    args.work.mkdir(parents=True, exist_ok=True)
    g32, g16 = sous_graphe(args.fp32, args.work), sous_graphe(args.fp16, args.work)
    plan32, plan16 = lit_plan(args.plan32), lit_plan(args.plan16)
    controle, natif = Emulation(pad_exact(plan32, args.fp32), False), Emulation(plan16, True)
    print(f"Squelette de GMFlow, images {args.width}x{args.height} ; écarts au sous-graphe FP32 (ONNX Runtime)")
    print("| paire | sortie | recette | max | moyen | relatif | PSNR (dB) |")
    cumul: dict[tuple, list] = {}
    for instant in INSTANTS:
        x = images(instant, args.width, args.height)
        verite = onnx_squelette(g32, x, np.float32)
        actuel = onnx_squelette(g16, x, np.float16)
        lignes = [("contrôle f32 (plan FP32)", controle(x)), ("actuel (ORT FP16)", actuel), ("natif FP16", natif(x))]
        for nom, sorties in lignes:
            for echelle, y, ref in zip(("1/4", "1/8"), sorties, verite):
                e = ecart(y, ref)
                cumul.setdefault((nom, echelle), []).append(e)
                print(f"| {instant:g} s | {echelle} | {nom} | {e['max']:.3e} | {e['moy']:.3e} | {e['rel']:.3e} | "
                      f"{e['psnr']:.1f} |", flush=True)
    print("\nSynthèse (pire max, moyenne des écarts moyens et relatifs, PSNR moyen / pire) :")
    print("| sortie | recette | max | moyen | relatif | PSNR moyen / pire |")
    for (nom, echelle), rows in cumul.items():
        ps = [r["psnr"] for r in rows]
        print(f"| {echelle} | {nom} | {max(r['max'] for r in rows):.3e} | "
              f"{sum(r['moy'] for r in rows) / len(rows):.3e} | {sum(r['rel'] for r in rows) / len(rows):.3e} | "
              f"{sum(ps) / len(ps):.1f} / {min(ps):.1f} |")


if __name__ == "__main__":
    main()
