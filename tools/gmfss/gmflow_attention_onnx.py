#!/usr/bin/env python3
"""Une couche d'attention de GMFlow seule en ONNX, pour la chronométrer sous TensorRT (trtexec) face au banc natif
gmflow_attention_banc (docs/gmflow-natif.md, « Attention native (PR 2) »).

La couche vient du plan vidé par gmflow_attention_vide (les poids tels que le lecteur les lit) et s'écrit comme
l'export de la paire l'écrit : projections sans biais, décalage de Swin par torch.roll, fenêtres par Reshape et
Transpose (split_feature, merge_splits), scores / √d, masque des fenêtres décalées répété sur le lot, Softmax, P · V,
fusion, LayerNorm, résiduel en self-attention ; en cross-attention, la couche échange elle-même les moitiés du lot
de keys (Concat de deux Slice, comme la paire). Taille fixe : le masque est une constante, comme une fois plié par
TensorRT. Entrées query (et keys en cross) [lot, h · w, 128], sortie out, en FP16 comme la paire du lecteur ; le
script écrit la commande trtexec de la recette de la paire (Recipe::Fp32Islands, prepare/trtexec_command.cpp).

Avant d'écrire, la couche est vérifiée sous ONNX Runtime (en FP32) contre le contrôle f32 de
gmflow_attention_emule.py, que l'émulation a montré égal à la paire FP32 : la couche exportée calcule ce que la
paire calcule.

    python -I tools/gmfss/gmflow_attention_onnx.py --plan attn32 --couche 2 --echelle 1 --sortie couche2_quart.onnx

Échelle 0 : le huitième, lot 2, 48 × 80 jetons (GMFlow sur 640 × 384) ; échelle 1 : le quart, lot 4, 96 × 160.
--hauteur et --largeur pour une autre taille (multiples des fenêtres par côté).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmflow_attention_emule import couche_native, ecart, lit_plan  # noqa: E402


def fenetres(x: torch.Tensor, s: int) -> torch.Tensor:
    """[lot, h, w, c] → [lot · s², (h/s) · (w/s), c], fenêtre wy · s + wx."""
    b, h, w, c = x.shape
    return x.view(b, s, h // s, s, w // s, c).permute(0, 1, 3, 2, 4, 5).reshape(b * s * s, -1, c)


def rassemble(x: torch.Tensor, b: int, h: int, w: int, s: int) -> torch.Tensor:
    c = x.shape[-1]
    return x.view(b, s, s, h // s, w // s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def masque(h: int, w: int, s: int, valeur: float) -> torch.Tensor:
    """[s², L, L] : valeur entre deux régions différentes de l'image décalée, 0 ailleurs."""
    wh, ww = h // s, w // s
    ry = (torch.arange(h) >= h - wh).long() + (torch.arange(h) >= h - wh // 2).long()
    rx = (torch.arange(w) >= w - ww).long() + (torch.arange(w) >= w - ww // 2).long()
    regions = fenetres((ry.view(h, 1) * 3 + rx.view(1, w)).float().view(1, h, w, 1), s).view(s * s, -1)
    differe = regions.unsqueeze(1) != regions.unsqueeze(2)
    return torch.where(differe, torch.tensor(valeur), torch.tensor(0.0))


class CoucheOnnx(torch.nn.Module):
    def __init__(self, c, plan, h: int, w: int, s: int):
        super().__init__()
        self.cross, self.decale, self.h, self.w, self.s = c.cross, c.decale, h, w, s
        self.qkv = torch.nn.Parameter(c.qkv.clone(), requires_grad=False)
        self.fusion = torch.nn.Parameter(c.fusion.clone(), requires_grad=False)
        self.norm = torch.nn.LayerNorm(128, eps=plan.epsilon)
        self.norm.weight.data.copy_(c.gamma)
        self.norm.bias.data.copy_(c.beta)
        self.racine = math.sqrt(128.0)
        self.register_buffer("masque", masque(h, w, s, plan.masque) if c.decale else torch.zeros(1))

    def forward(self, query: torch.Tensor, keys: torch.Tensor | None = None) -> torch.Tensor:
        b, h, w, s = query.shape[0], self.h, self.w, self.s
        cible = query if keys is None else torch.cat([keys[b // 2 :], keys[: b // 2]], 0)
        q = F.linear(query, self.qkv[:128])
        k = F.linear(cible, self.qkv[128:256])
        v = F.linear(cible, self.qkv[256:])
        q, k, v = (t.view(b, h, w, 128) for t in (q, k, v))
        if self.decale:
            q, k, v = (torch.roll(t, (-(h // s // 2), -(w // s // 2)), (1, 2)) for t in (q, k, v))
        q, k, v = (fenetres(t, s) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(1, 2)) / self.racine
        if self.decale:
            scores = scores + self.masque.repeat(b, 1, 1)
        o = rassemble(torch.matmul(torch.softmax(scores, -1), v), b, h, w, s)
        if self.decale:
            o = torch.roll(o, (h // s // 2, w // s // 2), (1, 2))
        y = self.norm(F.linear(o.reshape(b, h * w, 128), self.fusion))
        return y if self.cross else query + y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, help="les couches vidées par gmflow_attention_vide (préfixe)")
    parser.add_argument("--couche", type=int, required=True, help="0 à 11 : self du bloc b en 2b, cross en 2b + 1")
    parser.add_argument("--echelle", type=int, choices=(0, 1), required=True, help="0 : le huitième, 1 : le quart")
    parser.add_argument("--hauteur", type=int, help="jetons en hauteur (48 au huitième, 96 au quart)")
    parser.add_argument("--largeur", type=int, help="jetons en largeur (80 au huitième, 160 au quart)")
    parser.add_argument("--sortie", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    plan = lit_plan(args.plan)
    c = plan.couches[args.couche]
    s = plan.splits[args.echelle]
    lot = 2 if args.echelle == 0 else 4
    h = args.hauteur or (48 if args.echelle == 0 else 96)
    w = args.largeur or (80 if args.echelle == 0 else 160)
    if h % s or w % s:
        sys.exit(f"{h} x {w} ne se coupe pas en {s} x {s} fenêtres")
    couche = CoucheOnnx(c, plan, h, w, s).eval()
    gen = torch.Generator().manual_seed(1)
    entrees = [torch.randn(lot, h * w, 128, generator=gen) for _ in range(2 if c.cross else 1)]
    noms = ["query", "keys"][: len(entrees)]
    attendu = couche_native(c, plan, entrees[0], entrees[-1], h, w, s, False)
    with tempfile.TemporaryDirectory() as tmp:
        f32 = Path(tmp) / "f32.onnx"
        torch.onnx.export(couche, tuple(entrees), str(f32), input_names=noms, output_names=["out"],
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(f32), providers=["CPUExecutionProvider"])
        y = torch.from_numpy(sess.run(None, {n: e.numpy() for n, e in zip(noms, entrees)})[0])
    r = ecart(y, attendu)
    print(f"couche {args.couche} ({'cross' if c.cross else 'self'}{', décalée' if c.decale else ''}), lot {lot}, "
          f"{h} x {w}, fenêtres {s} x {s} : ONNX Runtime FP32 contre le contrôle f32 : max {r['max']:.2e}, "
          f"PSNR {r['psnr']:.1f} dB")
    if r["max"] > 1e-4:
        sys.exit("la couche exportée ne calcule pas ce que calcule la paire")
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(couche.half(), tuple(e.half() for e in entrees), str(args.sortie), input_names=noms,
                      output_names=["out"], opset_version=17, dynamo=False)
    formats = ",".join(["fp16:chw"] * len(noms))
    print(f"écrit {args.sortie} (FP16) ; sous TensorRT comme la paire du lecteur (Recipe::Fp32Islands) :\n"
          f"  trtexec --onnx={args.sortie} --stronglyTyped --inputIOFormats={formats} --outputIOFormats=fp16:chw "
          f"--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --tilingOptimizationLevel=3 --useCudaGraph --noDataTransfers "
          f"--warmUp=500 --duration=10")


if __name__ == "__main__":
    main()
