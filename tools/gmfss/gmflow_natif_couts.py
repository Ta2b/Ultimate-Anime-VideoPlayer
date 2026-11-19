#!/usr/bin/env python3
"""Coûts de GMFlow (graphe « paire » de GMFSS) opération par opération, et temps estimés du natif sur une
RTX 4070 Ti (docs/gmflow-natif.md).

    python3 tools/gmfss/gmflow_natif_couts.py [--width 1280 --height 768] [--scale 1|0.5] [--markdown]

La taille est celle de la paire (plans remplis : 1280x768 pour 720p de travail) ; GMFlow tourne sur les images
réduites de moitié (× scale au demi-flot). FLOP : 2 par multiplication-addition, sans le rembourrage des canaux.
Octets « natif » : ce que lit et écrit le noyau fondu prévu (FP16, 2 octets ; FP8, 1 octet à l'entrée des GEMM
FP8). Octets « graphe » : ce que lit et écrit le graphe ONNX non fondu (matrice des scores, couche cachée du MLP,
fenêtrage et décalages en copies), l'ordre de grandeur de ce que TensorRT doit éviter. Le temps estimé d'un
noyau : max(calcul / (pic × rendement), octets / (bande × 0,75)) + 4 µs de lancement et de queue.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# RTX 4070 Ti (AD104, 60 SM, 2,61 GHz) : tensor cores, TFLOPS denses.
PIC = {"fp16-f32": 80.0, "fp16-f16": 160.0, "fp8-f32": 160.0, "fp8-f16": 320.0}
BANDE = 504.0  # Go/s
# Rendements supposés : convolutions de la famille de rife_native_kernels.cuh, GEMM à la CUTLASS (SPAN FP8 : 76 à
# 83 % du pic en f32), attention fondue (FlashAttention-2 tient 50 à 70 % sur Ampere ; une tête de 128, des
# fenêtres courtes : prudence).
RENDEMENT = {"conv": 0.50, "gemm": 0.70, "attn": 0.40, "mem": 1.0}
LANCEMENT_US = 4.0


@dataclass
class Op:
    bloc: str
    nom: str
    flop: float
    natif: float  # octets
    graphe: float  # octets
    sorte: str  # conv, gemm, attn, mem
    fois: int = 1


def conv(bloc, nom, n, cin, cout, oh, ow, k, ih=None, iw=None, fois=1):
    ih, iw = ih or oh, iw or ow
    flop = 2.0 * n * oh * ow * cout * cin * k * k
    octets = 2.0 * n * (ih * iw * cin + oh * ow * cout)
    return Op(bloc, nom, flop, octets, octets, "conv", fois)


def squelette(w, h):
    """CNNEncoder sur les deux images (lot 2) ; une InstanceNorm ajoute la relecture d'une sortie (statistiques à
    l'épilogue en natif ; passes séparées dans le graphe)."""
    n, ops = 2, []
    h2, w2, h4, w4, h8, w8 = h // 2, w // 2, h // 4, w // 4, h // 8, w // 8
    ops.append(conv("squelette", "conv1 7x7/2 3->64", n, 3, 64, h2, w2, 7, h, w))
    for i in range(2):
        ops.append(conv("squelette", f"layer1.{i} 2 x conv 3x3 64->64", n, 64, 64, h2, w2, 3, fois=2))
    ops.append(conv("squelette", "layer2.0 conv1 3x3/2 64->96", n, 64, 96, h4, w4, 3, h2, w2))
    ops.append(conv("squelette", "layer2.0 ds 1x1/2 64->96", n, 64, 96, h4, w4, 1, h2, w2))
    ops.append(conv("squelette", "layer2 3 x conv 3x3 96->96", n, 96, 96, h4, w4, 3, fois=3))
    ops.append(conv("squelette", "layer3.0 conv1 3x3 96->128", n, 96, 128, h4, w4, 3))
    ops.append(conv("squelette", "layer3.0 ds 1x1 96->128", n, 96, 128, h4, w4, 1))
    ops.append(conv("squelette", "layer3 3 x conv 3x3 128->128", n, 128, 128, h4, w4, 3, fois=3))
    ops.append(conv("squelette", "conv2 1x1 128->128", n, 128, 128, h4, w4, 1))
    ops.append(conv("squelette", "trident 3x3 /1", n, 128, 128, h4, w4, 3))
    ops.append(conv("squelette", "trident 3x3 /2", n, 128, 128, h8, w8, 3, h4, w4))
    # Fins de bloc (normes, somme, Relu) : 6 blocs, 3 tenseurs lus et 1 écrit ; normes du graphe : 2 passes.
    fin = 2.0 * n * (h2 * w2 * 64 * 2 + h4 * w4 * 96 * 2 + h4 * w4 * 128 * 2) * 4
    ops.append(Op("squelette", "fins de bloc et InstanceNorm", 0, fin, fin * 2, "mem"))
    return ops


def transformeur(bloc, jetons, fenetre, fois=6):
    """Un TransformerBlock : self (QKV, attention, fusion + LayerNorm + résiduel), cross (même chose + MLP 256 ->
    1024 -> 128 fondu, LayerNorm, résiduel). jetons : tous les jetons du lot (2 ou 4 cartes)."""
    t = 2.0 * jetons * 128  # un tenseur de jetons en FP16
    ops = []
    for couche in ("self", "cross"):
        ops.append(Op(bloc, f"{couche} : QKV (GEMM 128->384)", 2.0 * jetons * 128 * 384, 4 * t, 4 * t, "gemm", fois))
        # Attention : S = QKᵀ et P·V ; natif : Q, K, V lus, sortie fusionnée (128x128 + LayerNorm) écrite.
        flop = 2 * 2.0 * jetons * fenetre * 128 + 2.0 * jetons * 128 * 128
        scores = 2.0 * jetons * fenetre  # une matrice de scores FP16
        graphe = 3 * t + 4 * scores + 2 * t + 6 * t  # scores écrits, softmax lu et écrit, relus ; fenêtres, roll
        ops.append(Op(bloc, f"{couche} : attention fenêtrée + fusion + LN", flop, 4 * t + (t if couche == "self" else 0),
                      graphe, "attn", fois))
    hidden = 2.0 * jetons * 1024
    ops.append(Op(bloc, "cross : MLP fondu 256->1024->128 + LN + résiduel", 2.0 * jetons * (256 * 1024 + 1024 * 128),
                  3 * t, 4 * t + 2 * hidden + 2 * t, "gemm", fois))
    return ops


def globaux(l, c=128):
    """Appariement global bidirectionnel et propagation globale à 1/8 : softmax fondus sur des tuiles de clés,
    S recalculé pour le second sens ; v = coordonnées ou flot (2 canaux) en f32 sur les cœurs CUDA."""
    t = 2.0 * l * c
    corr = 2 * (2.0 * l * l * c)  # deux passes (lignes, puis colonnes)
    prop = 2 * (2.0 * l * l * c)  # deux cartes, k = k_proj(q_proj(f))
    graphe_corr = 4.0 * l * l * 2 * 3  # corrélation FP16 : écrite, transposée, softmax lu et écrit
    return [
        Op("1/8", "appariement global (2 sens)", corr, 2 * t * 2, 2 * t + graphe_corr, "attn"),
        Op("1/8", "propagation : q, k (GEMM 128->256)", 2.0 * 2 * l * c * 256, 2 * t + 4 * t, 6 * t, "gemm"),
        Op("1/8", "propagation globale (2 cartes)", prop, 2 * 2 * t, 4 * t + graphe_corr, "attn"),
    ]


def fins(w4, h4):
    l, t = w4 * h4, 2.0 * w4 * h4 * 128
    up = conv("1/4", "upsampler conv 3x3 130->256", 2, 130, 256, h4, w4, 3)
    up.natif = 2.0 * 2 * l * (144 + 256)
    up1 = conv("1/4", "upsampler conv 1x1 256->144 + softmax 9 + somme convexe", 2, 256, 144, h4, w4, 1)
    up1.natif = 2.0 * 2 * l * 256 + 4.0 * 2 * 16 * l * 2
    up1.graphe = up1.natif + 2.0 * 2 * l * 144 * 4 + 4.0 * 2 * l * 9 * 2 * 3
    return [
        Op("1/4", "déformation de f1 par le flot ×2", 0, 2 * 2 * t, 2 * 2 * t, "mem"),
        Op("1/4", "appariement local r = 4 (81 voisins)", 2.0 * 2 * l * 81 * 128, 2 * 2 * t, 2 * 2 * t + 4.0 * 2 * l * 81 * 3,
           "attn"),
        Op("1/4", "propagation locale 3x3 (q, k + fenêtre)", 2.0 * 2 * l * 128 * 256 + 2.0 * 2 * l * 9 * 128,
           2 * t + 4 * t, 2 * t + 4 * t + 2.0 * 2 * l * 128 * 9 * 2, "gemm"),
        up,
        up1,
    ]


def operations(w, h):
    """GMFlow sur une image de w × h (la demi-image de la paire, × scale)."""
    w8, h8, w4, h4 = w // 8, h // 8, w // 4, h // 4
    ops = squelette(w, h)
    ops += transformeur("1/8 transformeur (6 blocs)", 2 * w8 * h8, (w8 // 2) * (h8 // 2))
    ops += globaux(w8 * h8)
    ops += transformeur("1/4 transformeur (6 blocs)", 4 * w4 * h4, (w4 // 8) * (h4 // 8))
    ops += fins(w4, h4)
    return ops


def precision(op, recette):
    """Le pic du tensor core de l'opération selon la recette du natif."""
    # L'attention reste en FP16 à sommes f32 (S et le softmax) dans les trois recettes.
    if op.sorte == "attn":
        return PIC["fp16-f32"]
    if recette == "fp16-acc16":
        return PIC["fp16-f16"]
    if recette == "fp8-gemm" and op.sorte == "gemm" and "transformeur" in op.bloc:
        return PIC["fp8-f32"]
    return PIC["fp16-f32"]


def temps_us(op, recette):
    octets = op.natif * (0.75 if recette.startswith("fp8") and op.sorte == "gemm" and "transformeur" in op.bloc else 1)
    calcul = op.flop / (precision(op, recette) * 1e12 * RENDEMENT[op.sorte]) * 1e6 if op.flop else 0.0
    memoire = octets / (BANDE * 1e9 * 0.75) * 1e6
    return op.fois * (max(calcul, memoire) + LANCEMENT_US)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1280, help="largeur de la paire (plans remplis)")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0)
    args = parser.parse_args()
    w, h = int(args.width // 2 * args.scale), int(args.height // 2 * args.scale)
    ops = operations(w, h)
    recettes = ("fp16", "fp16-acc16", "fp8-gemm")
    print(f"GMFlow sur {w}x{h} (paire {args.width}x{args.height}, échelle {args.scale})\n")
    print("| Bloc | Opération | × | GFLOP | Mo natif | Mo graphe | µs fp16 | µs acc16 | µs fp8 |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = {"flop": 0.0, "natif": 0.0, "graphe": 0.0, **{r: 0.0 for r in recettes}}
    par_bloc: dict[str, list[float]] = {}
    for op in ops:
        t = [temps_us(op, r) for r in recettes]
        print(f"| {op.bloc} | {op.nom} | {op.fois} | {op.flop * op.fois / 1e9:.2f} | {op.natif * op.fois / 1e6:.1f} | "
              f"{op.graphe * op.fois / 1e6:.1f} | " + " | ".join(f"{x:.0f}" for x in t) + " |")
        tot["flop"] += op.flop * op.fois
        tot["natif"] += op.natif * op.fois
        tot["graphe"] += op.graphe * op.fois
        b = par_bloc.setdefault(op.bloc, [0.0, 0.0] + [0.0] * len(recettes))
        b[0] += op.flop * op.fois
        b[1] += op.natif * op.fois
        for i, r in enumerate(recettes):
            tot[r] += t[i]
            b[2 + i] += t[i]
    print(f"| **total** | | | **{tot['flop'] / 1e9:.1f}** | **{tot['natif'] / 1e6:.0f}** | **{tot['graphe'] / 1e6:.0f}** | "
          + " | ".join(f"**{tot[r] / 1e3:.2f} ms**" for r in recettes) + " |")
    print("\n| Bloc | GFLOP | Mo natif | ms fp16 | ms acc16 | ms fp8 |\n|---|---|---|---|---|---|")
    for bloc, v in par_bloc.items():
        print(f"| {bloc} | {v[0] / 1e9:.1f} | {v[1] / 1e6:.0f} | " + " | ".join(f"{x / 1e3:.2f}" for x in v[2:]) + " |")


if __name__ == "__main__":
    main()
