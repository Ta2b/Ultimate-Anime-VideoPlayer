#!/usr/bin/env python3
"""Ce que Winograd et le FP8 ôteraient au plafond du GridNet de la fusion (docs/gmfss-natif.md, « Winograd et FP8
du GridNet, émulés »), au modèle de plafond_gmfss.py.

Compte chaque convolution du GridNet (transposées comprises, que plafond_gmfss.py laisse de côté) aux formes d'une
lecture en 1080p, la range dans son morceau (lignes 0, 1, 2 et queue, comme emul_gridnet_leviers.py) et donne
pour chaque morceau et chaque variante deux temps :
- le plafond : le plus lent du calcul et des octets, couche par couche ;
- « à la suite » : calcul plus octets, ce que fait à peu près un noyau qui ne recouvre pas son chargement
  (conv_kernel aujourd'hui, docs/gmfss-bilan.md, partie 3.1).

Variantes, chacune sur les seules couches qu'elle touche :
- direct f16 : 160 Tflop/s ;
- wino23, wino43 : sur les 3 × 3 de pas 1 sauf conv_last, les mma ne font plus que (m + 2)² / (9 m²) des
  multiplications (0,444 et 0,25), en sommes f16 ; les transformées passent sur les cœurs f32 (20 T instructions
  par seconde : 60 SM × 128 voies × 2,61 GHz), une instruction par coefficient non nul de Bᵀ et Aᵀ, entrée par tuile
  et par canal d'entrée, sortie par tuile et par canal de sortie. Le calcul vaut mma + transformées (sans
  recouvrement : le temps le plus prudent) ;
- fp8 : 321 Tflop/s (E4M3 à sommes f16 ; en sommes f32 la GeForce n'en fait que la moitié, 160 : aucun gain),
  les octets restent ceux du f16 (l'entrée est quantifiée au chargement) ; « fp8, octets FP8 » : en plus,
  chaque tenseur écrit et relu en E4M3 (les octets des activations divisés par deux).

    python3 -I tools/gmfss/plafond_leviers.py FUSION.onnx --height 1152 --width 1920

    --modele-actuel calcule aussi la borne avec les canaux des MMA, la queue composée sur Ada et F(2×2,3×3).
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
from plafond_gmfss import BANDWIDTH, PEAK, inferred  # noqa: E402

PIECES = ["ligne 0", "ligne 1", "ligne 2", "queue"]
# La ligne du tenseur qu'écrit chaque bloc (emul_mix_native.BLOCKS et LINE) : 0, 1 ou 2.
BLOCK_LINE = {
    "residual_model_head0": 0, "residual_model_head1": 0, "residual_model_01": 0, "residual_model_04": 0,
    "upsample_model_04": 0, "residual_model_05": 0, "upsample_model_05": 0,
    "residual_model_head2": 1, "downsample_model_10": 1, "residual_model_11": 1, "downsample_model_11": 1,
    "residual_model_14": 1, "upsample_model_14": 1, "residual_model_15": 1, "upsample_model_15": 1,
    "residual_model_head3": 2, "downsample_model_20": 2, "residual_model_21": 2, "downsample_model_21": 2,
    "residual_model_24": 2, "residual_model_25": 2,
}
CUDA_RATE = 60 * 128 * 2.61e9  # instructions f32 par seconde
# Coefficients non nuls de Bᵀ et Aᵀ (les matrices de winograd_db.py) : instructions d'une transformée 1D.
WINO = {"wino23": (2, 8, 6), "wino43": (4, 22, 18)}  # m, nnz(Bᵀ 4×4 ou 6×6), nnz(Aᵀ)


def conv_layers(model, full_size=None):
    """(morceau, nom, FLOP, octets d'entrée et de sortie, octets des poids, 3 × 3 de pas 1, tuiles par forme)."""
    values = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in values}
    shapes.update({i.name: list(i.dims) for i in model.graph.initializer})
    rows = []
    for node in model.graph.node:
        parts = node.name.strip("/").split("/")
        if parts[0] != "fusionnet" or node.op_type not in ("Conv", "ConvTranspose"):
            continue
        piece = 3 if parts[1] == "residual_model_tail" else BLOCK_LINE[parts[1]]
        inp, weight, out = shapes[node.input[0]], shapes[node.input[1]], shapes[node.output[0]]
        if full_size is not None and parts[2] == "conv_last":
            # La taille de sortie du PixelShuffle peut rester symbolique après l'inférence ONNX.
            inp, out = [1, weight[1], *full_size], [1, weight[0], *full_size]
        attrs = {a.name: list(a.ints) for a in node.attribute}
        if node.op_type == "Conv":
            flop = 2 * np.prod(out) * weight[1] * np.prod(weight[2:])
        else:  # transposée 4 × 4 de pas 2 : quatre prises par phase
            flop = 2 * np.prod(out) * weight[0] * 4
        acts = 2 * (np.prod(inp) + np.prod(out))
        winograd = (node.op_type == "Conv" and weight[2:] == [3, 3] and attrs.get("strides", [1, 1]) == [1, 1]
                    and "conv_last" not in node.name)
        rows.append((piece, node.name, float(flop), float(acts), float(2 * np.prod(weight)), winograd,
                     (inp[1], out[1], out[2], out[3])))
    return rows


def layer_time(row, variant):
    """(calcul, octets) d'une couche, en secondes, pour une variante qui la touche."""
    _, _, flop, acts, weights, winograd, (cin, cout, h, w) = row
    moved = acts + weights
    if variant == "direct" or (variant.startswith("wino") and not winograd):
        return flop / PEAK["f16"], moved / BANDWIDTH
    if variant.startswith("wino"):
        m, nnz_b, nnz_a = WINO[variant]
        a = m + 2
        tiles = -(-h // m) * -(-w // m)
        mma = flop * a * a / (9 * m * m) / PEAK["f16"]
        cuda = tiles * (cin * 2 * a * nnz_b + cout * (a + m) * nnz_a) / CUDA_RATE
        return mma + cuda, moved / BANDWIDTH
    if variant == "fp8":
        return flop / PEAK["fp8"], moved / BANDWIDTH
    return flop / PEAK["fp8"], (acts / 2 + weights / 2) / BANDWIDTH  # fp8, octets FP8


def native_cost(row, winograd):
    _, name, flop, acts, weights, eligible, (cin, cout, h, w) = row
    pin, pout = ((c + 15) // 16 * 16 for c in (cin, cout))
    padding = pin * pout / (cin * cout)
    packed_weights = weights * padding
    if winograd and eligible:
        tiles = ((h + 1) // 2) * ((w + 1) // 2)
        # Bᵀ : 4 additions par passage 1D. Aᵀ : 4 additions par passage, sur 4 puis 2 points.
        transforms = tiles * (32 * cin + 24 * cout)
        math = tiles * 16 * 2 * pin * pout / PEAK["f16"] + transforms / CUDA_RATE
        return math, (acts + packed_weights * 16 / 9) / BANDWIDTH
    peak = PEAK["f32"] if "/conv_last/" in name else PEAK["f16"]
    return flop * padding / peak, (acts + packed_weights) / BANDWIDTH


def folded_cost(h, w):
    # Deux groupes de huit sorties MMA, chacun couvrant une empreinte de 20 positions × 64 canaux.
    low_pixels = (h // 2) * (w // 2)
    products = low_pixels * 2 * 8 * 20 * 64
    boundary = 2 * (h + w) - 4
    edge_products = boundary * 3 * 16 * 64  # borne de l'épilogue scalaire aux bords
    math = 2 * products / PEAK["f32"] + 2 * edge_products / (2 * CUDA_RATE)
    interior_weights = 2 * 8 * 20 * 64 * 2
    edge_weights = 4 * 4 * 3 * 16 * 64 * 4
    bandwidth = (2 * (low_pixels * 64 + h * w * 3) + interior_weights + edge_weights) / BANDWIDTH
    return math, bandwidth


def report_current(rows, h, w):
    last = [r for r in rows if "/conv_last/" in r[1] or "/upsample/upsample.0/" in r[1]]
    expected = {(64, 3, h, w), (64, 256, h // 2, w // 2)}
    if len(rows) != 45 or len(last) != 2 or {r[-1] for r in last} != expected:
        raise ValueError("formes inattendues pour la queue GridNet composée")
    print(f"GridNet à {h} × {w}, {len(rows)} convolutions, RTX 4070 Ti")
    print("Calcul et durée calcul/octets couche par couche, pour le GridNet seul")
    print(f"  {'voie':35s} {'calcul (ms)':>12s} {'borne (ms)':>12s} {'cible 90 % (ms)':>16s}")
    for label, folded, winograd in (
            ("direct avec canaux MMA", False, False),
            ("queue composée", True, False),
            ("queue composée + Winograd F(2×2)", True, True)):
        costs = [native_cost(r, winograd) for r in rows if not folded or r not in last]
        if folded:
            costs.append(folded_cost(h, w))
        compute = 1000 * sum(c[0] for c in costs)
        bound = 1000 * sum(max(c) for c in costs)
        print(f"  {label:35s} {compute:12.3f} {bound:12.3f} {bound / .9:16.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="la fusion d'export_gmfss.py")
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--modele-actuel", action="store_true", help="avec le calcul de la queue composée Ada")
    args = parser.parse_args()
    if args.modele_actuel:
        if (args.height, args.width) != (1152, 1920):
            parser.error("la queue composée actuelle est qualifiée uniquement à 1152 × 1920")
        model = inferred(args.model, args.height, args.width, symbolic=False)
        report_current(conv_layers(model, (args.height, args.width)), args.height, args.width)
        return
    rows = conv_layers(inferred(args.model, args.height, args.width))
    variants = ["direct", "wino23", "wino43", "fp8", "fp8, octets FP8"]
    total = collections.defaultdict(lambda: np.zeros(2))
    print(f"GridNet à {args.height} × {args.width} : {len(rows)} convolutions, "
          f"{sum(r[2] for r in rows) / 1e9:.0f} Gflop ; ms au plafond / à la suite")
    print(f"  {'morceau':10s}" + "".join(f"{v:>22s}" for v in variants))
    for p, piece in enumerate(PIECES):
        cells = []
        for v in variants:
            ceil = sum(max(*layer_time(r, v)) for r in rows if r[0] == p)
            serial = sum(sum(layer_time(r, v)) for r in rows if r[0] == p)
            total[v] += (ceil, serial)
            cells.append(f"{ceil * 1e3:9.2f} / {serial * 1e3:5.2f}")
        flop = sum(r[2] for r in rows if r[0] == p)
        print(f"  {piece:10s}" + "".join(f"{c:>22s}" for c in cells) + f"   ({flop / 1e9:.0f} Gflop)")
    print(f"  {'GridNet':10s}" + "".join(f"{total[v][0] * 1e3:9.2f} / {total[v][1] * 1e3:5.2f}".rjust(22)
                                         for v in variants))


if __name__ == "__main__":
    main()
