"""Estime le coût d'une passe SPAN FP16 entière sur RTX 4070 Ti."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import onnx


PEAK_MMA = 160e12  # FLOP/s, RTX 4070 Ti, sommes FP16, sans sparsité
PEAK_FP32 = 60 * 128 * 2.61e9  # instructions/s pour des additions et multiplications simples
DRAM = 504e9
BLOCK = re.compile(r"block_(\d+)\.(c[123]_r\.eval_conv|att_conv)\.weight")


@dataclass(frozen=True)
class Stage:
    outputs: int
    inputs: int
    taps: int
    skip: int = 0


def model_stages(path):
    model = onnx.load(path)
    initializers = {item.name: tuple(item.dims) for item in model.graph.initializer}
    names = [node.input[1] for node in model.graph.node if node.op_type == "Conv"]
    layers = {}
    blocks = defaultdict(dict)
    for name in names:
        out, inp, kh, kw = initializers[name]
        if kh != kw or kh not in (1, 3):
            raise ValueError(f"taille de filtre inconnue pour {name}")
        layer = Stage(out, inp, kh * kw)
        if match := BLOCK.fullmatch(name):
            index, kind = match.groups()
            blocks[int(index)][kind] = layer
        else:
            layers[name] = layer
    if layers.keys() != {"conv_1.1.eval_conv.weight", "conv_cat.weight", "to_pixel.eval_conv.weight"}:
        raise ValueError("trous dans le corps SPAN")
    c = layers["conv_1.1.eval_conv.weight"].outputs
    expected_depth = {48: 5, 64: 8}.get(c, 0)
    if set(blocks) != set(range(1, expected_depth + 1)) or not expected_depth:
        raise ValueError("nombre de blocs ou de canaux inattendu")
    triplet = {f"c{k}_r.eval_conv": Stage(c, c, 9) for k in (1, 2, 3)}
    if any(block != {**triplet, "att_conv": Stage(c, c, 1)} for block in blocks.values()):
        raise ValueError("couplage 3×3 et porte inattendu")
    if layers != {"conv_1.1.eval_conv.weight": Stage(c, 12, 9), "conv_cat.weight": Stage(c, 4 * c, 1),
                  "to_pixel.eval_conv.weight": Stage(48, c, 9)}:
        raise ValueError("entrée ou queue SPAN inattendue")
    return c, expected_depth


def ceil_to(n, unit):
    return (n + unit - 1) // unit


def convolution(h, w, stage, form):
    area = h * w
    products, weights = area * stage.taps, stage.taps
    transform = 0
    match form:
        case "vertical":
            pairs = ceil_to(h, 2) * w
            products, weights = 12 * pairs, 12
            transform = pairs * 4 * (stage.inputs + stage.outputs)
        case "F2":
            tiles = ceil_to(h, 2) * ceil_to(w, 2)
            products, weights = 16 * tiles, 16
            transform = tiles * (32 * stage.inputs + 24 * stage.outputs)
        case "F4":
            tiles = ceil_to(h, 4) * ceil_to(w, 4)
            products, weights = 36 * tiles, 36
            # Estimation avec 22 termes d'entrée et 18 de sortie par passage 1D.
            transform = tiles * (12 * 22 * stage.inputs + 10 * 18 * stage.outputs)
    cin = ceil_to(stage.inputs, 16) * 16
    mma = 2 * products * cin * stage.outputs
    traffic = 2 * (area * (stage.inputs + stage.outputs + stage.skip) + weights * cin * stage.outputs)
    return mma, transform, traffic


def calculate(c, count, h, w, form, fused_tail):
    first = convolution(h, w, Stage(c, 12, 9), "direct")
    cat = convolution(h, w, Stage(c, 4 * c, 1), "direct")
    pixel = convolution(h, w, Stage(48, c, 9), "direct")
    normal_form = "vertical" if form == "hybride" else form
    lower = convolution(h, w, Stage(c, c, 9), normal_form)
    second = convolution(h, w, Stage(c, c, 9), "direct") if form == "hybride" else lower
    third = convolution(h, w, Stage(c, c, 9, skip=c), normal_form)
    # La porte 1×1 lit le fragment 3×3 local. Ses accumulateurs ne sortent pas vers la VRAM.
    gate = (2 * h * w * c * c, 0, 2 * c * c)
    finish = tuple(a + b for a, b in zip(third, gate, strict=True))
    body = [stage for _ in range(count) for stage in (lower, second, finish)]
    if fused_tail:
        mma, transform, traffic = body.pop()
        body.append((mma + cat[0], transform + cat[1], traffic + cat[2] - 4 * h * w * c))
    else:
        body.append(cat)
    stages = [first, *body, pixel]
    compute = sum(m / PEAK_MMA for m, _, _ in stages)
    estimated = sum(max(m / PEAK_MMA + f / PEAK_FP32, b / DRAM) for m, f, b in stages)
    return 1000 * compute, 1000 * estimated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("dimensions strictement positives attendues")
    h, w = ceil_to(args.height, 2), ceil_to(args.width, 2)
    print("Calcul MMA seul et modèle par étage avec transformations et transfert VRAM")
    print("La mémoire du modèle suppose une lecture et une écriture des tenseurs à chaque étage.")
    print("Des lectures intermédiaires peuvent être servies par le cache. La qualité se mesure séparément.")
    for model in args.models:
        channels, blocks = model_stages(model)
        print(f"{Path(model).name} · {args.width} × {args.height}")
        for form in ("direct", "vertical", "hybride", "F2", "F4"):
            if form == "hybride" and channels != 48:
                continue
            mma, separate = calculate(channels, blocks, h, w, form, False)
            _, fused = calculate(channels, blocks, h, w, form, True)
            print(f"  {form:9} · MMA {mma:5.3f} ms · modèle {separate:5.3f} ms"
                  f" · queue fusionnée {fused:5.3f} ms · 90 % de ce modèle {fused/.9:5.3f} ms")


if __name__ == "__main__":
    main()
