#!/usr/bin/env python3
"""Exporte GMFSS Fortuna en deux graphes ONNX pour le lecteur (kernels/spec/gmfss.h).

Le réseau vient de vsgmfss_fortuna 1.0.0 (HolyWu, MIT) : son code et ses poids sont dans la roue
PyPI, que l'on extrait sans l'installer (son __init__ importe TensorRT et VapourSynth) :

    pip download --no-deps vsgmfss_fortuna==1.0.0 -d gmfss
    python -m zipfile -e gmfss/vsgmfss_fortuna-1.0.0-py3-none-any.whl gmfss/roue
    python -I tools/gmfss/export_gmfss.py gmfss/roue/vsgmfss_fortuna --out modeles/gmfss

Le softmax splatting (softsplat.py, noyau CuPy) n'a pas d'export ONNX : le réseau est coupé
autour de lui. Le graphe « paire » tourne une fois par paire (FeatureNet, GMFlow dans les deux
sens, MetricNet) ; le lecteur fait les six splats à chaque instant, puis le graphe « fusion »
(GridNet). Ici le splat est réécrit en PyTorch pour vérifier la coupe : le réseau coupé doit
rendre exactement ce que rend GMFSS.forward, à la taille de l'export et à une autre taille.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I : le dossier du script n'y est pas
from gmfss_graph import (  # noqa: E402
    Fusion,
    Pair,
    backwarp,
    dims,
    flow_warp,
    fusion_axes,
    fusion_names,
    l1_distance,
    load_gmfss,
    local_correlation_softmax,
    normalize_coords,
    pair_axes,
    pair_names,
    shift_window_attn_mask,
    splat_all,
    split_window_attention,
)


@contextlib.contextmanager
def exportable():
    """Les fonctions du paquet remplacées par leurs versions exportables, le temps de l'export et
    des passes PyTorch qui le vérifient ; le réseau de référence, lui, tourne sans elles."""
    transformer = importlib.import_module("vsgmfss_fortuna.gmflow.transformer")
    matching = importlib.import_module("vsgmfss_fortuna.gmflow.matching")
    geometry = importlib.import_module("vsgmfss_fortuna.gmflow.geometry")
    gmflow = importlib.import_module("vsgmfss_fortuna.gmflow.gmflow")
    metric = importlib.import_module("vsgmfss_fortuna.MetricNet")
    patches = [
        (transformer, "single_head_split_window_attention", split_window_attention),
        (transformer, "generate_shift_window_attn_mask", shift_window_attn_mask),
        (matching, "normalize_coords", normalize_coords),
        (gmflow, "local_correlation_softmax", local_correlation_softmax),
        (geometry, "flow_warp", flow_warp),
        (gmflow, "flow_warp", flow_warp),
        (metric, "backwarp", backwarp),
        (F, "l1_loss", l1_distance),
    ]
    saved = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, value in patches:
        setattr(module, name, value)
    warplayer = sys.modules.get("vsgmfss_fortuna.warplayer")
    if warplayer is not None:
        warplayer.backwarp_tenGrid.clear()  # union : la grille de IFNet, recalculée dans le graphe
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
        if warplayer is not None:
            warplayer.backwarp_tenGrid.clear()


def export(module, args, path, inputs, outputs, axes):
    # Sans autograd : les entrées de la fusion sortent d'inference_mode, et l'IFNet de « union » les
    # garderait pour la rétropropagation, ce qu'interdit PyTorch.
    with torch.no_grad(), exportable():
        torch.onnx.export(
            module,
            tuple(args),
            str(path),
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )


def settle_dims(path, model_type):
    """Les dimensions que l'export laisse nommées alors qu'elles sont fixes (canaux du mouvement,
    lot de la sortie) prennent leur valeur ; la métadonnée dit ce que le graphe contient."""
    import onnx

    model = onnx.load(str(path))
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if value.name.startswith("motion"):
            dims[1].dim_value = 3
        if value.name == "out":
            dims[0].dim_value = 1
        for d in dims:
            if d.HasField("dim_value"):
                continue
            assert d.dim_param in {"H", "W", "H/2", "W/2", "H/4", "W/4", "H/8", "W/8"}, (value.name, d.dim_param)
    entry = model.metadata_props.add()
    entry.key = "ajp.gmfss"
    entry.value = f"vsgmfss_fortuna 1.0.0 {model_type}, kernels/spec/gmfss.h"
    onnx.save(model, str(path))


def run_onnx(path, feeds):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {k: v.numpy() for k, v in feeds.items()})


def frames_for(width, height, seed):
    """Deux images lisses décalées : un mouvement réel, pas du bruit que GMFlow ne suit pas."""
    g = torch.Generator().manual_seed(seed)
    base = F.interpolate(torch.rand(1, 3, height // 16 + 2, width // 16 + 2, generator=g), (height + 16, width + 16),
                         mode="bicubic", align_corners=False).clamp(0, 1)
    img0 = base[:, :, 8 : 8 + height, 8 : 8 + width]
    img1 = base[:, :, 5 : 5 + height, 11 : 11 + width]
    return img0, img1


@torch.inference_mode()
def check(reference, gmfss, pair_path, fusion_path, width, height, t):
    """Écart à GMFSS.forward (FP32, fonctions d'origine) de la coupe en PyTorch puis des graphes
    ONNX, splats faits ici ; rend les écarts maximaux de la coupe et de l'ONNX, et l'écart moyen de l'ONNX."""
    dtype = next(gmfss.parameters()).dtype
    img0, img1 = frames_for(width, height, seed=width * 7 + height)
    expected = reference(img0, img1, torch.tensor([t]))
    frames = torch.cat([img0, img1], 1).to(dtype)
    union = gmfss.model_type == "union"
    extra = [torch.tensor([[[[t]]]], dtype=dtype)] if union else []
    with exportable():
        pair_outputs = Pair(gmfss)(frames)
        half0, half1, warped = splat_all(pair_outputs, t)
        cut = Fusion(gmfss)(half0, half1, *extra, *warped).float()
    torch_gap = (cut - expected).abs().max().item()

    onnx_pair = [torch.from_numpy(a) for a in run_onnx(pair_path, {"frames": frames})]
    pair_gap = max((a.float() - b.float()).abs().max().item() for a, b in zip(onnx_pair, pair_outputs))
    half0, half1, warped = splat_all(onnx_pair, t)
    feeds = dict(zip(fusion_names(gmfss.model_type), [half0, half1, *extra, *warped]))
    onnx_out = torch.from_numpy(run_onnx(fusion_path, feeds)[0]).float()
    onnx_gap = (onnx_out - expected).abs().max().item()
    mean_gap = (onnx_out - expected).abs().mean().item()
    print(f"  {width}x{height} t={t} : coupe {torch_gap:.2e}, graphe paire {pair_gap:.2e}, "
          f"ONNX de bout en bout {onnx_gap:.2e} (moyenne {mean_gap:.2e})")
    return torch_gap, onnx_gap, mean_gap


def export_pair(gmfss, model_type, out, stem, width, height):
    dtype = next(gmfss.parameters()).dtype
    pair_path = out / f"{stem}_pair.onnx"
    fusion_path = out / f"{stem}_fusion.onnx"
    img0, img1 = frames_for(width, height, seed=1)
    frames = torch.cat([img0, img1], 1).to(dtype)
    with torch.inference_mode(), exportable():
        pair_outputs = Pair(gmfss)(frames)
        half0, half1, warped = splat_all(pair_outputs, 0.5)
    export(Pair(gmfss), (frames,), pair_path, ["frames"], pair_names(), pair_axes())
    extra = [torch.tensor([[[[0.5]]]], dtype=dtype)] if model_type == "union" else []
    export(Fusion(gmfss), (half0, half1, *extra, *warped), fusion_path, fusion_names(model_type), ["out"],
           fusion_axes())
    settle_dims(pair_path, model_type)
    settle_dims(fusion_path, model_type)
    return pair_path, fusion_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="dossier vsgmfss_fortuna extrait de la roue (GMFSS.py, models/)")
    parser.add_argument("--out", type=Path, required=True, help="dossier des deux graphes")
    parser.add_argument("--model", choices=("base", "union"), default="base")
    parser.add_argument("--width", type=int, default=256, help="largeur de l'export, multiple de 64")
    parser.add_argument("--height", type=int, default=192, help="hauteur de l'export, multiple de 64")
    parser.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0,
                        help="échelle de GMFlow (scale de GMFSS) : 0,5 calcule le flot à un quart de la "
                             "taille de travail, dimensions multiples de 128 ; graphes suffixés _s05")
    args = parser.parse_args()
    align = 64 if args.scale == 1.0 else 128
    if args.width % align or args.height % align:
        parser.error(f"largeur et hauteur multiples de {align}")

    GMFSS = load_gmfss(args.package.resolve())
    models = str(args.package / "models")
    reference = GMFSS(models, args.model, args.scale, False).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"gmfss_fortuna_{args.model}" + ("" if args.scale == 1.0 else "_s05")
    sizes = ((args.width, args.height, 0.5), (args.width + align, args.height + 2 * align, 0.25))

    # FP32 d'abord, vérifié strictement à deux tailles : la coupe et les tailles variables tiennent.
    with tempfile.TemporaryDirectory() as scratch:
        pair_path, fusion_path = export_pair(reference, args.model, Path(scratch), stem, args.width, args.height)
        gaps = [check(reference, reference, pair_path, fusion_path, *size) for size in sizes]
    # La coupe doit être exacte. ONNX Runtime (CPU) calcule la métrique de MetricNet (|m| ≈ 15) à
    # 1e-3 près, et e^m l'amplifie dans le splat : quelques pixels à ~7e-3, la moyenne reste à
    # 1e-5. Une coupe fausse ou une taille figée à l'export lèverait la moyenne.
    cut, worst, mean = (max(gap[i] for gap in gaps) for i in range(3))
    if not all(np.isfinite((cut, worst, mean))) or cut > 1e-3 or worst > 1e-2 or mean > 1e-4:
        sys.exit(f"FP32 : coupe {cut:.2e} (≤ 1e-3), ONNX {worst:.2e} (≤ 1e-2), moyenne {mean:.2e} (≤ 1e-4) : "
                 "la coupe ou l'export ne tient pas")

    # Puis le FP16 que TensorRT construit (grilles d'échantillonnage en FP32), le même code tracé.
    half = GMFSS(models, args.model, args.scale, False).eval().half()
    pair_path, fusion_path = export_pair(half, args.model, args.out, stem, args.width, args.height)
    print(f"{pair_path}\n{fusion_path}")
    worst = max(max(check(reference, half, pair_path, fusion_path, *size)[:2]) for size in sizes)
    if not np.isfinite(worst) or worst > 0.1:
        sys.exit(f"FP16 : écart {worst:.2e} au-delà de 0,1")


if __name__ == "__main__":
    main()
