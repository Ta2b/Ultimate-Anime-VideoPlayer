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
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LEVELS = (1, 2, 3)
SIDES = (0, 1)


def soft_splat(source, flow, metric, strMode="soft"):  # noqa: N803, nom de softsplat.py
    """softsplat « soft » de softsplat.py, en PyTorch : addition bilinéaire vers l'avant pondérée
    par e^métrique, puis normalisation + 1e-7 (spec/gmfss.h)."""
    assert strMode == "soft"
    dtype = source.dtype
    source, flow, metric = source.float(), flow.float(), metric.float()
    scale = metric.exp()
    values = torch.cat([source * scale, scale], 1)
    b, c, h, w = values.shape
    tx = torch.arange(w, dtype=torch.float32).view(1, 1, w) + flow[:, 0]
    ty = torch.arange(h, dtype=torch.float32).view(1, h, 1) + flow[:, 1]
    finite = torch.isfinite(tx) & torch.isfinite(ty)
    tx = torch.where(finite, tx, torch.zeros_like(tx))
    ty = torch.where(finite, ty, torch.zeros_like(ty))
    x0 = torch.floor(tx)
    y0 = torch.floor(ty)
    out = torch.zeros(b, c, h * w)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cx = x0 + dx
        cy = y0 + dy
        wx = tx - x0 if dx else x0 + 1 - tx
        wy = ty - y0 if dy else y0 + 1 - ty
        valid = finite & (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        index = (cy * w + cx).long()
        for n in range(b):
            keep = valid[n].reshape(-1)
            contribution = (values[n].reshape(c, -1) * (wx * wy)[n].reshape(1, -1))[:, keep]
            out[n].index_add_(1, index[n].reshape(-1)[keep], contribution)
    out = out.view(b, c, h, w)
    return (out[:, :-1] / (out[:, -1:] + 1e-7)).to(dtype)


def load_gmfss(package: Path):
    """Le module GMFSS du paquet, sans son __init__ ni CuPy : softsplat est remplacé par
    soft_splat avant l'import de GMFSS.py."""
    root = types.ModuleType("vsgmfss_fortuna")
    root.__path__ = [str(package)]
    sys.modules["vsgmfss_fortuna"] = root
    splat = types.ModuleType("vsgmfss_fortuna.softsplat")
    splat.softsplat = soft_splat
    sys.modules["vsgmfss_fortuna.softsplat"] = splat
    return importlib.import_module("vsgmfss_fortuna.GMFSS").GMFSS


def half_size(image):
    return F.interpolate(image, scale_factor=0.5, mode="bilinear")


class Pair(nn.Module):
    """Une fois par paire : ce que chaque côté splatte, son mouvement aux trois niveaux, et les
    deux images réduites de moitié. Le flot de chaque niveau est déjà réduit (× 0,5, × 0,25) :
    l'instant ne fait plus que le multiplier, l'interpolation bilinéaire étant linéaire."""

    def __init__(self, gmfss):
        super().__init__()
        self.gmfss = gmfss

    def forward(self, frames):
        img0, img1 = frames[:, 0:3], frames[:, 3:6]
        flow01, flow10, metric0, metric1, feat0, feat1 = self.gmfss.reuse(img0, img1)
        half0, half1 = half_size(img0), half_size(img1)
        outputs = []
        for half, flow, metric, feat in ((half0, flow01, metric0, feat0), (half1, flow10, metric1, feat1)):
            for level in LEVELS:
                scale = 0.5 ** (level - 1)
                if level == 1:
                    source, level_flow, level_metric = torch.cat([half, feat[0]], 1), flow, metric
                else:
                    source = feat[level - 1]
                    level_flow = F.interpolate(flow, scale_factor=scale, mode="bilinear") * scale
                    level_metric = F.interpolate(metric, scale_factor=scale, mode="bilinear")
                outputs += [source, torch.cat([level_flow, level_metric], 1)]
        return (*outputs, half0, half1)


def pair_names():
    names = []
    for side in SIDES:
        for level in LEVELS:
            names += [f"source{side}_{level}", f"motion{side}_{level}"]
    return names + ["half0", "half1"]


def warped_names():
    return [f"warped{side}_{level}" for side in SIDES for level in LEVELS]


class Fusion(nn.Module):
    """Une fois par image intermédiaire : GridNet sur les sources splattées (warped{k}_{l}). Le
    modèle union reçoit aussi l'instant, pour son IFNet."""

    def __init__(self, gmfss):
        super().__init__()
        self.gmfss = gmfss

    def forward(self, half0, half1, *rest):
        if self.gmfss.model_type == "union":
            time, warped = rest[0], rest[1:]
        else:
            warped = rest
        w = dict(zip(warped_names(), warped))
        image0, image1 = w["warped0_1"][:, :3], w["warped1_1"][:, :3]
        feat0, feat1 = w["warped0_1"][:, 3:], w["warped1_1"][:, 3:]
        if self.gmfss.model_type == "union":
            rife = self.gmfss.ifnet(half0, half1, time.reshape(1))
            images = torch.cat([image0, rife, image1], 1)
        else:
            images = torch.cat([half0, image0, image1, half1], 1)
        out = self.gmfss.fusionnet(
            images,
            torch.cat([feat0, feat1], 1),
            torch.cat([w["warped0_2"], w["warped1_2"]], 1),
            torch.cat([w["warped0_3"], w["warped1_3"]], 1),
        )
        return torch.clamp(out, 0, 1)


def fusion_names(model_type):
    return ["half0", "half1"] + (["time"] if model_type == "union" else []) + warped_names()


def splat_all(pair_outputs, t):
    """Les six splats à l'instant t, comme le lecteur (gmfss_side_time)."""
    named = dict(zip(pair_names(), pair_outputs))
    warped = []
    for side in SIDES:
        s = t if side == 0 else 1.0 - t
        for level in LEVELS:
            motion = named[f"motion{side}_{level}"]
            warped.append(soft_splat(named[f"source{side}_{level}"], motion[:, 0:2] * s, motion[:, 2:3] * s))
    return named["half0"], named["half1"], warped


def dims(level):
    if level == 0:
        return {2: "H", 3: "W"}
    return {2: f"H/{2 ** level}", 3: f"W/{2 ** level}"}


def pair_axes():
    axes = {"frames": dims(0), "half0": dims(1), "half1": dims(1)}
    for side in SIDES:
        for level in LEVELS:
            axes[f"source{side}_{level}"] = dims(level)
            axes[f"motion{side}_{level}"] = dims(level)
    return axes


def fusion_axes():
    axes = {"half0": dims(1), "half1": dims(1), "out": dims(0)}
    for side in SIDES:
        for level in LEVELS:
            axes[f"warped{side}_{level}"] = dims(level)
    return axes


def roll_left(tensor, shift, dim):
    """torch.roll(tensor, -shift, dim) par deux tranches."""
    n = tensor.shape[dim]
    return torch.cat([tensor.narrow(dim, shift, n - shift), tensor.narrow(dim, 0, shift)], dim)


def roll_right(tensor, shift, dim):
    """torch.roll(tensor, shift, dim) par deux tranches."""
    n = tensor.shape[dim]
    return torch.cat([tensor.narrow(dim, n - shift, shift), tensor.narrow(dim, 0, n - shift)], dim)


def split_window_attention(q, k, v, num_splits=1, with_shift=False, h=None, w=None, attn_mask=None):
    """single_head_split_window_attention de gmflow/transformer.py, torch.roll en tranches :
    l'export ONNX de torch.roll veut des décalages constants, ceux-ci suivent la taille."""
    from vsgmfss_fortuna.gmflow.utils import merge_splits, split_feature

    b, _, c = q.size()
    b_new = b * num_splits * num_splits
    window_size_h = h // num_splits
    window_size_w = w // num_splits
    q, k, v = (x.view(b, h, w, c) for x in (q, k, v))
    scale_factor = c**0.5
    if with_shift:
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2
        q, k, v = (roll_left(roll_left(x, shift_size_h, 1), shift_size_w, 2) for x in (q, k, v))
    q, k, v = (split_feature(x, num_splits=num_splits, channel_last=True) for x in (q, k, v))
    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / scale_factor
    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.view(b_new, -1, c))
    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c), num_splits=num_splits,
                       channel_last=True)
    if with_shift:
        out = roll_right(roll_right(out, shift_size_h, 1), shift_size_w, 2)
    return out.view(b, -1, c)


def shift_window_attn_mask(input_resolution, window_size_h, window_size_w, shift_size_h, shift_size_w,
                           device=None):
    """generate_shift_window_attn_mask de gmflow/transformer.py, les régions comptées par
    comparaisons au lieu d'affectations de tranches, que l'export fige à sa taille."""
    from vsgmfss_fortuna.gmflow.utils import split_feature

    h, w = input_resolution
    ys = torch.arange(h)
    xs = torch.arange(w)
    region_y = (ys >= h - window_size_h).long() + (ys >= h - shift_size_h).long()
    region_x = (xs >= w - window_size_w).long() + (xs >= w - shift_size_w).long()
    img_mask = (region_y.view(h, 1) * 3 + region_x.view(1, w)).float().view(1, h, w, 1)
    mask_windows = split_feature(img_mask, num_splits=w // window_size_w, channel_last=True)
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


def normalize_coords(coords, h, w):
    """normalize_coords de gmflow/geometry.py sans torch.tensor([...]), qui fige la taille."""
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return torch.stack([(coords[..., 0] - cx) / cx, (coords[..., 1] - cy) / cy], -1)


def l1_distance(a, b, reduction="mean"):
    """F.l1_loss sans réduction (MetricNet), que l'export ONNX ne connaît pas."""
    assert reduction == "none"
    return (a - b).abs()


def flow_warp(feature, flow, mask=False, padding_mode="zeros"):
    """flow_warp de gmflow/geometry.py, la grille en FP32 : en coordonnées normalisées, le FP16
    arrondit la position d'échantillonnage à près d'un pixel (convert_rife_fp16.py de RIFE)."""
    from vsgmfss_fortuna.gmflow.geometry import bilinear_sample, coords_grid

    b, c, h, w = feature.size()
    grid = coords_grid(b, h, w).to(flow.device) + flow.float()
    out = bilinear_sample(feature.float(), grid, padding_mode=padding_mode, return_mask=mask)
    return (out[0].to(feature.dtype), out[1]) if mask else out.to(feature.dtype)


def local_correlation_softmax(feature0, feature1, local_radius, padding_mode="zeros"):
    """local_correlation_softmax de gmflow/matching.py, coordonnées et échantillonnage en FP32."""
    from vsgmfss_fortuna.gmflow.geometry import coords_grid, generate_window_grid

    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    side = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius, side, side,
                                       device=feature0.device)
    sample_coords = coords.unsqueeze(-2) + window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    window_feature = F.grid_sample(feature1.float(), normalize_coords(sample_coords, h, w),
                                   padding_mode=padding_mode, align_corners=True).to(feature0.dtype)
    window_feature = window_feature.permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c**0.5)
    corr[~valid] = -1e9 if feature0.dtype == torch.float else -1e4
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2).float(), sample_coords).squeeze(-2)
    flow = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2) - coords_init
    return flow.to(feature0.dtype), prob


def backwarp(tenIn, tenflow):  # noqa: N803, noms de MetricNet.py
    """backwarp de MetricNet.py, la grille en FP32 et recalculée à chaque appel (le cache par
    taille entrerait dans le graphe en constante)."""
    h, w = tenflow.shape[2], tenflow.shape[3]
    hor = torch.linspace(-1.0, 1.0, steps=w, device=tenflow.device).view(1, 1, 1, -1).repeat(1, 1, h, 1)
    ver = torch.linspace(-1.0, 1.0, steps=h, device=tenflow.device).view(1, 1, -1, 1).repeat(1, 1, 1, w)
    flow = tenflow.float()
    flow = torch.cat([flow[:, 0:1] / ((tenIn.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((tenIn.shape[2] - 1.0) / 2.0)],
                     1)
    grid = (torch.cat([hor, ver], 1) + flow).permute(0, 2, 3, 1)
    out = F.grid_sample(input=tenIn.float(), grid=grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.to(tenIn.dtype)


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
    args = parser.parse_args()
    if args.width % 64 or args.height % 64:
        parser.error("largeur et hauteur multiples de 64")

    GMFSS = load_gmfss(args.package.resolve())
    models = str(args.package / "models")
    reference = GMFSS(models, args.model, 1.0, False).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"gmfss_fortuna_{args.model}"
    sizes = ((args.width, args.height, 0.5), (args.width + 64, args.height + 128, 0.25))

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
    half = GMFSS(models, args.model, 1.0, False).eval().half()
    pair_path, fusion_path = export_pair(half, args.model, args.out, stem, args.width, args.height)
    print(f"{pair_path}\n{fusion_path}")
    worst = max(max(check(reference, half, pair_path, fusion_path, *size)[:2]) for size in sizes)
    if not np.isfinite(worst) or worst > 0.1:
        sys.exit(f"FP16 : écart {worst:.2e} au-delà de 0,1")


if __name__ == "__main__":
    main()
