"""Le graphe de GMFSS Fortuna coupé pour l'export (export_gmfss.py) : le softmax splatting réécrit en
PyTorch, le réseau chargé depuis la roue, les modules « paire » et « fusion », leurs noms et leurs axes
dynamiques, et les versions exportables des fonctions de GMFlow et MetricNet.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


LEVELS = (1, 2, 3)
SIDES = (0, 1)

# Version du format des graphes, écrite dans la métadonnée « ajp.gmfss.export » de chacun et lue par le
# lecteur (app/play/gmfss_model.h : kGmfssExportVersion) ; à monter avec lui quand un changement de
# l'export change le contrat, le résultat ou le coût.
#   1 (métadonnée absente) : les exports d'avant, source{k}_1 en un tenseur de 67 canaux (un Concat
#       3 + 64 canaux par côté dans la paire, 1,2 ms profilé à 1280x768) ; certains lisaient encore la
#       fenêtre locale de GMFlow par un GridSample de 159 millions de valeurs (avant a12132d).
#   2 : feature{k}_1 (64 canaux) et half{k} (RGB, 3 canaux) sortent séparés, sans Concat : le lecteur
#       splatte les deux dans les tranches de warped{k}_1 (67 canaux, RGB d'abord).
EXPORT_VERSION = 2
EXPORT_KEY = "ajp.gmfss.export"


def soft_splat(source, flow, metric, strMode="soft"):  # noqa: N803, nom de softsplat.py
    """softsplat « soft » de softsplat.py, en PyTorch : addition bilinéaire vers l'avant pondérée
    par e^métrique, puis normalisation + 1e-7 (spec/gmfss.h)."""
    assert strMode == "soft"
    dtype = source.dtype
    source, flow, metric = source.float(), flow.float(), metric.float()
    scale = metric.exp()
    values = torch.cat([source * scale, scale], 1)
    b, c, h, w = values.shape
    tx = torch.arange(w, dtype=torch.float32, device=flow.device).view(1, 1, w) + flow[:, 0]
    ty = torch.arange(h, dtype=torch.float32, device=flow.device).view(1, h, 1) + flow[:, 1]
    finite = torch.isfinite(tx) & torch.isfinite(ty)
    tx = torch.where(finite, tx, torch.zeros_like(tx))
    ty = torch.where(finite, ty, torch.zeros_like(ty))
    x0 = torch.floor(tx)
    y0 = torch.floor(ty)
    out = torch.zeros(b, c, h * w, device=flow.device)
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


def bidirectional_reuse(gmfss, img0, img1):
    """GMFSS.reuse, les deux flots d'un seul passage de GMFlow (pred_bidir_flow) : son
    transformeur traite déjà les deux ordres en lot, et le flot inverse de l'appariement global
    vient de la corrélation transposée. L'original appelait GMFlow deux fois (0 → 1 puis 1 → 0),
    refaisant l'encodeur des deux images et le transformeur du niveau grossier."""
    feat0 = list(gmfss.feat_ext(img0))
    feat1 = list(gmfss.feat_ext(img1))
    img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
    img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")
    if gmfss.scale != 1.0:
        imgf0 = F.interpolate(img0, scale_factor=gmfss.scale, mode="bilinear")
        imgf1 = F.interpolate(img1, scale_factor=gmfss.scale, mode="bilinear")
    else:
        imgf0, imgf1 = img0, img1
    flow01, flow10 = gmfss.flownet(imgf0, imgf1, pred_bidir_flow=True).chunk(2, 0)
    if gmfss.scale != 1.0:
        flow01 = F.interpolate(flow01, scale_factor=1.0 / gmfss.scale, mode="bilinear") / gmfss.scale
        flow10 = F.interpolate(flow10, scale_factor=1.0 / gmfss.scale, mode="bilinear") / gmfss.scale
    metric0, metric1 = gmfss.metricnet(img0, img1, flow01, flow10)
    return flow01, flow10, metric0, metric1, feat0, feat1


class Pair(nn.Module):
    """Une fois par paire : ce que chaque côté splatte (au niveau 1 les 64 caractéristiques, l'image
    réduite de moitié étant l'une des deux sorties half{k}), son mouvement aux trois niveaux, et les
    deux images réduites de moitié. Le flot de chaque niveau est déjà réduit (× 0,5, × 0,25) :
    l'instant ne fait plus que le multiplier, l'interpolation bilinéaire étant linéaire."""

    def __init__(self, gmfss):
        super().__init__()
        self.gmfss = gmfss

    def forward(self, frames):
        img0, img1 = frames[:, 0:3], frames[:, 3:6]
        flow01, flow10, metric0, metric1, feat0, feat1 = bidirectional_reuse(self.gmfss, img0, img1)
        half0, half1 = half_size(img0), half_size(img1)
        outputs = []
        for half, flow, metric, feat in ((half0, flow01, metric0, feat0), (half1, flow10, metric1, feat1)):
            for level in LEVELS:
                scale = 0.5 ** (level - 1)
                if level == 1:
                    # Les caractéristiques seules : l'image réduite sort déjà (half{k}), le lecteur
                    # splatte les deux sans que la paire les concatène.
                    source, level_flow, level_metric = feat[0], flow, metric
                else:
                    source = feat[level - 1]
                    level_flow = F.interpolate(flow, scale_factor=scale, mode="bilinear") * scale
                    level_metric = F.interpolate(metric, scale_factor=scale, mode="bilinear")
                outputs += [source, torch.cat([level_flow, level_metric], 1)]
        return (*outputs, half0, half1)


def source_name(side, level):
    """La sortie de la paire que le côté splatte au niveau : feature{k}_1 (64 caractéristiques, l'image
    est half{k}) au niveau 1, source{k}_{l} aux autres."""
    return f"feature{side}_1" if level == 1 else f"source{side}_{level}"


def pair_names():
    names = []
    for side in SIDES:
        for level in LEVELS:
            names += [source_name(side, level), f"motion{side}_{level}"]
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
            source = named[source_name(side, level)]
            if level == 1:  # l'image d'abord, puis les caractéristiques : 67 canaux comme warped{k}_1
                source = torch.cat([named[f"half{side}"], source], 1)
            warped.append(soft_splat(source, motion[:, 0:2] * s, motion[:, 2:3] * s))
    return named["half0"], named["half1"], warped


def dims(level):
    if level == 0:
        return {2: "H", 3: "W"}
    return {2: f"H/{2 ** level}", 3: f"W/{2 ** level}"}


def pair_axes():
    axes = {"frames": dims(0), "half0": dims(1), "half1": dims(1)}
    for side in SIDES:
        for level in LEVELS:
            axes[source_name(side, level)] = dims(level)
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
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
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
    """local_correlation_softmax de gmflow/matching.py, la fenêtre lue par décalages entiers.

    Les positions de la fenêtre sont des pixels entiers (pixel + décalage de -r à r) : le
    grid_sample de l'original n'interpole rien, il lit un voisin ou zéro hors de l'image. Ici, la
    carte est bordée de zéros puis lue par (2r+1)² tranches, et chaque corrélation est un produit
    sommé sur les canaux : ni le tenseur [B, HW, C, (2r+1)²] de l'original (159 millions de valeurs
    à 720p, 7,5 ms par appel sous TensorRT), ni grille. L'ordre de la fenêtre est celui de
    generate_window_grid : k = (dy + r) × (2r + 1) + (dx + r)."""
    from vsgmfss_fortuna.gmflow.geometry import coords_grid, generate_window_grid

    assert padding_mode == "zeros"
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    r = local_radius
    side = 2 * r + 1
    window_grid = generate_window_grid(-r, r, -r, r, side, side, device=feature0.device)
    sample_coords = coords.unsqueeze(-2) + window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    padded = F.pad(feature1, (r, r, r, r))
    corr = torch.cat([(feature0 * padded[:, :, dy:dy + h, dx:dx + w]).sum(1, keepdim=True)
                      for dy in range(side) for dx in range(side)], 1)
    corr = corr.view(b, side * side, h * w).permute(0, 2, 1) / (c**0.5)
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
