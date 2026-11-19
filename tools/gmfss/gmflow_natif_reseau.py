"""GMFlow de GMFSS Fortuna découpé comme le ferait le natif (docs/gmflow-natif.md), en PyTorch CPU.

Chaque étape est celle d'un noyau prévu : les tenseurs entre deux noyaux sont arrondis au format de stockage
(FP16, ou E4M3 à l'entrée d'une GEMM FP8), les sommes se font en f32 ou, si la recette le dit, en f16 par paquets
de k (16 en FP16, 32 en FP8, comme un mma.sync m16n8k16 ou m16n8k32 à accumulateur f16). L'attention par
fenêtres, l'appariement global et la propagation globale passent par tuiles de clés avec un softmax en ligne
(maximum et somme courants, P arrondi en FP16 avant le produit par V), jamais la matrice entière. Les poids sont
ceux de flownet.pkl ; la recette « fp32 » rend GMFlow d'origine à l'arrondi près (vérifié par gmflow_natif_emule.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


@dataclass
class Recette:
    """Les formats d'une émulation. stockage : fp32 ou fp16 entre deux noyaux. conv_acc16, gemm_acc16 : sommes en
    f16 par paquets de k. gemm8 : projections et MLP du transformeur en E4M3 (échelles d'activation par tenseur,
    de poids par canal de sortie). conv8 : convolutions du squelette et de l'upsampler en E4M3 aussi. p16 : P
    arrondi en FP16 avant P·V dans l'attention par fenêtres. flot_pv : « f32 » (P exact, flot en f32, cœurs CUDA)
    ou « f16 » (P et coordonnées en FP16 sur les tensor cores) pour l'appariement global et la propagation.
    tuile : clés par tuile du softmax en ligne (0 : softmax d'un bloc, comme l'original)."""

    nom: str = "fp32"
    stockage: str = "fp32"
    conv_acc16: bool = False
    gemm_acc16: bool = False
    gemm8: bool = False
    conv8: bool = False
    p16: bool = False
    flot_pv: str = "f32"
    tuile: int = 0
    echelles: dict = field(default_factory=dict)  # entrée de GEMM -> |max| relevé (calibration)
    releve: dict | None = None  # calibration en cours : on y range les |max|
    debordements: int = 0  # sommes f16 devenues infinies
    gemm8_noms: tuple = ()  # gemm8 limité à ces GEMM (qkv, merge, mlp1, mlp2) ; vide : toutes
    fp8_jeton: bool = False  # échelle d'activation par jeton (et par tranche de 128 de K au-delà de 256)


RECETTES = {
    "fp32": dict(),
    "fp16": dict(stockage="fp16", p16=True, tuile=64),
    "fp16-acc16": dict(stockage="fp16", p16=True, tuile=64, conv_acc16=True, gemm_acc16=True),
    "fp16-pv16": dict(stockage="fp16", p16=True, tuile=64, flot_pv="f16"),
    "fp8-gemm": dict(stockage="fp16", p16=True, tuile=64, gemm8=True),
    "fp8-gemm-acc16": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, gemm_acc16=True),
    "fp8-tout": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, conv8=True),
    "fp8-qkv": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, gemm8_noms=("qkv",)),
    "fp8-merge": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, gemm8_noms=("merge",)),
    "fp8-mlp1": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, gemm8_noms=("mlp1",)),
    "fp8-mlp2": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, gemm8_noms=("mlp2",)),
    "fp8-jeton": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, fp8_jeton=True),
    "fp8-jeton-mlp": dict(stockage="fp16", p16=True, tuile=64, gemm8=True, fp8_jeton=True,
                          gemm8_noms=("mlp1", "mlp2")),
}


def recette(nom: str, echelles: dict | None = None) -> Recette:
    return Recette(nom=nom, echelles=dict(echelles or {}), **RECETTES[nom])


def stocke(x: torch.Tensor, r: Recette) -> torch.Tensor:
    """Le tenseur tel qu'un noyau l'écrit en mémoire."""
    return x.half().float() if r.stockage == "fp16" else x


def e4m3(x: torch.Tensor, echelle: torch.Tensor | float) -> torch.Tensor:
    """Quantification E4M3 saturée à 448 (les noyaux de SPAN bornent avant de convertir), rendue en valeurs
    entières de l'échelle : x ≈ e4m3(x) · échelle."""
    return (x / echelle).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _somme_par_paquets(a: torch.Tensor, b: torch.Tensor, k: int, r: Recette) -> torch.Tensor:
    """a [..., K] · b [K, N] avec l'accumulateur en f16, arrondi après chaque paquet de k (un mma)."""
    acc = torch.zeros(*a.shape[:-1], b.shape[1])
    for k0 in range(0, a.shape[-1], k):
        acc = (acc + a[..., k0 : k0 + k] @ b[k0 : k0 + k]).half().float()
    r.debordements += int((~torch.isfinite(acc)).sum())
    return acc


def _gemm8_jeton(x: torch.Tensor, w: torch.Tensor, r: Recette, acc16: bool) -> torch.Tensor:
    """E4M3 à échelle par jeton : l'épilogue qui écrit x tient la ligne entière (LayerNorm, attention) et en
    connaît le maximum ; dans le MLP fondu, chaque tranche de 128 de la couche cachée a le sien, appliqué au
    résultat partiel de la tranche en f32."""
    pas = 128 if x.shape[-1] > 256 else x.shape[-1]
    plafond = E4M3_MAX / 16 if acc16 else E4M3_MAX
    sw = (w.abs().amax(1).clamp_min(1e-12) / plafond).view(1, -1)
    wq = e4m3(w, sw.view(-1, 1))
    y = torch.zeros(*x.shape[:-1], w.shape[0])
    for k0 in range(0, x.shape[-1], pas):
        xc = x[..., k0 : k0 + pas]
        sa = xc.abs().amax(-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
        part = e4m3(xc, sa)
        part = _somme_par_paquets(part, wq[:, k0 : k0 + pas].t(), 32, r) if acc16 else part @ wq[:, k0 : k0 + pas].t()
        y = y + part * sa
    return y * sw


def gemm(x: torch.Tensor, w: torch.Tensor, r: Recette, nom: str, acc16: bool, fp8: bool) -> torch.Tensor:
    """x [..., K] · w[N, K]ᵀ comme un noyau GEMM : entrées au format de la recette, somme en f32 ou en f16."""
    if r.releve is not None:
        r.releve[nom] = max(r.releve.get(nom, 0.0), float(x.abs().max()))
    if fp8 and r.gemm8_noms and not any(nom.endswith(n) for n in r.gemm8_noms):
        fp8 = False
    if fp8 and r.fp8_jeton:
        return _gemm8_jeton(x, w, r, acc16)
    if fp8:
        # Échelle d'activation statique (calibration × 1,25 / 448, comme SPAN), poids par canal de sortie ; à
        # accumulateur f16, les poids rangés à 448 / 16 au plus (kAcc16Headroom de SPAN).
        sa = r.echelles.get(nom, float(x.abs().max())) * 1.25 / E4M3_MAX
        plafond = E4M3_MAX / 16 if acc16 else E4M3_MAX
        sw = (w.abs().amax(1).clamp_min(1e-12) / plafond).view(1, -1)
        xa, wq = e4m3(x, sa), e4m3(w, sw.view(-1, 1))
        y = _somme_par_paquets(xa, wq.t(), 32, r) if acc16 else xa @ wq.t()
        return y * sa * sw
    if r.stockage == "fp16":
        x, w = x.half().float(), w.half().float()
    return _somme_par_paquets(x, w.t(), 16, r) if acc16 else x @ w.t()


def conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, stride: int, pad: int, r: Recette,
         nom: str) -> torch.Tensor:
    """Convolution NCHW par une GEMM implicite : K pris prise par prise (l'ordre des tranches du natif)."""
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    if not (r.conv_acc16 or r.conv8 or r.releve is not None):
        xs, ws = (x.half().float(), w.half().float()) if r.stockage == "fp16" else (x, w)
        y = F.conv2d(xs, ws, None, stride, pad)
    else:
        cols = F.unfold(x, (kh, kw), padding=pad, stride=stride)  # [N, C·kh·kw, L], C d'abord
        cols = cols.view(n, c, kh * kw, -1).permute(0, 3, 2, 1).reshape(n, oh * ow, kh * kw * c)
        wk = w.permute(0, 2, 3, 1).reshape(o, kh * kw * c)
        y = gemm(cols, wk, r, nom, r.conv_acc16, r.conv8).permute(0, 2, 1).reshape(n, o, oh, ow)
    return y if b is None else y + b.view(1, -1, 1, 1)


# --- squelette CNN -------------------------------------------------------------------------------------------


def instance_norm(x: torch.Tensor, r: Recette) -> torch.Tensor:
    """InstanceNorm2d sans paramètres : les sommes Σx et Σx² en f32 (épilogue de la convolution, atomiques par
    canal), la normalisation au chargement par le noyau suivant."""
    m = x.mean((2, 3), keepdim=True)
    var = (x * x).mean((2, 3), keepdim=True) - m * m
    return (x - m) * torch.rsqrt(var.clamp_min(0) + 1e-5)


def bloc_residuel(x: torch.Tensor, sd: dict, p: str, stride: int, r: Recette) -> torch.Tensor:
    y = stocke(conv(x, sd[p + "conv1.weight"], None, stride, 1, r, p + "conv1"), r)
    y = stocke(F.relu(instance_norm(y, r)), r)
    y = stocke(conv(y, sd[p + "conv2.weight"], None, 1, 1, r, p + "conv2"), r)
    y = F.relu(instance_norm(y, r))
    if p + "downsample.0.weight" in sd:
        x = stocke(conv(x, sd[p + "downsample.0.weight"], sd[p + "downsample.0.bias"], stride, 0, r, p + "ds"), r)
        x = instance_norm(x, r)
    # Fin de bloc : un noyau élément par élément (normes, somme, Relu), une écriture FP16.
    return stocke(F.relu(x + y), r)


def squelette(img: torch.Tensor, sd: dict, r: Recette) -> list[torch.Tensor]:
    """CNNEncoder à deux échelles : [1/8, 1/4] (de la plus grossière à la plus fine, comme extract_feature)."""
    p = "backbone."
    x = stocke(conv(img, sd[p + "conv1.weight"], None, 2, 3, r, p + "conv1"), r)
    x = stocke(F.relu(instance_norm(x, r)), r)
    for couche, stride in (("layer1", 1), ("layer2", 2), ("layer3", 1)):
        x = bloc_residuel(x, sd, f"{p}{couche}.0.", stride, r)
        x = bloc_residuel(x, sd, f"{p}{couche}.1.", 1, r)
    x = stocke(conv(x, sd[p + "conv2.weight"], sd[p + "conv2.bias"], 1, 0, r, p + "conv2"), r)
    w = sd[p + "trident_conv.weight"]
    return [stocke(conv(x, w, None, 2, 1, r, p + "trident2"), r), stocke(conv(x, w, None, 1, 1, r, p + "trident1"), r)]


# --- transformeur --------------------------------------------------------------------------------------------


def position(h: int, w: int, c: int = 128) -> torch.Tensor:
    """PositionEmbeddingSine normalisée sur une fenêtre h × w : [C, h, w]."""
    y = torch.arange(1, h + 1, dtype=torch.float32).view(h, 1).expand(h, w) / (h + 1e-6) * 2 * math.pi
    x = torch.arange(1, w + 1, dtype=torch.float32).view(1, w).expand(h, w) / (w + 1e-6) * 2 * math.pi
    d = torch.arange(c // 2, dtype=torch.float32)
    d = 10000 ** (2 * (d // 2) / (c // 2))
    px, py = x[..., None] / d, y[..., None] / d
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), -1).flatten(2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), -1).flatten(2)
    return torch.cat((py, px), -1).permute(2, 0, 1)


def ajoute_position(f: torch.Tensor, splits: int, r: Recette) -> torch.Tensor:
    """La position de la fenêtre ajoutée à chaque fenêtre (épilogue du noyau qui écrit f) ; f [B, C, H, W]."""
    _, c, h, w = f.shape
    pos = stocke(position(h // splits, w // splits, c), r).repeat(1, splits, splits)
    return stocke(f + pos, r)


def fenetres(x: torch.Tensor, s: int) -> torch.Tensor:
    """[B, H, W, C] -> [B·s·s, H/s · W/s, C] : ce que lit un bloc de l'attention."""
    b, h, w, c = x.shape
    return x.view(b, s, h // s, s, w // s, c).permute(0, 1, 3, 2, 4, 5).reshape(b * s * s, -1, c)


def defenetres(x: torch.Tensor, s: int, b: int, h: int, w: int) -> torch.Tensor:
    c = x.shape[-1]
    return x.view(b, s, s, h // s, w // s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def masque_decale(h: int, w: int, s: int) -> torch.Tensor:
    """Le masque -100 des fenêtres décalées (Swin) : régions comparées, une seule fois par échelle. Le natif le
    recalcule à la volée depuis les indices de ligne et de colonne, sans le lire."""
    wh, ww = h // s, w // s
    ry = (torch.arange(h) >= h - wh).long() + (torch.arange(h) >= h - wh // 2).long()
    rx = (torch.arange(w) >= w - ww).long() + (torch.arange(w) >= w - ww // 2).long()
    reg = (ry.view(h, 1) * 3 + rx.view(1, w)).float().view(1, h, w, 1)
    reg = fenetres(reg, s).squeeze(-1)
    diff = reg.unsqueeze(1) - reg.unsqueeze(2)
    return torch.where(diff != 0, torch.tensor(-100.0), torch.tensor(0.0))


def attention_tuiles(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, masque: torch.Tensor | None,
                     r: Recette, p16: bool) -> torch.Tensor:
    """softmax(q kᵀ / √C + masque) v, à la FlashAttention : S d'une tuile de clés en f32 (entrées FP16), maximum
    et somme courants, P arrondi en FP16 si p16, sortie divisée à la fin. q [B, L, C], k [B, M, C], v [B, M, D]."""
    echelle = q.shape[-1] ** -0.5
    if r.tuile <= 0:
        s = q @ k.transpose(1, 2) * echelle
        if masque is not None:
            s = s + masque
        return torch.softmax(s, -1) @ v
    b, n, _ = q.shape
    m = torch.full((b, n, 1), -float("inf"))
    l = torch.zeros(b, n, 1)
    o = torch.zeros(b, n, v.shape[-1])
    for j in range(0, k.shape[1], r.tuile):
        s = q @ k[:, j : j + r.tuile].transpose(1, 2) * echelle
        if masque is not None:
            s = s + masque[:, :, j : j + r.tuile]
        m_neuf = torch.maximum(m, s.amax(-1, keepdim=True))
        alpha = torch.exp(m - m_neuf)
        p = torch.exp(s - m_neuf)
        l = l * alpha + p.sum(-1, keepdim=True)
        pv = p.half().float() if p16 else p
        o = o * alpha + pv @ v[:, j : j + r.tuile]
        m = m_neuf
    return o / l


def couche_layernorm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """LayerNorm sur les 128 canaux d'un jeton, en f32 dans l'épilogue (la ligne entière est dans le bloc)."""
    return F.layer_norm(x, (x.shape[-1],), g, b, 1e-5)


def couche(src: torch.Tensor, tgt: torch.Tensor, sd: dict, p: str, splits: int, decale: bool, h: int, w: int,
           masque: torch.Tensor | None, r: Recette, ffn: bool) -> torch.Tensor:
    """TransformerLayer : projections (une GEMM QKV), attention par fenêtres fondue, fusion + LayerNorm, puis MLP
    fondu (256 -> 1024 -> 128, GELU, LayerNorm) ; src, tgt [B, H·W, C]."""
    b, _, c = src.shape
    acc, g8 = r.gemm_acc16, r.gemm8
    q = stocke(gemm(src, sd[p + "q_proj.weight"], r, p + "qkv", acc, g8), r)
    k = stocke(gemm(tgt, sd[p + "k_proj.weight"], r, p + "qkv", acc, g8), r)
    v = stocke(gemm(tgt, sd[p + "v_proj.weight"], r, p + "qkv", acc, g8), r)
    if splits > 1:
        sh, sw = (h // splits // 2, w // splits // 2) if decale else (0, 0)
        q, k, v = (fenetres(torch.roll(t.view(b, h, w, c), (-sh, -sw), (1, 2)), splits) for t in (q, k, v))
        mq = masque.repeat(b, 1, 1) if decale else None
        o = attention_tuiles(q, k, v, mq, r, r.p16)
        o = torch.roll(defenetres(o, splits, b, h, w), (sh, sw), (1, 2)).reshape(b, h * w, c)
    else:
        o = attention_tuiles(q, k, v, None, r, r.p16)
    o = stocke(o, r)  # sortie de l'attention, entrée de la fusion (dans le même noyau si la fusion y est)
    msg = couche_layernorm(gemm(o, sd[p + "merge.weight"], r, p + "merge", acc, g8), sd[p + "norm1.weight"],
                           sd[p + "norm1.bias"])
    msg = stocke(msg, r)
    if ffn:
        cat = torch.cat([src, msg], -1)
        hid = stocke(F.gelu(gemm(cat, sd[p + "mlp.0.weight"], r, p + "mlp1", acc, g8)), r)
        msg = couche_layernorm(gemm(hid, sd[p + "mlp.2.weight"], r, p + "mlp2", acc, g8), sd[p + "norm2.weight"],
                               sd[p + "norm2.bias"])
    return stocke(src + msg, r)


def transformeur(f0: torch.Tensor, f1: torch.Tensor, sd: dict, splits: int, r: Recette):
    b, c, h, w = f0.shape
    a0 = torch.cat([f0, f1], 0).flatten(2).transpose(1, 2)
    masque = masque_decale(h, w, splits) if splits > 1 else None
    for i in range(6):
        p = f"transformer.layers.{i}."
        decale = i % 2 == 1
        # La cible de l'attention croisée est l'autre moitié du lot telle qu'elle entrait dans le bloc.
        a1 = torch.cat(a0.chunk(2, 0)[::-1], 0)
        a0 = couche(a0, a0, sd, p + "self_attn.", splits, decale, h, w, masque, r, ffn=False)
        a0 = couche(a0, a1, sd, p + "cross_attn_ffn.", splits, decale, h, w, masque, r, ffn=True)
    g0, g1 = a0.chunk(2, 0)
    return (g0.transpose(1, 2).reshape(b, c, h, w), g1.transpose(1, 2).reshape(b, c, h, w))


# --- appariement, propagation, agrandissement ----------------------------------------------------------------


def grille(b: int, h: int, w: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return torch.stack([xs, ys], 0).unsqueeze(0).expand(b, 2, h, w)


def softmax_valeurs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, r: Recette) -> torch.Tensor:
    """softmax(q kᵀ/√C) · v quand v est un flot ou des coordonnées (2 canaux) : P exact et v en f32 (cœurs CUDA) ou
    P et v en FP16 (tensor cores, flot_pv = f16)."""
    if r.flot_pv == "f16":
        return attention_tuiles(q, k, v.half().float(), None, r, True)
    return attention_tuiles(q, k, v, None, r, False)


def appariement_global(f0: torch.Tensor, f1: torch.Tensor, r: Recette) -> torch.Tensor:
    """global_correlation_softmax bidirectionnel : le softmax des lignes de f0·f1ᵀ puis celui de ses colonnes,
    chacun par un passage fondu (la corrélation recalculée, jamais stockée) ; rend [2B, 2, H, W]."""
    b, c, h, w = f0.shape
    t0, t1 = f0.flatten(2).transpose(1, 2), f1.flatten(2).transpose(1, 2)
    coords = grille(b, h, w).flatten(2).transpose(1, 2)
    aller = softmax_valeurs(t0, t1, coords, r)
    retour = softmax_valeurs(t1, t0, coords, r)
    corr = torch.cat([aller, retour], 0).transpose(1, 2).reshape(2 * b, 2, h, w)
    return corr - grille(2 * b, h, w)


def appariement_local(f0: torch.Tensor, f1: torch.Tensor, rayon: int) -> torch.Tensor:
    """local_correlation_softmax : (2r+1)² produits par pixel, hors image masqués ; un noyau à tuile + halo."""
    b, c, h, w = f0.shape
    cote = 2 * rayon + 1
    pad = F.pad(f1, (rayon,) * 4)
    corr = torch.stack([(f0 * pad[:, :, dy : dy + h, dx : dx + w]).sum(1) for dy in range(cote) for dx in range(cote)],
                       -1) / c**0.5
    dys, dxs = torch.meshgrid(torch.arange(-rayon, rayon + 1), torch.arange(-rayon, rayon + 1), indexing="ij")
    ys, xs = torch.arange(h).view(h, 1, 1), torch.arange(w).view(1, w, 1)
    ok = ((ys + dys.reshape(1, 1, -1)) >= 0) & ((ys + dys.reshape(1, 1, -1)) < h)
    ok = ok & ((xs + dxs.reshape(1, 1, -1)) >= 0) & ((xs + dxs.reshape(1, 1, -1)) < w)
    corr = corr.masked_fill(~ok.unsqueeze(0), -1e9)
    p = torch.softmax(corr, -1)
    off = torch.stack([dxs.reshape(-1), dys.reshape(-1)], 0).float()  # [2, 81]
    return torch.einsum("bhwk,ck->bchw", p, off)


def deforme(f: torch.Tensor, flot: torch.Tensor, r: Recette) -> torch.Tensor:
    """flow_warp : lecture bilinéaire de f en (pixel + flot), zéros hors de l'image ; grille en f32."""
    b, c, h, w = f.shape
    g = grille(b, h, w) + flot
    gx = 2 * g[:, 0] / (w - 1) - 1
    gy = 2 * g[:, 1] / (h - 1) - 1
    out = F.grid_sample(f, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros", align_corners=True)
    return stocke(out, r)


def propagation(f: torch.Tensor, flot: torch.Tensor, sd: dict, rayon: int, r: Recette) -> torch.Tensor:
    """FeatureFlowAttention : globale (rayon < 0 ; k = k_proj(q_proj(f)), les deux matrices fondues en une
    au chargement) ou fenêtre 3 × 3 (k = k_proj(f), zéros hors de l'image, non masqués)."""
    b, c, h, w = f.shape
    p = "feature_flow_attn."
    t = f.flatten(2).transpose(1, 2)
    wq, bq, wk, bk = (sd[p + n] for n in ("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias"))
    q = stocke(gemm(t, wq, r, p + "q", r.gemm_acc16, False) + bq, r)
    if rayon < 0:
        wkq, bkq = wk @ wq, wk @ bq + bk
        k = stocke(gemm(t, wkq, r, p + "kq", r.gemm_acc16, False) + bkq, r)
        v = flot.flatten(2).transpose(1, 2)
        return softmax_valeurs(q, k, v, r).transpose(1, 2).reshape(b, 2, h, w)
    k = stocke(gemm(t, wk, r, p + "k", r.gemm_acc16, False) + bk, r).transpose(1, 2).reshape(b, c, h, w)
    cote = 2 * rayon + 1
    kw = F.unfold(k, cote, padding=rayon).view(b, c, cote * cote, h * w)
    fw = F.unfold(flot, cote, padding=rayon).view(b, 2, cote * cote, h * w)
    s = torch.einsum("blc,bckl->blk", q, kw) / c**0.5
    pr = torch.softmax(s, -1)
    return torch.einsum("blk,bckl->bcl", pr, fw).reshape(b, 2, h, w)


def agrandit(flot: torch.Tensor, f: torch.Tensor, sd: dict, r: Recette, facteur: int = 4) -> torch.Tensor:
    """Agrandissement convexe ×4 : conv 3 × 3 (130 -> 256, Relu) puis 1 × 1 (256 -> 144), softmax sur 9 voisins par
    sous-pixel, somme pondérée du flot ×4 (zéros hors de l'image). Le flot reste en f32 hors de la convolution."""
    b, _, h, w = flot.shape
    x = torch.cat([stocke(flot, r), f], 1)
    m = stocke(F.relu(conv(x, sd["upsampler.0.weight"], sd["upsampler.0.bias"], 1, 1, r, "up0")), r)
    m = conv(m, sd["upsampler.2.weight"], sd["upsampler.2.bias"], 1, 0, r, "up2")
    m = torch.softmax(stocke(m, r).view(b, 1, 9, facteur, facteur, h, w), 2)
    up = F.unfold(facteur * flot, [3, 3], padding=1).view(b, 2, 9, 1, 1, h, w)
    up = (m * up).sum(2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(b, 2, facteur * h, facteur * w)


MOYENNE = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
ECART = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.inference_mode()
def gmflow(img0: torch.Tensor, img1: torch.Tensor, sd: dict, r: Recette) -> torch.Tensor:
    """GMFlow(img0, img1, pred_bidir_flow=True) découpé : rend [2B, 2, H, W] (0 -> 1 puis 1 -> 0), en f32."""
    x = stocke((torch.cat([img0, img1], 0) - MOYENNE) / ECART, r)
    grossier, fin = squelette(x, sd, r)
    b = img0.shape[0]
    # Échelle 1/8 : fenêtres 2 × 2, appariement global, propagation globale.
    f0, f1 = grossier[:b], grossier[b:]
    f0, f1 = ajoute_position(f0, 2, r), ajoute_position(f1, 2, r)
    f0, f1 = transformeur(f0, f1, sd, 2, r)
    flot = appariement_global(f0, f1, r)
    flot = propagation(torch.cat([f0, f1], 0), flot, sd, -1, r)
    # Échelle 1/4 : les deux sens en lot, f1 déformé par le flot ×2, fenêtres 8 × 8, appariement local r = 4.
    f0, f1 = torch.cat([fin[:b], fin[b:]], 0), torch.cat([fin[b:], fin[:b]], 0)
    flot = F.interpolate(flot, scale_factor=2, mode="bilinear", align_corners=True) * 2
    f1 = deforme(f1, flot, r)
    f0, f1 = ajoute_position(f0, 8, r), ajoute_position(f1, 8, r)
    f0, f1 = transformeur(f0, f1, sd, 8, r)
    flot = flot + appariement_local(f0, f1, 4)
    flot = propagation(f0, flot, sd, 1, r)
    return agrandit(flot, f0, sd, r)
