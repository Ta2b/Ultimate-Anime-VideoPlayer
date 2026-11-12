"""Mesures du banc de qualité : une image d'essai contre sa référence, toutes deux en RGB 3 × H × W dans [0, 1].

  psnr          dB sur les trois plans ; 200 quand les images sont identiques ;
  ssim          luminance BT.709, fenêtres 8 × 8 disjointes ;
  ecarts        écart absolu en codes 8 bits (255 × |a − b|) : moyen, 99,9e centile, pire ;
  delta_itp     ΔE_ITP (BT.2124) d'une image SDR affichée à 203 cd/m² (BT.2408) : moyen et 99,9e centile ;
                1 est le seuil de l'écart tout juste visible.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from hdr import BT2020_TO_LMS, BT709_TO_2020, LMS_TO_ICTCP, nits_to_pq  # noqa: E402

SDR_WHITE = 203.0  # cd/m², blanc de référence d'une image SDR (BT.2408)


def psnr(ref, test):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 200.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def luma(x):
    x = x.astype(np.float64)
    return x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722


def ssim(ref, test):
    """SSIM par blocs 8×8 disjoints de la luminance BT.709. Quatre définitions coexistent (B46-09) : blocs 8×8 (mesures.py, interp_fp8.py), gaussien 11 σ 1,5 (quantize_span.py), 7×7 uniforme par canal (commun.py) ; chaque seuil a été fixé avec la sienne, ne pas les comparer d'un outil à l'autre."""
    a, b = luma(ref), luma(test)
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8

    def blocks(x):
        return x[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)

    a, b = blocks(a), blocks(b)
    ma, mb, va, vb = a.mean(1), b.mean(1), a.var(1), b.var(1)
    cov = ((a - ma[:, None]) * (b - mb[:, None])).mean(1)
    c1, c2 = 0.01**2, 0.03**2
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))))


def ecarts(ref, test):
    d = 255.0 * np.abs(ref.astype(np.float64) - test.astype(np.float64))
    return {"moyen": float(d.mean()), "p999": float(np.percentile(d, 99.9)), "pire": float(d.max())}


def ictcp(rgb):
    """RGB BT.709 gamma 2,4 (3 × H × W) vers ICtCp, le blanc à SDR_WHITE."""
    linear = np.clip(rgb.astype(np.float64), 0.0, 1.0) ** 2.4 * SDR_WHITE
    lms = np.einsum("ij,jhw->ihw", BT2020_TO_LMS @ BT709_TO_2020, linear)
    return np.einsum("ij,jhw->ihw", LMS_TO_ICTCP, nits_to_pq(lms))


def delta_itp(ref, test):
    a, b = ictcp(ref), ictcp(test)
    d = 720.0 * np.sqrt((a[0] - b[0]) ** 2 + 0.25 * (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return {"moyen": float(d.mean()), "p999": float(np.percentile(d, 99.9))}


def compare(ref, test):
    """Toutes les mesures d'une image contre sa référence, à plat."""
    e, d = ecarts(ref, test), delta_itp(ref, test)
    return {"psnr": psnr(ref, test), "ssim": ssim(ref, test), "ecart_moyen": e["moyen"], "ecart_p999": e["p999"],
            "ecart_pire": e["pire"], "itp_moyen": d["moyen"], "itp_p999": d["p999"]}
