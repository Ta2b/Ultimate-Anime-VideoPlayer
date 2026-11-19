"""Recalage d'une paire d'images par un flux, pour la mesure du scintillement (scintillement.py).

  luma      luminance BT.709 en niveaux 8 bits ;
  Sampler   lecture bilinéaire aux mêmes points de plusieurs images (indices et poids calculés une fois) ;
  scale     agrandissement bilinéaire d'un champ (un flux, la grille NVOF), axe par axe ;
  Aligned   une paire recalée vers son milieu, son masque « dans l'image », et la correction sous-pixel d'ensemble
            (Lucas-Kanade global) qui rattrape un panoramique lent que le flux n'a pas vu.
"""
import functools

import numpy as np


def luma(rgb):
    """Luminance BT.709 en niveaux 8 bits, d'une image H × W × 3 en uint8 ou uint16 ou 3 × H × W dans [0, 1]."""
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        y = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return y / 257.0 if rgb.dtype == np.uint16 else y
    return 255.0 * (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def box(img, r=1):
    """Moyenne (2r + 1) × (2r + 1), bords répétés."""
    p = np.pad(img, r, mode="edge").astype(np.float64)
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * r + 1
    h, w = img.shape
    return ((c[n:n + h, n:n + w] - c[:h, n:n + w] - c[n:n + h, :w] + c[:h, :w]) / (n * n)).astype(np.float32)


class Sampler:
    """Lecture bilinéaire aux points (x, y) d'images H × W, bords répétés. Les indices et les poids se calculent une
    fois, puis servent à chaque image lue aux mêmes points (la source et toutes les sorties)."""

    def __init__(self, x, y, h, w):
        x, y = np.clip(x, 0, w - 1).astype(np.float32), np.clip(y, 0, h - 1).astype(np.float32)
        x0, y0 = np.minimum(x.astype(np.int32), max(w - 2, 0)), np.minimum(y.astype(np.int32), max(h - 2, 0))
        self.ax = (x - x0.astype(np.float32)).ravel()
        self.ay = (y - y0.astype(np.float32)).ravel()
        self.idx = (y0.astype(np.intp) * w + x0).ravel()
        self.shape, self.dx, self.dy = x.shape, int(w > 1), w if h > 1 else 0

    def __call__(self, img):
        f = np.ascontiguousarray(img, np.float32).ravel()
        # Les quatre voisins : le même tableau d'indices dans l'image décalée d'un pixel, d'une ligne, ou des deux.
        p00, p01 = f.take(self.idx), f[self.dx:].take(self.idx)
        p10, p11 = f[self.dy:].take(self.idx), f[self.dy + self.dx:].take(self.idx)
        top = p00 + (p01 - p00) * self.ax
        return (top + (p10 + (p11 - p10) * self.ax - top) * self.ay).reshape(self.shape)


def sample(img, x, y):
    """Lecture bilinéaire de img (H × W) aux points (x, y), bords répétés."""
    return Sampler(x, y, *img.shape)(img)


@functools.lru_cache(maxsize=4)
def grid(h, w):
    """Les coordonnées (x, y) de chaque pixel, en lecture seule (partagées entre les paires)."""
    g = np.mgrid[0:h, 0:w].astype(np.float32)[::-1]
    g.setflags(write=False)
    return g


def _axis(n_out, n_in, step):
    pos = (np.arange(n_out, dtype=np.float64) + 0.5) * (1.0 / step if step else n_in / n_out) - 0.5
    pos = np.clip(pos, 0, n_in - 1)
    i0 = np.minimum(pos.astype(np.intp), max(n_in - 2, 0))
    return i0, np.minimum(i0 + 1, n_in - 1), (pos - i0).astype(np.float32)


def scale(field, w, h, step=None):
    """Un champ C × h0 × w0 au bilinéaire en C × h × w (centres demi-pixel), axe par axe. Un flux garde ses unités :
    à multiplier. step : un point du champ pour step × step pixels (la grille NVOF, dont le dernier bloc d'une taille
    qui n'en est pas un multiple déborde de l'image) ; sinon le champ couvre l'image."""
    _, h0, w0 = field.shape
    x0, x1, fx = _axis(w, w0, step)
    y0, y1, fy = _axis(h, h0, step)
    rows = field[:, :, x0] * (1 - fx) + field[:, :, x1] * fx
    return (rows[:, y0, :] * (1 - fy)[:, None] + rows[:, y1, :] * fy[:, None]).astype(np.float32)


class Aligned:
    """Une paire recalée : warp(N, fa) face à warp(N + 1, fb) ; fb absent vaut l'identité."""

    def __init__(self, fa, fb, coherent=None):
        _, h, w = fa.shape
        gx, gy = grid(h, w)
        self.a = (gx + fa[0], gy + fa[1])
        self.b = (gx + fb[0], gy + fb[1]) if fb is not None else None
        self.coherent = coherent
        self.readers = None
        self._bounds()

    def _bounds(self):
        h, w = self.a[0].shape
        inside = np.ones((h, w), bool)
        for x, y in (self.a, self.b) if self.b else (self.a,):
            inside &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        self.inside = inside if self.coherent is None else inside & self.coherent

    def pair(self, ya, yb):
        if self.readers is None:
            h, w = self.a[0].shape
            self.readers = (Sampler(*self.a, h, w), Sampler(*self.b, h, w) if self.b else None)
        sa, sb = self.readers
        return sa(ya), (sb(yb) if sb else yb)

    def refine(self, ya, yb, steps=4, step=2):
        """Corrige le flux d'une translation d'ensemble sous-pixel, par Lucas-Kanade global sur la source (un point sur
        step × step), chaque image prenant la moitié de la correction. RIFE ignore les mouvements de moins d'un pixel
        par image et NVOF est au quart de pixel : sur un panoramique lent, le reste du décalage compterait sinon comme
        du scintillement. Rend la translation ajoutée (x, y)."""
        total = np.zeros(2)
        for _ in range(steps):
            sub = (slice(None, None, step), slice(None, None, step))
            ra = sample(ya, self.a[0][sub], self.a[1][sub])
            rb = sample(yb, self.b[0][sub], self.b[1][sub]) if self.b else yb[sub]
            r = ra - rb
            gy, gx = np.gradient(0.5 * (ra + rb), step)
            m = self.inside[sub] & (np.abs(r) < 30.0)  # les occlusions et les coupes ne tirent pas la solution
            gx, gy, r = gx[m], gy[m], r[m]
            mat = np.array([[gx @ gx, gx @ gy], [gx @ gy, gy @ gy]], np.float64)
            if r.size < 64 or np.linalg.det(mat) < 1e-6 * max(1.0, np.trace(mat)) ** 2:
                break  # image unie ou trait dans un seul sens : la translation n'est pas déterminée
            d = np.clip(-np.linalg.solve(mat, np.array([gx @ r, gy @ r], np.float64)), -2.0, 2.0)
            self.a = (self.a[0] + 0.5 * d[0], self.a[1] + 0.5 * d[1])
            gxf, gyf = (self.b if self.b else grid(*ya.shape))
            self.b = (gxf - 0.5 * d[0], gyf - 0.5 * d[1])
            total += d
            if np.hypot(*d) < 0.01:
                break
        self.readers = None
        self._bounds()
        return tuple(total)
