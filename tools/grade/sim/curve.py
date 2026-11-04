#!/usr/bin/env python3
"""La courbe par plan du contrat v2 (src/render/curve.cpp), portée en Python pour la simulation (chantier 8.11).

Même calcul que le lecteur, pas à pas : décalage E (pente de l'APL du plan contre la série, plus celui de
l'utilisateur, |E| ≤ 0,25), gain constant sous le genou 0,75 × 203, Hermite monotone en PQ jusqu'à
T = min(C_pk, T_aire, T_ABL), budget ABL par bissection, pied (contraste sous 20 nits) et vibrance hors
des peaux. simulate.py s'en sert ; `curve.py --self-test` compare à des valeurs relevées sur la version C++.
"""
import math
import sys
from dataclasses import dataclass, field

M1 = 2610.0 / 16384.0
M2 = 2523.0 / 4096.0 * 128.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0
MAX_E = 0.25
SLOPE = 0.25
MAX_TOE = 0.2
TOE_NITS = 20.0
MAX_VIBRANCE = 1.15
HIST_BINS = 32


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pq(nits):
    y = (clamp(nits, 0.0, 10000.0) / 10000.0) ** M1
    return ((C1 + C2 * y) / (1.0 + C3 * y)) ** M2


def nits(code):
    p = clamp(code, 0.0, 1.0) ** (1.0 / M2)
    return 10000.0 * (max(p - C1, 0.0) / (C2 - C3 * p)) ** (1.0 / M1)


@dataclass
class CurveInput:
    shot_apl_nits: float = 0.0
    series_apl_nits: float = 0.0
    user_e: float = 0.0
    content_max_nits: float = 1000.0
    panel_peak_nits: float = 1000.0
    area_limit_nits: float = 0.0
    abl_limit_nits: float = 0.0
    apl_budget_nits: float = 0.0
    lum_hist: list = field(default_factory=lambda: [0.0] * HIST_BINS)
    vibrance: float = 1.0
    toe: float = 0.0


@dataclass
class ShotCurve:
    e: float = 0.0
    gain: float = 1.0
    knee_in: float = 0.0
    knee_out: float = 0.0
    top_in: float = 0.0
    top_out: float = 0.0
    m0: float = 0.0
    m1: float = 0.0
    vibrance: float = 1.0
    toe: float = 0.0


def _shape(c):
    x0, y0 = pq(c.knee_in), pq(c.knee_out)
    x1, y1 = pq(c.top_in), pq(c.top_out)
    h = 1e-4
    slope0 = (pq(nits(x0 + h) * c.gain) - pq(nits(x0 - h) * c.gain)) / (2.0 * h)
    d = (y1 - y0) / (x1 - x0)
    c.m0 = 0.0 if d <= 0.0 else min(slope0, 3.0 * d)
    c.m1 = 0.0


def _body(c, n):
    if n < 0.5:
        t = clamp((n - 0.1) / 0.4, 0.0, 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return n * (1.0 + (c.gain - 1.0) * s)
    if n <= c.knee_in:
        return n * c.gain
    if n >= c.top_in:
        return c.top_out
    x0, x1 = pq(c.knee_in), pq(c.top_in)
    y0, y1 = pq(c.knee_out), pq(c.top_out)
    w = x1 - x0
    t = (pq(n) - x0) / w
    t2, t3 = t * t, t * t * t
    y = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * w * c.m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * w * c.m1
    return nits(y)


def map_luminance(c, n):
    if n <= 0.0:
        return 0.0
    out = _body(c, n)
    if c.toe == 0.0 or n >= TOE_NITS:
        return out
    return out * (n / TOE_NITS) ** c.toe


def predicted_apl(c, hist):
    weight = total = 0.0
    for i, part in enumerate(hist):
        if part <= 0.0:
            continue
        weight += part
        total += part * map_luminance(c, nits((i + 0.5) / HIST_BINS))
    return total / weight if weight > 0.0 else 0.0


def plan_curve(i):
    c = ShotCurve()
    e = i.user_e
    if i.series_apl_nits > 0.0 and i.shot_apl_nits > 0.0:
        e -= SLOPE * math.log2(i.shot_apl_nits / i.series_apl_nits)
    c.e = clamp(e, -MAX_E, MAX_E)
    c.gain = 2.0 ** c.e
    c.vibrance = clamp(i.vibrance, 1.0, MAX_VIBRANCE)
    c.toe = clamp(i.toe, -MAX_TOE, MAX_TOE)
    c.knee_in = 0.75 * 203.0
    c.knee_out = c.knee_in * c.gain
    top = i.panel_peak_nits
    if i.area_limit_nits > 0.0:
        top = min(top, i.area_limit_nits)
    if i.abl_limit_nits > 0.0:
        top = min(top, i.abl_limit_nits)
    c.top_in = max(i.content_max_nits, c.knee_in * 1.01)
    floor_top = c.knee_out * 1.05
    c.top_out = max(top, floor_top)
    _shape(c)
    if i.apl_budget_nits > 0.0 and predicted_apl(c, i.lum_hist) > i.apl_budget_nits:
        lo, hi = floor_top, c.top_out
        for _ in range(40):
            c.top_out = 0.5 * (lo + hi)
            _shape(c)
            if predicted_apl(c, i.lum_hist) > i.apl_budget_nits:
                hi = c.top_out
            else:
                lo = c.top_out
        c.top_out = lo
        _shape(c)
    return c


# Valeurs relevées sur render::plan_curve puis render::map_luminance (C++), pour --self-test :
# (APL du plan, APL de la série, E, maximum du contenu, C_pk, pied, budget ABL, case pleine de l'histogramme
# ou −1 pour un histogramme plat) → sorties à _NITS.
_NITS = (0.3, 5.0, 100.0, 180.0, 400.0, 900.0, 3000.0)
_REFERENCE = [
    ((40, 20, 0, 1000, 800, 0, 0, -1),
     [0.276134462, 4.20448208, 84.0896415, 153.2068, 415.551155, 791.087199, 800]),
    ((10, 20, 0.1, 4000, 600, 0.15, 0, -1),
     [0.174900801, 4.82968164, 118.920712, 212.4799, 377.847286, 519.037669, 598.42519]),
    ((20, 20, 0, 1000, 1000, -0.2, 150, 24),
     [0.694869105, 6.59753955, 100, 154.021822, 158.927243, 159.861048, 159.8625]),
    ((80, 25, -0.1, 2000, 700, 0.05, 120, 20),
     [0.223833546, 3.92292049, 84.0896415, 129.134186, 132.78291, 134.223581, 134.427803]),
]


def self_test():
    flat = [1.0 / HIST_BINS] * HIST_BINS
    failures = []

    def check(ok, what):
        if not ok:
            failures.append(what)

    for (apl, series, e, top, peak, toe, budget, full), want in _REFERENCE:
        hist = [1.0 / HIST_BINS] * HIST_BINS if full < 0 else [0.0] * HIST_BINS
        if full >= 0:
            hist[full] = 1.0
        c = plan_curve(CurveInput(shot_apl_nits=apl, series_apl_nits=series, user_e=e, content_max_nits=top,
                                  panel_peak_nits=peak, toe=toe, apl_budget_nits=budget, lum_hist=hist))
        for n, w in zip(_NITS, want):
            got = map_luminance(c, n)
            check(abs(got - w) <= 1e-4 * max(1.0, w), f"C++ {apl}/{series}/{e} à {n} nits : {got:.6g} contre {w:.6g}")
    # Neutre : E = 0, gain 1 sous le genou, le maximum du contenu tombe sur C_pk.
    c = plan_curve(CurveInput(panel_peak_nits=800.0, lum_hist=flat))
    check(abs(map_luminance(c, 100.0) - 100.0) < 1e-9, "neutre sous le genou")
    check(abs(map_luminance(c, 1000.0) - 800.0) < 1e-6, "maximum sur C_pk")
    # Un plan deux fois plus clair que la série : E = −0,25.
    c = plan_curve(CurveInput(shot_apl_nits=40.0, series_apl_nits=20.0, lum_hist=flat))
    check(abs(c.e + 0.25) < 1e-12, "pente de l'APL")
    # Monotone partout, pied compris.
    for toe in (-MAX_TOE, 0.0, MAX_TOE):
        c = plan_curve(CurveInput(panel_peak_nits=600.0, content_max_nits=4000.0, toe=toe, lum_hist=flat))
        last, n = 0.0, 0.01
        while n < 4000.0:
            out = map_luminance(c, n)
            check(out > last, f"monotone (pied {toe}) à {n:.2f} nits")
            last, n = out, n * 1.05
        check(map_luminance(c, 4000.0) <= 600.0 * 1.005, f"pic ≤ C_pk (pied {toe})")
    # Budget ABL tenu.
    bright = [0.0] * HIST_BINS
    bright[24] = 1.0
    c = plan_curve(CurveInput(panel_peak_nits=1000.0, apl_budget_nits=150.0, lum_hist=bright))
    check(predicted_apl(c, bright) <= 150.0 * 1.001 or c.top_out <= c.knee_out * 1.05 + 1e-6, "budget ABL")
    for f in failures:
        print("échec :", f)
    print("curve.py : " + ("ok" if not failures else f"{len(failures)} échecs"))
    return 0 if not failures else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    print(__doc__)
