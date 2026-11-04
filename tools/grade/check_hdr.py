#!/usr/bin/env python3
"""Les portes HDR du contrat v2 sur des instantanés PQ (chantier 8.11).

Les instantanés viennent de `ajp play … --snapshot=NOM.pq.ppm --snapshot-at=S` sur une sortie HDR10
(PPM 16 bits, PQ BT.2020, nits affichés ; compare_shots.sh pour les 30 plans de shots30.tsv). Une suite
NOM-001.pq.ppm, NOM-002.pq.ppm… (--snapshot-count) est un plan vu image par image. Critères, par plan :

  pic        Y max ≤ C_pk + 0,5 %                           (--peak, nits affichés)
  APL        Y moyen ≤ 0,9 × MaxFALL                       (--max-fall, nits affichés)
  ΔE_ITP     sous K contre la référence BT.2408 : moyenne ≤ 3, p95 ≤ 6         (--ref DOSSIER, --k)
  teinte     |Δh| moyen ≤ 2°, ≤ 1° sur la peau (pixels assez colorés)          (avec --ref)
  peau       |écart de lumière| ≤ 0,5 diaphragme                                (avec --ref)
  stabilité  σ de l'APL dans un plan fixe ≤ 0,5 %          (suites nommées par --static)
  coupes     0 image d'erreur : une image qui ne ressemble ni à la précédente ni aux deux suivantes, elles
             d'accord entre elles (la première image d'un plan montrée avec la courbe d'un autre)
  CVVDP      ≥ −0,1 JOD contre la référence, si pycvvdp est installé (--cvvdp) ; sinon « non mesuré »

La référence BT.2408 : les mêmes plans rendus sans grade (graphiques à 203 nits), en nits du contenu ; K est
le facteur du contrat entre contenu et écran (DisplayContract::k), le candidat est divisé par K avant la
comparaison. Code retour 1 si un critère manque.

  check_hdr.py CANDIDATS --peak 766 --max-fall 230 [--ref REF --k 1.0] [--static plan-07,plan-12] [--json]
  check_hdr.py --self-test
"""
import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # de_itp.py, à côté (python -I ne l'ajoute pas)
from de_itp import itp, linear_to_pq, pq_to_linear, read_ppm16  # noqa: E402

LUMA = np.array([0.2627, 0.6780, 0.0593])
PEAK_SLACK = 1.005
BUDGET_SHARE = 0.9
MAX_DE_MEAN, MAX_DE_P95 = 3.0, 6.0
MAX_HUE, MAX_HUE_SKIN = 2.0, 1.0
MAX_SKIN_STOPS = 0.5
MAX_APL_SIGMA = 0.005
CUT_GAP, CUT_AGREE = 0.02, 0.005
MIN_CVVDP = -0.1
CHROMA_MIN = 0.02  # sous ce chroma ICtCp (T = Ct / 2), la teinte ne se lit pas
SNAPSHOT = re.compile(r"^(?P<name>.+?)(?:-(?P<index>\d{3}))?\.pq\.ppm$")


def skin_hue():
    """La teinte ICtCp de la peau sRGB (255, 224, 196) à 100 nits, comme src/render/curve.cpp."""
    srgb = np.array([255.0, 224.0, 196.0]) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    to2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])
    codes = linear_to_pq(100.0 * (to2020 @ lin) / 10000.0)
    _, t, p = itp(codes[None, :])[0]
    return float(np.arctan2(p, 2.0 * t))


def nits_of(codes):
    return 10000.0 * pq_to_linear(codes)


def luminance(codes):
    return nits_of(codes) @ LUMA


def scaled(codes, k):
    return linear_to_pq(pq_to_linear(codes) / k) if k != 1.0 else codes


def hue_and_chroma(codes):
    i_t_p = itp(codes)
    ct = 2.0 * i_t_p[..., 1]
    cp = i_t_p[..., 2]
    return np.arctan2(cp, ct), np.hypot(ct, cp)


def hue_gap(a, b):
    d = np.abs(a - b)
    return np.degrees(np.minimum(d, 2.0 * np.pi - d))


def skin_mask(codes):
    hue, chroma = hue_and_chroma(codes)
    y = luminance(codes)
    return (hue_gap(hue, skin_hue()) <= 15.0) & (chroma >= CHROMA_MIN) & (y > 5.0)


def frame_checks(cand, ref, args):
    out = {}
    y = luminance(cand)
    out["pic"] = float(y.max())
    out["apl"] = float(y.mean())
    out["ok_pic"] = out["pic"] <= args.peak * PEAK_SLACK
    out["ok_apl"] = out["apl"] <= BUDGET_SHARE * args.max_fall
    if ref is None:
        return out
    if ref.shape != cand.shape:
        raise ValueError("candidat et référence de tailles différentes")
    under = scaled(cand, args.k)
    de = 720.0 * np.linalg.norm(itp(under) - itp(ref), axis=-1)
    out["de_moyenne"] = float(de.mean())
    out["de_p95"] = float(np.percentile(de, 95))
    out["ok_de"] = out["de_moyenne"] <= MAX_DE_MEAN and out["de_p95"] <= MAX_DE_P95
    hue_c, chroma_c = hue_and_chroma(under)
    hue_r, chroma_r = hue_and_chroma(ref)
    colored = (chroma_r >= CHROMA_MIN) & (chroma_c >= CHROMA_MIN)
    skin = skin_mask(ref) & colored
    out["teinte"] = float(hue_gap(hue_c, hue_r)[colored].mean()) if colored.any() else 0.0
    out["teinte_peau"] = float(hue_gap(hue_c, hue_r)[skin].mean()) if skin.any() else 0.0
    y_c, y_r = luminance(under), luminance(ref)
    lit = skin & (y_r > 1.0) & (y_c > 0.0)
    out["peau_diaph"] = float(np.log2(y_c[lit] / y_r[lit]).mean()) if lit.any() else 0.0
    out["ok_teinte"] = out["teinte"] <= MAX_HUE and out["teinte_peau"] <= MAX_HUE_SKIN
    out["ok_peau"] = abs(out["peau_diaph"]) <= MAX_SKIN_STOPS
    return out


def cut_errors(apls):
    """Les images d'erreur aux coupes d'une suite : une image loin de la précédente et de la suivante, les deux
    suivantes d'accord entre elles (le plan nouveau s'est installé, sauf sur cette image)."""
    errors = []
    rel = lambda a, b: abs(a - b) / max(a, b, 1e-6)  # noqa: E731
    for i in range(1, len(apls) - 2):
        if rel(apls[i], apls[i - 1]) > CUT_GAP and rel(apls[i], apls[i + 1]) > CUT_GAP and \
                rel(apls[i + 1], apls[i + 2]) <= CUT_AGREE:
            errors.append(i)
    return errors


def cvvdp_score(cand_path, ref_path):
    try:
        import pycvvdp  # noqa: F401
    except ImportError:
        return None
    import pycvvdp
    metric = pycvvdp.cvvdp(display_name="standard_hdr_pq")
    cand = read_ppm16(cand_path).astype(np.float32)
    ref = read_ppm16(ref_path).astype(np.float32)
    jod, _ = metric.predict(cand, ref, dim_order="HWC")
    return float(jod) - 10.0  # écart à l'image de référence (10 JOD)


def group(folder):
    shots = defaultdict(list)
    for entry in sorted(os.listdir(folder)):
        m = SNAPSHOT.match(entry)
        if m:
            shots[m["name"]].append(os.path.join(folder, entry))
    return shots


def check(args):
    static = set(filter(None, args.static.split(",")))
    results = {}
    failed = 0
    for name, paths in sorted(group(args.candidates).items()):
        frames = []
        for path in paths:
            ref = None
            if args.ref:
                ref_path = os.path.join(args.ref, os.path.basename(path))
                if not os.path.exists(ref_path):  # une seule image de référence pour la suite
                    ref_path = os.path.join(args.ref, f"{name}.pq.ppm")
                ref = read_ppm16(ref_path) if os.path.exists(ref_path) else None
            frames.append(frame_checks(read_ppm16(path), ref, args))
        apls = [f["apl"] for f in frames]
        row = {k: max(f[k] for f in frames) for k in frames[0] if not k.startswith("ok_")}
        row.update({k: all(f[k] for f in frames) for k in frames[0] if k.startswith("ok_")})
        if len(apls) >= 3:
            sigma = float(np.std(apls) / max(np.mean(apls), 1e-6))
            row["apl_sigma"] = sigma
            if name in static:
                row["ok_stable"] = sigma <= MAX_APL_SIGMA
            errors = cut_errors(apls)
            row["coupes"] = len(errors)
            row["ok_coupes"] = not errors
        if args.cvvdp and args.ref:
            score = cvvdp_score(paths[0], os.path.join(args.ref, os.path.basename(paths[0])))
            row["cvvdp"] = "non mesuré" if score is None else score
            if score is not None:
                row["ok_cvvdp"] = score >= MIN_CVVDP
        row["verdict"] = all(v for k, v in row.items() if k.startswith("ok_"))
        failed += not row["verdict"]
        results[name] = row
    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for name, row in results.items():
            misses = [k[3:] for k, v in row.items() if k.startswith("ok_") and not v]
            values = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in row.items() if not k.startswith("ok_") and k != "verdict")
            print(f"{name}\t{'ok' if row['verdict'] else 'manque : ' + ', '.join(misses)}\t{values}")
        print(f"# {len(results) - failed}/{len(results)} plans passent")
    return 1 if failed or not results else 0


def write_ppm16(path, codes):
    h, w, _ = codes.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n65535\n".encode())
        f.write(np.round(np.clip(codes, 0.0, 1.0) * 65535.0).astype(">u2").tobytes())


def self_test():
    rng = np.random.default_rng(11)
    base = rng.uniform(1.0, 300.0, size=(48, 64, 1)) * np.array([1.0, 0.8, 0.6])  # tons chauds, nits
    base[:8, :8] = np.array([180.0, 140.0, 110.0])  # une plage de peau
    ref = linear_to_pq(base / 10000.0)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        def run(cand, **kw):
            cdir, rdir = os.path.join(tmp, "c"), os.path.join(tmp, "r")
            for d in (cdir, rdir):
                os.makedirs(d, exist_ok=True)
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
            for i, frame in enumerate(cand):
                write_ppm16(os.path.join(cdir, f"plan-{i + 1:03d}.pq.ppm"), frame)
            write_ppm16(os.path.join(rdir, "plan.pq.ppm"), ref)
            ns = argparse.Namespace(candidates=cdir, ref=rdir, k=1.0, peak=kw.get("peak", 1000.0),
                                    max_fall=kw.get("max_fall", 400.0), static="plan", json=True, cvvdp=False)
            saved = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                code = check(ns)
            finally:
                sys.stdout.close()
                sys.stdout = saved
            return code
        if run([ref] * 4) != 0:
            failures.append("identique : doit passer")
        if run([ref] * 4, peak=200.0) != 1:
            failures.append("pic au-dessus de C_pk : doit manquer")
        hot = linear_to_pq(base * np.array([1.0, 0.7, 0.5]) / 10000.0)
        if run([hot] * 4) != 1:
            failures.append("teinte tournée : doit manquer")
        bright = linear_to_pq(base * 1.6 / 10000.0)
        if run([ref, bright, ref, ref]) != 1:
            failures.append("image d'erreur à une coupe : doit manquer")
    for f in failures:
        print("échec :", f)
    print("check_hdr.py : " + ("ok" if not failures else f"{len(failures)} échecs"))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="?")
    ap.add_argument("--peak", type=float)
    ap.add_argument("--max-fall", type=float)
    ap.add_argument("--ref")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--static", default="")
    ap.add_argument("--cvvdp", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.candidates or args.peak is None or args.max_fall is None:
        ap.error("CANDIDATS, --peak et --max-fall attendus")
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
