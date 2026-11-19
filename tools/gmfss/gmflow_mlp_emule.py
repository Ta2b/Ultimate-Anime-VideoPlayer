#!/usr/bin/env python3
"""Le MLP natif de GMFlow rejoué tel que le C++ le lit, contre ONNX Runtime (docs/gmflow-natif.md, PR 3).

Les couches (poids rangés, ε) et leurs douze emplois (échelle, couche, noms ONNX de x, du message et de la sortie)
viennent de gmflow_mlp_vide (tools/gmfss/gmflow_mlp), qui appelle la lecture du lecteur (prepare/gmflow_mlp.h) sur
la paire ONNX. La paire FP32 tourne sous ONNX Runtime sur de vraies paires d'images (data/calibration/
calibration.mkv, deux images à 2/24 s d'écart) en rendant, pour chaque emploi, x, le message et la sortie de la
couche (après le résiduel) et la branche (norm2) : la vérité. Sur ces entrées :
  - contrôle f32 : le plan lu dans la paire FP32 sans aucun arrondi, entrées f32. Doit rendre le graphe à l'ordre
    des sommes près (indexation : W1 [caché][x puis message], W2 [128][caché], GELU, LayerNorm, résiduel) ;
  - natif FP16 : les arrondis du noyau (gmflow_mlp_kernels.cuh) sur le plan de la paire FP16 : x et message en
    FP16, poids en FP16, H = X · W1 sommé en f32, GELU exacte en f32 arrondie en FP16 (les fragments A), O = H ·
    W2 sommé en f32, LayerNorm en f32 (moyenne, puis variance des écarts), résiduel en f32, un arrondi FP16 ;
  - actuel : le sous-graphe du même MLP dans la paire FP16 du lecteur, sous ONNX Runtime (CPU : poids FP16, calcul
    f32 ; ce n'est pas TensorRT) ;
  - stockage FP16 de chaque nœud (façon TensorRT FP16) : chaque sortie du graphe FP16 arrondie en FP16 (MatMul, Div
    par le √2 FP16, Erf, Add, Mul, Mul, MatMul, LayerNorm, Add), sommes en f32.
Écarts à la vérité, sur la sortie de la couche (ce que lit la suite) et sur la branche (ce que le MLP ajoute, à sa
propre échelle) : PSNR sur la crête de la référence et écart relatif √EQM / √(moyenne des carrés), moyens sur les
paires et les six couches, et le pire.

    cmake -S tools/gmfss/gmflow_mlp -B build/gmflow_mlp && cmake --build build/gmflow_mlp --target gmflow_mlp_vide
    build/gmflow_mlp/gmflow_mlp_vide paire_fp32.onnx mlp32
    build/gmflow_mlp/gmflow_mlp_vide gmfss_fortuna_union_pair.onnx mlp16
    python -I tools/gmfss/gmflow_mlp_emule.py --fp32 paire_fp32.onnx --fp16 gmfss_fortuna_union_pair.onnx \\
        --plan32 mlp32 --plan16 mlp16 [--width 1280 --height 768] [--work DOSSIER] [--banc DOSSIER]
    python -I tools/gmfss/gmflow_mlp_emule.py --fp16 gmfss_fortuna_union_pair.onnx --plan16 mlp16 \\
        --sous-graphe mlp_couche0.onnx

paire_fp32.onnx : la paire union exportée en FP32 pur (gmflow_natif_emule.py --work DOSSIER l'écrit ; paire_s05 pour
le demi-flot). --banc DOSSIER : les entrées réelles de la première paire (couches 0 et 5 aux deux échelles) pour
gmflow_mlp_banc --check. --sous-graphe : le MLP de la couche 0 seul (entrées x et message [lot, jetons, 128] FP16,
sortie out), pour trtexec (gmflow_mlp_banc.sh).
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.utils
import onnxruntime as ort
import torch

DEPOT = Path(__file__).resolve().parents[2]
VIDEO = DEPOT / "data/calibration/calibration.mkv"
INSTANTS = (15.0, 39.0, 69.0, 99.0)
FP16_MAX = 65504.0
INV_SQRT2 = 0.70710678118654752
ECHELLES = ("1/8", "1/4")
MORCEAU = 8192  # jetons émulés à la fois : tout ce qui suit les GEMM est par jeton, le découpage est exact


@dataclass
class Couche:
    eps: float
    w1: torch.Tensor  # [caché][256]
    w2: torch.Tensor  # [128][caché]
    gamma: torch.Tensor
    beta: torch.Tensor


@dataclass
class Plan:
    channels: int = 128
    hidden: int = 0
    couches: list = field(default_factory=list)
    emplois: list = field(default_factory=list)  # (échelle, couche, source, message, sortie)


def lit_plan(prefixe: str) -> Plan:
    """Le MLP vidé par gmflow_mlp_vide : PREFIXE.txt et les poids float32 de PREFIXE.bin."""
    poids = torch.from_numpy(np.fromfile(prefixe + ".bin", dtype="<f4").copy())
    p = Plan()
    for ligne in Path(prefixe + ".txt").read_text().splitlines():
        m = ligne.split()
        if m[0] in ("channels", "hidden"):
            setattr(p, m[0], int(m[1]))
        elif m[0] == "layer":
            a, b, c, d = (int(v) for v in m[3:7])
            h, ch = p.hidden, p.channels
            p.couches.append(Couche(float(m[2]), poids[a:b].view(h, 2 * ch), poids[b:c].view(ch, h),
                                    poids[c:d], poids[d : d + ch]))
        elif m[0] == "use":
            p.emplois.append((int(m[1]), int(m[2]), m[3], m[4], m[5]))
    return p


def f16(x: torch.Tensor) -> torch.Tensor:
    """Une valeur f32 telle qu'un noyau l'écrit en FP16 (bornée au plus grand fini, comme to_half2)."""
    return x.clamp(-FP16_MAX, FP16_MAX).half().float()


def layernorm(o: torch.Tensor, c: Couche) -> torch.Tensor:
    """La LayerNorm du noyau, en f32 : moyenne, puis variance des écarts."""
    moyenne = o.mean(-1, keepdim=True)
    d = o - moyenne
    return d * torch.rsqrt((d * d).mean(-1, keepdim=True) + c.eps) * c.gamma + c.beta


def par_morceaux(f, *entrees: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """f sur des morceaux de MORCEAU jetons (la mémoire du poste), les sorties recollées."""
    parts = [f(*t) for t in zip(*(e.split(MORCEAU) for e in entrees))]
    return tuple(torch.cat(p) for p in zip(*parts))


def natif(c: Couche, x: torch.Tensor, m: torch.Tensor, arrondi: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Une couche avec les arrondis du noyau (arrondi), ou sans aucun : (sortie, branche)."""
    if x.shape[0] > MORCEAU:
        return par_morceaux(lambda a, b: natif(c, a, b, arrondi), x, m)
    r = f16 if arrondi else (lambda v: v)
    xin = r(x)
    h = torch.cat([xin, r(m)], -1) @ r(c.w1).t()
    g = r(0.5 * h * (1.0 + torch.erf(h * INV_SQRT2)))
    branche = layernorm(g @ r(c.w2).t(), c)
    return r(xin + branche), branche


def facon_trt(c: Couche, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Une couche du graphe FP16, chaque sortie de nœud arrondie en FP16, sommes et statistiques en f32."""
    if x.shape[0] > MORCEAU:
        return par_morceaux(lambda a, b: facon_trt(c, a, b), x, m)
    x16, w1, w2 = f16(x), f16(c.w1), f16(c.w2)
    h = f16(torch.cat([x16, f16(m)], -1) @ w1.t())
    e = f16(torch.erf(f16(h / f16(torch.tensor(math.sqrt(2.0))))))
    g = f16(f16(h * f16(e + 1.0)) * 0.5)
    branche = f16(layernorm(f16(g @ w2.t()), c))
    return f16(x16 + branche), branche


def ajoute_sorties(chemin: Path, noms: list[str]) -> bytes:
    """La paire avec ces tenseurs intermédiaires en sorties (f32)."""
    modele = onnx.load(str(chemin))
    connus = {o.name for o in modele.graph.output}
    for nom in noms:
        if nom not in connus:
            modele.graph.output.append(onnx.helper.make_tensor_value_info(nom, onnx.TensorProto.FLOAT, None))
            connus.add(nom)
    return modele.SerializeToString()


def branches(chemin: Path, plan: Plan) -> dict[str, str]:
    """La sortie de norm2 de chaque emploi : la seconde entrée de l'Add du résiduel."""
    sorties = {e[4] for e in plan.emplois}
    graphe = onnx.load(str(chemin), load_external_data=False).graph
    return {n.output[0]: n.input[1] for n in graphe.node if n.op_type == "Add" and n.output[0] in sorties}


def sous_graphe(paire16: Path, emploi: tuple, sortie: Path, renomme: bool) -> Path:
    """Le MLP d'un emploi seul, extrait de la paire FP16 ; renomme : entrées x, message et sortie out de forme
    [lot, jetons, 128] (pour trtexec)."""
    if sortie.exists():
        return sortie
    _, _, source, message, out = emploi
    onnx.utils.extract_model(str(paire16), str(sortie), [source, message], [out])
    if renomme:
        modele = onnx.load(str(sortie))
        noms = {source: "x", message: "message", out: "out"}
        for n in modele.graph.node:
            n.input[:] = [noms.get(i, i) for i in n.input]
            n.output[:] = [noms.get(o, o) for o in n.output]
        for v in list(modele.graph.input) + list(modele.graph.output):
            v.name = noms[v.name]
            dims = v.type.tensor_type.shape.dim
            for d, nom in zip(dims, ("lot", "jetons")):
                d.ClearField("dim_value")
                d.dim_param = nom
        del modele.graph.value_info[:]
        onnx.checker.check_model(modele)
        onnx.save(modele, str(sortie))
    return sortie


def images(instant: float, w: int, h: int) -> np.ndarray:
    """frames [1, 6, h, w] : deux images de la vidéo à 2/24 s d'écart, w × (hauteur 16:9) complétée en recopiant
    le bord comme gmfss_frames, arrondies en FP16 comme les plans que donne le lecteur."""
    hu = w * 9 // 16
    plans = []
    for t in (instant, instant + 2 / 24):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(VIDEO), "-frames:v", "1", "-vf",
                              f"scale={w}:{hu}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(hu, w, 3).astype(np.float32) / 255
        plans.append(np.pad(a.transpose(2, 0, 1), ((0, 0), (0, h - hu), (0, 0)), mode="edge"))
    return np.concatenate(plans, 0)[None].astype(np.float16).astype(np.float32)


def ecart(y: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (y - ref).double()
    eqm = float((d * d).mean())
    crete = float(ref.abs().max())
    return {"max": float(d.abs().max()), "rel": math.sqrt(eqm) / max(float(ref.double().pow(2).mean().sqrt()), 1e-30),
            "psnr": 10 * math.log10(crete**2 / eqm) if eqm > 0 else float("inf")}


def ecrit_banc(dossier: Path, emploi: tuple, x: torch.Tensor, m: torch.Tensor, ref: torch.Tensor,
               sortie: torch.Tensor, lignes: list[str]):
    """Les entrées d'un emploi pour gmflow_mlp_banc --check : x, message, sortie émulée (FP16) et vérité (f32)."""
    echelle, couche = emploi[0], emploi[1]
    prefixe = f"s{echelle}_c{couche}"
    for suffixe, t, dtype in ((".x.f16", x, np.float16), (".message.f16", m, np.float16),
                              (".ref.f32", ref, np.float32), (".natif.f16", sortie, np.float16)):
        t.numpy().astype(dtype).tofile(dossier / (prefixe + suffixe))
    lignes.append(f"{echelle} {couche} {x.shape[0]} {prefixe}")


def options(optimise: bool) -> ort.SessionOptions:
    """Sans arène : la mémoire rendue au système entre deux passages (le poste la partage)."""
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    if not optimise:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return so


class Actuel:
    """Le MLP de chaque couche sous ONNX Runtime, extrait de la paire FP16 du lecteur."""

    def __init__(self, paire16: Path, plan16: Plan, travail: Path):
        self.sessions = {}
        for e in plan16.emplois:
            if e[0] == 0:
                chemin = sous_graphe(paire16, e, travail / f"mlp16_couche{e[1]}.onnx", False)
                sess = ort.InferenceSession(str(chemin), options(True), providers=["CPUExecutionProvider"])
                self.sessions[e[1]] = (sess, e)

    def __call__(self, couche: int, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        sess, e = self.sessions[couche]
        sorties = []
        for a, b in zip(x.split(MORCEAU), m.split(MORCEAU)):
            feeds = {e[2]: a[None].numpy().astype(np.float16), e[3]: b[None].numpy().astype(np.float16)}
            sorties.append(torch.from_numpy(sess.run(None, feeds)[0].astype(np.float32))[0])
        return torch.cat(sorties)


def mesure(args, plan32: Plan, plan16: Plan) -> dict:
    """Les écarts de chaque recette, par (échelle, sortie, recette) : la liste des écarts (paires × couches)."""
    sorties_b = branches(args.fp32, plan32)
    noms = [n for e in plan32.emplois for n in (e[2], e[3], e[4], sorties_b[e[4]])]
    graphe = ajoute_sorties(args.fp32, noms)
    actuel = Actuel(args.fp16, plan16, args.work)
    cumul: dict[tuple, list] = {}
    lignes_banc: list[str] = []
    for k, instant in enumerate(INSTANTS):
        verite = ort.InferenceSession(graphe, options(False), providers=["CPUExecutionProvider"])
        valeurs = dict(zip(noms, verite.run(noms, {"frames": images(instant, args.width, args.height)})))
        del verite
        for e32, e16 in zip(plan32.emplois, plan16.emplois):
            x, m, ref = (torch.from_numpy(valeurs[n]).reshape(-1, plan32.channels) for n in e32[2:5])
            ref_b = torch.from_numpy(valeurs[sorties_b[e32[4]]]).reshape(-1, plan32.channels)
            c32, c16 = plan32.couches[e32[1]], plan16.couches[e16[1]]
            nat = natif(c16, x, m, True)
            act = actuel(e16[1], x, m)
            recettes = {"contrôle f32 (plan FP32)": natif(c32, x, m, False), "natif FP16": nat,
                        "actuel (ORT, paire FP16)": (act, act - f16(x)), "stockage FP16 de chaque nœud": facon_trt(c16, x, m)}
            for nom, (y, b) in recettes.items():
                cumul.setdefault((ECHELLES[e32[0]], "couche", nom), []).append(ecart(y, ref))
                cumul.setdefault((ECHELLES[e32[0]], "branche", nom), []).append(ecart(b, ref_b))
            if args.banc is not None and k == 0 and e32[1] in (0, 5):
                ecrit_banc(args.banc, e16, f16(x), f16(m), ref, nat[0], lignes_banc)
        del valeurs
        print(f"paire {instant:g} s faite", flush=True)
    if args.banc is not None:
        (args.banc / "banc.txt").write_text("\n".join(lignes_banc) + "\n")
    return cumul


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, help="la paire union exportée en FP32 pur")
    parser.add_argument("--fp16", type=Path, required=True, help="la paire FP16 du lecteur")
    parser.add_argument("--plan32", help="le MLP vidé de la paire FP32 (préfixe)")
    parser.add_argument("--plan16", required=True, help="le MLP vidé de la paire FP16 (préfixe)")
    parser.add_argument("--width", type=int, default=1280, help="largeur de la paire (multiple de 128)")
    parser.add_argument("--height", type=int, default=768, help="hauteur de la paire (multiple de 128)")
    parser.add_argument("--work", type=Path, default=Path("."), help="dossier des sous-graphes extraits")
    parser.add_argument("--banc", type=Path, help="écrire les entrées de gmflow_mlp_banc --check ici")
    parser.add_argument("--sous-graphe", type=Path, help="écrire le MLP de la couche 0 seul (trtexec) et finir")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    plan16 = lit_plan(args.plan16)
    if args.sous_graphe is not None:
        print(sous_graphe(args.fp16, plan16.emplois[0], args.sous_graphe, True))
        return
    if args.fp32 is None or args.plan32 is None:
        parser.error("--fp32 et --plan32 sont demandés pour l'émulation")
    args.work.mkdir(parents=True, exist_ok=True)
    if args.banc is not None:
        args.banc.mkdir(parents=True, exist_ok=True)
    cumul = mesure(args, lit_plan(args.plan32), plan16)
    print(f"\nMLP de GMFlow, paire {args.width}x{args.height}, {len(INSTANTS)} paires × 6 couches ; écarts au graphe FP32")
    print("| échelle | sortie | recette | pire max | relatif moyen | PSNR moyen / pire (dB) |\n|---|---|---|---|---|---|")
    for (echelle, sortie, nom), rows in cumul.items():
        ps = [r["psnr"] for r in rows]
        print(f"| {echelle} | {sortie} | {nom} | {max(r['max'] for r in rows):.3e} | "
              f"{sum(r['rel'] for r in rows) / len(rows):.3e} | {sum(ps) / len(ps):.1f} / {min(ps):.1f} |")


if __name__ == "__main__":
    main()
