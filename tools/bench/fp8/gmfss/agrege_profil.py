#!/usr/bin/env python3
"""Agrège le profil par couche de trtexec (--exportProfile / --exportLayerInfo, mode detailed).

Pour chaque couche TensorRT : temps moyen mesuré, octets lus + écrits d'après les formes et les
types des tenseurs, borne mémoire à 504 Go/s, catégorie d'après le type de couche et le nom
(les noyaux Myelin fusionnés portent dans leur nom les opérations ONNX qu'ils contiennent).

Usage : agrege_profil.py profil.json couches.json [--top N] [--md]
"""
import argparse
import json
import re
from collections import defaultdict

BANDE = 504e9  # octets/s, RTX 4070 Ti
OCTETS = {"Half": 2, "Float": 4, "Int8": 1, "Int32": 4, "Int64": 8, "Bool": 1, "FP8": 1, "UInt8": 1}


def octets(tenseurs):
    total = 0
    for t in tenseurs:
        n = 1
        for d in t.get("Dimensions", []):
            n *= max(int(d), 1)
        total += n * OCTETS.get(t.get("Datatype", "Half"), 2)
    return total


def categorie(nom, type_couche, tactique):
    n = nom
    nl = n.lower()
    if type_couche in ("signal", "wait"):
        return "synchro"
    # Chemin « classique » de TensorRT : noms fusionnés « Conv + PWN(...) », « X copy », « Reformatting »
    if nl.startswith("reformatting") or " copy" in nl[-6:] or nl.endswith(" copy"):
        return "reformatage/copies"
    if re.search(r"/conv(transpose)?(_\d+)? \+|/conv(transpose)?(_\d+)?$|conv_last|\.\d/conv", nl) and "matmul" not in nl:
        return "convolutions + épilogue fusionné" if "pwn(" in nl or " + " in nl else "convolutions"
    # Noyau de convolution/GEMM (tactique xmma/cutlass...) : « Corr » dans un nom Myelin = convolution
    # avec son activation fusionnée dans l'épilogue.
    if re.search(r"xmma|cutlass|implicit_gemm|cudnn|gemm|sm\d\d_", tactique or "", re.I) and "Soft" not in n:
        est_mat = "MatM" in n or "/MatMul" in n or "attn" in nl or "transformer" in nl
        if est_mat:
            return "attention/MatMul/softmax"
        fusionne = any(k in n for k in ("Neg", "Relu", "Mul", "Sub", "Add", "Cast"))
        return "convolutions + épilogue fusionné" if fusionne else "convolutions"
    if type_couche == "correlation" or "/conv" in nl and type_couche != "kgen" and "convtranspose" not in nl:
        return "convolutions"
    if "convtranspose" in nl or type_couche == "deconvolution":
        return "convolutions"
    if "gridsamp" in nl or "grid" in nl and "samp" in nl or "gridsa" in nl:
        return "warp/GridSample"
    if "resi" in nl and "resh" not in nl or "resize" in nl:
        return "Resize"
    if any(k in n for k in ("Soft", "MatM", "Gemm", "Attn", "attn")) or "matmul" in nl:
        return "attention/MatMul/softmax"
    if "Corr" in n and "Relu" not in n:
        return "attention/MatMul/softmax"
    if any(k in nl for k in ("norm", "instancenorm", "layernorm")):
        return "normalisations"
    # Noyau « kgen » / Myelin : on lit les opérations dans le nom
    ops = re.findall(r"[A-Z][a-z]+", n)
    copies = {"Move", "Conc", "Slic", "Tran", "Resh", "Gath", "Pad", "Unsq", "Sque", "Tile", "Expa", "Iden"}
    elem = {"Neg", "Relu", "Mul", "Add", "Sub", "Div", "Cast", "Clip", "Erf", "Pow", "Sin", "Cos", "Tanh", "Abs", "Whe",
            "Gre", "Less", "Equa", "Not", "And", "Cumsum", "Redu", "Sqrt", "Exp", "Corr"}
    if ops and all(o[:4] in {c[:4] for c in copies} for o in ops):
        return "copies/Transpose/Reshape/Concat/Slice"
    if tactique and "reformat" in tactique.lower() or type_couche == "reformat" or nl.startswith("reformat"):
        return "reformatage"
    if ops and any(o[:3] in {e[:3] for e in elem} for o in ops):
        return "élémentaires"
    return "autres (" + type_couche + ")"


def module_de(nom, meta):
    """Module du réseau d'après le premier noeud ONNX porté par la couche (feat_ext, flownet/...)."""
    m = re.search(r"\[ONNX Layer: (/[^\]]*)\]", meta or "")
    chemin = m.group(1) if m else nom
    parts = [p for p in chemin.split("/") if p]
    if not parts:
        return "inconnu"
    if parts[0] == "flownet" and len(parts) > 1:
        return "flownet/" + re.sub(r"_?\d+$", "", parts[1])
    return re.sub(r"_?\d+$", "", parts[0]) if not parts[0].startswith(("__", "entry")) else "myelin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profil")
    ap.add_argument("couches")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    prof = json.load(open(a.profil))
    info = json.load(open(a.couches))["Layers"]
    temps = {e["name"]: e["averageMs"] for e in prof if "name" in e}
    total_prof = sum(temps.values())
    lignes = []
    for c in info:
        nom = c["Name"]
        ms = temps.get(nom, 0.0)
        o_lus = octets(c.get("Inputs", []))
        o_ecrits = octets(c.get("Outputs", []))
        cat = categorie(nom, c.get("LayerType", ""), c.get("TacticName", ""))
        forme_in = c["Inputs"][0]["Dimensions"] if c.get("Inputs") else []
        forme_out = c["Outputs"][0]["Dimensions"] if c.get("Outputs") else []
        onnx_noeuds = re.findall(r"\[ONNX Layer: ([^\]]*)\]", c.get("Metadata", "") or "")
        lignes.append(dict(onnx=onnx_noeuds, module=module_de(nom, c.get("Metadata", "")), nom=nom, ms=ms, cat=cat, lu=o_lus, ecrit=o_ecrits, fin=forme_in, fout=forme_out,
                           type=c.get("LayerType", ""), tactique=c.get("TacticName", "")))
    par_cat = defaultdict(lambda: [0.0, 0, 0, 0])
    for l in lignes:
        p = par_cat[l["cat"]]
        p[0] += l["ms"]
        p[1] += l["lu"] + l["ecrit"]
        p[2] += 1
        p[3] += 1 if l["ms"] > 0 else 0
    print(f"Temps total des couches : {total_prof:.3f} ms ({len(lignes)} couches, "
          f"{sum(1 for l in lignes if l['ms'] > 0)} avec temps > 0)\n")
    print("| catégorie | temps (ms) | part | couches | octets lus+écrits (Mo) | borne mémoire (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    for cat, (ms, b, n, nz) in sorted(par_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"| {cat} | {ms:.3f} | {100 * ms / total_prof:.1f} % | {nz} | {b / 1e6:.0f} | {b / BANDE * 1e3:.3f} |")
    par_mod = defaultdict(float)
    for l in lignes:
        par_mod[l["module"]] += l["ms"]
    print("\n| module | temps (ms) | part |")
    print("|---|---:|---:|")
    for mod, ms in sorted(par_mod.items(), key=lambda kv: -kv[1])[:14]:
        print(f"| {mod} | {ms:.3f} | {100 * ms / total_prof:.1f} % |")
    print(f"\nTop {a.top} noyaux :")
    print("| # | ms | part | catégorie | noyau | noeuds ONNX (premiers) | forme entrée -> sortie | Mo | borne ms | tactique |")
    print("|---:|---:|---:|---|---|---|---|---:|---:|---|")
    for i, l in enumerate(sorted(lignes, key=lambda x: -x["ms"])[: a.top], 1):
        b = l["lu"] + l["ecrit"]
        ox = ", ".join(x.split("/", 1)[-1][-48:] for x in l["onnx"][:3]) + (" ..." if len(l["onnx"]) > 3 else "")
        print(f"| {i} | {l['ms']:.3f} | {100 * l['ms'] / total_prof:.1f} % | {l['cat']} | `{l['nom'][:70]}` | {ox} | "
              f"{'x'.join(map(str, l['fin']))} -> {'x'.join(map(str, l['fout']))} | {b / 1e6:.0f} | "
              f"{b / BANDE * 1e3:.3f} | {l['tactique'][:50]} |")
    json.dump(lignes, open(a.profil.replace(".json", "_agrege.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
