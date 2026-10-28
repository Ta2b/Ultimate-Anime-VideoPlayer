#!/usr/bin/env python3
"""Contrôle du rapport des couches d'un moteur TensorRT (étude 4, 6.1).

Le lecteur fait écrire à trtexec, à chaque construction, « <moteur>.layers.json » à côté du moteur
(--profilingVerbosity=detailed --exportLayerInfo). Ce script le relit et dit :
  - les reformatages (LayerType « Reformat ») hors des E/S du réseau : une fusion manquée, à
    corriger dans l'export ONNX quand le tenseur reformaté fait au moins --full-image de la plus
    grande E/S (pleine image) ;
  - les couches dont une sortie est en FP32 : hors îlots déclarés, une précision perdue en vitesse.
    Les îlots sont les nœuds de l'ONNX dont une sortie est déclarée FP32 (--onnx, typage fort :
    tools/rife/convert_rife_fp16.py garde GridSample, Resize et le calcul de la grille) ; une
    couche y appartient si son nom contient le nom d'un de ces nœuds. Sans --onnx, aucun îlot :
    toute couche FP32 est signalée (SPAN, entièrement FP16).

Code de sortie 1 si un reformatage pleine image ou une couche FP32 hors îlots est trouvé, 0 sinon.
Myelin regroupe parfois un sous-graphe entier en une couche (« ForeignNode ») : ce qui s'y passe
n'est pas dans le rapport.

Avec --profile (un seul rapport), le profil par couche de trtexec (--exportProfile, sur le PC) donne
la part du temps des reformatages, celle des E/S à part : c'est elle qui décide de l'essai d'E/S
NHWC (6.4, --inputIOFormats=hwc8 et des noyaux d'ingestion et de lecture qui suivent ce format).

  python3 tools/trt/layer_report.py moteurs/aji-v3-….engine.layers.json --onnx modele.onnx
  python3 tools/trt/layer_report.py moteurs/aji-v3-….engine.layers.json --profile profil.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

FP32 = re.compile(r"FP32|\bFloat\b|float32", re.IGNORECASE)


def tensors(layer: dict, key: str) -> list[dict]:
    value = layer.get(key, [])
    return value if isinstance(value, list) else []


def elements(tensor: dict) -> int:
    dims = tensor.get("Dimensions", [])
    if not isinstance(dims, list) or not dims:
        return 0
    return math.prod(max(int(d), 1) for d in dims)


def is_fp32(tensor: dict) -> bool:
    return bool(FP32.search(str(tensor.get("Format/Datatype", ""))))


def onnx_islands(path: Path) -> set[str]:
    """Noms des nœuds de l'ONNX dont une sortie est déclarée FP32."""
    import onnx  # seulement avec --onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    float_type = onnx.TensorProto.FLOAT
    declared = {v.name for v in list(graph.value_info) + list(graph.output) if v.type.tensor_type.elem_type == float_type}
    islands = set()
    for node in graph.node:
        if node.op_type == "Cast":
            to = next((a.i for a in node.attribute if a.name == "to"), None)
            if to == float_type:
                islands.add(node.name)
            continue
        if any(o in declared for o in node.output):
            islands.add(node.name)
    return {name for name in islands if name}


def check(report: Path, islands: set[str], full_image: float) -> int:
    data = json.loads(report.read_text(encoding="utf-8"))
    layers = data["Layers"] if isinstance(data, dict) else data
    bindings = set(data.get("Bindings", [])) if isinstance(data, dict) else set()
    largest = max(
        (elements(t) for layer in layers for key in ("Inputs", "Outputs") for t in tensors(layer, key)
         if t.get("Name") in bindings),
        default=0,
    )
    failures = 0
    reformats = [layer for layer in layers if layer.get("LayerType") == "Reformat"]
    internal = []
    for layer in reformats:
        names = {t.get("Name") for key in ("Inputs", "Outputs") for t in tensors(layer, key)}
        if not names & bindings:
            internal.append(layer)
    print(f"{report.name} : {len(layers)} couches, {len(reformats)} reformatages dont {len(internal)} hors des E/S")
    for layer in internal:
        size = max((elements(t) for t in tensors(layer, "Outputs")), default=0)
        full = largest > 0 and size >= full_image * largest
        failures += 1 if full else 0
        print(f"  {'ÉCHEC' if full else 'note'} reformatage {layer.get('Name')} ({size} éléments)")
    fp32 = [
        layer
        for layer in layers
        if layer.get("LayerType") != "Reformat" and any(is_fp32(t) for t in tensors(layer, "Outputs"))
    ]
    outside = [layer for layer in fp32 if not any(name in str(layer.get("Name", "")) for name in islands)]
    print(f"  {len(fp32)} couches FP32, dont {len(outside)} hors îlots déclarés")
    for layer in outside:
        failures += 1
        print(f"  ÉCHEC FP32 {layer.get('Name')} ({layer.get('LayerType')})")
    return failures


# Part du temps des reformatages d'E/S au-delà de laquelle des E/S NHWC valent l'essai (6.4).
NHWC_TRIAL_SHARE = 3.0


def profile_shares(report: Path, profile: Path) -> None:
    """Temps des reformatages, d'après le profil par couche de trtexec (--exportProfile)."""
    data = json.loads(report.read_text(encoding="utf-8"))
    layers = data["Layers"] if isinstance(data, dict) else data
    bindings = set(data.get("Bindings", [])) if isinstance(data, dict) else set()
    kinds = {}
    for layer in layers:
        names = {t.get("Name") for key in ("Inputs", "Outputs") for t in tensors(layer, key)}
        if layer.get("LayerType") == "Reformat":
            kinds[layer.get("Name")] = "io" if names & bindings else "internal"
    rows = [r for r in json.loads(profile.read_text(encoding="utf-8")) if "name" in r]
    total = sum(float(r.get("averageMs", 0.0)) for r in rows)
    if total <= 0:
        sys.exit(f"{profile} : profil vide")
    io = sum(float(r["averageMs"]) for r in rows if kinds.get(r["name"]) == "io")
    internal = sum(float(r["averageMs"]) for r in rows if kinds.get(r["name"]) == "internal")
    unknown = [r["name"] for r in rows if "Reformat" in r["name"] and r["name"] not in kinds]
    print(f"{profile.name} : {total:.3f} ms par passe ; reformatages d'E/S {io:.3f} ms ({100 * io / total:.1f} %), "
          f"internes {internal:.3f} ms ({100 * internal / total:.1f} %)")
    for name in unknown:
        print(f"  note : {name} absent du rapport des couches (autre moteur ?)")
    if 100 * io / total >= NHWC_TRIAL_SHARE:
        print(f"  E/S NHWC à essayer (au moins {NHWC_TRIAL_SHARE:.0f} % du temps) : --inputIOFormats=hwc8")
    else:
        print(f"  E/S NHWC sans intérêt (moins de {NHWC_TRIAL_SHARE:.0f} % du temps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+", help="rapports « <moteur>.layers.json »")
    parser.add_argument("--onnx", type=Path, help="le modèle construit : ses nœuds FP32 sont les îlots admis")
    parser.add_argument(
        "--full-image", type=float, default=0.25,
        help="part de la plus grande E/S à partir de laquelle un reformatage interne est un échec (0,25)",
    )
    parser.add_argument("--profile", type=Path, help="profil par couche de trtexec (--exportProfile) du même moteur")
    args = parser.parse_args()
    if args.profile:
        if len(args.reports) != 1:
            parser.error("--profile va avec un seul rapport")
        profile_shares(args.reports[0], args.profile)
    islands = onnx_islands(args.onnx) if args.onnx else set()
    failures = sum(check(report, islands, args.full_image) for report in args.reports)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
