"""Sorties du banc de qualité : tableau, rapport Markdown, et seuils (seuils.json).

Un seuil est une porte : PSNR de fidélité le plus bas admis, pire écart de fidélité le plus haut admis (codes
8 bits), PSNR de vérité moyen le plus bas admis. Les seuils ne valent que pour la configuration qu'ils nomment.
"""
import json
import os

# Marges prises par write_gates sur la mesure du jour : un autre processeur arrondit autrement.
MARGIN_FIDELITY_DB = 1.0
MARGIN_TRUTH_DB = 0.3
MARGIN_CODES = 1.25


def _rows(report):
    for r in report["reseaux"]:
        yield (r["modele"], r["fidelite_psnr"], r["fidelite_psnr_pire"], r["fidelite_ecart_pire"],
               r["fidelite_ecart_p999"], r["fidelite_itp_p999"], r["verite_psnr"], r["verite_psnr_fp32"],
               r["verite_ssim"], r["ms"])


HEAD = ("réseau", "fidélité dB", "pire dB", "pire écart", "écart p99,9", "ΔE_ITP p99,9", "vérité dB",
        "vérité FP32 dB", "SSIM", "ms (processeur)")


def _short(name):
    return name.replace("2x_AnimeJaNai_HD_", "").replace("_unshuffle_fp16", "").replace(".onnx", "")


def _config(report):
    c = report["config"]
    return (f"interpolation {c['taille']}, {c['scenes']} "
            f"{'fenêtres de la vidéo' if c['source'] == 'video' else 'scènes de synthèse'} ; "
            f"agrandissement 960x540 vers 1920x1080, {c['plans']} plans")


def table(report):
    lines = [_config(report), f"{HEAD[0]:34s}" + "".join(f"{h:>16s}" for h in HEAD[1:])]
    for name, *v in _rows(report):
        lines.append(f"{_short(name):34s}{v[0]:16.2f}{v[1]:16.2f}{v[2]:16.2f}{v[3]:16.2f}{v[4]:16.2f}{v[5]:16.2f}"
                     f"{v[6]:16.2f}{v[7]:16.4f}{v[8]:16.1f}")
    b = report.get("bicubique")
    if b:
        lines.append(f"{'bicubique ×2 (repère)':34s}{'':80s}{b['psnr']:16.2f}{'':16s}{b['ssim']:16.4f}")
    return "\n".join(lines)


def markdown(report):
    lines = ["# Banc de qualité", "", _config(report) + ".", "",
             "Fidélité : le graphe livré contre sa référence FP32. Vérité : contre la vraie image. Écarts en codes "
             "8 bits. Les temps sont ceux d'ONNX Runtime sur processeur : ils ne valent pas ceux du lecteur.", "",
             "| " + " | ".join(HEAD) + " |", "|---|" + "---:|" * (len(HEAD) - 1)]
    for name, *v in _rows(report):
        lines.append(f"| {_short(name)} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} | {v[3]:.2f} | {v[4]:.2f} | "
                     f"{v[5]:.2f} | {v[6]:.2f} | {v[7]:.4f} | {v[8]:.0f} |")
    b = report.get("bicubique")
    if b:
        lines.append(f"| bicubique ×2 (repère) | | | | | | {b['psnr']:.2f} | | {b['ssim']:.4f} | |")
    lines += ["", "## Par cas", "", "| réseau | cas | fidélité dB | pire écart | vérité dB | vérité FP32 dB |",
              "|---|---|---:|---:|---:|---:|"]
    for r in report["reseaux"]:
        for c in r["cas"]:
            lines.append(f"| {_short(r['modele'])} | {c['cas']} | {c['fidelite']['psnr']:.2f} | "
                         f"{c['fidelite']['ecart_pire']:.2f} | {c['verite']['psnr']:.2f} | "
                         f"{c['verite_fp32']['psnr']:.2f} |")
    return "\n".join(lines) + "\n"


def write_gates(path, report):
    gates = {r["modele"]: {"fidelite_psnr_pire_min": round(r["fidelite_psnr_pire"] - MARGIN_FIDELITY_DB, 2),
                           "fidelite_ecart_pire_max": round(r["fidelite_ecart_pire"] * MARGIN_CODES + 0.5, 2),
                           "verite_psnr_min": round(r["verite_psnr"] - MARGIN_TRUTH_DB, 2)}
             for r in report["reseaux"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": report["config"], "reseaux": gates}, f, ensure_ascii=False, indent=1)
        f.write("\n")


def check_gates(path, report):
    """Les seuils non tenus, en clair ; rien si la configuration n'est pas celle des seuils."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        gates = json.load(f)
    if gates["config"] != report["config"]:
        print("seuils non appliqués : la configuration n'est pas celle de", path)
        return []
    out = []
    for r in report["reseaux"]:
        g = gates["reseaux"].get(r["modele"])
        if g is None:
            continue
        if r["fidelite_psnr_pire"] < g["fidelite_psnr_pire_min"]:
            out.append(f"{r['modele']} : fidélité {r['fidelite_psnr_pire']:.2f} dB au pire, sous le seuil "
                       f"{g['fidelite_psnr_pire_min']}")
        if r["fidelite_ecart_pire"] > g["fidelite_ecart_pire_max"]:
            out.append(f"{r['modele']} : pire écart au FP32 {r['fidelite_ecart_pire']:.2f} codes, au-dessus du "
                       f"seuil {g['fidelite_ecart_pire_max']}")
        if r["verite_psnr"] < g["verite_psnr_min"]:
            out.append(f"{r['modele']} : {r['verite_psnr']:.2f} dB contre la vraie image, sous le seuil "
                       f"{g['verite_psnr_min']}")
    return out
