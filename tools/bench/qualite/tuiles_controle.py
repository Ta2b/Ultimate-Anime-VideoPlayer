#!/usr/bin/env python3
"""Le contrôle des tuiles de SPAN (--span-tiles-check) rejoué sur processeur : la sortie que les tuiles composent
contre un calcul complet, image par image, sur les décisions écrites par tuiles_rejeu --masques.

  build/tuiles_rejeu/tuiles_rejeu --diviseur 2 --masques MASQUES < taches.tsv  # le lecteur (--span-tiles)
  python3 -I tools/bench/qualite/tuiles_controle.py --dossier EXTRAIT --masques MASQUES \\
      --modele tools/upscale/models/2x_ArtCNN_C4F32_rgb_fp16.onnx [--diviseur 2] [--images 80]

EXTRAIT : un dossier de flux-carte (k.y, lumas 8 bits du flux, et taches.tsv pour la taille). Le réseau 2× (SPAN
du lecteur ou tout réseau 2× à trois plans) agrandit chaque luma, répétée sur trois plans. La sortie composée
suit les décisions : une tuile calculée y prend la sortie complète de l'image (une tuile du cœur est exactement
celle d'un calcul complet, kernels/spec/span_tiles.h) ; une tuile reprise y garde la sortie de l'image où elle a
été calculée pour la dernière fois. Écarts comme la porte des tuiles : pire bloc 16 × 16 de la sortie (écart
moyen des trois plans, niveaux 8 bits) et écart moyen, pour chaque image reprise entière (R) ou par tuiles (T) ;
porte : pire bloc sous 1 niveau, écart moyen sous 0,05.

Écarts au lecteur : la luma du flux (réduite d'un facteur D) au lieu du RGB de travail, et un autre réseau que
SPAN s'il est donné. Ce qui compte est l'écart d'une sortie reprise à celle d'un calcul complet, que fait l'écart
des entrées (glissement, grain, dessin) ; le contrôle final reste celui du PC, sur SPAN et le RGB de travail.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOCK = 16
TILE_W, TILE_H = 64, 16  # tuile du SPAN FP16 natif, pixels de travail (tuiles_rejeu, native_grid)


def frame_size(folder):
    with open(os.path.join(folder, "taches.tsv"), encoding="utf-8") as fh:
        width, height = fh.readline().split("\t")[:2]
    return int(width), int(height)


def gate(full, composed):
    """Pire bloc et écart moyen, en niveaux."""
    err = np.abs(full - composed).mean(axis=0) * 255.0
    h, w = err.shape[0] // BLOCK * BLOCK, err.shape[1] // BLOCK * BLOCK
    blocks = err[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))
    return float(blocks.max()), float(err.mean())


def expand(mask, cols, rows, out_h, out_w, scale):
    """Le masque des tuiles (rows × cols) en pixels de sortie ; scale : pixels de sortie par pixel de travail."""
    tw, th = round(TILE_W * scale), round(TILE_H * scale)
    big = np.repeat(np.repeat(mask.reshape(rows, cols).astype(bool), th, 0), tw, 1)
    return big[:out_h, :out_w]


def control(args, net):
    width, height = frame_size(args.dossier)
    work_w, work_h = width * args.diviseur, height * args.diviseur
    cols, rows = -(-(work_w // 2) // (TILE_W // 2)), -(-(work_h // 2) // (TILE_H // 2))

    def upscale(k):
        luma = np.fromfile(os.path.join(args.dossier, f"{k}.y"), np.uint8).reshape(height, width)
        return net.upscale(np.repeat((luma / 255.0)[None], 3, 0).astype(np.float32))

    composed = upscale(0)
    scale = composed.shape[2] / work_w
    rows_out = []
    for k in range(1, args.images + 1):
        path = os.path.join(args.masques, f"{k}.coeur")
        if not os.path.exists(path):
            break
        mask = np.fromfile(path, np.uint8)
        if mask.size != cols * rows:
            raise SystemExit(f"{path} : {mask.size} tuiles, la grille en a {cols} × {rows}")
        full = upscale(k)
        sort = "C" if mask.all() else "R" if not mask.any() else "T"
        where = expand(mask, cols, rows, full.shape[1], full.shape[2], scale)
        composed[:, where] = full[:, where]
        worst, mean = gate(full, composed)
        rows_out.append((k, sort, int(mask.sum()), worst, mean))
        if args.detail:
            print(f"- image {k} {sort} : {mask.sum()} tuiles, pire bloc {worst:.2f}, écart moyen {mean:.4f}")
    return rows_out


def report(rows, name):
    print(f"## Contrôle des tuiles sur processeur ({name}, {len(rows)} images)")
    print("| sorte | images | pire bloc (max) | pire bloc (médiane) | écart moyen (max) | hors porte |")
    print("|---|---|---|---|---|---|")
    for sort, label in (("R", "reprise entière"), ("T", "par tuiles")):
        sel = [r for r in rows if r[1] == sort]
        if not sel:
            continue
        worst = np.array([r[3] for r in sel])
        mean = np.array([r[4] for r in sel])
        out = int(np.sum((worst >= 1.0) | (mean >= 0.05)))
        print(f"| {label} | {len(sel)} | {worst.max():.2f} | {np.median(worst):.2f} | {mean.max():.4f} | {out} |")
    print("\nPorte des tuiles : pire bloc sous 1 niveau, écart moyen sous 0,05.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dossier", required=True, help="un extrait de flux-carte (k.y, taches.tsv)")
    p.add_argument("--masques", required=True, help="les K.coeur de tuiles_rejeu --masques")
    p.add_argument("--modele", required=True, help="un réseau 2× à trois plans (SPAN, ArtCNN)")
    p.add_argument("--diviseur", type=int, default=2, help="pixels de travail par pixel des lumas")
    p.add_argument("--images", type=int, default=159)
    p.add_argument("--fournisseur", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--detail", action="store_true", help="une ligne par image")
    args = p.parse_args()
    import modeles

    net = modeles.Network(args.modele, reference=args.fournisseur == "cpu", provider=args.fournisseur)
    report(control(args, net), net.name)


if __name__ == "__main__":
    main()
