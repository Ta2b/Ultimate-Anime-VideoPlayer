"""Triplets d'anime (ATD-12K et toute arborescence de la même forme) et leurs flux NVOF.

Chaque dossier de triplet contient frame1, frame2, frame3 (png ou jpg) et nvof.npy, écrit par
precompute_nvof.py : 4 × H/4 × W/4 en float16, (F13x, F13y, F31x, F31y) en pixels, de la
première image vers la troisième puis l'inverse. frame2 est la cible, à t = 1/2.

ATD-12K (Siyao et al., CVPR 2021) se télécharge depuis le dépôt AnimeInterp ; sa licence est
celle de la recherche, à respecter pour tout modèle publié.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

GRID = 4  # grille de sortie de NVOF utilisée partout : la seule de toutes les générations


def frame(folder: Path, name: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = folder / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(folder / name)


def triplets(root: Path) -> list[Path]:
    """Les dossiers de triplets sous root qui ont leurs flux."""
    found = sorted(p.parent for p in root.rglob("nvof.npy"))
    if not found:
        raise SystemExit(f"{root} : aucun nvof.npy (lancer precompute_nvof.py d'abord)")
    return found


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


class Triplets(Dataset):
    """Recadrages alignés sur la grille de NVOF, miroirs et inversion du temps."""

    def __init__(self, root: Path, crop: int = 256, train: bool = True) -> None:
        if crop % 64 != 0:
            raise ValueError("crop : multiple de 64")
        self.folders = triplets(root)
        self.crop = crop
        self.train = train

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, index: int):
        folder = self.folders[index]
        a = load_rgb(frame(folder, "frame1"))
        gt = load_rgb(frame(folder, "frame2"))
        b = load_rgb(frame(folder, "frame3"))
        flow = np.load(folder / "nvof.npy").astype(np.float32)
        h, w = a.shape[:2]
        if self.train:
            crop = self.crop
            y = random.randrange(0, (h - crop) // GRID + 1) * GRID
            x = random.randrange(0, (w - crop) // GRID + 1) * GRID
        else:  # validation : la plus grande image entière multiple de 64, en haut à gauche
            crop = None
            y = x = 0
        ch = crop if crop else h // 64 * 64
        cw = crop if crop else w // 64 * 64
        a, gt, b = (img[y : y + ch, x : x + cw] for img in (a, gt, b))
        flow = flow[:, y // GRID : (y + ch) // GRID, x // GRID : (x + cw) // GRID]
        if self.train:
            if random.random() < 0.5:  # miroir horizontal : x change de signe
                a, gt, b = (img[:, ::-1] for img in (a, gt, b))
                flow = flow[:, :, ::-1] * np.array([-1, 1, -1, 1], np.float32)[:, None, None]
            if random.random() < 0.5:  # miroir vertical
                a, gt, b = (img[::-1] for img in (a, gt, b))
                flow = flow[:, ::-1] * np.array([1, -1, 1, -1], np.float32)[:, None, None]
            if random.random() < 0.5:  # temps inversé : les deux flux échangés, t = 1/2 inchangé
                a, b = b, a
                flow = flow[[2, 3, 0, 1]]
        to = lambda img: torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # noqa: E731
        return to(a), to(b), to(gt), torch.from_numpy(np.ascontiguousarray(flow))
