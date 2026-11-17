"""Échantillons à t quelconque tirés des suites de sequences.py et de leurs flux (precompute_pairs.py).

Un échantillon : deux images d'une suite à l'écart g (2 à 8), une image du milieu comme vérité, à t = (k − i) / g,
et le flux NVOF de la paire. Tout reste en uint8 jusqu'à la carte (moins de copies, moins de mémoire épinglée) ;
le modèle convertit. Les images sont lues par np.load(mmap_mode="r") : seule la découpe quitte le disque.

    img0, img1, gt : 3 × C × C uint8 ; t : scalaire float32 ; flow : 4 × C/4 × C/4 float16 (F01 x, y puis F10 x, y)
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GRID = 4  # grille de NVOF : les découpes sont alignées dessus


def runs(root: Path) -> list[Path]:
    found = sorted(f.parent for f in root.rglob("flows.npy"))
    if not found:
        raise SystemExit(f"{root} : aucune suite avec ses flux (lancer sequences.py puis precompute_pairs.py)")
    return found


def split(found: list[Path], val_share: float) -> tuple[list[Path], list[Path]]:
    """Validation par vidéo entière (jamais deux suites voisines de part et d'autre), stable d'un lancement à
    l'autre : la vidéo va en validation si l'empreinte de son nom tombe sous val_share."""
    train, val = [], []
    for run in found:
        video = json.loads((run / "meta.json").read_text(encoding="utf-8"))["video"]
        (val if zlib.crc32(video.encode()) / 2**32 < val_share else train).append(run)
    return train, val


class SeqPairs(Dataset):
    """big_share des échantillons viennent des paires dont le mouvement (mag.npy) dépasse big_px : c'est là que
    NVOF et RIFE se trompent, et c'est sur elles que le banc juge ; le reste est tiré uniformément, pour ne rien
    désapprendre des petits mouvements."""

    def __init__(self, folders: list[Path], crop: int = 256, length: int = 100_000, train: bool = True,
                 seed: int = 0, big_share: float = 0.5, big_px: float = 16.0) -> None:
        if crop % 32 != 0:
            raise ValueError("crop : multiple de 32")
        self.folders = folders
        self.pairs = [np.load(f / "pairs.npy") for f in folders]
        flat = [(r, p, float(m)) for r, f in enumerate(folders) for p, m in enumerate(np.load(f / "mag.npy"))]
        self.all = [(r, p) for r, p, _ in flat]
        self.big = [(r, p) for r, p, m in flat if m >= big_px]
        self.crop, self.length, self.train, self.seed = crop, length, train, seed
        self.big_share = big_share if self.big else 0.0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # En validation, le tirage dépend seulement de l'indice : les mêmes échantillons à chaque passe.
        rng = random.Random(index * 7919 + self.seed) if not self.train else random
        pool = self.big if rng.random() < self.big_share else self.all
        r, p = pool[rng.randrange(len(pool))]
        folder, pairs = self.folders[r], self.pairs[r]
        i, j = (int(v) for v in pairs[p])
        k = rng.randrange(i + 1, j)
        t = (k - i) / (j - i)
        frames = np.load(folder / "frames.npy", mmap_mode="r")
        flows = np.load(folder / "flows.npy", mmap_mode="r")
        _, h, w, _ = frames.shape
        c = self.crop
        y = rng.randrange(0, (h - c) // GRID + 1) * GRID
        x = rng.randrange(0, (w - c) // GRID + 1) * GRID
        a, b, gt = (np.asarray(frames[n, y : y + c, x : x + c]) for n in (i, j, k))
        flow = np.asarray(flows[p, :, y // GRID : (y + c) // GRID, x // GRID : (x + c) // GRID], np.float32)
        if self.train:
            if random.random() < 0.5:  # miroir horizontal : x change de signe
                a, b, gt = (img[:, ::-1] for img in (a, b, gt))
                flow = flow[:, :, ::-1] * np.array([-1, 1, -1, 1], np.float32)[:, None, None]
            if random.random() < 0.5:  # miroir vertical
                a, b, gt = (img[::-1] for img in (a, b, gt))
                flow = flow[:, ::-1] * np.array([1, -1, 1, -1], np.float32)[:, None, None]
            if random.random() < 0.5:  # temps inversé : images et flux échangés, t devient 1 − t
                a, b, t = b, a, 1.0 - t
                flow = flow[[2, 3, 0, 1]]
        to = lambda img: torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # noqa: E731
        return (to(a), to(b), to(gt), torch.tensor(t, dtype=torch.float32),
                torch.from_numpy(np.ascontiguousarray(flow, np.float16)))


def intermediate(flow: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Les flux NVOF ramenés à l'instant t (Super SloMo, Jiang et al. 2018), en pixels :
    F_t→0 = −(1−t)·t·F01 + t²·F10 et F_t→1 = (1−t)²·F01 − t·(1−t)·F10. flow : B × 4 × h × w ; t : B."""
    t = t.view(-1, 1, 1, 1).to(flow.dtype)
    f01, f10 = flow[:, :2], flow[:, 2:]
    return torch.cat((-(1 - t) * t * f01 + t * t * f10, (1 - t) * (1 - t) * f01 - t * (1 - t) * f10), 1)


def consistency(flow: torch.Tensor) -> torch.Tensor:
    """Écart aller-retour |F01(x) + F10(x + F01(x))| en pixels, B × 1 × h × w : grand là où NVOF se trompe ou
    sur une occultation, ce que le réseau doit apprendre à ne pas croire."""
    _, _, h, w = flow.shape
    f01, f10 = flow[:, :2].float(), flow[:, 2:].float()
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    # Le flux est en pixels de l'image ; sur la grille de NVOF, un pas vaut GRID pixels.
    gx = (xs + f01[:, 0] / GRID) / max(w - 1, 1) * 2 - 1
    gy = (ys + f01[:, 1] / GRID) / max(h - 1, 1) * 2 - 1
    back = torch.nn.functional.grid_sample(f10, torch.stack((gx, gy), -1), align_corners=True, padding_mode="border")
    return (f01 + back).norm(dim=1, keepdim=True)
