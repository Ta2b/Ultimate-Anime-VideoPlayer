"""FlowJaNai : interpolation d'anime guidée par le flux optique matériel de NVIDIA (NVOF).

NVOF donne, sans réseau, le mouvement grossier de la paire dans les deux sens (grille 4 × 4, un
vecteur par bloc) ; il suit bien les grands déplacements, le point faible des estimateurs légers.
Le réseau ne fait que l'affiner :

  1. le mouvement à l'instant t est tiré des deux flux NVOF par la loi quadratique de QVI
     (F_t→0 = −(1 − t) t F01 + t² F10, F_t→1 = (1 − t)² F01 − t (1 − t) F10) ;
  2. la cohérence avant-arrière des deux flux (|F01 + F10 ∘ F01|) dit au réseau où NVOF se
     trompe (occultations, aplats sans texture, traits fins) ;
  3. trois étages de type RIFE, à 1/4, 1/2 puis 1/1 de la résolution, corrigent le mouvement et
     estiment le masque de mélange, chacun sur les images déformées par l'étage précédent.

La sortie est celle de la variante « mouvement » de RIFE dans le lecteur (kernels/spec/warp.h) :
flot (f0x, f0y, f1x, f1y) en pixels, de l'instant vers a et vers b, et masque (poids de a, dans
[0, 1]). Le warp et le mélange se font donc dans le lecteur, à la taille de travail ou de sortie,
et une passe au milieu de la paire peut servir à tous ses instants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp arrière bilinéaire, bords en clamp, centres des pixels aux entiers (comme le lecteur)."""
    _, _, h, w = image.shape
    # Range sur des entiers, puis converti : le moteur Vulkan (gpu/ir) n'a pas de Range flottant.
    ys = torch.arange(h, device=image.device).to(image.dtype).view(1, h, 1).expand(1, h, w)
    xs = torch.arange(w, device=image.device).to(image.dtype).view(1, 1, w).expand(1, h, w)
    x = (xs + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    y = (ys + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((x, y), dim=3)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def resize_flow(flow: torch.Tensor, factor: float) -> torch.Tensor:
    """Le flot mis à une autre échelle : la grille et les déplacements ensemble."""
    return F.interpolate(flow, scale_factor=factor, mode="bilinear", align_corners=False) * factor


def conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.PReLU(cout))


class Refine(nn.Module):
    """Un étage : deux réductions par 2, des convolutions résiduelles, puis une remontée ×4 qui
    rend la correction du flot (4 plans) et du logit du masque (1 plan), à l'échelle de l'étage."""

    def __init__(self, cin: int, width: int, depth: int) -> None:
        super().__init__()
        self.down = nn.Sequential(conv(cin, width // 2, 2), conv(width // 2, width, 2))
        self.body = nn.Sequential(*[conv(width, width) for _ in range(depth)])
        self.up = nn.ConvTranspose2d(width, 4 * 5, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.body(y) + y
        return F.pixel_shuffle(self.up(y), 2)


class FlowJaNai(nn.Module):
    """forward(img0, img1, t, nvof) → (flot, masque), à la taille des images.

    img0, img1 : 1 × 3 × H × W dans [0, 1] ; t : 1 × 1 × H × W (le plan d'instant de rife_input) ;
    nvof : 1 × 4 × H/4 × W/4, (F01x, F01y, F10x, F10y) en pixels de l'image, la grille 4 de NVOF.
    H et W multiples de 64.
    """

    STAGES = ((4, 128, 6), (2, 96, 4), (1, 48, 2))  # (réduction, largeur, profondeur)

    def __init__(self) -> None:
        super().__init__()
        # Entrées d'un étage : deux images, deux images déformées, le flot (4), le logit du
        # masque (1), la cohérence (2), l'instant (1).
        cin = 3 * 4 + 4 + 1 + 2 + 1
        self.stages = nn.ModuleList(Refine(cin, width, depth) for _, width, depth in self.STAGES)

    @staticmethod
    def initial_flow(nvof: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f01, f10 = nvof[:, 0:2], nvof[:, 2:4]
        ft0 = -(1.0 - t) * t * f01 + t * t * f10
        ft1 = (1.0 - t) * (1.0 - t) * f01 - t * (1.0 - t) * f10
        return torch.cat((ft0, ft1), 1)

    @staticmethod
    def consistency(nvof: torch.Tensor) -> torch.Tensor:
        """Écart avant-arrière de chaque flux, en pixels, compressé (log1p) pour le réseau."""
        f01, f10 = nvof[:, 0:2], nvof[:, 2:4]
        e0 = (f01 + warp(f10, f01)).abs().sum(1, keepdim=True)
        e1 = (f10 + warp(f01, f10)).abs().sum(1, keepdim=True)
        return torch.log1p(torch.cat((e0, e1), 1))

    def forward(self, img0, img1, t, nvof):
        t4 = F.interpolate(t, scale_factor=0.25, mode="nearest")
        flow = self.initial_flow(nvof, t4)  # à 1/4, en pixels de l'image
        consistency = self.consistency(nvof)
        logit = torch.zeros_like(flow[:, :1])
        scale = 4
        for (reduce, _, _), stage in zip(self.STAGES, self.stages):
            if reduce != scale:  # l'étage précédent, remonté à l'échelle de celui-ci
                factor = scale / reduce
                flow = F.interpolate(flow, scale_factor=factor, mode="bilinear", align_corners=False)
                logit = F.interpolate(logit, scale_factor=factor, mode="bilinear", align_corners=False)
                consistency = F.interpolate(consistency, scale_factor=factor, mode="bilinear", align_corners=False)
                scale = reduce
            a = F.interpolate(img0, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) if scale > 1 else img0
            b = F.interpolate(img1, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) if scale > 1 else img1
            ts = F.interpolate(t, scale_factor=1.0 / scale, mode="nearest") if scale > 1 else t
            local = flow / scale  # en pixels de l'étage
            x = torch.cat((a, b, warp(a, local[:, 0:2]), warp(b, local[:, 2:4]), local, logit, consistency, ts), 1)
            delta = stage(x)
            flow = flow + delta[:, 0:4] * scale
            logit = logit + delta[:, 4:5]
        return flow, torch.sigmoid(logit)


def blend(img0, img1, flow, mask):
    """L'image à l'instant t, comme le warp du lecteur : a m + b (1 − m)."""
    return warp(img0, flow[:, 0:2]) * mask + warp(img1, flow[:, 2:4]) * (1.0 - mask)
