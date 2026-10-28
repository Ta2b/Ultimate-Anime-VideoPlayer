#!/usr/bin/env python3
"""Entraîne FlowJaNai (model.py) sur des triplets d'anime et leurs flux NVOF (dataset.py).

    python -I tools/nvof_interp/train.py --train atd12k/train_10k --val atd12k/test_2k_540p \\
        --out runs/flowjanai

Perte : Charbonnier sur l'image mélangée, plus une perte laplacienne (pyramide de 5 niveaux,
celle de RIFE) qui garde les traits nets. AdamW, taux en cosinus, précision mixte. Un point de
reprise par époque (last.pt) et le meilleur PSNR de validation (best.pt).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import Triplets  # noqa: E402
from model import FlowJaNai, blend  # noqa: E402


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def gauss_kernel(channels: int, device) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    k = torch.outer(k, k)
    return (k / k.sum()).expand(channels, 1, 5, 5).contiguous()


def laplacian_loss(x: torch.Tensor, y: torch.Tensor, levels: int = 5) -> torch.Tensor:
    kernel = gauss_kernel(x.shape[1], x.device)
    total = x.new_zeros(())
    for _ in range(levels):
        bx = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel, groups=x.shape[1])
        by = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), kernel, groups=y.shape[1])
        total = total + (x - bx - (y - by)).abs().mean()
        x, y = bx[:, :, ::2, ::2], by[:, :, ::2, ::2]
    return total


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x.clamp(0, 1), y).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def half_plane(img: torch.Tensor) -> torch.Tensor:
    return torch.full_like(img[:, :1], 0.5)


@torch.no_grad()
def validate(net: FlowJaNai, loader: DataLoader, device) -> float:
    net.eval()
    scores = []
    for a, b, gt, flow in loader:
        a, b, gt, flow = (v.to(device) for v in (a, b, gt, flow))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out_flow, mask = net(a, b, half_plane(a), flow)
        scores.append(psnr(blend(a, b, out_flow.float(), mask.float()), gt))
    net.train()
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="repart de out/last.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    train = DataLoader(Triplets(args.train, args.crop, train=True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    val = DataLoader(Triplets(args.val, train=False), batch_size=1, num_workers=2) if args.val else None

    net = FlowJaNai().to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=steps, pct_start=0.02,
                                                anneal_strategy="cos", final_div_factor=100.0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    start, best = 0, 0.0
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        net.load_state_dict(state["net"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"reprise à l'époque {start}, meilleur PSNR {best:.2f} dB")

    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M paramètres, {len(train)} lots par époque")
    for epoch in range(start, args.epochs):
        total = 0.0
        for a, b, gt, flow in train:
            a, b, gt, flow = (v.to(device, non_blocking=True) for v in (a, b, gt, flow))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out_flow, mask = net(a, b, half_plane(a), flow)
            out = blend(a, b, out_flow.float(), mask.float())
            loss = charbonnier(out, gt) + 0.5 * laplacian_loss(out, gt)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item()
        line = f"époque {epoch} : perte {total / len(train):.4f}"
        if val is not None:
            score = validate(net, val, device)
            line += f", PSNR {score:.2f} dB"
            if score > best:
                best = score
                torch.save({"net": net.state_dict(), "psnr": score}, args.out / "best.pt")
        print(line, flush=True)
        torch.save({"net": net.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, args.out / "last.pt")


if __name__ == "__main__":
    main()
