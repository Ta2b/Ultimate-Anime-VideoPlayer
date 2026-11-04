#!/usr/bin/env python3
"""L'entraînement du réseau des émetteurs (méga plan A6) et son export ONNX. Le code est ici ; l'entraînement
se fera plus tard, sur un PC avec PyTorch (pip install torch onnx), pas dans le cloud.

Le contrat du modèle, celui que l'étage TensorRT lira :
  entrée  « pq »        1 × 3 × H × W, float16, codes PQ (ST 2084) BT.2020 : l'image que voit le grade ;
  sortie  « emitters »  1 × 1 × H/8 × W/8, float16, de 0 à 1 : la part d'émetteur de chaque bloc 8 × 8,
                        à la place de la carte des portes du shader (anime-hdr-anime/3-anime.hook).
H et W multiples de 8 ; le réseau est entièrement convolutif (toute taille).

    train.py --data DOSSIER --epochs 20 --out emitters.onnx
    train.py --self-test          (sans PyTorch : le contrat et la conversion PQ)
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

INPUT_NAME = "pq"
OUTPUT_NAME = "emitters"
STRIDE = 8
OPSET = 17

M1 = 2610.0 / 16384.0
M2 = 2523.0 / 4096.0 * 128.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0


def nits_to_pq(nits):
    y = np.power(np.clip(np.asarray(nits, dtype=np.float32) / 10000.0, 0.0, 1.0), M1)
    return np.power((C1 + C2 * y) / (1.0 + C3 * y), M2)


def load(data):
    files = sorted(glob.glob(os.path.join(data, "tiles-*.npz")))
    corrections = {}
    path = os.path.join(data, "corrections.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4 and not parts[0].startswith("#"):
                    corrections.setdefault(parts[0], []).append((int(parts[1]), int(parts[2]), float(parts[3])))
    for name in files:
        with np.load(name) as z:
            rgb = nits_to_pq(z["rgb"].astype(np.float32))
            label = z["label"].astype(np.float32).copy()
        for x8, y8, value in corrections.get(os.path.basename(name), []):
            label[y8, x8] = value
        yield rgb, label


def build_model(torch):
    nn = torch.nn

    # Trois réductions par 2 (le bloc 8 × 8), la couronne vue par des convolutions dilatées au 1/8.
    class Emitters(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU())
            self.context = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU())
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, pq):
            d = self.down(pq)
            return torch.sigmoid(self.head(torch.cat([d, self.context(d)], dim=1)))

    return Emitters()


def train(args):
    import torch  # noqa: PLC0415 (seulement à l'entraînement)

    pairs = list(load(args.data))
    if not pairs:
        print(f"aucune tuile dans {args.data} (dataset.py d'abord)", file=sys.stderr)
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x = torch.tensor(np.stack([p[0] for p in pairs]))
    y = torch.tensor(np.stack([p[1] for p in pairs]))[:, None]
    for epoch in range(args.epochs):
        order = torch.randperm(len(x))
        total = 0.0
        for at in range(0, len(x), args.batch):
            idx = order[at:at + args.batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            if torch.rand(1).item() < 0.5:  # l'image retournée : la même lumière
                xb, yb = xb.flip(-1), yb.flip(-1)
            loss = torch.nn.functional.binary_cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"époque {epoch + 1}/{args.epochs} : perte {total / len(x):.4f}")
    model = model.eval().half().to(device)
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device)
    torch.onnx.export(model, dummy, args.out, input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                      opset_version=OPSET,
                      dynamic_axes={INPUT_NAME: {2: "h", 3: "w"}, OUTPUT_NAME: {2: "h8", 3: "w8"}})
    print(f"modèle écrit : {args.out}")
    return 0


def self_test():
    # Le blanc SDR (203 nits) et 1 000 nits aux codes PQ connus ; le contrat tient en ses constantes.
    ok = abs(float(nits_to_pq(203.0)) - 0.58069) < 1e-4 and abs(float(nits_to_pq(1000.0)) - 0.75183) < 1e-4
    ok = ok and STRIDE == 8 and INPUT_NAME == "pq" and OUTPUT_NAME == "emitters"
    print(f"train.py : contrat et PQ {'ok' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="emitters-data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", default="emitters.onnx")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    return self_test() if args.self_test else train(args)


if __name__ == "__main__":
    sys.exit(main())
