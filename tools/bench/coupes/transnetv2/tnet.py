# TransNetV2 (Souček & Lokoč, 2020, MIT) réécrit pour l'essai : seuls les poids viennent de l'extérieur.
import torch, torch.nn as nn, torch.nn.functional as F

class Conv(nn.Module):  # (2+1)D : 3x3 spatial, puis 3 temporel dilaté
    def __init__(s, cin, f, dil):
        super().__init__()
        s.layers = nn.ModuleList([nn.Conv3d(cin, 2 * f, (1, 3, 3), padding=(0, 1, 1), bias=False),
                                  nn.Conv3d(2 * f, f, (3, 1, 1), dilation=(dil, 1, 1), padding=(dil, 0, 0), bias=False)])
    def forward(s, x):
        return s.layers[1](s.layers[0](x))

class DDCNN(nn.Module):
    def __init__(s, cin, f, act):
        super().__init__()
        s.Conv3D_1, s.Conv3D_2, s.Conv3D_4, s.Conv3D_8 = (Conv(cin, f, d) for d in (1, 2, 4, 8))
        s.bn = nn.BatchNorm3d(4 * f, eps=1e-3)
        s.act = act
    def forward(s, x):
        x = s.bn(torch.cat([s.Conv3D_1(x), s.Conv3D_2(x), s.Conv3D_4(x), s.Conv3D_8(x)], 1))
        return F.relu(x) if s.act else x

class Stacked(nn.Module):
    def __init__(s, cin, f, n=2):
        super().__init__()
        s.DDCNN = nn.ModuleList([DDCNN(cin if i == 0 else 4 * f, f, i != n - 1) for i in range(n)])
    def forward(s, x):
        short = None
        for b in s.DDCNN:
            x = b(x)
            short = x if short is None else short
        return F.avg_pool3d(F.relu(x) + short, (1, 2, 2))

def band(sim, w=101):  # sim[b, t, t + k - 50], k de 0 à 100, zéro hors de la fenêtre
    h = (w - 1) // 2
    p = F.pad(sim, [h, h])
    T = sim.shape[1]
    idx = torch.arange(T).view(T, 1) + torch.arange(w).view(1, w)
    return p[:, torch.arange(T).view(T, 1), idx]

class FrameSim(nn.Module):
    def __init__(s, cin):
        super().__init__()
        s.projection = nn.Linear(cin, 128)
        s.fc = nn.Linear(101, 128)
    def forward(s, feats):
        x = torch.cat([f.mean(dim=[3, 4]) for f in feats], 1).transpose(1, 2)
        x = F.normalize(s.projection(x), dim=2)
        return F.relu(s.fc(band(x @ x.transpose(1, 2))))

class ColorHist(nn.Module):
    def __init__(s):
        super().__init__()
        s.fc = nn.Linear(101, 128)
    def forward(s, frames):  # uint8 [B, T, H, W, 3]
        q = (frames.long() >> 5)
        bins = (q[..., 0] << 6) + (q[..., 1] << 3) + q[..., 2]
        B, T = bins.shape[:2]
        h = F.one_hot(bins.view(B, T, -1), 512).sum(2).float()
        h = F.normalize(h, dim=2)
        return F.relu(s.fc(band(h @ h.transpose(1, 2))))

class TransNetV2(nn.Module):
    def __init__(s, f=16, L=3):
        super().__init__()
        s.SDDCNN = nn.ModuleList([Stacked(3, f)] + [Stacked(f * 2 ** (i - 1) * 4, f * 2 ** i) for i in range(1, L)])
        s.frame_sim_layer = FrameSim(sum(f * 2 ** i * 4 for i in range(L)))
        s.color_hist_layer = ColorHist()
        s.fc1 = nn.Linear(f * 2 ** (L - 1) * 4 * 3 * 6 + 256, 1024)
        s.cls_layer1 = nn.Linear(1024, 1)
        s.cls_layer2 = nn.Linear(1024, 1)
    def forward(s, frames):  # uint8 [B, T, 27, 48, 3] -> probabilité de coupe après chaque image [B, T]
        x = frames.permute(0, 4, 1, 2, 3).float() / 255
        feats = []
        for blk in s.SDDCNN:
            x = blk(x)
            feats.append(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], x.shape[2], -1)
        x = torch.cat([s.color_hist_layer(frames), s.frame_sim_layer(feats), x], 2)
        return torch.sigmoid(s.cls_layer1(F.relu(s.fc1(x))))[..., 0]

def predict(model, frames):  # uint8 [N, 27, 48, 3] : fenêtres de 100, pas de 50, 25 de bord
    n = len(frames)
    end_pad = 25 + 50 - (n % 50 if n % 50 else 50)
    padded = torch.cat([frames[:1].repeat(25, 1, 1, 1), frames, frames[-1:].repeat(end_pad, 1, 1, 1)])
    out = []
    with torch.no_grad():
        for p in range(0, len(padded) - 99, 50):
            out.append(model(padded[p:p + 100][None])[0, 25:75])
    return torch.cat(out)[:n]
