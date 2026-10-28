"""Aides communes : images PPM, entree RIFE (11 plans), modeles ONNX a formes fixes."""
import numpy as np
import onnx
from onnx import shape_inference

H, W = 768, 1280
MODELE = "/home/ta2b/ajp/models/rife/rife_v4.25.onnx"
IMAGES = {n: f"/tmp/claude-1000/poses-A/{n}.ppm" for n in ("s5", "s9", "s14")}


def lire_ppm(chemin):
    """PPM P6 8 bits -> tableau (h, w, 3) uint8."""
    with open(chemin, "rb") as f:
        donnees = f.read()
    jetons, pos = [], 0
    while len(jetons) < 4:
        while donnees[pos:pos + 1].isspace():
            pos += 1
        if donnees[pos:pos + 1] == b"#":
            pos = donnees.index(b"\n", pos)
            continue
        debut = pos
        while not donnees[pos:pos + 1].isspace():
            pos += 1
        jetons.append(donnees[debut:pos])
    pos += 1
    w, h = int(jetons[1]), int(jetons[2])
    return np.frombuffer(donnees, dtype=np.uint8, offset=pos).reshape(h, w, 3)


def entree_rife(a, b, t=0.5):
    """a, b : (H, W, 3) uint8 -> tenseur (1, 11, H, W) fp16 comme rife_input (grille normalisee fp16)."""
    h, w = a.shape[:2]
    x = np.zeros((1, 11, h, w), np.float16)
    x[0, 0:3] = (a.astype(np.float32) / 255.0).transpose(2, 0, 1)
    x[0, 3:6] = (b.astype(np.float32) / 255.0).transpose(2, 0, 1)
    x[0, 6] = np.float16(t)
    gx = 2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0
    gy = 2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0
    x[0, 7] = gx[None, :]
    x[0, 8] = gy[:, None]
    x[0, 9] = np.float16(2.0 / (w - 1))
    x[0, 10] = np.float16(2.0 / (h - 1))
    return x


def paire(nom, decalage, x0=900, y0=300):
    """Recadrage W x H de l'image `nom` et du meme recadrage decale de `decalage` px (horizontal)."""
    img = lire_ppm(IMAGES[nom])
    a = img[y0:y0 + H, x0:x0 + W]
    b = img[y0:y0 + H, x0 + decalage:x0 + decalage + W]
    return a, b


def modele_fixe(chemin=MODELE):
    """Modele ONNX avec entree 1x11xHxW fixe et formes inferees."""
    m = onnx.load(chemin)
    d = m.graph.input[0].type.tensor_type.shape.dim
    for dim, v in zip(d, (1, 11, H, W)):
        dim.ClearField("dim_param")
        dim.dim_value = v
    del m.graph.value_info[:]
    return shape_inference.infer_shapes(m)


def psnr(a, b, crete=1.0):
    d = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 200.0 if d == 0 else 10.0 * np.log10(crete * crete / d)


def ssim(a, b, crete=1.0, fenetre=7):
    """SSIM a fenetre uniforme 7x7 en float64, moyenne sur les canaux (a, b : (C, H, W))."""
    from numpy.lib.stride_tricks import sliding_window_view
    c1, c2 = (0.01 * crete) ** 2, (0.03 * crete) ** 2
    n = fenetre * fenetre
    out = []
    for c in range(a.shape[0]):
        x, y = a[c].astype(np.float64), b[c].astype(np.float64)

        def moy(z):
            s = np.cumsum(np.cumsum(np.pad(z, ((1, 0), (1, 0))), 0), 1)
            return (s[fenetre:, fenetre:] - s[:-fenetre, fenetre:] - s[fenetre:, :-fenetre] + s[:-fenetre, :-fenetre]) / n

        mx, my = moy(x), moy(y)
        vx = (moy(x * x) - mx * mx) * n / (n - 1)
        vy = (moy(y * y) - my * my) * n / (n - 1)
        cxy = (moy(x * y) - mx * my) * n / (n - 1)
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        out.append(s.mean())
    return float(np.mean(out))
