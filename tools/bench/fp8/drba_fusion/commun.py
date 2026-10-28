"""Aides communes : images PPM, entrées DistilDRBA (pair01, pair12 en 7 plans), noms de tenseurs aux coupes."""
import numpy as np

H, W = 1088, 1920
MODELE = "/home/ta2b/.local/share/ajp/ajp/modeles/distil_drba_v2_lite-1920x1088-7048367e94f6c43d.onnx"
IMAGES = {n: f"/tmp/claude-1000/poses-A/{n}.ppm" for n in ("s5", "s9", "s14")}

# Tenseurs aux coupes (graphe plié par plie.py).
ENC_IN = ["/Resize_output_0/cast16", "/Resize_1_output_0/cast16", "/Resize_2_output_0/cast16"]
ENC_OUT = ["/encode/cnn3/ConvTranspose_output_0", "/encode/cnn3_1/ConvTranspose_output_0",
           "/encode/cnn3_2/ConvTranspose_output_0"]
BLOC_IN = ["/Concat_output_0", "/block1/Concat_output_0", "/block2/Concat_output_0"]
BLOC_OUT = [f"/block{b}/lastconv/lastconv.0/ConvTranspose_output_0" for b in range(3)]


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


def trois_images(nom, decalage, x0=300, y0=150):
    """Trois recadrages W x H de l'image `nom`, décalés de `decalage` px chacun (mouvement régulier)."""
    img = lire_ppm(IMAGES[nom])
    return [img[y0:y0 + H, x0 + k * decalage:x0 + k * decalage + W] for k in range(3)]


def entrees_drba(imgs, tau=0.5):
    """(img0, img1, img2) uint8 (H, W, 3) -> (pair01, pair12) 1 x 7 x H x W fp16 comme rife_input (plan 6 = tau / 2)."""
    f = [np.ascontiguousarray((i.astype(np.float32) / 255.0).transpose(2, 0, 1)) for i in imgs]
    u = np.float16(tau / 2)
    p01 = np.zeros((1, 7, H, W), np.float16)
    p12 = np.zeros((1, 7, H, W), np.float16)
    p01[0, 0:3], p01[0, 3:6], p01[0, 6] = f[0], f[1], u
    p12[0, 0:3], p12[0, 3:6], p12[0, 6] = f[1], f[2], u
    return p01, p12


# (image, décalage en px, tau) : six cas à tau = 0,5 et un cas de l'autre côté de img1.
CAS = [(n, d, 0.5) for n in ("s5", "s9", "s14") for d in (6, 20)] + [("s9", 20, 1.25)]


def nom_cas(c):
    return f"{c[0]}_{c[1]}" + ("" if c[2] == 0.5 else f"_t{c[2]}")


def psnr(a, b, crete=1.0):
    d = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 200.0 if d == 0 else 10.0 * np.log10(crete * crete / d)


def ssim(a, b, crete=1.0, fenetre=7):
    """SSIM à fenêtre uniforme 7x7 en float64, moyenne sur les canaux (a, b : (C, H, W))."""
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
