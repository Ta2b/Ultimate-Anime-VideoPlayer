"""Images du banc de qualité, en RGB float32 3 × H × W dans [0, 1].

  stills    les plans fixes de data/calibration/calibration.mkv (un par chapitre de 6 s), en 1920 × 1080 ;
  scenes    du mouvement de synthèse dont l'image du milieu est connue au pixel près : un fond (un plan fixe) qui
            glisse, et devant lui un personnage (une ellipse aux bords doux découpée dans un autre plan) qui glisse
            autrement ou tient la pose. Les déplacements sont entiers : aucune image n'est rééchantillonnée, la
            vérité est exacte, et le personnage découvre et recouvre le fond comme un cellulo ;
  frames    les images d'une vidéo (un épisode), pour le mouvement réel ;
  windows   dans cette vidéo, les fenêtres k … k+4 dont les trois premières images diffèrent.
"""
import subprocess

import numpy as np

# Déplacement par image, en pixels (x, y), du fond puis du personnage. Les réseaux reçoivent une image sur deux :
# entre leurs deux entrées, le mouvement est le double.
MOTIONS = (
    ("lent", (2, 0), (-1, 1)),
    ("moyen", (6, 2), (-4, 0)),
    ("rapide", (16, 4), (-12, 6)),
    ("pose", (8, 0), (0, 0)),
)
STEPS = 5  # images k … k+4 : entrées k, k+2 (et k+4 pour DistilDRBA), vérité k+1


def _decode(cmd, w, h):
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return [f.transpose(2, 0, 1).astype(np.float32) / 255.0 for f in arr]


def frames(video, size, limit):
    """Les images de la vidéo à la taille donnée, au plus limit."""
    w, h = size
    return _decode(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h}:flags=bicubic", "-frames:v",
                    str(limit), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], w, h)


def stills(video, count, size=(1920, 1080), period=144, first=72):
    """Une image au milieu de chaque plan de calibration.mkv (plans de 144 images), au plus count."""
    w, h = size
    select = f"select='not(mod(n-{first}\\,{period}))',scale={w}:{h}:in_color_matrix=bt709:in_range=tv:out_range=pc"
    return _decode(["ffmpeg", "-v", "error", "-i", video, "-vf", select, "-fps_mode", "passthrough", "-frames:v",
                    str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], w, h)


def _window(master, size, shift):
    """La fenêtre size de master, centrée puis décalée de shift (x, y)."""
    w, h = size
    _, mh, mw = master.shape
    x, y = (mw - w) // 2 + shift[0], (mh - h) // 2 + shift[1]
    if x < 0 or y < 0 or x + w > mw or y + h > mh:
        raise SystemExit(f"scène {w}x{h} : le mouvement sort de l'image {mw}x{mh} ; prendre une taille plus petite")
    return master[:, y:y + h, x:x + w]


def _mask(size, shift, edge=6.0):
    """Ellipse aux bords doux (edge pixels), centrée dans l'image puis décalée de shift."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ry, rx = 0.36 * h, 0.27 * h
    r = np.sqrt(((xs - w / 2 - shift[0]) / rx) ** 2 + ((ys - h / 2 - shift[1]) / ry) ** 2)
    t = np.clip((1.0 - r) * min(rx, ry) / edge, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t))[None]


def scene(background, figure, size, v_back, v_fig):
    """Les STEPS images d'une scène : le fond glisse de v_back par image, le personnage de v_fig."""
    out = []
    for s in range(STEPS):
        k = s - STEPS // 2
        back = _window(background, size, (k * v_back[0], k * v_back[1]))
        shift = (k * v_fig[0], k * v_fig[1])
        fig = _window(figure, size, (-shift[0], -shift[1]))  # le dessin suit son masque
        alpha = _mask(size, shift)
        out.append((back * (1.0 - alpha) + fig * alpha).astype(np.float32))
    return out


def scenes(plans, size, count):
    """count scènes nommées, tirées des plans fixes : (nom, images)."""
    out = []
    for i in range(count):
        name, v_back, v_fig = MOTIONS[i % len(MOTIONS)]
        b, f = i % len(plans), (i + 7) % len(plans)
        out.append((f"{name}-{b + 1}-{f + 1}", scene(plans[b], plans[f], size, v_back, v_fig)))
    return out


def differs(a, b, level=0.7 / 255.0):
    """Deux images différentes : écart moyen de luminance au-dessus de 0,7 niveau 8 bits (le seuil des tenus)."""
    return float(np.abs(a.mean(0) - b.mean(0)).mean()) > level


def windows(images, count, cut=0.12):
    """Dans une vidéo, count fenêtres de STEPS images (nom, images) où k, k+1 et k+2 diffèrent (l'anime se joue
    « sur deux » : un dessin tenu au milieu ne teste rien). Si k+4 est d'un autre plan (écart moyen au-dessus de
    cut), il est remplacé par k+2, comme le lecteur répète le bord devant une coupe."""
    cand = [k for k in range(len(images) - STEPS + 1)
            if differs(images[k], images[k + 1]) and differs(images[k + 1], images[k + 2])]
    if not cand:
        return []
    out = []
    for k in cand[::max(1, len(cand) // count)][:count]:
        win = list(images[k:k + STEPS])
        if float(np.abs(win[2] - win[4]).mean()) > cut:
            win[4] = win[2]
        out.append((f"image-{k}", win))
    return out


def halve(img):
    """Réduction ×0,5 par moyenne 2 × 2 : le filtre fixé du banc d'agrandissement."""
    _, h, w = img.shape
    return img[:, :h // 2 * 2, :w // 2 * 2].reshape(3, h // 2, 2, w // 2, 2).mean((2, 4)).astype(np.float32)


def _cubic(x, a=-0.5):
    x = np.abs(x)
    return np.where(x <= 1, (a + 2) * x**3 - (a + 3) * x**2 + 1,
                    np.where(x < 2, a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a, 0.0))


def bicubic2(img):
    """×2 bicubique séparable (Keys, a = −0,5), centres demi-pixel, bords répétés : le repère sans réseau."""
    def axis(n):
        pos = (np.arange(2 * n) + 0.5) / 2 - 0.5
        base = np.floor(pos).astype(int)
        frac = pos - base
        return [np.clip(base + o, 0, n - 1) for o in (-1, 0, 1, 2)], [_cubic(frac - o) for o in (-1, 0, 1, 2)]

    _, h, wd = img.shape
    (iy, wy), (ix, wx) = axis(h), axis(wd)
    rows = sum(img[:, i, :] * w[None, :, None] for i, w in zip(iy, wy))
    return sum(rows[:, :, i] * w[None, None, :] for i, w in zip(ix, wx))
