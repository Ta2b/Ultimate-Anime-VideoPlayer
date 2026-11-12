"""Les réseaux du lecteur sous ONNX Runtime, nourris comme le lecteur les nourrit.

  SPAN (agrandissement ×2)   une entrée RGB 1 × 3 × H × W ;
  RIFE                       une entrée de 11 plans (v1) ou 7 (v2) : src/kernels/spec/rife.h ;
  DistilDRBA                 deux entrées de 7 plans, pair01 et pair12 : src/kernels/spec/distil_drba.h.

Un réseau s'ouvre tel que livré (FP16) ou en référence FP32 (fp32.py). Sur processeur, ONNX Runtime range les
tenseurs FP16 en demi-précision mais fait les sommes de chaque couche en FP32 : la mesure « livré contre FP32 » est
donc un plancher de ce que perd un moteur qui somme aussi en FP16 (TensorRT, Vulkan).
"""
import os
import time

import numpy as np

ALIGN = 64  # kRifeAlign ; 128 pour un export de DistilDRBA à l'échelle 0,5 (« _s05 ») et pour RIFE 4.25 lite
# (kRifeAlignLite, rife_padding_multiple : son premier bloc travaille au 1/32, ses convolutions au 1/128)


def _session(model, provider):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(model, so, providers=providers)


def session(path, provider="cpu"):
    """Le fichier tel quel (utilisé aussi par span_flicker.py)."""
    return _session(path, provider)


def pad(img, align=ALIGN):
    """L'image centrée dans un tenseur au multiple de align, bords noirs : rife_window (bord gauche et haut pairs).
    Rend le tenseur et la fenêtre (haut, gauche)."""
    _, h, w = img.shape
    ph, pw = (h + align - 1) // align * align, (w + align - 1) // align * align
    top, left = ((ph - h) // 2) & ~1, ((pw - w) // 2) & ~1
    out = np.zeros((3, ph, pw), np.float32)
    out[:, top:top + h, left:left + w] = img
    return out, (top, left)


def rife_input(a, b, t, planes):
    """rife_input du lecteur sur deux images déjà remplies : 7 plans, ou 11 avec la grille du v1."""
    _, h, w = a.shape
    x = np.empty((1, planes, h, w), np.float32)
    x[0, 0:3], x[0, 3:6], x[0, 6] = a, b, t
    if planes == 11:
        xs = 2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0
        ys = 2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0
        x[0, 7], x[0, 8] = xs[None, :], ys[:, None]
        x[0, 9], x[0, 10] = 2.0 / (w - 1), 2.0 / (h - 1)
    return x


class Network:
    """Un réseau ouvert : kind vaut « span », « rife » ou « drba » ; seconds cumule le temps d'ONNX Runtime."""

    def __init__(self, path, reference=False, provider="cpu"):
        self.name = os.path.basename(path)
        if reference:
            import fp32  # ici seulement : span_flicker.py ouvre un réseau sans le paquet onnx

            self.sess = _session(fp32.promoted_bytes(path), provider)
        else:
            self.sess = _session(path, provider)
        self.inputs = self.sess.get_inputs()
        self.dtype = np.float16 if "float16" in self.inputs[0].type else np.float32
        planes = self.inputs[0].shape[1]
        if len(self.inputs) == 2 and planes == 7:
            self.kind = "drba"
        elif len(self.inputs) == 1 and planes in (7, 11):
            self.kind = "rife"
        elif len(self.inputs) == 1 and planes == 3:
            self.kind = "span"
        else:
            raise SystemExit(f"{path} : ni SPAN (3 plans), ni RIFE (7 ou 11), ni DistilDRBA (deux entrées de 7)")
        self.planes = planes
        self.align = 128 if "_s05" in self.name or self.name.startswith("rife_v4.25_lite") else ALIGN
        self.seconds, self.runs = 0.0, 0

    def _run(self, feeds):
        feeds = {k: v.astype(self.dtype) for k, v in feeds.items()}
        start = time.perf_counter()
        out = self.sess.run(None, feeds)[0]
        self.seconds += time.perf_counter() - start
        self.runs += 1
        return np.clip(out[0].astype(np.float32), 0.0, 1.0)

    def upscale(self, img):
        return self._run({self.inputs[0].name: img[None]})

    def middle(self, win):
        """L'image du milieu de win[0] et win[2] (la vraie est win[1]). DistilDRBA reçoit aussi win[4], centré sur
        win[2] comme drba_call(0,5) : u = 0,25."""
        _, h, w = win[0].shape
        (a, (top, left)), (b, _), (c, _) = (pad(win[i], self.align) for i in (0, 2, 4))
        if self.kind == "drba":
            feeds = {self.inputs[0].name: rife_input(a, b, 0.25, 7), self.inputs[1].name: rife_input(b, c, 0.25, 7)}
        else:
            feeds = {self.inputs[0].name: rife_input(a, b, 0.5, self.planes)}
        return self._run(feeds)[:3, top:top + h, left:left + w]

    def ms(self):
        return 1000.0 * self.seconds / max(1, self.runs)
