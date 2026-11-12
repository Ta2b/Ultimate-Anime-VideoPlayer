"""Ordre du warp et de SPAN : ce que l'image créée perd quand le warp passe après SPAN (audits C09-02, C32-01).

Le lecteur crée une image intermédiaire de deux façons, avec la variante mouvement de RIFE (flot et masque) :

  avant    le warp et le mélange à la taille de travail, puis SPAN sur l'image créée (une passe de SPAN par
           image créée) ;
  après    SPAN sur chaque source (une passe par source), puis le warp et le mélange des images agrandies, le
           flot et le masque lus à la taille de travail et mis à l'échelle (src/kernels/spec/warp.h).

Le banc joue les deux sur le mouvement de synthèse de sources.py, à l'échelle ½ comme travail.py : la scène est
faite à la taille de sortie (la vérité), sa moitié (moyenne 2 × 2) est la source donnée à RIFE et à SPAN. Chaque
image créée se compare :

  à la vraie image du milieu, à la taille de sortie ;
  à SPAN de la vraie image du milieu à la taille de travail (« comme une source ») : ce que serait l'image si
  elle était une source. L'écart dit si l'image créée « pompe » face aux sources qui l'entourent ;
  par sa netteté : l'énergie du gradient de la luminance, rapportée à celle de SPAN de la vraie image.

  python3 -I tools/bench/qualite/ordre.py --modeles ~/.cache/ajp-release-models/modeles --md ordre.md

Limites : un mouvement de translation de calques (pas de déformation ni d'animation « sur deux »), à l'échelle ½
(les traits sont deux fois plus fins que ceux que SPAN a appris) ; ONNX Runtime sur processeur, pas TensorRT ;
le warp en float64, sans l'arrondi du warp CUDA par texture.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesures  # noqa: E402
import modeles  # noqa: E402
import sources  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SPAN = "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
RIFE = ("rife_v4.22_lite.onnx", "rife_v4.26.onnx")
# Les tenseurs du flot final et du masque (src/app/play/motion_model.cpp, kFlowTensors et kMaskTensor).
FLOW_TENSORS = ("/Add_7_output_0", "/Add_5_output_0")
MASK_TENSOR = "/Sigmoid_output_0"
WARP_DEPTH = 8


def _final_flow(graph):
    """Le flot final comme is_final_flow : un seul consommateur, un Reshape qui mène à un GridSample."""
    users = {}
    for node in graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    for flow in FLOW_TENSORS:
        used = users.get(flow, [])
        if len(used) != 1 or used[0].op_type != "Reshape":
            continue
        front = list(used[0].output)
        for _ in range(WARP_DEPTH):
            nxt = []
            for t in front:
                for node in users.get(t, []):
                    if node.op_type == "GridSample":
                        return flow
                    nxt.extend(node.output)
            front = nxt
    raise SystemExit("pas de flot final reconnu")


class Motion:
    """RIFE avec le flot final et le masque en sorties, comme la variante mouvement du lecteur."""

    def __init__(self, path):
        import onnx

        model = onnx.load(path)
        flow = _final_flow(model.graph)
        elem = model.graph.input[0].type.tensor_type.elem_type
        for name in (flow, MASK_TENSOR):
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, elem, None))
        self.sess = modeles._session(model.SerializeToString(), "cpu")
        self.input = self.sess.get_inputs()[0]
        self.dtype = np.float16 if "float16" in self.input.type else np.float32
        self.planes = self.input.shape[1]
        self.align = 128 if os.path.basename(path).startswith("rife_v4.25_lite") else modeles.ALIGN
        self.names = [flow, MASK_TENSOR]

    def flow_mask(self, a, b):
        """Flot (4 × h × w, en pixels de la fenêtre) et masque (h × w, poids de a) au milieu de a et b."""
        _, h, w = a.shape
        (pa, (top, left)), (pb, _) = modeles.pad(a, self.align), modeles.pad(b, self.align)
        x = modeles.rife_input(pa, pb, 0.5, self.planes).astype(self.dtype)
        flow, mask = self.sess.run(self.names, {self.input.name: x})
        flow = flow[0].astype(np.float64)[:, top:top + h, left:left + w]
        mask = mask[0, 0].astype(np.float64)[top:top + h, left:left + w]
        return flow, mask


def bilinear(plane, x, y):
    """Échantillon bilinéaire d'un plan (… × h × w) en (x, y), centres aux entiers, bords en clamp (warp.h)."""
    h, w = plane.shape[-2:]
    x = np.clip(np.nan_to_num(x), 0.0, w - 1)
    y = np.clip(np.nan_to_num(y), 0.0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = plane[..., y0, x0] * (1 - fx) + plane[..., y0, x1] * fx
    bottom = plane[..., y1, x0] * (1 - fx) + plane[..., y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def warp_blend(a, b, flow, mask):
    """Le warp et le mélange de warp.h à t = 1/2 : a et b (3 × H × W) peuvent être plus grandes que le flot."""
    _, height, width = a.shape
    fh, fw = mask.shape
    sx, sy = width / fw, height / fh
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    u, v = (xs + 0.5) / sx - 0.5, (ys + 0.5) / sy - 0.5
    f = bilinear(flow, u, v)
    m = bilinear(mask, u, v)
    wa = bilinear(a.astype(np.float64), xs + sx * f[0], ys + sy * f[1])
    wb = bilinear(b.astype(np.float64), xs + sx * f[2], ys + sy * f[3])
    return np.clip(wa * m + wb * (1 - m), 0.0, 1.0).astype(np.float32)


def sharpness(img):
    """Énergie moyenne du gradient de la luminance."""
    y = mesures.luma(img)
    return float(np.mean(np.diff(y, axis=0) ** 2) + np.mean(np.diff(y, axis=1) ** 2))


def run(args):
    plans = sources.stills(args.calibration, args.plans)
    if not plans:
        raise SystemExit(f"{args.calibration} : aucun plan lu")
    size = (args.largeur, args.largeur * 9 // 16 // 2 * 2)
    span = modeles.Network(os.path.join(args.modeles, "onnx", SPAN))
    rows = {}
    for rife_file in RIFE:
        motion = Motion(os.path.join(args.modeles, "rife", rife_file))
        for name, images in sources.scenes(plans, size, args.scenes):
            allure = name.split("-")[0]
            truth = images[2]  # la vraie image du milieu, à la taille de sortie
            a, mid, b = (sources.halve(images[i]) for i in (1, 2, 3))  # entrées k, k+2 ; vérité k+1
            flow, mask = motion.flow_mask(a, b)
            span_a, span_b, span_mid = span.upscale(a), span.upscale(b), span.upscale(mid)
            made = {
                "avant": span.upscale(warp_blend(a, b, flow, mask)),
                "après": warp_blend(span_a, span_b, flow, mask),
            }
            base = sharpness(span_mid)
            for order, img in made.items():
                for key in (allure, "toutes"):
                    rows.setdefault((rife_file, key, order), []).append({
                        "vrai": mesures.psnr(truth, img),
                        "source": mesures.psnr(span_mid, img),
                        "nettete": sharpness(img) / base,
                        "ssim": mesures.ssim(truth, img),
                    })
            print(f"{rife_file} {name} : avant {rows[(rife_file, allure, 'avant')][-1]['vrai']:.2f} dB, "
                  f"après {rows[(rife_file, allure, 'après')][-1]['vrai']:.2f} dB", flush=True)
    lines = [f"Scènes de synthèse en {size[0]} × {size[1]} (source {size[0] // 2} × {size[1] // 2}), SPAN {SPAN}.",
             "",
             "| RIFE | Allure | Ordre | PSNR vrai (dB) | PSNR face à SPAN de la vraie (dB) | Netteté | SSIM |",
             "|---|---|---|---:|---:|---:|---:|"]
    allures = [m[0] for m in sources.MOTIONS] + ["toutes"]
    keys = sorted(rows, key=lambda k: (RIFE.index(k[0]), allures.index(k[1]), k[2] != "avant"))
    for rife_file, allure, order in keys:
        got = rows[(rife_file, allure, order)]
        mean = {k: float(np.mean([g[k] for g in got])) for k in got[0]}
        lines.append(f"| {rife_file} | {allure} | {order} | {mean['vrai']:.2f} | {mean['source']:.2f} | "
                     f"{mean['nettete']:.3f} | {mean['ssim']:.4f} |".replace(".", ",")
                     .replace(rife_file.replace(".", ","), rife_file))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as out:
            out.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modeles", required=True, help="dossier des modèles (onnx/, rife/)")
    parser.add_argument("--calibration", default=os.path.join(ROOT, "data", "calibration", "calibration.mkv"))
    parser.add_argument("--plans", type=int, default=6)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--largeur", type=int, default=1280, help="largeur de la sortie (la vérité)")
    parser.add_argument("--md", help="écrire le tableau dans ce fichier")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
