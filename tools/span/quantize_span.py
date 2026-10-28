#!/usr/bin/env python3
"""SPAN en INT8, FP8 ou FP32 pour TensorRT (étude 4, 6.3) : nœuds Q/DQ posés par NVIDIA Model Optimizer,
calibration « entropy » (INT8) ou « max » (FP8) sur des images d'anime, garde-fous PSNR et SSIM contre le
modèle d'origine.

--precision choisit la variante : int8 (par défaut), fp8 (E4M3, le cran Performance du lecteur, RTX 40 et
après) ou fp32 (le cran Qualité : calcul en FP32, E/S en FP16, sans quantification). Un SPAN en FP8 n'est écrit
que s'il va plus vite que le FP16 : une mesure faite à la main (RTX 4070 Ti, TensorRT 11.3, SPAN Balanced HD
V3.1, 1088×1920) donne 19,90 ms en FP8 contre 17,34 ms en FP16 quand les Q/DQ ne sont pas fusionnés. Le
script chronomètre donc les deux avec trtexec (--trtexec) et exige --min-speedup ; --skip-speed-check écrit
la variante sans mesure.

TensorRT 11 est fortement typé : l'INT8 vient des nœuds QuantizeLinear/DequantizeLinear du graphe,
le reste en FP16, entrée et sortie comprises (le lecteur refuse d'autres E/S). Le lecteur prend la
variante d'un modèle sous TensorRT quand elle est dans son dossier « modeles » avec un reçu qui
passe ses garde-fous (src/app/play/span_int8.h) ; supprimer les deux fichiers ramène le FP16.

Le corpus : des images SDR 8 bits d'anime, tirées de vos vidéos, par exemple
    ffmpeg -i episode.mkv -vf "select=not(mod(n\\,480))" -fps_mode vfr corpus/ep01_%04d.png
Il ne doit contenir aucune sortie de RTX HDR, TrueHDR ou RTX Video (licence NVIDIA : pas
d'entraînement ni de réglage sur leurs sorties). Le script refuse les fichiers dont le chemin le
laisse croire et toute image de plus de 8 bits par canal ; le reste est de votre ressort. Un
cinquième des images (choisies par leur SHA-256) sert à la validation, jamais à la calibration.

Le modèle INT8 est dérivé des poids d'AnimeJaNai : il se construit chez vous et ne se publie pas.

    pip install "nvidia-modelopt[onnx]" pillow
    python -I tools/span/quantize_span.py 2x_AnimeJaNai_HD_V3_….onnx --corpus images/ \\
        --out "%APPDATA%/ajp/ajp/modeles"

Écrit dans --out « <nom>-int8-<16 premiers chiffres du SHA-256 du modèle>.onnx » et son reçu
« .toml » ; si les garde-fous échouent, seul le reçu est écrit (passed = false).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Ce que le lecteur exige d'un reçu (span_int8.cpp, kMin*) : les mêmes valeurs par défaut.
MIN_PSNR_MEAN_DB = 42.0
MIN_PSNR_WORST_DB = 38.0
MIN_SSIM_WORST = 0.985

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
HDR_SUFFIXES = {".exr", ".hdr", ".jxr", ".avif", ".heic", ".heif", ".tif", ".tiff"}
NVIDIA_OUTPUTS = re.compile(r"rtx[ _.-]?(hdr|vsr|video)|true[ _.-]?hdr|nvidia[ _.-]?(hdr|vsr)", re.IGNORECASE)
F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16
# Par précision : le format du reçu, l'infixe du fichier et les seuils que le lecteur exige (span_int8.cpp).
PRECISIONS = {
    "int8": {"format": "ajp-span-int8-v1", "infix": "int8", "psnr": 42.0, "worst": 38.0, "ssim": 0.985, "mode": "int8"},
    "fp8": {"format": "ajp-span-fp8-v1", "infix": "fp8", "psnr": 30.0, "worst": 26.0, "ssim": 0.93, "mode": "fp8"},
    "fp32": {"format": "ajp-span-fp32-v1", "infix": "fp32", "psnr": 42.0, "worst": 38.0, "ssim": 0.985, "mode": None},
}


def fail(message: str) -> None:
    sys.exit(f"quantize_span : {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def png_bit_depth(path: Path) -> int:
    head = path.read_bytes()[:25]
    return head[24] if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) == 25 else 8


def corpus(folder: Path) -> tuple[list[Path], list[Path], str]:
    """Images de calibration et de validation, et le condensé du corpus pour le reçu."""
    images = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        if NVIDIA_OUTPUTS.search(relative):
            fail(f"{relative} : sortie de RTX HDR, TrueHDR ou RTX Video ? Retirez-la du corpus (licence NVIDIA)")
        suffix = path.suffix.lower()
        if suffix in HDR_SUFFIXES:
            fail(f"{relative} : format HDR ou à plus de 8 bits refusé, convertissez en PNG 8 bits")
        if suffix not in IMAGE_SUFFIXES:
            continue
        if suffix == ".png" and png_bit_depth(path) > 8:
            fail(f"{relative} : PNG de plus de 8 bits par canal refusé (SPAN lit des images SDR)")
        images.append((sha256_file(path), path))
    images.sort()
    if len(images) < 10:
        fail(f"{folder} : {len(images)} images, 10 au moins (8 de calibration, 2 de validation)")
    calibration = [p for i, (_, p) in enumerate(images) if i % 5 != 4]
    validation = [p for i, (_, p) in enumerate(images) if i % 5 == 4]
    manifest = hashlib.sha256("\n".join(sha for sha, _ in images).encode()).hexdigest()
    return calibration, validation, manifest


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb.transpose(2, 0, 1)  # CHW, comme l'ingestion du lecteur


def crops(images: list[Path], size: int, count: int, seed: int) -> np.ndarray:
    """count découpes size × size tirées au hasard (graine fixe), réparties sur les images."""
    rng = np.random.default_rng(seed)
    out = np.empty((count, 3, size, size), dtype=np.float32)
    for i in range(count):
        image = load_rgb(images[i % len(images)])
        _, h, w = image.shape
        if h < size or w < size:
            fail(f"{images[i % len(images)]} : {w}x{h}, plus petit que la découpe {size}")
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        out[i] = image[:, y : y + size, x : x + size]
    return out


def centre(image: np.ndarray, width: int, height: int) -> np.ndarray:
    _, h, w = image.shape
    width, height = min(width, w), min(height, h)
    y, x = (h - height) // 2, (w - width) // 2
    return image[:, y : y + height, x : x + width]


def to_float32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Le modèle FP16 en FP32 : Model Optimizer calibre en FP32 et rend du FP16 (high_precision_dtype)."""
    graph = model.graph
    for i, tensor in enumerate(graph.initializer):
        if tensor.data_type == F16:
            graph.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.type.tensor_type.elem_type == F16:
            value.type.tensor_type.elem_type = F32
    for node in graph.node:
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == F16:
                attribute.i = F32
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == F16:
                attribute.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float32), attribute.t.name))
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                fail(f"nœud {node.name} ({node.op_type}) à sous-graphe : pas un SPAN")
    return model


def fp16_io(model: onnx.ModelProto) -> onnx.ModelProto:
    """Entrée et sortie en FP16. Model Optimizer garde les E/S FP32 du modèle de calibration et les
    encadre de Cast vers et depuis le FP16 : ces Cast disparaissent ; ailleurs, un Cast est posé."""
    graph = model.graph
    for value in graph.input:
        if value.type.tensor_type.elem_type != F32:
            continue
        users = [n for n in graph.node if value.name in n.input]
        casts = [n for n in users if n.op_type == "Cast" and n.attribute[0].i == F16 and len(users) == 1]
        if casts:
            for node in graph.node:
                node.input[:] = [value.name if name == casts[0].output[0] else name for name in node.input]
            graph.node.remove(casts[0])
        else:
            inner = value.name + "_fp32"
            for node in users:
                node.input[:] = [inner if name == value.name else name for name in node.input]
            graph.node.insert(0, helper.make_node("Cast", [value.name], [inner], name=inner + "_cast", to=F32))
        value.type.tensor_type.elem_type = F16
    for value in graph.output:
        if value.type.tensor_type.elem_type != F32:
            continue
        producer = next(n for n in graph.node if value.name in n.output)
        users = [n for n in graph.node if value.name in n.input]
        if producer.op_type == "Cast" and producer.attribute[0].i == F32 and not users:
            half = producer.input[0]
            graph.node.remove(producer)
            for node in graph.node:
                node.output[:] = [value.name if name == half else name for name in node.output]
                node.input[:] = [value.name if name == half else name for name in node.input]
        else:
            inner = value.name + "_fp32"
            for node in graph.node:
                node.output[:] = [inner if name == value.name else name for name in node.output]
                node.input[:] = [inner if name == value.name else name for name in node.input]
            graph.node.append(helper.make_node("Cast", [inner], [value.name], name=value.name + "_cast", to=F16))
        value.type.tensor_type.elem_type = F16
    return model


def quantize_modelopt(source32: Path, output: Path, data: np.ndarray, input_name: str, method: str, eps: list[str], mode: str = "int8") -> str:
    try:
        import modelopt
        from modelopt.onnx.quantization import quantize
    except ImportError as error:
        fail(f"Model Optimizer introuvable ({error}) : pip install \"nvidia-modelopt[onnx]\"")
    size = data.shape[2]
    wanted = {
        "onnx_path": str(source32),
        "quantize_mode": mode,
        "calibration_data": {input_name: data},
        "calibration_method": method,
        "calibration_shapes": f"{input_name}:1x3x{size}x{size}",
        "calibration_eps": eps,
        "output_path": str(output),
        "high_precision_dtype": "fp16",
    }
    accepted = inspect.signature(quantize).parameters
    quantize(**{k: v for k, v in wanted.items() if k in accepted})
    return f"modelopt {getattr(modelopt, '__version__', '?')}"


def trtexec_ms(trtexec: str, model: Path, height: int, width: int) -> float:
    """Médiane du temps de calcul GPU d'une image (ms), mesurée par trtexec (TensorRT 11 : réseau fortement
    typé, ni --fp16 ni --fp8). Le modèle a une entrée « input » en 1×3×hauteur×largeur."""
    import subprocess

    shape = f"input:1x3x{height}x{width}"
    command = [trtexec, f"--onnx={model}", f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}",
               "--warmUp=500", "--iterations=200", "--useCudaGraph", "--noDataTransfers"]
    out = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    match = re.search(r"GPU Compute Time:.*?median = ([0-9.]+) ms", out.stdout)
    if out.returncode != 0 or not match:
        fail(f"trtexec a échoué sur {model.name} :\n{out.stdout[-800:]}{out.stderr[-400:]}")
    return float(match.group(1))


def session(path: Path, providers: list[str]):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=chosen)


def run(sess, image: np.ndarray) -> np.ndarray:
    feed = sess.get_inputs()[0]
    dtype = np.float16 if feed.type == "tensor(float16)" else np.float32
    out = sess.run(None, {feed.name: image[None].astype(dtype)})[0][0]
    return np.clip(out.astype(np.float64), 0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-10 else min(99.0, 10.0 * math.log10(1.0 / mse))


def gaussian(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    w = np.exp(-(x**2) / (2 * sigma**2))
    return w / w.sum()


def blur(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    k = len(w)
    rows = sum(w[i] * x[i : x.shape[0] - k + 1 + i, :] for i in range(k))
    return sum(w[i] * rows[:, i : rows.shape[1] - k + 1 + i] for i in range(k))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM de la luminance BT.709 (fenêtre gaussienne 11, σ 1,5 ; Wang et al. 2004)."""
    luma = np.array([0.2126, 0.7152, 0.0722])[:, None, None]
    x, y = (a * luma).sum(0), (b * luma).sum(0)
    w = gaussian()
    c1, c2 = 0.01**2, 0.03**2
    mx, my = blur(x, w), blur(y, w)
    sxx = blur(x * x, w) - mx * mx
    syy = blur(y * y, w) - my * my
    sxy = blur(x * y, w) - mx * my
    ratio = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(ratio.mean())


def toml_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_receipt(path: Path, fields: dict) -> None:
    lines = [
        "# tools/span/quantize_span.py : SPAN dérivé des poids d'AnimeJaNai, à ne pas publier.",
        *(f"{key} = {toml_text(value)}" for key, value in fields.items()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="le SPAN FP16 (ou FP32) d'AnimeJaNai")
    parser.add_argument("--corpus", type=Path, required=True, help="dossier d'images SDR 8 bits d'anime")
    parser.add_argument("--out", type=Path, required=True, help="le dossier « modeles » des données du lecteur")
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="int8", help="int8 (défaut), fp8 ou fp32")
    parser.add_argument("--method", choices=("entropy", "max"), default=None, help="entropy pour l'INT8, max pour le FP8")
    parser.add_argument("--trtexec", default="trtexec", help="trtexec (TensorRT 11) pour la mesure de vitesse du FP8")
    parser.add_argument("--speed-shape", default="1088x1920", help="hauteur x largeur de la mesure de vitesse")
    parser.add_argument("--min-speedup", type=float, default=1.05, help="gain exigé du FP8 sur le FP16 (1,05)")
    parser.add_argument("--skip-speed-check", action="store_true", help="écrire le FP8 sans le chronométrer")
    parser.add_argument("--crop", type=int, default=256, help="côté des découpes de calibration (256)")
    parser.add_argument("--crops", type=int, default=256, help="nombre de découpes de calibration (256)")
    parser.add_argument("--valid-size", default="960x540", help="taille des images de validation (960x540)")
    parser.add_argument("--eps", default="cuda:0,cpu", help="fournisseurs ONNX Runtime de la calibration")
    parser.add_argument("--min-psnr", type=float, default=None)
    parser.add_argument("--min-worst-psnr", type=float, default=None)
    parser.add_argument("--min-ssim", type=float, default=None)
    args = parser.parse_args()
    precision = PRECISIONS[args.precision]
    # Les seuils par défaut sont ceux que le lecteur exige de ce cran (span_int8.cpp).
    args.min_psnr = precision["psnr"] if args.min_psnr is None else args.min_psnr
    args.min_worst_psnr = precision["worst"] if args.min_worst_psnr is None else args.min_worst_psnr
    args.min_ssim = precision["ssim"] if args.min_ssim is None else args.min_ssim
    args.method = args.method or ("max" if args.precision == "fp8" else "entropy")

    source = args.model.resolve()
    if not source.is_file():
        fail(f"{source} introuvable")
    calibration, validation, manifest = corpus(args.corpus)
    source_sha = sha256_file(source)
    stem = f"{source.stem}-{precision['infix']}-{source_sha[:16]}"
    args.out.mkdir(parents=True, exist_ok=True)
    target, receipt = args.out / f"{stem}.onnx", args.out / f"{stem}.toml"
    width, height = (int(v) for v in args.valid_size.lower().split("x"))

    model = onnx.load(str(source))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        fail("un SPAN a une entrée et une sortie")
    input_name = model.graph.input[0].name
    print(f"{source.name} : {len(calibration)} images de calibration, {len(validation)} de validation")
    data = crops(calibration, args.crop, args.crops, seed=int(source_sha[:8], 16))
    with tempfile.TemporaryDirectory(prefix="ajp-int8-") as work:
        source32 = Path(work) / "source32.onnx"
        onnx.save(to_float32(model), str(source32))
        quantized = Path(work) / "int8.onnx"
        providers = [e.strip() for e in args.eps.split(",") if e.strip()]
        if precision["mode"] is None:  # fp32 : le calcul en FP32, E/S en FP16, aucun Q/DQ
            tool = "fp32 (to_float32 + fp16_io)"
            result = fp16_io(onnx.load(str(source32)))
            nodes = 0
        else:
            tool = quantize_modelopt(source32, quantized, data, input_name, args.method, providers, precision["mode"])
            result = fp16_io(onnx.load(str(quantized)))
            nodes = sum(1 for n in result.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
            if nodes == 0:
                fail("Model Optimizer n'a posé aucun nœud Q/DQ")
        onnx.checker.check_model(result)
        onnx.save(result, str(quantized))

        ort = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        reference, candidate = session(source, ort), session(quantized, ort)
        scores = []
        for path in validation:
            image = centre(load_rgb(path), width, height)
            a, b = run(reference, image), run(candidate, image)
            scores.append((psnr(a, b), ssim(a, b)))
            print(f"  {path.name} : PSNR {scores[-1][0]:.2f} dB, SSIM {scores[-1][1]:.4f}")
        psnr_mean = float(np.mean([s[0] for s in scores]))
        psnr_worst = min(s[0] for s in scores)
        ssim_worst = min(s[1] for s in scores)
        passed = psnr_mean >= args.min_psnr and psnr_worst >= args.min_worst_psnr and ssim_worst >= args.min_ssim
        speedup = 0.0
        if passed and args.precision == "fp8" and not args.skip_speed_check:
            speed_h, speed_w = (int(v) for v in args.speed_shape.lower().split("x"))
            half_ms = trtexec_ms(args.trtexec, source, speed_h, speed_w)
            fp8_ms = trtexec_ms(args.trtexec, quantized, speed_h, speed_w)
            speedup = half_ms / fp8_ms
            print(f"  trtexec {speed_h}x{speed_w} : FP16 {half_ms:.2f} ms, FP8 {fp8_ms:.2f} ms (×{speedup:.2f})")
            if speedup < args.min_speedup:
                print(f"  FP8 pas assez rapide (×{speedup:.2f}, il faut ×{args.min_speedup:.2f}) : non écrit")
                passed = False
        if passed:
            target.write_bytes(quantized.read_bytes())
        elif target.exists():
            target.unlink()  # une variante d'avant ne correspond plus à ce reçu

    write_receipt(
        receipt,
        {
            "format": precision["format"],
            "source": source.name,
            "source_sha256": source_sha,
            "model_sha256": sha256_file(target) if passed else "",
            "quantizer": tool,
            "precision": args.precision,
            "speedup": speedup,
            "method": args.method,
            "qdq_nodes": nodes,
            "calibration_images": len(calibration),
            "calibration_crops": args.crops,
            "calibration_crop": args.crop,
            "corpus_sha256": manifest,
            "validation_images": len(validation),
            "validation_size": f"{width}x{height}",
            "psnr_mean_db": psnr_mean,
            "psnr_worst_db": psnr_worst,
            "ssim_worst": ssim_worst,
            "passed": passed,
        },
    )
    verdict = "passe" if passed else "ÉCHEC, modèle non écrit"
    print(f"PSNR moyen {psnr_mean:.2f} dB, pire {psnr_worst:.2f} dB, pire SSIM {ssim_worst:.4f} : {verdict}")
    print(f"reçu {receipt}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
