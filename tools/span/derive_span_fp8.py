#!/usr/bin/env python3
"""SPAN entier en FP8 pour les noyaux CUTLASS du lecteur (docs/span-fp8-cutlass.md) : le paquet de poids, ses
échelles d'activation par maximum et le reçu à garde-fous, sur le modèle de tools/span/quantize_span.py.

Le lecteur prend ce paquet au cran « Performance » de SPAN, sur une RTX 40 (sm 8.9), à la place du SPAN en
TensorRT, quand son reçu passe les garde-fous (src/app/play/span_int8.h) ; sinon, ou si quoi que ce soit ne va
pas (autre GPU, taille non prise, erreur CUDA), il reste sur TensorRT FP16. Seul le SPAN à 8 blocs de 64 canaux
(« b8f64 », par exemple 2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx) est pris ; le modèle
« Performance » b5f48 (48 canaux, 5 blocs) est refusé ici, avant tout calcul.

  1. Lecture et contrôle du graphe : noms et formes des poids, dégroupage 2x2.
  2. Calibration : le maximum de |valeur| des 25 tenseurs que les noyaux quantifient en E4M3 (sortie de conv_1 ;
     relu1, relu2 et sortie de chacun des 8 blocs) sur des découpes d'images d'anime, avec ONNX Runtime CPU en
     FP32. Échelle = maximum / 448 (le plus grand E4M3 fini).
  3. Écriture du paquet « .span8 » (src/gpu/span_fp8/package.h) : poids FP16 du modèle, biais f32, échelles.
  4. Qualité : pour des images de validation, la sortie du lecteur (ajp_span_fp8_run, les noyaux eux-mêmes, sous
     le verrou GPU) contre le SPAN d'origine en FP32 : PSNR moyen et pire, pire SSIM, aux seuils du reçu INT8
     (42 dB, 38 dB, 0,985).
  5. Vitesse (si --trtexec est donné) : le temps des noyaux contre TensorRT FP16 ; le paquet exige --min-speedup.

Le corpus est celui de quantize_span.py : des images SDR 8 bits d'anime, tirées de vos vidéos, sans sortie de
RTX HDR, TrueHDR ou RTX Video (licence NVIDIA) ; un cinquième des images (choisies par leur SHA-256) sert à la
validation, jamais à la calibration.

Le paquet est dérivé des poids d'AnimeJaNai (CC BY-NC-SA 4.0) : il se construit chez vous et ne se publie pas.

    ~/.animejanai-tools/venv-gpu/bin/python3 -I tools/span/derive_span_fp8.py MODELE.onnx --corpus images/ \\
        --out "$XDG_DATA_HOME/ajp/ajp/modeles" --runner build/linux-prefix-cuda/src/gpu/cuda/backend/ajp_span_fp8_run \\
        --gpu-prefix ~/ajp-tools/bin/ajp-gpu

Écrit dans --out « <nom>-fp8c-<16 premiers chiffres du SHA-256 du modèle>.span8 » et son reçu « .toml » ; si les
garde-fous échouent, seul le reçu est écrit (passed = false).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# quantize_span.py : corpus, découpes, conversion en FP32, métriques, reçu (une seule définition de chacun).
_spec = importlib.util.spec_from_file_location("quantize_span", Path(__file__).with_name("quantize_span.py"))
q = importlib.util.module_from_spec(_spec)
sys.modules["quantize_span"] = q
_spec.loader.exec_module(q)

MAGIC = b"AJPSPAN8"
VERSION = 1
BLOCKS, CHANNELS = 8, 64
FORMAT = "ajp-span-fp8c-v1"
E4M3_MAX = 448.0
# Seuils du reçu : ceux de l'INT8 (span_int8.cpp), pas ceux du FP8 par Q/DQ.
MIN_PSNR_MEAN_DB, MIN_PSNR_WORST_DB, MIN_SSIM_WORST = 42.0, 38.0, 0.985

# Les tenseurs que les noyaux quantifient : conv_1, puis (relu1, relu2, sortie) de chaque bloc.
CALIBRATED = ["/conv_1/conv_1.1/eval_conv/Conv_output_0"]
for _k in range(1, BLOCKS + 1):
    CALIBRATED += [f"/block_{_k}/act1/Relu_output_0", f"/block_{_k}/act1_1/Relu_output_0", f"/block_{_k}/Mul_output_0"]
UNSHUFFLED = "/conv_1/conv_1.0/Reshape_1_output_0"


def fail(message: str) -> None:
    sys.exit(f"derive_span_fp8 : {message}")


def weights_of(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Les poids du SPAN b8f64, contrôlés : formes, nombre de blocs, présence de chaque nœud calibré."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    first = init.get("conv_1.1.eval_conv.weight")
    if first is None:
        fail("pas un SPAN AnimeJaNai « unshuffle » (conv_1.1.eval_conv.weight absent)")
    if first.shape != (CHANNELS, 12, 3, 3):
        fail(f"conv_1 de forme {first.shape} : seul le SPAN à {CHANNELS} canaux est pris "
             f"(le modèle b5f48 à 48 canaux reste sur TensorRT FP16)")
    blocks = sorted({int(m.group(1)) for n in init for m in [re.match(r"block_(\d+)\.", n)] if m})
    if blocks != list(range(1, BLOCKS + 1)):
        fail(f"{len(blocks)} blocs trouvés, {BLOCKS} attendus")
    expected = {"conv_cat.weight": (CHANNELS, 4 * CHANNELS, 1, 1), "to_pixel.eval_conv.weight": (48, CHANNELS, 3, 3)}
    for k in range(1, BLOCKS + 1):
        for n in ("c1_r", "c2_r", "c3_r"):
            expected[f"block_{k}.{n}.eval_conv.weight"] = (CHANNELS, CHANNELS, 3, 3)
        expected[f"block_{k}.att_conv.weight"] = (CHANNELS, CHANNELS, 1, 1)
    for name, shape in expected.items():
        if name not in init or init[name].shape != shape:
            fail(f"{name} : {'absent' if name not in init else init[name].shape}, {shape} attendu")
    produced = {o for n in model.graph.node for o in n.output}
    for name in CALIBRATED + [UNSHUFFLED]:
        if name not in produced:
            fail(f"tenseur {name} absent du graphe : ce SPAN n'a pas la forme que les noyaux supposent")
    return init


def krsc(w: np.ndarray, pad_to: int | None = None) -> np.ndarray:
    """(sortie, entrée, ligne, colonne) -> (sortie, ligne, colonne, entrée), FP16, entrées complétées de zéros."""
    out = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
    if pad_to is not None and out.shape[3] < pad_to:
        out = np.pad(out, [(0, 0), (0, 0), (0, 0), (0, pad_to - out.shape[3])])
    return np.ascontiguousarray(out.astype("<f2"))


def f32(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a.astype("<f4")).tobytes()


def build_package(init: dict[str, np.ndarray], scales: np.ndarray) -> bytes:
    """Le fichier .span8 : voir src/gpu/span_fp8/package.h (l'ordre des sections est celui de parse)."""
    body = bytearray(f32(scales))

    def conv(weight: np.ndarray, bias: np.ndarray) -> None:
        body.extend(weight.tobytes())
        body.extend(f32(bias))

    conv(krsc(init["conv_1.1.eval_conv.weight"], 16), init["conv_1.1.eval_conv.bias"])
    for k in range(1, BLOCKS + 1):
        for n in ("c1_r", "c2_r", "c3_r"):
            conv(krsc(init[f"block_{k}.{n}.eval_conv.weight"]), init[f"block_{k}.{n}.eval_conv.bias"])
        conv(init[f"block_{k}.att_conv.weight"].reshape(CHANNELS, CHANNELS).astype("<f2"), init[f"block_{k}.att_conv.bias"])
    conv(init["conv_cat.weight"].reshape(CHANNELS, 4 * CHANNELS).astype("<f2"), init["conv_cat.bias"])
    conv(krsc(init["to_pixel.eval_conv.weight"]), init["to_pixel.eval_conv.bias"])
    header = MAGIC + struct.pack("<IIIIQII", VERSION, BLOCKS, CHANNELS, 0, len(body), zlib.crc32(bytes(body)), 0)
    assert len(header) == 40
    return header + bytes(body)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Découpe centrée d'au plus width x height, ramenée à une taille que les noyaux prennent :
    hauteur multiple de 4, largeur multiple de 128 ((h/2)*(w/2) multiple de 128, tout est pair)."""
    image = q.centre(image, width, height)
    _, h, w = image.shape
    h, w = h // 4 * 4, w // 128 * 128
    if h == 0 or w == 0:
        fail("image de validation trop petite pour les noyaux (128 colonnes au moins)")
    return image[:, :h, :w]


def run_candidate(runner: str, gpu_prefix: list[str], package: Path, image: np.ndarray, work: Path, bench: int = 0):
    _, h, w = image.shape
    src, dst = work / "in.f32", work / "out.f32"
    np.ascontiguousarray(image, dtype="<f4").tofile(src)
    command = gpu_prefix + [runner, str(package), str(h), str(w), str(src), str(dst)]
    if bench:
        command += ["--bench", str(bench)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        fail(f"{runner} a échoué :\n{done.stdout[-600:]}{done.stderr[-600:]}")
    out = np.fromfile(dst, "<f4").reshape(3, 2 * h, 2 * w)
    return np.clip(out.astype(np.float64), 0.0, 1.0), done.stdout


def trtexec_ms(gpu_prefix: list[str], trtexec: str, model: Path, height: int, width: int) -> float:
    """Médiane du temps GPU de TensorRT sur le SPAN d'origine (FP16), sous le verrou GPU."""
    shape = f"input:1x3x{height}x{width}"
    command = gpu_prefix + [trtexec, f"--onnx={model}", f"--minShapes={shape}", f"--optShapes={shape}",
                            f"--maxShapes={shape}", "--warmUp=500", "--iterations=200", "--useCudaGraph", "--noDataTransfers"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    match = re.search(r"GPU Compute Time:.*?median = ([0-9.]+) ms", done.stdout)
    if done.returncode != 0 or not match:
        fail(f"trtexec a échoué sur {model.name} :\n{done.stdout[-600:]}{done.stderr[-300:]}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="le SPAN FP16 b8f64 d'AnimeJaNai")
    parser.add_argument("--corpus", type=Path, required=True, help="dossier d'images SDR 8 bits d'anime")
    parser.add_argument("--out", type=Path, required=True, help="le dossier « modeles » des données du lecteur")
    parser.add_argument("--runner", required=True, help="ajp_span_fp8_run (cible CMake du backend CUDA)")
    parser.add_argument("--gpu-prefix", default="", help="préfixe des commandes GPU (verrou), par exemple ajp-gpu")
    parser.add_argument("--trtexec", default="", help="trtexec (TensorRT 11) pour exiger un gain sur le FP16")
    parser.add_argument("--speed-shape", default="1088x1920", help="hauteur x largeur de la mesure de vitesse")
    parser.add_argument("--min-speedup", type=float, default=1.05, help="gain exigé sur TensorRT FP16 (1,05)")
    parser.add_argument("--calib-size", type=int, default=576, help="côté des découpes de calibration (576)")
    parser.add_argument("--calib-crops", type=int, default=32, help="nombre de découpes de calibration (32)")
    parser.add_argument("--valid-size", default="1920x1088", help="taille maximale des images de validation")
    parser.add_argument("--max-valid", type=int, default=4, help="images de validation au plus (4)")
    parser.add_argument("--min-psnr", type=float, default=MIN_PSNR_MEAN_DB)
    parser.add_argument("--min-worst-psnr", type=float, default=MIN_PSNR_WORST_DB)
    parser.add_argument("--min-ssim", type=float, default=MIN_SSIM_WORST)
    args = parser.parse_args()

    source = args.model.resolve()
    if not source.is_file():
        fail(f"{source} introuvable")
    calibration, validation, manifest = q.corpus(args.corpus)
    validation = validation[: args.max_valid]
    source_sha = q.sha256_file(source)
    stem = f"{source.stem}-fp8c-{source_sha[:16]}"
    args.out.mkdir(parents=True, exist_ok=True)
    target, receipt = args.out / f"{stem}.span8", args.out / f"{stem}.toml"
    width, height = (int(v) for v in args.valid_size.lower().split("x"))
    gpu_prefix = args.gpu_prefix.split() if args.gpu_prefix else []

    model = onnx.load(str(source))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        fail("un SPAN a une entrée et une sortie")
    init = weights_of(model)
    input_name = model.graph.input[0].name
    print(f"{source.name} : {len(calibration)} images de calibration, {len(validation)} de validation")

    # --- calibration : maximum de |valeur| de chaque tenseur, sur des découpes, en FP32 (CPU)
    import onnxruntime as ort

    ref32 = q.to_float32(onnx.load(str(source)))
    calib_model = q.to_float32(onnx.load(str(source)))
    for name in CALIBRATED + [UNSHUFFLED]:
        calib_model.graph.output.append(onnx.helper.make_tensor_value_info(name, q.F32, None))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    sess_calib = ort.InferenceSession(calib_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    sess_ref = ort.InferenceSession(ref32.SerializeToString(), options, providers=["CPUExecutionProvider"])
    data = q.crops(calibration, args.calib_size, args.calib_crops, seed=int(source_sha[:8], 16))
    peaks = np.zeros(len(CALIBRATED), np.float64)
    for i, crop in enumerate(data):
        res = sess_calib.run(CALIBRATED + [UNSHUFFLED], {input_name: crop[None].astype(np.float32)})
        if i == 0:  # le dégroupage que les noyaux supposent (canal = c*4 + a*2 + b) est celui du modèle
            n = args.calib_size
            mine = crop.reshape(3, n // 2, 2, n // 2, 2).transpose(0, 2, 4, 1, 3).reshape(12, n // 2, n // 2)
            gap = float(np.abs(mine - res[-1][0]).max())
            if gap > 1e-6:
                fail(f"dégroupage : écart {gap} avec le modèle, ce SPAN n'a pas l'ordre des canaux supposé")
        for j, a in enumerate(res[:-1]):
            peaks[j] = max(peaks[j], float(np.abs(a).max()))
    if not np.all(peaks > 0):
        fail("un tenseur calibré est nul partout : corpus ou découpes inutilisables")
    scales = (peaks / E4M3_MAX).astype(np.float32)
    print(f"calibration : {args.calib_crops} découpes de {args.calib_size}², maxima {peaks.min():.2f} à {peaks.max():.2f}")

    package = build_package(init, scales)
    with tempfile.TemporaryDirectory(prefix="ajp-span8-") as work_dir:
        work = Path(work_dir)
        candidate_path = work / "candidate.span8"
        candidate_path.write_bytes(package)

        # --- qualité : les noyaux du lecteur contre le SPAN d'origine
        scores = []
        for path in validation:
            image = fit(q.load_rgb(path), width, height)
            ref = np.clip(sess_ref.run(None, {input_name: image[None].astype(np.float32)})[0][0].astype(np.float64), 0, 1)
            got, _ = run_candidate(args.runner, gpu_prefix, candidate_path, image, work)
            scores.append((q.psnr(ref, got), q.ssim(ref, got)))
            print(f"  {path.name} : PSNR {scores[-1][0]:.2f} dB, SSIM {scores[-1][1]:.4f}")
        psnr_mean = float(np.mean([s[0] for s in scores]))
        psnr_worst = min(s[0] for s in scores)
        ssim_worst = min(s[1] for s in scores)
        passed = psnr_mean >= args.min_psnr and psnr_worst >= args.min_worst_psnr and ssim_worst >= args.min_ssim

        # --- vitesse : les noyaux contre TensorRT FP16, à la taille demandée
        speedup, fp8_ms, fp16_ms = 0.0, 0.0, 0.0
        if passed and args.trtexec:
            sh, sw = (int(v) for v in args.speed_shape.lower().split("x"))
            _, text = run_candidate(args.runner, gpu_prefix, candidate_path, np.zeros((3, sh, sw), np.float32), work, 200)
            m = re.search(r"médiane ([0-9.]+) ms", text)
            if not m:
                fail(f"temps des noyaux illisible : {text}")
            fp8_ms = float(m.group(1))
            fp16_ms = trtexec_ms(gpu_prefix, args.trtexec, source, sh, sw)
            speedup = fp16_ms / fp8_ms
            print(f"  {sh}x{sw} : TensorRT FP16 {fp16_ms:.2f} ms, noyaux FP8 {fp8_ms:.2f} ms (x{speedup:.2f})")
            if speedup < args.min_speedup:
                print(f"  pas assez rapide (x{speedup:.2f}, il faut x{args.min_speedup:.2f}) : non écrit")
                passed = False
        if passed:
            target.write_bytes(package)
        elif target.exists():
            target.unlink()  # un paquet d'avant ne correspond plus à ce reçu

    q.write_receipt(
        receipt,
        {
            "format": FORMAT,
            "source": source.name,
            "source_sha256": source_sha,
            "model_sha256": q.sha256_file(target) if passed else "",
            "quantizer": "derive_span_fp8.py (CUTLASS E4M3, maximum)",
            "precision": "fp8",
            "speedup": speedup,
            "kernels_ms": fp8_ms,
            "trt_fp16_ms": fp16_ms,
            "method": "max",
            "calibration_images": len(calibration),
            "calibration_crops": args.calib_crops,
            "calibration_crop": args.calib_size,
            "corpus_sha256": manifest,
            "validation_images": len(validation),
            "validation_size": f"{width}x{height}",
            "psnr_mean_db": psnr_mean,
            "psnr_worst_db": psnr_worst,
            "ssim_worst": ssim_worst,
            "passed": passed,
        },
    )
    verdict = "passe" if passed else "ÉCHEC, paquet non écrit"
    print(f"PSNR moyen {psnr_mean:.2f} dB, pire {psnr_worst:.2f} dB, pire SSIM {ssim_worst:.4f} : {verdict}")
    print(f"reçu {receipt}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
