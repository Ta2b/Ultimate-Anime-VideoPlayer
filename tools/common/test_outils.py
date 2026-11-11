#!/usr/bin/env python3
"""Tests des outils de mesure et de modèles (audit D12), sans GPU ni modèle livré : python3 -I tools/common/test_outils.py

Couvre les correctifs B46 : lecteur PPM et PQ communs, seuil sombre en nits du comptage de paliers, verdict de la
porte de l'anneau, clé de cache d'abab_trt.sh, contrôles de quantize_span.py, bruit de banc_resume.py."""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "common"))
import hdr  # noqa: E402


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, TOOLS / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_ppm16(path, pixels):
    h, w, _ = pixels.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n65535\n".encode())
        f.write(pixels.astype(">u2").tobytes())


class Hdr(unittest.TestCase):
    def test_pq_aller_retour_et_valeurs_connues(self):
        nits = np.array([0.1, 1.0, 100.0, 1000.0])
        self.assertTrue(np.allclose(hdr.pq_to_nits(hdr.nits_to_pq(nits)), nits, rtol=1e-6))
        self.assertAlmostEqual(float(hdr.nits_to_pq(203.0)), 0.58069, places=4)
        self.assertAlmostEqual(float(hdr.nits_to_pq(0.325)), 0.10, places=3)  # l'ancien « --dark 0,10 »

    def test_lecteur_ppm_refuse_un_fichier_tronque(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.ppm")
            with open(path, "wb") as f:
                f.write(b"P6\n2 2\n65535\n")
            with self.assertRaises(ValueError):
                hdr.read_ppm16(path)
            write_ppm16(path, np.full((2, 2, 3), 65535))
            self.assertEqual(hdr.read_ppm16(path).max(), 1.0)
            with open(path, "wb") as f:
                f.write(b"P6\n1 1\n255\n\x10\x20\x30")
            with self.assertRaises(ValueError):
                hdr.read_ppm16(path)  # strict : 65535 attendu
            self.assertEqual(hdr.read_ppm16(path, strict16=False).shape, (1, 1, 3))


class Banding(unittest.TestCase):
    def gradient(self, nits, steps=40, plateau=20, height=4):
        """Des plateaux d'un code 10 bits de pas, autour du niveau `nits`."""
        step = 65535.0 / 1023.0
        base = float(hdr.nits_to_pq(nits)) * 65535.0
        row = np.concatenate([np.full(plateau, round(base + k * step)) for k in range(steps)])
        return np.repeat(row[None, :, None], 3, axis=2).repeat(height, axis=0).astype(np.float64)

    def test_paliers_a_5_nits_comptes_avec_le_seuil_en_nits(self):
        banding = load("banding", "ring/banding.py")
        image = self.gradient(5.0)
        # B46-04 : l'ancien seuil (code PQ 0,10 = 0,33 nit) ne voyait rien à 5 nits.
        self.assertEqual(banding.bands(image, 16, 0.10), 0.0)
        self.assertGreater(banding.bands(image), 0.0)
        self.assertGreater(banding.dark_code(20.0), 0.35)

    def test_pas_de_palier_au_dessus_du_seuil(self):
        banding = load("banding", "ring/banding.py")
        self.assertEqual(banding.bands(self.gradient(100.0)), 0.0)


@unittest.skipUnless(shutil.which("bash"), "bash absent")
class Scripts(unittest.TestCase):
    def test_porte_de_l_anneau_echoue_sur_un_ppm_tronque(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "stub.sh"
            stub.write_text(
                '#!/bin/bash\nfor a in "$@"; do case $a in --snapshot=*) o=${a#--snapshot=};; esac; done\n'
                "printf 'P6\\n2 2\\n65535\\n' > \"$o\"\n")
            stub.chmod(0o755)
            (Path(tmp) / "cal.mkv").write_bytes(b"")
            (Path(tmp) / "shots.tsv").write_text("plan01\tcal\t3\tcalibration\n")
            env = dict(os.environ, L=str(stub), CAL=str(Path(tmp) / "cal.mkv"), STILLS=tmp,
                       SHOTS=str(Path(tmp) / "shots.tsv"))
            run = subprocess.run(["bash", str(TOOLS / "grade/compare_shots.sh"), "", "--x", str(Path(tmp) / "out")],
                                 env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)  # B46-01 : 0 avant
            self.assertIn("mesure en échec", (Path(tmp) / "out/resume.tsv").read_text(encoding="utf-8"))

    def test_cache_d_abab_suit_le_contenu_de_l_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            trtexec = t / "trtexec"
            trtexec.write_text(
                '#!/usr/bin/env bash\nfor a in "$@"; do case $a in --saveEngine=*) echo x > "${a#--saveEngine=}";;\n'
                '--loadEngine=*) echo "[I] GPU Compute Time: min = 1.0 ms, max = 2 ms, mean = 1.1 ms, median = 1.2 ms";;\n'
                "esac; done\n")
            trtexec.chmod(0o755)
            (t / "a.onnx").write_bytes(b"a")
            (t / "b.onnx").write_bytes(b"b")
            env = dict(os.environ, TRTEXEC=str(trtexec), ABAB_CACHE=str(t / "cache"))
            cmd = ["bash", str(TOOLS / "fp8/abab_trt.sh"), str(t / "a.onnx"), str(t / "b.onnx"), "static", "1"]
            subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=120)
            first = sorted(p.name for p in (t / "cache").glob("B-*.engine"))
            (t / "b.onnx").write_bytes(b"b2")  # reconstruit sous le même nom (B46-07)
            subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=120)
            second = sorted(p.name for p in (t / "cache").glob("B-*.engine"))
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 2)


class QuantizeSpan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.q = load("quantize_span", "span/quantize_span.py")
        except ImportError as error:  # onnx absent
            raise unittest.SkipTest(str(error))

    def model(self, zero_point_type):
        from onnx import TensorProto, helper, numpy_helper
        zp = helper.make_tensor("zp", zero_point_type, [], [0])
        graph = helper.make_graph(
            [helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["y"])],
            "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.array(1.0, np.float32), "s"), zp])
        return helper.make_model(graph)

    def test_le_format_ecrit_est_controle(self):
        from onnx import TensorProto
        fp8, int8 = self.model(TensorProto.FLOAT8E4M3FN), self.model(TensorProto.INT8)
        self.q.check_qdq_format(fp8, "fp8")
        self.q.check_qdq_format(int8, "int8")
        with self.assertRaises(SystemExit):
            self.q.check_qdq_format(int8, "fp8")  # B46-08 : un « FP8 » qui n'en est pas un
        with self.assertRaises(SystemExit):
            self.q.check_qdq_format(fp8, "int8")

    def test_corpus_trop_petit_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(self.q.MIN_CORPUS - 1):
                (Path(tmp) / f"i{i:03}.png").write_bytes(bytes([i]))
            with self.assertRaises(SystemExit):
                self.q.corpus(Path(tmp))
            (Path(tmp) / "i999.png").write_bytes(b"x")
            calibration, validation, _ = self.q.corpus(Path(tmp))
            self.assertEqual(len(validation), self.q.MIN_VALIDATION)  # B46-10 : 8, pas 2
            self.assertEqual(len(calibration) + len(validation), self.q.MIN_CORPUS)


class WrapLumaRgb(unittest.TestCase):
    def test_les_deux_remontees_de_chroma_donnent_un_modele_valide(self):
        try:
            import onnx
            import onnxruntime as ort
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as error:
            self.skipTest(str(error))
        wrap = load("wrap_luma_rgb", "upscale/wrap_luma_rgb.py")
        weights = numpy_helper.from_array(np.full((4, 1, 3, 3), 0.1, np.float32), "w")
        graph = helper.make_graph(
            [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
             helper.make_node("DepthToSpace", ["c"], ["y"], blocksize=2)],
            "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "h", "w"])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, "h2", "w2"])], [weights])
        luma = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        luma.ir_version = 9
        image = np.random.default_rng(0).random((1, 3, 8, 8)).astype(np.float16)
        for mode in ("linear", "cubic"):
            out = wrap.wrap(luma, mode)
            sess = ort.InferenceSession(out.SerializeToString(), providers=["CPUExecutionProvider"])
            result = sess.run(None, {"input": image})[0]
            self.assertEqual(result.shape, (1, 3, 16, 16), mode)
            self.assertTrue(np.isfinite(result.astype(np.float32)).all())


class BancResume(unittest.TestCase):
    def test_bruit_des_deux_cotes_et_nombre_de_tours(self):
        resume = load("banc_resume", "bench/banc_resume.py")
        calme = [10.0, 10.1, 10.0, 9.9, 10.0]
        self.assertEqual(resume.verdict(calme, [11.0, 15.0, 7.0, 13.0, 9.0]), "bruit")
        self.assertTrue(resume.verdict([10.0, 10.1, 10.0], [12.0, 12.1, 12.0]).startswith("à confirmer"))
        self.assertEqual(resume.verdict(calme, [12.0, 12.1, 12.0, 11.9, 12.0]), "gain acquis")


if __name__ == "__main__":
    unittest.main()
