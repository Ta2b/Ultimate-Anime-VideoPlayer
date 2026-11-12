#!/usr/bin/env python3
"""Tests des outils de mesure et de modèles (audit D12), sans GPU ni modèle livré : python3 -I tools/common/test_outils.py

Couvre les correctifs B46 : lecteur PPM et PQ communs, seuil sombre en nits du comptage de paliers, verdict de la
porte de l'anneau, clé de cache d'abab_trt.sh, contrôles de quantize_span.py, bruit de banc_resume.py ; et le
résumé du profil GPU d'un banc A/B (banc_gpu_resume.py)."""
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


class BancGpuResume(unittest.TestCase):
    def test_etapes_rangees_par_famille_et_ecart_des_medianes(self):
        resume = load("banc_gpu_resume", "bench/banc_gpu_resume.py")
        header = "etape\tmesures\ttotal_ms\tmediane_ms\tp95_ms\n"
        with tempfile.TemporaryDirectory() as d:
            for name, rife, ms in [("extrait1-tour1-A-1.gpu.tsv", "rife_v4.22_lite", 1.0),
                                   ("extrait1-tour1-B-2.gpu.tsv", "rife_v4.25_lite", 1.5),
                                   ("extrait1-tour2-A-3.gpu.tsv", "rife_v4.22_lite", 1.2),
                                   ("extrait1-tour2-B-4.gpu.tsv", "rife_v4.25_lite", 1.3),
                                   ("prechauffage-1-A.gpu.tsv", "rife_v4.22_lite", 9.0)]:
                Path(d, name).write_text(header + f"RIFE · {rife}\t10\t10\t{ms}\t2\n", encoding="utf-8")
            text = resume.report(resume.collect(d))
        self.assertIn("RIFE (RIFE · rife_v4.22_lite / RIFE · rife_v4.25_lite)", text)
        self.assertIn("médiane 1.10 ms", text)  # A : 1,0 et 1,2 ; le préchauffage ne compte pas
        self.assertIn("médiane 1.40 ms", text)
        self.assertIn("+27.3 %", text)


class AuditE16Grade(unittest.TestCase):
    def test_pire_mesure_signee_est_le_plus_grand_ecart_absolu(self):  # C46-07
        check = load("check_hdr", "grade/check_hdr.py")
        self.assertEqual(check.worst("peau_diaph", [-0.8, 0.1]), -0.8)
        self.assertEqual(check.worst("peau_diaph", [0.2, -0.1]), 0.2)
        self.assertEqual(check.worst("pic", [3.0, 5.0, 4.0]), 5.0)

    def test_reference_d_une_suite_retombe_sur_l_image_du_plan(self):  # C46-08
        check = load("check_hdr", "grade/check_hdr.py")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "plan.pq.ppm").write_bytes(b"x")
            self.assertEqual(check.reference_path(tmp, "plan", "c/plan-002.pq.ppm"), os.path.join(tmp, "plan.pq.ppm"))
            Path(tmp, "plan-002.pq.ppm").write_bytes(b"x")
            self.assertEqual(check.reference_path(tmp, "plan", "c/plan-002.pq.ppm"),
                             os.path.join(tmp, "plan-002.pq.ppm"))

    def test_banding_choisit_la_zone_sombre_en_nits_et_non_en_codes(self):  # C46-10
        banding = load("banding", "ring/banding.py")
        step = 65535.0 / 1023.0

        def vert(nits):  # des plateaux d'un code 10 bits sur le seul canal vert
            base = float(hdr.nits_to_pq(nits)) * 65535.0
            row = np.zeros((40 * 20, 3))
            for k in range(40):
                row[k * 20:(k + 1) * 20, 1] = round(base + k * step)
            return np.repeat(row[None], 4, axis=0)

        # Vert pur à 40 nits : 27 nits de luminance BT.2020, hors de la zone sombre (20 nits). Pondérer les codes
        # PQ donnait 0,68 × 0,43 = 0,29 < 0,36 : la zone y paraissait sombre.
        self.assertEqual(banding.bands(vert(40.0)), 0.0)
        self.assertGreater(banding.bands(vert(20.0)), 0.0)  # 13,6 nits de luminance : bien dans la zone


class AuditE16Scripts(unittest.TestCase):
    def run_script(self, args, env, timeout=120):
        return subprocess.run(["bash", *args], env=env, capture_output=True, text=True, timeout=timeout)

    def test_capture_reste_dans_le_plan_nomme(self):  # C46-01
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            stub = t / "stub.sh"
            stub.write_text('#!/bin/bash\necho "$@" >> "%s"\nfor a in "$@"; do case $a in --snapshot=*) o=${a#--snapshot=};; esac; done\n'
                            "printf 'P6\\n2 2\\n65535\\n' > \"$o\"\n" % (t / "appels.txt"))
            stub.chmod(0o755)
            (t / "cal.mkv").write_bytes(b"")
            (t / "shots.tsv").write_text("cal01\tcal\t3\tcalibration\ncal02\tcal\t9\tcalibration\n")
            env = dict(os.environ, L=str(stub), CAL=str(t / "cal.mkv"), STILLS=tmp, SHOTS=str(t / "shots.tsv"))
            self.run_script([str(TOOLS / "grade/compare_shots.sh"), "", "--x", str(t / "out")], env)
            calls = (t / "appels.txt").read_text()
            # cal01 dure 6 s (3 → 9) : la capture à 4 s après son début est dans le plan, pas à 6 s (le plan suivant).
            self.assertIn("--start=3 ", calls)
            self.assertIn("--snapshot-at=4 ", calls)
            self.assertNotIn("--snapshot-at=6 ", calls)
            (t / "appels.txt").unlink()
            env["SNAP_AT"] = "10"  # demandé au-delà du plan : ramené à sa durée moins 1 s
            self.run_script([str(TOOLS / "grade/compare_shots.sh"), "", "--x", str(t / "out2")], env)
            self.assertIn("--snapshot-at=5 ", (t / "appels.txt").read_text())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg absent")
    def test_extrait_coupe_au_plan_et_non_sur_l_image_cle(self):  # C46-02
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            source = t / "e.mkv"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=d=30:r=24",
                            "-c:v", "libx264", "-g", "240", str(source)], check=True)
            (t / "ep.tsv").write_text(f"a\t{source}\t10\n")
            run = self.run_script([str(TOOLS / "grade/make_stills.sh"), str(t / "ep.tsv"), str(t / "o")], os.environ)
            self.assertEqual(run.returncode, 0, run.stderr)
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                    "stream=codec_name:format=duration", "-of", "csv=p=0", str(t / "o/a.mkv")],
                                   capture_output=True, text=True, check=True).stdout
            self.assertIn("ffv1", probe)
            self.assertIn("8.0", probe)  # 8 s exactement : pas coupé sur l'image clé d'avant
            (t / "ep.tsv").write_text(f"a\t{source}\t1\n")
            self.assertNotEqual(self.run_script([str(TOOLS / "grade/make_stills.sh"), str(t / "ep.tsv"),
                                                 str(t / "o")], os.environ).returncode, 0)

    def test_banc_alterne_l_ordre_des_variantes(self):  # C46-05
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            ajp = t / "ajp"
            ajp.write_text('#!/bin/bash\nfor a in "$@"; do case $a in --data-dir=*) echo "${a##*data-}" >> "%s";; esac; done\n' % (t / "ordre.txt"))
            ajp.chmod(0o755)
            (t / "e.tsv").write_text("video.mkv\t0\n")
            self.run_script([str(TOOLS / "bench/banc.sh"), "--a=-a", "--b=-b", f"--extraits={t / 'e.tsv'}",
                             f"--sortie={t / 'out'}", "--tours=2", "--secondes=1", f"--ajp={ajp}"], os.environ)
            ordre = (t / "ordre.txt").read_text().split()[2:]  # sans les deux passages de préchauffage
            self.assertEqual("".join(ordre), "ABABBABA")

    def abab_env(self, t, output):
        trtexec = t / "trtexec"
        trtexec.write_text('#!/usr/bin/env bash\nfor a in "$@"; do case $a in --saveEngine=*) echo x > "${a#--saveEngine=}";;\n'
                           f'--loadEngine=*) echo "${{a#--loadEngine=}}" >> "{t / "appels.txt"}"; {output};;\nesac; done\n')
        trtexec.chmod(0o755)
        (t / "a.onnx").write_bytes(b"a")
        (t / "b.onnx").write_bytes(b"b")
        return dict(os.environ, TRTEXEC=str(trtexec), ABAB_CACHE=str(t / "cache"))

    def test_abab_alterne_l_ordre_et_refuse_une_mesure_vide(self):  # C46-05, C46-24
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            env = self.abab_env(t, 'echo "[I] GPU Compute Time: min = 1.0 ms, max = 2 ms, mean = 1.1 ms, median = 1.2 ms"')
            cmd = [str(TOOLS / "fp8/abab_trt.sh"), str(t / "a.onnx"), str(t / "b.onnx"), "static", "2"]
            self.assertEqual(self.run_script(cmd, env).returncode, 0)
            ordre = [Path(line).name[0] for line in (t / "appels.txt").read_text().split()]
            self.assertEqual("".join(ordre), "ABBA")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            env = self.abab_env(t, "true")  # trtexec réussit mais ne rend rien de lisible
            run = self.run_script([str(TOOLS / "fp8/abab_trt.sh"), str(t / "a.onnx"), str(t / "b.onnx"), "static", "1"], env)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("illisible", run.stderr)

    def test_presets_vulkan_refuse_de_vider_la_table_sans_source(self):  # C46-04
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "net_presets.cpp")
            out.write_text("// table en place\n")
            cmd = [sys.executable, "-I", str(TOOLS / "vk/presets_from_cache.py"), "--out", str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if (TOOLS / "vk/presets").is_dir() and list((TOOLS / "vk/presets").glob("*.tsv")):
                self.skipTest("des caches sont rangés dans tools/vk/presets")
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(out.read_text(), "// table en place\n")
            self.assertEqual(subprocess.run(cmd + ["--allow-empty"], capture_output=True, text=True, timeout=60).returncode, 0)


class AuditE16Fp8(unittest.TestCase):
    def test_derive_span_fp8_exige_trtexec(self):  # C46-17
        try:
            import onnx  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        run = subprocess.run([sys.executable, "-I", str(TOOLS / "span/derive_span_fp8.py"), "m.onnx", "--corpus", "c",
                              "--out", "o", "--runner", "r"], capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 2)
        self.assertIn("--trtexec", run.stderr)

    def test_plans_drba_jamais_a_l_instant_de_l_image_1(self):  # C46-18
        try:
            interp = load("interp_fp8", "fp8/interp_fp8.py")
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        us = [u for u, _ in interp.DRBA_U]
        self.assertNotIn(0.5, us)  # u = tau / 2 = 0,5 est tau = 1, l'image 1 elle-même
        self.assertTrue(all(0.0 < u < 1.0 for u in us))
        self.assertEqual(len({suffix for _, suffix in interp.DRBA_U}), len(us))
        self.assertEqual(interp.DEPLOYED_SIZE, "1088x1920")  # C46-19

    def test_echelles_fp16_nulles_refusees_et_poids_par_canal(self):  # C46-22, C46-23
        try:
            interp = load("interp_fp8", "fp8/interp_fp8.py")
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        with self.assertRaises(SystemExit):
            interp.activation_scale("t", 1e-6, np.float16)  # 1e-6 / 448 = 2e-9 : zéro en FP16
        with self.assertRaises(SystemExit):
            interp.activation_scale("t", 0.0, np.float16)
        self.assertGreater(float(interp.activation_scale("t", 3.0, np.float16)), 0.0)
        w = np.zeros((4, 2, 3, 3), np.float32)
        w[0] += 1.0
        w[1] += 0.01
        w[3] += 0.5  # le canal 2 est mort
        per_tensor = interp.weight_scale(w, "w", np.float16, False)
        per_channel = interp.weight_scale(w, "w", np.float16, True)
        self.assertEqual(per_channel.shape, (4,))
        self.assertTrue(np.all(per_channel > 0))  # un canal mort garde une échelle représentable
        self.assertLess(float(per_channel[1]), float(per_tensor) / 50)  # le canal faible garde ses bits

    def test_ilot_exact_et_non_sous_chaine(self):  # C46-14
        report = load("layer_report", "trt/layer_report.py")
        self.assertTrue(report.in_island("/Cast_1", {"/Cast_1"}))
        self.assertTrue(report.in_island("PWN(/Cast_1 + /Add)", {"/Cast_1"}))
        self.assertFalse(report.in_island("/Cast_10", {"/Cast_1"}))
        self.assertFalse(report.in_island("/block/Cast_1", {"/Cast_1"}))
        self.assertFalse(report.in_island("/Cast_1_output", {"/Cast_1"}))


class AuditE16Nvof(unittest.TestCase):
    def test_precompute_separe_les_champs_par_tabulation_et_compte_les_trous(self):  # C46-11, C46-12
        try:
            pre = load("precompute_nvof", "nvof_interp/precompute_nvof.py")
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="dossier avec espace ") as tmp:
            t = Path(tmp)
            tool = t / "outil.py"
            tool.write_text(
                "#!%s\nimport sys\nfor line in sys.stdin:\n    f = line.rstrip('\\n').split('\\t')\n"
                "    assert len(f) == 5, f\n    w, h = int(f[0]), int(f[1])\n    if f[4].endswith('1.flow'):\n        continue\n"
                "    open(f[4], 'wb').write(bytes(((w + 3) // 4) * ((h + 3) // 4) * 2 * 2 * 2))\n"
                "    print('ok ' + f[4])\n" % sys.executable)
            tool.chmod(0o755)
            folders = []
            for k in range(2):
                d = t / f"t{k}"
                d.mkdir()
                for name in ("frame1", "frame3"):
                    Image.new("RGB", (16, 8)).save(d / f"{name}.png")
                folders.append(d)
            holes = pre.run_chunk(folders, tool, 0)
            self.assertEqual(holes, 1)
            self.assertTrue((folders[0] / "nvof.npy").exists())
            self.assertFalse((folders[1] / "nvof.npy").exists())


if __name__ == "__main__":
    unittest.main()
