#!/usr/bin/env python3
"""Tests du banc de qualité, sans modèle livré ni vidéo : python3 -I tools/bench/qualite/test_banc.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32  # noqa: E402
import mesures  # noqa: E402
import modeles  # noqa: E402
import rapport  # noqa: E402
import sources  # noqa: E402
import travail  # noqa: E402

F16 = TensorProto.FLOAT16


def save(nodes, inputs, outputs, initializers, folder, name):
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info(n, F16, s) for n, s in inputs],
                              [helper.make_tensor_value_info(n, F16, s) for n, s in outputs], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = os.path.join(folder, name)
    onnx.save(model, path)
    return path


def fake_rife(folder, planes=11):
    """Un « RIFE » FP16 qui rend (a + b) × gain : assez pour suivre l'entrée, la fenêtre et la précision."""
    def const(name, values):
        return numpy_helper.from_array(np.array(values, np.int64), name)

    gain = numpy_helper.from_array(np.array([1 / 3], np.float16).reshape(1, 1, 1, 1), "gain")
    nodes = [helper.make_node("Slice", ["input", "a0", "a1", "axis"], ["a"]),
             helper.make_node("Slice", ["input", "b0", "b1", "axis"], ["b"]),
             helper.make_node("Add", ["a", "b"], ["sum"]),
             helper.make_node("Mul", ["sum", "gain"], ["output"])]
    inits = [const("a0", [0]), const("a1", [3]), const("b0", [3]), const("b1", [6]), const("axis", [1]), gain]
    return save(nodes, [("input", [1, planes, "h", "w"])], [("output", [1, 3, "h", "w"])], inits, folder, "rife.onnx")


def fake_drba(folder):
    """Un « DistilDRBA » FP16 : la somme des images du centre des deux paires (plans 3 à 5 et 0 à 2)."""
    def const(name, values):
        return numpy_helper.from_array(np.array(values, np.int64), name)

    nodes = [helper.make_node("Slice", ["pair01", "b0", "b1", "axis"], ["x"]),
             helper.make_node("Slice", ["pair12", "a0", "a1", "axis"], ["y"]),
             helper.make_node("Add", ["x", "y"], ["s"]),
             helper.make_node("Mul", ["s", "half"], ["out"])]
    inits = [const("a0", [0]), const("a1", [3]), const("b0", [3]), const("b1", [6]), const("axis", [1]),
             numpy_helper.from_array(np.array([0.5], np.float16).reshape(1, 1, 1, 1), "half")]
    return save(nodes, [("pair01", [1, 7, "h", "w"]), ("pair12", [1, 7, "h", "w"])], [("out", [1, 3, "h", "w"])],
                inits, folder, "drba.onnx")


def picture(seed, h=1080, w=1920):
    return np.random.default_rng(seed).random((3, h, w), dtype=np.float32)


class Mesures(unittest.TestCase):
    def test_images_identiques(self):
        a = picture(1, 64, 64)
        m = mesures.compare(a, a)
        self.assertEqual(m["psnr"], 200.0)
        self.assertAlmostEqual(m["ssim"], 1.0)
        self.assertEqual(m["ecart_pire"], 0.0)
        self.assertEqual(m["itp_p999"], 0.0)

    def test_un_code_d_ecart(self):
        a = np.full((3, 64, 64), 0.5, np.float32)
        m = mesures.compare(a, a + 1 / 255)
        self.assertAlmostEqual(m["psnr"], 20 * np.log10(255), places=3)
        self.assertAlmostEqual(m["ecart_pire"], 1.0, places=4)
        self.assertAlmostEqual(m["ecart_moyen"], 1.0, places=4)
        self.assertGreater(m["itp_moyen"], 0.3)  # un code de gris moyen : sous le seuil visible, mais pas nul
        self.assertLess(m["itp_moyen"], 3.0)

    def test_blanc_sdr_en_ictcp(self):
        i, ct, cp = mesures.ictcp(np.ones((3, 2, 2), np.float32))[:, 0, 0]
        self.assertAlmostEqual(i, 0.5807, places=3)  # PQ de 203 cd/m²
        self.assertAlmostEqual(ct, 0.0, places=3)
        self.assertAlmostEqual(cp, 0.0, places=3)


class Sources(unittest.TestCase):
    def test_le_fond_glisse_d_un_nombre_entier_de_pixels(self):
        win = sources.scene(picture(2), picture(3), (640, 384), (6, 2), (0, 0))
        self.assertEqual(len(win), sources.STEPS)
        # Hors du personnage (coin haut gauche), l'image k+1 est l'image k décalée de (6, 2).
        np.testing.assert_array_equal(win[1][:, 0:40, 0:40], win[0][:, 2:42, 6:46])
        np.testing.assert_array_equal(win[2][:, 0:40, 0:40], win[1][:, 2:42, 6:46])

    def test_le_personnage_tient_la_pose_ou_glisse(self):
        back, fig = picture(4), picture(5)
        held = sources.scene(back, fig, (640, 384), (8, 0), (0, 0))
        np.testing.assert_array_equal(held[0][:, 180:200, 310:330], held[4][:, 180:200, 310:330])
        moved = sources.scene(back, fig, (640, 384), (0, 0), (-4, 0))
        np.testing.assert_array_equal(moved[1][:, 180:200, 300:320], moved[0][:, 180:200, 304:324])

    def test_mouvement_hors_de_l_image_refuse(self):
        with self.assertRaises(SystemExit):
            sources.scene(picture(6), picture(7), (1920, 1080), (2, 0), (0, 0))

    def test_reduction_et_bicubique(self):
        flat = np.full((3, 16, 16), 0.25, np.float32)
        np.testing.assert_allclose(sources.halve(flat), 0.25)
        np.testing.assert_allclose(sources.bicubic2(flat), 0.25, atol=1e-6)
        self.assertEqual(sources.bicubic2(flat).shape, (3, 32, 32))

    def test_fenetres_d_une_video(self):
        images = [picture(i, 16, 16) for i in range(8)]
        images[3] = images[2]  # un dessin tenu : les fenêtres 1 et 2 n'ont pas trois images différentes
        names = [n for n, _ in sources.windows(images, 10, cut=2.0)]
        self.assertEqual(names, ["image-0", "image-3"])
        cut = sources.windows(images, 10, cut=0.0)[0][1]
        self.assertIs(cut[4], cut[2])  # coupe devant k+4 : le bord est répété


class Modeles(unittest.TestCase):
    def test_fenetre_comme_rife_window(self):
        out, (top, left) = modeles.pad(np.ones((3, 720, 1280), np.float32))
        self.assertEqual((out.shape, top, left), ((3, 768, 1280), 24, 0))
        self.assertEqual(float(out[:, :24].max()), 0.0)
        _, (top, left) = modeles.pad(np.ones((3, 90, 150), np.float32))
        self.assertEqual((top, left), (18, 20))  # (128 − 90) / 2 = 19 et (192 − 150) / 2 = 21, ramenés au pair

    def test_grille_du_v1(self):
        x = modeles.rife_input(np.zeros((3, 64, 128), np.float32), np.zeros((3, 64, 128), np.float32), 0.5, 11)
        self.assertEqual((x[0, 7, 0, 0], x[0, 7, 0, -1], x[0, 8, 0, 0], x[0, 8, -1, 0]), (-1.0, 1.0, -1.0, 1.0))
        self.assertAlmostEqual(float(x[0, 9, 0, 0]), 2 / 127)
        self.assertEqual(float(x[0, 6].min()), 0.5)

    def test_rife_recoit_k_et_k_plus_2(self):
        with tempfile.TemporaryDirectory() as d:
            net = modeles.Network(fake_rife(d))
            self.assertEqual((net.kind, net.planes), ("rife", 11))
            win = [np.full((3, 90, 150), v, np.float32) for v in (0.3, 0.9, 0.6, 0.9, 0.9)]
            out = net.middle(win)
            self.assertEqual(out.shape, (3, 90, 150))
            np.testing.assert_allclose(out, 0.3, atol=2e-3)  # (0,3 + 0,6) / 3

    def test_drba_recoit_k_plus_2_au_centre(self):
        with tempfile.TemporaryDirectory() as d:
            net = modeles.Network(fake_drba(d))
            self.assertEqual(net.kind, "drba")
            win = [np.full((3, 64, 64), v, np.float32) for v in (0.1, 0.9, 0.5, 0.9, 0.2)]
            np.testing.assert_allclose(net.middle(win), 0.5, atol=1e-3)

    def test_reference_fp32(self):
        with tempfile.TemporaryDirectory() as d:
            path = fake_rife(d)
            promoted = fp32.promote(onnx.load(path))
            self.assertTrue(fp32.has_fp16(onnx.load(path)))
            self.assertFalse(fp32.has_fp16(promoted))
            # Le poids garde la valeur du fichier livré : 1/3 arrondi en demi-précision.
            gain = [numpy_helper.to_array(t) for t in promoted.graph.initializer if t.name == "gain"][0]
            self.assertEqual(gain.dtype, np.float32)
            self.assertEqual(float(gain.ravel()[0]), float(np.float16(1 / 3)))
            ref, shipped = modeles.Network(path, reference=True), modeles.Network(path)
            self.assertEqual((ref.dtype, shipped.dtype), (np.float32, np.float16))
            win = [picture(i, 64, 64) for i in range(5)]
            m = mesures.compare(ref.middle(win), shipped.middle(win))
            self.assertGreater(m["psnr"], 60.0)
            self.assertLess(m["psnr"], 200.0)  # le FP16 arrondit : les deux sorties ne sont pas identiques


class Seuils(unittest.TestCase):
    def report(self, psnr=60.0, codes=4.0, truth=40.0):
        return {"config": {"taille": "1280x720", "scenes": 8, "plans": 6, "source": "synthese"},
                "reseaux": [{"modele": "r.onnx", "fidelite_psnr_pire": psnr, "fidelite_ecart_pire": codes,
                             "verite_psnr": truth}]}

    def test_une_baisse_passe_le_seuil(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seuils.json")
            rapport.write_gates(path, self.report())
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["reseaux"]["r.onnx"]["fidelite_psnr_pire_min"], 59.0)
            self.assertEqual(rapport.check_gates(path, self.report()), [])
            self.assertEqual(len(rapport.check_gates(path, self.report(psnr=58.5))), 1)
            self.assertEqual(len(rapport.check_gates(path, self.report(codes=6.0))), 1)
            self.assertEqual(len(rapport.check_gates(path, self.report(truth=39.5))), 1)

    def test_autre_configuration_sans_seuil(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seuils.json")
            rapport.write_gates(path, self.report())
            other = self.report(psnr=10.0)
            other["config"]["taille"] = "640x360"
            self.assertEqual(rapport.check_gates(path, other), [])


class Travail(unittest.TestCase):
    def test_reechantillonnage(self):
        flat = np.full((3, 54, 96), 0.25, np.float32)
        for size in ((64, 36), (96, 54), (144, 81), (192, 108)):
            out = travail.resize(flat, size, travail.hermite, 1)
            self.assertEqual(out.shape, (3, size[1], size[0]))
            self.assertLess(float(np.abs(out - 0.25).max()), 1e-6)  # un aplat reste un aplat, bords compris
        img = picture(3, 54, 96)
        self.assertIs(travail.resize(img, (96, 54), travail.hermite, 1), img)  # même taille : rien
        # Une rampe horizontale réduite de moitié reste la même rampe (centres demi-pixel, pas de décalage).
        ramp = np.broadcast_to(np.linspace(0.0, 1.0, 96, dtype=np.float32), (3, 54, 96)).copy()
        half = travail.resize(ramp, (48, 27), travail.hermite, 1)
        want = ramp.reshape(3, 27, 2, 48, 2).mean((2, 4))
        self.assertLess(float(np.abs(half - want)[:, :, 2:-2].max()), 1e-3)
        # Vers le haut, Lanczos rend l'image de départ une fois ramenée par moyenne : aucune dérive de phase.
        up = travail.to_screen(ramp, (192, 108))
        back = up.reshape(3, 54, 2, 96, 2).mean((2, 4))
        self.assertLess(float(np.abs(back - ramp)[:, :, 4:-4].max()), 2e-3)

    def test_largeur_paire(self):
        self.assertEqual(travail.width_of(540), 960)
        self.assertEqual(travail.width_of(360), 640)
        self.assertEqual(travail.width_of(270), 480)


if __name__ == "__main__":
    unittest.main()
