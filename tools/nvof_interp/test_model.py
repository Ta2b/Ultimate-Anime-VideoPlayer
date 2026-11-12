#!/usr/bin/env python3
"""Tests de model.py (FlowJaNai), avec PyTorch : python3 -I tools/nvof_interp/test_model.py

Hors de la CI (qualite.yml n'installe pas PyTorch) : à lancer avant de toucher model.py ou train.py."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FlowJaNai, warp  # noqa: E402


class Consistency(unittest.TestCase):
    def test_flux_coherents_sans_ecart_sur_un_mouvement_non_uniforme(self):  # C46-03
        # Un bloc de 4 × 4 cellules NVOF (16 × 16 px) se déplace de 24 px vers la droite (6 cellules) ; le fond
        # reste immobile. Les flux sont en pixels de l'image, sur la grille 1/4 : avant-arrière cohérents.
        h4, w4, d = 16, 40, 24.0
        nvof = torch.zeros(1, 4, h4, w4)
        nvof[0, 0, 4:8, 4:8] = d  # F01 sur le bloc d'origine
        nvof[0, 2, 4:8, 10:14] = -d  # F10 sur le bloc d'arrivée
        error = FlowJaNai.consistency(nvof)
        # Là où le bloc est, l'écart avant-arrière est nul (log1p(0) = 0). Avant le ÷4, l'écart valait ~ 24 px.
        self.assertLess(float(error[0, 0, 4:8, 4:8].max()), 1e-4)
        self.assertLess(float(error[0, 1, 4:8, 10:14].max()), 1e-4)


class Warp(unittest.TestCase):
    def test_grille_en_fp32_sous_autocast_et_entree_fp16(self):  # C46-13
        image = torch.rand(1, 3, 8, 1280)
        flow = torch.zeros(1, 2, 8, 1280)
        flow[:, 0] = 0.25  # un quart de pixel : disparaît si la grille est calculée en FP16 à 1280 px
        reference = warp(image, flow)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            sous_autocast = warp(image, flow)
        self.assertTrue(torch.allclose(reference, sous_autocast, atol=1e-5))
        decale = warp(image, flow)[0, :, 0, 100]
        voulu = 0.75 * image[0, :, 0, 100] + 0.25 * image[0, :, 0, 101]
        self.assertTrue(torch.allclose(decale, voulu, atol=1e-4))
        # Hors autocast, une image FP16 reçoit la grille dans son type (comme avant).
        self.assertEqual(warp(image.half(), flow.half()).dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
