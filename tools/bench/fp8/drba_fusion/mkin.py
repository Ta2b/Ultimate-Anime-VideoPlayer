#!/usr/bin/env python3
"""Écrit les entrées .npy d'un cas : mkin.py <image> <décalage> <tau> <préfixe>."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import entrees_drba, trois_images

nom, decalage, tau, sortie = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
p01, p12 = entrees_drba(trois_images(nom, decalage), tau)
np.save(sortie + "_p01.npy", p01)
np.save(sortie + "_p12.npy", p12)
