"""Cas du rejeu RIFE hors ligne, avec des déplacements et des occlusions contrôlés."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rife_budget as budget  # noqa: E402


class RifeBudgetTest(unittest.TestCase):
    grid = 4
    width, height = 157, 93
    shift = 8

    def setUp(self):
        y, x = np.indices((self.height, self.width))
        self.a = (36 + ((x // 11 + y // 7) % 2) * 120).astype(np.uint8)
        self.b = np.roll(self.a, self.shift, axis=1)
        self.b[:, :self.shift] = 12  # région découverte
        shape = len(budget.cell_centers(self.height, self.grid)), len(budget.cell_centers(self.width, self.grid))
        flow = np.zeros((2, *shape), dtype=np.float32)
        flow[0] = self.shift
        cost = np.zeros(shape, dtype=np.uint8)
        self.forward = budget.Direction(flow, cost.copy())
        self.backward = budget.Direction(-flow, cost.copy())

    def examine(self):
        return budget.examine(self.a, self.b, self.forward, self.backward, self.grid)

    def test_pan(self):
        row = self.examine()
        self.assertEqual(row["proposition"], budget.MID)
        self.assertLess(row["aller"]["recouvrement"], 1)
        self.assertEqual(row["aller"]["fermeture_p95_720"], 0)

    def test_small_hidden_detail(self):
        # Le champ reste affine et cohérent, mais un trait fin a changé entre les deux sources.
        self.b[43:48, 65] = 240
        row = self.examine()
        self.assertEqual(row["proposition"], budget.ALL)
        self.assertGreater(row["aller"]["contours_douteux"], 0)

    def test_reverse_mismatch(self):
        self.backward.flow[0, 10, 15] += 3
        row = self.examine()
        self.assertEqual(row["proposition"], budget.ALL)
        self.assertGreater(row["retour"]["contours_douteux"], 0)

    def test_reverse_cost(self):
        self.backward.cost[10, 15] = budget.COST_HIGH
        self.assertEqual(self.examine()["proposition"], budget.ALL)

    def test_legacy_capture(self):
        row = budget.examine(self.a, self.b, self.forward, None, self.grid)
        self.assertEqual(row["proposition"], budget.ONE_WAY)
        self.assertIsNone(row["retour"])

    def test_exact_duplicate(self):
        row = budget.examine(self.a, self.a, self.forward, None, self.grid)
        self.assertEqual(row["proposition"], budget.HELD)

    def test_direct_reference_reuses_the_middle(self):
        image = np.repeat(self.a[None], 3, axis=0).astype(np.float32) / 255.0
        flow = np.zeros((4, self.height, self.width), np.float32)
        mask = np.full((1, self.height, self.width), 0.5, np.float32)
        calls = []

        def estimate(t):
            calls.append(t)
            return flow, mask, 0.0

        rows = budget.compare_instants(estimate, image, image, 5)
        self.assertEqual(calls.count(0.5), 1)
        self.assertEqual(len(calls), 5)
        self.assertEqual([i["exact_en_trois_passes"] for i in rows], [True, False, True, False, True])
        self.assertLess(max(i["max_codes"] for i in rows), 0.01)

    def test_colour_edges_are_counted(self):
        ref = np.zeros((3, 8, 8), np.float32)
        ref[:, 3:5, :] = 0.5
        changed = ref.copy()
        changed[0, 3:5, :] += 0.2
        changed[1, 3:5, :] -= 0.2 * 0.2126 / 0.7152  # préserve la luma BT.709
        self.assertAlmostEqual(budget.edge_error(ref, changed), 51.0, places=4)

    def test_tsv_and_file_format(self):
        def packed(direction):
            return ((direction.flow.transpose(1, 2, 0) * 32).astype("<i2").tobytes()
                    + direction.cost.tobytes())

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tasks = base / "taches.tsv"
            self.a.tofile(base / "première image.y")
            self.b.tofile(base / "image b.y")
            path = base / "0.flux"
            path.write_bytes(packed(self.forward) + packed(self.backward))
            tasks.write_text(f"{self.width}\t{self.height}\tpremière image.y\timage b.y\t0.flux\t0.carte\n",
                             encoding="utf-8")
            self.assertEqual(budget.run(tasks, self.grid, 0)[0]["proposition"], budget.MID)
            path.write_bytes(packed(self.forward))
            self.assertEqual(budget.run(tasks, self.grid, 0)[0]["proposition"], budget.ONE_WAY)
            path.write_bytes(packed(self.forward)[:-1])
            with self.assertRaisesRegex(ValueError, "octets"):
                budget.run(tasks, self.grid, 0)


if __name__ == "__main__":
    unittest.main()
