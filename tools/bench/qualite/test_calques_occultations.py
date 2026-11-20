import unittest

import numpy as np

from calques_occultations import GRID, direction_evidence, extend_protection
from rife_budget import Direction, cell_centers


def motion(x, y, width, height):
    cx, cy = cell_centers(width, GRID), cell_centers(height, GRID)
    vx = np.broadcast_to(x(cx[None, :], cy[:, None]), (len(cy), len(cx)))
    vy = np.broadcast_to(y, vx.shape)
    return Direction(np.stack((vx, vy)).astype(np.float32), np.zeros(vx.shape, np.uint8))


class OcclusionTest(unittest.TestCase):
    def test_chromatic_edge_and_protected_neighbour(self):
        blue = np.array([78, 91, 165]) / 255
        pink = np.array([108, 78, 205]) / 255
        luma = np.array([0.2126, 0.7152, 0.0722])
        self.assertLess(abs(255 * ((blue - pink) @ luma)), 1)
        width, height = 64, 48
        a = np.broadcast_to(blue[:, None, None], (3, height, width)).copy()
        b = a.copy()
        a[:, 16:32, 16:32] = pink[:, None, None]
        b[:, 16:32, 24:40] = pink[:, None, None]
        f = motion(lambda x, y: 8 * ((16 <= x) & (x < 32) & (16 <= y) & (y < 32)), 0, width, height)
        r = motion(lambda x, y: -8 * ((24 <= x) & (x < 40) & (16 <= y) & (y < 32)), 0, width, height)
        _, _, guard = direction_evidence(a, b, f, r, (width, height), np.eye(3))
        self.assertTrue(guard[4:8, 8:10].all())  # Le décor de a disparaît derrière la forme.
        self.assertFalse(guard[4:8, 5:7].any())  # La forme se retrouve dans b.
        decor = np.ones((3, 4), dtype=bool)
        self.assertEqual(extend_protection(guard, decor, 4).sum(), 0)
        decor[1, 1] = False
        expected = np.zeros_like(decor)
        expected[1, 2] = True
        np.testing.assert_array_equal(extend_protection(guard, decor, 4), expected)
        isolated = np.zeros_like(guard)
        isolated[4, 8] = True
        self.assertEqual(extend_protection(isolated, decor, 4).sum(), 0)

    def test_stripes_that_follow_camera_are_not_occluded(self):
        width, height = 64, 48
        stripe = (np.arange(width) // 4 % 2) * 0.6 + 0.2
        a = np.broadcast_to(stripe[None, None, :], (3, height, width)).copy()
        b = np.roll(a, 4, axis=2)
        wrong = motion(lambda x, y: 0, 0, width, height)
        reverse = motion(lambda x, y: -4, 0, width, height)
        camera = np.eye(3)
        camera[0, 2] = 4
        _, photometric, guard = direction_evidence(a, b, wrong, reverse, (width, height), camera)
        self.assertTrue(photometric[:, 2:-2].all())
        self.assertFalse(guard.any())

    def test_brightness_jump_with_closed_flow_is_not_occlusion(self):
        width, height = 64, 48
        a = np.full((3, height, width), 0.2)
        b = a + 0.4
        zero = motion(lambda x, y: 0, 0, width, height)
        closure, _, guard = direction_evidence(a, b, zero, zero, (width, height), np.eye(3))
        self.assertFalse(closure.any())
        self.assertFalse(guard.any())


if __name__ == "__main__":
    unittest.main()
