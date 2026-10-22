import unittest
import numpy as np
from ssvep.data import preprocess_window
from ssvep.baselines import CCA


class WindowScope(unittest.TestCase):
    def test_outside_window_cannot_change_output(self):
        x = np.random.default_rng(1).normal(size=(6, 715))
        y = x.copy()
        y[:, :50] = np.nan
        y[:, 425:] = 1e9
        np.testing.assert_array_equal(preprocess_window(x, 375, 50),
                                      preprocess_window(y, 375, 50))

    def test_bounds_and_finite(self):
        for n, start in [(750, 0), (250, -1), (100, 0)]:
            with self.assertRaises(ValueError):
                preprocess_window(np.zeros((6, 465)), n, start)
        z = preprocess_window(np.zeros((6, 465)), 250)
        self.assertTrue(np.isfinite(z).all())
        invalid = np.zeros((6, 465))
        invalid[0, 100] = np.nan
        with self.assertRaises(ValueError):
            preprocess_window(invalid, 250)

    def test_all_frequency_harmonics_both_lengths(self):
        rng = np.random.default_rng(7)
        for n in [250, 375]:
            t = np.arange(n) / 250
            for f in range(8, 16):
                x = np.stack([np.sin(2*np.pi*f*t+p) + .4*np.sin(4*np.pi*f*t+p)
                              for p in np.linspace(0, 1, 6)])
                x += .1*rng.normal(size=x.shape)
                score = CCA(filter_bank=True).score_one(preprocess_window(x, n))
                self.assertTrue(np.isfinite(score).all())
                self.assertEqual(int(score.argmax()), f-8)
