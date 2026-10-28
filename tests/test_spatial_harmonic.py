import unittest
import numpy as np
from ssvep.expanded_v9 import score_signals as old_score
from ssvep.expanded_v10 import score_signals as new_score


class SpatialHarmonicTest(unittest.TestCase):
    def test_v10_baseline_preserves_deployed_v9_scores(self):
        t = np.arange(715) / 250
        signal = np.stack([np.sin(2*np.pi*11*t + k/8) for k in range(6)])
        expected = old_score([signal], ('full',))['full']
        actual = new_score([signal], ('fbcca3_all',))['fbcca3_all']
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_posterior_variant_ignores_cp_channels(self):
        t = np.arange(715) / 250
        signal = np.stack([np.sin(2*np.pi*(8+k)*t) for k in range(6)])
        changed = signal.copy()
        changed[:3] += np.stack([4*np.sin(2*np.pi*13*t+i/2) for i in range(3)])
        a = new_score([signal], ('fbcca5_posterior',))['fbcca5_posterior']
        b = new_score([changed], ('fbcca5_posterior',))['fbcca5_posterior']
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__': unittest.main()
