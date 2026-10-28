import tempfile
import unittest
from pathlib import Path

import numpy as np

from ssvep.data import CHANNELS, preprocess_window
from ssvep.baselines import CCA
from ssvep.duration_inference import read_examples
from ssvep.expanded_v9 import score_signals


class DurationInferenceTest(unittest.TestCase):
    def signal(self):
        t = np.arange(1000) / 250
        return np.stack([np.sin(2*np.pi*11*t + k/10) for k in range(6)])

    def test_full_uses_complete_record_and_first_matches_old_baseline(self):
        x = self.signal()
        y = x.copy()
        t = np.arange(625) / 250
        y[:, 375:] = np.stack([np.sin(2*np.pi*9*t + k/10) for k in range(6)])
        scores_x = score_signals([x])
        scores_y = score_signals([y])
        np.testing.assert_allclose(scores_x['first_1p5'], scores_y['first_1p5'])
        np.testing.assert_allclose(scores_x['first_1p5'][0],
            CCA(filter_bank=True).score_one(preprocess_window(x, 375)))
        self.assertGreater(np.linalg.norm(scores_x['full']-scores_y['full']), .01)

    def test_input_ignores_stim_id_and_checks_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.npz'
            signal = self.signal()[None]
            np.savez(path, X=signal, channels=CHANNELS, sampling_rate_hz=250,
                     task_id=[41], stim_id=[7])
            x, ids = read_examples(path)
            np.testing.assert_array_equal(x, signal)
            self.assertEqual(ids.tolist(), [41])
            np.savez(path, X=signal, channels=CHANNELS[::-1], sampling_rate_hz=250,
                     task_id=[41])
            with self.assertRaisesRegex(ValueError, 'electrode order'):
                read_examples(path)


if __name__ == '__main__':
    unittest.main()
