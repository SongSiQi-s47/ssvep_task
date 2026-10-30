import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ssvep.data import CHANNELS
from ssvep.robust_inference import infer, load_input, validate_frequency_map


class RobustInferenceTest(unittest.TestCase):
    def signal(self, fs=250):
        t = np.arange(fs*4)/fs
        return np.stack([np.sin(2*np.pi*11*t+k/6)+.1*np.sin(2*np.pi*22*t)
                         for k in range(6)])

    def write(self, path, x, names=CHANNELS, fs=250, freqs=np.arange(8,16)):
        np.savez(path, X=x[None], channels=names, sampling_rate_hz=fs,
                 task_id=[17], stim_frequencies_hz=freqs, stim_id=[7])

    def test_channel_order_scale_and_extra_channel_do_not_change_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder)/'first.npz'
            second = Path(folder)/'second.npz'
            signal = self.signal()
            self.write(first, signal)
            order = [5,2,0,4,1,3]
            extra = np.ones((1,signal.shape[-1]))
            scrambled = np.vstack([signal[order]*np.array([3,.5,2,4,1,5])[:,None],extra])
            names = [CHANNELS[i] for i in order]+['EXTRA']
            self.write(second, scrambled, names=names)
            ids_a, pred_a, _ = infer(first)
            ids_b, pred_b, _ = infer(second)
            np.testing.assert_array_equal(ids_a, ids_b)
            np.testing.assert_array_equal(pred_a, pred_b)

    def test_resampled_500_hz_and_frequency_map_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder)/'250.npz'
            b = Path(folder)/'500.npz'
            self.write(a, self.signal(250))
            self.write(b, self.signal(500), fs=500)
            self.assertEqual(int(infer(a)[1][0]), int(infer(b)[1][0]))
            with self.assertRaisesRegex(ValueError,'conflicts'):
                load_input(a, [9,8,10,11,12,13,14,15])
            with self.assertRaisesRegex(ValueError,'distinct'):
                validate_frequency_map([8]*8)

    def test_explicit_label_frequency_reordering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'reordered_labels.npz'
            frequencies = np.arange(8, 16, dtype=float)
            frequencies[[0, 3]] = frequencies[[3, 0]]
            self.write(path, self.signal(), freqs=frequencies)
            self.assertEqual(int(infer(path)[1][0]), 0)

    def test_missing_metadata_and_channel_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npz'
            np.savez(path,X=self.signal()[None],channels=CHANNELS,
                     sampling_rate_hz=250,task_id=[17])
            with self.assertRaisesRegex(ValueError,'Frequency map'):
                load_input(path)
            np.savez(path,X=self.signal()[None,:5,:],channels=CHANNELS[:5],
                     sampling_rate_hz=250,task_id=[17],stim_frequencies_hz=np.arange(8,16))
            with self.assertRaisesRegex(ValueError,'channel'):
                load_input(path)


if __name__=='__main__': unittest.main()
