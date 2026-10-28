import tempfile
import unittest
from pathlib import Path

import numpy as np

from ssvep.data import CHANNELS, preprocess_window
from ssvep.expanded_inference import read_examples


class ExpandedInferenceInputTest(unittest.TestCase):
    def test_later_samples_cannot_change_first_window(self):
        t=np.arange(1000)/250
        raw=np.vstack([np.sin(2*np.pi*(8+i)*t) for i in range(6)])
        changed=raw.copy()
        changed[:,375:]+=1000
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'{i}.npz' for i in (1,2)]
            for path,signal in zip(paths,(raw,changed)):
                np.savez(path,X=signal[None],task_id=[7],channels=CHANNELS,sampling_rate_hz=250)
            first,ids=read_examples(paths[0],375)
            second,_=read_examples(paths[1],375)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(first[0],preprocess_window(raw,375))
        self.assertEqual(ids.tolist(),[7])

    def test_rejects_wrong_channel_order(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.npz'
            np.savez(path,X=np.zeros((1,6,375)),task_id=[1],
                     channels=list(reversed(CHANNELS)),sampling_rate_hz=250)
            with self.assertRaisesRegex(ValueError,'electrode order'):
                read_examples(path,375)

if __name__=='__main__': unittest.main()
