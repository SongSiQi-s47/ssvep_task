"""Recompute the v11 BETA input-compatibility audit without modifying its record."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssvep.data import CHANNELS
from ssvep.expanded_v5 import ROOT, get_trials
from ssvep.expanded_v9 import score
from ssvep.robust_inference import infer

SUBJECTS = ('S64', 'S28', 'S60', 'S27')
RESULT = ROOT / 'runs/v11-one-shot-support/robustness_check.json'


def main():
    result = {}
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as folder:
        for subject in SUBJECTS:
            trials = get_trials([subject])
            raw = np.stack([t.x for t in trials])
            labels = np.asarray([t.y for t in trials])
            baseline = score(trials, ('full',))['full'].argmax(axis=1)
            order = [5, 2, 0, 4, 1, 3]
            extra = np.zeros((len(raw), 1, raw.shape[-1]))
            reordered = np.concatenate((raw[:, order, :] *
                np.asarray([3, .5, 2, 4, 1, 5])[None, :, None], extra), axis=1)
            variants = (
                ('original', raw, CHANNELS, 250),
                ('reordered', reordered, [CHANNELS[i] for i in order] + ['EXTRA'], 250),
                ('resampled500', resample_poly(raw, 2, 1, axis=-1), CHANNELS, 500),
            )
            for variant, signal, names, fs in variants:
                path = Path(folder) / f'{subject}_{variant}.npz'
                np.savez(path, X=signal, channels=names, sampling_rate_hz=fs,
                         task_id=np.arange(len(raw)), stim_frequencies_hz=np.arange(8, 16))
                _, prediction, _ = infer(path)
                result.setdefault(variant, []).append({
                    'subject': subject, 'accuracy': float(np.mean(prediction == labels)),
                    'prediction_matches_v9': int(np.sum(prediction == baseline)),
                    'n_trials': len(labels),
                })
    recorded = json.loads(RESULT.read_text(encoding='utf-8'))
    if result != recorded:
        raise ValueError('Input compatibility audit differs from recorded results')
    for variant, rows in result.items():
        print(variant, sum(r['prediction_matches_v9'] for r in rows), '/',
              sum(r['n_trials'] for r in rows), 'predictions match v9')


if __name__ == '__main__':
    main()
