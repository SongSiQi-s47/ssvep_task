"""Metadata-checked, label-free FBCCA for organizer-style NPZ inputs.

Frequency map must be explicit in the input or on the CLI. Input has no query labels.
"""
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from .baselines import CCA
from .data import CHANNELS, preprocess_window


def validate_frequency_map(values):
    f = np.asarray(values, dtype=np.float64)
    if f.shape != (8,) or not np.isfinite(f).all() or len(set(f.tolist())) != 8:
        raise ValueError('Need eight distinct finite frequencies for labels 0..7')
    if np.min(f) < 6 or np.max(f) > 25:
        raise ValueError('Frequencies outside validated 6-25 Hz analytic range')
    return f


def load_input(path, external_frequencies=None):
    with np.load(path, allow_pickle=False) as data:
        required = {'X', 'channels', 'sampling_rate_hz', 'task_id'}
        if not required <= set(data.files):
            raise ValueError(f'Missing fields: {sorted(required-set(data.files))}')
        raw = data['X'].copy()
        ids = data['task_id'].copy()
        names = [str(s).upper() for s in data['channels'].tolist()]
        fs = float(data['sampling_rate_hz'])
        embedded = (data['stim_frequencies_hz'].copy()
                    if 'stim_frequencies_hz' in data.files else None)
    if raw.ndim != 3 or raw.shape[1] != len(names) or raw.shape[0] != len(ids):
        raise ValueError('Expected X [trial,channel,time] with matching metadata')
    if not len(raw) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Empty input or duplicate task ID')
    if len(set(names)) != len(names) or not set(CHANNELS) <= set(names):
        raise ValueError('Missing or duplicated required channel names')
    if not np.isfinite(raw).all() or not np.isfinite(fs) or fs < 200 or fs > 1000:
        raise ValueError('Non-finite signal or unsupported sampling rate')
    if external_frequencies is None and embedded is None:
        raise ValueError('Frequency map is required; do not assume old 0..7 mapping')
    f = validate_frequency_map(embedded if external_frequencies is None else external_frequencies)
    if embedded is not None and external_frequencies is not None:
        if not np.allclose(f, validate_frequency_map(embedded), atol=1e-6, rtol=0):
            raise ValueError('Explicit frequency map conflicts with input metadata')
    index = [names.index(name) for name in CHANNELS]
    signals = raw[:, index, :]
    if fs != 250:
        ratio = Fraction(250 / fs).limit_denominator(1000)
        if abs(float(ratio) - 250 / fs) > 1e-8:
            raise ValueError('Sampling rate cannot be resampled accurately')
        signals = resample_poly(signals, ratio.numerator, ratio.denominator, axis=-1)
    if signals.shape[-1] < 375:
        raise ValueError('Less than 375 samples after alignment/resampling')
    audit = {'source': str(path), 'n_trials': len(ids), 'input_fs_hz': fs,
             'resampled_fs_hz': 250, 'input_channels': names,
             'selected_channels': CHANNELS, 'stim_frequencies_hz': f.tolist(),
             'samples_per_trial_after_resample': int(signals.shape[-1]),
             'query_labels_read': False}
    return signals, ids, f, audit


def infer(path, external_frequencies=None):
    signals, ids, frequencies, audit = load_input(path, external_frequencies)
    model = CCA(frequencies=frequencies, fs=250, harmonics=3, filter_bank=True)
    scores = np.stack([model.score_one(preprocess_window(raw, raw.shape[-1]))
                       for raw in signals])
    return ids, scores.argmax(axis=1), audit


def main():
    p = argparse.ArgumentParser(description='Metadata-checked development CSV; not organizer Excel template')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frequency-map', type=float, nargs=8, metavar='HZ')
    p.add_argument('--audit-output', type=Path)
    args = p.parse_args()
    if args.output.exists() or (args.audit_output and args.audit_output.exists()):
        raise FileExistsError('Refusing to overwrite an existing output')
    with threadpool_limits(limits=1):
        ids, predictions, audit = infer(args.input, args.frequency_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['taskID', 'predicted_stimID'])
        writer.writerows(zip(ids.tolist(), predictions.tolist()))
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote', len(ids), 'predictions:', args.output)


if __name__ == '__main__':
    main()
