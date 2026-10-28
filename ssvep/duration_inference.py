"""Label-free prediction from six-channel 250 Hz organizer-style NPZ records."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .data import CHANNELS, fingerprint
from .expanded_v9 import CONFIG_PATH, OUT, ROOT, score_signals


def read_examples(path):
    with np.load(path, allow_pickle=False) as data:
        required = {'X', 'channels', 'sampling_rate_hz', 'task_id'}
        if not required.issubset(data.files):
            raise ValueError(f'Missing input fields: {sorted(required - set(data.files))}')
        if int(data['sampling_rate_hz']) != 250 or data['channels'].tolist() != CHANNELS:
            raise ValueError('Sampling rate or electrode order differs from v9 protocol')
        x, ids = data['X'].copy(), data['task_id'].copy()
    if x.ndim != 3 or x.shape[1] != 6 or x.shape[-1] < 375 or len(x) != len(ids):
        raise ValueError('Expected [trial,6,time>=375] and one task ID per trial')
    if len(set(ids.tolist())) != len(ids) or not np.isfinite(x).all():
        raise ValueError('Duplicate task ID or non-finite signal')
    return x, ids


def infer(path):
    frozen = json.loads((OUT / 'selection_frozen.json').read_text(encoding='utf-8'))
    completed = json.loads((OUT / 'completed.json').read_text(encoding='utf-8'))
    if not frozen['complete'] or not completed['complete'] or frozen['config_sha256'] != fingerprint(CONFIG_PATH):
        raise ValueError('v9 selection or run is incomplete/changed')
    algorithm = completed['default_algorithm']
    if algorithm not in ('full', 'first_1p5'):
        raise ValueError('Unknown deployed v9 algorithm')
    raw, ids = read_examples(path)
    scores = score_signals(raw, (algorithm,))[algorithm]
    return ids, scores.argmax(axis=1)


def main():
    p = argparse.ArgumentParser(description='Label-free v9 prediction; internal CSV, not organizer template')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        ids, predictions = infer(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['taskID', 'predicted_stimID'])
        writer.writerows(zip(ids.tolist(), predictions.tolist()))
    print('Wrote', len(ids), 'predictions to', args.output)


if __name__ == '__main__':
    main()
