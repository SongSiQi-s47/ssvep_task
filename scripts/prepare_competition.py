"""Preserve every original trial; sampling rate and physical units remain unknown."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']


def main():
    alignment=json.loads((ROOT/'configs/data_alignment.json').read_text(encoding='utf-8'))
    sampling_rate=alignment['competition']['sampling_rate_hz']
    inventory = {}
    records = []
    for source in sorted((ROOT / 'data/raw/competition').glob('D*.csv')):
        with source.open(encoding='utf-8-sig') as stream:
            header = next(csv.reader(stream))
        if header != CHANNELS + ['taskID', 'stimID']:
            raise ValueError(f'Unexpected columns: {source.name}: {header}')
        data = np.loadtxt(source, delimiter=',', skiprows=1)
        if not np.isfinite(data).all():
            raise ValueError(f'Non-finite values in {source.name}')
        boundaries = np.r_[0, np.flatnonzero(np.diff(data[:, 6]) != 0) + 1, len(data)]
        trials, labels, task_ids, segments = [], [], [], []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            markers = data[begin:end, 6:]
            if not np.all(markers == markers[0]) or not np.all(markers[0] == np.floor(markers[0])):
                raise ValueError(f'Inconsistent markers in {source.name}')
            task, label = map(int, markers[0])
            if label not in range(8):
                raise ValueError('Unexpected sample stimulus ID')
            trials.append(data[begin:end, :6].T)
            labels.append(label)
            task_ids.append(task)
            segment = {'source_file': source.name, 'taskID': task, 'stimID': label,
                       'start_row_0based': int(begin), 'end_row_exclusive': int(end),
                       'samples': int(end - begin)}
            segments.append(segment)
            records.append(segment)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('Repeated task IDs require explicit interpretation')
        tensor = np.stack(trials)
        out = ROOT / 'data/processed/competition' / (source.stem + '_trials.npz')
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata={} if sampling_rate is None else {'sampling_rate_hz':np.array(sampling_rate)}
        np.savez_compressed(out, X=tensor, stim_id=np.array(labels),
                            task_id=np.array(task_ids), channels=np.array(CHANNELS),**metadata)
        with np.load(out, allow_pickle=False) as saved:
            if not np.array_equal(saved['X'], tensor):
                raise ValueError('Saved trials differ from source')
        inventory[source.name] = {'rows': len(data), 'trial_shape': list(tensor.shape),
                                 'class_counts': {str(i): labels.count(i) for i in range(8)},
                                 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'finite': True, 'sampling_rate_hz': sampling_rate,
                                 'frequency_mapping_hz': alignment['competition']['stim_id_to_frequency_hz'], 'units': None,
                                 'subject_identity': 'unconfirmed', 'segments': segments}
    (ROOT / 'docs/competition_inventory.json').write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    with (ROOT / 'data/processed/competition/trial_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: {f: v for f, v in d.items() if f != 'segments'}
                      for k, d in inventory.items()}, indent=2))


if __name__ == '__main__':
    main()
