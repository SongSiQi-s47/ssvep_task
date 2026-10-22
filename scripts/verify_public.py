"""Read completed MATLAB files and record scientific structure checks."""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']


def main():
    report = {}
    for source in sorted((ROOT / 'data/raw/public').rglob('*.mat')):
        record = {'file': str(source.relative_to(ROOT))}
        try:
            data = loadmat(source, simplify_cells=True)
            if source.parent.name == 'beta':
                item = data['data']
                eeg, info = item['EEG'], item['suppl_info']
                names = [str(x).upper() for x in info['chan'][:, 3]]
                indices = [names.index(ch) for ch in CHANNELS]
                if eeg.shape[0] != 64 or eeg.shape[2:] != (4, 40):
                    raise ValueError('Unexpected BETA dimensions')
                if int(info['srate']) != 250 or len(info['freqs']) != 40:
                    raise ValueError('Unexpected BETA metadata')
                record.update(sampling_rate_hz=int(info['srate']),
                              six_channel_indices_0based=indices,
                              frequencies_in_original_order_hz=info['freqs'].tolist(),
                              axes=['channel', 'time', 'block', 'condition'])
            else:
                eeg = data['eeg']
                if eeg.shape != (12, 8, 1114, 15):
                    raise ValueError('Unexpected Nakanishi dimensions')
                record.update(sampling_rate_hz=256, axes=['target', 'channel', 'time', 'trial'])
            record.update(shape=list(eeg.shape), finite=bool(np.isfinite(eeg).all()))
            if not record['finite']:
                raise ValueError('Non-finite EEG values')
            record['status'] = 'passed'
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        report[str(source.relative_to(ROOT))] = record
    (ROOT / 'docs/public_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    failures = sum(item['status'] != 'passed' for item in report.values())
    print('MAT_VALIDATION', len(report), 'files;', failures, 'failures', flush=True)
    return failures == 0


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
