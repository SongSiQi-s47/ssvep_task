import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, sosfilt

CHANNELS = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
FREQUENCIES = np.arange(8., 16.)


@dataclass
class Trial:
    x: np.ndarray
    y: int
    source: str
    subject: str
    uid: str
    block: int


def preprocess(x, fs=250, causal=False):
    """Per-record filtering and channel normalization; no cohort statistics fitted."""
    if x.ndim != 2 or x.shape[0] != 6 or x.shape[1] < 125:
        raise ValueError('Expected six channels and at least 0.5 seconds')
    if not np.isfinite(x).all():
        raise ValueError('Non-finite input')
    x = np.asarray(x, dtype=np.float64)
    # Offline competition mode. Causal mode is a separate experimental setting.
    sos = butter(4, [6, 90], fs=fs, btype='bandpass', output='sos')
    x = sosfilt(sos, x, axis=-1) if causal else sosfiltfilt(sos, x, axis=-1)
    x -= x.mean(axis=-1, keepdims=True)
    x /= np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    return x.astype(np.float32)


def preprocess_window(x, samples, start=0, fs=250):
    """Offline bounded observation: crop BEFORE filtering and normalization."""
    if not isinstance(samples, (int, np.integer)) or samples < 125:
        raise ValueError('Window must contain at least 125 samples')
    if not isinstance(start, (int, np.integer)) or start < 0:
        raise ValueError('Invalid window start')
    if x.ndim != 2 or start + samples > x.shape[-1]:
        raise ValueError('Window exceeds available signal; padding is forbidden')
    return preprocess(x[:, start:start + samples], fs=fs)


def competition_trials(root):
    result = []
    for file in sorted((root / 'data/processed/competition').glob('*_trials.npz')):
        with np.load(file, allow_pickle=False) as a:
            if a['channels'].tolist() != CHANNELS:
                raise ValueError('Competition electrode order mismatch')
            for x, y, tid in zip(a['X'], a['stim_id'], a['task_id']):
                result.append(Trial(x.copy(), int(y), 'competition', file.stem,
                                    f'{file.stem}/task{int(tid)}', int(tid) // 8))
    return result


def beta_trials(root, files, latency_seconds=0.14):
    result = []
    for relative in files:
        path = root / relative
        d = loadmat(path, simplify_cells=True)['data']
        info, eeg = d['suppl_info'], d['EEG']
        fs = int(info['srate'])
        if fs != 250 or eeg.shape[2:] != (4, 40):
            raise ValueError('Unsupported BETA sampling rate or layout')
        names = [str(ch).upper() for ch in info['chan'][:, 3]]
        ch = [names.index(name) for name in CHANNELS]
        freq = np.asarray(info['freqs'])
        # Exclude pre-stimulus 0.5 s and post-stimulus 0.5 s, then visual latency.
        start = round((0.5 + latency_seconds) * fs)
        end = eeg.shape[1] - round(0.5 * fs)
        for label, f in enumerate(FREQUENCIES):
            targets = np.flatnonzero(np.isclose(freq, f, atol=1e-4))
            if len(targets) != 1:
                raise ValueError(f'Cannot uniquely match frequency {f}')
            target = int(targets[0])
            for block in range(4):
                x = eeg[ch, start:end, block, target].copy()
                result.append(Trial(x, label, 'beta', path.stem,
                                    f'beta/{path.stem}/block{block}/target{target}', block))
    return result


def split_trials(trials, seed):
    """D2 locked as test; D1 trial split; BETA held out by subject."""
    rng = np.random.default_rng(seed)
    out = {key: [] for key in ['train', 'val', 'test']}
    comp = [t for t in trials if t.source == 'competition']
    d1 = [t for t in comp if t.subject == 'D1_trials']
    d2 = [t for t in comp if t.subject == 'D2_trials']
    if not d1 or not d2:
        raise ValueError('Require D1 and D2 for the declared evaluation protocol')
    for label in range(8):
        group = [t for t in d1 if t.y == label]
        rng.shuffle(group)
        out['val'].extend(group[:2])
        out['train'].extend(group[2:])
    out['test'].extend(d2)
    public = [t for t in trials if t.source == 'beta']
    subjects = sorted(set(t.subject for t in public))
    if subjects:
        if len(subjects) < 5:
            raise ValueError('At least five BETA subjects required for held-out evaluation')
        rng.shuffle(subjects)
        n = max(1, round(len(subjects) * 0.2))
        assignment = {s: 'test' if i < n else 'val' if i < 2*n else 'train'
                      for i, s in enumerate(subjects)}
        for t in public:
            out[assignment[t.subject]].append(t)
    ids = {key: set(t.uid for t in group) for key, group in out.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    return out


def make_windows(trials, seconds, stride_seconds=0.5, training=False, causal=False):
    """Split first; filter each ORIGINAL trial; then produce fixed-duration crops."""
    n, stride = round(seconds * 250), round(stride_seconds * 250)
    x, y, ids, sources = [], [], [], []
    for t in trials:
        filtered = preprocess(t.x, causal=causal)
        if filtered.shape[-1] < n:
            # No silent padding of shorter public stimuli with artificial signals.
            continue
        starts = list(range(0, filtered.shape[-1] - n + 1, stride)) if training else [0]
        for start in starts:
            x.append(filtered[:, start:start+n]); y.append(t.y)
            ids.append(t.uid); sources.append(t.source)
    if not x:
        raise ValueError('No trials long enough for requested window')
    return np.stack(x), np.array(y), ids, sources


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def benchmark_trials(root,files,latency_seconds=.14):
    """Wang2016 original MAT axes, electrode metadata, first eight integer-Hz targets.

    Target order is independently documented by MOABB's Wang2016 event map.
    This loader does not use its classifier implementation.
    """
    loc=root/'data/raw/public/benchmark/64-channels.loc'
    names=[line.split()[-1].upper() for line in loc.read_text().splitlines() if line.strip()]
    if len(names)!=64:raise ValueError('Expected complete 64-electrode location table')
    indices=[names.index(ch) for ch in CHANNELS]
    result=[]
    for relative in files:
        path=root/relative
        x=loadmat(path)['data']
        if x.shape!=(64,1500,40,6):raise ValueError('Unexpected Benchmark tensor shape')
        start=round((.5+latency_seconds)*250);end=round(5.5*250)
        for label in range(8):
            for block in range(6):
                sample=x[indices,start:end,label,block].copy()
                if not np.isfinite(sample).all():raise ValueError('Non-finite Benchmark signal')
                result.append(Trial(sample,label,'benchmark',path.stem,
                                    f'benchmark/{path.stem}/block{block}/target{label}',block))
    return result
