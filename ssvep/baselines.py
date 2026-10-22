"""CCA and filter-bank CCA, implemented from matrix decompositions."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def basis(x):
    x = x - x.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    if not len(singular) or singular[0] < 1e-12:
        return u[:, :0]
    return u[:, singular > singular[0] * 1e-8]


def canonical_correlation(x, y):
    a, b = basis(x), basis(y)
    if not a.shape[1] or not b.shape[1]: return 0.
    return float(np.clip(np.linalg.svd(a.T @ b, compute_uv=False)[0], 0, 1))


class CCA:
    def __init__(self, frequencies=range(8,16), fs=250, harmonics=3, filter_bank=False):
        self.frequencies, self.fs = list(frequencies), fs
        self.harmonics, self.filter_bank = harmonics, filter_bank

    def subband_scores(self, x):
        time = np.arange(x.shape[-1]) / self.fs
        refs = [np.stack([fun(2*np.pi*h*f*time)
                          for h in range(1,self.harmonics+1)
                          for fun in (np.sin,np.cos)], axis=1) for f in self.frequencies]
        bands = [6,14,22] if self.filter_bank else [None]
        scores = []
        for i, low in enumerate(bands):
            z = x if low is None else sosfiltfilt(
                butter(4,[low,90],fs=self.fs,btype='bandpass',output='sos'),x,axis=-1)
            scores.append([canonical_correlation(z.T,y)**2 for y in refs])
        return np.asarray(scores).T

    def score_one(self, x):
        scores = self.subband_scores(x)
        weights = (np.arange(1, scores.shape[1]+1)**-1.25 + .25
                   if self.filter_bank else np.ones(1))
        return scores @ weights

    def scores(self, x): return np.stack([self.score_one(a) for a in x])


def reference_template_scores(support_x, support_y, query_x):
    """One-example-per-class spectral template calibration, no query labels."""
    if len(support_y) != 8 or sorted(support_y.tolist()) != list(range(8)):
        raise ValueError('Calibration requires exactly one labeled original trial per class')
    def features(x):
        # Hann taper reduces edge leakage; retaining frequency positions is essential.
        fft = np.fft.rfft(x * np.hanning(x.shape[-1]),n=1000,axis=-1)
        power = np.log1p(np.abs(fft[:,:,24:361])**2).reshape(len(x),-1)
        power -= power.mean(axis=1,keepdims=True)
        power /= np.maximum(np.linalg.norm(power,axis=1,keepdims=True),1e-8)
        return power
    support = features(support_x)[np.argsort(support_y)]
    return features(query_x) @ support.T
