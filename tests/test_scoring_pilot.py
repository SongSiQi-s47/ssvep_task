import unittest
import numpy as np
import torch
from scipy.signal import butter,sosfiltfilt
from ssvep.baselines import CCA,canonical_correlation
from ssvep.scoring_pilot import SharedScorer,features


class ScoringPilot(unittest.TestCase):
    def test_fbcca_refactor_preserves_original_equation(self):
        x=np.random.default_rng(42).normal(size=(6,375))
        t=np.arange(375)/250
        expected=np.zeros(8)
        for i,low in enumerate([6,14,22]):
            z=sosfiltfilt(butter(4,[low,90],fs=250,btype='bandpass',output='sos'),x,axis=-1)
            for j,f in enumerate(range(8,16)):
                ref=np.stack([fun(2*np.pi*h*f*t) for h in [1,2,3] for fun in [np.sin,np.cos]],axis=1)
                expected[j]+=((i+1)**-1.25+.25)*canonical_correlation(z.T,ref)**2
        np.testing.assert_allclose(CCA(filter_bank=True).score_one(x),expected,atol=1e-12)

    def test_shared_scorer_candidate_permutation_and_gradient(self):
        model=SharedScorer(np.zeros(6),np.ones(6))
        x=torch.randn(4,8,6)
        permutation=torch.tensor([7,2,1,0,3,4,5,6])
        torch.testing.assert_close(model(x[:,permutation]),model(x)[:,permutation])
        torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1,2,3])).backward()
        self.assertTrue(torch.isfinite(model.linear.weight.grad).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()),6)

    def test_feature_zero_signal_finite(self):
        value=features(np.zeros((6,375),dtype='float32'))
        self.assertEqual(value.shape,(8,6))
        self.assertTrue(np.isfinite(value).all())
