import unittest
import numpy as np

from ssvep.expanded_v11 import score_query, fold_scores
from ssvep.data import Trial


class OneShotSupportTest(unittest.TestCase):
    def signal(self, f, shift=0):
        t = np.arange(465)/250
        return np.stack([np.sin(2*np.pi*f*t + shift + k/8) for k in range(6)])

    def test_exact_one_reference_per_class_and_order_independence(self):
        supports={i:self.signal(8+i) for i in range(8)}
        query=self.signal(11,.2)
        a=score_query(query,supports)
        b=score_query(query,dict(reversed(list(supports.items()))))
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        with self.assertRaisesRegex(ValueError,'Exactly one'):
            score_query(query,{0:supports[0]})

    def test_non_public_competition_trials_rejected(self):
        t=Trial(self.signal(8),0,'competition','D1_trials','D1/task0',0)
        with self.assertRaisesRegex(ValueError,'Only public BETA'):
            fold_scores([t])


if __name__=='__main__': unittest.main()
