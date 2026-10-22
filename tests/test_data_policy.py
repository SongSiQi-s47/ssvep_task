import unittest
from ssvep.policy import assert_allowed_sources
from ssvep.policy import public_subject_split
from ssvep.data import Trial
import numpy as np


class DataPolicy(unittest.TestCase):
    def test_reserved_sources_cannot_train_or_calibrate(self):
        policy={'training_sources':['beta'],'calibration_sources':['beta']}
        for stage in ['training','calibration']:
            with self.assertRaises(ValueError):assert_allowed_sources(['competition'],stage,policy)
            assert_allowed_sources(['beta'],stage,policy)

    def test_unconfirmed_scope_blocks_all_training(self):
        policy={'training_sources':[],'calibration_sources':[]}
        with self.assertRaises(ValueError):assert_allowed_sources(['beta'],'training',policy)

    def test_public_split_has_no_subject_overlap_and_rejects_competition(self):
        trials=[Trial(np.zeros((6,250)),i%8,'benchmark',f'S{i}',f'trial{i}',0) for i in range(6)]
        split=public_subject_split(trials,2026)
        subjects={k:{t.subject for t in v} for k,v in split.items()}
        self.assertFalse(subjects['train']&subjects['dev'] or subjects['train']&subjects['holdout'] or subjects['dev']&subjects['holdout'])
        self.assertEqual(sorted(map(len,subjects.values())),[1,1,4])
        trials.append(Trial(np.zeros((6,250)),0,'competition','D1','comp0',0))
        with self.assertRaises(ValueError):public_subject_split(trials,2026)


if __name__=='__main__':unittest.main()
