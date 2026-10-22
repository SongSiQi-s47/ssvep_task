"""Enforce dataset purpose independently of an experiment's model settings."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load_policy():
    return json.loads((ROOT/'configs/data_usage_policy.json').read_text(encoding='utf-8'))


def assert_allowed_sources(sources,stage='training',policy=None):
    policy=load_policy() if policy is None else policy
    field={'training':'training_sources','calibration':'calibration_sources'}[stage]
    allowed=set(policy[field])
    rejected=set(sources)-allowed
    if rejected:
        raise ValueError(f'Dataset policy forbids {stage} on {sorted(rejected)}; '
                         'evaluation-only data must not update model weights or calibration state')


def public_subject_split(trials,seed):
    """Development partition uses allowed public training data only."""
    import numpy as np
    if any(t.source=='competition' for t in trials):
        raise ValueError('Competition records are reserved for evaluation')
    assert_allowed_sources([t.source for t in trials])
    groups=sorted({(t.source,t.subject) for t in trials})
    if len(groups)<5:raise ValueError('Need at least five public training subjects')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    n=max(1,round(len(groups)*.2))
    dev=set(groups[:n]);holdout=set(groups[n:2*n]);train=set(groups[2*n:])
    assignment={group:stage for stage,subset in [('train',train),('dev',dev),('holdout',holdout)] for group in subset}
    return {stage:[t for t in trials if assignment[(t.source,t.subject)]==stage]
            for stage in ['train','dev','holdout']}
