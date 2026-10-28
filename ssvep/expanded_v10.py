"""Frozen full-record FBCCA spatial and harmonic variants on external BETA subjects."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .baselines import CCA
from .data import competition_trials, fingerprint, preprocess_window
from .expanded_v5 import ROOT, get_trials, report, write

CONFIG_PATH = ROOT / 'configs/expanded_v10.json'
OUT = ROOT / 'runs/v10-spatial-harmonic'
CANDIDATES = ('fbcca3_all', 'fbcca3_posterior', 'fbcca5_all',
              'fbcca5_posterior', 'fbcca2_all', 'fbcca3_fusion')


def checked_config():
    c = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    v9 = json.loads((ROOT / 'configs/expanded_v9.json').read_text(encoding='utf-8-sig'))
    old = json.loads((ROOT / 'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    used = set(sum(old.values(), []))
    for version, fields in ((5, ('new_test_subjects',)), (6, ('additional_train', 'new_test')),
                            (7, ('new_test',)), (8, ('new_test',))):
        prior = json.loads((ROOT / f'configs/expanded_v{version}.json').read_text())
        used.update(s + '.mat' for field in fields for s in prior[field])
    used.update(s + '.mat' for s in v9['confirmation_subjects'])
    unused = [f'S{i}' for i in range(1, 71) if f'S{i}.mat' not in used]
    ordered = sorted(unused, key=lambda s: hashlib.sha256(
        ('v10-spatial-harmonic:' + s + '.mat').encode()).hexdigest())
    assert c['confirmation_subjects'] == ordered[:5]
    assert c['development_subjects'] == v9['development_subjects'] + v9['confirmation_subjects']
    assert c['candidates'] == list(CANDIDATES)
    assert not set(c['development_subjects']) & set(c['confirmation_subjects'])
    return c


def _zscore(values):
    return (values - values.mean()) / max(values.std(), 1e-8)


def score_signals(signals, candidates=CANDIDATES):
    if not candidates or not set(candidates) <= set(CANDIDATES):
        raise ValueError('Unknown or empty candidate set')
    models = {2: CCA(harmonics=2, filter_bank=True),
              3: CCA(harmonics=3, filter_bank=True),
              5: CCA(harmonics=5, filter_bank=True)}
    output = {key: [] for key in candidates}
    for raw in signals:
        x = preprocess_window(raw, raw.shape[-1])
        values = {}
        need_all3 = 'fbcca3_all' in candidates or 'fbcca3_fusion' in candidates
        need_po3 = 'fbcca3_posterior' in candidates or 'fbcca3_fusion' in candidates
        if need_all3: values['fbcca3_all'] = models[3].score_one(x)
        if need_po3: values['fbcca3_posterior'] = models[3].score_one(x[3:])
        if 'fbcca5_all' in candidates: values['fbcca5_all'] = models[5].score_one(x)
        if 'fbcca5_posterior' in candidates: values['fbcca5_posterior'] = models[5].score_one(x[3:])
        if 'fbcca2_all' in candidates: values['fbcca2_all'] = models[2].score_one(x)
        if 'fbcca3_fusion' in candidates:
            values['fbcca3_fusion'] = (_zscore(values['fbcca3_all']) +
                                        _zscore(values['fbcca3_posterior'])) / 2
        for key in candidates: output[key].append(values[key])
    return {key: np.asarray(rows) for key, rows in output.items()}


def evaluate(trials, candidates=CANDIDATES):
    scores = score_signals((t.x for t in trials), candidates)
    return {key: report(values, trials) for key, values in scores.items()}


def dev():
    c = checked_config()
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'preregistered_config.json', c)
    write(OUT / 'development_source_fingerprints.json',
          {s + '.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat')
           for s in c['development_subjects']})
    results = evaluate(get_trials(c['development_subjects']))
    write(OUT / 'development_results.json', results)
    winner = max(CANDIDATES, key=lambda key: results[key]['subject_macro_accuracy'])
    baseline = results['fbcca3_all']['subject_macro_accuracy']
    selected = winner if results[winner]['subject_macro_accuracy'] - baseline >= .02 - 1e-12 else 'fbcca3_all'
    write(OUT / 'selection_frozen.json', {'complete': True, 'config_sha256': fingerprint(CONFIG_PATH),
          'selected': selected, 'confirmation_read': False, 'D1_D2_read': False})
    print('DEVELOPMENT', {key: round(val['subject_macro_accuracy'], 4)
                          for key,val in results.items()}, 'SELECTED', selected, flush=True)


def confirm():
    c = checked_config()
    frozen = json.loads((OUT / 'selection_frozen.json').read_text())
    if not frozen['complete'] or frozen['config_sha256'] != fingerprint(CONFIG_PATH):
        raise ValueError('Config/selection is not frozen')
    if (OUT / 'confirmation_results.json').exists():
        raise FileExistsError('Confirmation already recorded')
    write(OUT / 'confirmation_source_fingerprints.json',
          {s + '.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat')
           for s in c['confirmation_subjects']})
    trials = get_trials(c['confirmation_subjects'])
    selected = frozen['selected']
    candidates = ('fbcca3_all', selected) if selected != 'fbcca3_all' else ('fbcca3_all',)
    beta = evaluate(trials, candidates)
    base = beta['fbcca3_all']['subject_macro_accuracy']
    gain = beta[selected]['subject_macro_accuracy'] - base
    nonnegative = sum(beta[selected]['subjects'][s] >= beta['fbcca3_all']['subjects'][s]
                      for s in c['confirmation_subjects'])
    promoted = selected != 'fbcca3_all' and gain >= .02 - 1e-12 and nonnegative >= 4
    # Historical examples are read only after the confirmation promotion decision.
    comp = competition_trials(ROOT)
    verification = {d: evaluate([t for t in comp if t.subject == d], candidates)
                    for d in ('D1_trials', 'D2_trials')}
    write(OUT / 'confirmation_results.json', {'beta': beta, 'selected': selected,
          'gain': gain, 'nonnegative_subjects': nonnegative, 'promoted': promoted,
          'default_algorithm': selected if promoted else 'fbcca3_all',
          'competition_verification': verification,
          'D1_D2_evaluation_only': True})
    write(OUT / 'completed.json', {'complete': True, 'promoted': promoted,
          'default_algorithm': selected if promoted else 'fbcca3_all'})
    print('CONFIRM', selected, 'gain', round(gain, 4), 'nonnegative', nonnegative,
          'promoted', promoted, flush=True)
    for d,r in verification.items():
        print(d, {k: round(v['subject_macro_accuracy'], 4) for k,v in r.items()}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('dev', 'confirm'))
    stage = p.parse_args().stage
    with threadpool_limits(limits=1):
        dev() if stage == 'dev' else confirm()


if __name__ == '__main__':
    main()
