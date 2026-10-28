"""Duration-aware analytic FBCCA comparison with subject-isolated BETA confirmation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .baselines import CCA
from .data import competition_trials, fingerprint, preprocess_window
from .expanded_v5 import ROOT, get_trials, report, write

OUT = ROOT / 'runs/v9-duration-fbcca'
CONFIG_PATH = ROOT / 'configs/expanded_v9.json'
CANDIDATES = ('first_1p5', 'full', 'last_1p5', 'first_last_mean')


def config_checked():
    c = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    old = json.loads((ROOT / 'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    v5 = json.loads((ROOT / 'configs/expanded_v5.json').read_text())
    v6 = json.loads((ROOT / 'configs/expanded_v6.json').read_text())
    v7 = json.loads((ROOT / 'configs/expanded_v7.json').read_text())
    v8 = json.loads((ROOT / 'configs/expanded_v8.json').read_text())
    used = set(sum(old.values(), [])) | {s + '.mat' for s in
        v5['new_test_subjects'] + v6['additional_train'] + v6['new_test'] +
        v7['new_test'] + v8['new_test']}
    pool = [f'S{i}' for i in range(1, 71) if f'S{i}.mat' not in used]
    ordered = sorted(pool, key=lambda s: hashlib.sha256(
        ('v9-duration-fbcca:' + s + '.mat').encode()).hexdigest())
    assert c['confirmation_subjects'] == ordered[:5]
    assert c['development_subjects'] == v8['dev']
    assert c['candidates'] == list(CANDIDATES) and c['first_or_last_samples'] == 375
    assert not set(c['development_subjects']) & set(c['confirmation_subjects'])
    return c


def score_signals(signals, candidates=CANDIDATES):
    """Score unlabeled six-channel records of possibly different lengths."""
    if not candidates or not set(candidates) <= set(CANDIDATES):
        raise ValueError('Unknown or empty candidate set')
    fbcca = CCA(filter_bank=True)
    result = {key: [] for key in candidates}
    n = 375
    for raw in signals:
        length = raw.shape[-1]
        if length < n:
            raise ValueError(f'Insufficient samples: {length}')
        scores = {}
        if 'first_1p5' in candidates or 'first_last_mean' in candidates or (
                'full' in candidates and length == n):
            scores['first_1p5'] = fbcca.score_one(preprocess_window(raw, n))
        if 'full' in candidates:
            scores['full'] = (scores['first_1p5'] if length == n else
                              fbcca.score_one(preprocess_window(raw, length)))
        if 'last_1p5' in candidates or 'first_last_mean' in candidates:
            scores['last_1p5'] = fbcca.score_one(preprocess_window(raw, n, length - n))
        if 'first_last_mean' in candidates:
            scores['first_last_mean'] = (scores['first_1p5'] + scores['last_1p5']) / 2
        for key in candidates:
            result[key].append(scores[key])
    return {key: np.asarray(values) for key, values in result.items()}


def score(trials, candidates=CANDIDATES):
    return score_signals((trial.x for trial in trials), candidates)


def summarize(scores, trials):
    return {key: report(values, trials) for key, values in scores.items()}


def dev():
    c = config_checked()
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'preregistered_config.json', c)
    sources = {s + '.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat')
               for s in c['development_subjects']}
    write(OUT / 'development_source_fingerprints.json', sources)
    trials = get_trials(c['development_subjects'])
    results = summarize(score(trials), trials)
    write(OUT / 'development_results.json', results)
    baseline = results['first_1p5']['subject_macro_accuracy']
    winner = max(CANDIDATES, key=lambda key: results[key]['subject_macro_accuracy'])
    selected = winner if results[winner]['subject_macro_accuracy'] - baseline >= .02 - 1e-12 else 'first_1p5'
    frozen = {'complete': True, 'config_sha256': fingerprint(CONFIG_PATH),
              'development_subjects': c['development_subjects'],
              'confirmation_subjects': c['confirmation_subjects'],
              'selection': selected, 'confirmation_read': False,
              'D1_D2_read': False}
    write(OUT / 'selection_frozen.json', frozen)
    print('DEVELOPMENT', {k: round(v['subject_macro_accuracy'], 4) for k,v in results.items()},
          'SELECTED', selected, flush=True)


def _export(path, trials, predictions):
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['taskID', 'predicted_stimID'])
        for trial, label in zip(trials, predictions):
            w.writerow([int(trial.uid.rsplit('task', 1)[1]), int(label)])


def confirm():
    c = config_checked()
    frozen = json.loads((OUT / 'selection_frozen.json').read_text(encoding='utf-8'))
    if not frozen['complete'] or frozen['config_sha256'] != fingerprint(CONFIG_PATH):
        raise ValueError('No valid frozen selection/config')
    if (OUT / 'confirmation_results.json').exists():
        raise FileExistsError('Confirmation already recorded')
    subjects = c['confirmation_subjects']
    sources = {s + '.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat') for s in subjects}
    trials = get_trials(subjects)
    beta = summarize(score(trials), trials)
    selected = frozen['selection']
    base = beta['first_1p5']['subject_macro_accuracy']
    chosen = beta[selected]['subject_macro_accuracy']
    nonnegative = sum(beta[selected]['subjects'][s] >= beta['first_1p5']['subjects'][s]
                      for s in subjects)
    promoted = (selected != 'first_1p5' and chosen - base >= .02 - 1e-12
                and nonnegative >= 4)
    deployed = selected if promoted else 'first_1p5'
    # Read historical competition examples only after selection and promotion are fixed.
    comp = competition_trials(ROOT)
    if len(comp) != 96 or {t.subject for t in comp} != {'D1_trials', 'D2_trials'}:
        raise ValueError('Expected two original 48-trial competition examples')
    comp_scores = score(comp, ('first_1p5', selected) if selected != 'first_1p5' else ('first_1p5',))
    verification = {key: summarize({name: values[[t.subject == key for t in comp]]
                                   for name, values in comp_scores.items()},
                                   [t for t in comp if t.subject == key])
                    for key in ('D1_trials', 'D2_trials')}
    result = {'beta_confirmation': beta, 'selected_on_development': selected,
              'nonnegative_subjects': nonnegative, 'promoted': promoted,
              'default_algorithm': deployed, 'competition_verification': verification,
              'note': 'D1/D2 historically exposed, descriptive verification only'}
    write(OUT / 'confirmation_source_fingerprints.json', sources)
    write(OUT / 'verification_source_fingerprints.json',
          {name + '.csv': fingerprint(ROOT / f'data/raw/competition/{name}.csv')
           for name in ('D1', 'D2')})
    write(OUT / 'confirmation_results.json', result)
    folder = ROOT / 'output' / 'sample_results'
    folder.mkdir(parents=True, exist_ok=True)
    for key in ('D1_trials', 'D2_trials'):
        subset = [i for i,t in enumerate(comp) if t.subject == key]
        _export(folder / f'{key[:2]}_v9_{deployed}.csv', [comp[i] for i in subset],
                comp_scores[deployed][subset].argmax(axis=1))
    write(OUT / 'completed.json', {'complete': True, 'default_algorithm': deployed,
                                  'D1_D2_evaluation_only': True})
    print('CONFIRM', selected, 'gain', round(chosen - base, 4),
          'nonnegative', nonnegative, 'promoted', promoted, flush=True)
    for key, values in verification.items():
        print(key, {name: round(item['subject_macro_accuracy'], 4)
                    for name,item in values.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['dev', 'confirm'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        dev() if args.stage == 'dev' else confirm()


if __name__ == '__main__':
    main()
