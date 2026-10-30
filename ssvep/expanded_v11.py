"""Single reference per class: analytic template CCA plus frozen FBCCA fusion."""
import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from .baselines import CCA, canonical_correlation
from .data import fingerprint, preprocess_window
from .expanded_v5 import ROOT, get_trials, write

CONFIG_PATH = ROOT / 'configs/expanded_v11.json'
OUT = ROOT / 'runs/v11-one-shot-support'
CANDIDATES = ('fbcca', 'template_cca', 'fusion_25', 'fusion_50')


def config_checked():
    c = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    old = json.loads((ROOT / 'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    used = set(sum(old.values(), []))
    for ver, fields in ((5, ('new_test_subjects',)), (6, ('additional_train', 'new_test')),
                        (7, ('new_test',)), (8, ('new_test',))):
        v = json.loads((ROOT / f'configs/expanded_v{ver}.json').read_text())
        used.update(s + '.mat' for field in fields for s in v[field])
    for ver, field in ((9, 'confirmation_subjects'), (10, 'confirmation_subjects')):
        v = json.loads((ROOT / f'configs/expanded_v{ver}.json').read_text(encoding='utf-8-sig'))
        used.update(s + '.mat' for s in v[field])
    remaining = [f'S{i}' for i in range(1, 71) if f'S{i}.mat' not in used]
    ordered = sorted(remaining, key=lambda s: hashlib.sha256(
        ('v11-one-shot-support:' + s + '.mat').encode()).hexdigest())
    assert c['confirmation_subjects'] == ordered and len(ordered) == 4
    assert c['candidates'] == list(CANDIDATES)
    v10 = json.loads((ROOT / 'configs/expanded_v10.json').read_text(encoding='utf-8-sig'))
    assert c['development_subjects'] == v10['development_subjects']
    assert not set(c['development_subjects']) & set(c['confirmation_subjects'])
    return c


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return (scores - scores.mean()) / max(float(scores.std()), 1e-8)


def score_query(query, supports, *, need_template=True):
    if sorted(supports) != list(range(8)):
        raise ValueError('Exactly one labeled reference per class is required')
    q = preprocess_window(query, query.shape[-1])
    base = CCA(filter_bank=True).score_one(q)
    if not need_template:
        return {'fbcca': base}
    template = []
    for label in range(8):
        raw = supports[label]
        length = min(query.shape[-1], raw.shape[-1])
        if length < 375:
            raise ValueError('Support and query need >=375 shared samples')
        # Crop each original record before independent filtering and scaling.
        qs = q[:, :length] if length == query.shape[-1] else preprocess_window(query, length)
        ss = preprocess_window(raw, length)
        template.append(canonical_correlation(qs.T, ss.T) ** 2)
    template = np.asarray(template)
    return {'fbcca': base, 'template_cca': template,
            'fusion_25': .75*normalize(base) + .25*normalize(template),
            'fusion_50': .50*normalize(base) + .50*normalize(template)}


def fold_scores(trials):
    by_subject = defaultdict(list)
    for trial in trials:
        if trial.source != 'beta':
            raise ValueError('Only public BETA is allowed for support-set simulation')
        by_subject[trial.subject].append(trial)
    rows = {key: [] for key in CANDIDATES}
    details = []
    for subject in sorted(by_subject):
        subject_trials = by_subject[subject]
        for support_block in range(4):
            support = [t for t in subject_trials if t.block == support_block]
            query = [t for t in subject_trials if t.block != support_block]
            if len(support) != 8 or len(query) != 24:
                raise ValueError('Expected 8 distinct support and 24 distinct query trials per subject/fold')
            supports = {t.y: t.x for t in support}
            if sorted(supports) != list(range(8)):
                raise ValueError('Reference classes are incomplete')
            if set(t.uid for t in support) & set(t.uid for t in query):
                raise ValueError('Support/query trial leakage')
            for trial in query:
                result = score_query(trial.x, supports)
                details.append((subject, support_block, trial.uid, trial.y))
                for key in CANDIDATES:
                    rows[key].append(result[key])
    return details, {key: np.asarray(value) for key,value in rows.items()}


def summarize(details, scores):
    subjects = sorted({row[0] for row in details})
    labels = np.asarray([row[3] for row in details])
    output = {}
    for key, values in scores.items():
        pred = values.argmax(axis=1)
        per_subject = {s: float(np.mean(pred[[row[0] == s for row in details]] ==
                                         labels[[row[0] == s for row in details]])) for s in subjects}
        output[key] = {'subject_macro_accuracy': float(np.mean(list(per_subject.values()))),
                       'overall_accuracy': float(np.mean(pred == labels)),
                       'subjects': per_subject, 'n_query_folds': len(details),
                       'n_original_query_trials_per_subject': 32,
                       'labels': labels.tolist(), 'predictions': pred.tolist()}
    return output


def dev():
    c = config_checked()
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'preregistered_config.json', c)
    write(OUT / 'development_source_fingerprints.json',
          {s+'.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat')
           for s in c['development_subjects']})
    details,scores = fold_scores(get_trials(c['development_subjects']))
    result = summarize(details,scores)
    write(OUT / 'development_results.json', result)
    winner = max(CANDIDATES, key=lambda key: result[key]['subject_macro_accuracy'])
    baseline = result['fbcca']['subject_macro_accuracy']
    selected = winner if result[winner]['subject_macro_accuracy']-baseline >= .02-1e-12 else 'fbcca'
    write(OUT / 'selection_frozen.json', {'complete': True, 'config_sha256': fingerprint(CONFIG_PATH),
          'selected': selected, 'confirmation_read': False})
    print('DEVELOPMENT', {k: round(v['subject_macro_accuracy'],4) for k,v in result.items()},
          'SELECTED', selected, flush=True)


def confirm():
    c = config_checked()
    frozen = json.loads((OUT / 'selection_frozen.json').read_text())
    if not frozen['complete'] or frozen['config_sha256'] != fingerprint(CONFIG_PATH):
        raise ValueError('Selection/config is not frozen')
    if (OUT / 'confirmation_results.json').exists():
        raise FileExistsError('Confirmation already recorded')
    write(OUT / 'confirmation_source_fingerprints.json',
          {s+'.mat': fingerprint(ROOT / f'data/raw/public/beta/{s}.mat')
           for s in c['confirmation_subjects']})
    details,scores = fold_scores(get_trials(c['confirmation_subjects']))
    selected = frozen['selected']
    result = summarize(details,{key: scores[key] for key in dict.fromkeys(('fbcca',selected))})
    base = result['fbcca']['subject_macro_accuracy']
    selected_acc = result[selected]['subject_macro_accuracy']
    nonnegative = sum(result[selected]['subjects'][s] >= result['fbcca']['subjects'][s]
                      for s in c['confirmation_subjects'])
    promoted = selected != 'fbcca' and selected_acc-base >= .02-1e-12 and nonnegative >= 3
    write(OUT / 'confirmation_results.json', {'results':result,'selected':selected,
          'gain':selected_acc-base,'nonnegative_subjects':nonnegative,'promoted':promoted,
          'default_algorithm':selected if promoted else 'fbcca',
          'D1_D2_used':False})
    write(OUT / 'completed.json', {'complete':True,'promoted':promoted,
          'default_algorithm':selected if promoted else 'fbcca','D1_D2_used':False})
    print('CONFIRM',selected,'gain',round(selected_acc-base,4),'nonnegative',nonnegative,
          'promoted',promoted,flush=True)


def verify_mapping():
    """Recompute the recorded support-only mapping diagnostic after confirmation."""
    c = config_checked()
    completed = json.loads((OUT / 'completed.json').read_text())
    if not completed['complete']:
        raise ValueError('Confirmation must finish first')
    model = CCA(filter_bank=True)
    rows = []
    for subject in c['confirmation_subjects']:
        trials = get_trials([subject])
        for block in range(4):
            support = sorted((t for t in trials if t.block == block), key=lambda t: t.y)
            if len(support) != 8 or [t.y for t in support] != list(range(8)):
                raise ValueError('Expected one independent support per class')
            scores = np.stack([model.score_one(preprocess_window(t.x, t.x.shape[-1]))
                               for t in support])
            labels, frequencies = linear_sum_assignment(-scores)
            mapping = {int(a): int(b) for a, b in zip(labels, frequencies)}
            rows.append({'subject': subject, 'support_block': block,
                         'exact_mapping': mapping == {i: i for i in range(8)},
                         'mapping': mapping,
                         'support_prediction_accuracy':
                             float(np.mean(scores.argmax(axis=1) == np.arange(8)))})
    recorded = json.loads((OUT / 'mapping_check.json').read_text())
    if json.loads(json.dumps(rows)) != recorded:
        raise ValueError('Recomputed mapping diagnostic differs from the record')
    print('MAPPING',sum(row['exact_mapping'] for row in rows),'/',len(rows),'exact')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('dev','confirm','verify-mapping'))
    stage=p.parse_args().stage
    with threadpool_limits(limits=1):
        if stage == 'dev': dev()
        elif stage == 'confirm': confirm()
        else: verify_mapping()


if __name__=='__main__':
    main()
