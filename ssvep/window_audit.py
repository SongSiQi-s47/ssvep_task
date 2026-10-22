"""Frozen, development-only FBCCA observation-window audit. No model fitting."""
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from .data import beta_trials, preprocess, preprocess_window, fingerprint
from .baselines import CCA

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    out = ROOT/'runs/v4-window-audit'
    out.mkdir(exist_ok=False)
    subjects = ['S6', 'S20', 'S51']
    config = {'subjects': subjects, 'samples': [250, 375], 'fs': 250,
              'latency_seconds': .14, 'start': 0, 'algorithm': 'unchanged FBCCA',
              'selection': 'Highest bounded-window macro subject accuracy; tie selects 250',
              'old_scope': 'diagnostic only, never eligible for selection',
              'mode': 'offline within observed window, not streaming',
              'confirmation_subjects_read': False, 'holdout_read': False,
              'competition_read': False, 'hardware': platform.processor()}
    save(out/'preregistered_config.json', config)
    roles = json.loads((ROOT/'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    hashes = json.loads((ROOT/'runs/v2-beta-subject-isolated/source_fingerprints.json').read_text())['public_pool']
    files = []
    audit = {}
    for s in subjects:
        assert s+'.mat' in roles['dev']
        relative = str(Path('data/raw/public/beta')/(s+'.mat'))
        actual = fingerprint(ROOT/relative)
        assert actual == hashes[relative], relative
        files.append(relative)
        audit[relative] = actual
    save(out/'read_sources.json', audit)
    trials = beta_trials(ROOT, files)
    assert len(trials) == 96 and all(t.subject in subjects for t in trials)
    alg = CCA(filter_bank=True)
    alg.score_one(preprocess_window(trials[0].x, 250))  # warm-up, excluded timing
    results = {}
    for n in config['samples']:
        for mode in ['bounded', 'legacy_full_trial']:
            rows = []
            for trial in trials:
                begin = perf_counter()
                x = (preprocess_window(trial.x, n) if mode == 'bounded'
                     else preprocess(trial.x)[:, :n])
                middle = perf_counter()
                scores = alg.score_one(x)
                end = perf_counter()
                assert np.isfinite(scores).all()
                rows.append({'uid': trial.uid, 'subject': trial.subject,
                             'label': trial.y, 'prediction': int(scores.argmax()),
                             'raw_available_samples': trial.x.shape[-1],
                             'scores': scores.tolist(),
                             'preprocess_ms': (middle-begin)*1000,
                             'inference_ms': (end-middle)*1000,
                             'total_compute_ms': (end-begin)*1000})
            by_subject = {s: float(np.mean([r['label'] == r['prediction']
                                          for r in rows if r['subject'] == s])) for s in subjects}
            confusion = np.zeros((8, 8), dtype=int)
            for r in rows: confusion[r['label'], r['prediction']] += 1
            key = f'{mode}_{n}'
            results[key] = {'subject_accuracy': by_subject,
                            'macro_accuracy': float(np.mean(list(by_subject.values()))),
                            'confusion': confusion.tolist(), 'trials': 96,
                            'timing': {k: {'median': float(np.median([r[k] for r in rows])),
                                           'p95': float(np.percentile([r[k] for r in rows], 95))}
                                       for k in ['preprocess_ms', 'inference_ms', 'total_compute_ms']}}
            save(out/(key+'_predictions.json'), rows)
    chosen = max(config['samples'], key=lambda n: (results[f'bounded_{n}']['macro_accuracy'], -n))
    save(out/'results.json', results)
    save(out/'frozen_window_decision.json', {'samples': chosen, 'seconds': chosen/250,
         'basis': config['selection'], 'development_decision_only': True,
         'model_promotion': False, 'not_comparable_to_old_scope_accuracy': True})
    save(out/'completed.json', {'complete': True, 'predictions': 384, 'trained_models': 0})
    print(json.dumps({'results': results, 'selected_samples': chosen}, indent=2))


if __name__ == '__main__':
    main()
