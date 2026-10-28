"""Frozen BETA subject-scale experiment: fixed FBCCA and direct EEGNet."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .baselines import CCA
from .data import beta_trials, competition_trials, fingerprint, preprocess_window
from .models import EEGNetStudent
from .policy import assert_allowed_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v5-expanded-eegnet'
CONFIG_PATH = ROOT / 'configs/expanded_v5.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    roles = json.loads((ROOT/'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    used = set(sum(roles.values(), []))
    pool = [f'S{i}.mat' for i in range(1, 71) if f'S{i}.mat' not in used]
    chosen = sorted(pool, key=lambda s: hashlib.sha256(('expanded-v1:'+s).encode()).hexdigest())[:5]
    assert chosen == [s+'.mat' for s in config['new_test_subjects']]
    assert set(config['training_groups']['six']) <= set(config['training_groups']['fourteen'])
    assert not (set(config['training_groups']['fourteen']) & set(config['development_subjects']))
    assert not (set(config['training_groups']['fourteen']+config['development_subjects']) & set(config['new_test_subjects']))
    return config


def get_trials(subjects):
    files = [f'data/raw/public/beta/{s}.mat' for s in subjects]
    for file in files:
        if not (ROOT/file).exists():
            raise FileNotFoundError(ROOT/file)
    return beta_trials(ROOT, files)


def arrays(trials, samples, stride, training):
    data, labels = [], []
    for trial in trials:
        starts = range(0, trial.x.shape[-1]-samples+1, stride) if training else (0,)
        for start in starts:
            data.append(preprocess_window(trial.x, samples, start))
            labels.append(trial.y)
    return np.stack(data), np.asarray(labels, dtype=np.int64)


def report(scores, trials):
    prediction = np.asarray(scores).argmax(axis=1)
    labels = np.asarray([t.y for t in trials])
    subjects = sorted({t.subject for t in trials})
    by = {s: float(np.mean(prediction[[t.subject == s for t in trials]] ==
                           labels[[t.subject == s for t in trials]])) for s in subjects}
    return {'subject_macro_accuracy': float(np.mean(list(by.values()))),
            'subjects': by, 'n_trials': len(trials), 'labels': labels.tolist(),
            'prediction': prediction.tolist()}


def score_models(models, x):
    tensor = torch.from_numpy(x)
    with torch.no_grad():
        return {key: model(tensor).numpy() for key, model in models.items()}


def run_train():
    config = load_config()
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'preregistered_config.json', config)
    audit = {}
    all_subjects = config['training_groups']['fourteen'] + config['development_subjects']
    old = json.loads((ROOT/'runs/v2-beta-subject-isolated/source_fingerprints.json').read_text())['public_pool']
    for subject in all_subjects:
        relative = f'data/raw/public/beta/{subject}.mat'
        audit[relative] = fingerprint(ROOT/relative)
        assert audit[relative] == old[relative.replace('/', '\\')]
    write(OUT/'training_source_fingerprints.json', audit)
    dev_trials = get_trials(config['development_subjects'])
    dx, dy = arrays(dev_trials, config['samples'], config['stride'], False)
    fbcca = CCA(filter_bank=True)
    fb = fbcca.scores(dx)
    results = {'fbcca': report(fb, dev_trials)}
    torch.set_num_threads(1)
    for group, subjects in config['training_groups'].items():
        trials = get_trials(subjects)
        assert_allowed_sources([t.source for t in trials])
        tx, ty = arrays(trials, config['samples'], config['stride'], True)
        x, y, dev = torch.from_numpy(tx), torch.from_numpy(ty), torch.from_numpy(dx)
        for seed in config['seeds']:
            torch.manual_seed(seed)
            model = EEGNetStudent()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                          weight_decay=config['weight_decay'])
            rng = np.random.default_rng(seed)
            best = -1.0
            best_state = None
            selected_epoch = None
            history = []
            started = time.perf_counter()
            for epoch in range(config['epochs']):
                model.train()
                indices = rng.integers(len(tx), size=config['samples_per_epoch'])
                losses = []
                for batch in indices.reshape(-1, config['batch']):
                    loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    scores = model(dev).numpy()
                accuracy = report(scores, dev_trials)['subject_macro_accuracy']
                history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)),
                                'development_subject_macro_accuracy': accuracy})
                if accuracy > best:
                    best = accuracy
                    selected_epoch = epoch+1
                    best_state = copy.deepcopy(model.state_dict())
                if (epoch+1) % 10 == 0:
                    print('TRAIN', group, seed, epoch+1, accuracy, flush=True)
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                scores = model(dev).numpy()
            key = f'eegnet_{group}_{seed}'
            results[key] = report(scores, dev_trials)
            results[key].update(selected_epoch=selected_epoch, train_original_trials=len(trials),
                                train_windows=len(tx), training_seconds=time.perf_counter()-started)
            torch.save({'state_dict': best_state, 'group': group, 'seed': seed,
                        'config_sha256': fingerprint(CONFIG_PATH)}, OUT/(key+'.pt'))
            write(OUT/(key+'_history.json'), history)
            write(OUT/'development_partial.json', results)
    write(OUT/'development_results.json', results)
    write(OUT/'models_frozen.json', {'complete': True, 'config_sha256': fingerprint(CONFIG_PATH),
         'new_test_read': False, 'competition_read': False,
         'model_keys': [f'eegnet_{g}_{s}' for g in config['training_groups'] for s in config['seeds']]})
    print('MODELS_FROZEN', flush=True)


def run_confirm():
    config = load_config()
    frozen = json.loads((OUT/'models_frozen.json').read_text())
    assert frozen['complete'] and frozen['config_sha256'] == fingerprint(CONFIG_PATH)
    if (OUT/'confirmation_results.json').exists():
        raise FileExistsError('Confirmation already run; do not overwrite')
    models = {}
    for key in frozen['model_keys']:
        saved = torch.load(OUT/(key+'.pt'), map_location='cpu', weights_only=False)
        assert saved['config_sha256'] == frozen['config_sha256']
        model = EEGNetStudent()
        model.load_state_dict(saved['state_dict'])
        models[key] = model.eval()
    torch.set_num_threads(1)
    groups = {'new_beta_test': get_trials(config['new_test_subjects'])}
    # D1/D2 are historical examples, evaluated only after all models freeze.
    from scripts.prepare_competition import main as prepare_competition
    prepare_competition()
    for t in competition_trials(ROOT):
        groups.setdefault(t.subject, []).append(t)
    results = {}
    hashes = {}
    for name, trials in groups.items():
        x, labels = arrays(trials, config['samples'], config['stride'], False)
        assert labels.tolist() == [t.y for t in trials]
        prediction = {'fbcca': report(CCA(filter_bank=True).scores(x), trials)}
        prediction.update({key: report(score, trials) for key, score in score_models(models, x).items()})
        results[name] = prediction
        print('CONFIRM', name, {k: round(v['subject_macro_accuracy'], 4) for k,v in prediction.items()}, flush=True)
    for s in config['new_test_subjects']:
        relative = f'data/raw/public/beta/{s}.mat'
        hashes[relative] = fingerprint(ROOT/relative)
    for name in ('D1', 'D2'):
        relative = f'data/raw/competition/{name}.csv'
        hashes[relative] = fingerprint(ROOT/relative)
    write(OUT/'confirmation_source_fingerprints.json', hashes)
    write(OUT/'confirmation_results.json', results)
    write(OUT/'completed.json', {'complete': True, 'test_subjects': config['new_test_subjects'],
                                 'D1_D2_evaluation_only': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'confirm'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == 'train':
            run_train()
        else:
            run_confirm()


if __name__ == '__main__':
    main()
