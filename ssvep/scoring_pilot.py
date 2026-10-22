"""One bounded candidate: shared linear subband/harmonic scorer vs EEGNet/FBCCA."""
import copy
import json
import time
import platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from .data import beta_trials, preprocess_window, fingerprint
from .baselines import CCA
from .models import EEGNetStudent
from .policy import assert_allowed_sources

ROOT = Path(__file__).resolve().parents[1]


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def features(x):
    sub = CCA(filter_bank=True).subband_scores(x)
    power = (np.abs(np.fft.rfft(x*np.hanning(x.shape[-1]), n=1000))**2).mean(0)
    bins = np.array([[4*f*h for h in [1, 2, 3]] for f in range(8, 16)])
    harmonic = np.log1p(power[bins]/max(power[24:241].mean(), 1e-6))
    return np.concatenate([sub, harmonic], axis=1).astype('float32')


class SharedScorer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        # Shared across candidate frequencies; no frequency ID or class-specific bias.
        self.linear = nn.Linear(6, 1, bias=False)

    def forward(self, x):
        return self.linear((x-self.mean)/self.std).squeeze(-1)


def metric(scores, trials):
    pred = scores.argmax(1)
    y = np.array([t.y for t in trials])
    subjects = sorted(set(t.subject for t in trials))
    by = {s: float(np.mean(pred[[t.subject == s for t in trials]] ==
                          y[[t.subject == s for t in trials]])) for s in subjects}
    cm = np.zeros((8, 8), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {'macro_accuracy': float(np.mean(list(by.values()))), 'subjects': by, 'labels': y.tolist(),
            'confusion': cm.tolist(), 'prediction': pred.tolist(), 'scores': scores.tolist()}


def arrays(trials, train):
    x, y, manifest = [], [], []
    for t in trials:
        starts = range(0, t.x.shape[-1]-375+1, 125) if train else [0]
        for start in starts:
            x.append(preprocess_window(t.x, 375, start))
            y.append(t.y)
            manifest.append({'uid': t.uid, 'subject': t.subject, 'start': start})
    return np.stack(x), np.array(y), manifest


def main():
    out = ROOT/'runs/v4-scoring-pilot-cpu'
    out.mkdir(exist_ok=False)
    config = {'train': ['S2','S5','S10','S15','S18','S25'], 'dev': ['S6','S20','S51'],
              'seeds': [2026, 2027], 'samples': 375, 'stride': 125, 'epochs': 50,
              'samples_per_epoch': 768, 'batch': 64, 'lr': .001, 'weight_decay': .001,
              'augmentation': 'none for either fitted arm',
              'normalization': 'feature dimension pooled across train windows and candidate classes only',
              'selection': 'earliest maximum dev macro accuracy',
              'gate': 'scorer minus max(FBCCA, same-seed EEGNet): both >=0pp and mean >=3pp',
              'device_for_fit': 'CPU; CUDA attempt failed before first optimizer update', 'latency': 'CPU batch1, 1 numerical thread, interleaved 3 passes after warmup',
              'cpu': platform.processor(), 'confirmation_read': False, 'holdout_read': False,
              'D1_D2_read': False}
    save(out/'preregistered_config.json', config)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    decision = json.loads((ROOT/'runs/v4-window-audit/frozen_window_decision.json').read_text())
    assert decision['samples'] == 375
    parent = ROOT/'runs/v2-beta-subject-isolated'
    roles = json.loads((parent/'subject_roles.json').read_text())
    hashes = json.loads((parent/'source_fingerprints.json').read_text())['public_pool']
    audit = {}
    groups = {}
    for role in ['train', 'dev']:
        assert set(s+'.mat' for s in config[role]) <= set(roles[role])
        files = [str(Path('data/raw/public/beta')/(s+'.mat')) for s in config[role]]
        for f in files:
            audit[f] = fingerprint(ROOT/f)
            assert audit[f] == hashes[f]
        groups[role] = beta_trials(ROOT, files)
    assert not set(config['train']) & set(config['dev'])
    assert_allowed_sources([t.source for t in groups['train']])
    save(out/'read_sources.json', audit)
    tx, ty, manifest = arrays(groups['train'], True)
    dx, dy, devmanifest = arrays(groups['dev'], False)
    save(out/'window_manifest.json', {'train': manifest, 'dev': devmanifest})
    print('FEATURE_EXTRACTION', len(tx), len(dx), flush=True)
    tf = np.stack([features(x) for x in tx])
    df = np.stack([features(x) for x in dx])
    mean, std = tf.mean((0,1)), np.maximum(tf.std((0,1)), 1e-6)
    np.savez(out/'train_only_feature_stats.npz', mean=mean, std=std)
    fb = df[:,:,:3] @ (np.arange(1,4)**-1.25+.25)
    results = {'fbcca': metric(fb, groups['dev'])}
    device = torch.device('cpu')
    models = {}
    for seed in config['seeds']:
        # Identical window indices and update budgets, not identical architecture weights.
        rng = np.random.default_rng(seed)
        batches = rng.integers(len(tx), size=(50,768))
        np.save(out/f'sampling-{seed}.npy', batches)
        for arm in ['eegnet', 'shared_scorer']:
            torch.manual_seed(seed)
            model = (EEGNetStudent() if arm=='eegnet' else SharedScorer(mean, std)).to(device)
            x = torch.from_numpy(tx if arm=='eegnet' else tf).to(device)
            dev = torch.from_numpy(dx if arm=='eegnet' else df).to(device)
            y = torch.from_numpy(ty).long().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            history, best, state, selected = [], -1, None, None
            started = time.perf_counter()
            for epoch in range(50):
                model.train(); losses=[]
                for indices in batches[epoch].reshape(-1,64):
                    idx = torch.as_tensor(indices, device=device)
                    loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad(): score = model(dev).cpu().numpy()
                acc = metric(score, groups['dev'])['macro_accuracy']
                history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'dev_macro_accuracy': acc})
                if acc > best:
                    best, state, selected = acc, copy.deepcopy(model.state_dict()), epoch+1
                if (epoch+1)%10==0: print(seed,arm,epoch+1,round(acc,4),flush=True)
            model.load_state_dict(state); model.eval()
            with torch.no_grad(): score = model(dev).cpu().numpy()
            key = f'{arm}_{seed}'
            results[key] = metric(score, groups['dev'])
            results[key].update(selected_epoch=selected, training_seconds=time.perf_counter()-started,
                                parameters=sum(p.numel() for p in model.parameters()), updates=600)
            torch.save({'state_dict': state, 'config': config, 'arm': arm, 'seed': seed}, out/(key+'.pt'))
            save(out/(key+'_history.json'),history)
            models[key] = model.cpu()
            save(out/'partial_results.json', results)
    gains = [100*(results[f'shared_scorer_{s}']['macro_accuracy']-
                  max(results['fbcca']['macro_accuracy'],results[f'eegnet_{s}']['macro_accuracy'])) for s in config['seeds']]
    save(out/'frozen_screen_decision.json', {'gains_pp': gains, 'mean_gain_pp': float(np.mean(gains)),
         'eligible': bool(min(gains)>=0 and np.mean(gains)>=3), 'confirmation_read': False})
    # Real CPU signal-to-score path includes preprocessing and scorer feature extraction.
    def run(key, raw):
        z = preprocess_window(raw,375)
        if key == 'fbcca': return CCA(filter_bank=True).score_one(z)
        v = z if key.startswith('eegnet') else features(z)
        with torch.no_grad(): return models[key](torch.from_numpy(v[None])).numpy()[0]
    keys = list(results)
    timings = {k: [] for k in keys}
    for k in keys:
        for _ in range(3): run(k,groups['dev'][0].x)
    for repeat in range(3):
        for i,t in enumerate(groups['dev']):
            order = np.roll(keys,(repeat+i)%len(keys))
            for k in order:
                start = time.perf_counter(); scores = run(k,t.x)
                timings[k].append((time.perf_counter()-start)*1000)
                np.testing.assert_allclose(scores, np.asarray(results[k]['scores'])[i],rtol=2e-3,atol=2e-3)
    for k in keys:
        results[k]['cpu_compute_ms'] = {'median':float(np.median(timings[k])), 'p95':float(np.percentile(timings[k],95)), 'n':len(timings[k])}
    save(out/'results.json',results)
    save(out/'completed.json',{'complete':True,'fits':4,'trained_windows':len(tx),'development_trials':len(dx)})
    print('COMPLETE',gains,flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
