"""Run the predeclared v6 BETA subject expansion and frozen confirmation."""
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
from .data import competition_trials, fingerprint, preprocess_window
from .expanded_v5 import ROOT, arrays, get_trials, report, score_models, write
from .models import EEGNetStudent
from .policy import assert_allowed_sources

OUT = ROOT/'runs/v6-expanded-eegnet'
CONFIG_PATH = ROOT/'configs/expanded_v6.json'
V5 = ROOT/'runs/v5-expanded-eegnet'


def config_checked():
    c=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    v5=json.loads((ROOT/'configs/expanded_v5.json').read_text())
    used=set(sum(old.values(), [])) | {s+'.mat' for s in v5['new_test_subjects']}
    pool=[f'S{i}' for i in range(1,71) if f'S{i}.mat' not in used]
    ordered=sorted(pool,key=lambda s:hashlib.sha256(('expanded-v6:'+s+'.mat').encode()).hexdigest())
    assert c['additional_train']==ordered[:12] and c['new_test']==ordered[12:17]
    assert c['existing_train']==v5['training_groups']['fourteen']
    assert c['train']==c['existing_train']+c['additional_train']
    assert not(set(c['train']) & (set(c['dev']) | set(c['new_test'])))
    return c


def train():
    c=config_checked()
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'preregistered_config.json',c)
    fingerprints={}
    manifest=json.loads((ROOT/'configs/download_manifest.json').read_text())
    entries={f['name']: f for f in manifest if f['dataset']=='beta'}
    for s in c['train']+c['dev']:
        path=ROOT/f'data/raw/public/beta/{s}.mat'
        import hashlib as hashes
        digest=hashes.md5(path.read_bytes()).hexdigest()
        assert digest==entries[s+'.mat']['md5'], s
        fingerprints[s+'.mat']={'md5':digest,'sha256':fingerprint(path)}
    write(OUT/'training_source_fingerprints.json',fingerprints)
    trials=get_trials(c['train'])
    assert_allowed_sources([t.source for t in trials])
    dev_trials=get_trials(c['dev'])
    x,y=arrays(trials,c['samples'],c['stride'],True)
    dx,_=arrays(dev_trials,c['samples'],c['stride'],False)
    x,y,dev=torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(dx)
    torch.set_num_threads(1)
    results={'fbcca':report(CCA(filter_bank=True).scores(dx),dev_trials)}
    for seed in c['seeds']:
        torch.manual_seed(seed)
        model=EEGNetStudent()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        rng=np.random.default_rng(seed)
        history=[]; best=-1.;best_state=None;selected=None
        started=time.perf_counter()
        for epoch in range(c['epochs']):
            model.train()
            indices=rng.integers(len(x),size=c['samples_per_epoch'])
            losses=[]
            for batch in indices.reshape(-1,c['batch']):
                loss=nn.functional.cross_entropy(model(x[batch]),y[batch])
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad(): scores=model(dev).numpy()
            accuracy=report(scores,dev_trials)['subject_macro_accuracy']
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'dev_subject_macro':accuracy})
            if accuracy>best:
                best,best_state,selected=accuracy,copy.deepcopy(model.state_dict()),epoch+1
            if (epoch+1)%10==0: print('TRAIN',seed,epoch+1,round(accuracy,4),flush=True)
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad(): scores=model(dev).numpy()
        key=f'eegnet_26_{seed}'
        results[key]=report(scores,dev_trials)
        results[key].update(selected_epoch=selected,training_seconds=time.perf_counter()-started,
                            train_original_trials=len(trials),train_windows=len(x))
        torch.save({'state_dict':best_state,'config_sha256':fingerprint(CONFIG_PATH),'seed':seed},OUT/(key+'.pt'))
        write(OUT/(key+'_history.json'),history)
        write(OUT/'development_partial.json',results)
    write(OUT/'development_results.json',results)
    write(OUT/'models_frozen.json',{'complete':True,'config_sha256':fingerprint(CONFIG_PATH),
        'new_test_read':False,'D1_D2_read':False,'keys':[f'eegnet_26_{s}' for s in c['seeds']]})
    print('MODELS_FROZEN',flush=True)


def confirm():
    c=config_checked()
    frozen=json.loads((OUT/'models_frozen.json').read_text())
    assert frozen['complete'] and frozen['config_sha256']==fingerprint(CONFIG_PATH)
    if (OUT/'confirmation_results.json').exists(): raise FileExistsError('Confirmation already exists')
    torch.set_num_threads(1)
    models={}
    for seed in c['seeds']:
        key=f'eegnet_26_{seed}'
        saved=torch.load(OUT/(key+'.pt'),map_location='cpu',weights_only=False)
        assert saved['config_sha256']==frozen['config_sha256']
        m=EEGNetStudent().eval();m.load_state_dict(saved['state_dict']);models[key]=m
        old=torch.load(V5/(f'eegnet_fourteen_{seed}.pt'),map_location='cpu',weights_only=False)
        m=EEGNetStudent().eval();m.load_state_dict(old['state_dict']);models[f'eegnet_14_{seed}']=m
    groups={'new_beta_test':get_trials(c['new_test'])}
    # Existing processed competition trials are from original CSV; verify manifest before scoring.
    manifest={Path(f['path']).name:f for f in json.loads((ROOT/'data/competition_manifest.json').read_text())['files']}
    for name in ('D1','D2'):
        path=ROOT/f'data/raw/competition/{name}.csv'
        assert path.exists() and fingerprint(path)==manifest[name+'.csv']['sha256']
    for trial in competition_trials(ROOT): groups.setdefault(trial.subject,[]).append(trial)
    results={}
    for name,trials in groups.items():
        x,_=arrays(trials,c['samples'],c['stride'],False)
        scores=score_models(models,x)
        results[name]={'fbcca':report(CCA(filter_bank=True).scores(x),trials)}
        results[name].update({key:report(value,trials) for key,value in scores.items()})
        print('CONFIRM',name,{key:round(v['subject_macro_accuracy'],4) for key,v in results[name].items()},flush=True)
    hashes={s+'.mat':fingerprint(ROOT/f'data/raw/public/beta/{s}.mat') for s in c['new_test']}
    hashes.update({name+'.csv':fingerprint(ROOT/f'data/raw/competition/{name}.csv') for name in ('D1','D2')})
    write(OUT/'confirmation_source_fingerprints.json',hashes)
    write(OUT/'confirmation_results.json',results)
    write(OUT/'completed.json',{'complete':True,'train_subjects':len(c['train']),
        'new_test_subjects':c['new_test'],'D1_D2_evaluation_only':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','confirm']);stage=p.parse_args().stage
    with threadpool_limits(limits=1):
        if stage=='train':train()
        else:confirm()

if __name__=='__main__':main()
