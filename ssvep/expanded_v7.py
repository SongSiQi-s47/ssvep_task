"""Targeted training-budget follow-up with a separate untouched BETA confirmation set."""
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
from .data import competition_trials, fingerprint
from .expanded_v5 import ROOT, arrays, get_trials, report, score_models, write
from .models import EEGNetStudent
from .policy import assert_allowed_sources

OUT=ROOT/'runs/v7-expanded-eegnet'
CONFIG_PATH=ROOT/'configs/expanded_v7.json'
V6=ROOT/'runs/v6-expanded-eegnet'


def config_checked():
    c=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'runs/v2-beta-subject-isolated/subject_roles.json').read_text())
    v5=json.loads((ROOT/'configs/expanded_v5.json').read_text())
    v6=json.loads((ROOT/'configs/expanded_v6.json').read_text())
    used=set(sum(old.values(),[]))|{s+'.mat' for s in v5['new_test_subjects']+v6['additional_train']+v6['new_test']}
    pool=[f'S{i}' for i in range(1,71) if f'S{i}.mat' not in used]
    ordered=sorted(pool,key=lambda s:hashlib.sha256(('expanded-v7:'+s+'.mat').encode()).hexdigest())
    assert c['new_test']==ordered[:5] and c['train']==v6['train'] and c['dev']==v6['dev']
    assert c['samples_per_epoch']==1408 and c['batch']==64
    return c


def load_model(path,expected_hash=None):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if expected_hash is not None: assert saved['config_sha256']==expected_hash
    model=EEGNetStudent().eval()
    model.load_state_dict(saved['state_dict'])
    return model


def train():
    c=config_checked()
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'preregistered_config.json',c)
    old=json.loads((V6/'training_source_fingerprints.json').read_text())
    for s in c['train']+c['dev']:
        assert fingerprint(ROOT/f'data/raw/public/beta/{s}.mat')==old[s+'.mat']['sha256']
    write(OUT/'training_source_fingerprints.json',old)
    trials=get_trials(c['train'])
    assert_allowed_sources([t.source for t in trials])
    dev_trials=get_trials(c['dev'])
    tx,ty=arrays(trials,c['samples'],c['stride'],True)
    dx,_=arrays(dev_trials,c['samples'],c['stride'],False)
    x,y,dev=torch.from_numpy(tx),torch.from_numpy(ty),torch.from_numpy(dx)
    torch.set_num_threads(1)
    results={'fbcca':report(CCA(filter_bank=True).scores(dx),dev_trials)}
    for seed in c['seeds']:
        torch.manual_seed(seed)
        model=EEGNetStudent()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        rng=np.random.default_rng(seed)
        history=[];best=-1.;best_state=None;selected=None
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
            with torch.no_grad():scores=model(dev).numpy()
            acc=report(scores,dev_trials)['subject_macro_accuracy']
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'dev_subject_macro':acc})
            if acc>best:best,best_state,selected=acc,copy.deepcopy(model.state_dict()),epoch+1
            if (epoch+1)%10==0:print('TRAIN',seed,epoch+1,round(acc,4),flush=True)
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad():scores=model(dev).numpy()
        key=f'eegnet_26_more_updates_{seed}'
        results[key]=report(scores,dev_trials)
        results[key].update(selected_epoch=selected,training_seconds=time.perf_counter()-started,
                            train_original_trials=len(trials),train_windows=len(tx),updates=c['epochs']*c['samples_per_epoch']//c['batch'])
        torch.save({'state_dict':best_state,'config_sha256':fingerprint(CONFIG_PATH),'seed':seed},OUT/(key+'.pt'))
        write(OUT/(key+'_history.json'),history)
        write(OUT/'development_partial.json',results)
    write(OUT/'development_results.json',results)
    write(OUT/'models_frozen.json',{'complete':True,'config_sha256':fingerprint(CONFIG_PATH),
        'new_test_read':False,'D1_D2_read':False,'keys':[f'eegnet_26_more_updates_{s}' for s in c['seeds']]})
    print('MODELS_FROZEN',flush=True)


def confirm():
    c=config_checked()
    frozen=json.loads((OUT/'models_frozen.json').read_text())
    assert frozen['complete'] and frozen['config_sha256']==fingerprint(CONFIG_PATH)
    if (OUT/'confirmation_results.json').exists():raise FileExistsError('Confirmation already exists')
    torch.set_num_threads(1)
    models={}
    for seed in c['seeds']:
        models[f'eegnet_more_{seed}']=load_model(OUT/f'eegnet_26_more_updates_{seed}.pt',frozen['config_sha256'])
        models[f'eegnet_base_{seed}']=load_model(V6/f'eegnet_26_{seed}.pt')
    groups={'new_beta_test':get_trials(c['new_test'])}
    manifest={Path(f['path']).name:f for f in json.loads((ROOT/'data/competition_manifest.json').read_text())['files']}
    for name in ('D1','D2'):
        assert fingerprint(ROOT/f'data/raw/competition/{name}.csv')==manifest[name+'.csv']['sha256']
    for t in competition_trials(ROOT):groups.setdefault(t.subject,[]).append(t)
    results={}
    for name,trials in groups.items():
        x,_=arrays(trials,c['samples'],c['stride'],False)
        prediction={'fbcca':report(CCA(filter_bank=True).scores(x),trials)}
        prediction.update({key:report(scores,trials) for key,scores in score_models(models,x).items()})
        results[name]=prediction
        print('CONFIRM',name,{key:round(v['subject_macro_accuracy'],4) for key,v in prediction.items()},flush=True)
    hashes={s+'.mat':fingerprint(ROOT/f'data/raw/public/beta/{s}.mat') for s in c['new_test']}
    hashes.update({name+'.csv':fingerprint(ROOT/f'data/raw/competition/{name}.csv') for name in ('D1','D2')})
    write(OUT/'confirmation_source_fingerprints.json',hashes)
    write(OUT/'confirmation_results.json',results)
    write(OUT/'completed.json',{'complete':True,'train_subjects':len(c['train']),'new_test_subjects':c['new_test'],'D1_D2_evaluation_only':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','confirm']);stage=p.parse_args().stage
    with threadpool_limits(limits=1):
        if stage=='train':train()
        else:confirm()

if __name__=='__main__':main()
