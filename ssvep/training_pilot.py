"""Bounded training-method ablation, with subject-isolated development replication."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import beta_trials, preprocess, fingerprint
from .models import EEGNetStudent
from .policy import assert_allowed_sources
from .experiment import ROOT,write,predict


class TrialCrops(Dataset):
    def __init__(self,trials,mode,seed,n=250,stride=125):
        assert_allowed_sources([t.source for t in trials])
        self.trials=trials;self.signals=[preprocess(t.x) for t in trials]
        self.rng=np.random.default_rng(seed);self.n=n;self.current_n=n;self.mode=mode
        self.index=[]
        for i,x in enumerate(self.signals):
            if x.shape[-1]<n:raise ValueError('Training record shorter than window')
            if mode=='enumerated':self.index.extend((i,start) for start in range(0,x.shape[-1]-n+1,stride))
            else:self.index.append((i,None))
        self.subjects=[trials[i].subject for i,_ in self.index]
        counts={s:self.subjects.count(s) for s in set(self.subjects)}
        self.weights=[1. if mode=='enumerated' else 1/counts[s] for s in self.subjects]
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        trial,start=self.index[index];x=self.signals[trial]
        if self.current_n>x.shape[-1]:raise ValueError('Curriculum window exceeds original trial')
        if start is None:start=int(self.rng.integers(0,x.shape[-1]-self.current_n+1))
        elif self.current_n!=self.n:
            # Keep original trial/window sampling mass; map its relative position
            # into the valid range for a longer window instead of changing counts.
            span=x.shape[-1]-self.n
            start=round(start*(x.shape[-1]-self.current_n)/span) if span else 0
        return torch.from_numpy(x[:,start:start+self.current_n].copy()),self.trials[trial].y,trial


def augmentation(x,phase_gain,gain_std=.15):
    if phase_gain:
        # All electrodes of the same trial share the temporal shift.
        shift=torch.randint(x.shape[-1],(x.shape[0],1),device=x.device)
        index=(torch.arange(x.shape[-1],device=x.device)[None,:]-shift)%x.shape[-1]
        x=torch.gather(x,2,index[:,None,:].expand(-1,x.shape[1],-1))
        x=x*torch.exp(gain_std*torch.randn(x.shape[0],x.shape[1],1,device=x.device))
    return x+.02*torch.randn_like(x)


def cross_subject_contrastive(features,labels,subjects,temperature=.2):
    """Same-class/different-subject positives; other classes are negatives.

    Same-class/same-subject pairs are excluded from the denominator rather than
    treated as negatives. Anchors without cross-subject positives contribute zero.
    """
    if temperature<=0:raise ValueError('Temperature must be positive')
    z=F.normalize(features,dim=1,eps=1e-8)
    logits=z@z.T/temperature
    same_label=labels[:,None]==labels[None,:]
    different_subject=subjects[:,None]!=subjects[None,:]
    positive=same_label&different_subject
    denominator=(~same_label)|positive
    logp=logits-torch.logsumexp(logits.masked_fill(~denominator,-1e9),dim=1,keepdim=True)
    counts=positive.sum(dim=1);valid=counts>0
    if not bool(valid.any()):return features.sum()*0
    per_anchor=-torch.where(positive,logp,torch.zeros_like(logp)).sum(dim=1)/counts.clamp_min(1)
    return per_anchor[valid].mean()


def development_arrays(trials,n=250):
    return np.stack([preprocess(t.x)[:,:n] for t in trials]),np.array([t.y for t in trials]),[t.subject for t in trials]


def scores_by_subject(score,labels,subjects):
    out={}
    for subject in sorted(set(subjects)):
        mask=np.array(subjects)==subject
        out[subject]={'accuracy':float(np.mean(score[mask].argmax(1)==labels[mask])),'trials':int(mask.sum())}
    return {'macro_subject_accuracy':float(np.mean([r['accuracy'] for r in out.values()])),
            'accuracy':float(np.mean(score.argmax(1)==labels)),'subjects':out,'trials':len(labels)}


def fit(arm,seed,trials,development,config,folder,device):
    folder.mkdir(parents=True)
    torch.manual_seed(seed+2)
    model=EEGNetStudent().to(device)
    init={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    mode='enumerated' if arm in ['enumerated','cross_subject_contrastive','long_to_short_curriculum'] else 'random'
    dataset=TrialCrops(trials,mode,seed,config['window_samples'],config['enumerated_stride_samples'])
    generator=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(dataset.weights,config['samples_per_epoch'],replacement=True,generator=generator)
    loader=DataLoader(dataset,batch_size=config['batch_size'],sampler=sampler,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['minimum_learning_rate']) if arm=='phase_gain_cosine' else None
    best=-1.;history=[];best_state=None
    x,y,subjects=development;started=time.perf_counter()
    for epoch in range(config['epochs']):
        dataset.current_n=config['curriculum_warmup_window_samples'] if arm=='long_to_short_curriculum' and epoch<config['curriculum_warmup_epochs'] else config['window_samples']
        model.train();losses=[];auxiliary=[];draws={s:0 for s in sorted(set(dataset.subjects))}
        for bx,by,ids in loader:
            for i in ids.tolist():draws[trials[i].subject]+=1
            bx=augmentation(bx.to(device),arm in ['phase_gain','phase_gain_cosine'],config['augmentation_gain_log_std'])
            by=by.to(device)
            if arm=='cross_subject_contrastive':
                logits,features=model(bx,return_features=True)
                names=sorted(set(dataset.subjects));mapping={s:i for i,s in enumerate(names)}
                subject_ids=torch.tensor([mapping[trials[i].subject] for i in ids.tolist()],device=device)
                contrast=cross_subject_contrastive(features,by,subject_ids,config['contrastive_temperature'])
                loss=nn.functional.cross_entropy(logits,by)+config['contrastive_weight']*contrast
                auxiliary.append(float(contrast.detach()))
            else:loss=nn.functional.cross_entropy(model(bx),by)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        result=scores_by_subject(predict(model,x,device),y,subjects)
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'learning_rate':optimizer.param_groups[0]['lr'],
                'screen_development':result,'samples_by_subject':draws,'training_window_samples':dataset.current_n}
        if auxiliary:record['cross_subject_contrastive_loss']=float(np.mean(auxiliary))
        history.append(record)
        if result['macro_subject_accuracy']>best:
            best=result['macro_subject_accuracy'];best_state=copy.deepcopy(model.state_dict())
            torch.save({'model':'student','state_dict':best_state,'arm':arm,'seed':seed,
                        'config':config,'selected_epoch':epoch+1,'development':result},folder/'student.pt')
        if scheduler:scheduler.step()
    model.load_state_dict(best_state);model.eval()
    write(folder/'history.json',history)
    result=scores_by_subject(predict(model,x,device),y,subjects)
    result.update(seconds=time.perf_counter()-started,epochs=config['epochs'],gradient_steps=len(loader)*config['epochs'],
                  parameters=sum(p.numel() for p in model.parameters()))
    write(folder/'screen_result.json',result)
    print('PILOT_FIT',arm,seed,'screen_accuracy',round(result['macro_subject_accuracy'],4),flush=True)
    return model,result,init


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--config',default='configs/pilot_training_v3.json');args=parser.parse_args()
    root=ROOT/'runs'/args.run
    if root.exists():raise ValueError('Use a new pilot run name')
    config=json.loads((ROOT/args.config).read_text())
    parent=ROOT/'runs'/config['base_run'];roles=json.loads((parent/'subject_roles.json').read_text())
    training=config['training_subjects'];screen=config['screen_development_subjects'];confirm=config['confirmation_development_subjects']
    if not set(s+'.mat' for s in training)<=set(roles['train']):raise ValueError('Pilot training subjects must come from existing training partition')
    if not set(s+'.mat' for s in screen+confirm)<=set(roles['dev']):raise ValueError('Pilot development subjects must come from existing development partition')
    if set(training)&set(screen+confirm) or set(screen)&set(confirm):raise ValueError('Subject leakage')
    if set(s+'.mat' for s in training+screen+confirm)&set(roles['holdout']):raise ValueError('Holdout subject accessed')
    base_config=json.loads((parent/'resolved_config.json').read_text())
    fingerprints=json.loads((parent/'source_fingerprints.json').read_text())['public_pool']
    def paths(subjects):return [f'data/raw/public/beta/{s}.mat' for s in subjects]
    def validate(subjects):
        for rel in paths(subjects):
            # JSON paths from Windows may contain backslashes.
            old=fingerprints.get(rel,fingerprints.get(str(Path(rel))))
            if old is None or fingerprint(ROOT/rel)!=old:raise ValueError('Source fingerprint changed')
    validate(training+screen)
    root.mkdir(parents=True);write(root/'preregistered_config.json',config)
    write(root/'data_access_audit.json',{'fitting_and_selection_files':paths(training+screen),
          'confirmation_files_open_only_after_selection':paths(confirm),'competition_data_read':False,
          'beta_holdout_data_read':False,'all_models_freshly_initialized':True})
    train_trials=beta_trials(ROOT,paths(training),base_config['beta_latency_seconds'])
    screen_trials=beta_trials(ROOT,paths(screen),base_config['beta_latency_seconds'])
    dev=development_arrays(screen_trials,config['window_samples'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['cpu_threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    results={};initials={}
    for seed in config['screen_seeds']:
        for arm in config['arms']:
            _,result,initial=fit(arm,seed,train_trials,dev,config,root/f'seed-{seed}'/arm,device)
            results.setdefault(arm,{})[str(seed)]=result
            if seed not in initials:initials[seed]=initial
            elif not all(torch.equal(initials[seed][k],v) for k,v in initial.items()):raise ValueError('Arms did not share identical initialization')
    write(root/'screen_results.json',results)
    decisions={}
    for arm in config['arms'][1:]:
        gains=[100*(results[arm][str(s)]['macro_subject_accuracy']-results['enumerated'][str(s)]['macro_subject_accuracy']) for s in config['screen_seeds']]
        eligible=np.mean(gains)>=config['screen_minimum_gain_percentage_points'] and min(gains)>=0
        decisions[arm]={'paired_gains_percentage_points':gains,'mean_gain_percentage_points':float(np.mean(gains)),'eligible':bool(eligible)}
    eligible=[arm for arm in decisions if decisions[arm]['eligible']]
    chosen=max(eligible,key=lambda arm:decisions[arm]['mean_gain_percentage_points']) if eligible else None
    write(root/'frozen_screen_decision.json',{'candidates':decisions,'chosen':chosen,
           'confirmation_subjects_not_yet_opened':True,'parameter_search_complete':True})
    confirmation=None
    if chosen:
        # Checkpoint selection still uses screening developers. New records do not select epochs.
        pair={}
        for arm in ['enumerated',chosen]:
            model,result,_=fit(arm,config['confirmation_seed'],train_trials,dev,config,
                               root/'confirmation'/arm,device)
            pair[arm]=model
        validate(confirm)
        extra=development_arrays(beta_trials(ROOT,paths(confirm),base_config['beta_latency_seconds']),config['window_samples'])
        ex,ey,es=extra
        evaluated={arm:scores_by_subject(predict(model,ex,device),ey,es) for arm,model in pair.items()}
        gains={s:100*(evaluated[chosen]['subjects'][s]['accuracy']-evaluated['enumerated']['subjects'][s]['accuracy']) for s in confirm}
        mean=100*(evaluated[chosen]['macro_subject_accuracy']-evaluated['enumerated']['macro_subject_accuracy'])
        passed=mean>=config['confirmation_minimum_gain_percentage_points'] and min(gains.values())>=-config['confirmation_maximum_single_subject_drop_percentage_points']
        confirmation={'results':evaluated,'mean_gain_percentage_points':mean,'subject_gains_percentage_points':gains,
                      'passed':bool(passed),'seed':config['confirmation_seed']}
        write(root/'confirmation_results.json',confirmation)
    write(root/'conclusion.json',{'chosen_screen_direction':chosen,'confirmation':confirmation,
         'direction_supported':bool(confirmation and confirmation['passed']),
         'no_competition_or_holdout_evaluation':True,'small_scale_exploratory':True,
         'next_action':'Consider full BETA subject-isolated comparison only if confirmation passes; do not tune on D1/D2.'})
    write(root/'completed.json',{'complete':True})
    print('PILOT_COMPLETE',root,flush=True)


if __name__=='__main__':main()
