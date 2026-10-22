import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .baselines import CCA, reference_template_scores
from .data import competition_trials, beta_trials, split_trials, make_windows, fingerprint
from .models import build_model, distillation_loss
from .policy import assert_allowed_sources

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def accuracy(scores,y):
    return float(np.mean(np.argmax(scores,axis=1)==y))


def metrics(scores,y,sources):
    result={'accuracy':accuracy(scores,y),'samples':len(y)}
    for source in sorted(set(sources)):
        mask=np.array(sources)==source
        result[source]={'accuracy':accuracy(scores[mask],y[mask]),'samples':int(mask.sum())}
    result['mean_domain_accuracy']=float(np.mean([result[s]['accuracy'] for s in set(sources)]))
    confusion=np.zeros((8,8),dtype=int)
    np.add.at(confusion,(y,np.argmax(scores,axis=1)),1)
    result['confusion_matrix']=confusion.tolist()
    return result


def predict(model,x,device,features=False):
    model.eval(); out=[]
    with torch.no_grad():
        for batch in np.array_split(x,max(1,(len(x)+63)//64)):
            result=model(torch.from_numpy(batch).to(device),return_features=features)
            out.append((result[1] if features else result).cpu().numpy())
    return np.concatenate(out)


def train(name,config,arrays,run,device,teacher=None,soft_targets=None):
    assert_allowed_sources(arrays['train'][3],stage='training')
    # Equal initialization for direct and distilled students makes their comparison meaningful.
    torch.manual_seed(config['seed'] + (1 if name=='teacher' else 2))
    model=build_model(name).to(device)
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    x,y,_,sources=arrays['train']
    source_counts={s:sources.count(s) for s in set(sources)}
    weights=torch.tensor([1/source_counts[s] for s in sources],dtype=torch.double)
    generator=torch.Generator().manual_seed(config['seed'])
    sampler=WeightedRandomSampler(weights,len(y),replacement=True,generator=generator)
    tensors=[torch.from_numpy(x),torch.from_numpy(y).long()]
    if soft_targets is not None:
        if soft_targets.shape!=(len(x),8):raise ValueError('Soft targets must align with training windows')
        tensors.append(torch.from_numpy(soft_targets).float())
    loader=DataLoader(TensorDataset(*tensors),
                      batch_size=config['batch_size'],sampler=sampler,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best=-1.; stale=0; history=[]; best_state=None
    for epoch in range(config['epochs']):
        model.train(); losses=[]; started=time.perf_counter()
        for batch in loader:
            bx,by=batch[:2]
            bx,by=bx.to(device),by.to(device)
            # Identical input perturbation for teacher and student during distillation.
            bx=bx+0.02*torch.randn_like(bx)
            logits=model(bx)
            if teacher is None and soft_targets is None:loss=nn.functional.cross_entropy(logits,by)
            else:
                if soft_targets is not None:target=batch[2].to(device)
                else:
                    with torch.no_grad():target=teacher(bx)
                loss=distillation_loss(logits,target,by,config['distillation_temperature'],config['distillation_alpha'])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); losses.append(float(loss.detach()))
        vx,vy,_,vs=arrays['val']
        validation=metrics(predict(model,vx,device),vy,vs)
        score=validation['mean_domain_accuracy']
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),
                'validation':validation,'seconds':time.perf_counter()-started}
        history.append(record);write(run/f'{name}_history.json',history)
        print(name,'epoch',epoch+1,'val_domain_accuracy',round(score,4),flush=True)
        if score>best:
            best=score;stale=0;best_state=copy.deepcopy(model.state_dict())
            torch.save({'model':name,'state_dict':best_state,'config':config,
                        'best_validation':validation,'epoch':epoch+1},run/f'{name}.pt')
        else:stale+=1
        if stale>=config['patience']:break
    model.load_state_dict(best_state);model.eval()
    return model


def scale_rows(score):
    return (score-score.mean(axis=1,keepdims=True))/np.maximum(score.std(axis=1,keepdims=True),1e-6)


def calibrated_scores(trials,arrays,base_scores,model,device):
    """Use ONLY labeled support; never pass query labels to calibration functions."""
    assert_allowed_sources(arrays[3],stage='calibration')
    x,y,ids,sources=arrays
    lookup={t.uid:t for t in trials}
    groups={}
    for i,uid in enumerate(ids):
        t=lookup[uid]
        groups.setdefault((t.source,t.subject),[]).append(i)
    query_indices=[];new_scores=[];support_records=[]
    for group,indices in sorted(groups.items()):
        # Stable original-order support selection; independent of prediction correctness.
        support=[]
        for label in range(8):
            candidate=[i for i in indices if y[i]==label]
            if not candidate:raise ValueError('Missing calibration class')
            support.append(candidate[0])
        support_uids={ids[i] for i in support}
        query=[i for i in indices if ids[i] not in support_uids]
        if not query:continue
        if model is None:
            calibration=reference_template_scores(x[support],y[support],x[query])
        else:
            sf=predict(model,x[support],device,features=True)
            qf=predict(model,x[query],device,features=True)
            center=sf.mean(axis=0,keepdims=True)
            sf=sf-center;qf=qf-center
            sf/=np.maximum(np.linalg.norm(sf,axis=1,keepdims=True),1e-8)
            qf/=np.maximum(np.linalg.norm(qf,axis=1,keepdims=True),1e-8)
            calibration=qf@sf.T # supports are collected in label order 0..7
        query_indices.extend(query);new_scores.append(calibration)
        support_records.append({'group':list(group),'support_trial_ids':sorted(support_uids),
                                'query_trial_ids':[ids[i] for i in query]})
    q=np.array(query_indices,dtype=int)
    return q,scale_rows(base_scores[q]),scale_rows(np.concatenate(new_scores)),support_records


def latency(model,device,length):
    model.eval();x=torch.randn(1,6,length,device=device)
    with torch.no_grad():
        for _ in range(10):model(x)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=[]
        for _ in range(50):
            started=time.perf_counter();model(x)
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-started)*1000)
    return {'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
            'batch_size':1,'device':str(device),'includes':'model only; excludes signal collection and preprocessing'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiment_v1.json')
    parser.add_argument('--run',required=True)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--max-beta-subjects',type=int)
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    if args.epochs:config['epochs']=args.epochs
    if args.max_beta_subjects is not None:config['max_beta_subjects']=args.max_beta_subjects
    run=ROOT/'runs'/args.run
    if run.exists():raise ValueError('Use a new run name to preserve previous results')
    run.mkdir(parents=True)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.backends.cudnn.deterministic=True
    files=sorted((ROOT/'data/raw/public/beta').glob('S*.mat'),key=lambda p:int(p.stem[1:]))
    # Freeze the available, publisher-checksummed BETA files at the beginning of a run.
    status_path=ROOT/'logs/download_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    files=[p for p in files if status.get('beta/'+p.name,{}).get('status')=='verified']
    files=files[:config['max_beta_subjects']]
    config['beta_files']=[str(p.relative_to(ROOT)) for p in files]
    config['device']=str(device);config['torch_version']=str(torch.__version__)
    write(run/'resolved_config.json',config)
    data=competition_trials(ROOT)+beta_trials(ROOT,config['beta_files'],config['beta_latency_seconds'])
    split=split_trials(data,config['seed'])
    write(run/'split_manifest.json',{key:[{'uid':t.uid,'label':t.y,'source':t.source,
                 'group':t.subject,'samples':t.x.shape[-1]} for t in trials] for key,trials in split.items()})
    source_files=files+list((ROOT/'data/raw/competition').glob('*.csv'))
    write(run/'source_fingerprints.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in source_files})
    arrays={k:make_windows(v,config['window_seconds'],config['stride_seconds'],k=='train',config['causal_filter'])
            for k,v in split.items()}
    print('DATA', {k:len(v[1]) for k,v in arrays.items()},'DEVICE',device,flush=True)
    models={}
    models['teacher']=train('teacher',config,arrays,run,device)
    models['student']=train('student',config,arrays,run,device)
    models['distilled']=train('distilled',config,arrays,run,device,models['teacher'])
    algorithms={**models,'cca':CCA(),'fbcca':CCA(filter_bank=True)}
    validation_scores={name:predict(alg,arrays['val'][0],device) if name in models else alg.scores(arrays['val'][0])
                       for name,alg in algorithms.items()}
    # Select calibration weight using validation ONLY, then evaluate fixed settings on test.
    weights={};results={};support_manifests={}
    for name,alg in algorithms.items():
        model=alg if name in models else None
        vq,vbase,vcal,vrecord=calibrated_scores(split['val'],arrays['val'],validation_scores[name],model,device)
        candidates=[0.,.25,.5,1.]
        val_scores={str(w):metrics((1-w)*vbase+w*vcal,arrays['val'][1][vq],
                          [arrays['val'][3][i] for i in vq])['mean_domain_accuracy'] for w in candidates}
        w=max(candidates,key=lambda a:val_scores[str(a)])
        weights[name]=w
        tx,ty,_,ts=arrays['test']
        test=predict(alg,tx,device) if model is not None else alg.scores(tx)
        q,base,cal,record=calibrated_scores(split['test'],arrays['test'],test,model,device)
        qsource=[ts[i] for i in q]
        results[name]={'validation':metrics(validation_scores[name],arrays['val'][1],arrays['val'][3]),
                       'calibrated_validation_query':metrics((1-w)*vbase+w*vcal,arrays['val'][1][vq],
                                                       [arrays['val'][3][i] for i in vq]),
                       'zero_shot_all_test':metrics(test,ty,ts),
                       'zero_shot_query_only':metrics(base,ty[q],qsource),
                       'calibrated_query_only':metrics((1-w)*base+w*cal,ty[q],qsource),
                       'calibration_weight_selected_on_validation':w,'validation_weight_candidates':val_scores}
        support_manifests[name]={'validation':vrecord,'test':record}
        np.savez_compressed(run/f'{name}_predictions.npz',trial_ids=np.array(arrays['test'][2]),
                            true_labels=ty,scores=test,query_indices=q,calibrated_scores=(1-w)*base+w*cal)
        if model is not None:
            results[name]['parameters']=sum(p.numel() for p in model.parameters())
            results[name]['checkpoint_bytes']=(run/f'{name}.pt').stat().st_size
            results[name]['latency']=latency(model,device,round(config['window_seconds']*250))
    write(run/'calibration_support_manifest.json',support_manifests)
    write(run/'results.json',results)
    selected=max(results,key=lambda name:results[name]['validation']['mean_domain_accuracy'])
    write(run/'deployment_policy.json',{'algorithm':selected,'selected_by':'validation mean domain accuracy, zero-shot',
        'calibration_weight':weights[selected],'window_seconds':config['window_seconds'],
        'channels':['CP3','CPZ','CP4','PO3','POZ','PO4'],'sampling_rate_hz':250,
        'causal_filter':config['causal_filter'],'frequencies_hz':list(range(8,16))})
    teacher_val=results['teacher']['validation']['mean_domain_accuracy']
    student_val=results['student']['validation']['mean_domain_accuracy']
    write(run/'assessment.json',{'teacher_stronger_on_validation':teacher_val>student_val,
        'distillation_gain_on_validation':results['distilled']['validation']['mean_domain_accuracy']-student_val,
        'single_seed_exploratory':True,'competition_protocol':'held-out file, subject relationship unknown',
        'window_seconds':config['window_seconds'],'offline_zero_phase_filter':not config['causal_filter'],
        'note':'Test results describe this run; do not choose deployment model or tuning parameters from these test results.'})
    write(run/'completed.json',{'complete':True,'run':args.run})
    print('COMPLETE',run,flush=True)


if __name__=='__main__':main()
