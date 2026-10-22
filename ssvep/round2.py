"""Fresh independent-source training; existing datasets remain evaluation-only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import CCA
from .data import benchmark_trials,beta_trials,competition_trials,make_windows,fingerprint
from .models import build_model
from .policy import load_policy,public_subject_split,assert_allowed_sources
from .experiment import ROOT,write,metrics,train,predict,latency


def snapshot_trials(trials):
    return [{'uid':t.uid,'source':t.source,'subject':t.subject,'label':t.y} for t in trials]


def freeze_model_scores(models,x,device):
    return {name:predict(model,x,device) for name,model in models.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    root=ROOT/'runs'/args.run
    if root.exists():raise ValueError('Use a new run name')
    config=json.loads((ROOT/'configs/experiment_v2.json').read_text())
    policy=load_policy()
    assert_allowed_sources([config['training_source']],policy=policy)
    beta_files=sorted((ROOT/'data/raw/public/beta').glob('S*.mat'),key=lambda f:int(f.stem[1:]))
    beta_state=json.loads((ROOT/'logs/download_status.json').read_text())
    beta_files=[f for f in beta_files if beta_state.get('beta/'+f.name,{}).get('status')=='verified']
    if config['training_source']=='benchmark':
        status=json.loads((ROOT/'logs/benchmark_download_status.json').read_text())
        files=[ROOT/f'data/raw/public/benchmark/S{i}.mat' for i in config['benchmark_subjects']]
        if any(status.get('benchmark/'+f.name,{}).get('status')!='verified' for f in files):
            raise ValueError('All six pilot subjects must pass publisher checksum before training')
        reserved_beta=beta_files
    elif config['training_source']=='beta':
        files=beta_files[:config['max_beta_subjects']]
        if len(files)<5:raise ValueError('At least five verified BETA subjects required')
        reserved_beta=[] # BETA's separate holdout subjects are scored as public_holdout.
    else:raise ValueError('Unsupported public training source')
    train_files=[str(f.relative_to(ROOT)) for f in files]
    config['training_files']=train_files
    config['reserved_beta_files']=[str(f.relative_to(ROOT)) for f in reserved_beta]
    config['torch_version']=str(torch.__version__)
    train_hash={str(f.relative_to(ROOT)):fingerprint(f) for f in files}
    eval_files=reserved_beta+list((ROOT/'data/raw/competition').glob('*.csv'))
    eval_hash={str(f.relative_to(ROOT)):fingerprint(f) for f in eval_files}
    if set(train_hash.values())&set(eval_hash.values()):raise ValueError('Duplicate source file across training and evaluation')
    root.mkdir(parents=True);write(root/'resolved_config.json',config);write(root/'data_usage_policy.json',policy)
    write(root/'source_fingerprints.json',{'public_pool':train_hash,'evaluation_only':eval_hash})
    if config['training_source']=='beta':original=beta_trials(ROOT,train_files,config['beta_latency_seconds'])
    else:original=benchmark_trials(ROOT,train_files,config['benchmark_latency_seconds'])
    split=public_subject_split(original,config['split_seed'])
    write(root/'split_manifest.json',{k:snapshot_trials(v) for k,v in split.items()})
    role_files={stage:sorted({t.subject+'.mat' for t in trials}) for stage,trials in split.items()}
    if (set(role_files['train'])&set(role_files['dev']) or set(role_files['train'])&set(role_files['holdout']) or set(role_files['dev'])&set(role_files['holdout'])):
        raise ValueError('Subject file overlap between public roles')
    write(root/'subject_roles.json',role_files)
    arrays={k:make_windows(v,config['window_seconds'],config['stride_seconds'],k=='train') for k,v in split.items()}
    train_arrays={'train':arrays['train'],'val':arrays['dev']}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['cpu_threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # No existing v1 checkpoint is loaded. Never inspect reserved evaluation during fitting.
    all_models={};development={};baseline=CCA(filter_bank=True)
    baseline_dev=metrics(baseline.scores(arrays['dev'][0]),arrays['dev'][1],arrays['dev'][3])
    for seed in config['repeat_seeds']:
        path=root/f'seed-{seed}';path.mkdir();local=copy.deepcopy(config);local['seed']=seed
        models={}
        models['frequency_teacher']=train('frequency_teacher',local,train_arrays,path,device)
        models['student']=train('student',local,train_arrays,path,device)
        vx,vy,_,vs=arrays['dev']
        teacher_score=metrics(predict(models['frequency_teacher'],vx,device),vy,vs)
        student_score=metrics(predict(models['student'],vx,device),vy,vs)
        stronger=teacher_score['accuracy']>student_score['accuracy']
        if stronger:
            models['distilled']=train('distilled',local,train_arrays,path,device,models['frequency_teacher'])
        development[str(seed)]={name:metrics(predict(model,vx,device),vy,vs) for name,model in models.items()}
        development[str(seed)]['teacher_gate']={'passed':stronger,'criterion':'teacher development accuracy strictly exceeds direct student'}
        write(path/'development.json',development[str(seed)])
        options={name:development[str(seed)][name]['accuracy'] for name in models}
        options['fbcca']=baseline_dev['accuracy']
        selected=max(options,key=options.get)
        write(path/'deployment_policy.json',{'algorithm':selected,
            'selected_by':config['training_source']+' development subject accuracy only',
            'window_seconds':config['window_seconds'],'sampling_rate_hz':250,
            'channels':['CP3','CPZ','CP4','PO3','POZ','PO4'],'frequencies_hz':list(range(8,16)),
            'causal_filter':False,'calibration_weight':0.,'allow_support_calibration':False})
        all_models[seed]=models
    write(root/'all_models_frozen.json',{'complete':True,'development':development,
         'evaluation_parameters_frozen':True,'reserved_support_calibration':False})
    # Only now load original competition signals and reserved BETA labels for reporting.
    reserved=competition_trials(ROOT)+beta_trials(ROOT,config['reserved_beta_files'],config['beta_latency_seconds'])
    evaluation=make_windows(reserved,config['window_seconds'],training=False)
    write(root/'reserved_trial_manifest.json',snapshot_trials(reserved))
    ex,ey,eids,es=evaluation;hx,hy,hids,hs=arrays['holdout'];results={}
    for seed,models in all_models.items():
        row={};path=root/f'seed-{seed}'
        for name,model in models.items():
            scores=predict(model,ex,device);holdout=predict(model,hx,device)
            row[name]={'reserved_evaluation':metrics(scores,ey,es),'public_holdout':metrics(holdout,hy,hs),
                       'development':development[str(seed)][name],
                       'parameters':sum(p.numel() for p in model.parameters()),
                       'latency':latency(model,device,round(config['window_seconds']*250))}
            comp=np.array(es)=='competition'
            for file in ['D1_trials','D2_trials']:
                mask=np.array([uid.startswith(file+'/') for uid in eids])
                row[name][file]=metrics(scores[mask],ey[mask],['competition']*int(mask.sum()))
            np.savez_compressed(path/f'{name}_reserved_predictions.npz',trial_ids=np.array(eids),scores=scores,true_labels=ey)
        results[str(seed)]=row
    results['fbcca']={'development':baseline_dev,'reserved_evaluation':metrics(baseline.scores(ex),ey,es),
                      'public_holdout':metrics(baseline.scores(hx),hy,hs)}
    write(root/'results.json',results)
    names=sorted({name for seed,row in results.items() if seed!='fbcca' for name in row})
    summary={}
    for name in names:
        values=[row[name]['reserved_evaluation']['competition']['accuracy'] for seed,row in results.items() if seed!='fbcca' and name in row]
        summary[name]={'competition_mean_accuracy':float(np.mean(values)),
                       'competition_std_accuracy':float(np.std(values)), 'seeds_evaluated':len(values)}
    summary['fbcca']=results['fbcca']['reserved_evaluation']
    write(root/'summary.json',summary)
    write(root/'completed.json',{'complete':True,'note':'Reserved data used for evaluation only; repeated historical exposure prevents calling it a pristine external test.'})
    print('ROUND2_COMPLETE',root,flush=True)


if __name__=='__main__':main()
