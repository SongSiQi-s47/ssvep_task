"""Distill a development-calibrated FBCCA teacher using BETA training subjects only."""
import argparse
import copy
import json
import shutil

import numpy as np
import torch
from scipy.special import logsumexp

from .baselines import CCA
from .data import beta_trials,competition_trials,make_windows,fingerprint
from .policy import public_subject_split,load_policy,assert_allowed_sources
from .experiment import ROOT,write,metrics,train,predict,latency


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-run',required=True);p.add_argument('--run',required=True)
    args=p.parse_args();base=ROOT/'runs'/args.base_run;run=ROOT/'runs'/args.run
    if run.exists():raise ValueError('Use a new run name')
    if not (base/'completed.json').exists():raise ValueError('Base experiment must complete')
    config=json.loads((base/'resolved_config.json').read_text())
    if config['training_source']!='beta':raise ValueError('This experiment requires BETA-only public training')
    assert_allowed_sources(['beta'])
    frozen=json.loads((base/'source_fingerprints.json').read_text())
    for source,digest in frozen['public_pool'].items():
        if fingerprint(ROOT/source)!=digest:raise ValueError('Public data changed since base run')
    original=beta_trials(ROOT,config['training_files'],config['beta_latency_seconds'])
    split=public_subject_split(original,config['split_seed'])
    expected=json.loads((base/'split_manifest.json').read_text())
    for stage,trials in split.items():
        if [t.uid for t in trials]!=[t['uid'] for t in expected[stage]]:
            raise ValueError('Original subject/trial split changed')
    arrays={k:make_windows(v,config['window_seconds'],config['stride_seconds'],k=='train') for k,v in split.items()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['cpu_threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run.mkdir(parents=True)
    for name in ['split_manifest.json','subject_roles.json','source_fingerprints.json']:
        shutil.copy2(base/name,run/name)
    write(run/'data_usage_policy.json',load_policy())
    teacher=CCA(filter_bank=True)
    vx,vy,_,vs=arrays['dev'];dev=teacher.scores(vx)
    dev=dev-dev.mean(axis=1,keepdims=True)
    # Global score scale calibrated by public development NLL, not competition ACC.
    choices=[1.,2.,4.,8.,16.,32.]
    nll={str(scale):float(np.mean(logsumexp(dev*scale,axis=1)-(dev*scale)[np.arange(len(vy)),vy])) for scale in choices}
    scale=min(choices,key=lambda k:nll[str(k)])
    teacher_accuracy=metrics(dev,vy,vs)['accuracy']
    config['teacher_type']='FBCCA algorithm scores with public-development NLL calibration'
    config['teacher_score_scale']=scale;config['teacher_scale_nll_candidates']=nll
    config['base_run']=args.base_run
    write(run/'resolved_config.json',config)
    print('ALGORITHM_TEACHER','dev_accuracy',teacher_accuracy,'score_scale',scale,flush=True)
    tx=arrays['train'][0];targets=teacher.scores(tx).astype(np.float32)
    targets=(targets-targets.mean(axis=1,keepdims=True))*scale
    np.savez_compressed(run/'training_soft_targets.npz',scores=targets,trial_ids=np.array(arrays['train'][2]))
    models={};gates={}
    for seed in config['repeat_seeds']:
        prior=json.loads((base/f'seed-{seed}'/'development.json').read_text())
        direct=prior['student']['accuracy'];passed=teacher_accuracy>direct
        gates[str(seed)]={'teacher_development_accuracy':teacher_accuracy,'direct_student_development_accuracy':direct,'passed':passed}
        if not passed:continue
        path=run/f'seed-{seed}';path.mkdir();local=copy.deepcopy(config);local['seed']=seed
        models[seed]=train('distilled',local,{'train':arrays['train'],'val':arrays['dev']},path,device,soft_targets=targets)
        development=metrics(predict(models[seed],vx,device),vy,vs)
        write(path/'development.json',{'distilled':development,'teacher_gate':gates[str(seed)]})
        # Selection depends entirely on BETA development, never on D1/D2 results.
        candidates={'distilled':development['accuracy'],'student':direct,'fbcca':teacher_accuracy}
        selected=max(candidates,key=candidates.get)
        shutil.copy2(base/f'seed-{seed}'/'student.pt',path/'student.pt')
        write(path/'deployment_policy.json',{'algorithm':selected,'selected_by':'BETA development subject accuracy only',
            'window_seconds':config['window_seconds'],'sampling_rate_hz':250,'causal_filter':False,
            'channels':['CP3','CPZ','CP4','PO3','POZ','PO4'],'frequencies_hz':list(range(8,16)),
            'calibration_weight':0.,'allow_support_calibration':False})
    write(run/'all_models_frozen.json',{'complete':True,'teacher_gates':gates,'reserved_support_calibration':False})
    # Open competition data only after training and all selection decisions finish.
    evaluation=make_windows(competition_trials(ROOT),config['window_seconds'],training=False)
    ex,ey,eids,es=evaluation;hx,hy,_,hs=arrays['holdout']
    results={}
    for seed,model in models.items():
        scores=predict(model,ex,device)
        row={'reserved_evaluation':metrics(scores,ey,es),
             'public_holdout':metrics(predict(model,hx,device),hy,hs),
             'development':metrics(predict(model,vx,device),vy,vs),
             'parameters':sum(p.numel() for p in model.parameters()),
             'latency':latency(model,device,round(config['window_seconds']*250))}
        for file in ['D1_trials','D2_trials']:
            mask=np.array([uid.startswith(file+'/') for uid in eids])
            row[file]=metrics(scores[mask],ey[mask],['competition']*int(mask.sum()))
        results[str(seed)]=row
        np.savez_compressed(run/f'seed-{seed}'/'distilled_reserved_predictions.npz',trial_ids=np.array(eids),true_labels=ey,scores=scores)
    write(run/'results.json',results)
    values=[v['reserved_evaluation']['accuracy'] for v in results.values()]
    write(run/'summary.json',{'competition_mean_accuracy':float(np.mean(values)) if values else None,
        'competition_std_accuracy':float(np.std(values)) if values else None,'seeds_evaluated':len(values),
        'note':'All training and selection uses BETA only; paired comparisons use the same split and initialization seeds.'})
    write(run/'completed.json',{'complete':True,'run':args.run})
    print('ALGORITHM_DISTILL_COMPLETE',run,flush=True)


if __name__=='__main__':main()
