"""Validation-selected FBCCA/CNN teacher; reuse original held-out split."""
import argparse
import inspect
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .baselines import CCA
from .data import competition_trials,beta_trials,split_trials,make_windows
from .models import build_model
from .experiment import ROOT,write,metrics,predict,scale_rows,train,calibrated_scores,latency


def load_model(run,name,device):
    model=build_model(name).to(device)
    kwargs={'map_location':device}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    model.load_state_dict(torch.load(run/f'{name}.pt',**kwargs)['state_dict']);model.eval()
    return model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-run',required=True)
    parser.add_argument('--run',required=True);args=parser.parse_args()
    base=ROOT/'runs'/args.base_run;run=ROOT/'runs'/args.run
    if run.exists():raise ValueError('Choose a new experiment name')
    if not (base/'completed.json').exists():raise ValueError('Base experiment must be complete')
    run.mkdir(parents=True)
    config=json.loads((base/'resolved_config.json').read_text())
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(config['device'])
    data=competition_trials(ROOT)+beta_trials(ROOT,config['beta_files'],config['beta_latency_seconds'])
    split=split_trials(data,config['seed'])
    previous=json.loads((base/'split_manifest.json').read_text())
    for key,trials in split.items():
        if [t.uid for t in trials]!=[t['uid'] for t in previous[key]]:
            raise ValueError('Original trial split changed')
    for name in ['split_manifest.json','source_fingerprints.json','teacher.pt','student.pt','distilled.pt']:
        shutil.copy2(base/name,run/name)
    arrays={k:make_windows(v,config['window_seconds'],config['stride_seconds'],k=='train',config['causal_filter'])
            for k,v in split.items()}
    teacher=load_model(base,'teacher',device);fbcca=CCA(filter_bank=True)
    vx,vy,_,vs=arrays['val'];cnn_val=predict(teacher,vx,device);fb_val=fbcca.scores(vx)
    weights=[0.,.25,.5,.75,1.]
    def fuse(cnn,fb,w):return 4*((1-w)*scale_rows(cnn)+w*scale_rows(fb))
    candidates={str(w):metrics(fuse(cnn_val,fb_val,w),vy,vs)['mean_domain_accuracy'] for w in weights}
    w=max(weights,key=lambda k:candidates[str(k)])
    guided_teacher_val=fuse(cnn_val,fb_val,w)
    original=json.loads((base/'results.json').read_text())
    stronger=metrics(guided_teacher_val,vy,vs)['mean_domain_accuracy']>original['student']['validation']['mean_domain_accuracy']
    config['teacher_fbcca_weight']=w;config['teacher_score_scale']=4.
    config['base_run']=args.base_run;config['teacher_validation_weight_candidates']=candidates
    write(run/'resolved_config.json',config)
    print('HYBRID_TEACHER',w,candidates,'STRONGER_THAN_STUDENT',stronger,flush=True)
    if not stronger:
        write(run/'completed.json',{'complete':False,'reason':'Teacher does not exceed direct student validation; distillation stopped'})
        return
    tx=arrays['train'][0]
    targets=fuse(predict(teacher,tx,device),fbcca.scores(tx),w).astype(np.float32)
    np.savez_compressed(run/'training_soft_targets.npz',trial_ids=np.array(arrays['train'][2]),scores=targets)
    # Cached target refers to original training crop; the student's tiny noise adds consistency training.
    guided=train('guided_student',config,arrays,run,device,soft_targets=targets)
    results=original.copy();support={};testx,testy,_,tests=arrays['test']
    for name,model,validation,test_score in [
        ('hybrid_teacher',teacher,guided_teacher_val,fuse(predict(teacher,testx,device),fbcca.scores(testx),w)),
        ('guided_student',guided,predict(guided,vx,device),predict(guided,testx,device))]:
        qv,bv,cv,sv=calibrated_scores(split['val'],arrays['val'],validation,model,device)
        choice=[0.,.25,.5,1.]
        values={str(k):metrics((1-k)*bv+k*cv,vy[qv],[vs[i] for i in qv])['mean_domain_accuracy'] for k in choice}
        k=max(choice,key=lambda a:values[str(a)])
        qt,bt,ct,st=calibrated_scores(split['test'],arrays['test'],test_score,model,device)
        results[name]={'validation':metrics(validation,vy,vs),
            'calibrated_validation_query':metrics((1-k)*bv+k*cv,vy[qv],[vs[i] for i in qv]),
            'zero_shot_all_test':metrics(test_score,testy,tests),
            'zero_shot_query_only':metrics(bt,testy[qt],[tests[i] for i in qt]),
            'calibrated_query_only':metrics((1-k)*bt+k*ct,testy[qt],[tests[i] for i in qt]),
            'calibration_weight_selected_on_validation':k,'validation_weight_candidates':values}
        if name=='guided_student':
            results[name].update(parameters=sum(p.numel() for p in model.parameters()),
                                  checkpoint_bytes=(run/'guided_student.pt').stat().st_size,
                                  latency=latency(model,device,round(config['window_seconds']*250)))
        support[name]={'validation':sv,'test':st}
        np.savez_compressed(run/f'{name}_predictions.npz',trial_ids=np.array(arrays['test'][2]),
                            true_labels=testy,scores=test_score,query_indices=qt,calibrated_scores=(1-k)*bt+k*ct)
    write(run/'results.json',results);write(run/'calibration_support_manifest.json',support)
    selected=max(results,key=lambda name:results[name]['validation']['mean_domain_accuracy'])
    policy=json.loads((base/'deployment_policy.json').read_text())
    policy.update(algorithm=selected,calibration_weight=results[selected]['calibration_weight_selected_on_validation'],
                  teacher_fbcca_weight=w,teacher_score_scale=4.)
    write(run/'deployment_policy.json',policy)
    write(run/'assessment.json',{'teacher_stronger_on_validation':stronger,
        'distillation_gain_on_validation':results['guided_student']['validation']['mean_domain_accuracy']-original['student']['validation']['mean_domain_accuracy'],
        'refinement_selected_using':'validation only; all original test trials unchanged',
        'single_seed_exploratory':True,'hybrid_teacher_is_not_neural_only':True})
    write(run/'completed.json',{'complete':True,'run':args.run})
    print('COMPLETE',run,flush=True)


if __name__=='__main__':main()
