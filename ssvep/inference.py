"""Predict existing samples without opening query-label fields.

    Input NPZ: X [trial,6,time], channels, sampling_rate_hz, task_id.
    Optional support NPZ additionally contains stim_id, exactly one per class.
    Formal competition format requires a future adapter using organizer metadata.
"""
import argparse
import csv
import json
import inspect
from pathlib import Path

import numpy as np
import torch

from .baselines import CCA,reference_template_scores
from .data import CHANNELS,preprocess
from .models import build_model
from .experiment import predict,scale_rows


def read_signal(path,window,causal,support=False):
    with np.load(path,allow_pickle=False) as data:
        if 'sampling_rate_hz' not in data:
            raise ValueError('Input must explicitly state sampling_rate_hz; do not infer it from length')
        if int(data['sampling_rate_hz'])!=250 or data['channels'].tolist()!=CHANNELS:
            raise ValueError('Sampling rate or electrode order differs from trained model')
        raw=data['X'];ids=data['task_id'].copy()
        if raw.ndim!=3 or len(ids)!=len(raw) or len(set(ids.tolist()))!=len(ids):
            raise ValueError('Expected unique task IDs and [trial,channel,time] tensor')
        n=round(window*250)
        if raw.shape[-1]<n:raise ValueError('Input shorter than trained window')
        x=np.stack([preprocess(t,causal=causal)[:,:n] for t in raw])
        labels=data['stim_id'].copy() if support else None
        return x,ids,labels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--support',type=Path)
    p.add_argument('--algorithm',choices=['teacher','frequency_teacher','student','distilled','guided_student','hybrid_teacher','cca','fbcca'])
    args=p.parse_args()
    policy=json.loads((args.run/'deployment_policy.json').read_text())
    algorithm=args.algorithm or policy['algorithm']
    x,ids,_=read_signal(args.input,policy['window_seconds'],policy['causal_filter'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=None
    if algorithm in ['teacher','frequency_teacher','student','distilled','guided_student','hybrid_teacher']:
        neural_name='teacher' if algorithm=='hybrid_teacher' else algorithm
        model=build_model(neural_name).to(device)
        kwargs={'map_location':device}
        if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
        # Older validated local PyTorch: load only this project's own training checkpoints.
        checkpoint=torch.load(args.run/f'{neural_name}.pt',**kwargs)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        score=predict(model,x,device)
        if algorithm=='hybrid_teacher':
            w=policy['teacher_fbcca_weight']
            score=policy['teacher_score_scale']*((1-w)*scale_rows(score)+w*scale_rows(CCA(filter_bank=True).scores(x)))
    else:score=CCA(filter_bank=algorithm=='fbcca').scores(x)
    if args.support:
        if policy.get('allow_support_calibration') is False:
            raise ValueError('This experiment reserves current data for evaluation; support calibration is disabled')
        if args.support.resolve()==args.input.resolve():raise ValueError('Support and query inputs must differ')
        sx,sids,sy=read_signal(args.support,policy['window_seconds'],policy['causal_filter'],True)
        if len(sy)!=8 or sorted(sy.tolist())!=list(range(8)):
            raise ValueError('Provide exactly one labeled reference per class')
        if set(ids.tolist())&set(sids.tolist()):
            raise ValueError('Support/query task IDs overlap; preserve globally unique record identities')
        if model is None:cal=reference_template_scores(sx,sy,x)
        else:
            sf=predict(model,sx,device,features=True)[np.argsort(sy)]
            qf=predict(model,x,device,features=True)
            center=sf.mean(axis=0,keepdims=True);sf=sf-center;qf=qf-center
            sf/=np.maximum(np.linalg.norm(sf,axis=1,keepdims=True),1e-8)
            qf/=np.maximum(np.linalg.norm(qf,axis=1,keepdims=True),1e-8)
            cal=qf@sf.T
        results=json.loads((args.run/'results.json').read_text())
        w=results[algorithm]['calibration_weight_selected_on_validation']
        score=(1-w)*scale_rows(score)+w*scale_rows(cal)
    if args.output.exists():raise ValueError('Refusing to overwrite existing predictions')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['taskID','predicted_stimID'])
        writer.writerows(zip(ids.tolist(),score.argmax(axis=1).tolist()))
    print('Predicted',len(ids),'trials with',algorithm,'; CSV is a development output, not organizer result template.')


if __name__=='__main__':main()
