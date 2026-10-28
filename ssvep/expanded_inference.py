"""Frozen v6/v7 offline inference for organizer-style six-channel NPZ examples.

The output CSV is an internal development artifact, not the organizer Excel template.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baselines import CCA
from .data import CHANNELS, preprocess_window
from .models import EEGNetStudent, EEGNetStudentLongKernel


def read_examples(path, samples):
    with np.load(path, allow_pickle=False) as data:
        required={'X','channels','sampling_rate_hz','task_id'}
        if not required.issubset(data.files):
            raise ValueError(f'Missing input metadata: {sorted(required-set(data.files))}')
        if int(data['sampling_rate_hz'])!=250 or data['channels'].tolist()!=CHANNELS:
            raise ValueError('Sampling rate or electrode order differs from this trained protocol')
        raw=data['X']
        ids=data['task_id'].copy()
    if raw.ndim!=3 or raw.shape[1]!=6 or raw.shape[0]!=len(ids):
        raise ValueError('Expected [trial,6,time] and one task ID per trial')
    if len(np.unique(ids))!=len(ids):
        raise ValueError('Task IDs must be unique')
    if raw.shape[-1]<samples:
        raise ValueError('Input shorter than trained window; no padding or silent alternative window')
    # No access to stim_id or other label columns. Match v6/v7 training exactly.
    x=np.stack([preprocess_window(t,samples) for t in raw])
    return x,ids


def infer(run, input_path, algorithm, seed=None):
    config_path=run/'preregistered_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    frozen=json.loads((run/'models_frozen.json').read_text(encoding='utf-8'))
    if not frozen['complete']:
        raise ValueError('Run models are not frozen')
    x,ids=read_examples(input_path,config['samples'])
    if algorithm=='fbcca':
        scores=CCA(filter_bank=True).scores(x)
    else:
        if seed not in config['seeds']:
            raise ValueError('Choose an existing trained seed')
        if run.name.startswith('v6-'):
            name=f'eegnet_26_{seed}'
            model_type=EEGNetStudent
        elif run.name.startswith('v7-'):
            name=f'eegnet_26_more_updates_{seed}'
            model_type=EEGNetStudent
        elif run.name.startswith('v8-'):
            name=f'eegnet_long_{seed}'
            model_type=EEGNetStudentLongKernel
        else:
            raise ValueError('Only v6/v7/v8 frozen runs are supported')
        saved=torch.load(run/(name+'.pt'),map_location='cpu',weights_only=True)
        if saved['config_sha256']!=frozen['config_sha256']:
            raise ValueError('Checkpoint/config fingerprint mismatch')
        model=model_type().eval()
        model.load_state_dict(saved['state_dict'])
        with torch.no_grad():
            scores=model(torch.from_numpy(x)).numpy()
    return ids,np.asarray(scores).argmax(axis=1)


def main():
    p=argparse.ArgumentParser(description='Internal development CSV; not the official Excel result template')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',choices=['fbcca','eegnet'],required=True)
    p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        ids,predictions=infer(args.run,args.input,args.algorithm,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['taskID','predicted_stimID'])
        writer.writerows(zip(ids,predictions))
    print('Wrote',len(ids),'development predictions:',args.output)

if __name__=='__main__':main()
