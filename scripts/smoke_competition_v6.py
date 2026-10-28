"""Validate organizer CSV -> trials -> both frozen algorithm input/output paths."""
import json
from pathlib import Path

import numpy as np
import torch

from ssvep.baselines import CCA
from ssvep.data import competition_trials, fingerprint, preprocess_window
from ssvep.models import EEGNetStudent

ROOT=Path(__file__).resolve().parents[1]


def main():
    records=competition_trials(ROOT)
    checkpoint=ROOT/'runs/v5-expanded-eegnet/eegnet_fourteen_2026.pt'
    model=EEGNetStudent().eval()
    model_source='fresh architecture only'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'])
        model_source='v5 frozen EEGNet checkpoint'
    result={}
    for stem in ('D1','D2'):
        trials=[t for t in records if t.subject==stem+'_trials']
        raw=np.stack([t.x for t in trials])
        x=np.stack([preprocess_window(t.x,375) for t in trials])
        fb=CCA(filter_bank=True).scores(x)
        with torch.no_grad(): eegnet=model(torch.from_numpy(x)).numpy()
        labels=np.bincount([t.y for t in trials],minlength=8)
        assert raw.shape==(48,6,1000) and x.shape==(48,6,375)
        assert fb.shape==eegnet.shape==(48,8)
        assert (labels==6).all()
        assert np.isfinite(raw).all() and np.isfinite(x).all()
        assert np.isfinite(fb).all() and np.isfinite(eegnet).all()
        result[stem]={'model_source':model_source,'source_sha256':fingerprint(ROOT/f'data/raw/competition/{stem}.csv'),
                      'trials':48,'raw_shape':list(raw.shape),'model_input_shape':list(x.shape),
                      'fbcca_output_shape':list(fb.shape),'eegnet_output_shape':list(eegnet.shape),
                      'finite':True,'class_counts':labels.tolist()}
    out=ROOT/'runs/v6-input-smoke.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(out)

if __name__=='__main__': main()
