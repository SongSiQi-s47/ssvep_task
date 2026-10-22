"""Export the neural student chosen by BETA development accuracy only."""
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ssvep.models import build_model


def main():
    candidates=[]
    for run,model_name in [('v2-beta-subject-isolated','student'),('v2-beta-algorithm-distilled','distilled')]:
        for seed in [2026,2027,2028]:
            folder=ROOT/'runs'/run/f'seed-{seed}'
            dev=json.loads((folder/'development.json').read_text())
            score=dev[model_name]['accuracy']
            candidates.append((score,run,seed,model_name,folder))
    # Stable order resolves equal development scores without consulting evaluation labels.
    score,run,seed,name,folder=max(candidates,key=lambda x:x[0])
    kwargs={'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    checkpoint=torch.load(folder/f'{name}.pt',**kwargs)
    model=build_model(name);model.load_state_dict(checkpoint['state_dict']);model.eval()
    example=torch.randn(1,6,250)
    traced=torch.jit.trace(model,example)
    # Test save/load and batch handling on model inputs without examining any validation labels.
    output=folder/'student_inference.pt';traced.save(str(output));restored=torch.jit.load(str(output));restored.eval()
    with torch.no_grad():
        for batch in [1,8]:
            x=torch.randn(batch,6,250)
            np.testing.assert_allclose(model(x).numpy(),restored(x).numpy(),rtol=1e-5,atol=1e-5)
    deliver=Path(r'C:\Users\64443\Documents\Codex\2026-10-14\siunners-chatgpt-conversation-6ace5c53-2184-83e8\outputs')
    target=deliver/'SSVEP_v2_student_inference.pt';shutil.copy2(output,target)
    metadata={'chosen_by':'highest BETA development accuracy among direct/distilled candidates; never D1/D2',
              'run':run,'seed':seed,'model':name,'beta_development_accuracy':score,
              'parameters':sum(p.numel() for p in model.parameters()),'artifact_bytes':target.stat().st_size,
              'input':'float32 [batch,6,250] after documented offline bandpass and per-original-trial normalization',
              'channels':['CP3','CPZ','CP4','PO3','POZ','PO4'],'sample_rate_hz':250,
              'outputs':'eight logits, argmax maps 0..7 to 8..15 Hz',
              'export_device':'CPU; use original model/checkpoint for GPU inference',
              'validation':'CPU original/save-load outputs match for batch sizes 1 and 8',
              'warning':'Neural candidate export; the development-selected overall default remains FBCCA.'}
    (folder/'export_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (deliver/'SSVEP_v2_model_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
