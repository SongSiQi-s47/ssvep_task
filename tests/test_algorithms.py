import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from ssvep.baselines import CCA,canonical_correlation,reference_template_scores
from ssvep.data import Trial,split_trials,make_windows
from ssvep.models import EEGNetStudent,MultiScaleAttentionTeacher,distillation_loss
from ssvep.inference import read_signal


class Algorithms(unittest.TestCase):
    def test_cca_frequency_and_degenerate_signal(self):
        time=np.arange(500)/250
        x=np.stack([np.sin(2*np.pi*11*time+phase) for phase in np.linspace(0,1,6)])
        self.assertEqual(int(CCA().scores(x[None]).argmax()),3)
        self.assertEqual(int(CCA(filter_bank=True).scores(x[None]).argmax()),3)
        self.assertEqual(canonical_correlation(np.zeros((500,6)),np.zeros((500,6))),0)

    def test_group_split_before_windowing(self):
        rng=np.random.default_rng(2);trials=[]
        for file in ['D1_trials','D2_trials']:
            for y in range(8):
                for repeat in range(6):
                    trials.append(Trial(rng.normal(size=(6,1000)),y,'competition',file,f'{file}/{y}/{repeat}',repeat))
        for s in range(10):
            for y in range(8):
                trials.append(Trial(rng.normal(size=(6,500)),y,'beta',f'S{s}',f'beta/S{s}/{y}',0))
        split=split_trials(trials,1)
        groups={k:{t.subject for t in v if t.source=='beta'} for k,v in split.items()}
        self.assertFalse(groups['train']&groups['val']|groups['train']&groups['test']|groups['val']&groups['test'])
        windows={k:make_windows(v,1,training=k=='train') for k,v in split.items()}
        uid={k:set(v[2]) for k,v in windows.items()}
        self.assertFalse(uid['train']&uid['val']|uid['train']&uid['test'])

    def test_models_accept_variable_lengths_and_gradients(self):
        for model in [EEGNetStudent(),MultiScaleAttentionTeacher()]:
            for length in [125,250,500,1000]:
                logits=model(torch.randn(2,6,length))
                self.assertEqual(tuple(logits.shape),(2,8))
                self.assertTrue(torch.isfinite(logits).all())
            logits.sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_full_gpu_models_and_distillation_backward(self):
        x=torch.randn(4,6,250,device='cuda');labels=torch.tensor([0,1,2,3],device='cuda')
        teacher=MultiScaleAttentionTeacher().cuda().eval()
        student=EEGNetStudent().cuda()
        with torch.no_grad():target=teacher(x)
        logits=student(x)
        loss=distillation_loss(logits,target,labels)
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None))

    def test_distillation_detaches_teacher_and_scales_temperature(self):
        s=torch.randn(4,8,requires_grad=True);t=torch.randn(4,8,requires_grad=True)
        labels=torch.tensor([0,1,2,3])
        loss=distillation_loss(s,t,labels,temperature=3,alpha=.5)
        loss.backward();self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue(torch.allclose(distillation_loss(s,t,labels,alpha=0),torch.nn.functional.cross_entropy(s,labels)))

    def test_calibration_requires_all_classes(self):
        with self.assertRaises(ValueError):reference_template_scores(np.zeros((7,6,250)),np.arange(7),np.zeros((1,6,250)))

    def test_inference_ignores_query_labels_and_checks_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'query.npz'
            data={'X':np.random.default_rng(1).normal(size=(2,6,1000)),
                  'channels':np.array(['CP3','CPZ','CP4','PO3','POZ','PO4']),
                  'task_id':np.array([10,11]),'stim_id':np.array([-99,-99])}
            np.savez(path,**data)
            with self.assertRaises(ValueError):read_signal(path,1,False)
            np.savez(path,**data,sampling_rate_hz=np.array(250))
            x,ids,labels=read_signal(path,1,False)
            self.assertIsNone(labels);self.assertEqual(x.shape,(2,6,250))
            data['channels']=data['channels'][::-1]
            np.savez(path,**data,sampling_rate_hz=np.array(250))
            with self.assertRaises(ValueError):read_signal(path,1,False)


if __name__=='__main__':unittest.main()
