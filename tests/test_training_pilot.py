import unittest
import numpy as np
import torch

from ssvep.data import Trial
from ssvep.training_pilot import TrialCrops,augmentation,cross_subject_contrastive


class TrainingPilot(unittest.TestCase):
    def test_balancing_removes_duration_based_subject_weight(self):
        rng=np.random.default_rng(3)
        trials=[Trial(rng.normal(size=(6,n)),i,'beta',f'S{i}',f'trial{i}',0)
                for i,n in enumerate([500,750])]
        fixed=TrialCrops(trials,'enumerated',1)
        random=TrialCrops(trials,'random',1)
        self.assertEqual([fixed.subjects.count(s) for s in ['S0','S1']],[3,5])
        mass=[sum(w for w,s in zip(random.weights,random.subjects) if s==subject) for subject in ['S0','S1']]
        self.assertEqual(mass[0],mass[1])
        draws=[random[1][0].numpy().tobytes() for _ in range(10)]
        self.assertGreater(len(set(draws)),1)

    def test_augmentation_preserves_frequency_and_channel_alignment(self):
        torch.manual_seed(4)
        t=torch.arange(250)/250
        x=torch.sin(2*torch.pi*11*t)[None,None,:].repeat(8,6,1)
        z=augmentation(x,True)
        spectrum=torch.fft.rfft(z,dim=-1)
        power=spectrum.real.square()+spectrum.imag.square()
        self.assertTrue((power.argmax(-1)==11).all())
        zn=z-z.mean(-1,keepdim=True)
        zn=zn/torch.linalg.norm(zn,dim=-1,keepdim=True)
        self.assertTrue(((zn[:,0,:]*zn[:,1,:]).sum(-1)>.99).all())

    def test_curriculum_preserves_sampling_mass_and_valid_window_bounds(self):
        trials=[Trial(np.random.default_rng(3).normal(size=(6,500)),0,'beta','S1','trial0',0)]
        dataset=TrialCrops(trials,'enumerated',1)
        original=len(dataset);dataset.current_n=375
        for i in range(len(dataset)):self.assertEqual(dataset[i][0].shape,(6,375))
        self.assertEqual(len(dataset),original)
        dataset.current_n=250
        for i in range(len(dataset)):self.assertEqual(dataset[i][0].shape,(6,250))

    def test_cross_subject_objective_prefers_matching_class_and_handles_no_pairs(self):
        labels=torch.tensor([0,0,1,1]);subjects=torch.tensor([0,1,0,1])
        good=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True)
        bad=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
        loss=cross_subject_contrastive(good,labels,subjects)
        self.assertLess(loss.item(),cross_subject_contrastive(bad,labels,subjects).item())
        loss.backward();self.assertTrue(torch.isfinite(good.grad).all())
        none=cross_subject_contrastive(good,labels,torch.zeros(4,dtype=torch.long))
        self.assertEqual(none.item(),0.)


if __name__=='__main__':unittest.main()
