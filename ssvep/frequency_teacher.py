"""Class-shared harmonic scoring teacher with electrode attention.

The scorer is shared over candidate frequencies: it learns spectral evidence,
not a different memorized waveform/phase template for each class.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FrequencyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('bins',torch.tensor([[4*f*h for h in (1,2,3)] for f in range(8,16)]))
        self.register_buffer('background_offsets',torch.tensor([-10,-8,-6,6,8,10]))
        self.encoder=nn.Sequential(nn.Linear(9,64),nn.LayerNorm(64),nn.ELU(),
                                   nn.Linear(64,64),nn.ELU())
        self.electrode_attention=nn.Sequential(nn.Linear(64,16),nn.Tanh(),nn.Linear(16,1))
        self.scorer=nn.Sequential(nn.Linear(128,128),nn.ELU(),nn.Dropout(.2),
                                  nn.Linear(128,32),nn.ELU(),nn.Linear(32,1))
        self.harmonic_weight=nn.Parameter(torch.zeros(3))

    def forward(self,x,return_features=False):
        if x.ndim!=3 or x.shape[1]!=6 or not 125<=x.shape[-1]<=1000:
            raise ValueError('Expected [batch,6,time], 0.5 to 4 seconds at 250 Hz')
        taper=torch.hann_window(x.shape[-1],dtype=x.dtype,device=x.device)
        spectrum=torch.fft.rfft(x*taper,n=1000,dim=-1)
        power=spectrum.real.square()+spectrum.imag.square()
        # [batch, channel, candidate frequency, harmonic]
        peak=power[:,:,self.bins]
        narrow=(peak+power[:,:,self.bins-1]+power[:,:,self.bins+1])/3
        wide=(narrow*3+power[:,:,self.bins-2]+power[:,:,self.bins+2])/5
        background=power[:,:,self.bins[:,:,None]+self.background_offsets].mean(dim=-1).clamp_min(1e-6)
        ratios=[torch.log((energy+1e-6)/background).clamp(-6,8) for energy in [peak,narrow,wide]]
        feature=torch.cat(ratios,dim=-1).permute(0,2,1,3)
        encoded=self.encoder(feature)
        attention=torch.softmax(self.electrode_attention(encoded),dim=2)
        representation=torch.cat([(attention*encoded).sum(dim=2),encoded.max(dim=2).values],dim=-1)
        harmonic=F.softplus(self.harmonic_weight)
        prior=(ratios[0].mean(dim=1)*harmonic).sum(dim=-1)/harmonic.sum()
        logits=prior+.1*self.scorer(representation).squeeze(-1)
        return (logits,representation.flatten(1)) if return_features else logits
