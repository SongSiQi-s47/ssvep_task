"""EEGNet-inspired student and original multiscale attention teacher."""
import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels,max(4,channels//4)), nn.ELU(),
                                  nn.Linear(max(4,channels//4),channels),nn.Sigmoid())
    def forward(self,x):
        return x * self.gate(x.mean(dim=(-2,-1)))[:,:,None,None]


class HarmonicFeatures(nn.Module):
    """Log power near target harmonics, using physical frequency, not label indices."""
    def __init__(self):
        super().__init__()
        bins = torch.tensor([[round(4*f*h) for h in (1,2,3)] for f in range(8,16)])
        self.register_buffer('bins',bins)
    def forward(self,x):
        taper = torch.hann_window(x.shape[-1],device=x.device,dtype=x.dtype)
        spectrum=torch.fft.rfft(x*taper,n=1000,dim=-1)
        # Identical |FFT|^2, without old CUDA's runtime-compiled complex abs kernel.
        power=(spectrum.real.square()+spectrum.imag.square()).mean(dim=1)
        energy = power[:,self.bins]
        # Relative spectrum removes recording amplitude as a shortcut.
        return torch.log1p(energy / power[:,24:241].mean(dim=-1,keepdim=True).clamp_min(1e-6)[:,:,None])


class EEGNetStudent(nn.Module):
    """Temporal -> depthwise spatial -> separable temporal; light channel attention.

    Uses adaptive temporal pooling and an SSVEP harmonic head, so it is an EEGNet
    adaptation rather than a claim of reproducing the original paper exactly.
    """
    def __init__(self, channels=6, classes=8, f1=8, depth=2, dropout=.25):
        super().__init__()
        f2 = f1*depth
        self.temporal = nn.Sequential(nn.Conv2d(1,f1,(1,63),padding=(0,31),bias=False),nn.BatchNorm2d(f1))
        self.spatial = nn.Sequential(nn.Conv2d(f1,f2,(channels,1),groups=f1,bias=False),
                                     nn.BatchNorm2d(f2),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(dropout))
        self.separable = nn.Sequential(nn.Conv2d(f2,f2,(1,15),padding=(0,7),groups=f2,bias=False),
                                       nn.Conv2d(f2,f2,1,bias=False),nn.BatchNorm2d(f2),nn.ELU(),
                                       ChannelAttention(f2),nn.AvgPool2d((1,4)),nn.Dropout(dropout))
        self.pool = nn.AdaptiveAvgPool2d((1,8))
        self.harmonics = HarmonicFeatures()
        self.feature = nn.Sequential(nn.Linear(f2*8+24,32),nn.ELU(),nn.Dropout(dropout))
        self.classifier = nn.Linear(32,classes)
    def forward(self,x,return_features=False):
        t=self.pool(self.separable(self.spatial(self.temporal(x[:,None])))).flatten(1)
        z=self.feature(torch.cat([t,self.harmonics(x).flatten(1)],dim=1))
        logits=self.classifier(z)
        return (logits,z) if return_features else logits


class MultiScaleAttentionTeacher(nn.Module):
    def __init__(self, channels=6, classes=8):
        super().__init__()
        self.branches=nn.ModuleList([nn.Sequential(
            nn.Conv2d(1,16,(1,k),padding=(0,k//2),bias=False),nn.BatchNorm2d(16),
            nn.Conv2d(16,32,(channels,1),groups=16,bias=False),nn.BatchNorm2d(32),nn.ELU(),
            nn.AvgPool2d((1,4))) for k in (15,31,63)])
        self.fusion=nn.Sequential(nn.Conv2d(96,64,1,bias=False),nn.BatchNorm2d(64),nn.ELU(),
                                  ChannelAttention(64),nn.Conv2d(64,64,(1,15),padding=(0,7),bias=False),
                                  nn.BatchNorm2d(64),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.35))
        self.pool=nn.AdaptiveAvgPool2d((1,8))
        self.harmonics=HarmonicFeatures()
        self.feature=nn.Sequential(nn.Linear(64*8+24,128),nn.ELU(),nn.Dropout(.35))
        self.classifier=nn.Linear(128,classes)
    def forward(self,x,return_features=False):
        t=self.pool(self.fusion(torch.cat([b(x[:,None]) for b in self.branches],dim=1))).flatten(1)
        z=self.feature(torch.cat([t,self.harmonics(x).flatten(1)],dim=1))
        logits=self.classifier(z)
        return (logits,z) if return_features else logits


def distillation_loss(student_logits,teacher_logits,labels,temperature=3.,alpha=.5):
    if temperature<=0 or not 0<=alpha<=1:
        raise ValueError('Require temperature > 0 and alpha in [0,1]')
    hard=F.cross_entropy(student_logits,labels)
    soft=F.kl_div(F.log_softmax(student_logits/temperature,dim=1),
                 F.softmax(teacher_logits.detach()/temperature,dim=1),reduction='batchmean')*temperature**2
    return (1-alpha)*hard+alpha*soft


def build_model(name):
    if name=='frequency_teacher':
        from .frequency_teacher import FrequencyTeacher
        return FrequencyTeacher()
    if name=='teacher':return MultiScaleAttentionTeacher()
    if name in ('student','distilled','guided_student'):return EEGNetStudent()
    raise ValueError(name)
