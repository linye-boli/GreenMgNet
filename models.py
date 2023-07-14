import copy 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layers import SpectralConv1d
from utils import DenseNet, ZeroLayer, BandDiagLayer1d, MultiLevelLayer1d

class FNO1d(nn.Module):
    def __init__(self, modes, width, clevel=0, mlevel=0, nblocks=4):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.clevel = clevel 
        self.mlevel = mlevel
        self.nblocks = nblocks

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = SpectralConv1d(self.width, self.width, self.modes1)

        local_correction = MultiLevelLayer1d(self.width, mlevel)

        self.local_corrections = nn.ModuleList([copy.deepcopy(local_correction) for _ in range(nblocks)])
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint) in enumerate(zip(self.local_corrections, self.kernel_integrals)):

            # local correction
            x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevel != 0:
                x2 = kint(x[:,::2**self.clevel])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)
            
            # nonlinear 
            x = F.relu(x1 + x2)

        x = self.q(x)
        return x

if __name__ == '__main__':
    # inputs :
    bsz = 5 
    seq_len = 512 
    width = 64
    x = torch.rand((bsz, seq_len, 2))
    print('x : ', x.shape)

    # fno cfg:
    modes = 16
    res = ['null', 'diag', 'band']
    clevel = [0, 1, 2, 3]

    for r in res:
        for c in clevel:
            bw = 3 if r == 'band' else 0
            print('------- spec_conv -------')
            print('-'.join([r, str(c), str(bw)]))

            fno1d = FNO1d(modes, width, res=r, clevel=c, bw=3)
            y = fno1d(x)
            
            print('y : ', y.shape)
