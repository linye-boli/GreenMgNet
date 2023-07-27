import copy 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layers import SpectralConv1d, LowRank1d, FourierAttention1d, GalerkinAttention1d
from utils import DenseNet, MultiLevelLayer1d

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

class LNO1d(nn.Module):
    def __init__(self, width, rank, clevel=0, mlevel=0, nblocks=4):
        super(LNO1d, self).__init__()
        self.width = width
        self.rank = rank
        self.clevel = clevel 
        self.mlevel = mlevel
        self.nblocks = nblocks

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = LowRank1d(self.width, self.rank)
        layer_norm = nn.LayerNorm(self.width)
        local_correction = MultiLevelLayer1d(self.width, mlevel)

        self.local_corrections = nn.ModuleList([copy.deepcopy(local_correction) for _ in range(nblocks)])
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])
        self.layer_norms = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        a = x.clone()
        x = self.p(x)

        for i, (lc, kint, ln) in enumerate(zip(self.local_corrections, self.kernel_integrals, self.layer_norms)):

            # local correction
            x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevel != 0:
                x2 = kint(x[:,::2**self.clevel], a[:,::2**self.clevel])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x, a)
            
            x = ln(x1+x2)
            # nonlinear 
            if i != self.nblocks-1:
                x = F.relu(x)


        x = self.q(x)
        return x

class FT1d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4):
        super(FT1d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.clevel = clevel 
        self.mlevel = mlevel
        self.nblocks = nblocks

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = FourierAttention1d(self.width, self.width, self.nhead)
        layer_norm = nn.LayerNorm(self.width)
        local_correction = MultiLevelLayer1d(self.width, mlevel)
        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)

        self.local_corrections = nn.ModuleList([copy.deepcopy(local_correction) for _ in range(nblocks)])
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        
    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            # local correction
            x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevel != 0:
                x2 = kint(x[:,::2**self.clevel])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)

            x = ln1(x1+x2) # f' = ln(f + attn(f))
            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

        x = self.q(x)
        return x

class GT1d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4):
        super(GT1d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.clevel = clevel 
        self.mlevel = mlevel
        self.nblocks = nblocks

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = GalerkinAttention1d(self.width, self.width, self.nhead)
        layer_norm = nn.LayerNorm(self.width)
        local_correction = MultiLevelLayer1d(self.width, mlevel)
        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)

        self.local_corrections = nn.ModuleList([copy.deepcopy(local_correction) for _ in range(nblocks)])
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        
    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            # local correction
            x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevel != 0:
                x2 = kint(x[:,::2**self.clevel])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)

            x = ln1(x1+x2) # f' = ln(f + attn(f))
            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

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
