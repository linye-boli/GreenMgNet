import copy 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange, repeat
from layers import (
    SmoothKernel1d,
    SpectralConv1d, SpectralConv2d,
    LowRank1d, LowRank2d,
    FourierAttention1d, FourierAttention2d,
    GalerkinAttention1d, GalerkinAttention2d)
from utils import DenseNet, MultiLevelLayer1d, MultiLevelLayer2d

def clevels_n_mlevels(clevel, mlevel, nblocks):
    if isinstance(clevel, int):
        clevels = [clevel] * nblocks
    elif isinstance(clevel, list) & (len(clevel) == nblocks):
        clevels = clevel

    if isinstance(mlevel, int):
        mlevels = [mlevel] * nblocks
    elif isinstance(mlevel, list) & (len(mlevel) == nblocks):
        mlevels = mlevel
    
    return clevels, mlevels

class ML1d(nn.Module):
    def __init__(self, modes, width, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(ML1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.clevel = clevel 
        self.mlevel = mlevel
        self.nblocks = nblocks

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = SmoothKernel1d(self.width, self.modes1)

        local_correction = MultiLevelLayer1d(self.width, mlevel)

        self.local_corrections = nn.ModuleList([copy.deepcopy(local_correction) for _ in range(nblocks)])
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = rearrange(x, "b l -> b 1 l")

        # x = torch.stack([a, x],dim=-1)
        # x = self.p(x)

        for i, (lc, kint) in enumerate(zip(self.local_corrections, self.kernel_integrals)):

            if self.mlevel >= 0:
                # local correction
                x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevel != 0:
                x2 = kint(x[:,::2**self.clevel])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)

            if self.mlevel >= 0:
                x = x1 + x2
            else:
                x = x2
            # # nonlinear 
            # if self.mlevel >= 0:
            #     x = F.relu(x1 + x2)
            # else:
            #     x = F.relu(x2)

        # x = self.q(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = SpectralConv1d(self.width, self.width, self.modes1)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer1d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

        # mws = []
        # for i in range(nblocks):
        #     ws = []
        #     for j in mlevels:
        #         if mw == 'same':
        #             ws.append(1)
        #         elif mw == 'learn':
        #             ws.append(nn.Parameter(torch.rand(, dtype=torch.float32)))
            
        #     mws.append(ws)

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint) in enumerate(zip(self.local_corrections, self.kernel_integrals)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)
            
            # nonlinear 
            if self.mlevels[i] >= 0:
                x = F.relu(x1 + x2)
            else:
                x = F.relu(x2)

        x = self.q(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(3, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*4, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer2d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        # x : [b, x, y, 2]
        # a : [b, x, y, 1]

        _, seq_lx, seq_ly, _ = x.shape
        
        x = torch.cat([a, x],dim=-1)

        x = self.p(x)

        for i, (lc, kint) in enumerate(zip(self.local_corrections, self.kernel_integrals)):
            
            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i],::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,3,1,2), (seq_lx, seq_ly), mode='bilinear').permute(0,2,3,1)
            else:
                x2 = kint(x)

            if self.mlevels[i] >= 0:
                # nonlinear 
                x = F.relu(x1 + x2)
            else:
                x = F.relu(x2)

        x = self.q(x)
        return x

class LNO1d(nn.Module):
    def __init__(self, width, rank, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(LNO1d, self).__init__()
        self.width = width
        self.rank = rank
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)

        kernel_integral = LowRank1d(self.width, self.rank)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer1d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        a = x.clone()
        x = self.p(x)

        for i, (lc, kint, ln) in enumerate(zip(self.local_corrections, self.kernel_integrals, self.layer_norms)):
            
            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i]], a[:,::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x, a)
            
            if self.mlevels[i] >= 0:
                x = ln(x1+x2)
            else:
                x = ln(x2)

            # nonlinear 
            if i != self.nblocks-1:
                x = F.relu(x)


        x = self.q(x)
        return x

class LNO2d(nn.Module):
    def __init__(self, width, rank, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(LNO2d, self).__init__()
        self.width = width
        self.rank = rank
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(3, self.width) # input channel_dim is 2: (u0(x,y), x,y)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = LowRank2d(self.width, self.rank)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer2d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        # x : [b, x, y, 2]
        # a : [b, x, y, 1]

        _, seq_lx, seq_ly, _ = x.shape
        
        x = torch.cat([a, x],dim=-1)

        a = x.clone()
        x = self.p(x)

        for i, (lc, kint, ln) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.layer_norms)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i],::2**self.clevels[i]], a[:,::2**self.clevels[i],::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,3,1,2), (seq_lx, seq_ly), mode='bilinear').permute(0,2,3,1)
            else:
                x2 = kint(x, a)

            if self.mlevels[i] >= 0:
                x = ln(x1+x2)
            else:
                x = ln(x2)

            # nonlinear 
            if i != self.nblocks-1:
                x = F.relu(x)


        x = self.q(x)
        return x

class FT1d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(FT1d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = FourierAttention1d(self.width, self.width, self.nhead)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])
        
        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])
    
        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer1d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)

            if self.mlevels[i] >= 0:
                x = ln1(x1+x2) # f' = ln(f + attn(f))
            else:
                x = ln1(x2)
            
            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

        x = self.q(x)
        return x

class FT2d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(FT2d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(3, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = FourierAttention2d(self.width, self.width, self.nhead)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])
        
        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])

        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer2d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

        
    def forward(self, x, a):
        _, seq_lx, seq_ly, _ = x.shape
        x = torch.cat([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i],::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,3,1,2), (seq_lx, seq_ly), mode='bilinear').permute(0,2,3,1)
            else:
                x2 = kint(x)

            if self.mlevels[i] >= 0:
                x = ln1(x1+x2) # f' = ln(f + attn(f))
            else:
                x = ln1(x2)
            
            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

        x = self.q(x)
        return x


class GT1d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(GT1d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = GalerkinAttention1d(self.width, self.width, self.nhead)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])

        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])
        
        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer1d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        seq_len = x.shape[1]
        x = torch.stack([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 2, 1)).permute(0, 2, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,2,1), seq_len, mode='linear').permute(0,2,1)
            else:
                x2 = kint(x)

            if self.mlevels[i] >= 0:
                # local correction
                x = ln1(x1+x2) # f' = ln(f + attn(f))
            else:
                x = ln1(x2)

            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

        x = self.q(x)
        return x

class GT2d(nn.Module):
    def __init__(self, width, nhead, clevel=0, mlevel=0, nblocks=4, mw='same'):
        super(GT2d, self).__init__()
        self.width = width
        self.nhead = nhead
        self.nblocks = nblocks

        clevels, mlevels = clevels_n_mlevels(clevel, mlevel, nblocks)
        self.clevels = clevels
        self.mlevels = mlevels

        self.p = nn.Linear(3, self.width) # input channel_dim is 2: (u0(x), x)
        self.q = DenseNet([self.width, self.width*2, 1], nn.ReLU)  # output channel_dim is 1: u1(x)
        kernel_integral = GalerkinAttention2d(self.width, self.width, self.nhead)
        self.kernel_integrals = nn.ModuleList([copy.deepcopy(kernel_integral) for _ in range(nblocks)])

        fnn = DenseNet([self.width, self.width, self.width], nn.ReLU)
        self.fnns = nn.ModuleList([copy.deepcopy(fnn) for _ in range(nblocks)])

        layer_norm = nn.LayerNorm(self.width)
        self.layer_norms1 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        self.layer_norms2 = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(nblocks)])
        
        local_corrections = []
        for i in range(nblocks):
            local_corrections.append(MultiLevelLayer2d(self.width, self.mlevels[i]))
        self.local_corrections = nn.ModuleList(local_corrections)

    def forward(self, x, a):
        _, seq_lx, seq_ly, _ = x.shape
        x = torch.cat([a, x],dim=-1)
        x = self.p(x)

        for i, (lc, kint, fnn, ln1, ln2) in enumerate(
            zip(self.local_corrections, self.kernel_integrals, self.fnns, self.layer_norms1, self.layer_norms2)):

            if self.mlevels[i] >= 0:
                # local correction
                x1 = lc(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            # smooth kernel integral
            if self.clevels[i] != 0:
                x2 = kint(x[:,::2**self.clevels[i],::2**self.clevels[i]])
                x2 = F.interpolate(x2.permute(0,3,1,2), (seq_lx, seq_ly), mode='bilinear').permute(0,2,3,1)
            else:
                x2 = kint(x)

            if self.mlevels[i] >= 0:
                x = ln1(x1+x2) # f' = ln(f + attn(f))
            else:
                x = ln1(x2)
                
            x = ln2(x+fnn(x)) # f = ln(f' + fnn(f'))

        x = self.q(x)
        return x




if __name__ == '__main__':
    # # 1d inputs:
    # bsz = 5 
    # modes = 12 
    # width = 32
    
    # seq_len = 4096
    # a = torch.rand((bsz, seq_len))
    # x = torch.rand((bsz, seq_len))
    # u = torch.rand((bsz, seq_len))
    
    # print('FNO1d test:')
    # model = FNO1d(modes, width, clevel=0, mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FNO1d(modes, width, clevel=[0, 2, 3, 1], mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FNO1d(modes, width, clevel=0, mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FNO1d(modes, width, clevel=[1, 3, 2, 0], mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)
    
    # width = 64
    # rank = 4

    # print('LNO1d test:')
    # model = LNO1d(width, rank, clevel=0, mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = LNO1d(width, rank, clevel=[0, 2, 3, 1], mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = LNO1d(width, rank, clevel=0, mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = LNO1d(width, rank, clevel=[1, 3, 2, 0], mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)

    # width = 64
    # head = 8

    # print('FT1d test:')
    # model = FT1d(width, head, clevel=0, mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FT1d(width, head, clevel=[0, 2, 3, 1], mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FT1d(width, head, clevel=0, mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = FT1d(width, head, clevel=[1, 3, 2, 0], mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)

    # width = 64
    # head = 8

    # print('GT1d test:')
    # model = GT1d(width, head, clevel=0, mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = GT1d(width, head, clevel=[0, 2, 3, 1], mlevel=[3, 2, 1, 0], nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = GT1d(width, head, clevel=0, mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)
    # model = GT1d(width, head, clevel=[1, 3, 2, 0], mlevel=1, nblocks=4)
    # print(model(x=x, a=a).shape)

    # inputs :
    bsz = 5 
    seq_lx = 211 
    seq_ly = 211
    width = 32
    modes = 12
    x = torch.rand((bsz, seq_lx, seq_ly, 2))
    a = torch.rand((bsz, seq_lx, seq_ly, 1))

    print('FNO2d test:')
    model = FNO2d(modes1=12, modes2=12, width=32, clevel=3, mlevel=3)
    print(model(x=x, a=a).shape)
    model = FNO2d(modes1=12, modes2=12, width=32, clevel=[3, 2, 1, 0], mlevel=3)
    print(model(x=x, a=a).shape)
    model = FNO2d(modes1=12, modes2=12, width=32, clevel=2, mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)
    model = FNO2d(modes1=12, modes2=12, width=32, clevel=[3, 2, 1, 0], mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)

    print('LNO2d test:')
    model = LNO2d(width=64, rank=4, clevel=3, mlevel=3)
    print(model(x=x, a=a).shape)
    model = LNO2d(width=64, rank=4, clevel=[3, 2, 1, 0], mlevel=3)
    print(model(x=x, a=a).shape)
    model = LNO2d(width=64, rank=4, clevel=2, mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)
    model = LNO2d(width=64, rank=4, clevel=[3, 2, 1, 0], mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)

    print('FT2d test:')
    model = FT2d(width=64, nhead=8, clevel=3, mlevel=3)
    print(model(x=x, a=a).shape)
    model = FT2d(width=64, nhead=8, clevel=[3, 2, 2, 1], mlevel=3)
    print(model(x=x, a=a).shape)
    model = FT2d(width=64, nhead=8, clevel=2, mlevel=[3, 2, 2, 0])
    print(model(x=x, a=a).shape)
    model = FT2d(width=64, nhead=8, clevel=[3, 2, 2, 1], mlevel=[3, 2, 2, 0])
    print(model(x=x, a=a).shape)

    print('GT2d test:')
    model = GT2d(width=64, nhead=8, clevel=3, mlevel=3)
    print(model(x=x, a=a).shape)
    model = GT2d(width=64, nhead=8, clevel=[3, 2, 1, 0], mlevel=3)
    print(model(x=x, a=a).shape)
    model = GT2d(width=64, nhead=8, clevel=2, mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)
    model = GT2d(width=64, nhead=8, clevel=[3, 2, 1, 0], mlevel=[3, 2, 1, 0])
    print(model(x=x, a=a).shape)
