import copy 
import torch 
import torch.nn as nn 
from einops import rearrange, repeat
from utils import DenseNet
import torch.nn.functional as F


################################################################
#  1d smooth kernel layer
################################################################
class SmoothKernel1d(nn.Module):
    def __init__(self, channels, modes):
        super(SmoothKernel1d, self).__init__()
        """
        1D Smooth Kernel layer.
        """
        self.channels = channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (modes*modes))

        # x = torch.linspace(0,1,modes)
        # y = torch.linspace(0,1,modes)
        # X, Y = torch.meshgrid(x, y)
        # init_sk = torch.cos(X - Y)
        # init_sk = repeat(init_sk, 'm n -> c m n', c=channels)

        # self.sk = nn.Parameter(init_sk + self.scale * torch.rand(channels, self.modes, self.modes, dtype=torch.float32))
        self.sk = nn.Parameter(self.scale * torch.rand(channels, self.modes, self.modes, dtype=torch.float32))

    def forward(self, w):
        seq_len = w.shape[1]
        x_c = F.interpolate(w.permute(0,2,1), self.modes, mode='linear')
        w_c = torch.einsum("bcm, cnm-> bcn", x_c, self.sk)
        w = F.interpolate(w_c, seq_len, mode='linear') / (self.modes ** 2)
        w = w.permute(0,2,1)
        return w


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        x = x.permute(0,2,1)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = x.permute(0,2,1)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        x = x.permute(0,2,3,1)
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    def forward(self, x):
        x = x.permute(0,4,1,2,3)

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))

        x = x.permute(0,2,3,4,1)
        return x

class SpectralConvLite1d(nn.Module):
    def __init__(self, channels, modes):
        super(SpectralConvLite1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.channels = channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = 1 / channels
        self.weights1 = nn.Parameter(self.scale * torch.rand(channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, channel, x ), (channel, x) -> (batch, channels, x)
        return torch.einsum("bix, ix -> bix", input, weights)

    def forward(self, x):
        x = x.permute(0,2,1)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = x.permute(0,2,1)
        return x

################################################################
# lowrank layer
################################################################
class LowRank1d(nn.Module):
    def __init__(self, width, rank=1):
        super(LowRank1d, self).__init__()
        self.width = width
        self.rank = rank

        self.phi = DenseNet([2, 64, 128, 256, width*rank], torch.nn.ReLU)
        self.psi = DenseNet([2, 64, 128, 256, width*rank], torch.nn.ReLU)

    def forward(self, v, a):
        # a (batch, n, 2)
        # v (batch, n, f)
        batch_size = v.shape[0]
        n = v.shape[1]

        phi_eval = self.phi(a).reshape(batch_size, n, self.width, self.rank)
        psi_eval = self.psi(a).reshape(batch_size, n, self.width, self.rank)

        Q = rearrange(phi_eval, 'b l d h -> b h l d', d=self.width, h=self.rank)
        K = rearrange(psi_eval, 'b l d h -> b h l d', d=self.width, h=self.rank)
        V = rearrange(v, 'b l d -> b 1 l d', d=self.width)

        attn_map = torch.einsum('bhmd,bhmc->bhdc', K, V)
        attn_out = torch.einsum('bhld,bhdc->blc', Q, attn_map)

        return attn_out / n
    
class LowRank2d(nn.Module):
    def __init__(self, width, rank=1):
        super(LowRank2d, self).__init__()
        self.width = width
        self.rank = rank

        self.phi = DenseNet([3, 64, 128, 256, width*rank], torch.nn.ReLU)
        self.psi = DenseNet([3, 64, 128, 256, width*rank], torch.nn.ReLU)

    def forward(self, v, a):
        # a (batch, nx, ny, 3)
        # v (batch, nx, ny, f)
        batch_size, nx, ny, cdim = v.shape
        n = nx * ny

        phi_eval = self.phi(a).reshape(batch_size, n, self.width, self.rank)
        psi_eval = self.psi(a).reshape(batch_size, n, self.width, self.rank)

        Q = rearrange(phi_eval, 'b l d h -> b h l d', d=self.width, h=self.rank)
        K = rearrange(psi_eval, 'b l d h -> b h l d', d=self.width, h=self.rank)
        V = rearrange(v, 'b x y d -> b 1 (x y) d', d=self.width)

        attn_map = torch.einsum('bhmd,bhmc->bhdc', K, V)
        attn_out = torch.einsum('bhld,bhdc->blc', Q, attn_map) / n 
        attn_out = attn_out.reshape(batch_size, nx, ny, cdim)

        return attn_out

################################################################
# fourier transformer layer
################################################################
class FourierAttention1d(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=5):
        super(FourierAttention1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead

        self.Wq = nn.Linear(in_channels, out_channels*nhead)
        self.Wk = nn.Linear(in_channels, out_channels*nhead)
        self.Wv = nn.Linear(in_channels, out_channels*nhead)
        self.oproj = nn.Linear(out_channels*nhead, out_channels)

        self.LnQ = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])
        self.LnK = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])

    def forward(self, v):
        # v (batch, n, f)
        batch_size = v.shape[0]
        n = v.shape[1]
        d = self.out_channels
        h = self.nhead

        Q = self.Wq(v)
        K = self.Wk(v)
        V = self.Wv(v)

        Q = rearrange(Q, 'b l (h d) -> b h l d', d=d, h=h)
        K = rearrange(K, 'b l (h d) -> b h l d', d=d, h=h)
        V = rearrange(V, 'b l (h d) -> b h l d', d=d, h=h)

        Q = torch.stack([norm(x) for norm, x in zip(self.LnQ, (Q[:,i,...] for i in range(h)))], dim=1)
        K = torch.stack([norm(x) for norm, x in zip(self.LnK, (K[:,i,...] for i in range(h)))], dim=1)
        
        attn_map = torch.einsum('bhmd,bhnd->bhmn', Q, K)
        attn_out = torch.einsum('bhmn,bhnd->bhmd', attn_map, V)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')

        attn_out = self.oproj(attn_out)
        attn_out = attn_out / n

        return attn_out 
    
class FourierAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=5):
        super(FourierAttention2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead

        self.Wq = nn.Linear(in_channels, out_channels*nhead)
        self.Wk = nn.Linear(in_channels, out_channels*nhead)
        self.Wv = nn.Linear(in_channels, out_channels*nhead)
        self.oproj = nn.Linear(out_channels*nhead, out_channels)

        self.LnQ = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])
        self.LnK = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])

    def forward(self, v):
        # v (batch, n, f)
        _, nx, ny, _ = v.shape
        n = nx * ny
        d = self.out_channels
        h = self.nhead

        Q = self.Wq(v)
        K = self.Wk(v)
        V = self.Wv(v)

        Q = rearrange(Q, 'b x y (h d) -> b h (x y) d', d=d, h=h)
        K = rearrange(K, 'b x y (h d) -> b h (x y) d', d=d, h=h)
        V = rearrange(V, 'b x y (h d) -> b h (x y) d', d=d, h=h)
        
        Q = torch.stack([norm(x) for norm, x in zip(self.LnQ, (Q[:,i,...] for i in range(h)))], dim=1)
        K = torch.stack([norm(x) for norm, x in zip(self.LnK, (K[:,i,...] for i in range(h)))], dim=1)

        attn_map = torch.einsum('bhmd,bhnd->bhmn', Q, K)
        attn_out = torch.einsum('bhmn,bhnd->bhmd', attn_map, V)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')

        attn_out = self.oproj(attn_out)
        attn_out = attn_out / n

        attn_out = rearrange(attn_out, 'b (x y) l -> b x y l', x=nx, y=ny)

        return attn_out 


################################################################
# galerkin Attention layer
################################################################
class GalerkinAttention1d(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=5):
        super(GalerkinAttention1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead

        self.Wq = nn.Linear(in_channels, out_channels*nhead)
        self.Wk = nn.Linear(in_channels, out_channels*nhead)
        self.Wv = nn.Linear(in_channels, out_channels*nhead)
        self.oproj = nn.Linear(out_channels*nhead, out_channels)

        self.LnK = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])
        self.LnV = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])

    def forward(self, v):
        # v (batch, n, f)
        batch_size = v.shape[0]
        n = v.shape[1]
        d = self.out_channels
        h = self.nhead

        Q = self.Wq(v)
        K = self.Wk(v)
        V = self.Wv(v)

        Q = rearrange(Q, 'b l (h d) -> b h l d', d=d, h=h)
        K = rearrange(K, 'b l (h d) -> b h l d', d=d, h=h)
        V = rearrange(V, 'b l (h d) -> b h l d', d=d, h=h)

        K = torch.stack([norm(x) for norm, x in zip(self.LnK, (K[:,i,...] for i in range(h)))], dim=1)
        V = torch.stack([norm(x) for norm, x in zip(self.LnV, (V[:,i,...] for i in range(h)))], dim=1)
        
        attn_map = torch.einsum('bhmd,bhmc->bhdc', K, V)
        attn_out = torch.einsum('bhld,bhdc->bhlc', Q, attn_map)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        attn_out = self.oproj(attn_out)
        attn_out = attn_out / n

        return attn_out 

class GalerkinAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=5):
        super(GalerkinAttention2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead

        self.Wq = nn.Linear(in_channels, out_channels*nhead)
        self.Wk = nn.Linear(in_channels, out_channels*nhead)
        self.Wv = nn.Linear(in_channels, out_channels*nhead)
        self.oproj = nn.Linear(out_channels*nhead, out_channels)

        self.LnK = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])
        self.LnV = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])

    def forward(self, v):
        # v (batch, n, f)
        _, nx, ny, _ = v.shape
        n = nx * ny
        d = self.out_channels
        h = self.nhead

        Q = self.Wq(v)
        K = self.Wk(v)
        V = self.Wv(v)

        Q = rearrange(Q, 'b x y (h d) -> b h (x y) d', d=d, h=h)
        K = rearrange(K, 'b x y (h d) -> b h (x y) d', d=d, h=h)
        V = rearrange(V, 'b x y (h d) -> b h (x y) d', d=d, h=h)

        K = torch.stack([norm(x) for norm, x in zip(self.LnK, (K[:,i,...] for i in range(h)))], dim=1)
        V = torch.stack([norm(x) for norm, x in zip(self.LnV, (V[:,i,...] for i in range(h)))], dim=1)
        
        attn_map = torch.einsum('bhmd,bhmc->bhdc', K, V)
        attn_out = torch.einsum('bhld,bhdc->bhlc', Q, attn_map)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        attn_out = self.oproj(attn_out)
        attn_out = attn_out / n

        attn_out = rearrange(attn_out, 'b (x y) l -> b x y l', x=nx, y=ny)

        return attn_out 

class GalerkinAttention3d(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=5):
        super(GalerkinAttention3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead

        self.Wq = nn.Linear(in_channels, out_channels*nhead)
        self.Wk = nn.Linear(in_channels, out_channels*nhead)
        self.Wv = nn.Linear(in_channels, out_channels*nhead)
        self.oproj = nn.Linear(out_channels*nhead, out_channels)

        self.LnK = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])
        self.LnV = nn.ModuleList([copy.deepcopy(nn.LayerNorm(out_channels)) for _ in range(nhead)])

    def forward(self, v):
        # v (batch, n, f)
        _, nx, ny, nt, _ = v.shape
        n = nx * ny
        d = self.out_channels
        h = self.nhead

        Q = self.Wq(v)
        K = self.Wk(v)
        V = self.Wv(v)

        Q = rearrange(Q, 'b x y t (h d) -> b h (x y t) d', d=d, h=h)
        K = rearrange(K, 'b x y t (h d) -> b h (x y t) d', d=d, h=h)
        V = rearrange(V, 'b x y t (h d) -> b h (x y t) d', d=d, h=h)

        K = torch.stack([norm(x) for norm, x in zip(self.LnK, (K[:,i,...] for i in range(h)))], dim=1)
        V = torch.stack([norm(x) for norm, x in zip(self.LnV, (V[:,i,...] for i in range(h)))], dim=1)
        
        attn_map = torch.einsum('bhmd,bhmc->bhdc', K, V)
        attn_out = torch.einsum('bhld,bhdc->bhlc', Q, attn_map)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        attn_out = self.oproj(attn_out)
        attn_out = attn_out / n

        attn_out = rearrange(attn_out, 'b (x y t) l -> b x y t l', x=nx, y=ny)

        return attn_out 


if __name__ == '__main__':

    # # inputs :
    # bsz = 5 
    # seq_len = 128 
    # cin = 32
    # cout = 32
    # v = torch.rand((bsz, seq_len, cin))
    # a = torch.rand((bsz, seq_len, 2))
    # print('v : ', v.shape)
    # print('a : ', a.shape)

    # # smooth_kernel cfg:
    # modes = 16
    # sk1d = SmoothKernel1d(cin, modes)
    # u = sk1d(v)
    # print('------- smooth kernel -------')
    # print('u : ', u.shape)

    # # spec_conv cfg:
    # modes = 16
    # spec_conv1d = SpectralConv1d(cin, cout, modes)
    # u = spec_conv1d(v)
    # print('------- spec_conv -------')
    # print('u : ', u.shape)

    # # Low Rank cfg
    # rank = 4
    # lr_1d = LowRank1d(cin, rank)
    # u = lr_1d(v, a)
    # print('------- lowrank -------')
    # print('u : ', u.shape)

    # # Fourier Attention 
    # nhead = 4
    # fattn = FourierAttention1d(cin, cout, nhead)
    # u = fattn(v)
    # print('------- fourier_attn -------')
    # print('u : ', u.shape)

    # # Galerkin Attention 
    # nhead = 4
    # gattn = GalerkinAttention1d(cin, cout, nhead)
    # u = gattn(v)
    # print('------- galerkin_attn -------')
    # print('u : ', u.shape)

    # # inputs :
    # bsz = 5 
    # seq_lx = 141
    # seq_ly = 141 
    # cin = 64
    # cout = 32
    # v = torch.rand((bsz, seq_lx, seq_ly, cin))
    # a = torch.rand((bsz, seq_lx, seq_ly, 3))
    # print('------ data 2d ------')
    # print('v : ', v.shape)
    # print('a : ', a.shape)

    # lr_2d = LowRank2d(cin, rank)
    # u = lr_2d(v, a)
    # print('------- lowrank -------')
    # print('u : ', u.shape)

    # nhead = 4
    # fattn = FourierAttention2d(cin, cout, nhead)
    # u = fattn(v)
    # print('------- fourier_attn -------')
    # print('u : ', u.shape)

    # nhead = 4
    # fattn = GalerkinAttention2d(cin, cout, nhead)
    # u = fattn(v)
    # print('------- fourier_attn -------')
    # print('u : ', u.shape)

    # inputs : 
    bsz = 5
    seq_lx = 64
    seq_ly = 64
    seq_lt = 30
    cin = 20
    cout = 20
    
    a = torch.rand((bsz, seq_lx, seq_ly, seq_lt, cin))

    # modes = 8
    # width = 20

    # spec_conv3d = SpectralConv3d(cin, cout, modes, modes, modes)
    # u = spec_conv3d(a)
    # print('------- spec_conv3d -------')
    # print('u : ', u.shape)

    nhead = 4 
    gattn = GalerkinAttention3d(cin, cout, nhead)
    u = gattn(a)
    print('------- galerkin_attn3d -------')
    print('u : ', u.shape)

