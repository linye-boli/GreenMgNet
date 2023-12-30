import torch 
import torch.nn.functional as F 
from siren_pytorch import Sine
import torch.nn as nn
from einops import repeat, rearrange
from utils import (
    toeplitz_matrix_vector_multiplication, 
    lowrank_matrix_vector_multiplication,
    ml_matrix_vector_multiplication,
    fullml_matrix_vector_multiplication,
    full_matrix_vector_multiplication,
    injection1d, 
    interp1d, interp2d, interp1d_rows, interp1d_cols,
    gauss_smooth1d, gauss_smooth2d,
    fetch_nbrs)
import kornia

# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()

        nonlinearity = Activations[nonlinearity]        
        out_nonlinearity = Activations[out_nonlinearity] if out_nonlinearity is not None else None

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Conv1d(layers[j], layers[j+1], kernel_size=1))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class CNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(CNN, self).__init__()

        nonlinearity = Activations[nonlinearity]        
        out_nonlinearity = Activations[out_nonlinearity] if out_nonlinearity is not None else None

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Conv1d(layers[j], layers[j+1], kernel_size=3, padding=1))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class Gauss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1))
        self.mu = nn.Parameter(torch.randn(1))
        self.sigma = nn.Parameter(torch.randn(1))
        
    def forward(self, input):
        output = self.s * torch.exp(-(input - self.mu)**2 / (2*self.sigma**2))
        return output

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output

Activations = {
    'relu' : nn.ReLU,
    'rational' : Rational,
    'tanh' : nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu' : nn.GELU,
    'elu': nn.ELU,
    'sine' : Sine,
    'gauss' : Gauss
}

# Green Multi-grid Network
class GMGN(nn.Module):
    def __init__(self, in_channels, out_channels, n, k, m, r, mtype='toep_mg'):
        super(GMGN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        
        self.mtype = mtype
        self.nn = nn # type of model
        self.n = n # number points on finest grid
        self.m = m # correction range dict
        self.r = r
        self.k = k

        if self.mtype == 'toep_mg':
            kernels, nbrs = self.init_toep_kernel()
            self.nbrs = nbrs

            # kernels, nbrs = self.init_nntoep_kernel()
            # self.nbrs = nbrs
        
        # if self.mtype == 'lr_mg':
        #     kernels, nbrs = self.init_lrdd_kernel()
        #     self.nbrs = nbrs
        
        if self.mtype == 'dd_mg':
            kernels, nbrs = self.init_dd_kernel()
            self.nbrs = nbrs
        
        # if self.mtype == 'mlplr_mg':
        #     kernels, idx_i_dict, idx_j_dict, idx_j_mask_dict = self.init_mlplrdd_kernel()
        #     self.idx_i_dict = idx_i_dict
        #     self.idx_j_dict = idx_j_dict
        #     self.idx_j_mask_dict = idx_j_mask_dict

        self.kernels = torch.nn.ParameterDict(kernels)
    
    def init_toep_kernel(self):
        mg_kernels = {}
        idx_lst = []
        n = 2*(self.n-1)+1
        for l in range(self.k):
            mg_kernels['bandcorr-'+str(l)] = torch.nn.Parameter(torch.zeros(1,1,2*self.m+1))
            # neighbors index
            idx = torch.arange((n-1)//2-self.m, (n-1)//2+self.m+1)
            idx_lst.append(idx)
            n = (n-1)//2+1

        idx_lst = idx_lst[::-1]
        mg_kernels["smooth"] = torch.nn.Parameter(torch.zeros(1,1,n))
        return mg_kernels, idx_lst

    def init_nntoep_kernel(self):
        mg_kernels = {}
        idx_lst = []
        n = 2*(self.n-1)+1
        for l in range(self.k):
            mg_kernels['bandcorr-'+str(l)] = MLP([self.in_channels, 64, 64, self.out_channels], nonlinearity='relu')
            # neighbors index
            idx = torch.arange((n-1)//2-self.m, (n-1)//2+self.m+1)
            idx_lst.append(idx)
            n = (n-1)//2+1

        idx_lst = idx_lst[::-1]
        mg_kernels["smooth"] = MLP([self.in_channels, 64, 64, self.out_channels], nonlinearity='relu')
        return mg_kernels, idx_lst

    def init_dd_kernel(self):
        mg_kernels = {}
        Khh_bandcorr_lst = []

        idx_i_lst = []
        idx_j_lst = []
        
        n = self.n
        for i in range(self.k):
            # fetch nbr idx
            idx_i, idx_j = fetch_nbrs(n, self.m)
            idx_mask = (idx_j >= 0) & (idx_j <= n-1)
            idx_j[idx_j < 0] = 0 
            idx_j[idx_j > n-1] = n-1
            idx_i_lst.append(idx_i)
            idx_j_lst.append(idx_j)

            # init band correction part of Khh
            Khh_bandcorr = torch.nn.Parameter(torch.zeros(1,1, n, 2*self.m+1))
            Khh_bandcorr_lst.append(Khh_bandcorr)

            n = (n-1)//2+1

        # reverse order
        Khh_bandcorr_lst = Khh_bandcorr_lst[::-1]
        idx_i_lst = idx_i_lst[::-1]
        idx_j_lst = idx_j_lst[::-1]

        # init smooth part of Khh
        KHH = torch.nn.Parameter(torch.zeros(1,1,n,n))
        print("smooth : ", KHH.shape)
        mg_kernels['smooth'] = KHH       

        for i in range(self.k):
            mg_kernels[f'bandcorr-{i}'] = Khh_bandcorr_lst[i]
            print(f"bandcorr-{i}  : ", Khh_bandcorr_lst[i].shape)

        return mg_kernels, [idx_i_lst, idx_j_lst]

    def init_mlpdd_kernel(self):
        kernels = {}
        nbrs = {}
        mg_n = [(self.n-1)//2**(self.nl-l-1)+1 for l in range(self.nl)]
        for l in range(self.nl):
            if l == 0:
                kernels['phi-'+str(l)] = torch.nn.Parameter(torch.randn(1,self.r,mg_n[l])/self.r)
                kernels['psi-'+str(l)] = torch.nn.Parameter(torch.randn(1,self.r,mg_n[l])/self.r)
            else:
                if l in self.ms.keys():
                    kernels['phi-'+str(l)] = torch.nn.Parameter(torch.zeros(1,self.r,mg_n[l]))
                    kernels['psi-'+str(l)] = torch.nn.Parameter(torch.zeros(1,self.r,2*self.ms[l]+1))
                    nbrs[l] = fetch_nbrs(mg_n[l], self.ms[l])
        return kernels, nbrs

    def init_mlplrdd_kernel(self):
        kernels = {}
        nbrs = {}
        mg_n = [(self.n-1)//2**(self.nl-l-1)+1 for l in range(self.nl)]
        for l in range(self.nl):
            if l == 0:
                kernels['phi-'+str(l)] = MLP([self.in_channels//2, 256, 256, 256, 256, self.r], 'relu')
                kernels['psi-'+str(l)] = MLP([self.in_channels//2, 256, 256, 256, 256, self.r], 'relu')
            else:
                if l in self.ms.keys():
                    kernels['cor-'+str(l)] = MLP([self.in_channels//2, 256, 256, 256, 256, 2*self.ms[l]+1], 'relu')
                    nbrs[l] = fetch_nbrs(mg_n[l], self.ms[l])
        return kernels, nbrs

    def toep_kernel_approximation(self, x):
        # x = interp1d(x)
        # for l in range(self.nl-1):
        #     x = injection1d(x)

        Ah = self.kernels['smooth'] # 65
        # Ah = gauss_smooth1d(Ah)
        for i in range(self.k):
            Ah = interp1d(Ah) # 129
            Aband = self.kernels[f'bandcorr-{i}'] # 16
            Aband.data[:,:,0] = 0
            Aband.data[:,:,-1] = 0       
            Ah[:,:,self.nbrs[i]] += Aband
        
        return Ah
    
    def lrdd_kernel_approximation(self, x):

        ks = {}
        for l in range(self.nl):
            if l == 0:
                phi = self.kernels['phi-'+str(0)]
                psi = self.kernels['psi-'+str(0)]
                a = torch.einsum('b r n, b r m -> b n m', phi, psi)[None]
                ks[l] = a
                # assert a.shape[-1] == x.shape[-1]
            else:
                a = interp2d(a)
                if l in self.ms.keys():
                    phi = self.kernels['phi-'+str(l)]
                    psi = self.kernels['psi-'+str(l)]
                    a_band = torch.einsum('b r n, b r m -> b n m', phi, psi)[None]
                    a[:,:,self.nbrs[l][0], self.nbrs[l][1]] += a_band
                    ks[l] = a_band
        
        # for l in range(self.nl):
        #     if l == 0:
        #         phi = self.kernels['phi-'+str(0)]
        #         psi = self.kernels['psi-'+str(0)]
        #         a = torch.einsum('b r n, b r m -> b n m', phi, psi)[None]
        #         # assert a.shape[-1] == x.shape[-1]
        #     else:
        #         a = interp2d(a)
        #         if l in self.ms.keys():
        #             phi = self.kernels['phi-'+str(l)]
        #             psi = self.kernels['psi-'+str(l)]
        #             a_band = torch.einsum('b r n, b r m -> b n m', phi, psi)[None]
        #             a[:,:,self.nbrs[l][0], self.nbrs[l][1]] += a_band

            if self.k == 0:
                pass
            else:
                a = gauss_smooth2d(a,ksize=self.k)
        
        return a, ks

    def dd_kernel_approximation(self, x):
        K = self.kernels['smooth']
        # K = gauss_smooth2d(K, ksize=31)
        for i in range(self.k):
            K = interp1d_rows(interp1d_cols(K))
            Kband_corr = self.kernels[f'bandcorr-{i}']
            Kband_corr.data[:,:,:,0] = 0
            Kband_corr.data[:,:,:,-1] = 0
            K[:,:,self.nbrs[0][i], self.nbrs[1][i]] += Kband_corr
        
        return K

    def nntoep_kernel_approximation(self, x):
        xm = torch.linspace(-1,1,2*self.m+1)[None,None].to(x)
        
        x = interp1d(x)
        for l in range(self.k-1):
            x = injection1d(x)

        Ah = self.kernels['smooth'](x)
        for i in range(self.k):
            Ah = interp1d(Ah)
            Aband = self.kernels[f'bandcorr-{i}'](xm)
            Aband[:,:,0] = 0
            Aband[:,:,-1] = 0      
            Ah[:,:,self.nbrs[i]] += Aband

        return Ah

    def mlplrdd_kernel_approximation(self, x):
        for l in range(self.nl-1):
            x = injection1d(x)
        
        for l in range(self.nl):
            if l == 0:
                phi = self.kernels['phi-'+str(0)](x)
                psi = self.kernels['psi-'+str(0)](x)
                a = torch.einsum('b r n, b r m -> b n m', phi, psi)[None]
                # assert a.shape[-1] == x.shape[-1]
            else:
                a = interp2d(a)
                x = interp1d(x)
                if l in self.ms.keys():
                    a_band = self.kernels['cor-'+str(l)](x).permute(0,2,1)[None]
                    a[:,:,self.nbrs[l][0], self.nbrs[l][1]] += a_band

            if self.k == 0:
                pass 
            else:
                a = gauss_smooth2d(a,ksize=self.k)
        
        return a

    def forward(self, u, x):
        
        if self.mtype == 'toep_mg':
            a = self.toep_kernel_approximation(x)
            w = toeplitz_matrix_vector_multiplication(a, u)
            # a = self.nntoep_kernel_approximation(x)
            # w = toeplitz_matrix_vector_multiplication(a, u)


        
        if self.mtype == 'lr_mg':
            a, ks = self.lrdd_kernel_approximation(x)
            w = full_matrix_vector_multiplication(a, u)
        
        if self.mtype == 'dd_mg':
            a = self.dd_kernel_approximation(x)
            # h = x[0,0,1]-x[0,0,0]
            w = full_matrix_vector_multiplication(a, u)
            # w = fullml_matrix_vector_multiplication(u, a, h, self.nl, m=self.m)
            # import pdb 
            # pdb.set_trace()

        if self.mtype == 'mlplr_mg':
            a = self.mlplrdd_kernel_approximation(x)
            w = full_matrix_vector_multiplication(a, u)

        return w, a

class GL(nn.Module):
    def __init__(self, in_channels, out_channels, r, act='rational', mtype='toep_gl'):
        super(GL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.mtype = mtype

        if mtype == 'toep_gl':
            self.kernel = MLP([in_channels,256,256,256,256,out_channels], act)
        elif mtype == 'lr_gl':
            self.phi = MLP([in_channels//2, 256, 256, 256, 256, r], act)
            self.psi = MLP([in_channels//2, 256, 256, 256, 256, r], act)
        elif mtype == 'gl':
            self.kernel = MLP([in_channels,256,256,256,256,out_channels], act)
    
    def kernel_approximation(self, x):
        if self.mtype == 'toep_gl':
            x = interp1d(x)
            k = self.kernel(x)
        
        if self.mtype == 'lr_gl':
            phi = self.phi(x)
            psi = self.psi(x)
            k = [phi, psi]

        if self.mtype == 'gl':
            n = x.shape[-1]
            x = torch.cartesian_prod(x[0,0], x[0,0])
            k = self.kernel(rearrange(x, 'b c -> b c 1'))
            k = k.reshape(1,1,n,n)

        return k 

    def forward(self, u, x):
        k = self.kernel_approximation(x)

        if self.mtype == 'toep_gl':
            w = toeplitz_matrix_vector_multiplication(k, u)
        
        if self.mtype == 'lr_gl':
            w = lowrank_matrix_vector_multiplication(k[0], k[1], u)
        
        if self.mtype == 'gl':
            w = full_matrix_vector_multiplication(k, u)

        return w, k
