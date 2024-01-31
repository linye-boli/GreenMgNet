import torch 
import torch.nn.functional as F 
from siren_pytorch import Sine
import torch.nn as nn
import numpy as np
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
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False, init=None, bias=True):
        super(MLP, self).__init__()

        nonlinearity = Activations[nonlinearity]        
        out_nonlinearity = Activations[out_nonlinearity] if out_nonlinearity is not None else None

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            if bias:
                conv = nn.Conv1d(layers[j], layers[j+1], kernel_size=1)
            else:
                conv = nn.Conv1d(layers[j], layers[j+1], kernel_size=1, bias=False)

            if init == 'zero':
                torch.nn.init.constant(conv.weight, 0)
            self.layers.append(conv)

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.InstanceNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class CNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False, bias=True):
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
    def __init__(self, in_channels, out_channels, hidden_channels, n, k, m, r, act, mtype='toep_mg'):
        super(GMGN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        
        self.mtype = mtype# type of model
        self.n = n # number points on finest grid
        self.m = m # correction range dict
        self.r = r
        self.k = k
        self.act = act 

        if self.mtype == 'toep_mg':
            kernels, nbrs, xs = self.init_toep_kernel()
            self.nbrs = nbrs
            self.xs = xs
            print(f'num points to eval : {self.xs.shape}')
        
        if self.mtype in ['dd_mg', 'lrdd_mg']:
            kernels, xs, intervals, nbrs, masks = self.init_dd_kernel(self.mtype)
            self.xs = xs 
            print(f'num points to eval : {self.xs.shape}')
            self.intervals = intervals
            self.nbrs = nbrs
            self.masks = masks
        
        self.kernels = torch.nn.ParameterDict(kernels)

    def init_toep_kernel(self):
        mg_kernels = {}
        idx_lst = []
        x_lst = []

        n = 2*(self.n-1)+1
        x = torch.linspace(-1,1,n)[None,None]

        for l in range(self.k):
            n = x.shape[-1]
            print("level : ", l, " len : ", n, ' nbr : ', 2*self.m+1)
            # neighbors index
            if self.m > 0:
                idx = torch.arange((n-1)//2-self.m+1, (n-1)//2+self.m)[1::2]
                idx_lst.append(idx)
                x_lst.append(x[:,:,idx])
            x = injection1d(x)

        x_lst.append(x)
        xs = torch.concat(x_lst, axis=-1)

        idx_lst = idx_lst[::-1]
        x_lst = x_lst[::-1]
        mg_kernels["kernel"] = MLP([self.in_channels] + [self.hidden_channels]*4 + [self.out_channels], nonlinearity=self.act)
        return mg_kernels, idx_lst, xs

    def toep_kernel_approximation(self,x):     

        Amg = self.kernels['kernel'](self.xs)

        if self.m > 0:
            Abands = []
            for i in range(self.k):
                Abands.append(Amg[:,:,(self.m-1)*i:(self.m-1)*(i+1)])
            Abands = Abands[::-1]
            Ah = Amg[:,:,(self.m-1)*(i+1):]
        else:
            Ah = Amg

        for i in range(self.k):
            Ah = interp1d(Ah)
            if self.m > 0:
                Ah[:,:,self.nbrs[i]] = Abands[i]

        return Ah

    def toep_kernel_reconstruction(self):
        return self.toep_kernel_approximation(self.xs)
        
    def init_dd_kernel(self, mtype):
        kernels = {}
        xs = []
        idx_i_lst = []
        idx_j_lst = []
        mask_lst = []
        interval_dict = {}

        n = self.n
        xh = torch.linspace(-1,1,n)
        xhh = torch.cartesian_prod(xh, xh).reshape(n,n,-1)

        idx_h = torch.arange(n)
        idx_hh = torch.cartesian_prod(idx_h, idx_h).reshape(n,n,-1)

        for l in range(self.k):
            print("level : ", l, " len : ", n, ' nbr : ', 2*self.m+1)
            # fetch nbrs at level l
            idx_i, idx_j = fetch_nbrs(n, self.m)
            idx_mask = (idx_j >= 0) & (idx_j <= n-1)
            idx_j[idx_j < 0] = 0 
            idx_j[idx_j > n-1] = n-1
            
            idx_i_lst.append(idx_i)
            idx_j_lst.append(idx_j)
            mask_lst.append(idx_mask)            

            # extract band idx
            idx_band = idx_hh[idx_i, idx_j]
            idx_band_even = idx_band[::2,::2].reshape(-1,2)
            idx_band_odd = idx_band[1::2].reshape(-1,2)
            interval_dict[f'band_even_{self.k-l-1}'] = np.prod(idx_band[::2,::2].shape[:2])
            interval_dict[f'band_odd_{self.k-l-1}'] = np.prod(idx_band[1::2].shape[:2])

            # extract band xs 
            xband = xhh[idx_i, idx_j]
            xband_even = xband[::2,::2].reshape(-1,2)
            xband_odd = xband[1::2].reshape(-1,2)
            xs.append(xband_odd)
            xs.append(xband_even)

            print(l, idx_band_odd.shape[0])
            print(l, idx_band_even.shape[0])  

            xh = injection1d(xh[None,None])[0,0]
            idx_h = injection1d(idx_h[None,None])[0,0]
            n = (n-1)//2+1

            xhh = torch.cartesian_prod(xh, xh).reshape(n,n,-1)
            idx_hh = torch.cartesian_prod(idx_h, idx_h).reshape(n,n,-1)
        
        print('smooth ', np.prod(idx_hh.shape[:2]))
        interval_dict[f'full_coarse'] = np.prod(idx_hh.shape[:2])
        idx_i_lst = idx_i_lst[::-1]
        idx_j_lst = idx_j_lst[::-1]
        mask_lst = mask_lst[::-1]

        xs.append(xhh.reshape(-1,2))
        xs = xs[::-1]
        xs = torch.concat(xs)

        idx_interval = {}
        idx_start = 0
        idx_end = interval_dict['full_coarse']
        idx_interval['full_coarse'] = [idx_start, idx_end]
        print('smooth ', idx_end-idx_start)
        for i in range(self.k):
            idx_start = idx_end 
            idx_end = idx_start + interval_dict[f'band_even_{i}']
            idx_interval[f'band_even_{i}'] = [idx_start, idx_end]
            print(i, idx_end-idx_start)
            idx_start = idx_end 
            idx_end = idx_start + interval_dict[f'band_odd_{i}']
            idx_interval[f'band_odd_{i}'] = [idx_start, idx_end]
            print(i, idx_end-idx_start)

        if mtype == 'dd_mg':
            kernels['kernel'] = MLP([self.in_channels] + [self.hidden_channels]*4 + [self.out_channels], nonlinearity=self.act)
        elif mtype == 'lrdd_mg':
            kernels['psi'] = MLP([self.in_channels//2] + [self.hidden_channels]*4 + [self.r], nonlinearity=self.act)
            kernels['phi'] = MLP([self.in_channels//2] + [self.hidden_channels]*4 + [self.r], nonlinearity=self.act)
            
        xs = rearrange(xs, 'n c->n c 1')
        nbrs = [idx_i_lst, idx_j_lst]
        
        return kernels, xs, idx_interval, nbrs, mask_lst

    def dd_kernel_approximation(self, x):

        Kmg = self.kernels['kernel'](self.xs)
        Kmg = rearrange(Kmg, 'n c l -> l c n', c=1, l=1)

        nH = int(2**(np.log2(self.n-1) - self.k)+1)
        KHH = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,nH,nH)
        # fetch KHH band
        idx_i, idx_j = fetch_nbrs(nH, (self.m-1)//2+1) 
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > nH-1] = nH-1 
        Kband = KHH[:,:,idx_i, idx_j]        
        K_lst = [KHH]

        if self.m > 0:
            for i in range(self.k):
                Kband_even_interp = interp1d_cols(Kband)
                beven_start = self.intervals[f'band_even_{i}'][0]
                beven_end = self.intervals[f'band_even_{i}'][1]
                Kband_even = Kmg[:,:,beven_start:beven_end].reshape(1,1,-1, self.m+1)
                Kband_even_corr = Kband_even - Kband_even_interp[:,:,:,1:-1:2]

                Kband_even_interp[:,:,:,1:-1:2] = Kband_even
                bodd_start = self.intervals[f'band_odd_{i}'][0]
                bodd_end = self.intervals[f'band_odd_{i}'][1]
                Kband_odd_interp = (Kband_even_interp[:,:,:-1,:-2] + Kband_even_interp[:,:,1:,2:])/2 # interpolation
                Kband_odd = Kmg[:,:,bodd_start:bodd_end].reshape(1,1,-1, 2*self.m+1)
                Kband_odd_corr = Kband_odd - Kband_odd_interp

                K_lst.append([Kband_even_corr, Kband_odd_corr])

                Kband = interp1d_rows(Kband_even_interp)
                Kband[:,:,1:-1:2,1:-1] = Kband_odd
                Kband = Kband[:,:,:,self.m//2+1:self.m//2+self.m+1+2]

        return K_lst

    def lrdd_kernel_approximation(self, x):
        psi = self.kernels['psi'](x)
        phi = self.kernels['phi'](x)

        psi_mg = psi[:,:,self.idx[:,0]]
        phi_mg = phi[:,:,self.idx[:,1]]

        Kmg = (psi_mg * phi_mg).sum(axis=1)[None]

        n_lst = []
        n = self.n 
        n_lst.append(n)
        for i in range(self.k):
            n = (n-1)//2+1
            n_lst.append(n)
        n_lst = n_lst[::-1]

        Khh = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,n_lst[0],n_lst[0])
        K_lst = [Khh]
        for i in range(self.k):
            Khh = interp1d_rows(interp1d_cols(Khh))

            if self.m > 0:
                Kband_interp = Khh[:,:,self.nbrs[0][i],self.nbrs[1][i]]
                
                Kband_even_interp = Kband_interp[:,:,::2,::2]
                Kband_odd_interp = Kband_interp[:,:,1::2]
                Kband_even = Kmg[:,:,self.intervals[f'band_even_{i}'][0]:self.intervals[f'band_even_{i}'][1]].reshape(1,1,n_lst[i+1]//2+1, self.m+1)
                Kband_odd = Kmg[:,:,self.intervals[f'band_odd_{i}'][0]:self.intervals[f'band_odd_{i}'][1]].reshape(1,1,n_lst[i+1]//2, 2*self.m+1)

                Kband_even_corr = Kband_even - Kband_even_interp
                Kband_odd_corr = Kband_odd - Kband_odd_interp

                K_lst.append([Kband_even_corr, Kband_odd_corr])

                Khh[:,:,self.nbrs[0][i][::2,::2],self.nbrs[1][i][::2,::2]] += Kband_even_corr
                Khh[:,:,self.nbrs[0][i][1::2],self.nbrs[1][i][1::2]] += Kband_odd_corr

        return K_lst

    def dd_kernel_reconstruction(self):
        Kmg = self.kernels['kernel'](self.xs)
        Kmg = rearrange(Kmg, 'n c l -> l c n', c=1, l=1)

        nH = int(2**(np.log2(self.n-1) - self.k)+1)
        Khh = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,nH,nH)

        for i in range(self.k):
            Khh = interp1d_rows(interp1d_cols(Khh))
            
            if self.m > 0:
                beven_start = self.intervals[f'band_even_{i}'][0]
                beven_end = self.intervals[f'band_even_{i}'][1]
                Kband_even = Kmg[:,:,beven_start:beven_end].reshape(1,1,-1, self.m+1)
                
                bodd_start = self.intervals[f'band_odd_{i}'][0]
                bodd_end = self.intervals[f'band_odd_{i}'][1]
                Kband_odd = Kmg[:,:,bodd_start:bodd_end].reshape(1,1,-1, 2*self.m+1)
                
                Khh[:,:,self.nbrs[0][i][::2,::2],self.nbrs[1][i][::2,::2]] = Kband_even
                Khh[:,:,self.nbrs[0][i][1::2],self.nbrs[1][i][1::2]] = Kband_odd
        
        return Khh

    def fetch_kernel(self):
        if self.mtype == 'toep_mg':
            A = self.toep_kernel_reconstruction()
            return A.detach().cpu().numpy()[0,0]
        elif self.mtype == 'dd_mg':
            K = self.dd_kernel_reconstruction()
            return K.detach().cpu().numpy()[0,0]

    def forward(self, u, x):
        if self.mtype == 'toep_mg':
            A = self.toep_kernel_approximation(x)
            w = toeplitz_matrix_vector_multiplication(A, u)
              
        if self.mtype == 'dd_mg':
            Ks = self.dd_kernel_approximation(x)
            h = x[0,0,1]-x[0,0,0]
            w = ml_matrix_vector_multiplication(Ks, u, self.nbrs[1], self.masks, h, k=self.k)

        if self.mtype == 'lrdd_mg':
            Ks = self.lrdd_kernel_approximation(x)
            h = x[0,0,1]-x[0,0,0]
            w = ml_matrix_vector_multiplication(Ks, u, self.nbrs[1], self.masks, h, k=self.k)

        return w


class GMGN2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n, k, m, r, act, mtype='toep_mg'):
        super(GMGN2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        
        self.mtype = mtype# type of model
        self.n = n # number points on finest grid
        self.m = m # correction range dict
        self.r = r
        self.k = k
        self.act = act 

        if self.mtype == 'toep_mg':
            kernels, nbrs, xs = self.init_toep_kernel()
            self.nbrs = nbrs
            self.xs = xs
            print(f'num points to eval : {self.xs.shape}')
        
        if self.mtype in ['dd_mg', 'lrdd_mg']:
            kernels, xs, intervals, nbrs, masks = self.init_dd_kernel(self.mtype)
            self.xs = xs 
            print(f'num points to eval : {self.xs.shape}')
            self.intervals = intervals
            self.nbrs = nbrs
            self.masks = masks
        
        self.kernels = torch.nn.ParameterDict(kernels)

    def init_toep_kernel(self):
        mg_kernels = {}
        idx_lst = []
        x_lst = []

        n = 2*(self.n-1)+1
        x = torch.linspace(-1,1,n)[None,None]

        for l in range(self.k):
            n = x.shape[-1]
            print("level : ", l, " len : ", n, ' nbr : ', 2*self.m+1)
            # neighbors index
            if self.m > 0:
                idx = torch.arange((n-1)//2-self.m+1, (n-1)//2+self.m)[1::2]
                idx_lst.append(idx)
                x_lst.append(x[:,:,idx])
            x = injection1d(x)

        x_lst.append(x)
        xs = torch.concat(x_lst, axis=-1)

        idx_lst = idx_lst[::-1]
        x_lst = x_lst[::-1]
        mg_kernels["kernel"] = MLP([self.in_channels] + [self.hidden_channels]*4 + [self.out_channels], nonlinearity=self.act)
        return mg_kernels, idx_lst, xs

    def toep_kernel_approximation(self,x):     

        Amg = self.kernels['kernel'](self.xs)

        if self.m > 0:
            Abands = []
            for i in range(self.k):
                Abands.append(Amg[:,:,(self.m-1)*i:(self.m-1)*(i+1)])
            Abands = Abands[::-1]
            Ah = Amg[:,:,(self.m-1)*(i+1):]
        else:
            Ah = Amg

        for i in range(self.k):
            Ah = interp1d(Ah)
            if self.m > 0:
                Ah[:,:,self.nbrs[i]] = Abands[i]

        return Ah

    def toep_kernel_reconstruction(self):
        return self.toep_kernel_approximation(self.xs)
        
    def init_dd_kernel(self, mtype):
        kernels = {}
        xs = []
        idx_i_lst = []
        idx_j_lst = []
        mask_lst = []
        interval_dict = {}

        n = self.n
        xh = torch.linspace(-1,1,n)
        xhh = torch.cartesian_prod(xh, xh).reshape(n,n,-1)

        idx_h = torch.arange(n)
        idx_hh = torch.cartesian_prod(idx_h, idx_h).reshape(n,n,-1)

        for l in range(self.k):
            print("level : ", l, " len : ", n, ' nbr : ', 2*self.m+1)
            # fetch nbrs at level l
            idx_i, idx_j = fetch_nbrs(n, self.m)
            idx_mask = (idx_j >= 0) & (idx_j <= n-1)
            idx_j[idx_j < 0] = 0 
            idx_j[idx_j > n-1] = n-1
            
            idx_i_lst.append(idx_i)
            idx_j_lst.append(idx_j)
            mask_lst.append(idx_mask)            

            # extract band idx
            idx_band = idx_hh[idx_i, idx_j]
            idx_band_even = idx_band[::2,::2].reshape(-1,2)
            idx_band_odd = idx_band[1::2].reshape(-1,2)
            interval_dict[f'band_even_{self.k-l-1}'] = np.prod(idx_band[::2,::2].shape[:2])
            interval_dict[f'band_odd_{self.k-l-1}'] = np.prod(idx_band[1::2].shape[:2])

            # extract band xs 
            xband = xhh[idx_i, idx_j]
            xband_even = xband[::2,::2].reshape(-1,2)
            xband_odd = xband[1::2].reshape(-1,2)
            xs.append(xband_odd)
            xs.append(xband_even)

            print(l, idx_band_odd.shape[0])
            print(l, idx_band_even.shape[0])  

            xh = injection1d(xh[None,None])[0,0]
            idx_h = injection1d(idx_h[None,None])[0,0]
            n = (n-1)//2+1

            xhh = torch.cartesian_prod(xh, xh).reshape(n,n,-1)
            idx_hh = torch.cartesian_prod(idx_h, idx_h).reshape(n,n,-1)
        
        print('smooth ', np.prod(idx_hh.shape[:2]))
        interval_dict[f'full_coarse'] = np.prod(idx_hh.shape[:2])
        idx_i_lst = idx_i_lst[::-1]
        idx_j_lst = idx_j_lst[::-1]
        mask_lst = mask_lst[::-1]

        xs.append(xhh.reshape(-1,2))
        xs = xs[::-1]
        xs = torch.concat(xs)

        idx_interval = {}
        idx_start = 0
        idx_end = interval_dict['full_coarse']
        idx_interval['full_coarse'] = [idx_start, idx_end]
        print('smooth ', idx_end-idx_start)
        for i in range(self.k):
            idx_start = idx_end 
            idx_end = idx_start + interval_dict[f'band_even_{i}']
            idx_interval[f'band_even_{i}'] = [idx_start, idx_end]
            print(i, idx_end-idx_start)
            idx_start = idx_end 
            idx_end = idx_start + interval_dict[f'band_odd_{i}']
            idx_interval[f'band_odd_{i}'] = [idx_start, idx_end]
            print(i, idx_end-idx_start)

        if mtype == 'dd_mg':
            kernels['kernel'] = MLP([self.in_channels] + [self.hidden_channels]*4 + [self.out_channels], nonlinearity=self.act)
        elif mtype == 'lrdd_mg':
            kernels['psi'] = MLP([self.in_channels//2] + [self.hidden_channels]*4 + [self.r], nonlinearity=self.act)
            kernels['phi'] = MLP([self.in_channels//2] + [self.hidden_channels]*4 + [self.r], nonlinearity=self.act)
            
        xs = rearrange(xs, 'n c->n c 1')
        nbrs = [idx_i_lst, idx_j_lst]
        
        return kernels, xs, idx_interval, nbrs, mask_lst

    def dd_kernel_approximation(self, x):
        Kmg = self.kernels['kernel'](self.xs)
        Kmg = rearrange(Kmg, 'n c l -> l c n', c=1, l=1)

        nH = int(2**(np.log2(self.n-1) - self.k)+1)
        KHH = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,nH,nH)
        # fetch KHH band
        idx_i, idx_j = fetch_nbrs(nH, (self.m-1)//2+1) 
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > nH-1] = nH-1 
        Kband = KHH[:,:,idx_i, idx_j]        
        K_lst = [KHH]

        if self.m > 0:
            for i in range(self.k):
                Kband_even_interp = interp1d_cols(Kband)
                beven_start = self.intervals[f'band_even_{i}'][0]
                beven_end = self.intervals[f'band_even_{i}'][1]
                Kband_even = Kmg[:,:,beven_start:beven_end].reshape(1,1,-1, self.m+1)
                Kband_even_corr = Kband_even - Kband_even_interp[:,:,:,1:-1:2]

                Kband_even_interp[:,:,:,1:-1:2] = Kband_even
                bodd_start = self.intervals[f'band_odd_{i}'][0]
                bodd_end = self.intervals[f'band_odd_{i}'][1]
                Kband_odd_interp = (Kband_even_interp[:,:,:-1,:-2] + Kband_even_interp[:,:,1:,2:])/2 # interpolation
                Kband_odd = Kmg[:,:,bodd_start:bodd_end].reshape(1,1,-1, 2*self.m+1)
                Kband_odd_corr = Kband_odd - Kband_odd_interp

                K_lst.append([Kband_even_corr, Kband_odd_corr])

                Kband = interp1d_rows(Kband_even_interp)
                Kband[:,:,1:-1:2,1:-1] = Kband_odd
                Kband = Kband[:,:,:,self.m//2+1:self.m//2+self.m+1+2]

        return K_lst

    def lrdd_kernel_approximation(self, x):
        psi = self.kernels['psi'](x)
        phi = self.kernels['phi'](x)

        psi_mg = psi[:,:,self.idx[:,0]]
        phi_mg = phi[:,:,self.idx[:,1]]

        Kmg = (psi_mg * phi_mg).sum(axis=1)[None]

        n_lst = []
        n = self.n 
        n_lst.append(n)
        for i in range(self.k):
            n = (n-1)//2+1
            n_lst.append(n)
        n_lst = n_lst[::-1]

        Khh = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,n_lst[0],n_lst[0])
        K_lst = [Khh]
        for i in range(self.k):
            Khh = interp1d_rows(interp1d_cols(Khh))

            if self.m > 0:
                Kband_interp = Khh[:,:,self.nbrs[0][i],self.nbrs[1][i]]
                
                Kband_even_interp = Kband_interp[:,:,::2,::2]
                Kband_odd_interp = Kband_interp[:,:,1::2]
                Kband_even = Kmg[:,:,self.intervals[f'band_even_{i}'][0]:self.intervals[f'band_even_{i}'][1]].reshape(1,1,n_lst[i+1]//2+1, self.m+1)
                Kband_odd = Kmg[:,:,self.intervals[f'band_odd_{i}'][0]:self.intervals[f'band_odd_{i}'][1]].reshape(1,1,n_lst[i+1]//2, 2*self.m+1)

                Kband_even_corr = Kband_even - Kband_even_interp
                Kband_odd_corr = Kband_odd - Kband_odd_interp

                K_lst.append([Kband_even_corr, Kband_odd_corr])

                Khh[:,:,self.nbrs[0][i][::2,::2],self.nbrs[1][i][::2,::2]] += Kband_even_corr
                Khh[:,:,self.nbrs[0][i][1::2],self.nbrs[1][i][1::2]] += Kband_odd_corr

        return K_lst

    def dd_kernel_reconstruction(self):
        Kmg = self.kernels['kernel'](self.xs)
        Kmg = rearrange(Kmg, 'n c l -> l c n', c=1, l=1)

        nH = int(2**(np.log2(self.n-1) - self.k)+1)
        Khh = Kmg[:,:,self.intervals['full_coarse'][0]:self.intervals['full_coarse'][1]].reshape(1,1,nH,nH)

        for i in range(self.k):
            Khh = interp1d_rows(interp1d_cols(Khh))
            
            if self.m > 0:
                beven_start = self.intervals[f'band_even_{i}'][0]
                beven_end = self.intervals[f'band_even_{i}'][1]
                Kband_even = Kmg[:,:,beven_start:beven_end].reshape(1,1,-1, self.m+1)
                
                bodd_start = self.intervals[f'band_odd_{i}'][0]
                bodd_end = self.intervals[f'band_odd_{i}'][1]
                Kband_odd = Kmg[:,:,bodd_start:bodd_end].reshape(1,1,-1, 2*self.m+1)
                
                Khh[:,:,self.nbrs[0][i][::2,::2],self.nbrs[1][i][::2,::2]] = Kband_even
                Khh[:,:,self.nbrs[0][i][1::2],self.nbrs[1][i][1::2]] = Kband_odd
        
        return Khh

    def fetch_kernel(self):
        if self.mtype == 'toep_mg':
            A = self.toep_kernel_reconstruction()
            return A.detach().cpu().numpy()[0,0]
        elif self.mtype == 'dd_mg':
            K = self.dd_kernel_reconstruction()
            return K.detach().cpu().numpy()[0,0]

    def forward(self, u, x):
        if self.mtype == 'toep_mg':
            A = self.toep_kernel_approximation(x)
            w = toeplitz_matrix_vector_multiplication(A, u)
              
        if self.mtype == 'dd_mg':
            Ks = self.dd_kernel_approximation(x)
            h = x[0,0,1]-x[0,0,0]
            w = ml_matrix_vector_multiplication(Ks, u, self.nbrs[1], self.masks, h, k=self.k)

        if self.mtype == 'lrdd_mg':
            Ks = self.lrdd_kernel_approximation(x)
            h = x[0,0,1]-x[0,0,0]
            w = ml_matrix_vector_multiplication(Ks, u, self.nbrs[1], self.masks, h, k=self.k)

        return w



class GL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n, r, act='rational', mtype='toep_gl'):
        super(GL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        self.n = n
        self.mtype = mtype
        self.x = torch.linspace(-1,1,self.n)[None,None]

        if mtype == 'toep_gl':
            self.kernel = MLP([in_channels] + [hidden_channels]*4 + [out_channels], act)
            n = 2*(self.n-1)+1
            self.x = torch.linspace(-1,1,n)[None,None]
            print(self.x.shape)
        elif mtype == 'lr_gl':
            self.phi = MLP([in_channels//2] + [hidden_channels]*4 + [r], act)
            self.psi = MLP([in_channels//2] + [hidden_channels]*4 + [r], act)            
        elif mtype == 'gl':
            self.kernel = MLP([in_channels] + [hidden_channels]*4 + [out_channels], act)
            x = torch.linspace(-1,1,n)
            x = torch.cartesian_prod(x, x)
            self.x = rearrange(x, 'b c -> b c 1')
            print(self.x.shape)
    
    def kernel_approximation(self, x):
        if self.mtype == 'toep_gl':
            k = self.kernel(self.x)
            return k
        
        if self.mtype == 'lr_gl':
            phi = self.phi(self.x)
            psi = self.psi(self.x)
            return phi, psi

        if self.mtype == 'gl':
            k = self.kernel(self.x)
            k = k.reshape(1,1,self.n,self.n)
            return k

    def fetch_kernel(self):
        if self.mtype in ['toep_gl', 'gl']:
            K = self.kernel_approximation(self.x).detach().cpu().numpy()[0,0]
        elif self.mtype == 'lr_gl':
            phi, psi = self.kernel_approximation(self.x)
            K = torch.einsum('b r m, b r n -> b m n', phi, psi).detach().cpu().numpy()[0]      
        return K

    def forward(self, u, x):
        if self.mtype == 'toep_gl':
            k = self.kernel_approximation(x)
            w = toeplitz_matrix_vector_multiplication(k, u)
        
        if self.mtype == 'lr_gl':
            phi, psi = self.kernel_approximation(x)
            w = lowrank_matrix_vector_multiplication(phi, psi, u)
        
        if self.mtype == 'gl':
            k = self.kernel_approximation(x)
            w = full_matrix_vector_multiplication(k, u)

        return w
