import torch 
import torch.nn.functional as F 
from siren_pytorch import Sine
import torch.nn as nn
from utils import toeplitz_matrix_vector_multiplicaiton, injection1d, interp1d

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
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x 

# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()

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

class Gauss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.zeros(1))
        self.sigma = nn.Parameter(torch.ones(1))
        
    def forward(self, input):
        output = self.s * torch.exp(-(input - self.mu)**2 / (2*self.sigma**2))
        return output

class ToepGreenMgNet(nn.Module):
    def __init__(self, in_channels, out_channels, lc):
        super(ToepGreenMgNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.lc = lc
        self.kernels = nn.ModuleList([MLP([in_channels,64,64,out_channels], nn.Tanh) for _ in range(lc)])
    
    def kernel_approximation(self, x):
        for i in range(self.lc):
            x = injection1d(x)

        n = x.shape[-1]        
        for j in range(self.lc):
            if j == 0:
                a = self.kernels[j](x)
                n_ = (n-1) // 4
            else:
                kernel_band = self.kernels[j](x)
                a[:,:,n_:n_+n] = kernel_band
                n_ = n_*2
            
            a = interp1d(a)
        return a 

    def forward(self, u, x):
        a = self.kernel_approximation(x)
        w = toeplitz_matrix_vector_multiplicaiton(a, u)

        return w 

class ToepGreenNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ToepGreenNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel = MLP([in_channels,50,50,50,50,out_channels], Rational)
    
    def kernel_approximation(self, x):
        a = self.kernel(x)
        return a 

    def forward(self, u, x):
        a = self.kernel_approximation(x)
        w = toeplitz_matrix_vector_multiplicaiton(a, u)

        return w 
