import torch 
import torch.nn.functional as F 
import torch.nn as nn

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

# class MLP(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels):
#         super(MLP, self).__init__()
#         self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
#         self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.gelu(x)
#         x = self.mlp2(x)
#         return x
    
class MLNO(nn.Module):
    def __init__(self, in_channels, out_channels, k, m, modes):
        super(MLNO, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.modes = modes 
        self.k = k 
        self.m = m 

        self.K_smooth = MLP([1,50,50,50,50,1], Rational)
        self.K_corrs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=2*m+1, padding=m, bias=False) for _ in range(k)])
        self.K_diag = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, u, x):

        # us = []
        # for i in range(self.k):
        #     us.append(u)
        #     u = F.interpolate(u, scale_factor=0.5)
        #     x = F.interpolate(x, scale_factor=0.5)
        
        # us = us[::-1]
        K = self.K_smooth(x)
        K_fft = torch.fft.rfft(K)
        u_fft = torch.fft.rfft(u)
        w = torch.fft.irfft(K_fft * u_fft)

        # for i in range(self.k):
        #     w = F.interpolate(w, scale_factor=2) + self.K_corrs[i](us[i])
        #     if i == self.k - 1:
        #         w += self.K_diag(us[i])

        return w 