import torch 
import torch.nn.functional as F 
from siren_pytorch import Sine
import torch.nn as nn

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
}

# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, aug=None):
        super(MLP, self).__init__()

        nonlinearity = Activations[nonlinearity]
        self.n_layers = len(layers) - 1
        self.aug = aug

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):

        if self.aug == 'aug1_1d':
            z = (x[...,[0]] < x[...,[1]] - 0.5)*2
            x = torch.concat([x,z], axis=-1)

        if self.aug == 'aug2_1d':
            m1 = x[...,[0]] < x[...,[1]]
            m2 = x[...,[0]] >= x[...,[1]]

        for _, l in enumerate(self.layers):
            x = l(x)

        if self.aug == 'aug2_1d':
            x[...,[0]] = x[...,[0]] * m1
            x[...,[1]] = x[...,[1]] * m2
            x = x.sum(axis=-1)

        return x
    
