"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import argparse
from GreenMgNet.src.dataset import load_dataset_2d 
import torch.nn.functional as F
import torch 
import torch.nn as nn
from GreenMgNet.src.utils import rl2_error, init_records, save_hist, get_seed
import json
from tqdm import trange
from einops import rearrange
import numpy as np
import os 


################################################################
# fourier layer
################################################################
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
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        x = rearrange(x, 'b c m n -> b m n c')

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train Green-MgNet")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--res', type=int, default=129,
                        help='129, 257, 513')
    parser.add_argument('--model', type=str, default='fno',
                        help='model name. (fno)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--lr_adam', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--ep_adam', type=int, default=1000,
                        help='learning rate')
    parser.add_argument('--sch', action='store_true',
                        help='reduce rl on plateau scheduler')
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    get_seed(args.seed, printout=False)

    batch_size = 20
    learning_rate = args.lr_adam
    epochs = args.ep_adam

    modes = 12
    width = 32

    ################################################################
    # prepare log
    ################################################################
    device = torch.device(f'cuda:{args.device}')
    resolution = args.res

    data_root = f'/workdir/pde_data/green_learning/data2d_{resolution}'
    log_root = '/workdir/GreenMgNet/results/'
    hist_outpath, pred_outpath, nn_outpath, kernel_outpath, cfg_outpath = init_records(
        log_root, args)
    
    if os.path.exists(hist_outpath):
        print(f"{hist_outpath} file exists")
        print('-'*20)
        exit()

    print('output files:')
    print(hist_outpath)
    print(pred_outpath)
    print(nn_outpath)
    print(kernel_outpath)
    print(cfg_outpath)

    with open(cfg_outpath, 'w') as f:
        cfg_dict = vars(args)
        json.dump(cfg_dict, f)

    ################################################################
    # read data
    ################################################################
    train_loader, test_loader = load_dataset_2d(args.task, data_root, odd=False, normalize=True)

    ################################################################
    # build model
    ################################################################
    model = FNO2d(modes, modes, width).to(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True)

    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 5
    test_rl2_best = 5

    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description(
            "train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
        model.train()
        train_rl2 = 0

        for u, w in train_loader:
            u, w = u.to(device), w.to(device)
            w_ = model(u)
            loss = rl2_error(w_, w)

            optimizer.zero_grad()
            loss.backward() # use the l2 relative loss
            optimizer.step()
            # scheduler.step()
            train_rl2 += loss.item()

        model.eval()
        test_rl2 = 0.0
        with torch.no_grad():
            for u, w in test_loader:
                u, w = u.to(device), w.to(device)
                w_ = model(u)
                rl2 = rl2_error(w_, w)
                test_rl2 += rl2.item()

        train_rl2 = train_rl2/len(train_loader)
        test_rl2 = test_rl2/len(test_loader)
        sch.step(test_rl2)

        train_rl2_hist.append(train_rl2)
        test_rl2_hist.append(test_rl2)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
    
    print(f'save model at : {nn_outpath}')    
    torch.save(model.state_dict(), nn_outpath)
    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist)