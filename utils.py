import os 
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
import copy 

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

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

class ZeroLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.zeros_like(x)

class BandDiagLayer1d(torch.nn.Module):
    def __init__(self, width, bw):
        super(BandDiagLayer1d, self).__init__()
        self.conv = nn.Conv1d(width, width, bw, padding=bw//2, bias=False)
        self.diag = nn.Conv1d(width, width, 1, bias=False)
        
        # for param in self.diag.parameters():
        #     torch.nn.init.constant_(param, 0)
        #     param.data += torch.diag(torch.ones(
        #                     param.size(-1), dtype=torch.float))
        
        # for param in self.conv.parameters():
        #     torch.nn.init.constant_(param, 0)

    def forward(self, x):
        return self.conv(x) + self.diag(x)

class MultiLevelLayer1d(torch.nn.Module):
    def __init__(self, width, nlevel):
        super(MultiLevelLayer1d, self).__init__()

        self.nlevel = nlevel
        self.diag = nn.Conv1d(width, width, 1)
        
        if nlevel > 0:
            conv = nn.Conv1d(width, width, 3, padding=1)
            self.convs = nn.ModuleList([copy.deepcopy(conv) for _ in range(nlevel)])

    def forward(self, x):
        seq_len = x.size(-1)
        w = self.diag(x)

        if self.nlevel == 0:
            return w
        elif self.nlevel > 0:
            for i, conv in enumerate(self.convs):
                if i == 0:
                    w += conv(x)
                else:
                    xH = x[...,::2**i]
                    yH = conv(xH)
                    yh = F.interpolate(yH, seq_len, mode='linear')
                    w += yh 
            return w
        else:
            return torch.zeros_like(x).to(x)

class MultiLevelLayer2d(torch.nn.Module):
    def __init__(self, width, nlevel):
        super(MultiLevelLayer2d, self).__init__()

        self.nlevel = nlevel
        self.diag = nn.Conv2d(width, width, 1)
        
        if nlevel > 0:
            conv = nn.Conv2d(width, width, 3, padding=1)
            self.convs = nn.ModuleList([copy.deepcopy(conv) for _ in range(nlevel)])

    def forward(self, x):
        _, _, seq_lx, seq_ly = x.shape
        w = self.diag(x)

        if self.nlevel == 0:
            return w
        elif self.nlevel > 0:
            for i, conv in enumerate(self.convs):
                if i == 0:
                    w += conv(x)
                else:
                    xH = x[:,:,::2**i,::2**i]
                    yH = conv(xH)
                    yh = F.interpolate(yH, (seq_lx, seq_ly), mode='bilinear')
                    w += yh 
            return w
        else:
            return torch.zeros_like(x).to(x)


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("="*50)
        print(message)
        print("="*50)

def get_arguments(parser):
    # basic training settings
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Specifies learing rate for optimizer. (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--seed', type=int, default=0, 
                        help='random seed. (default: 0)')
    parser.add_argument('--save', type=int, default=0, 
                        help='save model. (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')    

    # ===================================
    # for dataset
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Size of each batch (default: 8)')
    parser.add_argument('--dataset_nm', type=str, default='burgers',
                        help='dataset name. (burgers, poisson, cosine, lnabs)')
    parser.add_argument('--ntrain', type=int, default=1000,
                        help='How many sequences in the training dataset.')
    parser.add_argument('--ntest', type=int, default=200, 
                        help='How many sequences in the training dataset.')
    parser.add_argument('--trasub', type=int, default=8,
                        help='The interval of when sample snapshots from train sequence')
    parser.add_argument('--testsub', type=int, default=8,
                        help='The interval of when sample snapshots from test sequence')
    
    
    # ====================================
    # for model
    parser.add_argument('--clevel', type=int, default=0, 
                        help='coarsen level for kernel integral calculation')
    parser.add_argument('--mlevel', type=int, default=0,
                        help='multi-level for ml residual type. ')
            
    return parser.parse_args()

def get_model_name(model_nm,
                   dataset_nm,
                   clevel,
                   resolution,
                   res_type,
                   proc_type,
                   enc_inp,
                   seed,
                   additional_str,
                   bw=3,
                   ):
    
    out_nm = []
    out_nm.append(model_nm)
    out_nm.append(dataset_nm)
    out_nm.append(f'c{clevel}')
    out_nm.append(f'r{resolution}')
    if res_type is None:
        out_nm.append('null')
    elif res_type == 'band':
        out_nm.append(f'bw{bw}')
    else:
        out_nm.append(res_type)
    out_nm.append(proc_type)
    out_nm.append(enc_inp)
    out_nm.append(f's{seed}')
    out_nm.append(f'{additional_str}')
    
    out_nm = '_'.join(out_nm)

    return out_nm

import nvsmi 
import json 
def profile_gpumem(gpu_id):
    gpu_proc = nvsmi.get_gpu_processes()
    gpu_proc_json = [json.loads(proc.to_json()) for proc in gpu_proc]
    valid_proc = [proc for proc in gpu_proc_json if int(proc['gpu_id']) == gpu_id]

    assert len(valid_proc) == 1
    return valid_proc[0]['used_memory']

def cuda_empty_cache(gpu_id):
    with torch.cuda.device('cuda:{:}'.format(gpu_id)):
        torch.cuda.empty_cache()

def pass_check(model_nm, res, clevel, mlevel, out_nm):
    if model_nm == 'ft2d':
        if res == 85:
            return False 
        elif res == 141:
            if clevel == 0:
                print('{:} : out of A100 mem'.format(out_nm))
                return True
            else:
                return False
        elif res == 211:
            if clevel == 0:
                print('{:} : out of A100 mem'.format(out_nm))
                return True
            elif clevel == 1:
                print('{:} : too long for training'.format(out_nm))
                return True
            else:
                return False
                