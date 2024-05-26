import scipy
import numpy as np
import h5py
from einops import rearrange, repeat
from tqdm import tqdm
import torch

def Cosine(pts):
    x = pts[...,0]
    y = pts[...,1]
    return torch.cos(torch.pi/2 * (x-y).abs())

def Logarithm(pts):
    x = pts[...,0]
    y = pts[...,1]
    return torch.nan_to_num(torch.log((x-y).abs()), neginf=-8)

def Poisson(pts):
    x = (pts[...,0]+1)/2
    y = (pts[...,1]+1)/2
    return (x+y - (x-y).abs())/2 - x*y

def Expdecay(pts):
    x = pts[...,0]
    y = pts[...,1]
    r1 = ((x-0.5)**2 + (y-0.5)**2)**0.5
    r2 = ((x+0.5)**2 + (y+0.5)**2)**0.5
    return torch.exp(-r1/0.05) + torch.exp(-r2/0.05)

def DoubleHoles(pts):
    x = pts[...,0]
    y = pts[...,1]
    r1 = ((x-0.1)**2 + (y-0.1)**2)**0.5
    r2 = ((x+0.1)**2 + (y+0.1)**2)**0.5
    return torch.nan_to_num(torch.log(r1), neginf=-8) - torch.nan_to_num(torch.log(r2), neginf=-8)

def DiskInvdist(pts):
    x1 = pts[...,0]
    y1 = pts[...,1]
    x2 = pts[...,2]
    y2 = pts[...,3]

    mask = ((x1**2+y1**2) < 1) & ((x2**2+y2**2) < 1)

    k = ((x1 - x2)**2 + (y1-y2)**2) ** (-0.5)
    k = torch.nan_to_num(k, posinf=55) * mask

    return k

def DiskPoisson(pts):
    x1 = pts[...,0]
    y1 = pts[...,1]
    x2 = pts[...,2]
    y2 = pts[...,3]

    mask = ((x1**2+y1**2) < 1) & ((x2**2+y2**2) < 1)    

    k = 1/(4*torch.pi) * torch.log(
        ((x1 - x2)**2 + (y1-y2)**2) / \
        ((x1*y2-x2*y1)**2 + (x1*x2+y1*y2-1)**2))
    k = torch.nan_to_num(k, neginf=-1) * mask
    return k

def sample_dataset1d(n=12):
    r = 15 - n
    s = 2**r
    device = torch.device('cpu')
    res = str(2**n + 1)

    # 1d dataset
    for data_path in ['./dataset/f1d_32769_3.00e-01.mat', './dataset/f1d_32769_3.00e-02.mat']:

        try:
            raw_data = scipy.io.loadmat(data_path)
            F = raw_data['F']
        except:
            raw_data = h5py.File(data_path)
            F = raw_data['F'][()]
            F = np.transpose(F, axes=range(len(F.shape) - 1, -1, -1))

        F = torch.from_numpy(F[::s,:400]).float()

        # cosine kernel
        model = GreenNet1D(n=n, kernel=Cosine, device=device)
        model.eval_K()
        U = model.full_kint(F)

        F_out_path = data_path.replace('32769', res)
        U_out_path = F_out_path.replace('f1d', 'cosine')

        scipy.io.savemat(F_out_path, {'F':F.numpy()})
        print('save F at {:}'.format(F_out_path))
        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))

        # logarithm kernel
        model = GreenNet1D(n=n, kernel=Logarithm, device=device)
        model.eval_K()
        U = model.full_kint(F)

        U_out_path = F_out_path.replace('f1d', 'logarithm')

        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))

        # poisson kernel
        model = GreenNet1D(n=n, kernel=Poisson, device=device)
        model.eval_K()
        U = model.full_kint(F)

        U_out_path = F_out_path.replace('f1d', 'poisson')

        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))
        
        # exp_decay kernel
        model = GreenNet1D(n=n, kernel=Expdecay, device=device)
        model.eval_K()
        U = model.full_kint(F)

        U_out_path = F_out_path.replace('f1d', 'expdecay')

        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))

        # double holes kernel
        model = GreenNet1D(n=n, kernel=DoubleHoles, device=device)
        model.eval_K()
        U = model.full_kint(F)

        U_out_path = F_out_path.replace('f1d', 'doubleholes')

        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))
 

def sample_dataset2d(n=6):
    r = 9 - n
    s = 2**r
    device = torch.device('cpu')
    res = str(2**n + 1)
    
    # 2d dataset
    for data_path in ['./dataset/fdisk_513x513_2.00e-01.mat', './dataset/fdisk_513x513_5.00e-01.mat']:

        try:
            raw_data = scipy.io.loadmat(data_path)
            F = raw_data['F']
        except:
            raw_data = h5py.File(data_path)
            F = raw_data['F'][()]
            F = np.transpose(F, axes=range(len(F.shape) - 1, -1, -1))
        
        F = torch.from_numpy(F[:400,::s,::s]).float()

        # 2d invdist kernel
        model = GreenNet2D(n=n, kernel=DiskInvdist, device=device)
        model.eval_K()
        U = model.full_kint(F.reshape(400,-1)).reshape(400,res,res)

        F_out_path = data_path.replace('513x513', f'{res}x{res}')
        U_out_path = F_out_path.replace('fdisk', 'invdist')

        scipy.io.savemat(F_out_path, {'F':F.numpy()})
        print('save F at {:}'.format(F_out_path))
        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))

        # 2d poisson kernel
        model = GreenNet2D(n=n, kernel=DiskPoisson, device=device)
        model.eval_K()
        U = model.full_kint(F.reshape(400,-1)).reshape(400,res,res)
        U_out_path = F_out_path.replace('fdisk', 'poisson')

        scipy.io.savemat(F_out_path, {'F':F.numpy()})
        print('save F at {:}'.format(F_out_path))
        scipy.io.savemat(U_out_path, {'U':U.numpy()})
        print('save U at {:}'.format(U_out_path))

if __name__ == '__main__':
    from src.green_net import GreenNet1D, GreenNet2D

    # for n in [9, 10, 11]:
    #     sample_dataset1d(n)

    for n in [8, 9]:
        sample_dataset2d(n)
        


    