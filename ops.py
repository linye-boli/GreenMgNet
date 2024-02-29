import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# injections 
def injection2d(Khh):
    KhH = torch.cat([Khh[...,[0]], Khh[...,1:-1][...,1::2], Khh[...,[-1]]], axis=-1)
    KHH = torch.cat([KhH[...,[0],:], KhH[...,1:-1,:][...,1::2,:], KhH[...,[-1],:]], axis=-2)
    return KHH

def injection4d(Khhhh):
    KhhhH = torch.cat([Khhhh[...,[0]], Khhhh[...,1:-1][...,1::2], Khhhh[...,[-1]]], axis=-1)
    KhhHH = torch.cat([KhhhH[...,[0],:], KhhhH[...,1:-1,:][...,1::2,:], KhhhH[...,[-1],:]], axis=-2)
    KhHHH = torch.cat([KhhHH[...,[0],:,:], KhhHH[...,1:-1,:,:][...,1::2,:,:], KhhHH[...,[-1],:,:]], axis=-3)
    KHHHH = torch.cat([KhHHH[...,[0],:,:,:], KhHHH[...,1:-1,:,:,:][...,1::2,:,:,:], KhHHH[...,[-1],:,:,:]], axis=-4)

    return KHHHH

def injection1d_cols(Khh):
    KhH = torch.cat([Khh[...,[0]], Khh[...,1:-1][...,1::2], Khh[...,[-1]]], axis=-1)
    return KhH

def injection1d_rows(Khh, order=2):
    # KhH : (batch, c, H, h)

    Khh = rearrange(Khh, 'b c I j -> b c j I')
    KHh = injection1d_cols(Khh)
    KHh = rearrange(KHh, 'b c j i -> b c i j')

    return KHh

def injection1d(vh):
    vH = torch.cat([vh[...,[0]], vh[...,1:-1][...,1::2], vh[...,[-1]]], axis=-1)
    return vH

# interploation
def interp1d(vH, order=2):
    # vH : (batch, c, H)

    bsz, c, h = vH.shape
    vH = rearrange(vH, 'b c h -> (b c) 1 h')

    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vH)
        w = 1/2
        s = 2
        p = 0

    if order == 4:
        kernel = torch.tensor([[[-1,0,9,16,9,0,-1]]]).to(vH)
        w = 1/16
        s = 2
        p = 2
    
    if order == 6:
        kernel = torch.tensor([[[3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3]]]).to(vH)
        w = 1/256
        s = 2 
        p = 4

    vh = w * F.conv_transpose1d(vH, kernel, stride=s, padding=p)

    vh = rearrange(vh, '(b c) 1 h -> b c h', b=bsz, c=c, h=h*2+1)[..., 1:-1]
    return  vh 

def interp1d_mat(n, order=2):
    if order == 2:
        mat = torch.zeros((2*n+1, n))
        kernel = torch.tensor([1., 2., 1.])/2
        klen = 3
    elif order == 4:
        mat = torch.zeros((2*n+1+4, n))
        kernel = torch.tensor([-1., 0, 9, 16, 9, 0, -1.])/16
        klen = 7
    elif order == 6:
        mat = torch.zeros((2*n+1+8, n))
        kernel = torch.tensor([3., 0, -25, 0, 150, 256, 150, 0, -25, 0, 3])/256
        klen = 11

    for i in range(n):
        mat[2*i:2*i+klen, i] = kernel 
        
    if order == 4:
        mat = mat[2:-2]
    elif order == 6:
        mat = mat[4:-4]

    return mat

def interp1d_matmul(vH, interpmat=None, order=2):
    n = vH.shape[-1]
    if interpmat is None:
        interpmat = interp1d_mat(n, order)
    return torch.einsum('mn, bcn->bcm', interpmat, vH)

def restrict1d_mat(n, order=2):
    interpmat = interp1d_mat(n, order)/2
    return interpmat.T

def restrict1d_matmul(vh, restrictmat=None, order=2):
    n = vh.shape[-1]
    if restrictmat is None:
        restrictmat = restrict1d_mat((n-1)//2, order)
    return torch.einsum('mn, bcn->bcm', restrictmat, vh)


def restrict1d(vh, order=2):
    # vh : (batch, c, h)

    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vh)
        w = 1/4
        s = 2
        p = 0
    
    if order == 4:
        kernel = torch.tensor([[[-1,0,9,16,9,0,-1]]]).to(vh)
        w = 1/32
        s = 2
        p = 2
    
    if order == 6:
        kernel = torch.tensor([[[3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3]]]).to(vh)
        w = 1/512
        s = 2
        p = 4

    vH = w * F.conv1d(vh[...,1:-1], kernel, stride=s, padding=p)
    vH = torch.cat([vh[...,[0]], vH, vh[...,[-1]]], axis=-1)
    return vH

def interp1d_cols(KhH, order=2):
    # KhH : (batch, c, i, J)

    bsz, c, i, J = KhH.shape
    KhH = rearrange(KhH, 'b c i J -> (b i) c J')
    Khh = interp1d(KhH, order=order)
    Khh = rearrange(Khh, '(b i) c j-> b c i j', b = bsz, c=c, i=i, j=2*J-1)

    return Khh

def restrict1d_cols(KhH, order=2):
    # KhH : (batch, c, i, J)

    bsz, c, i, J = KhH.shape
    KhH = rearrange(KhH, 'b c i J -> (b i) c J')
    Khh = restrict1d(KhH, order=order)
    Khh = rearrange(Khh, '(b i) c j-> b c i j', b = bsz, c=c, i=i, j=(J-1)//2+1)
    return Khh

def interp1d_rows(KHh, order=2):
    # KhH : (batch, c, H, h)

    KhH = rearrange(KHh, 'b c I j -> b c j I')
    Khh = interp1d_cols(KhH, order)
    Khh = rearrange(Khh, 'b c j i -> b c i j')

    return Khh

def restrict1d_rows(KHh, order=2):
    # KhH : (batch, c, H, h)

    KhH = rearrange(KHh, 'b c I j -> b c j I')
    Khh = restrict1d_cols(KhH, order)
    Khh = rearrange(Khh, 'b c j i -> b c i j')

    return Khh

def interp2d(KHH, order=2):
    KHh = interp1d_rows(KHH, order)
    Khh = interp1d_cols(KHh, order)
    return Khh

def restrict2d(KHH, order=2):
    KHh = restrict1d_rows(KHH, order)
    Khh = restrict1d_cols(KHh, order)
    return Khh

def fetch_nbrs(n, m):
    # n : int, lenth of inputs,
    # m : int, radius of window
 
    idx_h = torch.arange(n)
    idx_nbrs = torch.arange(-m, m+1)
    idx_j = torch.cartesian_prod(idx_h, idx_nbrs).sum(axis=1).reshape(-1, 2*m+1) # n x 2m+1
    idx_i = repeat(idx_h, 'i -> i m', m=2*m+1)

    return idx_i, idx_j