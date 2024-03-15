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

def fetch_nbrs1d(coords, mx1=2, mx2=2):
    n, d = coords.shape
    
    coords_nbrs_lst = []
    coords_max = coords.max()

    for i in range(-mx1, mx2+1):
        coords_nbr = coords.clone()
        coords_nbr[:,-1] += i

        coords_nbr[coords_nbr < 0] = 0
        coords_nbr[coords_nbr > coords_max] = coords_max

        coords_nbrs_lst.append(coords_nbr)

    # return coords_nbrs_lst
    return torch.cat(coords_nbrs_lst, axis=1).reshape(n, mx1+mx2+1, d)


# ------------------- 2D case ---------------------

def coord2idx2d(coord_xy, n):
    coord_x = coord_xy[...,[-2]]
    coord_y = coord_xy[...,[-1]]
    return coord_x * n + coord_y

def coord2idx4d(coord_ij, n):
    coord_i = coord2idx2d(coord_ij[...,-4:-2], n)
    coord_j = coord2idx2d(coord_ij[...,-2:], n)
    return coord2idx2d(torch.cat([coord_i, coord_j], axis=-1), n*n)

def grid1d_coords(nh):
    x_i = torch.linspace(-1,1,nh).reshape(-1,1)
    coords_i = torch.arange(nh).reshape(-1,1)
    return x_i, coords_i

def grid2d_coords(nh):
    xh = torch.linspace(-1,1,nh)
    idx_ix = torch.arange(nh)
    x_i = torch.cartesian_prod(xh, xh)                               # values : nh^2 x 2
    coords_i = torch.cartesian_prod(idx_ix, idx_ix)                  # coords : nh^2 x 2
    return x_i, coords_i

def grid4d_coords(nh):
    x_i, coords_i = grid2d_coords(nh)
    idx_i = coord2idx2d(coords_i, n=nh)            # index : nh^2 x 1
    # fine grid pts pairs           
    idx_ij = torch.cartesian_prod(idx_i.reshape(-1), idx_i.reshape(-1))                      # fine kernel index pairs: nh^2 x 1
    x_ij = torch.cat([x_i[idx_ij[:,0]], x_i[idx_ij[:,1]]], axis=1)
    coords_ij = torch.cat([coords_i[idx_ij[:,0]], coords_i[idx_ij[:,1]]], axis=1)
    return x_ij, coords_ij

def cat1d_nbr_coords(coords_i, coords_j):
    n, m, d = coords_j.shape 
    coords_i = repeat(coords_i, 'n d -> n m d', m=m)
    return torch.cat([coords_i, coords_j], axis=-1)


def cat2d_nbr_coords(coords_i, coords_j):
    n, m1, m2, d = coords_j.shape 
    coords_i = repeat(coords_i, 'n d -> n m1 m2 d', m1=m1, m2=m2)
    return torch.cat([coords_i, coords_j], axis=-1)


def fetch_nbrs2d(coords, mx1=2, mx2=2, my1=2, my2=2):
    n, d = coords.shape
    
    coords_nbrs_lst = []
    coords_max = coords.max()

    for i in range(-mx1, mx2+1):
        for j in range(-my1, my2+1):
            coords_nbr = coords.clone()
            coords_nbr[:,-2:-1] += i
            coords_nbr[:,-1:] += j            

            coords_nbr[coords_nbr < 0] = 0
            coords_nbr[coords_nbr > coords_max] = coords_max

            coords_nbrs_lst.append(coords_nbr)

    # return coords_nbrs_lst
    return torch.cat(coords_nbrs_lst, axis=1).reshape(n, mx1+mx2+1, my1+my2+1, d)

if __name__ == '__main__':
    from utils import l1_norm
    # test interp1d
    l = 8
    n = 2**l - 1
    lb = 0
    ub = 2*np.pi
    xh = torch.linspace(lb, ub, n+2)[1:-1][None][None]
    xH = xh[:,:,1::2]
    vh = torch.sin(xh)
    vH = torch.sin(xH)

    vh_ord2 = interp1d(vH, order=2)
    vh_ord4 = interp1d(vH, order=4)
    vh_ord6 = interp1d(vH, order=6)

    vh_ord2_mat = interp1d_matmul(vH, order=2)
    vh_ord4_mat = interp1d_matmul(vH, order=4)
    vh_ord6_mat = interp1d_matmul(vH, order=6)

    vH_ord2 = restrict1d(vh, order=2)
    vH_ord4 = restrict1d(vh, order=4)
    vH_ord6 = restrict1d(vh, order=6)

    vH_ord2_mat = restrict1d_matmul(vh, order=2)
    vH_ord4_mat = restrict1d_matmul(vh, order=4)
    vH_ord6_mat = restrict1d_matmul(vh, order=6)

    print('deconv interp error(L1Norm) : ')
    print('ord2 : ', l1_norm(vh_ord2,vh))
    print('ord4 : ', l1_norm(vh_ord4,vh))
    print('ord6 : ', l1_norm(vh_ord6,vh))
    
    print('matmul interp error : ')
    print('ord2 : ', l1_norm(vh_ord2_mat,vh))
    print('ord4 : ', l1_norm(vh_ord4_mat,vh))
    print('ord6 : ', l1_norm(vh_ord6_mat,vh))
    
    print('conv restrict error : ')
    print('ord2 : ', l1_norm(vH_ord2,vH))
    print('ord4 : ', l1_norm(vH_ord4,vH))
    print('ord6 : ', l1_norm(vH_ord6,vH))
    
    print('matmul restrict error : ')
    print('ord2 : ', l1_norm(vH_ord2_mat,vH))
    print('ord4 : ', l1_norm(vH_ord4_mat,vH))
    print('ord6 : ', l1_norm(vH_ord6_mat,vH))