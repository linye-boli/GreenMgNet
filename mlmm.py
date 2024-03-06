import torch 
from einops import rearrange
from ops import grid2d_coords, grid4d_coords
from ops import fetch_nbrs2d 
from ops import cat2d_nbr_coords 
from ops import coord2idx2d, coord2idx4d 
from ops import interp1d, interp2d, interp1d_cols, interp1d_rows

class Grid2D:
    def __init__(self, n, m):
        self.n = n 
        self.m = m
        self.nh = 2**n + 1
        self.init_grid()
        self.fetch_nbrs()
        self.h = 2/(self.nh-1)
    
    def init_grid(self):
        x_h, coords_h = grid2d_coords(self.nh)
        x_hh, coords_hh = grid4d_coords(self.nh)
        self.x_h = x_h 
        self.x_hh = x_hh 
        self.coords_h = coords_h 
        self.coords_hh = coords_hh
    
    def fetch_nbrs(self):
        ij_coords = fetch_nbrs2d(
            self.coords_h, mx1=self.m, mx2=self.m,
            my1=self.m, my2=self.m)
        
        self.j_coords = ij_coords
        self.ij_coords = cat2d_nbr_coords(
            self.coords_h, ij_coords)
        
        self.j_idx = coord2idx2d(self.ij_coords[...,2:], self.nh)        
        self.ij_idx = coord2idx4d(self.ij_coords, self.nh)
        self.x_ij = torch.squeeze(self.x_hh[self.ij_idx])

def f_local_fetching_2D(f_fine, coarse_grid):
    nH = coarse_grid.nh
    nh = coarse_grid.nh * 2 - 1

    ij = coarse_grid.ij_coords * 2
    ij_xeven_yfull = interp1d_cols(ij.permute(0,3,1,2)).permute(0,2,3,1).int()
    ij_xeven_yodd = ij_xeven_yfull[:,:,1::2]
    ij_xodd_yfull = (ij_xeven_yfull[:,:-1] + ij_xeven_yfull[:,1:])//2

    f_ij_xeven_yodd = f_fine[coord2idx2d(ij_xeven_yodd[...,2:], nh)].reshape(nH*nH,-1)
    f_ij_xodd_yfull = f_fine[coord2idx2d(ij_xodd_yfull[...,2:], nh)].reshape(nH*nH,-1)

    i_xeven_yeven_j = interp2d(ij.permute(0,3,1,2)).permute(0,2,3,1).int()
    i_xeven_yeven_j = rearrange(
        i_xeven_yeven_j, '(m n) x y c -> m n x y c', m=nH,n=nH)
    i_xodd_yeven_j = (i_xeven_yeven_j[:-1] + i_xeven_yeven_j[1:])//2
    i_xeven_yodd_j = (i_xeven_yeven_j[:,:-1] + i_xeven_yeven_j[:,1:])//2
    i_xodd_yodd_j = (i_xeven_yodd_j[:-1] + i_xeven_yodd_j[1:])//2

    f_i_xodd_yeven_j = f_fine[coord2idx2d(i_xodd_yeven_j[...,2:], nh)].reshape(nH*(nH-1),-1)
    f_i_xeven_yodd_j = f_fine[coord2idx2d(i_xeven_yodd_j[...,2:], nh)].reshape(nH*(nH-1),-1)
    f_i_xodd_yodd_j = f_fine[coord2idx2d(i_xodd_yodd_j[...,2:], nh)].reshape((nH-1)*(nH-1),-1)

    return [
        f_ij_xeven_yodd, f_ij_xodd_yfull, 
        f_i_xodd_yeven_j, f_i_xeven_yodd_j, f_i_xodd_yodd_j]

def K_local_interp_4D(K_2I2J, K_2Ij):
    nH = K_2I2J.shape[0]
    K_2I2J = rearrange(K_2I2J, 'm n x y -> (m n) x y 1')
    K_2I_j_xeven_yfull_ = interp1d_cols(K_2I2J.permute(0,3,1,2)).permute(0,2,3,1) 
    K_2I_j_xeven_yodd_ = K_2I_j_xeven_yfull_[:,:,1::2] 
    K_2I_j_xodd_yfull_ = (K_2I_j_xeven_yfull_[:,:-1] + K_2I_j_xeven_yfull_[:,1:])/2 
    
    K_2I_j_xeven_yodd_ = K_2I_j_xeven_yodd_.reshape(nH*nH,-1)
    K_2I_j_xodd_yfull_ = K_2I_j_xodd_yfull_.reshape(nH*nH,-1)

    K_i_xeven_yeven_j = K_2Ij
    K_i_xodd_yeven_j_ = (K_i_xeven_yeven_j[:-1] + K_i_xeven_yeven_j[1:])/2
    K_i_xeven_yodd_j_ = (K_i_xeven_yeven_j[:,:-1] + K_i_xeven_yeven_j[:,1:])/2
    K_i_xodd_yodd_j_ = (K_i_xeven_yodd_j_[:-1] + K_i_xeven_yodd_j_[1:])/2

    K_i_xodd_yeven_j_ = K_i_xodd_yeven_j_.reshape((nH-1)*nH,-1)
    K_i_xeven_yodd_j_ = K_i_xeven_yodd_j_.reshape(nH*(nH-1),-1)
    K_i_xodd_yodd_j_ = K_i_xodd_yodd_j_.reshape((nH-1)*(nH-1),-1)

    return [
        K_2I_j_xeven_yodd_, K_2I_j_xodd_yfull_,
        K_i_xodd_yeven_j_, K_i_xeven_yodd_j_, K_i_xodd_yodd_j_
    ]

def K_local_eval_4D(x_2Ij, kernel_func):
    nH = x_2Ij.shape[0]
    x_2I_j_xeven_yodd = x_2Ij[:,:,::2,1::2]
    x_2I_j_xodd_yfull = x_2Ij[:,:,1::2]

    x_i_xodd_yeven_j = (x_2Ij[:-1] + x_2Ij[1:])/2
    x_i_xeven_yodd_j = (x_2Ij[:,:-1] + x_2Ij[:,1:])/2
    x_i_xodd_yodd_j = (x_i_xeven_yodd_j[:-1] + x_i_xeven_yodd_j[1:])/2

    K_2I_j_xeven_yodd = kernel_func(
        x_2I_j_xeven_yodd.reshape(-1,4)).reshape(nH*nH,-1)    
    K_2I_j_xodd_yfull = kernel_func(
        x_2I_j_xodd_yfull.reshape(-1,4)).reshape(nH*nH,-1)
    K_i_xodd_yeven_j = kernel_func(
        x_i_xodd_yeven_j.reshape(-1,4)).reshape(nH*(nH-1),-1)
    K_i_xeven_yodd_j = kernel_func(
        x_i_xeven_yodd_j.reshape(-1,4)).reshape(nH*(nH-1),-1)
    K_i_xodd_yodd_j = kernel_func(
        x_i_xodd_yodd_j.reshape(-1,4)).reshape((nH-1)*(nH-1),-1)

    K_2Ij = kernel_func(x_2Ij.reshape(-1,4)).reshape(nH,nH,-1)

    return [
        K_2I_j_xeven_yodd, K_2I_j_xodd_yfull, 
        K_i_xodd_yeven_j, K_i_xeven_yodd_j, 
        K_i_xodd_yodd_j], K_2Ij
