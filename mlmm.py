import torch 
from einops import rearrange
from ops import grid2d_coords, grid4d_coords
from ops import fetch_nbrs2d 
from ops import cat2d_nbr_coords 
from ops import coord2idx2d, coord2idx4d 
from ops import interp1d, interp2d, interp1d_cols, interp1d_rows

class Grid2D:
    def __init__(self, nh, m):
        '''
        nh : number of nodes on each axis
        m : neighbor radius for a nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        self.m = m
        self.nh = nh
        self.init_grid()
        self.fetch_nbrs()
        self.h = 2/(self.nh-1)
        self.hh = self.h**2
    
    def init_grid(self):
        '''
        build a 2d mesh grid for input/output 2d functions
        x_h : physical coordinates (nh^2) x 2 
        coords_h : index coordinates (nh^2) x 2

        build a 4d mesh grid for 4d kernel functions
        x_hh : physical coordinates (nh^4) x 4
        coords_h : index coordinates (nh^4) x 4
        '''
        x_h, coords_h = grid2d_coords(self.nh)
        x_hh, coords_hh = grid4d_coords(self.nh)
        self.x_h = x_h 
        self.x_hh = x_hh 
        self.coords_h = coords_h 
        self.coords_hh = coords_hh
    
    def fetch_nbrs(self):
        '''
        ij_coords : host-neighbors pair index coordinates of i, (nh^2) x (2m+1)^2 x 4
        ij_idx : (4d)index of ij_coords, (nh^2) x (2m+1)^2 x 1
        x_ij : physical coords of ij_coords, (nh^2) x (2m+1)^2 x 4

        j_coords : neighbors index coordinates of i, (nh^2) x (2m+1)^2 x 2
        j_idx : (2d)index of j_coords, (nh^2) x (2m+1)^2 x 1
        '''        
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
    '''
    fetch locals of input function f_fine, based on interped coarse grid
    '''
    
    # nH: # nodes on each axis on COARSE grid
    # nh: # nodes on each axis on FINE grid
    nH = coarse_grid.nh
    nh = coarse_grid.nh * 2 - 1

    # coords_2I2J: index coordinates of nodes for which belongs to both COARSE grid and FINE grid
    #   COARSE grid I=(X,Y), J=(X',Y')
    #   FINE grid i=2I=(2X,2Y), j=(2X',2Y')
    coords_2I2J = coarse_grid.ij_coords * 2
    
    # coords_2I_j_xeven_yfull: i=(2X,2Y), j=(2X',y')
    coords_2I_j_xeven_yfull = interp1d_cols(coords_2I2J.permute(0,3,1,2)).permute(0,2,3,1).int()
    # coords_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',2Y'+1)
    coords_2I_j_xeven_yodd = coords_2I_j_xeven_yfull[:,:,1::2]
    # coords_2I_j_xodd_yfull: i=(2X,2Y), j=(2X'+1,y')
    coords_2I_j_xodd_yfull = (coords_2I_j_xeven_yfull[:,:-1] + coords_2I_j_xeven_yfull[:,1:])//2

    # coords_i_xeven_yeven_j : i=(2X,2Y), j=(x',y')
    coords_i_xeven_yeven_j = interp2d(coords_2I2J.permute(0,3,1,2)).permute(0,2,3,1).int()
    coords_i_xeven_yeven_j = rearrange(
        coords_i_xeven_yeven_j, '(m n) x y c -> m n x y c', m=nH,n=nH)
    
    # coords_i_xeven_yeven_j : i=(2X+1,2Y), j=(x',y')
    coords_i_xodd_yeven_j = (coords_i_xeven_yeven_j[:-1] + coords_i_xeven_yeven_j[1:])//2
    # coords_i_xeven_yeven_j : i=(2X,2Y+1), j=(x',y')
    coords_i_xeven_yodd_j = (coords_i_xeven_yeven_j[:,:-1] + coords_i_xeven_yeven_j[:,1:])//2
    # coords_i_xeven_yeven_j : i=(2X+1,2Y+1), j=(x',y')
    coords_i_xodd_yodd_j = (coords_i_xeven_yodd_j[:-1] + coords_i_xeven_yodd_j[1:])//2

    # only convert j, since f function is only defined on j
    f_ij_xeven_yodd = f_fine[coord2idx2d(coords_2I_j_xeven_yodd[...,2:], nh)].reshape(nH*nH,-1)
    f_ij_xodd_yfull = f_fine[coord2idx2d(coords_2I_j_xodd_yfull[...,2:], nh)].reshape(nH*nH,-1)
    f_i_xodd_yeven_j = f_fine[coord2idx2d(coords_i_xodd_yeven_j[...,2:], nh)].reshape(nH*(nH-1),-1)
    f_i_xeven_yodd_j = f_fine[coord2idx2d(coords_i_xeven_yodd_j[...,2:], nh)].reshape(nH*(nH-1),-1)
    f_i_xodd_yodd_j = f_fine[coord2idx2d(coords_i_xodd_yodd_j[...,2:], nh)].reshape((nH-1)*(nH-1),-1)

    return [
        f_ij_xeven_yodd, f_ij_xodd_yfull, 
        f_i_xodd_yeven_j, f_i_xeven_yodd_j, f_i_xodd_yodd_j]

def K_local_interp_4D(K_2I2J, K_2Ij):
    '''
    local interpolation of local K
    K_2I2J: nH x nH x (2m+1) x (2m+1), coarse nodes and their COARSE neighbors
    K_2Ij: nH x nH x (4m+1) x (4m+1), coarse nodes and their FINE neighbors
    '''

    nH = K_2I2J.shape[0]
    
    # Kernel values for i=(2X,2Y), j=(2X',2Y')
    K_2I2J = rearrange(K_2I2J, 'm n x y -> (m n) x y 1')
    # Kernel values for i=(2X,2Y), j=(2X',y')
    K_2I_j_xeven_yfull_ = interp1d_cols(K_2I2J.permute(0,3,1,2)).permute(0,2,3,1) 
    # Kernel values for i=(2X,2Y), j=(2X',2Y'+1)
    K_2I_j_xeven_yodd_ = K_2I_j_xeven_yfull_[:,:,1::2] 
    # Kernel values for i=(2X,2Y), j=(2X'+1,y')
    K_2I_j_xodd_yfull_ = (K_2I_j_xeven_yfull_[:,:-1] + K_2I_j_xeven_yfull_[:,1:])/2 
    
    # Kernel values for i=(2X,2Y), j=(x',y')
    K_i_xeven_yeven_j = K_2Ij
    # Kernel values for i=(2X,2Y), j=(x',y')
    K_i_xodd_yeven_j_ = (K_i_xeven_yeven_j[:-1] + K_i_xeven_yeven_j[1:])/2
    # Kernel values for i=(2X,2Y), j=(x',y')
    K_i_xeven_yodd_j_ = (K_i_xeven_yeven_j[:,:-1] + K_i_xeven_yeven_j[:,1:])/2
    # Kernel values for i=(2X,2Y), j=(x',y')
    K_i_xodd_yodd_j_ = (K_i_xeven_yodd_j_[:-1] + K_i_xeven_yodd_j_[1:])/2

    # reshape
    K_2I_j_xeven_yodd_ = K_2I_j_xeven_yodd_.reshape(nH*nH,-1)
    K_2I_j_xodd_yfull_ = K_2I_j_xodd_yfull_.reshape(nH*nH,-1)
    K_i_xodd_yeven_j_ = K_i_xodd_yeven_j_.reshape((nH-1)*nH,-1)
    K_i_xeven_yodd_j_ = K_i_xeven_yodd_j_.reshape(nH*(nH-1),-1)
    K_i_xodd_yodd_j_ = K_i_xodd_yodd_j_.reshape((nH-1)*(nH-1),-1)

    return [
        K_2I_j_xeven_yodd_, K_2I_j_xodd_yfull_,
        K_i_xodd_yeven_j_, K_i_xeven_yodd_j_, K_i_xodd_yodd_j_
    ]

def K_local_eval_4D(x_2Ij, kernel_func):
    '''
    local K evaluation
    x_2Ij: nH x nH x (4m+1) x (4m+1) x 4, coarse nodes and their FINE neighbors physical coordinates
    kernel_func : kernel function
    '''
    # x_2Ij: i=(2X,2Y), j=(x',y')
    nH = x_2Ij.shape[0]

    # x_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',2Y'+1)
    x_2I_j_xeven_yodd = x_2Ij[:,:,::2,1::2]
    # x_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',y')
    x_2I_j_xodd_yfull = x_2Ij[:,:,1::2]

    # x_i_xodd_yeven_j: i=(2X+1,2Y), j=(x',y')
    x_i_xodd_yeven_j = (x_2Ij[:-1] + x_2Ij[1:])/2
    # x_i_xeven_yodd_j: i=(2X,2Y+1), j=(x',y')
    x_i_xeven_yodd_j = (x_2Ij[:,:-1] + x_2Ij[:,1:])/2
    # x_i_xodd_yodd_j: i=(2X+1,2Y+1), j=(x',y')
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

