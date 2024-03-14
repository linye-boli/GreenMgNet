import torch 
from einops import rearrange
from ops import grid1d_coords, grid2d_coords, grid4d_coords
from ops import fetch_nbrs2d, fetch_nbrs1d
from ops import cat2d_nbr_coords, cat1d_nbr_coords
from ops import coord2idx2d, coord2idx4d 
from ops import interp1d, interp2d, interp1d_cols, interp1d_rows
from ops import restrict1d, restrict2d
from ops import injection1d, injection2d

# 1D MLMM algorithm
class DD_Grid1D:
    def __init__(self, nh, m, device):
        '''
        nh : number of nodes on each axis
        m : neighbor radius for a nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        self.m = m
        self.nh = nh
        self.h = 2/(self.nh-1)
        self.hh = self.h
        self.device = device
        self.init_grid_h()
        self.fetch_nbrs()
        
    
    def init_grid_h(self):
        '''
        build a 1d mesh grid for input/output 1d functions
        x_h : physical coordinates nh x 1
        coords_h : index coordinates nh x 1
        '''
        x_h, coords_h = grid1d_coords(self.nh)
        self.x_h = x_h.to(self.device)
        self.coords_h = coords_h.to(self.device)

    def init_grid_hh(self):
        '''
        build a 2d mesh grid for input/output 2d functions
        x_h : physical coordinates (nh^2) x 2 
        coords_h : index coordinates (nh^2) x 2
        '''
        x_hh, coords_hh = grid2d_coords(self.nh)
        self.x_hh = x_hh.to(self.device)
        self.coords_hh = coords_hh.to(self.device)
    
    def fetch_nbrs(self):
        ij_coords = fetch_nbrs1d(
            self.coords_h, mx1=self.m, mx2=self.m)
        self.j_coords = ij_coords 
        self.j_idx = ij_coords 
        self.x_j = torch.squeeze(self.x_h[self.j_idx], dim=-1)

        self.ij_coords = cat1d_nbr_coords(self.coords_h, ij_coords)
        
        self.ij_idx = coord2idx2d(self.ij_coords, self.nh)
        self.x_ij = cat1d_nbr_coords(self.x_h, self.x_j)
    
    def fetch_local_idx(self):
        # nH: # nodes on each axis on COARSE grid
        # nh: # nodes on each axis on FINE grid
        nH = self.nh 
        nh = nH * 2 - 1

        # coords_2I2J: index coordinates of nodes for which belongs to both COARSE grid and FINE grid
        #   COARSE grid I,J
        #   FINE grid i=2I, j=2J
        coords_2I2J = self.ij_coords * 2

        # coords_i_even_j : i=2I, j=j
        coords_2Ij = interp1d(coords_2I2J.permute(0,2,1).float()).permute(0,2,1).int()

        # coords_2I_j_odd: i=2I, j=2J+1
        coords_2I_j_odd = coords_2Ij[:,1::2]
        
        # coords_i_xeven_yeven_j : i=(2X+1,2Y), j=(x',y')
        coords_i_odd_j = (coords_2Ij[:-1] + coords_2Ij[1:])//2

        # idx_2Ij = coords_2Ij
        idx_2I_j_odd = coords_2I_j_odd      
        idx_i_odd_j = coords_i_odd_j
        
        idx_local_even = idx_2I_j_odd.reshape(-1,2).T
        idx_local_odd = idx_i_odd_j.reshape(-1,2).T
        return idx_local_even, idx_local_odd
    
    def fetch_K_local_x(self):
        x_2I2J = self.x_ij
        x_2Ij = interp1d(x_2I2J.permute(0,2,1)).permute(0,2,1)

        # x_2I_j_odd: i=2I, j=j
        x_2I_j_odd = x_2Ij[:,1::2]
        # x_i_odd_j: i=2I+1, j=j
        x_i_odd_j = (x_2Ij[:-1] + x_2Ij[1:])/2
        return x_2I_j_odd, x_i_odd_j

class DD_MLMM1D:
    def __init__(self, n, m, k, device):
        '''
        n : total level
        m : neighbor radius for a nodes on each axis
        k : coarse level
        '''
        self.n = n 
        self.m = m 
        self.k = k
        self.device = device
        self.build_ml_grids()
        self.fetch_eval_pts()

    def build_ml_grids(self):
        '''
        build multi-level grids
        '''
        ml_grids = []
        for l in range(self.k+1):
            nh = 2**(self.n-l)+1
            ml_grids.append(DD_Grid1D(nh, self.m, self.device))

            if l == 0:
                nfinest = nh
            if l == self.k:
                ncoarest = nh
                if nfinest ** 0.5 > ncoarest:
                    print("finest grid : {:}".format(nfinest))
                    print("coarest grid : {:}".format(ncoarest))
                    print('too coarse warning')

        self.ml_grids = ml_grids

    def restrict_ml_f(self, f_h):
        '''
        restrict f into multi-level
        '''
        ml_f = [f_h]
        for _ in range(self.k):
            f_h = restrict1d(f_h)
            ml_f.append(f_h)
        self.ml_f = ml_f
    
    def fetch_eval_pts(self):
        '''
        fetch coarest grid pts and neighbor pts on each grids
        '''
        # coarest grid
        self.ml_grids[-1].init_grid_hh()
        self.coarest_pts = self.ml_grids[-1].x_hh.reshape(-1,2)
        # local pts
        local_pts = []
        local_idx = []

        for l in range(self.k+1):
            x_2I_j_odd, x_i_odd_j = self.ml_grids[-1-l].fetch_K_local_x()
            idx_corr_even, idx_corr_odd = self.ml_grids[-1-l].fetch_local_idx()
            local_pts.append([x_2I_j_odd, x_i_odd_j])
            local_idx.append([idx_corr_even, idx_corr_odd])

        self.local_pts = local_pts
        self.local_idx = local_idx

    def eval_ml_K(self, kernel_func):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarest grid
        K_HH = kernel_func(self.coarest_pts)
        self.K_HH = K_HH 
        
        # local pts
        K_locals = []
        for l in range(self.k+1):
            m, n, _ = self.local_pts[l][0].shape
            K_local_even = kernel_func(self.local_pts[l][0].reshape(-1,2)).reshape(m, n)
            m, n, _ = self.local_pts[l][1].shape
            K_local_odd = kernel_func(self.local_pts[l][1].reshape(-1,2)).reshape(m, n)
            K_locals.append([K_local_even, K_local_odd])        
        self.K_locals = K_locals

    def coarest_full_kint(self):
        '''
        kernel integral on coarest level by dense matrix-vector product
        '''
        nH = self.ml_grids[-1].nh
        HH = self.ml_grids[-1].hh
        KHH = self.K_HH.reshape(nH, nH)
        fH = torch.squeeze(self.ml_f[-1]).T
        uH = HH * (KHH @ fH).T
        return uH

    def local_interp_K(self, K_2I2J, K_2Ij):
        '''
        local interpolation of local K
        K_2I2J: nH x (2m+1), coarse nodes and their COARSE neighbors
        K_2Ij: nH x (4m+1), coarse nodes and their FINE neighbors
        '''

        nH = K_2I2J.shape[0]
        
        # Kernel values for i=2I, j=2J+1
        K_2I_j_odd_ = (K_2I2J[:,:-1] + K_2I2J[:,1:])/2
        
        # Kernel values for i=2I+1, j=j
        K_i_odd_j_ = (K_2Ij[1:] + K_2Ij[:-1])/2

        K_local_even = K_2I_j_odd_.reshape(-1)
        K_local_odd = K_i_odd_j_.reshape(-1)
        
        return K_local_even, K_local_odd
    
    def local_assemble_K(self, K_IJ, K_local_even, K_local_odd):
        nH, M = K_IJ.shape
        K_ij = torch.zeros(2*nH-1, 2*M-1).to(K_IJ)

        K_ij[::2,::2] += K_IJ
        K_ij[::2,1::2] += K_local_even
        K_ij[1::2] += K_local_odd

        return K_ij

    def ml_kint(self):
        u_h = self.coarest_full_kint()
        K_IJ = torch.squeeze(self.K_HH[self.ml_grids[-1].ij_idx])

        for l in range(1,self.k+1):
            nh = self.ml_grids[-1-l].nh
            hh = self.ml_grids[-1-l].hh
            f_h = torch.squeeze(self.ml_f[-1-l]).T
            
            # local evaluation and assemblation
            K_local_even, K_local_odd = self.K_locals[l-1]
            idx_corr_even, idx_corr_odd = self.local_idx[l-1]
            K_ij = self.local_assemble_K(K_IJ, K_local_even, K_local_odd)
            K_2Ij = K_ij[::2]
            
            # local kernel interpolation
            K_local_even_, K_local_odd_ = self.local_interp_K(K_IJ, K_2Ij)

            # calculate difference
            K_corr_even = K_local_even.reshape(-1) - K_local_even_
            K_corr_odd = K_local_odd.reshape(-1) - K_local_odd_

            # correct even 
            K_corr_even_sparse = torch.sparse_coo_tensor(idx_corr_even, K_corr_even,(nh,nh))            
            u_corr_ = torch.sparse.mm(K_corr_even_sparse, f_h).T
            u_corr_ = hh * injection1d(u_corr_[None,None])[0,0]
            u_h_ = u_h + u_corr_
            u_h_ = interp1d(u_h_[:,None])[:,0]

            # correct odd 
            K_corr_odd_sparse = torch.sparse_coo_tensor(idx_corr_odd, K_corr_odd,(nh,nh))
            u_corr_ = hh*torch.sparse.mm(K_corr_odd_sparse, f_h).T
            u_h_ = u_h_ + u_corr_

            # get new K_IJ, u_h
            K_IJ = K_ij[:,self.m:-self.m]
            u_h = u_h_

        return u_h

# 2D MLMM algorithm

class DD_Grid2D:
    def __init__(self, nh, m, device):
        '''
        nh : number of nodes on each axis
        m : neighbor radius for a nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        self.m = m
        self.nh = nh
        self.h = 2/(self.nh-1)
        self.hh = self.h**2
        self.device = device
        self.init_grid_h()
        self.fetch_nbrs()
    
    def init_grid_h(self):
        '''
        build a 2d mesh grid for input/output 2d functions
        x_h : physical coordinates (nh^2) x 2 
        coords_h : index coordinates (nh^2) x 2
        '''
        x_h, coords_h = grid2d_coords(self.nh)
        self.x_h = x_h.to(self.device)
        self.coords_h = coords_h.to(self.device)
    
    def init_grid_hh(self):
        '''
        build a 4d mesh grid for 4d kernel functions
        x_hh : physical coordinates (nh^4) x 4
        coords_h : index coordinates (nh^4) x 4
        '''
        x_hh, coords_hh = grid4d_coords(self.nh)
        self.x_hh = x_hh.to(self.device) 
        self.coords_hh = coords_hh.to(self.device)
    
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
        self.j_idx = coord2idx2d(self.j_coords, self.nh)
        self.x_j = torch.squeeze(self.x_h[self.j_idx])

        self.ij_coords = cat2d_nbr_coords(self.coords_h, ij_coords)
        self.ij_idx = coord2idx4d(self.ij_coords, self.nh)
        self.x_ij = cat2d_nbr_coords(
            self.x_h, self.x_j)
               
    def fetch_local_idx(self):
        # nH: # nodes on each axis on COARSE grid
        # nh: # nodes on each axis on FINE grid
        nH = self.nh 
        nh = nH * 2 - 1

        # coords_2I2J: index coordinates of nodes for which belongs to both COARSE grid and FINE grid
        #   COARSE grid I=(X,Y), J=(X',Y')
        #   FINE grid i=2I=(2X,2Y), j=(2X',2Y')
        coords_2I2J = self.ij_coords * 2
        # coords_2I_j_xeven_yfull: i=(2X,2Y), j=(2X',y')
        coords_2I_j_xeven_yfull = interp1d_cols(coords_2I2J.float().permute(0,3,1,2)).permute(0,2,3,1).int()
        # coords_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',2Y'+1)
        coords_2I_j_xeven_yodd = coords_2I_j_xeven_yfull[:,:,1::2]
        # coords_2I_j_xodd_yfull: i=(2X,2Y), j=(2X'+1,y')
        coords_2I_j_xodd_yfull = (coords_2I_j_xeven_yfull[:,:-1] + coords_2I_j_xeven_yfull[:,1:])//2

        # coords_i_xeven_yeven_j : i=(2X,2Y), j=(x',y')
        coords_i_xeven_yeven_j = interp2d(coords_2I2J.float().permute(0,3,1,2)).permute(0,2,3,1).int()
        coords_i_xeven_yeven_j = rearrange(
            coords_i_xeven_yeven_j, '(m n) x y c -> m n x y c', m=nH,n=nH)
        
        # coords_i_xeven_yeven_j : i=(2X+1,2Y), j=(x',y')
        coords_i_xodd_yeven_j = (coords_i_xeven_yeven_j[:-1] + coords_i_xeven_yeven_j[1:])//2
        # coords_i_xeven_yeven_j : i=(2X,2Y+1), j=(x',y')
        coords_i_xeven_yodd_j = (coords_i_xeven_yeven_j[:,:-1] + coords_i_xeven_yeven_j[:,1:])//2
        # coords_i_xeven_yeven_j : i=(2X+1,2Y+1), j=(x',y')
        coords_i_xodd_yodd_j = (coords_i_xeven_yodd_j[:-1] + coords_i_xeven_yodd_j[1:])//2

        idx_2I_j_xeven_yodd_i = coord2idx2d(coords_2I_j_xeven_yodd[...,:2], nh).reshape(1,-1)
        idx_2I_j_xeven_yodd_j = coord2idx2d(coords_2I_j_xeven_yodd[...,2:], nh).reshape(1,-1)

        idx_2I_j_xodd_yfull_i = coord2idx2d(coords_2I_j_xodd_yfull[...,:2], nh).reshape(1,-1)
        idx_2I_j_xodd_yfull_j = coord2idx2d(coords_2I_j_xodd_yfull[...,2:], nh).reshape(1,-1)
        
        idx_i_xodd_yeven_j_i = coord2idx2d(coords_i_xodd_yeven_j[...,:2], nh).reshape(1,-1)
        idx_i_xodd_yeven_j_j = coord2idx2d(coords_i_xodd_yeven_j[...,2:], nh).reshape(1,-1)
        
        idx_i_xeven_yodd_j_i = coord2idx2d(coords_i_xeven_yodd_j[...,:2], nh).reshape(1,-1)
        idx_i_xeven_yodd_j_j = coord2idx2d(coords_i_xeven_yodd_j[...,2:], nh).reshape(1,-1)

        idx_i_xodd_yodd_j_i = coord2idx2d(coords_i_xodd_yodd_j[...,:2], nh).reshape(1,-1)
        idx_i_xodd_yodd_j_j = coord2idx2d(coords_i_xodd_yodd_j[...,2:], nh).reshape(1,-1)

        idx_2I_j_xeven_yodd = torch.concat([idx_2I_j_xeven_yodd_i, idx_2I_j_xeven_yodd_j], axis=0)
        idx_2I_j_xodd_yfull = torch.concat([idx_2I_j_xodd_yfull_i, idx_2I_j_xodd_yfull_j], axis=0)
        idx_i_xodd_yeven_j = torch.concat([idx_i_xodd_yeven_j_i, idx_i_xodd_yeven_j_j], axis=0)
        idx_i_xeven_yodd_j = torch.concat([idx_i_xeven_yodd_j_i, idx_i_xeven_yodd_j_j], axis=0)
        idx_i_xodd_yodd_j = torch.concat([idx_i_xodd_yodd_j_i, idx_i_xodd_yodd_j_j], axis=0)
        
        idx_local_even = torch.concat([idx_2I_j_xeven_yodd, idx_2I_j_xodd_yfull], axis=1)
        idx_local_odd = torch.concat([idx_i_xodd_yeven_j, idx_i_xeven_yodd_j, idx_i_xodd_yodd_j], axis=1)
        return idx_local_even, idx_local_odd

    def fetch_K_local_x(self):
        nH = self.nh 
        x_2I2J = self.x_ij
        x_2Ij = interp2d(x_2I2J.permute(0,3,1,2)).permute(0,2,3,1)
        x_2Ij = rearrange(x_2Ij, '(m n) x y c-> m n x y c', m=nH, n=nH)

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

        return [x_2I_j_xeven_yodd, x_2I_j_xodd_yfull], [x_i_xodd_yeven_j, x_i_xeven_yodd_j, x_i_xodd_yodd_j]

class DD_MLMM2D:
    def __init__(self, n, m, k, device):
        '''
        n : total level
        m : neighbor radius for a nodes on each axis
        k : coarse level
        '''
        self.n = n 
        self.m = m 
        self.k = k
        self.device = device
        self.build_ml_grids()
        self.fetch_eval_pts()
    
    def build_ml_grids(self):
        '''
        build multi-level grids
        '''
        ml_grids = []
        for l in range(self.k+1):
            nh = 2**(self.n-l)+1
            ml_grids.append(DD_Grid2D(nh, self.m, self.device))

            if l == 0:
                nfinest = nh
            if l == self.k:
                ncoarest = nh
                if nfinest ** 0.5 > ncoarest:
                    print("finest grid : {:}".format(nfinest))
                    print("coarest grid : {:}".format(ncoarest))
                    print('too coarse warning')

        self.ml_grids = ml_grids
    
    def fetch_eval_pts(self):
        '''
        fetch coarest grid pts and neighbor pts on each grids
        '''
        # coarest grid
        self.ml_grids[-1].init_grid_hh()
        self.coarest_pts = self.ml_grids[-1].x_hh.reshape(-1,4)

        # local pts
        local_pts = []
        local_idx = []

        for l in range(self.k+1):
            x_corr_even, x_corr_odd = self.ml_grids[-1-l].fetch_K_local_x()
            idx_corr_even, idx_corr_odd = self.ml_grids[-1-l].fetch_local_idx()
            local_pts.append([x_corr_even, x_corr_odd])
            local_idx.append([idx_corr_even, idx_corr_odd])

        self.local_pts = local_pts
        self.local_idx = local_idx
    
    def restrict_ml_f(self, f_h):
        '''
        restrict f into multi-level
        '''
        nh = self.ml_grids[0].nh
        f_h = rearrange(f_h, '(m n) b->b 1 m n', m=nh, n=nh)
        ml_f = [f_h]
        for _ in range(self.k):
            f_h = restrict2d(f_h)
            ml_f.append(f_h)
        self.ml_f = ml_f
    
    def eval_ml_K(self, kernel_func):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarse grid 
        K_HH = kernel_func(self.coarest_pts)
        self.K_HH = K_HH

        # local pts
        K_locals = []
        for l in range(self.k+1):
            x_2I_j_xeven_yodd, x_2I_j_xodd_yfull = self.local_pts[l][0]
            x_i_xodd_yeven_j, x_i_xeven_yodd_j, x_i_xodd_yodd_j = self.local_pts[l][1]

            mx, my, nx, ny, _ = x_2I_j_xeven_yodd.shape
            K_2I_j_xeven_yodd = kernel_func(
                x_2I_j_xeven_yodd.reshape(-1,4)).reshape(mx, my, nx, ny)
            
            mx, my, nx, ny, _ = x_2I_j_xodd_yfull.shape
            K_2I_j_xodd_yfull = kernel_func(
                x_2I_j_xodd_yfull.reshape(-1,4)).reshape(mx, my, nx, ny)
            
            mx, my, nx, ny, _ = x_i_xodd_yeven_j.shape
            K_i_xodd_yeven_j = kernel_func(
                x_i_xodd_yeven_j.reshape(-1,4)).reshape(mx, my, nx, ny)
            
            mx, my, nx, ny, _ = x_i_xeven_yodd_j.shape
            K_i_xeven_yodd_j = kernel_func(
                x_i_xeven_yodd_j.reshape(-1,4)).reshape(mx, my, nx, ny)
            
            mx, my, nx, ny, _ = x_i_xodd_yodd_j.shape
            K_i_xodd_yodd_j = kernel_func(
                x_i_xodd_yodd_j.reshape(-1,4)).reshape(mx, my, nx, ny)
            
            K_local_even = [K_2I_j_xeven_yodd, K_2I_j_xodd_yfull]
            K_local_odd = [K_i_xodd_yeven_j, K_i_xeven_yodd_j, K_i_xodd_yodd_j]

            K_locals.append([K_local_even, K_local_odd]) 
        
        self.K_locals = K_locals
    
    def coarest_full_kint(self):
        '''
        kernel integral on coarest level by dense matrix-vector product
        '''
        
        nH = self.ml_grids[-1].nh
        HH = self.ml_grids[-1].hh 
        KHH = self.K_HH.reshape(nH*nH, nH*nH)
        fH = self.ml_f[-1].reshape(-1,nH*nH).T
        uH = HH * (KHH @ fH).T
        uH = uH.reshape(-1,nH,nH)        
        return uH

    def local_interp_K(self, K_2I2J, K_2Ij):
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
        K_2I_j_xeven_yodd_ = K_2I_j_xeven_yodd_.reshape(-1)
        K_2I_j_xodd_yfull_ = K_2I_j_xodd_yfull_.reshape(-1)
        K_i_xodd_yeven_j_ = K_i_xodd_yeven_j_.reshape(-1)
        K_i_xeven_yodd_j_ = K_i_xeven_yodd_j_.reshape(-1)
        K_i_xodd_yodd_j_ = K_i_xodd_yodd_j_.reshape(-1)

        K_local_even = torch.concat([K_2I_j_xeven_yodd_, K_2I_j_xodd_yfull_], axis=0)
        K_local_odd = torch.concat([K_i_xodd_yeven_j_, K_i_xeven_yodd_j_, K_i_xodd_yodd_j_], axis=0)
        
        return K_local_even, K_local_odd
    
    def local_assemble_K(self, K_IJ, K_local_even, K_local_odd):
        nH, _, M, _ = K_IJ.shape
        K_ij = torch.zeros(2*nH-1, 2*nH-1, 2*M-1, 2*M-1).to(K_IJ)

        K_2I_j_xeven_yodd, K_2I_j_xodd_yfull = K_local_even
        K_i_xodd_yeven_j, K_i_xeven_yodd_j, K_i_xodd_yodd_j = K_local_odd

        K_ij[::2,::2,::2,::2] += K_IJ
        K_ij[::2,::2,::2,1::2] += K_2I_j_xeven_yodd
        K_ij[::2,::2,1::2,:] += K_2I_j_xodd_yfull
        K_ij[1::2,::2] += K_i_xodd_yeven_j
        K_ij[::2,1::2] += K_i_xeven_yodd_j
        K_ij[1::2,1::2] += K_i_xodd_yodd_j

        return K_ij
    
    def ml_kint(self):
        u_h = self.coarest_full_kint()
        nH = self.ml_grids[-1].nh 
        K_IJ = self.K_HH[self.ml_grids[-1].ij_idx]
        K_IJ = K_IJ.reshape(nH,nH,2*self.m+1,2*self.m+1)

        for l in range(1, self.k+1):
            nh = self.ml_grids[-1-l].nh
            hh = self.ml_grids[-1-l].hh
            f_h = self.ml_f[-1-l].reshape(-1,nh*nh).T

            # local evaluation and assemblation
            K_local_even, K_local_odd = self.K_locals[l-1]
            idx_corr_even, idx_corr_odd = self.local_idx[l-1]
            K_ij = self.local_assemble_K(K_IJ, K_local_even, K_local_odd)
            K_2Ij = K_ij[::2,::2]

            # local kernel interpolation
            K_local_even_, K_local_odd_ = self.local_interp_K(K_IJ, K_2Ij)

            # calculate difference
            K_local_even = torch.cat([k.reshape(-1) for k in K_local_even], axis=0)
            K_local_odd = torch.cat([k.reshape(-1) for k in K_local_odd], axis=0)
            K_corr_even = K_local_even - K_local_even_
            K_corr_odd = K_local_odd - K_local_odd_

            # correct even
            K_corr_even_sparse = torch.sparse_coo_tensor(idx_corr_even, K_corr_even,(nh**2,nh**2))
            u_corr_ = torch.sparse.mm(K_corr_even_sparse, f_h).T.reshape(-1,nh,nh)
            u_corr_ = hh * injection2d(u_corr_[:,None])[:,0]
            u_h_ = u_h + u_corr_
            u_h_ = interp2d(u_h_[:,None])[:,0]

            # correct odd 
            K_corr_odd_sparse = torch.sparse_coo_tensor(idx_corr_odd, K_corr_odd,(nh**2,nh**2))
            u_corr_ = hh*torch.sparse.mm(K_corr_odd_sparse, f_h).T.reshape(-1,nh,nh)
            u_h_ = u_h_ + u_corr_
            
            # get new K_IJ, u_h
            K_IJ = K_ij[:,:,self.m:-self.m,self.m:-self.m]
            u_h = u_h_

        return rearrange(u_h, 'b m n -> (m n) b', m=nh, n=nh)

    def ml_kint_wo(self):
        u_h = self.coarest_full_kint()
        nh = self.ml_grids[0].nh 
        nH = self.ml_grids[-1].nh 
        K_IJ = self.K_HH[self.ml_grids[-1].ij_idx]
        K_IJ = K_IJ.reshape(nH,nH,2*self.m+1,2*self.m+1)

        for l in range(1, self.k+1):
            u_h = interp2d(u_h[:,None])[:,0]
        return rearrange(u_h, 'b m n -> (m n) b', m=nh, n=nh)

if __name__ == '__main__':
    from utils import rl2_error, matrl2_error
    from utils import dd_kfunc_4D, ffunc_2D
    from utils import dd_kfunc_2D, ffunc_1D
    from tqdm import trange
    import time

    # device = torch.device(f'cuda:0')
    # device = torch.device(f'cuda:0')

    # # MLMM 1D example
    # n = 19
    # m = 3 
    # k = 7

    # # # kernel integral on finest grids
    # # mlmm1d = MLMM1D(n,m,1,device)
    # # f_h = ffunc_1D(mlmm1d.ml_grids[0].x_h).T[None].repeat(7,1,1)
    # # mlmm1d.restrict_ml_f(f_h)
    # # finest_grid = mlmm1d.ml_grids[0]
    # # fh = torch.squeeze(mlmm1d.ml_f[0]).T
    # # nh = finest_grid.nh
    # # hh = finest_grid.hh
    # # finest_grid.init_grid_hh()
    # # finest_pts = finest_grid.x_hh.reshape(-1,2)
    # # Khh = kernel_func_2D(finest_pts).reshape(nh, nh)
    # # uh = (hh * (Khh @ fh).T).cpu()

    # # kernel integral on ml grids with different m
    # for m in [31, 15, 7, 3]:
    #     mlmm1d = MLMM1D(n,m,k,device)
    #     f_h = ffunc_1D(mlmm1d.ml_grids[0].x_h).T[None].repeat(7,1,1)
    #     mlmm1d.restrict_ml_f(f_h)
    #     mlmm1d.eval_ml_K(kernel_func_2D)
    #     uh_ = mlmm1d.ml_kint().cpu()
    #     # print("m {:} - rl2 {:.4e} ".format(m, rl2_error(uh_, uh).numpy()))

    # # # time measure
    # # st = time.time()
    # # for _ in trange(1000):
    # #     uh = hh * (Khh @ fh).T
    # # et = time.time()
    # # print('GPU - full kint avg exec time : {:.5f}s'.format((et-st)/1000))

    # st = time.time()
    # for _ in trange(1000):
    #     uh_ = mlmm1d.ml_kint()
    # et = time.time()
    # print('GPU - ml kint avg exec time : {:.5f}s'.format((et-st)/1000))

    # ----------------------------------------------------------------------
    # 2D example
    # ----------------------------------------------------------------------
    bsz = 4
    n = 7
    m = 3 
    k = 4
    device = torch.device(f'cuda:0')

    finest_grid = DD_Grid2D(2**n+1, m, device)
    finest_grid.init_grid_hh()
    hh = finest_grid.hh
    nh = finest_grid.nh
    Khh = dd_kfunc_4D(finest_grid.x_hh).reshape(nh*nh, nh*nh)
    fh = ffunc_2D(finest_grid.x_h).repeat(1,bsz)

    uh = hh * (Khh @ fh)

    for k in [3, 2, 1]:
        dd2d = DD_MLMM2D(n, 3, k, device)
        dd2d.restrict_ml_f(fh)
        dd2d.eval_ml_K(dd_kfunc_4D)
        uh_ = dd2d.ml_kint()
        print("m {:} - k {:} - rl2 ".format(m, k), matrl2_error(uh_, uh).cpu().numpy())
    
    for m in [1, 3, 5, 7, 9]:
        dd2d = DD_MLMM2D(n, m, 3, device)
        dd2d.restrict_ml_f(fh)
        dd2d.eval_ml_K(dd_kfunc_4D)
        uh_ = dd2d.ml_kint()
        print("m {:} - k {:} - rl2 ".format(m, k), matrl2_error(uh_, uh).cpu().numpy())

    # # time measure
    # st = time.time()
    # for _ in trange(1000):
    #     uh = hh * (Khh @ fh).T
    # et = time.time()
    # print('GPU - full kint avg exec time : {:.5f}s'.format((et-st)/1000))

    # st = time.time()
    # for _ in trange(1000):
    #     uh_ = mlmm2d.ml_kint()
    # et = time.time()
    # print('GPU - ml kint avg exec time : {:.5f}s'.format((et-st)/1000))