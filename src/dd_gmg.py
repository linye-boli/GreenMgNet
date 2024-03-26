import torch 
import numpy as np
from einops import rearrange
from .ops import grid1d_coords, grid2d_coords, grid4d_coords
from .ops import fetch_nbrs2d, fetch_nbrs1d
from .ops import cat2d_nbr_coords, cat1d_nbr_coords
from .ops import coord2idx2d, coord2idx4d 
from .ops import interp1d, interp2d, interp1d_cols
from .ops import restrict1d, restrict2d
from .ops import injection1d, injection2d

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
        # coords_j : n x (mx1+mx2+1)
        # x_j : n x (mx1+mx2+1)
        coords_j = fetch_nbrs1d(
            self.coords_h, mx1=self.m, mx2=self.m)
        
        self.coords_j = coords_j 
        self.idx_j = coords_j 
        self.x_j = coords_j * self.h - 1

        # coords_ij : n x (mx1+mx2+1) x 2
        # x_ij_redundant : n x (mx1+mx2+1) x 2
        # x_ij : m x 2, not all n points are in domain
        # mask_ij : m x 2, True for in domain, False for outside domain
        coords_ij = cat1d_nbr_coords(self.coords_h, coords_j)
        x_ij = cat1d_nbr_coords(self.x_h, self.x_j)
        self.coords_ij = torch.clamp(coords_ij, min=0, max=self.nh-1)
        self.x_ij = x_ij
        self.mask_ij = (x_ij[...,1] >= -1) & (x_ij[...,1] <= 1)
        
    def fetch_K_local_pts(self):
        x_2I2J = self.x_ij
        x_2Ij = interp1d(x_2I2J.permute(0,2,1)).permute(0,2,1)
        
        # x_2I_j_odd: i=2I, j=j
        x_2I_j_odd = x_2Ij[:,1::2]
        mask_2I_j_odd = (x_2I_j_odd[...,1] >= -1) & (x_2I_j_odd[...,1] <= 1)        
        coords_2I_j_odd =  ((x_2I_j_odd + 1)/self.h * 2).int()[mask_2I_j_odd].T
        pts_2I_j_odd = (x_2I_j_odd, coords_2I_j_odd, mask_2I_j_odd)

        # x_i_odd_j: i=2I+1, j=j
        x_i_odd_j = (x_2Ij[:-1] + x_2Ij[1:])/2
        mask_i_odd_j = (x_i_odd_j[...,1] >= -1) & (x_i_odd_j[...,1] <= 1)
        coords_i_odd_j = ((x_i_odd_j[:,1:-1] + 1)/self.h * 2).int()[mask_i_odd_j[:,1:-1]].T
        pts_i_odd_j = ( x_i_odd_j, coords_i_odd_j, mask_i_odd_j)

        return pts_2I_j_odd, pts_i_odd_j

class DD_GMG1D:
    def __init__(self, n, m, k, kernel, device):
        '''
        n : total level
        m : neighbor radius for a nodes on each axis
        k : coarse level
        '''
        self.n = n 
        self.m = m 
        self.k = k
        self.device = device
        self.kernel = kernel 
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
        nh = self.ml_grids[0].nh
        f_h = rearrange(f_h, 'm b->b 1 m', m=nh)
        ml_f = [f_h]

        for _ in range(self.k):
            f_h = restrict1d(f_h)
            ml_f.append(f_h)
        self.ml_f = ml_f
    
    def fetch_eval_pts(self, verbose=True):
        '''
        fetch coarest grid pts and neighbor pts on each grids
        '''
        # coarest grid
        self.ml_grids[-1].init_grid_hh()
        self.coarest_pts = self.ml_grids[-1].x_hh.reshape(-1,2)
        
        num_coarse = self.coarest_pts.shape[0]

        # local pts
        pts_local = []

        num_corrects = 0
        if self.m > 0:
            for l in range(self.k):
                pts_2I_j_odd, pts_i_odd_j = self.ml_grids[-1-l].fetch_K_local_pts()
                pts_local.append([pts_2I_j_odd, pts_i_odd_j])
                num_corrects += pts_2I_j_odd[-1].sum().cpu().numpy() + pts_i_odd_j[-1].sum().cpu().numpy()
        
        if verbose:
            print("# coarest pts : ", num_coarse)
            print('# correction : ', num_corrects)
            print('ratio {:d}/{:d} = {:.2f}% \n'.format(
                num_coarse + num_corrects, (2**self.n+1)**2, (num_coarse + num_corrects)*100/(2**self.n+1)**2))
        
        self.pts_ratio = (num_coarse + num_corrects)*100/(2**self.n+1)**2

        self.pts_local = pts_local

    def eval_ml_K(self):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarest grid
        K_HH = self.kernel(self.coarest_pts)
        nH = self.ml_grids[-1].nh
        self.K_HH = K_HH.reshape(nH, nH)
        
        # local pts
        if self.m > 0:

            K_locals = []
            for l in range(self.k):
                x_2I_j_odd, _, mask_2I_j_odd = self.pts_local[l][0]
                x_i_odd_j, _, mask_i_odd_j = self.pts_local[l][1]
                K_local_even = torch.zeros_like(mask_2I_j_odd, dtype=torch.float)
                K_local_odd = torch.zeros_like(mask_i_odd_j, dtype=torch.float)
                # import pdb 
                # pdb.set_trace()
                K_local_even[mask_2I_j_odd] += self.kernel(x_2I_j_odd[mask_2I_j_odd]).squeeze()
                K_local_odd[mask_i_odd_j] += self.kernel(x_i_odd_j[mask_i_odd_j]).squeeze()
                K_locals.append([K_local_even, K_local_odd])

            self.K_locals = K_locals

    def coarest_full_kint(self):
        '''
        kernel integral on coarest level by dense matrix-vector product
        '''
        HH = self.ml_grids[-1].hh
        KHH = self.K_HH
        fH = torch.squeeze(self.ml_f[-1]).T
        uH = HH * (KHH @ fH).T
        return uH

    def local_interp_K(self, K_2I2J, K_2Ij):
        '''
        local interpolation of local K
        K_2I2J: nH x (2m+1), coarse nodes and their COARSE neighbors
        K_2Ij: nH x (4m+1), coarse nodes and their FINE neighbors
        '''
        
        # Kernel values for i=2I, j=2J+1
        K_2I_j_odd_ = (K_2I2J[:,:-1] + K_2I2J[:,1:])/2
        
        # Kernel values for i=2I+1, j=j
        K_i_odd_j_ = (K_2Ij[1:,:-2] + K_2Ij[:-1,2:])/2

        return K_2I_j_odd_, K_i_odd_j_
    
    def local_assemble_K(self, K_IJ, K_local_even, K_local_odd):        
        nH, M = K_IJ.shape
        K_ij = torch.zeros(2*nH-1, 2*M-1).to(K_IJ)

        K_ij[::2,::2] += K_IJ
        K_ij[::2,1::2] += K_local_even
        K_ij[1::2] += K_local_odd
        return K_ij

    def ml_kint(self, corr_odd=True):
        u_h = self.coarest_full_kint()
        coords_IJ = self.ml_grids[-1].coords_ij
        mask_IJ = self.ml_grids[-1].mask_ij
        K_IJ = self.K_HH[coords_IJ[...,0], coords_IJ[...,1]] * mask_IJ

        for l in range(1,self.k+1):
            if self.m > 0:
                nh = self.ml_grids[-1-l].nh
                hh = self.ml_grids[-1-l].hh
                f_h = torch.squeeze(self.ml_f[-1-l]).T
            
                # local evaluation and assemblation
                K_local_even, K_local_odd = self.K_locals[l-1]
                K_ij = self.local_assemble_K(K_IJ, K_local_even, K_local_odd)
                K_2Ij = K_ij[::2]
                
                # local kernel interpolation
                K_local_even_, K_local_odd_ = self.local_interp_K(K_IJ, K_2Ij)

                # calculate difference
                _, coords_even, mask_even = self.pts_local[l-1][0]
                _, coords_odd, mask_odd = self.pts_local[l-1][1]
                K_corr_even = (K_local_even - K_local_even_)[mask_even]
                K_corr_odd = (K_local_odd[:,1:-1] - K_local_odd_)[mask_odd[:,1:-1]]

                # correct even 
                K_corr_even_sparse = torch.sparse_coo_tensor(coords_even, K_corr_even,(nh,nh))
                u_corr_ = torch.sparse.mm(K_corr_even_sparse, f_h).T
                u_corr_ = hh * injection1d(u_corr_[None,None])[0,0]
                u_h_ = u_h + u_corr_
                u_h_ = interp1d(u_h_[:,None])[:,0]

                if corr_odd:
                    # correct odd 
                    K_corr_odd_sparse = torch.sparse_coo_tensor(coords_odd, K_corr_odd,(nh,nh))
                    u_corr_ = hh*torch.sparse.mm(K_corr_odd_sparse, f_h).T
                    u_h_ = u_h_ + u_corr_

                # get new K_IJ, u_h
                K_IJ = K_ij[:,self.m:-self.m]
            else:
                u_h_ = interp1d(u_h[:,None])[:,0]

            u_h = u_h_

        return u_h.T

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
        self.mask_hh = self.mask_x(x_hh)
    
    def mask_x(self, x_ij):
        mask_ij = (x_ij[...,2] >= -1) & (x_ij[...,2] <= 1) & \
              (x_ij[...,3] >= -1) & (x_ij[...,3] <= 1)
        return mask_ij

    def fetch_nbrs(self):
        '''
        ij_coords : host-neighbors pair index coordinates of i, (nh^2) x (2m+1)^2 x 4
        ij_idx : (4d)index of ij_coords, (nh^2) x (2m+1)^2 x 1
        x_ij : physical coords of ij_coords, (nh^2) x (2m+1)^2 x 4

        j_coords : neighbors index coordinates of i, (nh^2) x (2m+1)^2 x 2
        j_idx : (2d)index of j_coords, (nh^2) x (2m+1)^2 x 1
        '''

        coords_j = fetch_nbrs2d(
            self.coords_h, mx1=self.m, mx2=self.m,
            my1=self.m, my2=self.m)
        
        self.coords_j = coords_j
        self.idx_j = coord2idx2d(coords_j, self.nh)
        self.x_j = coords_j * self.h - 1

        coords_ij = cat2d_nbr_coords(self.coords_h, coords_j)
        coords_ij = coords_ij.reshape(self.nh, self.nh, 2*self.m+1, 2*self.m+1, -1)
        self.coords_ij = torch.clamp(coords_ij, min=0, max=self.nh-1)

        x_ij = cat2d_nbr_coords(self.x_h, self.x_j)
        self.x_ij = x_ij.reshape(self.nh, self.nh, 2*self.m+1, 2*self.m+1, -1)
        self.mask_ij = self.mask_x(self.x_ij)
    
    def flat_coords(self, coords):
        coords_i = coord2idx2d(coords[...,-4:-2], 2*self.nh-1)
        coords_j = coord2idx2d(coords[...,-2:], 2*self.nh-1)
        return torch.stack([coords_i, coords_j])        

    def fetch_K_local_pts(self):
        nH = self.nh 
        x_2I2J = rearrange(self.x_ij, 'm n x y c -> (m n) x y c')
        x_2Ij = interp2d(x_2I2J.permute(0,3,1,2)).permute(0,2,3,1)
        x_2Ij = rearrange(x_2Ij, '(m n) x y c-> m n x y c', m=nH, n=nH)

        # x_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',2Y'+1)
        x_2I_j_xeven_yodd = x_2Ij[:,:,::2,1::2]
        mask_2I_j_xeven_yodd = self.mask_x(x_2I_j_xeven_yodd)
        coords_2I_j_xeven_yodd =  self.flat_coords(
            ((x_2I_j_xeven_yodd + 1)/self.h * 2).int()[mask_2I_j_xeven_yodd])
        # coords_2I_j_xeven_yodd =  ((x_2I_j_xeven_yodd + 1)/self.h * 2).int()
        pts_2I_j_xeven_yodd = (x_2I_j_xeven_yodd, coords_2I_j_xeven_yodd, mask_2I_j_xeven_yodd)

        # x_2I_j_xeven_yodd: i=(2X,2Y), j=(2X',y')
        x_2I_j_xodd_yfull = x_2Ij[:,:,1::2]
        mask_2I_j_xodd_yfull = self.mask_x(x_2I_j_xodd_yfull)
        coords_2I_j_xodd_yfull =  self.flat_coords(
            ((x_2I_j_xodd_yfull + 1)/self.h * 2).int()[mask_2I_j_xodd_yfull])
        # coords_2I_j_xodd_yfull = ((x_2I_j_xodd_yfull + 1)/self.h * 2).int()
        pts_2I_j_xodd_yfull = (x_2I_j_xodd_yfull, coords_2I_j_xodd_yfull, mask_2I_j_xodd_yfull)

        # x_i_xodd_yeven_j: i=(2X+1,2Y), j=(x',y')
        x_i_xodd_yeven_j = (x_2Ij[:-1] + x_2Ij[1:])/2
        mask_i_xodd_yeven_j = self.mask_x(x_i_xodd_yeven_j)
        coords_i_xodd_yeven_j  =  self.flat_coords(
            ((x_i_xodd_yeven_j[:,:,1:-1]  + 1)/self.h * 2).int()[mask_i_xodd_yeven_j[:,:,1:-1]])
        pts_i_xodd_yeven_j  = (x_i_xodd_yeven_j , coords_i_xodd_yeven_j , mask_i_xodd_yeven_j)

        # x_i_xeven_yodd_j: i=(2X,2Y+1), j=(x',y')
        x_i_xeven_yodd_j = (x_2Ij[:,:-1] + x_2Ij[:,1:])/2
        mask_i_xeven_yodd_j = self.mask_x(x_i_xeven_yodd_j)
        coords_i_xeven_yodd_j  = self.flat_coords(
            ((x_i_xeven_yodd_j[:,:,:,1:-1]  + 1)/self.h * 2).int()[mask_i_xeven_yodd_j[:,:,:,1:-1]])
        pts_i_xeven_yodd_j  = (x_i_xeven_yodd_j , coords_i_xeven_yodd_j , mask_i_xeven_yodd_j)

        # x_i_xodd_yodd_j: i=(2X+1,2Y+1), j=(x',y')
        x_i_xodd_yodd_j = (x_i_xeven_yodd_j[:-1] + x_i_xeven_yodd_j[1:])/2
        mask_i_xodd_yodd_j = self.mask_x(x_i_xodd_yodd_j)
        coords_i_xodd_yodd_j  =  self.flat_coords(
            ((x_i_xodd_yodd_j[:,:,1:-1,1:-1]  + 1)/self.h * 2).int()[mask_i_xodd_yodd_j[:,:,1:-1,1:-1]])
        pts_i_xodd_yodd_j  = (x_i_xodd_yodd_j , coords_i_xodd_yodd_j , mask_i_xodd_yodd_j)

        return pts_2I_j_xeven_yodd, pts_2I_j_xodd_yfull, pts_i_xodd_yeven_j, pts_i_xeven_yodd_j, pts_i_xodd_yodd_j

    
class DD_GMG2D:
    def __init__(self, n, m, k, kernel, device):
        '''
        n : total level
        m : neighbor radius for a nodes on each axis
        k : coarse level
        '''
        self.n = n 
        self.m = m 
        self.k = k
        self.device = device
        self.kernel = kernel 
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
    
    def fetch_eval_pts(self, verbose=True):
        '''
        fetch coarest grid pts and neighbor pts on each grids
        '''
        # coarest grid
        self.ml_grids[-1].init_grid_hh()
        self.coarest_pts = self.ml_grids[-1].x_hh.reshape(-1,4)

        num_coarse = self.coarest_pts.shape[0]

        # local pts
        pts_local = []

        num_corrects = 0
        if self.m > 0:
            for l in range(self.k):
                pts = self.ml_grids[-1-l].fetch_K_local_pts()
                pts_local.append(pts)
                num_corrects += sum([p[1].shape[1] for p in pts])

        if verbose:
            print("# coarest pts : ", num_coarse)
            print('# correction : ', num_corrects)
            print('ratio {:d}/{:d} = {:.2f}% \n'.format(
                num_coarse + num_corrects, (2**self.n+1)**4, (num_coarse + num_corrects)*100/(2**self.n+1)**4))
        
        self.pts_ratio = (num_coarse + num_corrects)*100/(2**self.n+1)**2

        self.pts_local = pts_local
    
    def restrict_ml_f(self, f_h):
        '''
        restrict f into multi-level
        '''
        nh = self.ml_grids[0].nh
        f_h = rearrange(f_h, 'b (m n)->b 1 m n', m=nh, n=nh)
        ml_f = [f_h]
        for _ in range(self.k):
            f_h = restrict2d(f_h)
            ml_f.append(f_h)
        self.ml_f = ml_f
    
    def eval_ml_K(self):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarse grid 
        K_HH = self.kernel(self.coarest_pts)
        nH = self.ml_grids[-1].nh
        self.K_HH = K_HH.reshape(nH, nH, nH, nH)

        # local pts
        if self.m > 0:
            K_locals = []
            for l in range(self.k):
                local_pts = self.pts_local[l]

                x_2I_j_xeven_yodd, _, mask_2I_j_xeven_yodd = local_pts[0]
                x_2I_j_xodd_yfull, _, mask_2I_j_xodd_yfull = local_pts[1]
                x_i_xodd_yeven_j, _, mask_i_xodd_yeven_j = local_pts[2]
                x_i_xeven_yodd_j, _, mask_i_xeven_yodd_j = local_pts[3]
                x_i_xodd_yodd_j, _, mask_i_xodd_yodd_j = local_pts[4]

                K_2I_j_xeven_yodd = torch.zeros_like(mask_2I_j_xeven_yodd, dtype=torch.float)
                K_2I_j_xodd_yfull = torch.zeros_like(mask_2I_j_xodd_yfull, dtype=torch.float)
                K_i_xodd_yeven_j = torch.zeros_like(mask_i_xodd_yeven_j, dtype=torch.float)
                K_i_xeven_yodd_j = torch.zeros_like(mask_i_xeven_yodd_j, dtype=torch.float)
                K_i_xodd_yodd_j = torch.zeros_like(mask_i_xodd_yodd_j, dtype=torch.float)
                
                K_2I_j_xeven_yodd[mask_2I_j_xeven_yodd] += self.kernel(
                    x_2I_j_xeven_yodd[mask_2I_j_xeven_yodd]).squeeze()
                
                K_2I_j_xodd_yfull[mask_2I_j_xodd_yfull] += self.kernel(
                    x_2I_j_xodd_yfull[mask_2I_j_xodd_yfull]).squeeze()

                K_i_xodd_yeven_j[mask_i_xodd_yeven_j] += self.kernel(
                    x_i_xodd_yeven_j[mask_i_xodd_yeven_j]).squeeze()

                K_i_xeven_yodd_j[mask_i_xeven_yodd_j] += self.kernel(
                    x_i_xeven_yodd_j[mask_i_xeven_yodd_j]).squeeze()

                K_i_xodd_yodd_j[mask_i_xodd_yodd_j] += self.kernel(
                    x_i_xodd_yodd_j[mask_i_xodd_yodd_j]).squeeze()
                
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
        
        # Kernel values for i=(2X,2Y), j=(2X',y')
        K_2I_j_xeven_yfull_ = interp1d_cols(K_2I2J)
        # Kernel values for i=(2X,2Y), j=(2X',2Y'+1)
        K_2I_j_xeven_yodd_ = K_2I_j_xeven_yfull_[:,:,:,1::2] 
        # Kernel values for i=(2X,2Y), j=(2X'+1,y')
        K_2I_j_xodd_yfull_ = (K_2I_j_xeven_yfull_[:,:,:-1] + K_2I_j_xeven_yfull_[:,:,1:])/2 
        
        # Kernel values for i=(2X,2Y), j=(x',y')
        K_i_xeven_yeven_j = K_2Ij
        # Kernel values for i=(2X,2Y), j=(x',y')
        K_i_xodd_yeven_j_ = (K_i_xeven_yeven_j[1:,:,:-2] + K_i_xeven_yeven_j[:-1,:,2:])/2
        # Kernel values for i=(2X,2Y), j=(x',y')
        K_i_xeven_yodd_j_ = (K_i_xeven_yeven_j[:,:-1,:,2:] + K_i_xeven_yeven_j[:,1:,:,:-2])/2
        # Kernel values for i=(2X,2Y), j=(x',y')
        K_i_xodd_yodd_j_ = (K_i_xeven_yodd_j_[:-1,:,2:] + K_i_xeven_yodd_j_[1:,:,:-2])/2

        K_local_even = [K_2I_j_xeven_yodd_, K_2I_j_xodd_yfull_]
        K_local_odd = [K_i_xodd_yeven_j_, K_i_xeven_yodd_j_, K_i_xodd_yodd_j_]
        
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
    
    def ml_kint(self, corr_odd=True):
        u_h = self.coarest_full_kint()
        coords_IJ = self.ml_grids[-1].coords_ij
        mask_IJ = self.ml_grids[-1].mask_ij
        K_IJ = self.K_HH[coords_IJ[...,0], coords_IJ[...,1], 
                         coords_IJ[...,2], coords_IJ[...,3]] * mask_IJ

        for l in range(1, self.k+1):
            if self.m > 0:
                nh = self.ml_grids[-1-l].nh
                hh = self.ml_grids[-1-l].hh
                f_h = self.ml_f[-1-l].reshape(-1,nh*nh).T

                # local evaluation and assemblation
                K_local_even, K_local_odd = self.K_locals[l-1]
                K_ij = self.local_assemble_K(K_IJ, K_local_even, K_local_odd)
                K_2Ij = K_ij[::2,::2]

                # local kernel interpolation
                K_local_even_, K_local_odd_ = self.local_interp_K(K_IJ, K_2Ij)

                # calculate difference
                _, coords_2I_j_xeven_yodd, mask_2I_j_xeven_yodd = self.pts_local[l-1][0]
                _, coords_2I_j_xodd_yfull, mask_2I_j_xodd_yfull = self.pts_local[l-1][1]

                K_corr_2I_j_xeven_yodd = (K_local_even[0] - K_local_even_[0])[mask_2I_j_xeven_yodd]
                K_corr_2I_j_xodd_yfull = (K_local_even[1] - K_local_even_[1])[mask_2I_j_xodd_yfull]
                K_corr_even = torch.cat([K_corr_2I_j_xeven_yodd, K_corr_2I_j_xodd_yfull])
                coords_even = torch.cat([coords_2I_j_xeven_yodd, coords_2I_j_xodd_yfull], axis=1)
                
                _, coords_i_xodd_yeven_j, mask_i_xodd_yeven_j = self.pts_local[l-1][2]
                _, coords_i_xeven_yodd_j, mask_i_xeven_yodd_j = self.pts_local[l-1][3]
                _, coords_i_xodd_yodd_j, mask_i_xodd_yodd_j = self.pts_local[l-1][4]

                K_corr_i_xodd_yeven_j = (K_local_odd[0][:,:,1:-1]  - K_local_odd_[0])[mask_i_xodd_yeven_j[:,:,1:-1]]
                K_corr_i_xeven_yodd_j = (K_local_odd[1][:,:,:,1:-1] - K_local_odd_[1])[mask_i_xeven_yodd_j[:,:,:,1:-1]]
                K_corr_i_xodd_yodd_j = (K_local_odd[2][:,:,1:-1,1:-1] - K_local_odd_[2])[mask_i_xodd_yodd_j[:,:,1:-1,1:-1]]
                K_corr_odd = torch.cat([K_corr_i_xodd_yeven_j, K_corr_i_xeven_yodd_j, K_corr_i_xodd_yodd_j])
                coords_odd = torch.cat([coords_i_xodd_yeven_j, coords_i_xeven_yodd_j, coords_i_xodd_yodd_j], axis=1)

                # correct even
                K_corr_even_sparse = torch.sparse_coo_tensor(coords_even, K_corr_even,(nh**2,nh**2))
                u_corr_ = torch.sparse.mm(K_corr_even_sparse, f_h).T.reshape(-1,nh,nh)
                u_corr_ = hh*injection2d(u_corr_[:,None])[:,0]
                u_h_ = u_h + u_corr_
                u_h_ = interp2d(u_h_[:,None])[:,0]

                # correct odd 
                if corr_odd:
                    K_corr_odd_sparse = torch.sparse_coo_tensor(coords_odd, K_corr_odd,(nh**2,nh**2))
                    u_corr_ = hh*torch.sparse.mm(K_corr_odd_sparse, f_h).T.reshape(-1,nh,nh)
                    u_h_ = u_h_ + u_corr_
            
                # get new K_IJ, u_h
                K_IJ = K_ij[:,:,self.m:-self.m,self.m:-self.m]
            else:
                u_h_ = interp2d(u_h[:,None])[:,0]
            
            u_h = u_h_

        return rearrange(u_h, 'b m n -> b (m n)')

    def twolevel_kint(self, corr_odd=True):
        assert len(self.ml_grids) == 2
        fine_grid = self.ml_grids[0]
        coarse_grid = self.ml_grids[1]
        u_h = self.coarest_full_kint()

        fine_grid.init_grid_hh()
        nh = fine_grid.nh
        hh = fine_grid.hh
        mask_hh = fine_grid.mask_hh.reshape(nh,nh,nh,nh)
        K_hh = self.kernel(fine_grid.x_hh).reshape(nh,nh,nh,nh)

        nH = coarse_grid.nh
        K_HH = self.kernel(coarse_grid.x_hh).reshape(nH,nH,nH,nH)

        f_h = self.ml_f[0].reshape(-1,nh*nh).T

        K_Hh_ = interp2d(K_HH)
        K_corr_even = ((K_hh[::2,::2] - K_Hh_)*mask_hh[::2,::2]).reshape(nH**2, nh**2)
        u_corr_ = hh*torch.mm(K_corr_even, f_h).T.reshape(-1,nH,nH)
        u_h_ = u_h + u_corr_
        u_h_ = interp2d(u_h_[:,None])[:,0]

        K_hh_ = rearrange(interp2d(rearrange(K_hh[::2,::2], 'm n x y -> x y m n')), 'x y m n -> m n x y')
        K_corr_odd = ((K_hh - K_hh_)*mask_hh).reshape(nh**2, nh**2)

        if corr_odd:
            u_corr_ = hh*torch.mm(K_corr_odd, f_h).T.reshape(-1,nh,nh)
            u_h_ = u_h_ + u_corr_

        u_h = u_h_
        return rearrange(u_h, 'b m n -> b (m n)')

    

if __name__ == '__main__':
    from einops import repeat
    from .utils import rl2_error

    def Logarithm(pts):
        x = pts[...,0]
        y = pts[...,1]
        return torch.nan_to_num(torch.log((x-y).abs()), neginf=-8)
    
    n = 13
    device = torch.device('cuda:0')

    k = 5
    m = 3

    F = torch.squeeze(1 - torch.linspace(-1,1,2**n+1)**2)
    F = repeat(F, 'n -> n b', b=64)
    dd_gmg = DD_GMG1D(n,m,k,Logarithm,device)
    dd_gmg.restrict_ml_f(F)
    dd_gmg.eval_ml_K()
    U_dd = dd_gmg.ml_kint()

    print('k {:d} - m {:d} - rl2 {:.4e}'.format(k, m, rl2_error(U_dd.T, U_full.T)))