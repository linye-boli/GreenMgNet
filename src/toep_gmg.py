import torch 
from einops import rearrange
from .ops import grid1d_coords, grid2d_coords
from .ops import interp1d, interp2d

class Toep_Grid1D:
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
        assert nh-m >= 0
        self.init_grid_h()
        self.fetch_nbrs()
    
    def init_grid_h(self):
        '''
        build a 1d mesh grid for input/output 1d functions
        x_h : physical coordinates 2*nh-1 x 1
        coords_h : index coordinates 2*nh-1 x 1
        '''

        nh = 2*self.nh-1
        x_h, coords_h = grid1d_coords(nh)
        self.x_h = x_h.to(self.device)
        self.coords_h = coords_h.to(self.device)
    
    def fetch_nbrs(self):
        nh = 2*self.nh-1
        self.x_j = self.x_h[self.nh-self.m:self.nh-1+self.m][1::2]
        self.j_coords = self.coords_h[self.nh-self.m:self.nh-1+self.m][1::2]
        self.j_idx = self.j_coords
    
class Toep_GMG1D:
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
            ml_grids.append(Toep_Grid1D(nh, self.m, self.device))

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
        self.coarest_pts = self.ml_grids[-1].x_h
        num_coarse = self.coarest_pts.shape[0]

        # local pts
        local_pts = []
        local_idx = []

        num_corrects = 0
        for l in range(self.k):
            x_j = self.ml_grids[l].x_j
            idx_j = self.ml_grids[l].j_idx
            local_pts.append(x_j)
            local_idx.append(idx_j)
            num_corrects += x_j.shape[0]

        if verbose:
            print("# coarest pts : ", num_coarse)
            print('# correction : ', num_corrects)
            print('ratio {:d}/{:d} = {:.2f}% \n'.format(
                num_coarse + num_corrects, 2**(self.n+1)+1, (num_coarse + num_corrects)*100/(2**(self.n+1)+1)))

        self.local_pts = local_pts
        self.local_idx = local_idx

    def eval_ml_K(self):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarest grid
        K_H = self.kernel(self.coarest_pts)
        self.K_H = K_H
        
        # local pts
        K_locals = []
        for l in range(self.k):
            K_local = self.kernel(self.local_pts[l])
            K_locals.append(K_local)
        self.K_locals = K_locals

    def assemble_K(self):
        K_h = torch.squeeze(self.K_H)
        for l in range(self.k):
            K_h = interp1d(K_h[None,None])[0,0]

            if self.m > 0:
                K_local = self.K_locals[-1-l]
                idx_local = self.local_idx[-1-l]
                K_h[idx_local] = K_local
                
        self.K_h = K_h.reshape(-1,1)

    def fft_kint(self, f_h):
        K = self.K_h
        f = f_h

        m = (K.shape[0] - 1)//2
        K_neg = K[:m].flip(-1)
        K_pos = K[m+1:].flip(-1)

        K_ = torch.concat([K[[m]], K_pos, K[[m]], K_neg], axis=0)
        f_ = torch.concat([f, torch.zeros_like(f)], axis=0)
    
        K_ft = torch.fft.rfft(K_.T)
        f_ft = torch.fft.rfft(f_.T)
        u_h = torch.fft.irfft(K_ft*f_ft).T[:m+1]

        return u_h * self.ml_grids[0].hh

class Toep_Grid2D:
    def __init__(self, nh, m, device):
        '''
        nh : number of nodes on each axis
        m : neighbor radius for a nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        assert nh-m >= 0
        self.m = m
        self.nh = nh
        self.h = 2/(self.nh-1)
        self.hh = self.h**2
        self.device = device
        self.init_grid_h()
        self.fetch_nbrs()
    
    def init_grid_h(self):
        '''
        build a 2d mesh grid for input/output 1d functions
        x_h : physical coordinates (2*nh-1) x (2*nh-1) x 1
        coords_h : index coordinates (2*nh-1) x (2*nh-1) x 1
        '''

        nh = 2*self.nh-1
        x_h, coords_h = grid2d_coords(nh)
        self.x_h = x_h.to(self.device)
        self.coords_h = coords_h.to(self.device)

        x_h_, _ = grid2d_coords(self.nh)
        self.mask = ((x_h_[:,0]**2 + x_h_[:,1]**2) < 1).to(self.device)

    
    def fetch_nbrs(self):
        nh = 2*self.nh-1
        idx_local = torch.arange((nh-1)//2-self.m, (nh-1)//2+self.m+1)
        idx_local_even_rows = idx_local[1:-1:2]
        idx_local_even_cols = idx_local[::2]
        idx_local_even = torch.cartesian_prod(idx_local_even_rows, idx_local_even_cols)

        idx_local_odd_rows = idx_local[::2]
        idx_local_odd_cols = idx_local
        idx_local_odd = torch.cartesian_prod(idx_local_odd_rows, idx_local_odd_cols)

        xh = self.x_h.reshape(nh, nh, 2)
        x_local_even = xh[idx_local_even[:,0], idx_local_even[:,1]]
        x_local_odd = xh[idx_local_odd[:,0], idx_local_odd[:,1]]

        self.x_local_even = x_local_even 
        self.x_local_odd = x_local_odd 
        self.idx_local_even = idx_local_even 
        self.idx_local_odd = idx_local_odd 

class Toep_GMG2D:
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
            ml_grids.append(Toep_Grid2D(nh, self.m, self.device))

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
        self.coarest_pts = self.ml_grids[-1].x_h
        num_coarse = self.coarest_pts.shape[0]

        # local pts
        local_even_pts = []
        local_even_idx = []

        local_odd_pts = []
        local_odd_idx = []

        num_corrects = 0
        for l in range(self.k):
            x_local_even = self.ml_grids[l].x_local_even
            x_local_odd = self.ml_grids[l].x_local_odd

            idx_local_even = self.ml_grids[l].idx_local_even
            idx_local_odd = self.ml_grids[l].idx_local_odd
            
            local_even_pts.append(x_local_even)
            local_even_idx.append(idx_local_even)

            local_odd_pts.append(x_local_odd)
            local_odd_idx.append(idx_local_odd)

            num_corrects += x_local_even.shape[0] + x_local_odd.shape[0]


        if verbose:
            print("# coarest pts : ", num_coarse)
            print('# correction : ', num_corrects)
            print('ratio {:d}/{:d} = {:.2f}% \n'.format(
                num_coarse + num_corrects, 
                (2**(self.n+1)+1)**2, 
                (num_coarse + num_corrects)*100/((2**(self.n+1)+1)**2)
                ))


        self.local_even_pts = local_even_pts
        self.local_odd_pts = local_odd_pts

        self.local_even_idx = local_even_idx
        self.local_odd_idx = local_odd_idx
    
    def eval_ml_K(self):
        '''
        evaluate Kernel function on coarest grid and local pts on each grids
        '''
        # coarest grid
        K_H = self.kernel(self.coarest_pts*2)
        self.K_H = K_H
        
        # local pts
        K_locals_even = []
        K_locals_odd = []
        for l in range(self.k):
            K_local_even = self.kernel(self.local_even_pts[l]*2)
            K_local_odd = self.kernel(self.local_odd_pts[l]*2)
            K_locals_even.append(K_local_even)
            K_locals_odd.append(K_local_odd)
            
        self.K_locals_even = K_locals_even
        self.K_locals_odd = K_locals_odd 

    def assemble_K(self):
        nH = 2*self.ml_grids[-1].nh-1
        K_h = self.K_H.reshape(nH,nH)

        for l in range(self.k):
            K_h = interp2d(K_h[None,None])[0,0]

            if self.m > 0:
                K_local_even = self.K_locals_even[-1-l]
                idx_local_even = self.local_even_idx[-1-l]
                K_local_odd = self.K_locals_odd[-1-l]
                idx_local_odd = self.local_odd_idx[-1-l]

                K_h[idx_local_even[:,0], idx_local_even[:,1]] = K_local_even
                K_h[idx_local_odd[:,0], idx_local_odd[:,1]] = K_local_odd
            
        self.K_h = K_h
    
    def fft_kint(self, f_h):
        nh = self.ml_grids[0].nh
        mask_h = self.ml_grids[0].mask
        f_h = f_h.reshape(-1,nh, nh)

        hh = self.ml_grids[0].hh
        K = self.K_h[None]
        f_ = f_h
        l = (K.shape[-1] - 1)//2 + 1
        assert l == f_.shape[-1]
        f = torch.zeros_like(K).repeat(f_.shape[0],1,1)

        f[...,:l,:l] += f_
        K_ft = torch.fft.rfft2(K, s=(2*l,2*l))
        f_ft = torch.fft.rfft2(f, s=(2*l,2*l))
        u = hh * torch.fft.irfft2(K_ft*f_ft)[..., l-1:-1,l-1:-1]
        u_h = u.reshape(-1, nh*nh) * mask_h
        return u_h

if __name__ == '__main__':

    nh = 7 
    m = 3
    device = torch.device('cpu')

    import pdb 
    pdb.set_trace()
