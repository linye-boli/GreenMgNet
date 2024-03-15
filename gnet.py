import torch 
from einops import rearrange 
from ops import grid1d_coords, grid2d_coords, grid4d_coords

class Grid1D:
    def __init__(self, nh, device):
        '''
        nh : number of nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        self.nh = nh
        self.h = 2/(self.nh-1)
        self.hh = self.h
        self.device = device
        self.init_grid_h()
    
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

class GN1D:
    def __init__(self, n, kernel, device, sub_num=1024):
        '''
        n : total level
        '''
        self.n = n 
        self.kernel = kernel
        self.device = device
        self.sub_num = sub_num
        self.build_grid()
        self.fetch_eval_pts()
    
    def rand_sub(self):
        self.sub = torch.randint(low=0, high=self.grid.nh, size=(self.sub_num,))

    def build_grid(self):
        nh = 2**self.n + 1
        self.grid = Grid1D(nh, self.device)
    
    def fetch_eval_pts(self):
        self.grid.init_grid_hh()
        self.pts = self.grid.x_hh 
    
    def eval_K(self):
        K_hh = self.kernel(self.pts)
        self.K_hh = K_hh
    
    def full_kint(self, fh):
        hh = self.grid.hh
        nh = self.grid.nh
        Khh = self.K_hh.reshape(nh, nh)
        uh = hh * (Khh @ fh)
        return uh

    def eval_K_sub(self):
        nh = self.grid.nh
        pts = self.pts.reshape(nh, nh, 2)
        K_hh = self.kernel(pts[self.sub])
        self.K_hh = torch.squeeze(K_hh)

    def sub_kint(self, fh):
        hh = self.grid.hh
        Khh = self.K_hh
        uh = hh * (Khh @ fh)
        return uh

    def eval_K_batch(self):
        pts_batch = torch.split(self.pts, 20480)
        K_hh = []
        for pts in pts_batch:
            K_hh.append(self.kernel(pts).detach())
        self.K_hh = torch.cat(K_hh)

    def batch_kint(self, fh):
        hh = self.grid.hh
        nh = self.grid.nh
        Khh = self.K_hh.reshape(nh, nh)
        Khh_batch = torch.split(Khh, 4096)

        uh = []
        for Khh_sub in Khh_batch:
            uh.append(hh * (Khh_sub @ fh))
        uh = torch.cat(uh)
        return uh

class Grid2D:
    def __init__(self, nh, device):
        '''
        nh : number of nodes on each axis
        h : mesh size, domain size [-1, 1]
        hh : square mesh size(for 2d)
        '''
        self.nh = nh
        self.h = 2/(self.nh-1)
        self.hh = self.h**2
        self.device = device
        self.init_grid_h()

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
    
class GN2D:
    def __init__(self, n, kernel, device, sub_num=64):
        '''
        n : total level
        m : neighbor radius for a nodes on each axis
        k : coarse level
        '''
        self.n = n 
        self.device = device
        self.kernel = kernel
        self.sub_num = sub_num
        self.build_grid()
        self.fetch_eval_pts()
    
    def build_grid(self):
        nh = 2**self.n+1
        self.grid = Grid2D(nh, self.device)
    
    def fetch_eval_pts(self):
        self.grid.init_grid_hh()
        self.pts = self.grid.x_hh 
    
    def eval_K(self):
        K_hh = self.kernel(self.pts)
        self.K_hh = K_hh
        
    def full_kint(self, fh):
        hh = self.grid.hh
        nh = self.grid.nh
        Khh = self.K_hh.reshape(nh*nh, nh*nh)
        uh = hh * (Khh @ fh) 
        return uh
    
    def rand_sub(self):
        self.sub = torch.randint(low=0, high=self.grid.nh**2, size=(self.sub_num,))

    def eval_K_sub(self):
        nh = self.grid.nh
        pts = self.pts.reshape(nh*nh, nh, nh, 4)
        K_hh = self.kernel(pts[self.sub])
        self.K_hh = torch.squeeze(K_hh)

    def sub_kint(self, fh):
        hh = self.grid.hh
        Khh = self.K_hh.reshape(self.sub_num,-1)
        uh = hh * (Khh @ fh)
        return uh
    
    def evalint_batch(self, fh):
        hh = self.grid.hh
        nh = self.grid.nh
        fh = fh.reshape(nh*nh,-1)
        pts_batch = torch.split(self.pts.reshape(nh*nh, nh, nh, 4), 64)
        uh = []
        for pts in pts_batch:
            bsz = pts.shape[0]
            K_sub = self.kernel(pts).detach().reshape(bsz, -1)
            u_sub = hh * (K_sub @ fh)
            uh.append(u_sub)     
        
        return torch.cat(uh)
