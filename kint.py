import torch 
from ops import restrict1d
from ops import interp1d, interp1d_cols, interp1d_rows
from ops import injection1d_cols, injection1d_rows
from ops import fetch_nbrs

# matrix multiplication
def circulant_matrix_vector_multiplication(a, b):
    a = torch.flip(a, [-1])
    a_ft = torch.fft.rfft(a)
    b_ft = torch.fft.rfft(b)
    return torch.fft.irfft(a_ft*b_ft)

def toeplitz_matrix_vector_multiplication(a, b, method='fft'):
    m = (a.shape[-1] - 1)//2
    
    if method == 'fft':
        assert m == b.shape[-1] - 1
        a_neg = a[...,:m].flip(-1)
        a_pos = a[...,m+1:].flip(-1)
        a_ = torch.concat([a[...,[m]], a_neg, a[...,[m]], a_pos], axis=-1)
        b_ = torch.concat([b, torch.zeros_like(b)], axis=-1)
        a_ft = torch.fft.rfft(a_)
        b_ft = torch.fft.rfft(b_)
        return torch.fft.irfft(a_ft*b_ft)[...,:m+1]
    elif method == 'conv':        
        return torch.nn.functional.conv1d(b, a, padding=m)

def lowrank_matrix_vector_multiplication(phi, psi, v):
    return torch.einsum('brn, bcn, brm->bcm', phi, v, psi)

def full_matrix_vector_multiplication(a, v):
    return torch.einsum('bcmn, bcn->bcm', a, v)

def ml_matrix_vector_multiplication(Ks, uh, nbrs, idx_mask, h, k):
    if len(Ks) == 1:
        for i in range(k):
            uh = restrict1d(uh)
            # update       
            h = h*2
        
        wh = multi_summation(Ks[0], uh, h)

        for i in range(k):
            h = h/2
            wh_even = wh 
            wh = interp1d(wh_even)
    else:
        uh_band_lst = []
        nbrs = nbrs[::-1]
        idx_mask = idx_mask[::-1]

        for i in range(k):
            uh_band = uh[:,:,nbrs[i]] * idx_mask[i].to(uh)
            uh_band_even = uh_band[:,:,::2]
            uh_band_odd = uh_band[:,:,1::2]
            uh_band_lst.append([uh_band_even[:,:,:,::2], uh_band_odd])
            uh = restrict1d(uh)
            # update       
            h = h*2
        
        # coarsest approximation
        wh = multi_summation(Ks[0], uh, h)
        Kband_corr_lst = Ks[1:]

        # reverse list
        uh_band_lst = uh_band_lst[::-1]
        
        # multi-level correction
        for i in range(k):
            h = h/2
            wh_even_corr = (Kband_corr_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
            wh_even = wh 
            wh_even[:,:,1:-1] += wh_even_corr[:,:,1:-1]
            wh = interp1d(wh_even)
            wh[:,:,1::2] = wh[:,:,1::2] + (Kband_corr_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h

    return wh

def fullml_matrix_vector_multiplication(uh, Khh, h, k=3, m=7):
    Khh_banddiff_lst = []
    uh_band_lst = []
    lb_lst = []
    rb_lst = []
    idx_j_lst = []

    for i in range(k):
        # calculate boundary values
        w_lb = multi_summation(Khh[:,:,[0]], uh, h)
        w_rb = multi_summation(Khh[:,:,[-1]], uh, h)
        lb_lst.append(w_lb)
        rb_lst.append(w_rb)


        # evaluate kernel function on coarse grid
        KHh = injection1d_rows(Khh) 
        KHH = injection1d_cols(KHh)

        # smooth approximation of kernel function
        KHh_smooth = interp1d_cols(injection1d_cols(Khh))
        Khh_smooth = interp1d_rows(KHh)

        # fetch nbr idx
        n = Khh.shape[-1] 
        idx_i, idx_j = fetch_nbrs(n, m)
        idx_mask = (idx_j >= 0) & (idx_j <= n-1)
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > n-1] = n-1
        idx_j_lst.append(idx_j)

        # band diff between fine kernel and smooth approximation
        KHh_banddiff_even = (Khh - KHh_smooth)[:,:,idx_i, idx_j][:,:,::2]
        Khh_banddiff_odd =  (Khh - Khh_smooth)[:,:,idx_i, idx_j][:,:,1::2]

        # correction kernels
        Khh_banddiff_lst.append(
            [KHh_banddiff_even[:,:,:,::2], Khh_banddiff_odd])

        # uh band
        uh_band = uh[:,:,idx_j] * idx_mask
        uh_band_even = uh_band[:,:,::2]
        uh_band_odd = uh_band[:,:,1::2]
        uh_band_lst.append([uh_band_even[:,:,:,::2], uh_band_odd])

        # coarse uh
        uh = restrict1d(uh)

        # update       
        h = h*2
        Khh = KHH
    
    # reverse list
    Khh_banddiff_lst = Khh_banddiff_lst[::-1]
    uh_band_lst = uh_band_lst[::-1]
    boundary_lst = [lb_lst[::-1], rb_lst[::-1]]

    # coarsest approximation
    wh = multi_summation(Khh, uh, h)    

    # multi-level correction
    for i in range(k):
        h = h/2        
        wh_even_corr = (Khh_banddiff_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
        wh += wh_even_corr
        wh[:,:,[0]] = boundary_lst[0][i]
        wh[:,:,[-1]] = boundary_lst[1][i]
        wh = interp1d(wh)
        wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h
    
    return wh

def fourier_integral_transform(a, f):
    l = (a.shape[-1] - 1)//2 + 1
    assert l == f.shape[-1]
    f_ = torch.zeros_like(a).repeat(f.shape[0],1,1,1)

    f_[...,:l,:l] += f
    A = torch.fft.rfft2(a, s=(2*l,2*l))
    F = torch.fft.rfft2(f_, s=(2*l,2*l))
    u = torch.fft.irfft2(A*F)[..., l-1:-1,l-1:-1]
    return u 

def multi_summation(K, u, h):
    # KHH : (batch, c, m, n)
    # u : (batch, c, n)
    # h : float scalar
    return h * torch.einsum('bcmn, bcn-> bcm', K, u)

