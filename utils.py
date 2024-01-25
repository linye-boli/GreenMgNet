import os 
import numpy as np
import torch 
import torch.nn.functional as F
from einops import rearrange, repeat
import pandas as pd
import random 
import kornia

# gauss smoothing
def gauss_smooth1d(x, ksize=7, sigma=3.):
    kernel = kornia.filters.get_gaussian_kernel1d(ksize, sigma).to(x)
    x = F.conv1d(x, kernel[None], stride=1, padding=(ksize-1)//2)
    return x

def gauss_smooth2d(x, ksize=5, sigma=1.5):
    kernel = kornia.filters.get_gaussian_kernel2d((ksize,ksize),(sigma,sigma)).to(x)
    x = F.conv2d(x, kernel[None], stride=1, padding=(ksize-1)//2)
    return x

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
    l = (a.shape[-1] - 1)//2
    assert l == f.shape[-1] - 1
    f_ = torch.zeros_like(a)
    f_[:l,:l] += f
    A = torch.fft.rfft2(a, s=(2*l,2*l))
    F = torch.fft.rfft2(f_, s=(2*l,2*l))
    u = torch.fft.irfft2(A*F)[l-1:-1,l-1:-1]
    return u 

# injections 
def injection2d(Khh):
    KhH = torch.cat([Khh[...,[0]], Khh[...,1:-1][...,1::2], Khh[...,[-1]]], axis=-1)
    KHH = torch.cat([KhH[...,[0],:], KhH[...,1:-1,:][...,1::2,:], KhH[...,[-1],:]], axis=-2)
    return KHH

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

def interp1d_rows(KHh, order=2):
    # KhH : (batch, c, H, h)

    KhH = rearrange(KHh, 'b c I j -> b c j I')
    Khh = interp1d_cols(KhH, order)
    Khh = rearrange(Khh, 'b c j i -> b c i j')

    return Khh

def interp2d(KHH, order=2):
    KHh = interp1d_rows(KHH, order)
    Khh = interp1d_cols(KHh, order)
    return Khh

def multi_summation(K, u, h):
    # KHH : (batch, c, m, n)
    # u : (batch, c, n)
    # h : float scalar
    return h * torch.einsum('bcmn, bcn-> bcm', K, u)

def l1_norm(est, ref):
    if len(est.shape) == 2:
        b, n = est.shape 
    elif len(est.shape) == 3:
        b, c, n = est.shape 
    return ((est - ref).abs().sum(axis=-1)/(n+2)).mean() # here n+2 indicates i=0,1,2,...2^13
    
def rl2_error(est, ref):
    if len(est.shape) == 2:
        b, n = est.shape 
    elif len(est.shape) == 3:
        b, c, n = est.shape 
    return ((((est - ref)**2).sum(axis=-1))**0.5 / ((ref**2).sum(axis=-1))**0.5).mean()

def matrl2_error(est, ref):
    est = est.reshape(1,-1)
    ref = ref.reshape(1,-1)
    return rl2_error(est, ref)

def ml_rl2_error(est, ref, k, order=2):
    est = est[::-1]
    for i in range(k+1):
        if i == 0:
            rl2 = rl2_error(est[i], ref)
        else:
            ref = restrict1d(ref, order=order)
            rl2 += rl2_error(est[i], ref)
    return rl2

def gradient_loss(A, h, thr=3, w='log'):

    if len(A.shape) == 3:
        n = A.shape[-1]
        w = -torch.log(torch.linspace(-1,1,n).abs()+1e-10).to(A)
        A = A[0,0]
        Agrad = torch.gradient(A, spacing=h)[0]
        loss = (F.relu(Agrad.abs() - thr) * w).mean()
        return loss
    
    if len(A.shape) == 4:
        n = A.shape[-1]
        x = torch.linspace(-1,1,n)[None]
        d = (x - x.T + 1e-10).abs()
        if w == 'log':
            w = -torch.log(d).to(A)
        elif w == 'one':
            w = torch.ones_like(d).to(A)
        A = A[0,0]
        Agradx, Agrady = torch.gradient(A, spacing=h)
        Agrad = torch.max(Agradx.abs(),Agrady.abs())
        loss = (F.relu(Agrad - thr) * w).mean()

        return loss

def init_records(log_root, args):
    resolution = str(args.res)
    task_nm = '_'.join([args.task, resolution])

    if args.model == 'toep_mg':
        exp_nm = '-'.join([args.model, args.act, str(args.h), str(args.k), str(args.m), str(args.seed)])
    
    if args.model == 'dd_mg':
        exp_nm = '-'.join([args.model, args.act, str(args.h), str(args.k), str(args.m), str(args.seed)])    

    if args.model == 'lrdd_mg':
        exp_nm = '-'.join([args.model, args.act, str(args.h), str(args.k), str(args.m), str(args.r), str(args.seed)])
        
    if args.model in ['toep_gl', 'gl']:
        exp_nm = '-'.join([args.model, args.act, str(args.h), str(args.seed)])    
    
    if args.model == 'lr_gl':
        exp_nm = '-'.join([args.model, args.act, str(args.h), str(args.r), str(args.seed)])    

    if args.model == 'fno':
        exp_nm = '-'.join([args.model, str(args.seed)])

    exp_root = os.path.join(log_root, task_nm, exp_nm)
    os.makedirs(exp_root, exist_ok=True)

    hist_outpath = os.path.join(exp_root, 'hist.csv')
    pred_outpath = os.path.join(exp_root, 'pred.csv')
    model_operator_outpath = os.path.join(exp_root, 'model.pth')
    model_kernel_outpath = os.path.join(exp_root, 'approx_kernel.npy')
    cfg_outpath = os.path.join(exp_root, 'cfg.json')    
    
    return hist_outpath, pred_outpath, model_operator_outpath, model_kernel_outpath, cfg_outpath

def save_hist(hist_outpath, train_hist, test_hist):
    log_df = pd.DataFrame({'train_rl2': train_hist, 'test_rl2': test_hist})
    log_df.to_csv(hist_outpath, index=False)
    print('save train-test log at : ', hist_outpath)

def save_preds(pred_outpath, preds):
    preds = np.array(preds)
    preds = rearrange(preds, 'n b l -> (n b) l')
    np.savetxt(pred_outpath, preds, delimiter=',')
    print('save test predictions at : ', pred_outpath)

def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        # pass 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("="*50)
        print(message)
        print("="*50)

def train_mg_model(train_loader, model, opt, sch, device, cfg, ep):
    model.train()
    train_rl2 = 0

    for u, w in train_loader:
        # prepare data
        u, w = u.to(device), w.to(device)
        n = u.shape[-1]
        x = torch.linspace(-1,1,n)[None][None].to(device)

        # infer
        w_, _ = model(u, x)

        # calc loss
        loss = rl2_error(w_, w)

        # update params
        opt.zero_grad()
        loss.backward() # use the l2 relative loss

        if ep in cfg[0]:
            pass

        opt.step()
        train_rl2 += loss.item()

    sch.step()
    train_rl2 = train_rl2/len(train_loader)
    return model, train_rl2

def train_model(train_loader, model, opt, device):
    model.train()
    train_rl2 = 0

    for u, w in train_loader:
        # prepare data
        u, w = u.to(device), w.to(device)
        n = u.shape[-1]
        x = torch.linspace(-1,1,n)[None][None].to(device)
        h = 1/n

        # infer
        w_ = model(u, x)

        # calc loss
        loss = rl2_error(w_, w)
        
        # update params
        opt.zero_grad()
        loss.backward() # use the l2 relative loss
        opt.step()

        train_rl2 += loss.item()

    train_rl2 = train_rl2/len(train_loader)
    return model, train_rl2

def eval_model(test_loader, model, device):
    model.eval()
    test_rl2 = 0.0
    with torch.no_grad():
        for u, w in test_loader:
            u, w = u.to(device), w.to(device)
            n = u.shape[-1]
            x = torch.linspace(-1,1,n)[None][None].to(device)

            w_ = model(u, x)
            rl2 = rl2_error(w_, w)
            test_rl2 += rl2.item()

    test_rl2 = test_rl2/len(test_loader)

    return test_rl2

def fetch_nbrs(n, m):
    # n : int, lenth of inputs,
    # m : int, radius of window
 
    idx_h = torch.arange(n)
    idx_nbrs = torch.arange(-m, m+1)
    idx_j = torch.cartesian_prod(idx_h, idx_nbrs).sum(axis=1).reshape(-1, 2*m+1) # n x 2m+1
    idx_i = repeat(idx_h, 'i -> i m', m=2*m+1)

    return idx_i, idx_j

if __name__ == '__main__':

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
    