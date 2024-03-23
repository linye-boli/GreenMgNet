import os 
import numpy as np
import torch 
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
import random 

def dd_kfunc_4D(pts_pairs, eps=1e-5):
    x1 = pts_pairs[:,0]
    y1 = pts_pairs[:,1]

    x2 = pts_pairs[:,2]
    y2 = pts_pairs[:,3]

    mask = ((x1**2+y1**2) < 1) & ((x2**2+y2**2) < 1)

    k = 1/(4*torch.pi) * torch.log(
        ((x1 - x2)**2 + (y1-y2)**2) / \
        ((x1*y2-x2*y1)**2 + (x1*x2+y1*y2-1)**2))
    k = torch.nan_to_num(k, neginf=-2) * mask

    return k

def dd_kfunc_2D(pts_pairs, eps=1e-5):
    x = pts_pairs[:,0]
    y = pts_pairs[:,1]
    k = -torch.log((x-y).abs()+eps)
    return k

def toep_kfunc_2D(x, eps=1e-5):
    k = (torch.linalg.norm(x, axis=1)+eps)**(-0.5)
    return k

def toep_kfunc_1D(x, eps=1e-5):
    k = -torch.log(2*x.abs()+eps)
    return k

def ffunc_2D(pts):
    x = pts[:,[0]]
    y = pts[:,[1]]
    u = 1 - (x**2+y**2)
    u = torch.nan_to_num(u, posinf=0)
    return u

def ffunc_1D(pts):
    y = pts
    return 1-y**2

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

def init_records(log_root, task_nm, exp_nm):
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


    