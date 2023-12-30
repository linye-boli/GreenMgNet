import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
from dataset import load_dataset_1d 
from datetime import datetime 
from utils import rl2_error, init_records, save_hist, get_seed, injection1d, interp1d, toeplitz_matrix_vector_multiplicaiton
import json 
from tqdm import trange 
from einops import rearrange 
from model import GMGN, GL, MLP


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train Green-MgNet")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--fine', action='store_true',
                        help='fine grid(8193) or coarse grid(513) problem')
    parser.add_argument('--nn', type=str, default='mlp',
                        help='type of neural network to model kernel, mlp/low-rank')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation functions')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
        
    # parameters for GreenMgNet
    parser.add_argument('--lk', type=int, default=3,
                        help='level of kernel functions')
    parser.add_argument('--cs', type=int, default=2,
                        help='coarsen scale, 0 for no coarsen, 2 for half, 4 for quarter')
    parser.add_argument('--struct', type=str, default='GMGN',
                        help='local correction range')
    parser.add_argument('--fastMM', action='store_true',
                        help='whether to use fast algorithm to compute matrix-vector multiplicaiton')
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    get_seed(args.seed)

    batch_size = 1000
    lr_adam = 1e-3
    lr_lbfgs = .01
    epochs = 520

    if args.task in ['cosine', 'logarithm']:        
        in_channels = 1
        out_channels = 1
        is_toep = True
    elif args.task in ['laplace']:
        in_channels = 2
        out_channels = 1
        is_toep = False

    if args.struct == 'GL':
        args.lk = 0
        args.cs = 0

    ################################################################
    # prepare log
    ################################################################
    device = torch.device(f'cuda:{args.device}')
    if args.fine:
        resolution = 8193
    else:
        resolution = 513 

    if is_toep:
        n = 2*resolution-1

    data_root = f'/workdir/pde_data/green_learning/data1d_{resolution}'
    log_root = '/workdir/GreenMgNet/results/'
    hist_outpath, pred_outpath, nn_outpath, kernel_outpath, cfg_outpath = init_records(log_root, args)

    print('output files:')
    print(hist_outpath)
    print(pred_outpath)
    print(nn_outpath)
    print(kernel_outpath)
    print(cfg_outpath)

    with open(cfg_outpath, 'w') as f:
        cfg_dict = vars(args)
        json.dump(cfg_dict, f)
    
    ################################################################
    # read data
    ################################################################
    train_loader, test_loader = load_dataset_1d(args.task, data_root, normalize=False)
    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 1
    test_rl2_best = 1
    x = torch.linspace(-1,1,n)[None][None].to(device)

    ################################################################
    # build model
    ################################################################
    if args.struct == 'GMGN':

        def smooth_kernel_approx(kfunc, x, l):
            for _ in range(l):
                x = injection1d(x)
            
            a = kfunc(x)

            for _ in range(1, l+1):
                a = interp1d(a)
            
            return a
        
        def correction_kernel_approx(kcfunc, x, l):
            for _ in range(l):
                x = injection1d(x)
            ac = kcfunc(x)
            bump = torch.exp(-5/(1-x**2))
            bump = (bump - bump.min())/(bump.max()-bump.min())
            return ac * bump

        smooth_kernel = MLP([in_channels, 64, 64, out_channels], args.act).to(device)
        correction_kernel = MLP([in_channels, 32, 32, out_channels], args.act).to(device)

        ################################################################
        # training and evaluation
        ################################################################
        opt_adam = torch.optim.Adam(smooth_kernel.parameters(), lr=lr_adam)
        opt_lbfgs = torch.optim.LBFGS(smooth_kernel.parameters(), lr=lr_lbfgs)

        pbar = trange(epochs)
        for ep in pbar:
            pbar.set_description(
                "train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
            smooth_kernel.train()
            train_rl2 = 0

            for u, w in train_loader:
                u, w = u.to(device), w.to(device)
                bsz = u.shape[0]

                if ep <= 500:
                    ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                    w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                    loss = rl2_error(w_, w)
                    opt_adam.zero_grad()
                    loss.backward() # use the l2 relative loss
                    opt_adam.step()
                else:
                    def loss_closure():
                        ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                        w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                        loss = rl2_error(w_, w)
                        opt_lbfgs.zero_grad()
                        loss.backward()
                        return loss 
                    opt_lbfgs.step(loss_closure)
                    loss =loss_closure()
                train_rl2 += loss.item()

            smooth_kernel.eval()
            test_rl2 = 0.0
            with torch.no_grad():
                for u, w in test_loader:
                    u, w = u.to(device), w.to(device)
                    bsz = u.shape[0]

                    ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                    w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                    rl2 = rl2_error(w_, w)
                    test_rl2 += rl2.item()

            train_rl2 = train_rl2/len(train_loader)
            test_rl2 = test_rl2/len(test_loader)

            train_rl2_hist.append(train_rl2)
            test_rl2_hist.append(test_rl2)

            if test_rl2 < test_rl2_best:
                test_rl2_best = test_rl2
                torch.save(smooth_kernel, nn_outpath)
                np.save(kernel_outpath, ks.detach().cpu().numpy())

        smooth_kernel = torch.load(nn_outpath)
        opt_adam = torch.optim.Adam(correction_kernel.parameters(), lr=lr_adam*0.1)
        opt_lbfgs = torch.optim.LBFGS(correction_kernel.parameters(), lr=lr_lbfgs)

        pbar = trange(epochs)
        for ep in pbar:
            pbar.set_description(
                "train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
            smooth_kernel.eval()
            correction_kernel.train()
            
            train_rl2 = 0

            for u, w in train_loader:
                u, w = u.to(device), w.to(device)
                bsz = u.shape[0]

                if ep <= 500:
                    ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                    kc = correction_kernel_approx(correction_kernel, x, args.lk)
                    nglobal = ks.shape[-1]
                    nlocal = kc.shape[-1]
                    nmid = (nglobal-1)//2
                    nlocal_half = (nlocal-1)//2
                    ks[:,:,nmid-nlocal_half:nmid+nlocal_half+1] += kc
                    w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                    loss = rl2_error(w_, w)
                    opt_adam.zero_grad()
                    loss.backward() # use the l2 relative loss
                    opt_adam.step()
                else:
                    def loss_closure():
                        ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                        kc = correction_kernel_approx(correction_kernel, x, args.lk)
                        nglobal = ks.shape[-1]
                        nlocal = kc.shape[-1]
                        nmid = (nglobal-1)//2
                        nlocal_half = (nlocal-1)//2
                        ks[:,:,nmid-nlocal_half:nmid+nlocal_half+1] += kc
                        w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                        loss = rl2_error(w_, w)
                        opt_lbfgs.zero_grad()
                        loss.backward()
                        return loss 
                    opt_lbfgs.step(loss_closure)
                    loss =loss_closure()
                train_rl2 += loss.item()

            correction_kernel.eval()
            test_rl2 = 0.0
            with torch.no_grad():
                for u, w in test_loader:
                    u, w = u.to(device), w.to(device)
                    bsz = u.shape[0]

                    ks = smooth_kernel_approx(smooth_kernel, x, args.lk)
                    kc = correction_kernel_approx(correction_kernel, x, args.lk)
                    nglobal = ks.shape[-1]
                    nlocal = kc.shape[-1]
                    nmid = (nglobal-1)//2
                    nlocal_half = (nlocal-1)//2
                    ks[:,:,nmid-nlocal_half:nmid+nlocal_half+1] += kc
                    w_ = toeplitz_matrix_vector_multiplicaiton(ks, u)
                    rl2 = rl2_error(w_, w)
                    test_rl2 += rl2.item()

            train_rl2 = train_rl2/len(train_loader)
            test_rl2 = test_rl2/len(test_loader)

            train_rl2_hist.append(train_rl2)
            test_rl2_hist.append(test_rl2)

            if test_rl2 < test_rl2_best:
                test_rl2_best = test_rl2
                # torch.save(correction_kernel_approx, nn_outpath)
                np.save(kernel_outpath, ks.detach().cpu().numpy())
                

        # scheduler.step(test_rl2_best)
    
    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist)
