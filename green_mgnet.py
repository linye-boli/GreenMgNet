import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
from dataset import load_dataset_1d 
from datetime import datetime 
from utils import (
    rl2_error, init_records, save_hist, get_seed,
    train_model, eval_model)
import json 
from tqdm import trange 
from einops import rearrange 
from model import GMGN, GL


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train Green-MgNet")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--fine', action='store_true',
                        help='fine grid(8193) or coarse grid(513) problem')
    parser.add_argument('--model', type=str, default='mlp',
                        help='type of neural network to model kernel, mlp/low-rank')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation functions')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--lr_adam', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_lbfgs', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--ep_adam', type=int, default=1000,
                        help='learning rate')
    parser.add_argument('--ep_lbfgs', type=int, default=50,
                        help='learning rate')
    parser.add_argument('--sch', action='store_true',
                        help='reduce rl on plateau scheduler')

    # parameters for GreenMgNet
    parser.add_argument('--k', type=int, default=5,
                        help='number of levels to coarse from fine')
    parser.add_argument('--m', type=int, default=7,
                        help='local range for correction on each level')
    parser.add_argument('--r', type=int, default=1,
                        help='rank for multipole')
    parser.add_argument('--thr', type=float, default=1e-3,
                        help='maximum gradient acceptted in kernel function')
    parser.add_argument('--w', type=str, default='log',
                        help='weights for gradient norm loss function')
    
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    get_seed(args.seed)

    batch_size = 1000
    lr_adam = args.lr_adam
    lr_lbfgs = args.lr_lbfgs
    epochs = args.ep_adam + args.ep_lbfgs

    if args.task in ['cosine', 'logarithm']:        
        in_channels = 1
        out_channels = 1
        assert args.model in ['toep_mg', 'toep_gl']
    elif args.task in [
        'cusp',  'schrodinger', 'viscous_shock',
        'laplace', 'interior_layer']:
        in_channels = 2
        out_channels = 1
        assert args.model in ['gl', 'lr_mg', 'lr_gl', 'dd_mg', 'mlplr_mg']

    loss_cfg = {'rl2' : True}
    if args.thr != 0:
        loss_cfg['gnorm'] = {'thr':args.thr, 'w':args.w}
    else:
        loss_cfg['gnorm'] = None

    ################################################################
    # prepare log
    ################################################################
    device = torch.device(f'cuda:{args.device}')
    if args.fine:
        resolution = 8193
    else:
        resolution = 513

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

    ################################################################
    # build model
    ################################################################
    if args.model in ['lr_mg', 'dd_mg', 'mlplr_mg', 'toep_mg']:
        model = GMGN(
            in_channels=in_channels, 
            out_channels=out_channels,
            n = resolution, # finest resolution
            k = args.k, # coarse level
            m = args.m, # local range
            r = args.r, # rank
            mtype = args.model).to(device)

    elif args.model in ['lr_gl', 'toep_gl', 'gl']:
        model = GL(
            in_channels=in_channels, 
            out_channels=out_channels,
            act=args.act,
            r=args.r,
            mtype=args.model).to(device)

    opt_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)

    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lr_lbfgs, 
        history_size=50,
        tolerance_change=1e-20,
        tolerance_grad=1.0*np.finfo(float).eps,
        max_iter=5*10**4,
        max_eval=10**5)

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_adam, patience=20, verbose=True)

    ################################################################
    # training and evaluation
    ################################################################
    
    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 1
    test_rl2_best = 1

    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description(
            "train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
        
        if ep <= args.ep_adam:
            model, train_rl2 = train_model(train_loader, model, opt_adam, device, loss_cfg)
        else:
            model, train_rl2 = train_model(train_loader, model, opt_lbfgs, device)
            
        test_rl2, k_ = eval_model(test_loader, model, device)
        if args.sch:
            sch.step(test_rl2)

        train_rl2_hist.append(train_rl2)
        test_rl2_hist.append(test_rl2)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
            torch.save(model.state_dict(), nn_outpath)
            np.save(kernel_outpath, k_)
    
    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist)
