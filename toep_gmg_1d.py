import os
import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
import json 
from tqdm import trange 

from src.model import MLP
from src.toep_gmg import Toep_GMG1D
from src.dataset import load_dataset_1d 
from src.utils import (init_records, save_hist, get_seed, rl2_error)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train Toep Green MG Net")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--train_post', type=str, default='3.00e-01',
                        help='lambda of inputs funcitons, higher smoother')
    parser.add_argument('--test_post', type=str, default='3.00e-01',
                        help='lambda of inputs funcitons, higher smoother')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--lr_adam', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--ep_adam', type=int, default=1000,
                        help='learning rate')
    parser.add_argument('--sch', action='store_true',
                        help='reduce rl on plateau scheduler')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch size')
    
    # parameters for Toep GreenMGNet
    parser.add_argument('--n', type=int, default=9,
                        help='number of total levels')
    parser.add_argument('--k', type=int, default=5,
                        help='number of levels to coarse from fine')
    parser.add_argument('--m', type=int, default=7,
                        help='local range for correction on each level')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation functions')
    parser.add_argument('--h', type=int, default=50,
                        help='hidden channel for mlp')
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    get_seed(args.seed, printout=False)

    batch_size = args.bsz
    lr_adam = args.lr_adam
    epochs = args.ep_adam

    in_channels = 1 # FOR TOEP KERNEL
    out_channels = 1
    hidden_channels = args.h

    ################################################################
    # prepare log
    ################################################################
    device = torch.device(f'cuda:{args.device}')
    resolution = 2**args.n+1

    data_root = '/workdir/GreenMgNet/dataset'
    log_root = '/workdir/GreenMgNet/results/'
    task_nm = args.task
    exp_nm = '-'.join([
        'Toep_GMGN1D', args.act, str(2**args.n+1), 
        str(args.h), str(args.k), str(args.m), str(args.seed), 
        args.train_post, args.test_post])
    hist_outpath, pred_outpath, nn_outpath, kernel_outpath, cfg_outpath = init_records(log_root, task_nm, exp_nm)

    if os.path.exists(hist_outpath):
        print(f"{hist_outpath} file exists")
        print('-'*20)
        exit()

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
    r = 12 - args.n
    train_loader, test_loader = load_dataset_1d(args.task, data_root, r)

    ################################################################
    # build model
    ################################################################
    layers = [in_channels] + [hidden_channels]*4 + [out_channels]
    kernel = MLP(layers, nonlinearity=args.act).to(device)
    model = Toep_GMG1D(n=args.n, m=args.m, k=args.k, kernel=kernel, device=device)

    opt_adam = torch.optim.Adam(kernel.parameters(), lr=lr_adam)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt_adam, gamma=0.95)

    ################################################################
    # training and evaluation
    ################################################################
    
    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 1

    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description("train l2 {:.2e} - lr {:.4e}".format(train_rl2, sch.get_last_lr()[0]))
        
        model.kernel.train()
        train_rl2 = 0
        for f, u in train_loader:
            # fetch data batch 
            u, f = u.to(device), f.to(device)
            u = torch.squeeze(u).T
            f = torch.squeeze(f).T

            # eval kernel
            model.eval_ml_K()
            model.assemble_K()

            # calc kernel integral
            u_ = model.fft_kint(f)

            # calc loss 
            loss = rl2_error(u_.T, u.T)

            opt_adam.zero_grad()
            loss.backward() # use the l2 relative loss
            opt_adam.step()
            train_rl2 += loss.item()
        
        if args.sch:
            sch.step()

        train_rl2 = train_rl2/len(train_loader)
        train_rl2_hist.append(train_rl2)

    model.kernel.eval()
    test_rl2 = 0.0
    with torch.no_grad():
        for f, u in test_loader:
            u, f = u.to(device), f.to(device)
            u = torch.squeeze(u).T
            f = torch.squeeze(f).T

            u_ = model.fft_kint(f)
            rl2 = rl2_error(u_.T, u.T)
            test_rl2 += rl2.item()

    test_rl2 = test_rl2/len(test_loader)

    print('test_rl2 : {:.4e}'.format(test_rl2))
    print(f'save model at : {nn_outpath}')    
    torch.save(model.kernel.state_dict(), nn_outpath)
    save_hist(hist_outpath, train_rl2_hist, test_rl2)
    K = model.K_h.cpu().detach().numpy()
    print(f'save kernel at : {kernel_outpath} ', K.shape)
    np.save(kernel_outpath, K)
    print('-'*20)