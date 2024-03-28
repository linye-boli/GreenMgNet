import os
import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
import json 
from tqdm import trange 

from src.model import MLP
from src.green_net import GreenNet1D
from src.dataset import load_dataset_1d 
from src.utils import (
    init_records, save_hist, get_seed, rl2_error)


torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train Green Net")
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
    parser.add_argument('--bsz', type=int, default=8,
                        help='batch size')

    # parameters for GreenNet
    parser.add_argument('--n', type=int, default=9,
                        help='number of total levels')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation functions')
    parser.add_argument('--h', type=int, default=64,
                        help='hidden channel for mlp')
    args = parser.parse_args()
    print(args)

    ################################################################
    #  configurations
    ################################################################
    get_seed(args.seed, printout=False)

    batch_size = args.bsz
    lr_adam = args.lr_adam
    epochs = args.ep_adam

    in_channels = 2
    out_channels = 1
    hidden_channels = args.h

    ################################################################
    # prepare log
    ################################################################
    device = torch.device(f'cuda:{args.device}')
    print(device)
    res = str(2**args.n+1)

    data_root = '/workdir/GreenMgNet/dataset'
    log_root = '/workdir/GreenMgNet/results/'
    task_nm = args.task
    exp_nm = '-'.join([
        'GN1D', args.act, res,
        str(args.h), str(args.seed), 
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
    train_loader, test_loader = load_dataset_1d(args.task, data_root, bsz=args.bsz, res=res)

    ################################################################
    # build model
    ################################################################
    layers = [in_channels] + [hidden_channels]*4 + [out_channels]
    kernel = MLP(layers, nonlinearity=args.act).to(device)
    model = GreenNet1D(n=args.n, kernel=kernel, device=device)

    opt_adam = torch.optim.Adam(kernel.parameters(), lr=lr_adam)
    step_size = 100
    gamma = 0.9
    sch = torch.optim.lr_scheduler.StepLR(opt_adam, step_size=step_size, gamma=gamma)

    ################################################################
    # training and evaluation
    ################################################################
    
    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 1

    # training stage
    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description("train l2 {:.2e} - lr {:.4e}".format(train_rl2, sch.get_last_lr()[0]))
               
        model.kernel.train()
        train_rl2 = 0
        for f, u in train_loader:
            # fetch data batch 
            u, f = u.to(device), f.to(device)
            u = torch.squeeze(u).T # bsz x xn
            f = torch.squeeze(f).T # bsz x xn

            # # eval kernel
            # model.rand_sub()
            # model.eval_K_sub()

            # # calc kernel integral
            # u_ = model.sub_kint(f)

            model.eval_K()
            u_ = model.full_kint(f)


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

    # test stage
    model.kernel.eval()
    test_rl2 = 0.0
    model.eval_K_batch()
    with torch.no_grad():
        for f, u in test_loader:
            u, f = u.to(device), f.to(device)
            u = torch.squeeze(u).T
            f = torch.squeeze(f).T

            u_ = model.batch_kint(f)
            rl2 = rl2_error(u_.T, u.T)
            test_rl2 += rl2.item()
    test_rl2 = test_rl2/len(test_loader)
    print('test_rl2 : {:.4e}'.format(test_rl2))
    print(f'save model at : {nn_outpath}')    
    torch.save(model.kernel.state_dict(), nn_outpath)

    save_hist(hist_outpath, train_rl2_hist, test_rl2)
    K = model.K_hh.cpu().detach().numpy()
    print(f'save kernel at : {kernel_outpath} ', K.shape)
    np.save(kernel_outpath, K)
    print('-'*20)