import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
from dataset import load_dataset_1d 
from datetime import datetime 
from utils import rl2_error, init_records, save_hist 
import json 
from tqdm import trange 
from einops import rearrange 
from model import ToepGreenMgNet, ToepGreenNet

torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train toeplitz Green-MgNet in 1d")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--lc', type=int, default=3,
                        help='level of local correction')
    parser.add_argument('--mtype', type=str, default='GMG',
                        help='local correction range')
    
    
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    batch_size = 1000
    lr_adam = 0.001
    lr_lbfgs = .01
    epochs = 520

    in_channels = 1
    out_channels = 1
    lc = args.lc
    modes = 16
    l = 9
    n = 2**l+1

    ################################################################
    # prepare log
    ################################################################
    now = datetime.now()
    exp_nm = now.strftime("exp-%Y-%m-%d-%H-%M-%S")
    device = torch.device(f'cuda:{args.device}')
    data_root = '/workdir/pde_data/green_learning/data1d_513'
    log_root = f'/workdir/{args.mtype}/results/'
    model_nm = args.mtype
    task_nm = args.task
    hist_outpath, pred_outpath, model_operator_outpath, _, cfg_outpath = init_records(
        task_nm, log_root, model_nm, exp_nm)
    print('output files:')
    print(hist_outpath)
    print(pred_outpath)
    print(model_operator_outpath)
    print(cfg_outpath)

    with open(cfg_outpath, 'w') as f:
        cfg_dict = vars(args)
        cfg_dict['model_nm'] = model_nm
        json.dump(cfg_dict, f)
    
    ################################################################
    # read data
    ################################################################
    train_loader, test_loader = load_dataset_1d(task_nm, data_root, normalize=False)

    ################################################################
    # build model
    ################################################################
    if args.mtype == 'TGMG':
        model = ToepGreenMgNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            lc = lc).to(device)
    elif args.mtype == 'TG':
        model = ToepGreenNet(
            in_channels=in_channels, 
            out_channels=out_channels).to(device)

    ################################################################
    # training and evaluation
    ################################################################
    opt_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lr_lbfgs)

    train_rl2_hist = []
    test_rl2_hist = []
    train_rl2 = 1
    test_rl2_best = 1
    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description(
            "train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
        model.train()
        train_rl2 = 0

        for u, w in train_loader:
            u, w = u.to(device), w.to(device)
            bsz = u.shape[0]
            x = torch.linspace(-1,1,2*n-1)[None][None].to(device)
            x = x.repeat(bsz,1,1)

            if ep <= 500:
                w_ = model(u, x)
                loss = rl2_error(w_, w)
                opt_adam.zero_grad()
                loss.backward() # use the l2 relative loss
                opt_adam.step()
                # sch_adam.step()
            else:
                def loss_closure():
                    w_ = model(u, x)
                    loss = rl2_error(w_, w)
                    opt_lbfgs.zero_grad()
                    loss.backward()
                    return loss 
                opt_lbfgs.step(loss_closure)
                loss =loss_closure()
            train_rl2 += loss.item()
        
        model.eval()
        test_rl2 = 0.0
        with torch.no_grad():
            for u, w in test_loader:
                u, w = u.to(device), w.to(device)
                bsz = u.shape[0]
                x = torch.linspace(-1,1,2*n-1)[None][None].to(device)
                x = x.repeat(bsz,1,1)

                w_ = model(u, x)
                rl2 = rl2_error(w_, w)
                test_rl2 += rl2.item()

        train_rl2 = train_rl2/len(train_loader)
        test_rl2 = test_rl2/len(test_loader)

        train_rl2_hist.append(train_rl2)
        test_rl2_hist.append(test_rl2)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
            torch.save(model, model_operator_outpath)

        # scheduler.step(test_rl2_best)
    
    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist)
