import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

import pandas as pd
import copy 
import torch 
import torch.nn.functional as F
from easydict import EasyDict
from tqdm import trange
from timeit import default_timer
from data import load_dataset_2d
import argparse
import yaml 
import pprint
from utils import *
from models import GT2d

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a fourier neural operator 1d")
    args = get_arguments(parser)
    cfg_root = '/'.join(current_path.split('/')[:-1] + ['cfgs'])
    with open(os.path.join(cfg_root, f'gt2d-cfg.yaml')) as f:
        model_cfg = EasyDict(yaml.full_load(f))
    with open(os.path.join(cfg_root, f'data_log-cfg.yaml')) as f:
        data_cfg = EasyDict(yaml.full_load(f))
        vars(args)['dataset_path'] = data_cfg.dataset_path
    # get_seed(args.seed, printout=True)
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{args.device}')

    tra_res = int(((421-1)/args.trasub)+1)    
    test_res = int(((421-1)/args.trasub)+1)

    if args.mlevel == -1:
        mlevel = 'x'
    else:
        mlevel = args.mlevel

    model_nm = f'gt2d-h{model_cfg.nhead}-w{model_cfg.width}'+ \
               f'-{tra_res}-{test_res}'+ \
               f'-cl{args.clevel}-ml{mlevel}' + \
               f'-seed{args.seed}'
    log_root = os.path.join(data_cfg.log_dir, f'exp2d/gt2d/{args.dataset_nm}')
    os.makedirs(log_root, exist_ok=True)
    model_out_path = os.path.join(log_root, model_nm + '.pth')
    csv_out_path = os.path.join(log_root, model_nm + '.csv')
    if os.path.exists(csv_out_path):
        print(f"{csv_out_path} file exists")
        exit()

    ispass = pass_check('gt2d', tra_res, args.clevel, mlevel, model_nm)
    if ispass:
        exit()


    ################################################################
    # load_dataset
    ################################################################
    pprint.pprint(args, width=1)
    train_loader, test_loader, u_normalizer = load_dataset_2d(args)

    ################################################################
    # init_model
    ################################################################
    pprint.pprint(model_cfg, width=1)
    model = GT2d(model_cfg.width, 
                 model_cfg.nhead,
                 clevel=args.clevel,
                 mlevel=args.mlevel).to(device)
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.epochs

    myloss = LpLoss(size_average=False)
    u_normalizer.to(device)
    test_l2_best = 1

    train_log = []
    test_log = []

    print(f'out log csv : {csv_out_path}')
    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description("best l2 {:.6f}".format(test_l2_best))
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for a, x, u in train_loader:
            bsz, seq_lx, seq_ly, _ = a.shape
            a, x, u = a.to(device), x.to(device), u.to(device)
            optimizer.zero_grad()

            u_ = model(a=a,x=x).reshape(bsz, seq_lx, seq_ly)
            mse = F.mse_loss(u_.view(bsz, -1), u.view(bsz, -1), reduction='mean')

            u_ = u_normalizer.decode(u_)
            u = u_normalizer.decode(u)
            loss = myloss(u_.view(bsz,-1), u.view(bsz,-1))
            loss.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for a, x, u in test_loader:
                bsz, seq_lx, seq_ly, _ = a.shape
                a, x, u = a.to(device), x.to(device), u.to(device)

                u_ = model(a=a, x=x).reshape(bsz, seq_lx, seq_ly)
                u_ = u_normalizer.decode(u_)
                test_l2 += myloss(u_.view(bsz, -1), u.view(bsz, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= args.ntrain
        test_l2 /= args.ntest
        t2 = default_timer()
        

        train_log.append(train_l2)
        test_log.append(test_l2)

        if test_l2 < test_l2_best:
            test_l2_best = test_l2
            if args.save:
                torch.save(model, model_out_path)
        
        if ep > 1:
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            if remaining / 3600 > 3:
                print('too long for training')
                exit()
        
    log_df = pd.DataFrame({'train_l2': train_log, 'test_l2': test_log})
    log_df.to_csv(csv_out_path, index=False)

