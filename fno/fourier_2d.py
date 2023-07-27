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
from models import FNO2d

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a fourier neural operator 2d")
    args = get_arguments(parser)

    with open(os.path.join(args.cfg_path, f'fno2d-cfg.yaml')) as f:
        model_cfg = EasyDict(yaml.full_load(f))
    get_seed(args.seed, printout=True)
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{args.device}')

    tra_res = int(((421-1)/args.trasub)+1)    
    test_res = int(((421-1)/args.trasub)+1)

    if args.mlevel == -1:
        mlevel = 'x'
    else:
        mlevel = args.mlevel

    model_nm = f'fno2d-m{model_cfg.modes}-w{model_cfg.width}'+ \
               f'-{tra_res}-{test_res}'+ \
               f'-cl{args.clevel}-ml{mlevel}' + \
               f'-seed{args.seed}'
    log_root = os.path.join(args.log_dir, f'exp2d/fno2d/{args.dataset_nm}')
    os.makedirs(log_root, exist_ok=True)
    model_out_path = os.path.join(log_root, model_nm + '.pth')
    csv_out_path = os.path.join(log_root, model_nm + '.csv')

    ################################################################
    # load_dataset
    ################################################################
    pprint.pprint(args, width=1)
    train_loader, test_loader, u_normalizer = load_dataset_2d(args)

    ################################################################
    # init_model
    ################################################################
    pprint.pprint(model_cfg, width=1)
    model = FNO2d(model_cfg.modes,
                  model_cfg.modes,
                  model_cfg.width,
                  clevel=args.clevel,
                  mlevel=args.mlevel).to(device)
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################
    # iterations = args.epochs*(args.ntrain//args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
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
        
    log_df = pd.DataFrame({'train_l2': train_log, 'test_l2': test_log})
    log_df.to_csv(csv_out_path, index=False)

