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
import torch.autograd.profiler as profiler
from flopth import flopth 
import nvsmi

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="galerkin transformer 2d profile")
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

    gpu_nm = torch.cuda.get_device_name(device=args.device).split(' ')[1]
    model_nm = f'gt2d-h{model_cfg.nhead}-w{model_cfg.width}'+ \
               f'-{tra_res}-{test_res}'+ \
               f'-cl{args.clevel}-ml{mlevel}'
    
    log_root = os.path.join(data_cfg.plog_dir, gpu_nm, f'exp2d/gt2d')
    os.makedirs(log_root, exist_ok=True)
    prof_out_path = os.path.join(log_root, model_nm + '.json')
    if os.path.exists(prof_out_path):
        print(f"{prof_out_path} file exists")
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

    print(f'profile result out path : {prof_out_path}')

    # ---------------------------------------------------------
    # profile training time 
    tra_epochtime = []
    cuda_empty_cache(args.device)
    pbar = trange(epochs) # 10 for warm up
    model.train()
    for ep in pbar:    
        t1 = default_timer()
        for a, x, u in train_loader:
            bsz, seq_lx, seq_ly, _ = a.shape
            a, x, u = a.to(device), x.to(device), u.to(device)
            optimizer.zero_grad()
            u_ = model(a=a,x=x).reshape(bsz, seq_lx, seq_ly)
            loss = myloss(u_.view(bsz,-1), u.view(bsz,-1))
            loss.backward()
            optimizer.step()
        scheduler.step()
        scheduler.step()
        t2 = default_timer()
        tra_time = (t2 - t1) / len(train_loader.dataset)
        if ep == (epochs - 1):
            tra_mem = profile_gpumem(args.device)
        tra_epochtime.append(tra_time)
    
    tra_avg_epoch_time = np.mean(tra_epochtime[1:-1])
    print("train epoch time : {:.5}s".format(tra_avg_epoch_time))
    print("train mem : {:}MB".format(tra_mem))

    # ---------------------------------------------------------
    # profile inference time 
    infer_epochtime = []

    cuda_empty_cache(args.device)
    pbar = trange(epochs) # 10 for warm up
    model.eval()

    with torch.no_grad():
        for ep in pbar:
            t1 = default_timer()
            for a, x, u in test_loader:
                bsz, seq_lx, seq_ly, _ = a.shape
                a, x, u = a.to(device), x.to(device), u.to(device)
                u_ = model(a=a, x=x).reshape(bsz, seq_lx, seq_ly)
            t2 = default_timer()
            infer_time = (t2 - t1) / len(test_loader.dataset)
            if ep == (epochs - 1):
                infer_mem = profile_gpumem(args.device)
            infer_epochtime.append(infer_time)
    

    infer_avg_epoch_time = np.mean(infer_epochtime[1:-1])
    print("infer epoch time : {:.5}s".format(infer_avg_epoch_time))
    print("infer mem : {:}MB".format(infer_mem))
    

    # ---------------------------------------------------------
    # profile FLOPs and params
    model.eval()
    with torch.device('cuda:{:}'.format(args.device)):
        flops, params = flopth(model, inputs=(a, x))
    
    print("model FLOPs : {:}".format(flops))
    print("model #params : {:}".format(params))

    profile_dict = {
        'bsz' : args.batch_size,
        'tra_time': tra_avg_epoch_time,
        'infer_time': infer_avg_epoch_time,
        'tra_mem': tra_mem,
        'infer_mem': infer_mem,
        'model_FLOPs': flops, 
        'model_nparam': params       
    }

    with open(prof_out_path, "w") as f:
        json.dump(profile_dict, f)