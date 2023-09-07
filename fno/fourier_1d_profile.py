import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

import pandas as pd
import copy 
import torch 
import torch.nn.functional as F
from easydict import EasyDict
from tqdm import trange, tqdm
from timeit import default_timer
from data import load_dataset_1d
import argparse
import yaml 
import pprint
from utils import *
from models import FNO1d
import torch.autograd.profiler as profiler
from flopth import flopth 
import nvsmi

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a fourier neural operator 1d")
    args = get_arguments(parser)
    cfg_root = '/'.join(current_path.split('/')[:-1] + ['cfgs'])
    with open(os.path.join(cfg_root, f'fno1d-cfg.yaml')) as f:
        model_cfg = EasyDict(yaml.full_load(f))
    with open(os.path.join(cfg_root, f'data_log-cfg.yaml')) as f:
        data_cfg = EasyDict(yaml.full_load(f))
        vars(args)['dataset_path'] = data_cfg.dataset_path
    get_seed(args.seed, printout=True)
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{args.device}')

    tra_res = 2**(13 - int(np.log2(args.trasub)))
    test_res = 2**(13 - int(np.log2(args.testsub)))

    if args.mlevel == -1:
        mlevel = 'x'
    else:
        mlevel = args.mlevel

    gpu_nm = torch.cuda.get_device_name(device=args.device).split(' ')[1]
    model_nm = f'fno1d-m{model_cfg.modes}-w{model_cfg.width}'+ \
               f'-{tra_res}-{test_res}'+ \
               f'-cl{args.clevel}-ml{mlevel}'
    
    log_root = os.path.join(data_cfg.plog_dir, gpu_nm, f'exp1d/fno1d/')
    os.makedirs(log_root, exist_ok=True)
    prof_out_path = os.path.join(log_root, model_nm + '.json')
    if os.path.exists(prof_out_path):
        print(f"{prof_out_path} file exists")
        exit()

    ################################################################
    # load_dataset
    ################################################################
    pprint.pprint(args, width=1)

    ################################################################
    # init_model
    ################################################################
    pprint.pprint(model_cfg, width=1)
    model = FNO1d(model_cfg.modes, 
                 model_cfg.width,
                 clevel=args.clevel,
                 mlevel=args.mlevel).to(device)
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    nbatch = args.epochs
    myloss = LpLoss(size_average=False)

    a = torch.randn(args.batch_size, tra_res).to(device)
    x = torch.randn(args.batch_size, tra_res).to(device)
    u = torch.randn(args.batch_size, tra_res).to(device)

    print(f'profile result out path : {prof_out_path}')

    # ---------------------------------------------------------
    # profile training time 

    tra_batchtime = []
    torch.cuda.empty_cache()
    model.train()
    pbar = trange(nbatch+10+1) # 10 for warm up
    for ep in pbar:
        t1 = default_timer()
        bsz, seq_len = a.shape
        optimizer.zero_grad()

        u_ = model(a=a,x=x).reshape(bsz, seq_len)
        loss = myloss(u_.view(bsz,-1), u.view(bsz,-1))
        loss.backward()
        optimizer.step()
        t2 = default_timer()
        tra_btime = t2 - t1
        tra_batchtime.append(tra_btime)
        if ep == nbatch+10:
            tra_mem = profile_gpumem(args.device)

    tra_bavgt = np.mean(tra_batchtime[10:-1])
    print("train batch time : {:.5}s".format(tra_bavgt))
    print("train batch mem : {:}MB".format(tra_mem))
    
    # ---------------------------------------------------------
    # profile inference time 

    infer_batchtime = []
    torch.cuda.empty_cache()
    model.eval()
    pbar = trange(nbatch+10+1)
    with torch.no_grad():
        for ep in pbar:
            t1 = default_timer()
            bsz, seq_len = a.shape
            u_ = model(a=a, x=x).reshape(bsz, seq_len)
            t2 = default_timer()
            infer_btime = t2 - t1
            infer_batchtime.append(infer_btime)
            if ep == nbatch+10:
                infer_mem = profile_gpumem(args.device)
    
    infer_bavgt = np.mean(infer_batchtime[10:-1])
    print("infer batch time : {:.5}s".format(infer_bavgt))
    print("infer batch time : {:}MB".format(infer_mem))
    

    # ---------------------------------------------------------
    # profile FLOPs and params
    
    model.eval()
    flops, params = flopth(model, inputs=(a, x))

    print("model FLOPs : {:}".format(flops))
    print("model #params : {:}".format(params))

    profile_dict = {
        'bsz' : args.batch_size,
        'tra_time': tra_bavgt,
        'infer_time': infer_bavgt,
        'tra_mem': tra_mem,
        'infer_mem': infer_mem,
        'model_FLOPs': flops, 
        'model_nparam': params       
    }

    with open(prof_out_path, "w") as f:
        json.dump(profile_dict, f)