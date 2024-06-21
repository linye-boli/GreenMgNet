import os
import numpy as np
import torch 
import argparse 
import torch.nn.functional as F 
from src.dataset import load_dataset_1d 
from src.utils import (
    init_records, save_hist, get_seed, rl2_error)
import json 
from tqdm import trange
import time

from src.model import MLP
from src.dd_gmg import DD_GMG1D
from src.green_net import GreenNet1D

n = 9
bsz = 200
seq_len = 2**9+1
u = torch.randn(seq_len, bsz)
f = torch.randn(seq_len, bsz)

ps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25, 0.30, 0.4, 0.5, 0.7, 0.9, 1.0]

def gl_profile(cuda, p, u, f, warmup_iters=10, iters=100):
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    u = u.to(device)
    f = f.to(device)

    in_channels = 2
    out_channels = 1
    hidden_channels = 50 
    layers = [in_channels] + [hidden_channels]*4 + [out_channels]
    kernel = MLP(layers, nonlinearity='rational', aug='none_1d').to(device)
    model = GreenNet1D(n=n, kernel=kernel, device=device, p=p)
    model.rand_sub()
    opt_adam = torch.optim.Adam(kernel.parameters(), lr=1e-3)

    model.kernel.train()

    # warmup
    for _ in trange(warmup_iters):
        if p < 1:
            model.eval_K_sub()
            u_ = model.sub_kint(f)
            # calc loss 
            loss = rl2_error(u_.T, u[model.sub].T)
        else:
            model.eval_K()
            u_ = model.full_kint(f)
            # calc loss 
            loss = rl2_error(u_.T, u.T)
        
        opt_adam.zero_grad()
        loss.backward() # use the l2 relative loss
        opt_adam.step()

    if cuda:
        torch.cuda.synchronize()
    
    # training stage measurement
    times = []
    keval_times = []
    kint_times = []

    for _ in trange(iters):
        if cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        if p < 1:
            keval_start_time = time.time()
            model.eval_K_sub()
            keval_end_time = time.time()
            u_ = model.sub_kint(f)
            kint_end_time = time.time()
            # calc loss 
            loss = rl2_error(u_.T, u[model.sub].T)
        else:
            keval_start_time = time.time()
            model.eval_K()
            keval_end_time = time.time()
            u_ = model.full_kint(f)
            kint_end_time = time.time()
            # calc loss 
            loss = rl2_error(u_.T, u.T)
        
        opt_adam.zero_grad()
        loss.backward() # use the l2 relative loss
        opt_adam.step()

        if cuda:
            torch.cuda.synchronize()    
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        keval_elapsed = keval_end_time - keval_start_time
        keval_times.append(keval_elapsed)

        kint_elapsed = kint_end_time - keval_end_time
        kint_times.append(kint_elapsed)
    
    avg_time = sum(times) / iters
    keval_avg_time = sum(keval_times) / iters
    kint_avg_time = sum(kint_times) / iters

    # inference stage measurement
    model.kernel.eval()
    model.eval_K()
    if cuda:
        torch.cuda.synchronize()    

    infer_times = []
    with torch.no_grad():

        for _ in trange(iters):
            if cuda:
                torch.cuda.synchronize()        
            infer_start_time = time.time()

            u_ = model.full_kint(f)
            
            if cuda:
                torch.cuda.synchronize()    
            infer_end_time = time.time()

            infer_elapsed =  infer_end_time - infer_start_time
            infer_times.append(infer_elapsed)
    infer_avg_time = sum(infer_times) / iters


    pf_info = {
        "model" : 'GL',
        "p" : p,
        "cuda" : cuda,
        "train_time" : avg_time, 
        "keval_avg_time": keval_avg_time, 
        "kint_avg_time" : kint_avg_time, 
        "infer_avg_time" : infer_avg_time}

    return pf_info

def gmg_profile(cuda, m, k, u, f, warmup_iters=10, iters=100):
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    u = u.to(device)
    f = f.to(device)

    in_channels = 2
    out_channels = 1+1
    hidden_channels = 50 
    layers = [in_channels] + [hidden_channels]*4 + [out_channels]
    kernel = MLP(layers, nonlinearity='rational', aug='aug2_1d').to(device)
    model = DD_GMG1D(n=n, m=m, k=k, kernel=kernel, device=device)
    p = model.pts_ratio
    opt_adam = torch.optim.Adam(kernel.parameters(), lr=1e-3)

    model.kernel.train()

    # warmup
    for _ in trange(warmup_iters):
        model.restrict_ml_f(f)
        model.eval_ml_K()
        u_ = model.ml_kint()

        # calc loss 
        loss = rl2_error(u_.T, u.T)
        
        opt_adam.zero_grad()
        loss.backward() # use the l2 relative loss
        opt_adam.step()

    if cuda:
        torch.cuda.synchronize()
    
    # training stage measurement
    times = []
    keval_times = []
    kint_times = []

    for _ in trange(iters):
        if cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        
        keval_start_time = time.time()
        model.eval_ml_K()
        keval_end_time = time.time()
        
        model.restrict_ml_f(f)
        u_ = model.ml_kint()
        kint_end_time = time.time()
        
        # calc loss 
        loss = rl2_error(u_.T, u.T)
        
        opt_adam.zero_grad()
        loss.backward() # use the l2 relative loss
        opt_adam.step()

        if cuda:
            torch.cuda.synchronize()    
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        keval_elapsed = keval_end_time - keval_start_time
        keval_times.append(keval_elapsed)

        kint_elapsed = kint_end_time - keval_end_time
        kint_times.append(kint_elapsed)
    
    avg_time = sum(times) / iters
    keval_avg_time = sum(keval_times) / iters
    kint_avg_time = sum(kint_times) / iters

    # inference stage measurement
    model.kernel.eval()
    if cuda:
        torch.cuda.synchronize()    

    infer_times = []
    with torch.no_grad():

        for _ in trange(iters):
            if cuda:
                torch.cuda.synchronize()        
            infer_start_time = time.time()

            model.restrict_ml_f(f)
            u_ = model.ml_kint()
            
            if cuda:
                torch.cuda.synchronize()    
            infer_end_time = time.time()

            infer_elapsed =  infer_end_time - infer_start_time
            infer_times.append(infer_elapsed)
    infer_avg_time = sum(infer_times) / iters


    pf_info = {
        "model" : 'GreenMGNet',
        "m" : m,
        "k" : k,
        "p" : p,
        "cuda" : cuda,
        "train_time" : avg_time, 
        "keval_avg_time": keval_avg_time, 
        "kint_avg_time" : kint_avg_time, 
        "infer_avg_time" : infer_avg_time}

    return pf_info


if __name__ == "__main__":
    n = 9
    bsz = 200
    seq_len = 2**9+1
    u = torch.randn(seq_len, bsz)
    f = torch.randn(seq_len, bsz)

    # ps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25, 0.30, 0.4, 0.5, 0.7, 0.9, 1.0]

    # for p in ps:
    #     print(gl_profile(True, p, u, f))

    ms = [0, 1, 3, 7, 15, 31]
    ks = [1, 2, 3]

    for m in ms:
        for k in ks:
            print(gmg_profile(True, m, k, u, f))