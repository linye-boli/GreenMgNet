import os
import glob 
import pandas as pd

def create_expdf(data_root):
    exp_hists = sorted(glob.glob(data_root + '/*/*/hist.csv'))
    exp_lst = []
    for exp_hist in exp_hists:
        _, _, task_nm, exp_nm, _ = exp_hist.split('/')
        settings = exp_nm.split('-')
        if len(settings) == 7:
            model_nm, act, res, h, p, seed, aug = settings
            k, m = 0, 0
        elif len(settings) == 9:
            model_nm, act, res, h, k, m, p, aug, seed = settings
        else:
            print(exp_hist)

        rl2 = pd.read_csv(exp_hist).test_rl2.iloc[-1]
        exp_lst.append([task_nm, model_nm, act, res, int(h), int(k), int(m), float(p), seed, aug, float(rl2)])
    exp_df = pd.DataFrame(
        exp_lst, columns=['task_nm', 'model_nm', 'act', 'res', 'h', 'k', 'm', 'p', 'aug', 'seed', 'rl2'])
    return exp_df.sort_values(['res', 'h', 'k', 'm', 'p', 'aug'])

def fetch_subdf(
          exp_df, task_nm, 
          model_nm=None, act='relu', res=None, 
          h=50, k=None, m=None, p=None, 
          seed=None):
    
    cond = exp_df.task_nm == task_nm
    if model_nm is not None:
        cond = cond & (exp_df.model_nm == model_nm)
    
    if act is not None:
        cond = cond & (exp_df.act == act)
    
    if res is not None:
        cond = cond & (exp_df.res == res)
    
    if h is not None:
        cond = cond & (exp_df.h == h)
    
    if k is not None:
        cond = cond & (exp_df.k == k)
    
    if m is not None:
        cond = cond & (exp_df.m == m)
    
    if p is not None:
        cond = cond & (exp_df.p == p)
    
    if p is not None:
        cond = cond & (exp_df.seed == seed)
    
    return exp_df[cond]