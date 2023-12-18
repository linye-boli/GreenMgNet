import os
import scipy
import torch 
import numpy as np
from einops import rearrange, repeat

def load_dataset_1d(task_nm, data_root, ntrain=1000, ntest=200, bsz=64, normalize=True):

    data_path = os.path.join(data_root, task_nm+'.mat')
    raw_data = scipy.io.loadmat(data_path)

    F = raw_data['F']
    U = raw_data['U'] - raw_data['U_hom']
    X = raw_data['X']
        
    us = rearrange(F, 'n b -> b 1 n')
    ws = rearrange(U, 'n b-> b 1 n')
    xs = repeat(X, 'n c -> b c n', b=ntrain+ntest)

    if normalize:
        u_mean, u_std = us.mean(), us.std()
        us = (us - u_mean) / u_std
        w_mean, w_std = ws.mean(), ws.std()
        ws = (ws - w_mean) / w_std    

    us = torch.tensor(us).float()
    ws = torch.tensor(ws).float()
    xs = torch.tensor(xs).float()

    us_train = us[:ntrain]
    us_test = us[-ntest:]
    ws_train = ws[:ntrain]
    ws_test = ws[-ntest:]
    xs_train = xs[:ntrain]
    xs_test = xs[-ntest:]

    # us_train = torch.cat([us_train, xs_train], axis=1)
    # us_test = torch.cat([us_test, xs_test], axis=1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_train, xs_train, ws_train), batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_test, xs_test, ws_test), batch_size=bsz, shuffle=False)

    return train_loader, test_loader