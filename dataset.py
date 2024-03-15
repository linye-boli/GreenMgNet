import os
import scipy
import torch 
import numpy as np
from einops import rearrange, repeat
import numpy as np
import h5py

def load_dataset_1d(task_nm, data_root, ntrain=1000, ntest=200, bsz=64, normalize=False, odd=True):

    data_path = os.path.join(data_root, task_nm+'.mat')
    try:
        raw_data = scipy.io.loadmat(data_path)
        F = raw_data['F']
        U = raw_data['U']
        U_hom = raw_data['U_hom']
        U = U-U_hom
    except:
        raw_data = h5py.File(data_path)

        F = raw_data['F'][()]
        F = np.transpose(F, axes=range(len(F.shape) - 1, -1, -1))

        U = raw_data['U'][()]
        U = np.transpose(U, axes=range(len(U.shape) - 1, -1, -1))
        
        U_hom = raw_data['U_hom'][()]
        U_hom = np.transpose(U_hom, axes=range(len(U_hom.shape) - 1, -1, -1))

        U = U - U_hom
    # F = raw_data['F']
    # U = raw_data['U'] - raw_data['U_hom']
        
    us = rearrange(F, 'n b -> b 1 n')
    ws = rearrange(U, 'n b-> b 1 n')

    if U_hom.sum() == 0:
        s = ws.max()
        ws = ws/s
        us = us/s

    if normalize:
        u_mean, u_std = us.mean(), us.std()
        us = (us - u_mean) / u_std
        w_mean, w_std = ws.mean(), ws.std()
        ws = (ws - w_mean) / w_std

    us = torch.tensor(us).float()
    ws = torch.tensor(ws).float()

    if not odd:
        us = us[:,:,:-1]
        ws = ws[:,:,:-1]

    us_train = us[:ntrain]
    us_test = us[-ntest:]
    ws_train = ws[:ntrain]
    ws_test = ws[-ntest:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_train, ws_train), batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_test, ws_test), batch_size=bsz, shuffle=False)

    return train_loader, test_loader

def load_dataset_2d(task_nm, data_root, ntrain=1000, ntest=200, bsz=64, normalize=False, odd=True):

    data_path = os.path.join(data_root, task_nm+'.mat')
    raw_data = scipy.io.loadmat(data_path)

    F = raw_data['F']
    U = raw_data['U']
        
    us = rearrange(F, 'b m n -> b 1 m n')
    ws = rearrange(U, 'b m n-> b 1 m n')
        
    if normalize:
        u_mean, u_std = us.mean(), us.std()
        us = (us - u_mean) / (u_std + 1e-5)
        w_mean, w_std = ws.mean(), ws.std()
        ws = (ws - w_mean) / (w_std + 1e-5)
    else:
        s = ws.max()
        ws = ws/s
        us = us/s

    us = torch.tensor(us).float()
    ws = torch.tensor(ws).float()

    if not odd:
        us = us[:,:,:-1,:-1]
        ws = ws[:,:,:-1,:-1]

    us_train = us[:ntrain]
    us_test = us[-ntest:]
    ws_train = ws[:ntrain]
    ws_test = ws[-ntest:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_train, ws_train), batch_size=bsz, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_test, ws_test), batch_size=bsz, shuffle=False, drop_last=True)

    return train_loader, test_loader