import os
import scipy
import torch 
import numpy as np
from einops import rearrange, repeat
import numpy as np
import h5py


def load_mat1d(data_path, key, N, train=True):
    try:
        raw_data = scipy.io.loadmat(data_path)
        X = raw_data[key]
    except:
        raw_data = h5py.File(data_path)
        X = raw_data[key][()]
        X = np.transpose(X, axes=range(len(X.shape) - 1, -1, -1))

    if train:
        return X[:,:N]
    else:
        return X[:,-N:]
    
def load_mat2d(data_path, key, N, train=True):
    try:
        raw_data = scipy.io.loadmat(data_path)
        X = raw_data[key]
    except:
        raw_data = h5py.File(data_path)
        X = raw_data[key][()]
        X = np.transpose(X, axes=range(len(X.shape) - 1, -1, -1))

    if train:
        return X[:N,:,:]
    else:
        return X[-N:,:,:]
    
def load_dataset_1d(
        task_nm, data_root,
        train_post='3.00e-01.mat', test_post='3.00e-01.mat', res='4097',
        ntrain=100, ntest=100, bsz=64, normalize=False, odd=True):
    
    # F, U = next(iter(train_loader))
    # F : bsz x 1 x 16385
    # U : bsz x 1 x 16385

    F_train_path = os.path.join(data_root, '_'.join(['f1d', res, train_post]))
    U_train_path = os.path.join(data_root, '_'.join([task_nm, res, train_post]))

    F_test_path = os.path.join(data_root, '_'.join(['f1d', res, test_post]))
    U_test_path = os.path.join(data_root, '_'.join([task_nm, res, test_post]))
    
    F_train = load_mat1d(F_train_path, 'F', ntrain, True)
    U_train = load_mat1d(U_train_path, 'U', ntrain, True)

    F_test = load_mat1d(F_test_path, 'F', ntest, False)
    U_test = load_mat1d(U_test_path, 'U', ntest, False)
    
    F_train = rearrange(F_train, 'n b -> b 1 n')
    U_train = rearrange(U_train, 'n b-> b 1 n')
    F_test = rearrange(F_test, 'n b -> b 1 n')
    U_test = rearrange(U_test, 'n b-> b 1 n')

    if normalize:
        U_mean, U_std = U_train.mean(), U_train.std()
        U_train = (U_train - U_mean) / U_std
        U_test = (U_test - U_mean) / U_std

        F_mean, F_std = F_train.mean(), F_train.std()
        F_train = (F_train - F_mean) / F_std
        F_test = (F_test - F_mean) / F_std

    U_train = torch.tensor(U_train).float()
    F_train = torch.tensor(F_train).float()
    U_test = torch.tensor(U_test).float()
    F_test = torch.tensor(F_test).float()

    if not odd:
        U_train = U_train[:,:,:-1]
        F_train = F_train[:,:,:-1]
        U_test = U_test[:,:,:-1]
        F_test = F_test[:,:,:-1]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_train, U_train), batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_test, U_test), batch_size=bsz, shuffle=False)

    return train_loader, test_loader


def load_dataset_2d(
        task_nm, data_root, r,
        train_post='2.00e-01.mat', test_post='2.00e-01.mat',
        ntrain=100, ntest=100, bsz=64, normalize=False, odd=True):

    F_train_path = os.path.join(data_root, '_'.join(['fdisk', '65x65', train_post]))
    U_train_path = os.path.join(data_root, '_'.join([task_nm, '65x65', train_post]))

    F_test_path = os.path.join(data_root, '_'.join(['fdisk', '65x65', test_post]))
    U_test_path = os.path.join(data_root, '_'.join([task_nm, '65x65', test_post]))
    
    F_train = load_mat2d(F_train_path, 'F', ntrain, True)[:,::2**r,::2**r]
    U_train = load_mat2d(U_train_path, 'U', ntrain, True)[:,::2**r,::2**r]

    F_train = rearrange(F_train, 'b x y -> b 1 x y')
    U_train = rearrange(U_train, 'b x y -> b 1 x y')

    F_test = load_mat2d(F_test_path, 'F', ntest, False)[:,::2**r,::2**r]
    U_test = load_mat2d(U_test_path, 'U', ntest, False)[:,::2**r,::2**r]

    F_test = rearrange(F_test, 'b x y -> b 1 x y')
    U_test = rearrange(U_test, 'b x y -> b 1 x y')

    if normalize:
        U_mean, U_std = U_train.mean(), U_train.std()
        F_mean, F_std = F_train.mean(), F_train.std()
        U_train = (U_train - U_mean) / U_std
        U_test = (U_test - U_mean) / U_std

        F_train = (F_train - F_mean) / F_std
        F_test = (F_test - F_mean) / F_std

    U_train = torch.tensor(U_train).float()
    F_train = torch.tensor(F_train).float()
    U_test = torch.tensor(U_test).float()
    F_test = torch.tensor(F_test).float()

    if not odd:
        U_train = U_train[:,:,:-1,:-1]
        F_train = F_train[:,:,:-1,:-1]
        U_test = U_test[:,:,:-1,:-1]
        F_test = F_test[:,:,:-1,:-1]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_train, U_train), batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_test, U_test), batch_size=bsz, shuffle=False)

    return train_loader, test_loader