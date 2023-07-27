import os
import argparse
import torch
import numpy as np
from utils import MatReader, UnitGaussianNormalizer, get_arguments

def load_dataset_1d(cfg):
    ntrain = cfg.ntrain
    ntest = cfg.ntest
    batch_size = cfg.batch_size

    #full resolution is 8192(2**13), input resolution 2**(13-2**)
    sub = cfg.trasub 
    inp_seq_len = 2**(13 - int(np.log2(sub)))

    ################################################################
    # reading data and normalization
    ################################################################
    if cfg.dataset_nm == 'burgers':
        file_nm = 'burgers_data_R10.mat'
    elif cfg.dataset_nm == 'cosine':
        file_nm = 'cosine_data.mat'
    elif cfg.dataset_nm == 'lnabs':
        file_nm = 'lnabs_data.mat'
    elif cfg.dataset_nm == 'poisson':
        file_nm = 'poisson_data.mat'

    dataset_path = os.path.join(cfg.dataset_path, 'data1d', cfg.dataset_nm, file_nm)
    dataloader = MatReader(dataset_path)
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grid = np.linspace(0, 1, inp_seq_len).reshape(1, inp_seq_len)
    grid = torch.tensor(grid, dtype=torch.float)
    grid_train = grid.repeat(ntrain,1)
    grid_test = grid.repeat(ntest,1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, grid_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, grid_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, y_normalizer


def load_dataset_2d(cfg):
    ntrain = cfg.ntrain
    ntest = cfg.ntest
    batch_size = cfg.batch_size

    sub = cfg.trasub 
    s = int(((421-1)/sub)+1)

    ################################################################
    # reading data and normalization
    ################################################################
    if cfg.dataset_nm == 'darcy':
        tra_file_nm = 'piececonst_r421_N1024_smooth1.mat'
        test_file_nm = 'piececonst_r421_N1024_smooth2.mat'
    elif cfg.dataset_nm == 'cosine':
        file_nm = 'cosine_data.mat'
    elif cfg.dataset_nm == 'lnabs':
        file_nm = 'lnabs_data.mat'
    elif cfg.dataset_nm == 'poisson':
        file_nm = 'poisson_data.mat'

    tra_dataset_path = os.path.join(cfg.dataset_path, 'data2d', cfg.dataset_nm, tra_file_nm)
    tra_dataloader = MatReader(tra_dataset_path)
    test_dataset_path = os.path.join(cfg.dataset_path, 'data2d', cfg.dataset_nm, test_file_nm)
    test_dataloader = MatReader(test_dataset_path)

    x_train = tra_dataloader.read_field('coeff')[:ntrain,::sub,::sub][:,:s,:s]
    y_train = tra_dataloader.read_field('sol')[:ntrain,::sub,::sub][:,:s,:s]
    x_test = test_dataloader.read_field('coeff')[:ntest,::sub,::sub][:,:s,:s]
    y_test = test_dataloader.read_field('sol')[:ntest,::sub,::sub][:,:s,:s]
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    grid_train = grid.repeat(ntrain,1,1,1)
    grid_test = grid.repeat(ntest,1,1,1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, grid_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, grid_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, y_normalizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="debug dataset for transformer neural operator")
    cfg = get_arguments(parser)
    cfg.dataset_nm = 'darcy'
    cfg.trasub = 5

    train_loader, test_loader, y_normalizer = load_dataset_2d(cfg)
    x_train, grid_train, y_train = next(iter(train_loader))

    print(x_train.shape)
    print(grid_train.shape)
    print(y_train.shape)
    