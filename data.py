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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="debug dataset for transformer neural operator")
    parser = get_arguments(parser)
    cfg = parser.parse_args()
    train_loader, test_loader, y_normalizer = load_dataset_1d(cfg)
    x_train, grid_train, y_train = next(iter(train_loader))