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
    elif cfg.dataset_nm == 'invdist':
        tra_file_nm = 'invdist_r421_train.mat'
        test_file_nm = 'invdist_r421_test.mat'


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

def load_dataset_2dt(cfg):
    ntrain = cfg.ntrain
    ntest = cfg.ntest
    batch_size = cfg.batch_size

    sub = cfg.trasub 
    s = 64 // sub
    T_in = 10

    if "NS" in cfg.dataset_nm:
        if cfg.dataset_nm == 'NS_V1e-3':
            tra_file_nm = 'NavierStokes_V1e-3_N1000_T50.mat'
            test_file_nm = 'NavierStokes_V1e-3_N200_T50.mat'
            T = 40
        elif cfg.dataset_nm == 'NS_V1e-4':
            tra_file_nm = 'NavierStokes_V1e-4_N1000_T30.mat'
            test_file_nm = 'NavierStokes_V1e-4_N200_T30.mat'
            T = 20
        elif cfg.dataset_nm == 'NS_V1e-5':
            tra_file_nm = 'NavierStokes_V1e-5_N1000_T20.mat'
            test_file_nm = 'NavierStokes_V1e-5_N200_T20.mat'
            T = 10
        
        tra_dataset_path = os.path.join(cfg.dataset_path, 'data2d', cfg.dataset_nm, tra_file_nm)
        tra_reader = MatReader(tra_dataset_path)
        test_dataset_path = os.path.join(cfg.dataset_path, 'data2d', cfg.dataset_nm, test_file_nm)
        test_reader = MatReader(test_dataset_path)

        u_train = tra_reader.read_field('u')
        train_a = u_train[:ntrain,::sub,::sub,:T_in] # 1000x64x64x10
        train_u = u_train[:ntrain,::sub,::sub,T_in:T+T_in] # 1000x64x64x30

        u_test = test_reader.read_field('u')
        test_a = u_test[-ntest:,::sub,::sub,:T_in]
        test_u = u_test[-ntest:,::sub,::sub,T_in:T+T_in]
    elif 'ns' in cfg.dataset_nm:

        if cfg.dataset_nm == 'ns_V1e-3':
            file_nm = 'ns_V1e-3_N5000_T50.mat'
            T = 40
        elif cfg.dataset_nm == 'ns_V1e-4':
            file_nm = 'ns_V1e-4_N10000_T30.mat'
            T = 20
        
        dataset_path = os.path.join(cfg.dataset_path, 'data2d', cfg.dataset_nm, file_nm)
        data_reader = MatReader(dataset_path)

        u = data_reader.read_field('u')
        train_a = u[:ntrain,::sub,::sub,:T_in] # 1000x64x64x10
        train_u = u[:ntrain,::sub,::sub,T_in:T+T_in] # 1000x64x64x30
        test_a = u[-ntest:,::sub,::sub,:T_in]
        test_u = u[-ntest:,::sub,::sub,T_in:T+T_in]

    assert (s == train_u.shape[-2])
    assert (T == train_u.shape[-1])

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain,s,s,1,T_in).repeat([1,1,1,T,1]) # 1000x64x64x30x10
    test_a = test_a.reshape(ntest,s,s,1,T_in).repeat([1,1,1,T,1]) # 200x64x64x30x10

    batchsize, size_x, size_y, size_z = 1, s, s, T
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    grid = torch.cat((gridx, gridy, gridz), dim=-1)
    grid_train = grid.repeat(ntrain,1,1,1,1).float()
    grid_test = grid.repeat(ntest,1,1,1,1).float()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, grid_train, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, grid_test, test_u), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_normalizer   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="debug dataset for transformer neural operator")
    cfg = get_arguments(parser)
    cfg.dataset_nm = 'NS_V1e-5'
    cfg.dataset_path = '/workdir/pde_data/'
    cfg.trasub = 1

    train_loader, test_loader, y_normalizer = load_dataset_2dt(cfg)
    x_train, grid_train, y_train = next(iter(train_loader))

    print(x_train.shape)
    print(grid_train.shape)
    print(y_train.shape)
    