from data import load_dataset_1d, load_dataset_2d, load_dataset_2dt
from easydict import EasyDict
import torch
from utils import LpLoss
import matplotlib.pyplot as plt 
import numpy as np

def load_model(log_root, model_nm, dataset_nm, cl, ml, sub=2):
    if '1d' in model_nm:
        res = 8192 // sub
    
    if '2d' in model_nm:
        res = int(((421-1)/sub)+1)   

    if model_nm == 'lno1d':
        model_prefix = '-'.join([model_nm, 'r4', 'w64'])
    elif model_nm == 'fno1d':
        model_prefix = '-'.join([model_nm, 'h4', 'w64'])
    
    if model_nm == 'lno2d':
        model_prefix = '-'.join([model_nm, 'r4', 'w64'])
    elif model_nm in ['gt2d', 'ft2d']:
        model_prefix = '-'.join([model_nm, 'h4', 'w64'])
    elif model_nm == 'fno2d':
        model_prefix = '-'.join([model_nm, 'm12', 'w32'])
    
    weight_nm = '-'.join([model_prefix, str(res), str(res), cl, ml, 'seed100']) + '.pth'
    weight_path = '/'.join([log_root, model_nm, dataset_nm, weight_nm])
    print('load model : ', weight_nm)
    model = torch.load(weight_path)
    return model

def load_dataset(dataset_path, dataset_nm, ntrain, ntest, sub):

    data_cfg = EasyDict({
        'dataset_nm': dataset_nm,
        'dataset_path': dataset_path,
        'trasub': sub,
        'ntrain': ntrain,
        'ntest': ntest,
        'batch_size': 1})

    if dataset_nm in ['burgers', 'lnabs', 'cosine']:
        _, test_loader, u_normalizer = load_dataset_1d(data_cfg)
    elif dataset_nm in ['darcy', 'invdist']:
        _, test_loader, u_normalizer = load_dataset_2d(data_cfg)
    elif dataset_nm in ['NS_V1e-3', 'NS_V1e-4', 'NS_V1e-5']:
        _, test_loader, u_normalizer = load_dataset_2dt(data_cfg)
    
    return test_loader.dataset, u_normalizer

def inference(idx, model, dataset, normalizer, device, is1d=True):

    model = model.to(device)
    rl2 = LpLoss(size_average=False)
    normalizer.to(device)
    model.eval()
    with torch.no_grad():
        a, x, u = dataset[idx]
        a, x, u = a[None], x[None], u[None]

        if is1d:
            bsz, seq_len = a.shape
            a, x, u = a.to(device), x.to(device), u.to(device)
            u_ = model(a=a, x=x).reshape(bsz, seq_len)
            u_ = normalizer.decode(u_)
        else:
            bsz, seq_lx, seq_ly, _ = a.shape
            a, x, u = a.to(device), x.to(device), u.to(device)

            u_ = model(a=a, x=x).reshape(bsz, seq_lx, seq_ly)
            u_ = normalizer.decode(u_)
        
        test_l2 = rl2(u_.reshape(bsz, -1), u.reshape(bsz, -1)).item()

        a = a[0].cpu().numpy()
        x = x[0].cpu().numpy()
        u = u[0].cpu().numpy()
        u_ = u_[0].detach().cpu().numpy()
    
    return a, x, u, u_, test_l2

def vis1d_sample_result(
    idx, test_dataset, normalizer, log_root, model_nm, dataset_nm, cls, mls, device, sub=2):
    nc, nm = len(cls), len(mls)
    fig, axs = plt.subplots(nc,nm, figsize=(4*nm, 4*nc), sharey='all', sharex='all')

    for i, c in enumerate(cls):
        for j, m in enumerate(mls):
            model = load_model(log_root, model_nm, dataset_nm, 'cl{:}'.format(c), 'ml{:}'.format(m), sub=8)
            a, x, u, u_, test_l2 = inference(idx, model, test_dataset, normalizer, device)
            axs[i][j].plot(x, u, c='steelblue', linestyle='solid', label='reference')
            axs[i][j].plot(x, u_, c='firebrick', linestyle='dashed', label='cl {:} - ml {:}'.format(c, m))
            axs[i][j].legend(loc='upper left')
            axs[i][j].set_title("relative L2 error : {:.4f}".format(test_l2))

    fig.suptitle('{:}-{:}-{:}'.format(model_nm, dataset_nm, 8192//sub))
    fig.tight_layout()
    return fig 

def vis2d_sample_result(
    idx, test_dataset, normalizer, log_root, model_nm, dataset_nm, cls, mls, device, sub=3):

    nc, nm = len(cls)*2, len(mls)+2

    u_preds = []
    error_maps = []
    test_l2s = []
    for i, c in enumerate(cls):
        uc_preds = []
        ec_maps = []
        testc_l2s = []

        for j, m in enumerate(mls):
            model = load_model(log_root, model_nm, dataset_nm, 'cl{:}'.format(c), 'ml{:}'.format(m), sub=3)
            a, x, u, u_, test_l2 = inference(idx, model, test_dataset, normalizer, device, is1d=False)
            ec_maps.append(np.abs(u - u_))
            uc_preds.append(u_)
            testc_l2s.append(test_l2)
        
        u_preds.append(uc_preds)
        error_maps.append(ec_maps)
        test_l2s.append(testc_l2s)

    u_preds = np.array(u_preds)
    error_maps = np.array(error_maps)
    test_l2s = np.array(test_l2s)


    fig, axs = plt.subplots(nc,nm, figsize=(4*nm, 4*nc), sharey='all', sharex='all')
    evmin = error_maps.min()
    evmax = np.percentile(error_maps, 98)
    for i, c in enumerate(cls):
        for j, m in enumerate(mls):
            if (i == 0) & (j == 0):
                axs[0][0].pcolormesh(x[:,:,0], x[:,:,1], a[:,:,0], cmap='viridis')
                axs[0][0].axis('off')
                axs[0][0].set_title('inputs')
                axs[0][0].set_box_aspect(1)
                pcm_sol = axs[0][1].pcolormesh(
                    x[:,:,0], x[:,:,1], u, cmap='viridis', vmin=u.min(), vmax=u.max())
                axs[0][1].axis('off')
                axs[0][1].set_title('reference')
                axs[0][1].set_box_aspect(1)

            axs[2*i][j].axis('off')
            axs[2*i][j+1].axis('off')
            axs[2*i+1][j].axis('off')
            axs[2*i+1][j+1].axis('off')

            axs[2*i][j+2].pcolormesh(
                x[:,:,0], x[:,:,1], u_preds[i,j], cmap='viridis', vmin=u.min(), vmax=u.max())
            axs[2*i][j+2].axis('off')
            axs[2*i][j+2].set_title('cl {:} - ml {:}'.format(c,m))
            axs[2*i][j+2].set_box_aspect(1)

            pcm_error = axs[2*i+1][j+2].pcolormesh(
                x[:,:,0], x[:,:,1], error_maps[i,j], cmap='jet', vmin=evmin, vmax=evmax)
            axs[2*i+1][j+2].set_title('relative L2 error : {:.4f}'.format(test_l2s[i,j]))
            axs[2*i+1][j+2].axis('off')        
            axs[2*i+1][j+2].set_box_aspect(1)

    for i in range(nc):
        if i % 2 == 0:
            fig.colorbar(pcm_sol, ax=axs[i,:], location='right')
        else:
            fig.colorbar(pcm_error, ax=axs[i,:], location='right')
    
    fig.suptitle('{:}-{:}-{:}'.format(model_nm, dataset_nm, int(((421-1)/sub)+1)))

    return fig