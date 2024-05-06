import sys 
sys.path.append('./')
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as ticker
from dataset.generate_dataset_1d import poisson_kernel, logarithm_kernel
import pandas as pd
import numpy as np
import glob
import scienceplots
import matplotlib as mpl
from easydict import EasyDict as edict

mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 300
plt.style.use('science')
pd.set_option('display.float_format', lambda x: '%.4e' % x)


def err_map(G, G_pred):
   return np.abs(G_pred - G)/np.linalg.norm(G, ord = 2)

def plot_kernel1d(cfg, nh=513):
    pred_kernel = np.load(cfg.kernel_file_path)
    
    # process green's function 
    h = (cfg.xmax-cfg.xmin)/(nh-1)
    xh = torch.linspace(cfg.xmin,cfg.xmax,nh)
    x_i = torch.cartesian_prod(xh, xh)
    if cfg.task_nm == 'Poisson1D':
        kernel_func = poisson_kernel
    elif cfg.task_nm == 'Logarithm1D':
        kernel_func = logarithm_kernel
    G = kernel_func(x_i[:,0], x_i[:,1], h).reshape(nh, nh)
    G_pred = pred_kernel.reshape(nh, nh)/h 
    G_err = err_map(G.numpy(), G_pred)
    print(G_err.max(), G_err.min())
    x_i = x_i.reshape(nh,nh,2)
    xs = x_i[:,:,0]
    ys = x_i[:,:,1]

    plt.figure(figsize = (5,5))
    plt.pcolor(xs, ys, G, vmin=cfg.Gmin, vmax=cfg.Gmax, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title("Exact Green's function")
    plt.gca().set_aspect('equal')
    # plt.tight_layout()

    outnm = './vis/{:}_Exact.jpg'.format(cfg.task_nm)
    plt.savefig(outnm)
    print("save fig : {:}".format(outnm))

    plt.figure(figsize = (5,5))
    plt.pcolor(xs, ys, G_pred, vmin=cfg.Gmin, vmax=cfg.Gmax, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(cfg.method_nm)
    plt.gca().set_aspect('equal')
    # plt.tight_layout()

    outnm = './vis/{:}_{:}.jpg'.format(cfg.task_nm, cfg.method_nm)
    plt.savefig(outnm)
    print("save fig : {:}".format(outnm))

    plt.figure(figsize = (5,5))
    plt.pcolor(xs, ys, G_err, vmin=cfg.emin, vmax=cfg.emax, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Rel Err: ' + cfg.method_nm)
    plt.gca().set_aspect('equal')
    # plt.tight_layout()
    plt.colorbar(format='%.2e')

    outnm = './vis/{:}_{:}_Err.jpg'.format(cfg.task_nm, cfg.method_nm)
    plt.savefig(outnm)
    print("save fig : {:}".format(outnm))


def plot_kernel1d_slice(cfg, nh=513, idx=257):
    pred_kernel_gn = np.load(cfg.kernel_gn_file_path)
    pred_kernel_gnaug = np.load(cfg.kernel_gnaug_file_path)

    # process green's function 
    h = (cfg.xmax-cfg.xmin)/(nh-1)
    x_ = cfg.xmin + (idx-1)*h
    xh = torch.linspace(cfg.xmin,cfg.xmax,nh)
    x_i = torch.cartesian_prod(xh, xh)
    if cfg.task_nm == 'Poisson1D':
        kernel_func = poisson_kernel
    elif cfg.task_nm == 'Logarithm1D':
        kernel_func = logarithm_kernel

    G = kernel_func(x_i[:,0], x_i[:,1], h).reshape(nh, nh)
    G_gn_pred = pred_kernel_gn.reshape(nh, nh)/h 
    G_gnaug_pred = pred_kernel_gnaug.reshape(nh, nh)/h 
    
    # visualize kernel slice
    G_ = G[idx]
    G_gn_ = G_gn_pred[idx]
    G_gnaug_ = G_gnaug_pred[idx]
    
    f = plt.figure(figsize=(5,5))
    ax = f.add_subplot(111)
    ax.plot(xh, G_, '-k',label='Exact')
    ax.plot(xh, G_gnaug_, '-g',label='GL-Aug')
    ax.plot(xh, G_gn_, '-r', label='GL')
    ax.title.set_text('G(x, y={:.2f})'.format(x_))
    axins = ax.inset_axes(
        [cfg.ina, cfg.inb, cfg.inc, cfg.ind], xlim=(cfg.inx_min, cfg.inx_max),ylim=(cfg.iny_min, cfg.iny_max),
        xticklabels=[], yticklabels=[])
    axins.plot(xh, G_, '-k', label='Exact')
    axins.plot(xh, G_gnaug_, '-g', label='GL-aug')
    axins.plot(xh, G_gn_, '-r', label='GL')
    ax.indicate_inset_zoom(axins)
    ax.legend()

    outnm = './vis/{:}_{:.2f}_compare.jpg'.format(cfg.task_nm, x_)
    plt.savefig(outnm)
    print("save fig : {:}".format(outnm))


if __name__ == '__main__':
    # poisson 1D gnaug
    cfg_gnaug_poisson1d = edict({
        "kernel_file_path":'./results/poisson1d/GN1D-rational-513-50-1.0000-aug2-2/approx_kernel.npy',
        "task_nm" : "Poisson1D",
        "method_nm" : "GL-Aug",
        "xmax": 1,
        "xmin": 0,
        "ymax": 1,
        "ymin": 0,
        "Gmin": 0,
        "Gmax": 0.25,
        "emax": 7e-5,
        "emin": 0
    })

    plot_kernel1d(cfg_gnaug_poisson1d)

    # poisson 1D gn
    cfg_gn_poisson1d = cfg_gnaug_poisson1d
    cfg_gn_poisson1d.kernel_file_path = './results/poisson1d/GN1D-rational-513-50-1.0000-none-2/approx_kernel.npy'
    cfg_gn_poisson1d.method_nm = 'GreenLearning'
    plot_kernel1d(cfg_gn_poisson1d)

    # poisson 1D compare 
    cfg_poisson1d_compare = edict({
        "kernel_gn_file_path":'./results/poisson1d/GN1D-rational-513-50-1.0000-none-2/approx_kernel.npy',
        "kernel_gnaug_file_path":"./results/poisson1d/GN1D-rational-513-50-1.0000-aug2-2/approx_kernel.npy",
        "task_nm" : "Poisson1D",
        "xmax": 1,
        "xmin": 0,
        "ina": 0.1,
        "inb": 0.6,
        "inc": 0.3,
        "ind": 0.3,
        "inx_min": 0.45,
        "inx_max": 0.55,
        "iny_min": 0.24,
        "iny_max": 0.26
    })
    plot_kernel1d_slice(cfg_poisson1d_compare)

    # logarithm 1D gnaug
    cfg_gnaug_log1d = edict({
        "kernel_file_path":'./results/logarithm/GN1D-rational-513-50-1.0000-aug2-1/approx_kernel.npy',
        "task_nm" : "Logarithm1D",
        "method_nm" : "GL-Aug",
        "xmax": 1,
        "xmin": -1,
        "ymax": 1,
        "ymin": -1,
        "Gmin": -5,
        "Gmax": 0.,
        "emax": 5.e-3,
        "emin": 0
    })

    plot_kernel1d(cfg_gnaug_log1d)

    # logarithm 1D gn
    cfg_gn_log1d = cfg_gnaug_log1d 
    cfg_gn_log1d.kernel_file_path = './results/logarithm/GN1D-rational-513-50-1.0000-none-2/approx_kernel.npy'
    cfg_gn_log1d.method_nm = 'GreanLearning'
    plot_kernel1d(cfg_gn_log1d)

    # logarithm 1D compare
    cfg_log1d_compare = edict({
        "kernel_gn_file_path":'./results/logarithm/GN1D-rational-513-50-1.0000-none-2/approx_kernel.npy',
        "kernel_gnaug_file_path":"./results/logarithm/GN1D-rational-513-50-1.0000-aug2-2/approx_kernel.npy",
        "task_nm" : "Logarithm1D",
        "xmax": 1,
        "xmin": -1,
        "ina": 0.1,
        "inb": 0.6,
        "inc": 0.1,
        "ind": 0.3,
        "inx_min": -0.2,
        "inx_max": 0.2,
        "iny_min": -7,
        "iny_max": -2
    })
    plot_kernel1d_slice(cfg_log1d_compare)