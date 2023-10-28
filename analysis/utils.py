import numpy as np
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

def load_accuracy_log(log_paths):
    log_df = pd.DataFrame(
        columns=(
        'model', 'dataset', 'coarse_level', 
        'resolution', 'residual', 'seed', 'test_l2', 'mcode'))

    for i, log_path in tqdm(enumerate(log_paths), total=len(log_paths)):
        dataset, log_info = log_path.split('/')[-2:]
        model_nm, _, _, trares, testres, clevel, mlevel, seed = log_info.split('.')[0].split('-')

        clevel = int(clevel[2:])
        trares = int(trares)
        testres = int(testres)
        seed = int(seed[4:])

        log = pd.read_csv(log_path)

        if mlevel[2:] == 'x':
            mlevel = 'null'
            mcode = -1
        elif mlevel[2:] == '0':
            mlevel = 'diag'
            mcode = 0
        else:
            mcode = int(mlevel[2:])        

        # mlevel = int(mlevel[2:]) if mlevel[2:] != 'x' else -1
        best_l2 = log.test_l2.min()
        # if best_l2 < 0.1:
        log_df.loc[i] = [model_nm, dataset, clevel, trares, mlevel, seed, best_l2, mcode]

    return log_df 

def vis1d_single_model_dataset_result(df, model, dataset, ml='ml3', clevel=0):
    sub_df = df[(df.dataset == dataset) & (df.model == model) & (df.coarse_level == clevel) & ((df.residual == 'null') | (df.residual == 'diag') | (df.residual == ml))]
    l2_min = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.min).reset_index()
    l2_max = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.max).reset_index()
    l2_mean = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.mean).reset_index()

    resolution = [int(x) for x in np.sort(sub_df.resolution.unique()).tolist()]

    diag_min = l2_min[l2_min.residual == 'diag'].test_l2.tolist()
    diag_max = l2_max[l2_max.residual == 'diag'].test_l2.tolist()
    diag_mean = l2_mean[l2_mean.residual == 'diag'].test_l2.tolist()

    null_min = l2_min[l2_min.residual == 'null'].test_l2.tolist()
    null_max = l2_max[l2_max.residual == 'null'].test_l2.tolist()
    null_mean = l2_mean[l2_mean.residual == 'null'].test_l2.tolist()

    ml_min = l2_min[l2_min.residual == ml].test_l2.tolist()
    ml_max = l2_max[l2_max.residual == ml].test_l2.tolist()
    ml_mean = l2_mean[l2_mean.residual == ml].test_l2.tolist()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5), sharey=True)
    st = fig.suptitle("{:} - {:}".format(model, dataset), fontsize="x-large")
    ax1.plot(resolution, null_min, '-.r',label='min')
    ax1.plot(resolution, null_max, ':r', label='max')
    ax1.plot(resolution, null_mean, '.-r', label='mean')
    ax1.set_xticks(resolution)
    ax1.set_xticklabels(resolution, rotation=90)
    ax1.set_yscale('log')
    ax1.grid(axis='both', which='both')
    ax1.legend(loc='upper left')
    ax1.set_title('w/o residual')

    ax2.plot(resolution, diag_min, '-.b', label='min')
    ax2.plot(resolution, diag_max, ':b', label='max')
    ax2.plot(resolution, diag_mean, '.-b', label='mean')
    ax2.set_xticks(resolution)
    ax2.set_xticklabels(resolution, rotation=90)
    ax2.set_yscale('log')
    ax2.grid(axis='both', which='both')
    ax2.legend(loc='upper left')
    ax2.set_title('linear residual')

    ax3.plot(resolution, ml_min, '-.c', label='min')
    ax3.plot(resolution, ml_max, ':c', label='max')
    ax3.plot(resolution, ml_mean, '.-c', label='mean')
    ax3.set_xticks(resolution)
    ax3.set_xticklabels(resolution, rotation=90)
    ax3.set_yscale('log')
    ax3.grid(axis='both', which='both')
    ax3.legend(loc='upper left')
    ax3.set_title('{:} residual'.format(ml))

    # fig.tight_layout()

    return fig 

def vis1d_all_model_dataset_result(df, ml='ml3', clevel=0):
    fig, axs = plt.subplots(4,4, figsize=(20, 20), sharey='row')
    for d, dataset in enumerate(['lnabs', 'poisson', 'burgers', 'cosine']):
        sub_df = df[(df.dataset == dataset) & (df.coarse_level == clevel)]
        l2_min = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.min).reset_index()
        l2_max = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.max).reset_index()
        l2_mean = sub_df.groupby(["model", "dataset", "coarse_level", "resolution", "residual"])['test_l2'].apply(np.mean).reset_index()

        for m, model in enumerate(['fno1d', 'lno1d', 'ft1d', 'gt1d']):
            resolution = [str(x) for x in np.sort(sub_df[(sub_df.model == model)].resolution.unique()).tolist()]

            diag_max = l2_max[(l2_max.residual == 'diag') & (l2_max.model == model)].test_l2
            ml_max = l2_max[(l2_max.residual == ml) & (l2_max.model == model)].test_l2

            diag_mean = l2_mean[(l2_mean.residual == 'diag') & (l2_mean.model == model)].test_l2
            ml_mean = l2_mean[(l2_mean.residual == ml) & (l2_mean.model == model)].test_l2

            diag_min = l2_min[(l2_min.residual == 'diag') & (l2_min.model == model)].test_l2
            ml_min = l2_min[(l2_min.residual == ml) & (l2_min.model == model)].test_l2

            axs[m][d].plot(resolution, diag_mean, 'r', label=model)
            axs[m][d].fill_between(resolution, diag_min, diag_max, color='r', alpha=0.1)

            axs[m][d].plot(resolution, ml_mean, '-.b', label="{:}-".format(ml)+model)
            axs[m][d].fill_between(resolution, ml_min, ml_max, color='b', alpha=0.1)            

            axs[m][d].set_xticks(resolution)
            axs[m][d].set_xticklabels(resolution)
            axs[m][d].set_yscale('log')
            axs[m][d].legend(loc='upper left')
            axs[m][d].grid(which='minor')
            axs[m][d].grid(which='major')
            if model == 'fno1d':
                axs[m][d].set_title('{:}'.format(dataset))
    
    fig.supxlabel('Resolution')
    fig.supylabel('Relative error')
    fig.tight_layout()
    
    return fig 

def vis_all_model_dataset_residual_trend_on_fix_resolution(df, resolution=4096, colors=['r','b','c','m']):
    sub_df = df[df.resolution == resolution]
    table_mean = sub_df.pivot_table(values='test_l2', index=['dataset', 'model'], columns=['residual'], aggfunc=np.mean)
    table_min = sub_df.pivot_table(values='test_l2', index=['dataset', 'model'], columns=['residual'], aggfunc=np.min)
    table_max = sub_df.pivot_table(values='test_l2', index=['dataset', 'model'], columns=['residual'], aggfunc=np.max)

    fig, axs = plt.subplots(3, 4, figsize=(15, 5))
    residuals = ['null', 'diag', 'ml1', 'ml2', 'ml3', 'ml4']
    x = np.arange(len(residuals))
    for d, dataset in enumerate(['lnabs', 'burgers', 'cosine']):
        for m, model in enumerate(['fno1d', 'lno1d', 'ft1d', 'gt1d']):
            
            l2max = []
            l2min = []
            l2mean = []

            for i, residual in enumerate(residuals):
                l2mean.append(table_mean.loc[(dataset, model), residual])
                l2min.append(table_min.loc[(dataset, model), residual])
                l2max.append(table_max.loc[(dataset, model), residual])

            axs[d][m].plot(x, l2mean, "-", color=colors[m])
            axs[d][m].plot(x, l2min, ":", color=colors[m])
            axs[d][m].plot(x, l2max, ":", color=colors[m])
            axs[d][m].fill_between(x, l2min, l2max, color=colors[m], alpha=0.1)
            
            axs[d][m].set_yscale('log')
            axs[d][m].grid(axis='both', which='both')
            axs[d][m].set_title("{:}-{:}".format(model, dataset))
            axs[d][m].set_xticks(x)
            axs[d][m].set_xticklabels(residuals)
    fig.tight_layout()

    return fig 

def pass_check(model_nm, res, clevel, mlevel, out_nm):
    if model_nm == 'ft2d':
        if res == 85:
            return False 
        elif res == 141:
            if clevel == 0:
                print('{:} : out of A100 mem'.format(out_nm))
                return True
            else:
                return False
        elif res == 211:
            if clevel == 0:
                print('{:} : out of A100 mem'.format(out_nm))
                return True
            elif clevel == 1:
                print('{:} : too long for training'.format(out_nm)) # 13 hours on A100
                return True
            else:
                return False
        else:
            if clevel == 0:
                print('{:} : out of A100 mem'.format(out_nm))
                return True
            elif clevel == 1:
                print('{:} : too long for training'.format(out_nm))
                return True
            else:
                
                return False
    elif model_nm == 'gt2d':
        if res in [85, 141]:
            return False
        elif res == 211:
            if clevel == 0:
                if mlevel in ['x', 0, 1]:
                    return False 
                else:
                    print('{:} : too long for training'.format(out_nm))
                    return True
            else:
                return False
        else:
            return False
    elif model_nm == 'lno2d':
        if res in [85, 141]:
            return False 
        elif res == 211:
            return False
        else:
            return True
