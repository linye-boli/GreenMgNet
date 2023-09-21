import pandas as pd 
from tqdm import tqdm 

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
        if best_l2 < 0.1:
            log_df.loc[i] = [model_nm, dataset, clevel, trares, mlevel, seed, best_l2, mcode]

    return log_df 