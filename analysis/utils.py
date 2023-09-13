import pandas as pd 
from tqdm import tqdm 

def load_accuracy_log(log_paths):
    log_df = pd.DataFrame(
        columns=(
        'model', 'dataset', 'coarse_level', 
        'resolution', 'residual', 'seed', 'test_l2'))

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
        elif mlevel[2:] == '0':
            mlevel = 'diag'

        # mlevel = int(mlevel[2:]) if mlevel[2:] != 'x' else -1
        best_l2 = log.test_l2.min()

        log_df.loc[i] = [model_nm, dataset, clevel, trares, mlevel, seed, best_l2]

    return log_df 