from optuna import create_study
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--study',type=str,default='optuna/optuna.db')
    args = parser.parse_args()

    storage = f'sqlite:///{args.study}'
    study   = create_study(
        study_name     = 'optuna',
        storage        = storage,
        load_if_exists = True)
    
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    print(study.best_params)
    print(study.best_value)