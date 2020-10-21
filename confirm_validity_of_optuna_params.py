from word_and_categ_dict import WordAndCategDict
import chainer
import pandas as pd
import cupy as np
import argparse
from dialect_and_area_classifier import batch_converter_area
from class_balanced_loss import get_samples_per_cls
from make_confusion_matrix import(
    get_confusion_matrix_DAC,
    save_cmat_fig
)
from classify_dialect import(
    get_model_trainer_reporter,
    get_iter
)
import pathlib
from optuna import create_study
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch',type=int,default=500)
    parser.add_argument('-d','--desc',type=str,default='')
    parser.add_argument('-rd','--result_directory',type=str,default='')
    parser.add_argument('-pts',
        '--param_to_be_scanned',
        type=str,
        choices=['','grad_clipping','learning_rate','beta','n_lstm'],
        default='')
    parser.add_argument('-s','--study',type=str,default='optuna/optuna.db')
    args = parser.parse_args()

    BATCH_SIZE = 60
    DEVICE = 0
    BC = batch_converter_area    

    wd      = WordAndCategDict('spm/dialect_standard.model','corpus/all_pft.txt')
    wd_area = WordAndCategDict(categ_path='corpus/all_area.txt')

    TRAIN_DATASET_PATH = 'corpus/train_ft_area.pkl'
    TEST_DATASET_PATH  = 'corpus/test_ft_area.pkl'
    RESULT_PATH = 'result/{}'.format(args.result_directory)
    CMAT_CATEG_PATH    = RESULT_PATH + '/cmat_categ_{}'.format(args.desc)
    CMAT_AREA_PATH     = RESULT_PATH + '/cmat_area_{}'.format(args.desc)

    pathlib.Path(RESULT_PATH).mkdir(exist_ok=True)

    df_train      = pd.read_pickle(TRAIN_DATASET_PATH)
    df_test       = pd.read_pickle(TEST_DATASET_PATH)
    dataset_train = chainer.datasets.TupleDataset(
        *[df_train[c].values for c in df_train.columns])
    dataset_test  = chainer.datasets.TupleDataset(
        *[df_test[c].values for c in df_test.columns])

    spc_list = get_samples_per_cls(df_train,'PFT')

    print_list = ['epoch', 'main/loss', 'main/accuracy',
            'validation/main/loss', 'validation/main/accuracy', 
            'validation/main/acc_categ','validation/main/acc_area'
            'elapsed_time']
    plot_list = [('loss',['main/loss','validation/main/loss']),
            ('accuracy',['main/accuracy','validation/main/accuracy']),
            ('acc_categ',['main/acc_categ','validation/main/acc_categ']),
            ('acc_area',['main/acc_area','validation/main/acc_area'])]
    
    STORAGE     = f'sqlite:///{args.study}'
    study       = create_study(
        study_name     = 'optuna',
        storage        = STORAGE,
        load_if_exists = True)
    best_params = study.best_params

    DICT_PARAMS_TO_BE_SCANNED = {
        'grad_clipping'     : [1e-5,1e-4,1e-3,1e-2,1e-1,0],
        'learning_rate'     : np.arange(0,1.0,0.1),
        'beta'              : [0.5,0.6,0.7,0.8,0.9,0.99,0.999,0.9999],
        'n_lstm'            : np.arange(100,600,100),
    }
    dict_params_prototype = {
        'spc_list'        : spc_list,
        'n_lstm'          : best_params['lstm'],
        'beta'            : best_params['beta'],
        'df_test'         : df_test,
        'df_train'        : df_train,
        'batch_size'      : BATCH_SIZE,
        'batch_converter' : BC,
        'args'            : args,
        'device'          : DEVICE,
        'print_list'      : print_list,
        'plot_list'       : plot_list,
        'learning_rate'   : best_params['learning_rate'],
        'grad_clipping'   : best_params['grad_clipping'],
    }

    list_params = []
    list_params.append(dict_params_prototype.copy())
    if args.param_to_be_scanned is not '':
        list_params_to_be_scanned = DICT_PARAMS_TO_BE_SCANNED[args.param_to_be_scanned]
        for param in list_params_to_be_scanned:
            dict_params = dict_params_prototype.copy()
            dict_params[args.param_to_be_scanned] = param
            list_params.append(dict_params)

    dict_log = {}
    for dict_params in list_params:
        model,trainer,reporter = get_model_trainer_reporter(**dict_params)
        model.to_gpu()
        trainer.run()
        dict_log[str(dict_params[args.param_to_be_scanned])] = reporter.log
        cm_categ,cm_area = get_confusion_matrix_DAC(df_test,model)
        save_cmat_fig(cm_categ,wd.categories(),CMAT_CATEG_PATH)
        save_cmat_fig(cm_area,wd_area.categories(),CMAT_AREA_PATH)
    
    log_filename = 'result/{}/log_{}.pkl'.format(args.result_directory,args.desc)
    with open(log_filename, 'wb') as f:
        pickle.dump(dict_log,f)