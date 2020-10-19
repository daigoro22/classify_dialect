from optuna import(
    Trial,
    integration,
    pruners,
    create_study
)
from cb_loss_classifier import CbLossClassifier
from chainer import(
    training,
    iterators,
    optimizers,
    optimizer_hooks
)
from chainer.training import extensions
from classify_dialect import get_iter
import argparse
from word_and_categ_dict import WordAndCategDict
import pandas as pd
from dialect_and_area_classifier import batch_converter_area
from class_balanced_loss import get_samples_per_cls

def get_model(trial:Trial,spc_list:list):
    n_lstm = trial.suggest_categorical('lstm',[100,300,600])
    beta = trial.suggest_uniform('beta',0,1.0)
    
    model = CbLossClassifier(
        spc_list = spc_list,
        n_lstm = n_lstm,
        beta = beta,
        n_embed = 100,
        n_categ = 48,
        n_area = 8
    )
    return model

def get_trainer_and_reporter(
    trial:Trial,
    model:CbLossClassifier,
    iter_test:iterators.SerialIterator,
    iter_train:iterators.SerialIterator,
    batch_converter,
    args,
    device=0):

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    grad_clipping = trial.suggest_uniform('grad_clipping',0,1.0)

    optimizer = optimizers.SGD(lr=learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(optimizer_hooks.GradientClipping(threshold=grad_clipping))

    updater = training.StandardUpdater(
        iter_train,
        optimizer,
        device=device,
        converter=batch_converter
    )

    trainer = training.Trainer(updater,(args.epoch,'epoch'),out='optuna')
    trainer.extend(extensions.Evaluator(iter_test, model,device=device,converter=batch_converter))
    snapshot_writer = training.extensions.snapshot_writers.ThreadQueueWriter()
    trainer.extend(training.extensions.snapshot_object(
        target=model, 
        filename='model_{}.npz'.format(args.desc), 
        writer=snapshot_writer),trigger=(10,'epoch'))

    reporter = extensions.LogReport()
    trainer.extend(reporter)

    trainer.extend(integration.ChainerPruningExtension(
        trial,args.pruning_key,(args.pruning_trigger_epoch,'epoch')))

    return trainer,reporter

def get_bestresult(observation,loss_key):
    observation.sort(key=lambda x: x[loss_key])
    return (observation[0]['epoch'], observation[0][loss_key])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch',type=int,default=500)
    parser.add_argument('-d','--desc',type=str,default='')
    parser.add_argument('-oo','--optuna_output',type=str,default='optuna')
    parser.add_argument('-nt','--n_trials',type=int,default=100)
    parser.add_argument('-pk','--pruning_key',type=str,default='main/loss')
    parser.add_argument('-pte','--pruning_trigger_epoch',type=int,default=3)
    args_cl = parser.parse_args()

    BATCH_SIZE = 60

    wd = WordAndCategDict('spm/dialect_standard.model','corpus/all_pft.txt')
    wd_area = WordAndCategDict(categ_path='corpus/all_area.txt')

    train_dataset_path = 'corpus/train_ft_area.pkl'
    test_dataset_path = 'corpus/test_ft_area.pkl'

    df_train = pd.read_pickle(train_dataset_path)
    df_test = pd.read_pickle(test_dataset_path)

    iter_train = get_iter(df_train,BATCH_SIZE,shuffle=True,repeat=True)
    iter_test = get_iter(df_test,BATCH_SIZE,shuffle=False,repeat=False)

    spc_list = get_samples_per_cls(df_train,'PFT')

    def objective(trial:Trial):
        model = get_model(trial,spc_list)
        trainer,reporter = get_trainer_and_reporter(
            trial=trial,
            model=model,
            iter_train=iter_train,
            iter_test=iter_test,
            batch_converter=batch_converter_area,
            args=args_cl,
            device=0
        )
        trainer.run()
        (epoch,best_loss) = get_bestresult(reporter.log,args_cl.pruning_key)
        return best_loss
    
    study = create_study(study_name=args_cl.optuna_output,
                            storage=f'sqlite:///{args_cl.optuna_output}/optuna_{args_cl.desc}.db',
                            load_if_exists=True,
                            pruner=pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=args_cl.n_trials) 
