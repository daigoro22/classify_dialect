from word_and_categ_dict import WordAndCategDict
import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
from chainer import(
    initializers,
    training,
    iterators,
    optimizers,
    optimizer_hooks)
from chainer import Variable
#import numpy as np
import cupy as np
import argparse
from dialect_and_area_classifier import(
    DialectAndAreaClassifier,
    batch_converter_area)
from cb_loss_classifier import CbLossClassifier
from class_balanced_loss import get_samples_per_cls
from make_confusion_matrix import(
    get_confusion_matrix_DAC,
    save_cmat_fig)
from chunk_dialect_classifier import ChunkDialectClassifier

def get_model(spc_list:list,n_lstm:int,beta:float):
    model = CbLossClassifier(
        spc_list = spc_list,
        n_lstm   = n_lstm,
        beta     = beta,
        n_embed  = 100,
        n_categ  = 48,
        n_area   = 8
    )
    return model

def get_trainer_and_reporter(
    model,
    df_test:pd.DataFrame,
    df_train:pd.DataFrame,
    batch_size,
    batch_converter,
    args,
    device,
    plot_list,
    print_list=[],
    learning_rate=1e-5,
    grad_clipping=0.5):

    iter_train = get_iter(df_train,batch_size,shuffle=True ,repeat=True)
    iter_test  = get_iter(df_test ,batch_size,shuffle=False,repeat=False)

    optimizer = optimizers.SGD(lr=learning_rate)
    optimizer.setup(model)
    if grad_clipping != None: # grad_clipping ありの場合
        optimizer.add_hook(optimizer_hooks.GradientClipping(threshold=grad_clipping))

    updater = training.StandardUpdater(
        iter_train,
        optimizer,
        device=device,
        converter=batch_converter
    )

    trainer = training.Trainer(updater,(args.epoch,'epoch'),out='result')
    
    ext_evaluator       = training.extensions.Evaluator(iter_test, model,device=device,converter=batch_converter)
    ext_snapshot_object = training.extensions.snapshot_object(
        target   = model, 
        filename = 'model_{}.npz'.format(args.desc), 
        writer   = training.extensions.snapshot_writers.ThreadQueueWriter())
    ext_reporter        = training.extensions.LogReport()
    ext_plotreport_list = []
    for tag,plot in plot_list:
        ext_plotreport_list.append(
            training.extensions.PlotReport(
                plot,
                x_key     = 'epoch',
                file_name = '{}/{}_{}.png'.format(args.result_directory,tag,args.desc))
        )
    ext_dump_graph      = training.extensions.dump_graph('main/loss')

    trainer.extend(ext_evaluator)
    trainer.extend(ext_snapshot_object,trigger=(10,'epoch'))
    trainer.extend(ext_reporter)
    for pl in ext_plotreport_list:
        trainer.extend(pl)
    trainer.extend(ext_dump_graph)

    if print_list != []:
        ext_printreport     = training.extensions.PrintReport(print_list)
        trainer.extend(ext_printreport)

    iter_test.reset()
    iter_train.reset()

    return trainer,ext_reporter

def get_model_trainer_reporter(
    spc_list,
    n_lstm,
    beta,
    df_test:pd.DataFrame,
    df_train:pd.DataFrame,
    batch_size,
    batch_converter,
    args,
    device,
    plot_list,
    print_list=[],
    learning_rate=1e-5,
    grad_clipping=0.5):

    model = get_model(spc_list,n_lstm,beta)
    trainer,reporter = get_trainer_and_reporter(
        model           = model,
        df_test         = df_test,
        df_train        = df_train,
        batch_size      = batch_size,
        batch_converter = batch_converter,
        args            = args,
        device          = device,
        print_list      = print_list,
        plot_list       = plot_list,
        learning_rate   = learning_rate,
        grad_clipping   = grad_clipping
    )
    return model,trainer,reporter

class DialectClassifier(chainer.Chain):
    def __init__(self,n_vocab,n_categ,n_embed,n_lstm,fasttext):
        super(DialectClassifier,self).__init__()
        self.fasttext = fasttext
        with self.init_scope():
            self.embed_d = L.EmbedID(n_vocab,n_embed)
            self.embed_c = L.EmbedID(n_vocab,n_embed)
            self.lstm    = L.NStepLSTM(1,n_embed,n_lstm,dropout=0.2)
            self.categ   = L.Linear(None,n_categ)
    
    def sequence_embed(self,embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1]).astype(np.int32).tolist()
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    def __call__(self,dialect,standard):
        if self.fasttext:
            h_emb_d = dialect 
        else:
            h_emb_d = self.sequence_embed(self.embed_d,dialect)
        h2,_,_  = self.lstm(None,None,h_emb_d)
        h3      = F.relu(h2[0])
        return self.categ(h3)

def batch_converter(batch,device):
    dialect  = [np.array(b[0]) for b in batch]
    standard = [np.array(b[1]) for b in batch]
    category = np.array([b[2] for b in batch],dtype=np.int32)
    return {'dialect':dialect,'standard':standard,'category':category}

def get_iter(df:pd.DataFrame,batch_size:int,shuffle:bool,repeat:bool):
    _dataset_train = chainer.datasets.TupleDataset(
        *[df[c].values for c in df.columns])
    _iter = iterators.SerialIterator(_dataset_train,batch_size,shuffle=shuffle,repeat=repeat)
    return _iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch',type=int,default=100)
    parser.add_argument('-ft','--fasttext',action='store_true')
    parser.add_argument('-ac','--area_classify',action='store_true')
    parser.add_argument('-cb','--cb_loss',type=float,default=None)
    parser.add_argument('-ch','--chunk_character',action='store_true')
    parser.add_argument('-lstm','--n_lstm',type=int,default=300)
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-5)
    parser.add_argument('-gc','--grad_clipping',type=float,default=0.01)
    parser.add_argument('-d','--desc',type=str,default='')
    parser.add_argument('-rd','--result_directory',type=str,default='')
    args = parser.parse_args()

    BATCH_SIZE = 60

    wd      = WordAndCategDict('spm/dialect_standard.model','corpus/all_pft.txt')
    wd_area = WordAndCategDict(categ_path='corpus/all_area.txt')

    if args.fasttext:
        train_dataset_path = 'corpus/train_ft.pkl'
        test_dataset_path  = 'corpus/test_ft.pkl'
    elif args.area_classify or args.cb_loss != None:
        train_dataset_path = 'corpus/train_ft_area.pkl'
        test_dataset_path  = 'corpus/test_ft_area.pkl'
    elif args.chunk_character:
        train_dataset_path = 'corpus/train_character.pkl'
        test_dataset_path  = 'corpus/test_character.pkl'
    else:
        train_dataset_path = 'corpus/train.pkl'
        test_dataset_path  = 'corpus/test.pkl'

    df_train      = pd.read_pickle(train_dataset_path)
    df_test       = pd.read_pickle(test_dataset_path)
    dataset_train = chainer.datasets.TupleDataset(
        *[df_train[c].values for c in df_train.columns])
    dataset_test  = chainer.datasets.TupleDataset(
        *[df_test[c].values for c in df_test.columns])

    if args.area_classify:
        model = DialectAndAreaClassifier(
            n_categ = 48,
            n_embed = 100,
            n_lstm  = 600,
            n_area  = 8
        )
        bc = batch_converter_area
    elif args.cb_loss != None:
        spc_list = get_samples_per_cls(df_train,'PFT')
        model = get_model(
            spc_list = spc_list,
            beta     = args.cb_loss,
            n_lstm   = args.n_lstm
        )
        bc = batch_converter_area
    elif args.chunk_character:
        model = L.Classifier(
            ChunkDialectClassifier(
                n_vocab = 98,
                n_categ = 48,
                n_lstm  = args.n_lstm
        ),
        label_key='category')
        bc = batch_converter
    else:
        model = L.Classifier(
            DialectClassifier(
                n_vocab  = 16000,
                n_categ  = 48,
                n_embed  = 100 if args.fasttext else 300,
                n_lstm   = 600,
                fasttext = args.fasttext
            ),
            label_key='category')
        bc = batch_converter
    model.to_gpu()

    if args.area_classify or args.cb_loss != None:
        print_list = ['epoch', 'main/loss', 'main/accuracy',
            'validation/main/loss', 'validation/main/accuracy', 
            'validation/main/acc_categ','validation/main/acc_area'
            'elapsed_time']
        plot_list = [('loss',['main/loss','validation/main/loss']),
            ('accuracy',['main/accuracy','validation/main/accuracy']),
            ('acc_categ',['main/acc_categ','validation/main/acc_categ']),
            ('acc_area',['main/acc_area','validation/main/acc_area'])]
    else:
        print_list = ['epoch', 'main/loss', 'main/accuracy',
            'validation/main/loss', 'validation/main/accuracy','elapsed_time']
        plot_list = [('loss',['main/loss','validation/main/loss']),
            ('accuracy',['main/accuracy','validation/main/accuracy'])]

    trainer,reporter = get_trainer_and_reporter(
        model           = model,
        df_test         = df_test,
        df_train        = df_train,
        batch_size      = BATCH_SIZE,
        batch_converter = bc,
        args            = args,
        device          = 0,
        print_list      = print_list,
        plot_list       = plot_list,
        learning_rate=args.learning_rate,
        grad_clipping=args.grad_clipping
    )
    trainer.run()

    cm_categ,cm_area = get_confusion_matrix_DAC(df_test,model)
    save_cmat_fig(cm_categ,wd.categories(),'result/{}/cmat_categ_{}'.format(args.result_directory,args.desc))
    save_cmat_fig(cm_area,wd_area.categories(),'result/{}/cmat_area_{}'.format(args.result_directory,args.desc))