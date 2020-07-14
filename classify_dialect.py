from word_and_categ_dict import WordAndCategDict
import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
from chainer import initializers, training, iterators, optimizers
from chainer.training import extensions
from chainer import Variable
import numpy as np

class DialectClassifier(chainer.Chain):
    
    def __init__(self,n_vocab,n_categ,n_embed,n_lstm):
        super(DialectClassifier,self).__init__()
        with self.init_scope():
            self.embed_d = L.EmbedID(n_vocab,n_embed)
            self.embed_c = L.EmbedID(n_vocab,n_embed)
            self.lstm = L.NStepLSTM(1,n_embed,n_lstm,dropout=0.2)
            self.categ = L.Linear(None,n_categ)
    
    def sequence_embed(self,embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    def __call__(self,dialect,standard):
        h_emb_d = self.sequence_embed(self.embed_d,dialect)
        h2,_,_ = self.lstm(None,None,h_emb_d)
        return self.categ(h2[0]).data

if __name__ == "__main__":
    BATCH_SIZE = 60

    model = L.Classifier(DialectClassifier(
        n_vocab=16000,
        n_categ=48,
        n_embed=300,
        n_lstm=600
    ),label_key='category')

    wd = WordAndCategDict('spm/dialect_standard.model','corpus/all_pft.txt')

    df_train = pd.read_pickle('corpus/train.pkl')
    df_test = pd.read_pickle('corpus/test.pkl')

    # print(df_train.head())
    
    dataset_train = chainer.datasets.TupleDataset(
        df_train['DIALECT'].values,
        df_train['STANDARD'].values,
        df_train['PFT'].values)
    dataset_test = chainer.datasets.TupleDataset(
        df_test['DIALECT'].values,
        df_test['STANDARD'].values,
        df_test['PFT'].values)

    iter_train = iterators.SerialIterator(dataset_train,BATCH_SIZE,shuffle=True)
    iter_test = iterators.SerialIterator(dataset_test,BATCH_SIZE,shuffle=False,repeat=False)

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    def batch_converter(batch,device):
        dialect = [np.array(b[0]) for b in batch]
        standard = [np.array(b[1]) for b in batch]
        category = np.array([b[2] for b in batch],dtype=np.int32)
        return {'dialect':dialect,'standard':standard,'category':category}

    updater = training.StandardUpdater(
        iter_train,
        optimizer,
        device=-1,
        converter=batch_converter
    )

    trainer = training.Trainer(updater,(100,'epoch'),out='result')
    trainer.extend(extensions.Evaluator(iter_test, model,device=-1,converter=batch_converter))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                                    'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()