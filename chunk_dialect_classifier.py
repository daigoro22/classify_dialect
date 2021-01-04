import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
import numpy as np

class ChunkDialectClassifier(chainer.Chain):
    def __init__(self,n_vocab,n_categ,n_lstm):
        super(ChunkDialectClassifier,self).__init__()
        with self.init_scope():
            self.lstm_s = L.NStepLSTM(1,n_vocab,n_lstm,dropout=0.2)
            self.lstm_d = L.NStepLSTM(1,n_vocab,n_lstm,dropout=0.2)
            self.categ  = L.Linear(None,n_categ)

    def __call__(self,dialect,standard):
        h1_s,_,_  = self.lstm_s(None,None,standard)
        h1_d,_,_  = self.lstm_d(None,None,dialect)
        h3        = F.relu(F.concat(h1_s[0],h1_d[0]))
        return self.categ(h3)

def get_one_hot(df:pd.DataFrame,label:str):
    """one-hotベクトルを pd.DataFrame の１カラムから取得する.
    Args:
        df (pd.DataFrame): one-hot ベクトルを取得したいデータ列が格納された DataFrame.
        label (str): one-hot ベクトルを取得したいデータ列のラベル.
    Returns:
        df_oh (pd.DataFrame): 変換した one-hot ベクトルが格納されている DataFrame.
    
    >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    >>> get_one_hot(df,'standard')
        standard
    0  [[1, 0, 0, 0],[0, 1, 0, 0]]
    1  [[0, 0, 1, 0],[0, 0, 0, 1]]
    """
    se_data = df[label]
    print(se_data)

if __name__ == "__main__":
    df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    get_one_hot(df,'standard')