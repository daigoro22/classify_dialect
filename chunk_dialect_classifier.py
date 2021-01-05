import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder

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
    """文字レベルの one-hotベクトルを pd.DataFrame の１カラムから取得する.
    Args:
        df (pd.DataFrame): one-hot ベクトルを取得したいデータ列が格納された DataFrame.
        label (str): one-hot ベクトルを取得したいデータ列のラベル.
    Returns:
        df_one_hot (pd.DataFrame): 変換した one-hot ベクトルが格納されている DataFrame.
    
    >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    >>> get_one_hot(df,'standard')
                           standard
    0  [[1, 0, 0, 0], [0, 1, 0, 0]]
    1  [[0, 0, 1, 0], [0, 0, 0, 1]]
    """
    
    # エンコーダを作成するために, データ列を構成する文字を flat なリストにする
    se_listed       = df[label].map(list)
    list_data_flat  = list(itertools.chain.from_iterable(se_listed.tolist()))
    
    # 文字->整数に変換するエンコーダ
    le = LabelEncoder()
    le.fit(list_data_flat)
    len_le = len(le.classes_)
    # 文字->整数に変換
    se_listed_encoded = se_listed.map(le.transform)

    # 整数->one-hot　ベクトルへの変換
    to_one_hot = lambda x:np.identity(len_le,dtype=np.int32)[x]
    se_listed_one_hot = se_listed_encoded.map(to_one_hot)

    # 元のラベルを付けた DataFrame へ変換
    df_one_hot = pd.DataFrame(se_listed_one_hot,columns=[label])
    return df_one_hot

if __name__ == "__main__":
    df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    get_one_hot(df,'standard')