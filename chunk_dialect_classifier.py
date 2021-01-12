import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
from typing import List

class ChunkDialectClassifier(chainer.Chain):
    def __init__(self,n_vocab=98,n_categ=48,n_lstm=600):
        super(ChunkDialectClassifier,self).__init__()
        with self.init_scope():
            self.lstm_s = L.NStepLSTM(1,n_vocab,n_lstm,dropout=0.2)
            self.lstm_d = L.NStepLSTM(1,n_vocab,n_lstm,dropout=0.2)
            self.categ  = L.Linear(2*n_lstm,n_categ)

    def __call__(self,dialect,standard):
        h1_s,_,_ = self.lstm_s(None,None,standard)
        h1_d,_,_ = self.lstm_d(None,None,dialect)
        h3       = F.concat([F.relu(h1_s[0]),F.relu(h1_d[0])])
        return self.categ(h3)

def get_one_hot(df:pd.DataFrame,label:str,classes:List[str]=None):
    """文字レベルの one-hotベクトルを pd.DataFrame の１カラムから取得する.
    Args:
        df (pd.DataFrame): one-hot ベクトルを取得したいデータ列が格納された DataFrame.
        label (str): one-hot ベクトルを取得したいデータ列のラベル.
    Returns:
        df_one_hot (pd.DataFrame): 変換した one-hot ベクトルが格納されている DataFrame.
    
    >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    >>> get_one_hot(df,'standard')
                                           standard
    0  [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    1  [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    """
    
    df = df.dropna(how='any',axis=0)

    # データ列を構成する文字を flat なリストにする
    se_listed       = df[label].map(list)
    
    # 文字->整数に変換するエンコーダ
    le = LabelEncoder()

    # classes が指定されていない場合, 構成する文字からエンコーダ作成
    if classes == None:
        list_data_flat  = list(itertools.chain.from_iterable(se_listed.tolist()))
        le.fit(list_data_flat)
    # classes が指定されている場合, classes からエンコーダ作成
    else:
        le.fit(classes)
    
    len_le = len(le.classes_) 
    # 文字->整数に変換
    # transform で ValueError 出たら * のベクトル返す関数
    def transform(line):
        try:
            return le.transform(line)
        except ValueError:
            return le.transform(['*'])
    se_listed_encoded = se_listed.map(transform)

    # 整数->one-hot　ベクトルへの変換
    to_one_hot = lambda x:np.identity(len_le,dtype=np.float32)[x]
    se_listed_one_hot = se_listed_encoded.map(to_one_hot)

    # 元のラベルを付けた DataFrame へ変換
    df_one_hot = pd.DataFrame(se_listed_one_hot,columns=[label])
    return df_one_hot

if __name__ == "__main__":
    df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    get_one_hot(df,'standard')