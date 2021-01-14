import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import(
    LabelEncoder,
    OneHotEncoder
)
from typing import List

class ChunkDialectClassifier(chainer.Chain):
    def __init__(self,n_vocab=98,n_categ=48,n_lstm=600,dropout=0.2):
        super(ChunkDialectClassifier,self).__init__()
        with self.init_scope():
            self.lstm_s = L.NStepLSTM(1,n_vocab,n_lstm,dropout=dropout)
            self.lstm_d = L.NStepLSTM(1,n_vocab,n_lstm,dropout=dropout)
            self.categ  = L.Linear(2*n_lstm,n_categ)

    def __call__(self,dialect,standard):
        h1_s,_,_ = self.lstm_s(None,None,standard)
        h1_d,_,_ = self.lstm_d(None,None,dialect)
        h3       = F.concat([F.relu(h1_s[0]),F.relu(h1_d[0])])
        return self.categ(h3)

class CharacterOneHotEncoder(OneHotEncoder):
    """
    文字単位の one-hot ベクトルのエンコーダ.
    sklearn.preprocessing.OneHotEncoder を継承している.
    """
    def __init__(self,classes,**args):
        super(CharacterOneHotEncoder,self).__init__(**args)
        # n行1列の array に変換
        array_classes = np.array(classes).reshape(-1,1)
        self.fit(array_classes)
        self.len_enc  = len(self.categories_[0])
    
    def get_one_hot(self,df:pd.DataFrame,label:str):
        """文字レベルの one-hotベクトルを pd.DataFrame の１カラムから取得する.
        Args:
            df (pd.DataFrame): one-hot ベクトルを取得したいデータ列が格納された DataFrame.
            label (str): one-hot ベクトルを取得したいデータ列のラベル.
        Returns:
            df_one_hot (pd.DataFrame): 変換した one-hot ベクトルが格納されている DataFrame.
        
        >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
        >>> classes = ['ア','イ','ウ','エ']
        >>> encoder = CharacterOneHotEncoder(classes=classes,categories='auto',sparse=False,dtype=np.float32)
        >>> encoder.get_one_hot(df,'standard')
                                               standard
        0  [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        1  [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        """
        
        df = df.dropna(how='any',axis=0)

        # データ列を構成する文字を flat なリストにする
        se_listed = df[label].map(list)
        
        # n行1列の array に変換
        se_listed = se_listed.map(lambda x: np.array(x).reshape(-1,1))
        
        # 文字->整数に変換
        # transform で ValueError 出たら * のベクトル返す関数
        def transform(line):
            try:
                return self.transform(line)
            except ValueError:
                return self.transform([['*']])
        se_listed_encoded = se_listed.map(transform)

        # 元のラベルを付けた DataFrame へ変換
        df_one_hot = pd.DataFrame(se_listed_encoded,columns=[label])
        return df_one_hot

if __name__ == "__main__":
    df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
    classes = ['ア','イ','ウ','エ']

    encoder = CharacterOneHotEncoder(classes=classes,categories='auto',sparse=False,dtype=np.float32)
    print(encoder.get_one_hot(df,'standard'))
