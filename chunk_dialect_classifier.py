import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import(
    OneHotEncoder,
    LabelEncoder
)
from typing import List

class ChunkDialectClassifier(chainer.Chain):
    """文節単位, 文字レベルの県識別モデル
    """
    def __init__(self,n_vocab=98,n_embed=100,n_categ=48,n_lstm=600,dropout=0.2):
        super(ChunkDialectClassifier,self).__init__()
        with self.init_scope():
            self.embed_d = L.EmbedID(n_vocab,n_embed)
            self.embed_s = L.EmbedID(n_vocab,n_embed)
            self.lstm_s = L.NStepLSTM(1,n_embed,n_lstm,dropout=dropout)
            self.lstm_d = L.NStepLSTM(1,n_embed,n_lstm,dropout=dropout)
            self.categ  = L.Linear(2*n_lstm,n_categ)
    
    def sequence_embed(self,embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1]).astype(np.int32).tolist()
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    def __call__(self,dialect,standard):
        h_emb_d = self.sequence_embed(self.embed_d,dialect)
        h_emb_s = self.sequence_embed(self.embed_s,standard)
        h1_d,_,_ = self.lstm_d(None,None,h_emb_d)
        h1_s,_,_ = self.lstm_s(None,None,h_emb_s)
        h3       = F.concat([F.relu(h1_s[0]),F.relu(h1_d[0])])
        return self.categ(h3)

class CharacterOneHotEncoder(OneHotEncoder):
    """文字単位の one-hot ベクトルのエンコーダ.
    sklearn.preprocessing.OneHotEncoder を継承している.
    """
    def __init__(self,classes,**args):
        super(CharacterOneHotEncoder,self).__init__(**args)
        # n行1列の array に変換
        array_classes = np.array(classes).reshape(-1,1)
        self.fit(array_classes)
        self.len_enc  = len(self.categories_[0])
    
    def get_encoded(self,df:pd.DataFrame,label:str):
        """文字レベルの one-hotベクトルを pd.DataFrame の１カラムから取得する.
        Args:
            df (pd.DataFrame): one-hot ベクトルを取得したいデータ列が格納された DataFrame.
            label (str): one-hot ベクトルを取得したいデータ列のラベル.
        Returns:
            df_one_hot (pd.DataFrame): 変換した one-hot ベクトルが格納されている DataFrame.
        
        >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
        >>> classes = ['ア','イ','ウ','エ']
        >>> encoder = CharacterOneHotEncoder(classes=classes,categories='auto',sparse=False,dtype=np.float32)
        >>> encoder.get_encoded(df,'standard')
                                               standard
        0  [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        1  [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        """
        
        df = df.dropna(how='any',axis=0)

        # データ列を構成する文字を flat なリストにする
        se_listed = df[label].map(list)
        
        # n行1列の array に変換
        se_listed = se_listed.map(lambda x: np.array(x).reshape(-1,1))
        
        # 文字-> one-hot に変換
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

class CharacterLabelEncoder(LabelEncoder):
    """文字単位のラベルエンコーダ.
    sklearn.preprocessing.LabelEncoder を継承している.
    """
    def __init__(self,classes,**args):
        super(CharacterLabelEncoder,self).__init__(**args)
        self.fit(classes)
        self.len_enc = len(self.classes_)

    def get_encoded(self,df:pd.DataFrame,label:str):
        """文字レベルでエンコードされたラベルを pd.DataFrame の１カラムから取得する.
        Args:
            df (pd.DataFrame): ラベルを取得したいデータ列が格納された DataFrame.
            label (str): ラベルを取得したいデータ列のラベル.
        Returns:
            df_one_hot (pd.DataFrame): 変換したラベルが格納されている DataFrame.
        
        >>> df=pd.DataFrame({'standard':['アイ','ウエ'],'dialect':['ウエ','オ'],'pref':['gunma','tokyo']})
        >>> classes = ['ア','イ','ウ','エ']
        >>> encoder = CharacterLabelEncoder(classes=classes)
        >>> encoder.get_encoded(df,'standard')
          standard
        0   [0, 1]
        1   [2, 3]
        """
        df = df.dropna(how='any',axis=0)

        # データ列を構成する文字を flat なリストにする
        se_listed = df[label].map(list)

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
    print(encoder.get_encoded(df,'standard'))
