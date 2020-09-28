import chainer.functions as F
import cupy as np
import pandas as pd

def cb_loss(x,t,beta,samples_per_cls):
    effective_num = 1 - np.power(beta,samples_per_cls)
    weight = (1 - beta) / effective_num
    weight = weight.astype(np.float32)
    return F.softmax_cross_entropy(x,t,class_weight=weight)

def get_samples_per_cls(df:pd.DataFrame,key:str):
    counts = df[key].value_counts()
    counts_list = [(i,c) for i,c in counts.items()]
    counts_list = sorted(counts_list,key=lambda x:x[0])
    return np.array([c for i,c in counts_list],dtype=np.int32)

if __name__ == "__main__":
    df_train = pd.read_pickle('corpus/train_ft_area.pkl')
    spc = get_samples_per_cls(df_train,'PFT')