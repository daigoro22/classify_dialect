import chainer.functions as F
import cupy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cb_loss_weight(beta,samples_per_cls):
    effective_num = 1 - np.power(beta,samples_per_cls)
    weight = (1 - beta) / effective_num
    weight = weight.astype(np.float32)
    return weight

def get_samples_per_cls(df:pd.DataFrame,key:str):
    counts = df[key].value_counts()
    counts_list = [(i,c) for i,c in counts.items()]
    counts_list = sorted(counts_list,key=lambda x:x[0])
    return np.array([c for i,c in counts_list],dtype=np.int32)

if __name__ == "__main__":
    number_of_samples = np.arange(10000)
    for beta in [0.001,0.9,0.99,0.999,0.9999]:
        weight = cb_loss_weight(beta,number_of_samples)
        plt.plot(weight.tolist())
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('cb_weights.png')