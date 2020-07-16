import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from word_and_categ_dict import WordAndCategDict
import argparse

def to_index_and_save(data_d,data_s,data_c,wcdict,data_name,index):
    data_name += '.pkl'

    data_d_new = data_d.map(wcdict.stoi)
    data_s_new = data_s.map(wcdict.stoi)
    data_c_new = data_c.map(wcdict.ctoi)

    pd.concat([data_d_new,data_s_new,data_c_new],axis=1).to_pickle(data_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','-seed',type=int,default=1)
    args = parser.parse_args()

    index=['DIALECT','STANDARD','PFT']
    df = pd.read_csv('corpus/concat_corpus_all.txt',sep='\t',names=index)
    train,test = train_test_split(df,test_size=0.2,random_state=args.s)
    train,valid = train_test_split(train,test_size=0.25,random_state=args.s)
    
    wcdict = WordAndCategDict()

    to_index_and_save(
        train['DIALECT'],
        train['STANDARD'],
        train["PFT"],
        wcdict,
        'corpus/train',index)

    to_index_and_save(
        test['DIALECT'],
        test['STANDARD'],
        test["PFT"],wcdict,
        'corpus/test',index)

    to_index_and_save(
        valid['DIALECT'],
        valid['STANDARD'],
        valid["PFT"],
        wcdict,
        'corpus/valid',index)